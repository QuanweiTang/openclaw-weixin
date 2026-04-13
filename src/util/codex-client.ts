import { setTimeout as delay } from "node:timers/promises";

const DEFAULT_BASE_URL = "https://api.openai.com/v1";
const DEFAULT_MODEL = "gpt-4.1";
const DEFAULT_TIMEOUT_MS = 30_000;
const RETRY_STATUS = new Set([408, 409, 429, 500, 502, 503, 504]);

export interface CodexClientOptions {
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  timeoutMs?: number;
  maxRetries?: number;
}

export interface CodexTextResult {
  id: string;
  model: string;
  outputText: string;
  raw: unknown;
}

interface ResponsesApiSuccess {
  id: string;
  model: string;
  output_text?: string;
  output?: Array<{
    type?: string;
    content?: Array<{
      type?: string;
      text?: string;
    }>
  }>
}

export class CodexClientError extends Error {
  readonly status?: number;
  readonly requestId?: string;
  readonly details?: unknown;

  constructor(message: string, meta?: { status?: number; requestId?: string; details?: unknown }) {
    super(message);
    this.name = "CodexClientError";
    this.status = meta?.status;
    this.requestId = meta?.requestId;
    this.details = meta?.details;
  }
}

export class CodexClient {
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly model: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;

  constructor(options: CodexClientOptions = {}) {
    this.apiKey = options.apiKey ?? process.env.OPENAI_API_KEY ?? "";
    this.baseUrl = trimTrailingSlash(options.baseUrl ?? process.env.OPENAI_BASE_URL ?? DEFAULT_BASE_URL);
    this.model = options.model ?? process.env.OPENAI_MODEL ?? DEFAULT_MODEL;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.maxRetries = options.maxRetries ?? 2;

    if (!this.apiKey) {
      throw new CodexClientError("Missing OpenAI API key. Please set OPENAI_API_KEY.");
    }
  }

  async runCodex(prompt: string): Promise<CodexTextResult> {
    if (!prompt?.trim()) {
      throw new CodexClientError("Prompt must not be empty.");
    }

    const payload = {
      model: this.model,
      input: prompt,
    };

    const response = await this.requestWithRetry("/responses", payload);
    const outputText = extractOutputText(response);

    if (!outputText) {
      throw new CodexClientError("Responses API returned no text output.", {
        details: response,
      });
    }

    return {
      id: response.id,
      model: response.model,
      outputText,
      raw: response,
    };
  }

  private async requestWithRetry(path: string, payload: object): Promise<ResponsesApiSuccess> {
    let attempt = 0;
    let lastError: unknown;

    while (attempt <= this.maxRetries) {
      try {
        return await this.request(path, payload);
      } catch (error) {
        lastError = error;
        const status = error instanceof CodexClientError ? error.status : undefined;

        const shouldRetry =
          attempt < this.maxRetries &&
          (status === undefined || (typeof status === "number" && RETRY_STATUS.has(status)));

        if (!shouldRetry) {
          throw error;
        }

        const backoffMs = 300 * 2 ** attempt;
        await delay(backoffMs);
      }

      attempt += 1;
    }

    throw lastError instanceof Error ? lastError : new CodexClientError("Unknown request failure.");
  }

  private async request(path: string, payload: object): Promise<ResponsesApiSuccess> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      const requestId = response.headers.get("x-request-id") ?? undefined;
      const body = (await safeJson(response)) as unknown;

      if (!response.ok) {
        const message = mapErrorMessage(response.status, body);
        throw new CodexClientError(message, {
          status: response.status,
          requestId,
          details: body,
        });
      }

      return body as ResponsesApiSuccess;
    } catch (error) {
      if (isAbortError(error)) {
        throw new CodexClientError(`OpenAI request timed out after ${this.timeoutMs}ms.`);
      }

      if (error instanceof CodexClientError) {
        throw error;
      }

      throw new CodexClientError("Failed to call OpenAI Responses API.", {
        details: toErrorDetails(error),
      });
    } finally {
      clearTimeout(timeout);
    }
  }
}

export async function runCodex(prompt: string, options?: CodexClientOptions): Promise<CodexTextResult> {
  const client = new CodexClient(options);
  return client.runCodex(prompt);
}

function extractOutputText(body: ResponsesApiSuccess): string {
  if (typeof body.output_text === "string" && body.output_text.trim()) {
    return body.output_text.trim();
  }

  const textChunks: string[] = [];

  for (const item of body.output ?? []) {
    for (const content of item.content ?? []) {
      if (content.type === "output_text" && content.text) {
        textChunks.push(content.text);
      }
    }
  }

  return textChunks.join("\n").trim();
}

function mapErrorMessage(status: number, body: unknown): string {
  if (status === 401) {
    return "Unauthorized (401): invalid or missing OPENAI_API_KEY.";
  }

  if (status === 429) {
    return "Rate limited (429): too many requests to OpenAI API.";
  }

  if (status >= 500) {
    return `OpenAI server error (${status}).`;
  }

  const apiMessage = extractApiErrorMessage(body);
  return apiMessage ? `OpenAI request failed (${status}): ${apiMessage}` : `OpenAI request failed (${status}).`;
}

function extractApiErrorMessage(body: unknown): string | undefined {
  if (!body || typeof body !== "object") {
    return undefined;
  }

  const maybeError = (body as Record<string, unknown>).error;
  if (!maybeError || typeof maybeError !== "object") {
    return undefined;
  }

  const maybeMessage = (maybeError as Record<string, unknown>).message;
  return typeof maybeMessage === "string" ? maybeMessage : undefined;
}

async function safeJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return {};
  }

  try {
    return JSON.parse(text) as unknown;
  } catch {
    return { raw: text };
  }
}

function trimTrailingSlash(input: string): string {
  return input.endsWith("/") ? input.slice(0, -1) : input;
}

function isAbortError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    (error as { name?: string }).name === "AbortError"
  );
}

function toErrorDetails(error: unknown): Record<string, unknown> {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack,
    };
  }

  return { error };
}