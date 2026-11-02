/**
 * Pure helpers for the backend proxy (app/api/backend/[...path]/route.ts).
 * Kept free of Next.js imports so they can be unit-tested directly.
 */

/** Request headers the browser may send through to the API. Cookies and auth never pass. */
export const FORWARD_REQUEST_HEADERS = [
  "accept",
  "content-type",
  "last-event-id", // resume a live progress stream
  "x-groq-key", // a user's own LLM keys (used for one run, never stored)
  "x-cerebras-key",
  "x-openrouter-key",
];

/** Response headers passed back to the browser. */
export const FORWARD_RESPONSE_HEADERS = ["content-type", "cache-control", "x-accel-buffering", "retry-after"];

/** API paths the browser may reach through the proxy. */
const ALLOWED_ROOTS = new Set(["health", "runs"]);
const SEGMENT = /^[A-Za-z0-9._-]{1,128}$/;

export function forwardHeaders(source: Headers, allow: string[]): Headers {
  const out = new Headers();
  for (const name of allow) {
    const value = source.get(name);
    if (value !== null) out.set(name, value);
  }
  return out;
}

/** Build the API URL, refusing anything outside the allowed paths (no "..", no encoded slashes). */
export function backendUrl(base: string, path: string[], search: string): string {
  if (path.length === 0 || !ALLOWED_ROOTS.has(path[0])) throw new Error("path not allowed");
  for (const segment of path) {
    if (!SEGMENT.test(segment) || segment === "." || segment === "..") throw new Error("path not allowed");
  }
  return `${base.replace(/\/+$/, "")}/${path.join("/")}${search}`;
}

/** State-changing requests must come from this site (defence in depth on top of SameSite cookies). */
export function isSameOrigin(method: string, origin: string | null, host: string | null): boolean {
  if (method === "GET" || method === "HEAD") return true;
  if (!origin || !host) return false;
  try {
    return new URL(origin).host === host;
  } catch {
    return false;
  }
}
