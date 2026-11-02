import { describe, expect, it } from "vitest";

import { FORWARD_REQUEST_HEADERS, backendUrl, forwardHeaders, isSameOrigin } from "../lib/proxy";

describe("forwardHeaders", () => {
  it("passes only allow-listed headers: never cookies, auth or host", () => {
    const incoming = new Headers({
      cookie: "authjs.session-token=secret",
      authorization: "Bearer forged",
      host: "evil.example",
      "content-type": "multipart/form-data; boundary=x",
      "x-groq-key": "gsk_user",
      "last-event-id": "1-0",
    });
    const out = forwardHeaders(incoming, FORWARD_REQUEST_HEADERS);
    expect(Object.fromEntries(out)).toEqual({
      "content-type": "multipart/form-data; boundary=x",
      "x-groq-key": "gsk_user",
      "last-event-id": "1-0",
    });
  });
});

describe("backendUrl", () => {
  const base = "http://api:8000/";

  it("maps allowed paths and keeps the query string", () => {
    expect(backendUrl(base, ["runs"], "?limit=5")).toBe("http://api:8000/runs?limit=5");
    expect(backendUrl(base, ["runs", "abc-123", "events"], "")).toBe("http://api:8000/runs/abc-123/events");
    expect(backendUrl(base, ["health"], "")).toBe("http://api:8000/health");
  });

  it.each([[[]], [["docs"]], [["openapi.json"]], [["runs", ".."]], [["runs", "a/b"]], [["runs", "%2F"]], [["runs", ""]]])(
    "refuses %j",
    (path) => {
      expect(() => backendUrl(base, path as string[], "")).toThrow("path not allowed");
    },
  );
});

describe("isSameOrigin", () => {
  it("allows reads from anywhere and writes only from this site", () => {
    expect(isSameOrigin("GET", null, "app.example")).toBe(true);
    expect(isSameOrigin("POST", "https://app.example", "app.example")).toBe(true);
    expect(isSameOrigin("POST", "https://evil.example", "app.example")).toBe(false);
    expect(isSameOrigin("DELETE", null, "app.example")).toBe(false);
    expect(isSameOrigin("POST", "not a url", "app.example")).toBe(false);
  });
});
