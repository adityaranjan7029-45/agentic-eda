import { decodeProtectedHeader, jwtVerify } from "jose";
import { describe, expect, it } from "vitest";

import { API_TOKEN_TTL_SECONDS, mintApiToken } from "../lib/api-token";

const SECRET = "web-test-secret-that-is-long-enough-0123456789";
const key = new TextEncoder().encode(SECRET);
const user = { provider: "github", accountId: "583231", email: "a@example.com", name: "Aditya" };

describe("mintApiToken", () => {
  it("signs an HS256 token the API will accept", async () => {
    const now = Math.floor(Date.now() / 1000);
    const token = await mintApiToken(user, SECRET, now);
    expect(decodeProtectedHeader(token)).toEqual({ alg: "HS256", typ: "JWT" });
    const { payload } = await jwtVerify(token, key, {
      issuer: "agentic-eda-web",
      audience: "agentic-eda-api",
      algorithms: ["HS256"],
    });
    expect(payload.sub).toBe("github:583231");
    expect(payload.email).toBe("a@example.com");
    expect(payload.exp! - payload.iat!).toBe(API_TOKEN_TTL_SECONDS);
  });

  it("expires after five minutes", async () => {
    const issued = Math.floor(Date.now() / 1000) - API_TOKEN_TTL_SECONDS - 60;
    const token = await mintApiToken(user, SECRET, issued);
    await expect(jwtVerify(token, key)).rejects.toThrow(/exp/);
  });

  it("signs Google identities the same way", async () => {
    const token = await mintApiToken({ provider: "google", accountId: "109876543210987654321" }, SECRET);
    const { payload } = await jwtVerify(token, new TextEncoder().encode(SECRET), {
      issuer: "agentic-eda-web",
      audience: "agentic-eda-api",
    });
    expect(payload.sub).toBe("google:109876543210987654321");
  });

  it("omits missing profile fields", async () => {
    const token = await mintApiToken({ provider: "github", accountId: "1" }, SECRET);
    const { payload } = await jwtVerify(token, key);
    expect(payload).not.toHaveProperty("email");
    expect(payload).not.toHaveProperty("name");
  });

  it("refuses a missing or short secret", async () => {
    await expect(mintApiToken(user, undefined)).rejects.toThrow(/API_JWT_SECRET/);
    await expect(mintApiToken(user, "short")).rejects.toThrow(/API_JWT_SECRET/);
  });

  it("refuses identities that could confuse the API", async () => {
    await expect(mintApiToken({ ...user, accountId: "../admin" }, SECRET)).rejects.toThrow(/identity/);
    await expect(mintApiToken({ ...user, provider: "GitHub:x" }, SECRET)).rejects.toThrow(/identity/);
  });
});
