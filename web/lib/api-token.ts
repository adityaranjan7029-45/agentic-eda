/**
 * Short-lived tokens for calls from this server to the FastAPI backend.
 *
 * The browser never sees these: it holds only Auth.js's httpOnly session cookie.
 * For each API call the Next.js server signs a 5-minute HS256 token with the secret
 * shared with the API (API_JWT_SECRET). The API pins the algorithm and checks
 * issuer, audience and expiry (see api/src/eda_api/auth.py).
 */
import { SignJWT } from "jose";

export const API_TOKEN_TTL_SECONDS = 300;
export const API_TOKEN_ISSUER = "agentic-eda-web";
export const API_TOKEN_AUDIENCE = "agentic-eda-api";
const PROVIDER = /^[a-z][a-z0-9-]{1,31}$/;
const ACCOUNT_ID = /^[A-Za-z0-9._-]{1,255}$/;

export type ApiUser = {
  provider: string; // "github"
  accountId: string; // the provider's account id
  email?: string | null;
  name?: string | null;
};

export async function mintApiToken(
  user: ApiUser,
  secret: string | undefined,
  now: number = Math.floor(Date.now() / 1000),
): Promise<string> {
  if (!secret || secret.length < 32) {
    throw new Error("API_JWT_SECRET must be set (at least 32 characters)");
  }
  if (!PROVIDER.test(user.provider) || !ACCOUNT_ID.test(user.accountId)) {
    throw new Error("invalid user identity");
  }
  const claims: Record<string, string> = {};
  if (user.email) claims.email = user.email;
  if (user.name) claims.name = user.name;
  return new SignJWT(claims)
    .setProtectedHeader({ alg: "HS256", typ: "JWT" })
    .setSubject(`${user.provider}:${user.accountId}`)
    .setIssuer(API_TOKEN_ISSUER)
    .setAudience(API_TOKEN_AUDIENCE)
    .setIssuedAt(now)
    .setExpirationTime(now + API_TOKEN_TTL_SECONDS)
    .sign(new TextEncoder().encode(secret));
}
