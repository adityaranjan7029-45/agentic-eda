/**
 * Signs a test in without GitHub: mints the same encrypted session cookie Auth.js
 * would set after a real login, using the web app's AUTH_SECRET. Every call makes a
 * new user (random GitHub id), so tests are isolated from each other and from old runs.
 */
import { test as base, expect, type BrowserContext, type Page } from "@playwright/test";
import { encode } from "@auth/core/jwt";
import { randomInt } from "node:crypto";

export const SESSION_COOKIE = "authjs.session-token"; // the http:// name; https uses a __Secure- prefix

export type TestUser = { id: string; name: string };

export async function signIn(context: BrowserContext, baseURL: string, name = "Test User"): Promise<TestUser> {
  const secret = process.env.E2E_AUTH_SECRET;
  if (!secret) throw new Error("Set E2E_AUTH_SECRET to the AUTH_SECRET of the web app under test");
  const id = String(randomInt(100_000_000, 999_999_999));
  const value = await encode({
    secret,
    salt: SESSION_COOKIE,
    maxAge: 60 * 60,
    token: { sub: id, name, email: `e2e-${id}@example.test`, provider: "github", providerAccountId: id },
  });
  await context.addCookies([
    { name: SESSION_COOKIE, value, domain: new URL(baseURL).hostname, path: "/", httpOnly: true, sameSite: "Lax" },
  ]);
  return { id, name };
}

/** Collects uncaught page errors and console errors, so a test can assert there were none. */
export function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  return errors;
}

/** Starts the built-in sample analysis and waits for the verified report. Returns the run URL. */
export async function runSample(page: Page): Promise<string> {
  await page.goto("/runs");
  await page.getByRole("button", { name: "Try a sample credit dataset" }).click();
  await page.waitForURL(/\/runs\/[0-9a-f-]{36}$/);
  await expect(page.getByText("Every number verified")).toBeVisible({ timeout: 90_000 });
  return page.url();
}

export const test = base.extend<{ user: TestUser }>({
  user: async ({ context, baseURL }, use) => {
    await use(await signIn(context, baseURL!));
  },
});

export { expect };
