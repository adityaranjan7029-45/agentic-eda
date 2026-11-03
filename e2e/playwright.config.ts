/**
 * End-to-end tests run a real Chromium against the whole stack: web -> API -> Redis -> workers -> Postgres.
 *
 *   docker compose up -d --build --wait      (or any running stack)
 *   cd e2e; npm ci; npx playwright install chromium
 *   $env:E2E_AUTH_SECRET = "<AUTH_SECRET from web/.env.local>"; npm test
 *
 * Against the live site: $env:E2E_BASE_URL = "https://eda.<your-domain>" and its AUTH_SECRET.
 *
 * Each test signs in as a brand-new user (see fixtures.ts), so tests never share data
 * and can run in parallel.
 */
import { defineConfig, devices } from "@playwright/test";

const CI = !!process.env.CI;

export default defineConfig({
  testDir: "./tests",
  timeout: 120_000,
  expect: { timeout: 15_000 },
  fullyParallel: true,
  workers: CI ? 2 : undefined,
  retries: CI ? 1 : 0,
  forbidOnly: CI,
  reporter: CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:3000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    // a local HTTPS test (e.g. Caddy with its own certificate); never needed for the live site
    ignoreHTTPSErrors: !!process.env.E2E_IGNORE_HTTPS_ERRORS,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
