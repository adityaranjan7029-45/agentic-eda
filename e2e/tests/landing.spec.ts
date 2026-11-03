import { expect, test } from "@playwright/test";

import { watchErrors } from "../fixtures";

test.describe("signed out", () => {
  test("landing page explains the product and offers GitHub sign-in", async ({ page }) => {
    const errors = watchErrors(page);
    await page.goto("/");
    await expect(page.getByRole("button", { name: "Sign in with GitHub" }).first()).toBeVisible();
    // the report preview on the landing page is labelled as an example, not real data
    await expect(page.getByText("Example", { exact: true })).toBeVisible();
    expect(errors).toEqual([]);
  });

  test("private pages send you back to the landing page", async ({ page }) => {
    await page.goto("/runs");
    await expect(page).toHaveURL("/");
    await page.goto("/runs/00000000-0000-0000-0000-000000000000");
    await expect(page).toHaveURL("/");
  });

  test("the API proxy refuses requests without a session", async ({ request }) => {
    const response = await request.get("/api/backend/runs");
    expect(response.status()).toBe(401);
  });
});
