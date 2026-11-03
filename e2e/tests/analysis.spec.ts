import path from "node:path";

import { expect, runSample, test, watchErrors } from "../fixtures";

const SAMPLE_CSV = path.join(__dirname, "..", "..", "data", "samples", "credit_applications.csv");

test("sample dataset: live progress, then a verified report with charts", async ({ page, user }) => {
  const errors = watchErrors(page);
  await page.goto("/");
  await expect(page).toHaveURL("/runs"); // signed in: straight to the dashboard
  await expect(page.getByRole("heading", { level: 1 })).toContainText(user.name.split(" ")[0]);

  await runSample(page);
  await expect(page.locator("article h2").first()).toBeVisible();
  // Vega-Lite charts are drawn as SVG
  await expect(page.locator("figure svg").first()).toBeVisible({ timeout: 30_000 });
  expect(await page.locator("figure svg").count()).toBeGreaterThanOrEqual(4);
  // every finding cited in the text links to its evidence row
  const rows = page.locator("tr[id^=F]");
  expect(await rows.count()).toBeGreaterThan(0);
  const chip = page.locator("a[href^='#F']").first();
  const target = (await chip.getAttribute("href"))!;
  await chip.click();
  await expect(page).toHaveURL(new RegExp(`${target}$`));

  // the run is listed in the history
  await page.goto("/runs");
  await expect(page.getByRole("link", { name: /sample_credit_applications\.csv/ })).toBeVisible();
  expect(errors).toEqual([]);
});

test("upload a CSV: the target is guessed in the browser, then analysed", async ({ page, user: _ }) => {
  await page.goto("/runs");
  await page.setInputFiles("input[type=file]", SAMPLE_CSV);
  await expect(page.locator("#target")).toHaveValue("default_flag");
  await page.getByRole("button", { name: "Analyse" }).click();
  await page.waitForURL(/\/runs\/[0-9a-f-]{36}$/);
  await expect(page.getByText("Every number verified")).toBeVisible({ timeout: 90_000 });
  await expect(page.locator("h1")).toContainText("credit_applications.csv");
});

test("a file that can't be analysed gets a plain message", async ({ page, user: _ }) => {
  await page.goto("/runs");
  await page.setInputFiles("input[type=file]", {
    name: "one_column.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("only_column\n1\n2\n3\n"),
  });
  await expect(page.locator("p[role=alert]")).toBeVisible();
  await expect(page).toHaveURL("/runs"); // nothing was started
});
