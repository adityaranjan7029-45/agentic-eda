import { writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
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

test("a file over the server's size limit is refused before uploading", async ({ page, user: _ }) => {
  await page.goto("/runs");
  // the limit shown comes from the server (10 MB on the free 1 GB server, 50 MB otherwise)
  const limitMb = Number((await page.getByText(/CSV up to \d+ MB/).innerText()).match(/(\d+) MB/)![1]);
  const big = path.join(tmpdir(), `too-big-${Date.now()}.csv`);
  const row = "1,2,3,4,5,6,7,8,9,10\n";
  writeFileSync(big, "a,b,c,d,e,f,g,h,i,j\n" + row.repeat(Math.ceil(((limitMb + 1) * 1024 * 1024) / row.length)));
  let uploads = 0;
  page.on("request", (r) => {
    if (r.method() === "POST" && r.url().endsWith("/api/backend/runs")) uploads += 1;
  });
  await page.setInputFiles("input[type=file]", big);
  await expect(page.locator("p[role=alert]")).toContainText(`limit on this server is ${limitMb} MB`);
  expect(uploads).toBe(0);
});
