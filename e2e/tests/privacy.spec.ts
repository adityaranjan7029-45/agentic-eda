import { expect, runSample, signIn, test } from "../fixtures";

test("another user can't see or delete your run", async ({ page, browser, baseURL, user: _ }) => {
  const runUrl = await runSample(page);
  const runId = runUrl.split("/").pop()!;

  const other = await browser.newContext();
  await signIn(other, baseURL!, "Someone Else");
  const theirs = await other.newPage();
  await theirs.goto(runUrl);
  await expect(theirs.locator("p[role=alert]")).toContainText("doesn't exist, or it isn't yours");
  await theirs.goto("/runs");
  await expect(theirs.getByText("No analyses yet")).toBeVisible();
  // straight at the API through the proxy: same answer
  expect((await theirs.request.get(`/api/backend/runs/${runId}`)).status()).toBe(404);
  const origin = new URL(baseURL!).origin;
  expect((await theirs.request.delete(`/api/backend/runs/${runId}`, { headers: { origin } })).status()).toBe(404);
  await other.close();

  // the owner still has it
  await page.reload();
  await expect(page.getByText("Every number verified")).toBeVisible();
});

test("delete asks for confirmation, then removes the run", async ({ page, user: _ }) => {
  const runUrl = await runSample(page);
  await page.goto("/runs");
  const del = page.getByRole("button", { name: "Delete sample_credit_applications.csv" });
  await del.click();
  await page.getByRole("button", { name: "Cancel" }).click(); // changing your mind keeps it
  await expect(page.getByRole("link", { name: /sample_credit_applications\.csv/ })).toBeVisible();
  await del.click();
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page.getByText("No analyses yet")).toBeVisible();
  await page.goto(runUrl);
  await expect(page.locator("p[role=alert]")).toContainText("doesn't exist");
});

test("at most two analyses in progress per user", async ({ page, baseURL, user: _ }) => {
  await page.goto("/runs");
  // Three requests at once: the API serialises them per user, so exactly one is refused.
  // (The proxy only accepts writes from its own origin, like a real browser sends.)
  const post = () =>
    page.request.post("/api/backend/runs/sample", { headers: { origin: new URL(baseURL!).origin } });
  const statuses = await Promise.all(
    [0, 1, 2].map(() => post().then((r) => r.status())),
  );
  expect(statuses.sort()).toEqual([202, 202, 429]);
});
