import { expect, runSample, signIn, test } from "../fixtures";

// Narrow phones are where layouts overflow: nothing should scroll sideways.
for (const width of [360, 390]) {
  test(`no sideways scrolling at ${width}px`, async ({ browser, baseURL }) => {
    const context = await browser.newContext({ viewport: { width, height: 800 }, isMobile: true, hasTouch: true });
    const page = await context.newPage();
    const overflow = () => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);

    await page.goto("/");
    expect(await overflow()).toBeLessThanOrEqual(0);

    await signIn(context, baseURL!);
    await runSample(page);
    await page.locator("figure svg").first().waitFor({ timeout: 30_000 });
    expect(await overflow()).toBeLessThanOrEqual(0);
    await page.goto("/runs");
    expect(await overflow()).toBeLessThanOrEqual(0);
    await context.close();
  });
}
