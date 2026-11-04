import { describe, expect, it } from "vitest";

import { enabledProviders } from "../lib/providers";

describe("enabledProviders", () => {
  it("offers GitHub only when Google isn't configured", () => {
    expect(enabledProviders({})).toEqual(["github"]);
    expect(enabledProviders({ AUTH_GOOGLE_ID: "id-only" })).toEqual(["github"]);
    expect(enabledProviders({ AUTH_GOOGLE_ID: "  ", AUTH_GOOGLE_SECRET: "  " })).toEqual(["github"]);
  });

  it("adds Google, first, when both its id and secret are set", () => {
    expect(enabledProviders({ AUTH_GOOGLE_ID: "123.apps.googleusercontent.com", AUTH_GOOGLE_SECRET: "s" })).toEqual([
      "google",
      "github",
    ]);
  });
});
