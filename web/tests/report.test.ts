import { describe, expect, it } from "vitest";

import { errorMessage } from "../lib/errors";
import { BAND_INFO, bandInfo, directionLabel, findingIds, formatBytes, isActive, keyProblem, pct, tokenUsage, verificationBadge } from "../lib/report";
import type { Finding, Report } from "../lib/types";

const finding = (column: string, extra: Partial<Finding> = {}): Finding => ({
  kind: "driver", column, direction: "+", gini: 0.3, q_value: 1e-6, summary: "", evidence: {},
  related_to: null, correlation: null, confidence: "high", ...extra,
});

function report(over: Partial<Report> = {}): Report {
  const findings = { target: "default_flag", target_rate: 0.126, n_rows: 5000, n_tested: 118, findings: [], redundant_pairs: [], quality_notes: [], not_reported: {} };
  return {
    target: "default_flag", n_rows: 5000, plan: [], findings,
    fintech: { findings: { ...findings, findings: [finding("interest_rate", { kind: "lender_decision" }), finding("credit_score", { direction: "-" })] },
      variable_strength: [], lender_decision_columns: ["interest_rate"], protected_attributes: {}, proxies: [], notes: [] },
    charts: [], narrative: { headline: "h", sections: [], caveats: [] }, narrative_source: "llm",
    verification: { status: "passed", revisions: 0, remaining_issues: [] }, tokens: { limit: 10000, used: 2900, calls: 2 },
    notes: [], seconds: 3, ...over,
  };
}

describe("verificationBadge", () => {
  it("describes each verification outcome", () => {
    expect(verificationBadge(report()).label).toBe("Every number verified");
    const revised = verificationBadge(report({ verification: { status: "passed_after_revision", revisions: 1, remaining_issues: [] } }));
    expect(revised.detail).toContain("once");
    expect(verificationBadge(report({ verification: { status: "repaired", revisions: 2, remaining_issues: [] } })).label).toBe(
      "Unverified sentences removed",
    );
    expect(verificationBadge(report({ narrative_source: "template" })).detail).toContain("without AI");
  });
});

describe("findingIds", () => {
  it("numbers findings exactly as the narrator saw them (after the lender-decision rule)", () => {
    const ids = findingIds(report());
    expect([...ids.keys()]).toEqual(["F1", "F2"]);
    expect(ids.get("F1")?.kind).toBe("lender_decision");
    expect(ids.get("F2")?.column).toBe("credit_score");
  });
});

describe("keyProblem", () => {
  it("explains why a report was written without AI", () => {
    expect(keyProblem(report())).toBeNull();
    const rejected = report({ notes: ["narrator unavailable, used template report (AllProvidersFailed: all LLM providers failed: groq: key rejected (401))"] });
    expect(keyProblem(rejected)).toMatch(/key was rejected/);
    expect(keyProblem(report({ notes: ["narrator unavailable, used template report (BudgetExceeded: x)"] }))).toMatch(/budget/);
  });
});

describe("formatting", () => {
  it("formats numbers for people", () => {
    expect(pct(0.1258)).toBe("12.6%");
    expect(pct(null)).toBe("–");
    expect(formatBytes(3_718_791)).toBe("3.5 MB");
    expect(formatBytes(2048)).toBe("2 KB");
    expect(directionLabel(finding("x", { direction: "-" }))).toBe("higher → fewer");
    expect(isActive({ status: "queued" })).toBe(true);
    expect(isActive({ status: "failed" })).toBe(false);
  });
});

describe("errorMessage", () => {
  it("uses plain language and the API's own detail when specific", () => {
    expect(errorMessage(429, "you already have 2 runs in progress; wait for one to finish")).toBe(
      "You already have 2 runs in progress; wait for one to finish.",
    );
    expect(errorMessage(422, "target column 'x' is not in the file")).toBe("Target column 'x' is not in the file.");
    expect(errorMessage(503)).toMatch(/unavailable/);
    expect(errorMessage(401)).toMatch(/sign in/);
  });
});

describe("tokenUsage", () => {
  it("shows the budget when AI was used, and says so plainly when it wasn't", () => {
    expect(tokenUsage(report())).toBe("2,900 / 10,000");
    expect(tokenUsage({ narrative_source: "template", tokens: { limit: 0, used: 0, calls: 0 } })).toBe("None (no AI used)");
    expect(tokenUsage({ narrative_source: "template", tokens: { limit: 10000, used: 0, calls: 0 } })).toBe("None (no AI used)");
  });
});

describe("band labels", () => {
  it("shows 'suspicious' as very strong with a leakage check, and explains every band", () => {
    expect(bandInfo("suspicious").label).toBe("Very strong · check for leakage");
    expect(bandInfo("suspicious").tip).toMatch(/leakage/);
    expect(Object.keys(BAND_INFO)).toEqual(["useless", "weak", "medium", "strong", "suspicious"]);
    expect(Object.values(BAND_INFO).every((b) => b.tip.length > 0)).toBe(true);
    expect(bandInfo("unknown")).toEqual({ label: "unknown", tip: "" });
  });
});
