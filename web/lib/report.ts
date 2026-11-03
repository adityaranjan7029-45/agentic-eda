/** Turn the API's report JSON into what the report page shows. */
import type { Finding, Report, RunSummary } from "./types";

export const pct = (x: number | null | undefined, digits = 1) =>
  x === null || x === undefined ? "–" : `${(100 * x).toFixed(digits)}%`;

export type Badge = { tone: "good" | "info" | "warn"; label: string; detail: string };

export function verificationBadge(report: Report): Badge {
  const v = report.verification;
  const source = report.narrative_source === "llm" ? "AI-written" : "written without AI";
  if (v.status === "passed") return { tone: "good", label: "Every number verified", detail: `Report ${source}; passed the checker first time.` };
  if (v.status === "passed_after_revision")
    return {
      tone: "good",
      label: "Every number verified",
      detail: `Report ${source}; the checker sent it back ${v.revisions === 1 ? "once" : `${v.revisions} times`} before it passed.`,
    };
  if (v.status === "repaired")
    return { tone: "info", label: "Unverified sentences removed", detail: "Sentences the checker couldn't verify were deleted." };
  return { tone: "warn", label: "Not fully verified", detail: "Some statements could not be verified." };
}

/** F1, F2, ... in the order the narrator saw them (findings after the lender-decision rule). */
export function findingIds(report: Report): Map<string, Finding> {
  return new Map(report.fintech.findings.findings.map((f, i) => [`F${i + 1}`, f] as const));
}

export const KIND_LABEL: Record<Finding["kind"], string> = {
  driver: "Driver",
  secondary_driver: "Restates another driver",
  lender_decision: "Lender decision",
  correlation: "Relationship",
};

export function directionLabel(f: Finding): string {
  if (f.kind === "correlation") return f.direction === "+" ? "move together" : "move oppositely";
  return f.direction === "+" ? "higher → more" : f.direction === "-" ? "higher → fewer" : "differs by group";
}

/** If the user's own key failed, say so plainly (the report was then written without AI). */
export function keyProblem(report: Report): string | null {
  const note = report.notes.find((n) => n.startsWith("narrator unavailable"));
  if (!note) return null;
  if (/key rejected/.test(note)) return "Your API key was rejected, so this report was written without AI. Check the key and try again.";
  if (/rate limited/.test(note)) return "The AI provider was rate-limited, so this report was written without AI.";
  if (/BudgetExceeded/.test(note)) return "The token budget ran out, so this report was written without AI.";
  return "The AI provider was unavailable, so this report was written without AI.";
}

export const STATUS_LABEL: Record<RunSummary["status"], string> = {
  queued: "Queued",
  running: "Running",
  succeeded: "Done",
  failed: "Failed",
};

export function isActive(run: Pick<RunSummary, "status">): boolean {
  return run.status === "queued" || run.status === "running";
}

/** "2,904 / 10,000", or a plain statement when no AI was used for this report. */
export function tokenUsage(report: Pick<Report, "tokens" | "narrative_source">): string {
  if (report.tokens.limit === 0 || (report.narrative_source === "template" && report.tokens.used === 0)) return "None (no AI used)";
  return `${report.tokens.used.toLocaleString("en-US")} / ${report.tokens.limit.toLocaleString("en-US")}`;
}

/**
 * How Information Value bands are shown. The engine keeps the industry's own word ("suspicious")
 * in the data; people see a less alarming label and a tooltip that explains it.
 */
export const BAND_INFO: Record<string, { label: string; tip: string }> = {
  useless: { label: "Useless", tip: "IV below 0.02: no meaningful predictive power." },
  weak: { label: "Weak", tip: "IV 0.02–0.1: some signal, usually too little to rely on." },
  medium: { label: "Medium", tip: "IV 0.1–0.3: a useful predictor." },
  strong: { label: "Strong", tip: "IV 0.3–0.5: a very useful predictor." },
  suspicious: {
    label: "Very strong · check for leakage",
    tip:
      "IV above 0.5. Genuinely strong variables (like a credit score) can reach this, but it is also the classic sign of " +
      "leakage: a column recorded after the outcome. Confirm the value is known at application time before relying on it.",
  },
};

export function bandInfo(band: string) {
  return BAND_INFO[band] ?? { label: band, tip: "" };
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
