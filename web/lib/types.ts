/** Shapes returned by the FastAPI backend (see api/src/eda_api/schemas.py and engine AnalysisReport.to_dict). */

export type RunStatus = "queued" | "running" | "succeeded" | "failed";

export type RunSummary = {
  id: string;
  status: RunStatus;
  stage: string | null;
  filename: string;
  size_bytes: number;
  n_rows: number | null;
  n_columns: number | null;
  target: string | null;
  narrative_source: "llm" | "template" | null;
  verification_status: string | null;
  tokens_used: number;
  key_source: "user" | "server" | "none" | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};

export type RunDetail = RunSummary & { report: Report | null };

export type Finding = {
  kind: "driver" | "secondary_driver" | "lender_decision" | "correlation";
  column: string;
  direction: "+" | "-" | "mixed";
  gini: number;
  q_value: number;
  summary: string;
  evidence: Record<string, unknown>;
  related_to: string | null;
  correlation: number | null;
  confidence: "high" | "tentative";
};

export type Note = { severity: "info" | "warning" | "critical"; column: string | null; message: string };

export type FindingsReport = {
  target: string | null;
  target_rate: number | null;
  n_rows: number;
  n_tested: number;
  findings: Finding[];
  redundant_pairs: { columns: [string, string]; spearman: number; note: string }[];
  quality_notes: Note[];
  not_reported: Record<string, number>;
};

export type Chart = { id: string; finding_id: string | null; title: string; kind: string; spec: Record<string, unknown> };

export type Narrative = {
  headline: string;
  sections: { title: string; text: string; findings: string[]; charts: string[] }[];
  caveats: string[];
};

export type Report = {
  target: string | null;
  n_rows: number;
  plan: string[];
  findings: FindingsReport;
  fintech: {
    findings: FindingsReport; // findings after the lender-decision rule: these are the F1, F2... of the narrative
    variable_strength: { column: string; iv: number; band: string; ks: number | null }[];
    lender_decision_columns: string[];
    protected_attributes: Record<string, string>;
    proxies: { feature: string; protected: string; strength: number; measure: string }[];
    notes: Note[];
  };
  charts: Chart[];
  narrative: Narrative;
  narrative_source: "llm" | "template";
  verification: { status: string; revisions: number; remaining_issues: unknown[] };
  tokens: { limit: number; used: number; calls: number };
  notes: string[];
  seconds: number;
};
