/**
 * Live progress: turns the SSE events from GET /runs/{id}/events into the
 * step-by-step tracker shown on the run page.
 */

export type ProgressEvent = { stage: string; detail: string; status: string; at?: number };
export type StepState = "pending" | "active" | "done" | "failed";

export const STEPS = [
  { key: "queued", label: "Queued", hint: "Waiting for a worker" },
  { key: "plan", label: "Planning", hint: "Choosing preprocessing steps" },
  { key: "analyse", label: "Analysing", hint: "Statistics, fintech checks, charts" },
  { key: "narrate", label: "Writing", hint: "Drafting the report from verified facts" },
  { key: "critic", label: "Checking", hint: "Verifying every number" },
  { key: "done", label: "Done", hint: "" },
] as const;

/** Engine stages that belong to a visible step ("started" -> Planning is about to begin, etc.). */
const STAGE_TO_STEP: Record<string, string> = {
  queued: "queued",
  started: "plan",
  plan: "plan",
  analyse: "analyse",
  narrate: "narrate",
  critic: "critic",
  repair: "critic",
  finalize: "critic",
  done: "done",
};

export type Progress = {
  current: string; // step key
  status: string; // queued | running | succeeded | failed
  details: Record<string, string>; // step key -> latest detail
  finished: boolean;
};

export const initialProgress = (status = "queued"): Progress => ({
  current: "queued",
  status,
  details: {},
  finished: status === "succeeded" || status === "failed",
});

export function applyEvent(progress: Progress, event: ProgressEvent): Progress {
  const step = STAGE_TO_STEP[event.stage];
  if (!step) return progress;
  const finished = event.status === "succeeded" || event.status === "failed";
  return {
    current: step,
    status: event.status,
    details: event.detail ? { ...progress.details, [step]: event.detail } : progress.details,
    finished: progress.finished || finished,
  };
}

export function stepState(progress: Progress, key: string): StepState {
  const order = STEPS.map((s) => s.key as string);
  const at = order.indexOf(progress.current);
  const index = order.indexOf(key);
  if (progress.finished && progress.status === "succeeded") return "done";
  if (progress.finished && progress.status === "failed") {
    return index < at ? "done" : index === at ? "failed" : "pending";
  }
  if (index < at) return "done";
  return index === at ? "active" : "pending";
}
