import { describe, expect, it } from "vitest";

import { applyEvent, initialProgress, stepState, STEPS, type ProgressEvent } from "../lib/progress";

const ev = (stage: string, status = "running", detail = ""): ProgressEvent => ({ stage, status, detail });

function run(events: ProgressEvent[]) {
  return events.reduce(applyEvent, initialProgress());
}

describe("progress tracking", () => {
  it("follows the engine's stages through the visible steps", () => {
    const p = run([ev("queued", "queued"), ev("started"), ev("plan", "running", "llm: missing_indicators"), ev("analyse", "running", "13 findings")]);
    expect(p.current).toBe("analyse");
    expect(stepState(p, "queued")).toBe("done");
    expect(stepState(p, "plan")).toBe("done");
    expect(stepState(p, "analyse")).toBe("active");
    expect(stepState(p, "narrate")).toBe("pending");
    expect(p.details).toEqual({ plan: "llm: missing_indicators", analyse: "13 findings" });
  });

  it("marks every step done when the run succeeds", () => {
    const p = run(["queued", "started", "plan", "analyse", "narrate", "critic", "finalize"].map((s) => ev(s)).concat(ev("done", "succeeded", "passed")));
    expect(p.finished).toBe(true);
    expect(STEPS.map((s) => stepState(p, s.key))).toEqual(STEPS.map(() => "done"));
  });

  it("shows where a failed run stopped", () => {
    const p = run([ev("queued", "queued"), ev("started"), ev("plan"), ev("analyse"), ev("done", "failed", "MemoryError")]);
    expect(stepState(p, "analyse")).toBe("done");
    expect(stepState(p, "done")).toBe("failed");
  });

  it("treats repair and finalize as part of checking, and ignores unknown stages", () => {
    const p = run([ev("critic"), ev("repair", "running", "removed 1 sentence"), ev("mystery")]);
    expect(p.current).toBe("critic");
    expect(p.details.critic).toBe("removed 1 sentence");
  });

  it("starts finished for runs that already ended", () => {
    expect(initialProgress("succeeded").finished).toBe(true);
    expect(initialProgress("running").finished).toBe(false);
  });
});
