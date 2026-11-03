import { Check, X } from "lucide-react";

import { STEPS, stepState, type Progress } from "@/lib/progress";

export function ProgressTracker({ progress }: { progress: Progress }) {
  const states = STEPS.map((s) => stepState(progress, s.key));
  const done = states.filter((s) => s === "done").length;
  const percent = Math.round((done / (STEPS.length - 1)) * 100);
  return (
    <div>
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium">
          {progress.finished ? (progress.status === "failed" ? "Stopped" : "Finished") : "Analysing your data…"}
        </span>
        <span className="tabular-nums text-slate-500">{Math.min(percent, 100)}%</span>
      </div>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
        <div
          className={`h-full rounded-full transition-all duration-700 ${
            progress.status === "failed" ? "bg-rose-500" : "bg-gradient-to-r from-brand-500 to-indigo-500"
          }`}
          style={{ width: `${Math.min(percent, 100)}%` }}
        />
      </div>
      <ol className="mt-6 grid gap-4 sm:grid-cols-6" aria-label="Analysis progress">
        {STEPS.map((step, i) => {
          const state = states[i];
          return (
            <li key={step.key} className="flex items-start gap-3 sm:flex-col sm:items-center sm:text-center" aria-current={state === "active" ? "step" : undefined}>
              <span
                className={`relative grid h-9 w-9 shrink-0 place-items-center rounded-full text-sm font-semibold transition ${
                  state === "done"
                    ? "bg-emerald-500 text-white"
                    : state === "active"
                      ? "bg-brand-600 text-white"
                      : state === "failed"
                        ? "bg-rose-600 text-white"
                        : "bg-slate-100 text-slate-400 dark:bg-slate-800"
                }`}
              >
                {state === "active" && <span className="absolute inset-0 animate-ping rounded-full bg-brand-500/40" />}
                {state === "done" ? <Check className="h-4 w-4" /> : state === "failed" ? <X className="h-4 w-4" /> : i + 1}
              </span>
              <span>
                <span className={`block text-sm font-medium ${state === "pending" ? "text-slate-400" : ""}`}>{step.label}</span>
                <span className="block text-xs text-slate-500">{progress.details[step.key] ?? (state === "active" ? step.hint : "")}</span>
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
