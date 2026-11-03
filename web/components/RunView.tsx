"use client";

import { AlertTriangle, ArrowLeft, KeyRound, Rows3, Target } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { ProgressTracker } from "@/components/ProgressTracker";
import { ReportView } from "@/components/ReportView";
import { StatusBadge } from "@/components/StatusBadge";
import { applyEvent, initialProgress, type ProgressEvent } from "@/lib/progress";
import { isActive } from "@/lib/report";
import type { RunDetail } from "@/lib/types";

/** Follows a run live (SSE through the proxy), then shows its report. */
export function RunView({ initial }: { initial: RunDetail }) {
  const [run, setRun] = useState(initial);
  const [progress, setProgress] = useState(() => initialProgress(initial.status));
  const active = isActive(run);

  useEffect(() => {
    if (!isActive(initial)) return;
    // EventSource reconnects by itself and resends Last-Event-ID, so missed stages are replayed.
    const source = new EventSource(`/api/backend/runs/${initial.id}/events`);
    const onEvent = (message: MessageEvent<string>) => {
      const event = JSON.parse(message.data) as ProgressEvent;
      setProgress((p) => applyEvent(p, event));
      if (event.status === "succeeded" || event.status === "failed") {
        source.close();
        void fetch(`/api/backend/runs/${initial.id}`, { cache: "no-store" })
          .then((r) => (r.ok ? r.json() : null))
          .then((detail: RunDetail | null) => detail && setRun(detail));
      }
    };
    for (const stage of ["queued", "started", "plan", "analyse", "narrate", "critic", "repair", "finalize", "done"]) {
      source.addEventListener(stage, onEvent as EventListener);
    }
    return () => source.close();
  }, [initial]);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <Link href="/runs" className="btn-ghost -ml-2.5 text-sm">
        <ArrowLeft className="h-4 w-4" /> All analyses
      </Link>
      <div className="mt-3 flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="truncate text-2xl font-bold tracking-tight sm:text-3xl">{run.filename}</h1>
          <div className="mt-2 flex flex-wrap gap-2 text-xs">
            <span className="pill bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300">
              <Rows3 className="h-3.5 w-3.5" /> {run.n_rows?.toLocaleString()} rows × {run.n_columns} columns
            </span>
            <span className="pill bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300">
              <Target className="h-3.5 w-3.5" /> {run.target ? <span className="font-mono">{run.target}</span> : "no outcome column"}
            </span>
            {run.key_source === "user" && (
              <span className="pill bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300">
                <KeyRound className="h-3.5 w-3.5" /> your API key
              </span>
            )}
          </div>
        </div>
        <StatusBadge status={run.status} />
      </div>

      {(active || (!run.report && run.status !== "failed")) && (
        <section className="card mt-6" aria-live="polite">
          <ProgressTracker progress={progress} />
        </section>
      )}

      {run.status === "failed" && (
        <section role="alert" className="card mt-6 border-rose-200 dark:border-rose-500/30">
          <div className="flex gap-3">
            <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-rose-50 text-rose-600 dark:bg-rose-500/10">
              <AlertTriangle className="h-5 w-5" />
            </span>
            <div>
              <h2 className="font-semibold">This analysis didn&apos;t finish</h2>
              <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">{failureMessage(run.error)}</p>
              <Link href="/runs" className="btn-secondary mt-4">
                Start a new analysis
              </Link>
            </div>
          </div>
        </section>
      )}

      {run.report && <ReportView report={run.report} />}
    </div>
  );
}

function failureMessage(error: string | null): string {
  if (!error) return "Something went wrong during the analysis.";
  if (error.startsWith("WorkerLost") || error.startsWith("ServerRestarted"))
    return "The server restarted while this ran. Your file was deleted for privacy, so please upload it again.";
  if (error.startsWith("QueueUnavailable")) return "The analysis queue was unavailable. Please try again in a minute.";
  return "The analysis hit an error with this file. Your file was deleted for privacy; please check it and try again.";
}
