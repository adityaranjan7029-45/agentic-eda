"use client";

import { ChevronRight, FileSpreadsheet, Inbox, Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { StatusBadge } from "@/components/StatusBadge";
import { errorFromResponse } from "@/lib/errors";
import { formatBytes, isActive } from "@/lib/report";
import type { RunSummary } from "@/lib/types";

function when(iso: string): string {
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}

export function RunHistory({ runs }: { runs: RunSummary[] }) {
  const router = useRouter();
  const [confirming, setConfirming] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const anyActive = runs.some(isActive);

  useEffect(() => {
    if (!anyActive) return;
    const timer = setInterval(() => router.refresh(), 3000); // keep statuses fresh while something runs
    return () => clearInterval(timer);
  }, [anyActive, router]);

  async function remove(id: string) {
    setError(null);
    const response = await fetch(`/api/backend/runs/${id}`, { method: "DELETE" });
    setConfirming(null);
    if (!response.ok) setError(await errorFromResponse(response));
    router.refresh();
  }

  return (
    <section className="card p-0" aria-labelledby="history-title">
      <div className="flex items-baseline justify-between px-6 pt-6">
        <h2 id="history-title" className="text-lg font-semibold tracking-tight">
          Recent analyses
        </h2>
        <span className="text-sm text-slate-500">{runs.length === 0 ? "" : `${runs.length} total`}</span>
      </div>
      {error && (
        <p role="alert" className="mx-6 mt-3 rounded-xl bg-rose-50 px-4 py-2 text-sm text-rose-700 dark:bg-rose-500/10 dark:text-rose-300">
          {error}
        </p>
      )}
      {runs.length === 0 ? (
        <div className="flex flex-col items-center px-6 py-12 text-center">
          <span className="grid h-12 w-12 place-items-center rounded-2xl bg-slate-100 text-slate-400 dark:bg-slate-800">
            <Inbox className="h-6 w-6" />
          </span>
          <p className="mt-3 font-medium">No analyses yet</p>
          <p className="mt-1 text-sm text-slate-500">Upload a CSV or try the sample dataset to see your first report.</p>
        </div>
      ) : (
        <ul className="mt-4 divide-y divide-slate-100 dark:divide-slate-800">
          {runs.map((run) => (
            <li key={run.id} className="group flex flex-wrap items-center gap-4 px-6 py-4 transition hover:bg-slate-50/80 dark:hover:bg-slate-800/40">
              <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-slate-100 text-slate-500 dark:bg-slate-800">
                <FileSpreadsheet className="h-5 w-5" />
              </span>
              <Link href={`/runs/${run.id}`} className="min-w-0 flex-1">
                <p className="truncate font-medium group-hover:text-brand-600">{run.filename}</p>
                <p className="mt-0.5 text-xs text-slate-500">
                  {when(run.created_at)} · {run.n_rows?.toLocaleString() ?? "?"} rows × {run.n_columns ?? "?"} cols ·{" "}
                  {formatBytes(run.size_bytes)}
                  {run.target && <> · outcome <span className="font-mono">{run.target}</span></>}
                </p>
              </Link>
              <StatusBadge status={run.status} />
              {!isActive(run) &&
                (confirming === run.id ? (
                  <span className="flex items-center gap-1">
                    <button className="btn px-3 py-1.5 text-rose-700 hover:bg-rose-50 dark:text-rose-300 dark:hover:bg-rose-500/10" onClick={() => void remove(run.id)}>
                      Delete
                    </button>
                    <button className="btn-ghost" onClick={() => setConfirming(null)}>
                      Cancel
                    </button>
                  </span>
                ) : (
                  <button className="btn-ghost hover:text-rose-600" aria-label={`Delete ${run.filename}`} onClick={() => setConfirming(run.id)}>
                    <Trash2 className="h-4 w-4" />
                  </button>
                ))}
              <Link href={`/runs/${run.id}`} aria-hidden tabIndex={-1} className="hidden text-slate-300 group-hover:text-brand-500 sm:block">
                <ChevronRight className="h-5 w-5" />
              </Link>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
