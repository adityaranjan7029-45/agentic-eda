import { CheckCircle2, Clock, Loader2, XCircle } from "lucide-react";

import { STATUS_LABEL } from "@/lib/report";
import type { RunStatus } from "@/lib/types";

const STYLES: Record<RunStatus, { className: string; Icon: typeof Clock; spin?: boolean }> = {
  queued: { className: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300", Icon: Clock },
  running: { className: "bg-brand-50 text-brand-700 dark:bg-brand-500/15 dark:text-brand-200", Icon: Loader2, spin: true },
  succeeded: { className: "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300", Icon: CheckCircle2 },
  failed: { className: "bg-rose-50 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300", Icon: XCircle },
};

export function StatusBadge({ status }: { status: RunStatus }) {
  const { className, Icon, spin } = STYLES[status];
  return (
    <span className={`pill ${className}`}>
      <Icon className={`h-3.5 w-3.5 ${spin ? "animate-spin" : ""}`} aria-hidden />
      {STATUS_LABEL[status]}
    </span>
  );
}
