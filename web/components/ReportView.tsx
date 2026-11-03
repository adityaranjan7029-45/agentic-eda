"use client";

import {
  AlertTriangle,
  BadgeCheck,
  BarChart3,
  Coins,
  Database,
  Info,
  Percent,
  Scale,
  ShieldAlert,
  Sparkles,
  TableProperties,
  TrendingUp,
} from "lucide-react";
import type { ComponentType, ReactNode } from "react";

import { VegaChart } from "@/components/VegaChart";
import { KIND_LABEL, bandInfo, directionLabel, findingIds, keyProblem, pct, tokenUsage, verificationBadge } from "@/lib/report";
import type { Chart, Report } from "@/lib/types";

const TONE = {
  good: "bg-emerald-50 text-emerald-800 ring-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-200 dark:ring-emerald-500/30",
  info: "bg-brand-50 text-brand-700 ring-brand-200 dark:bg-brand-500/10 dark:text-brand-200 dark:ring-brand-700/40",
  warn: "bg-amber-50 text-amber-800 ring-amber-200 dark:bg-amber-500/10 dark:text-amber-200 dark:ring-amber-500/30",
};

const BAND = {
  useless: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300",
  weak: "bg-sky-50 text-sky-700 dark:bg-sky-500/10 dark:text-sky-300",
  medium: "bg-brand-50 text-brand-700 dark:bg-brand-500/15 dark:text-brand-200",
  strong: "bg-indigo-100 text-indigo-800 dark:bg-indigo-500/20 dark:text-indigo-200",
  suspicious: "bg-amber-50 text-amber-800 ring-1 ring-amber-200 dark:bg-amber-500/15 dark:text-amber-200 dark:ring-amber-500/30",
} as Record<string, string>;

const SEVERITY = {
  critical: { bar: "bg-rose-500", Icon: ShieldAlert, icon: "text-rose-500" },
  warning: { bar: "bg-amber-500", Icon: AlertTriangle, icon: "text-amber-500" },
  info: { bar: "bg-slate-300 dark:bg-slate-600", Icon: Info, icon: "text-slate-400" },
};

const WIDE = new Set(["correlation_matrix", "variable_strength"]);

export function ReportView({ report }: { report: Report }) {
  const badge = verificationBadge(report);
  const ids = findingIds(report);
  const charts = new Map(report.charts.map((c) => [c.id, c]));
  const usedCharts = new Set(report.narrative.sections.flatMap((s) => s.charts));
  const otherCharts = report.charts.filter((c) => !usedCharts.has(c.id));
  const problem = keyProblem(report);
  const fintech = report.fintech;
  const fairness = fintech.notes.filter((n) => /protected attribute|proxy/.test(n.message));
  const quality = report.findings.quality_notes.filter((n) => n.severity !== "info");
  const drivers = fintech.findings.findings.filter((f) => f.kind === "driver").length;
  const maxGini = Math.max(0.01, ...[...ids.values()].map((f) => f.gini));
  const maxIv = Math.max(0.01, ...fintech.variable_strength.slice(0, 12).map((v) => v.iv));

  const nav = [
    ["summary", "Summary"],
    ["insights", "Insights"],
    otherCharts.length ? ["charts", "Charts"] : null,
    ids.size ? ["evidence", "Evidence"] : null,
    fintech.variable_strength.length ? ["scorecard", "Scorecard"] : null,
    fairness.length || quality.length ? ["checks", "Fairness & quality"] : null,
  ].filter(Boolean) as [string, string][];

  return (
    <article className="mt-6 grid min-w-0 grid-cols-1 gap-6">
      <nav
        aria-label="Report sections"
        className="sticky top-16 z-20 -mx-4 flex gap-1 overflow-x-auto border-b border-slate-200/70 bg-slate-50/85 px-4 py-2 backdrop-blur sm:mx-0 sm:rounded-xl sm:border sm:px-2 dark:border-slate-800 dark:bg-slate-950/85"
      >
        {nav.map(([id, label]) => (
          <a key={id} href={`#${id}`} className="btn-ghost shrink-0 text-sm">
            {label}
          </a>
        ))}
      </nav>

      <section id="summary" className="scroll-mt-32 overflow-hidden rounded-2xl bg-gradient-to-br from-brand-600 via-brand-700 to-indigo-800 p-px shadow-lg shadow-brand-900/20">
        <div className="rounded-[15px] bg-white p-6 sm:p-8 dark:bg-slate-900">
          <p className="eyebrow flex items-center gap-1.5">
            <Sparkles className="h-3.5 w-3.5" /> Summary
          </p>
          <h2 className="mt-2 break-words text-2xl font-bold leading-snug tracking-tight sm:text-3xl">{report.narrative.headline}</h2>
          <div className={`mt-5 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-xl px-4 py-3 text-sm ring-1 ${TONE[badge.tone]}`}>
            <BadgeCheck className="h-5 w-5 shrink-0" />
            <strong>{badge.label}</strong>
            <span className="opacity-90">{badge.detail}</span>
          </div>
          {problem && (
            <p role="alert" className={`mt-3 flex items-start gap-2 rounded-xl px-4 py-3 text-sm ring-1 ${TONE.warn}`}>
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              {problem}
            </p>
          )}
          <dl className="mt-6 grid grid-cols-2 gap-3 lg:grid-cols-5">
            <Kpi icon={Database} label="Rows analysed" value={report.n_rows.toLocaleString("en-US")} />
            <Kpi icon={TableProperties} label="Columns tested" value={String(report.findings.n_tested)} />
            <Kpi icon={Percent} label="Outcome rate" value={report.findings.target ? pct(report.findings.target_rate) : "–"} />
            <Kpi icon={TrendingUp} label="Independent drivers" value={String(drivers)} />
            <Kpi icon={Coins} label="AI tokens" value={tokenUsage(report)} />
          </dl>
        </div>
      </section>

      <div id="insights" className="grid min-w-0 scroll-mt-32 grid-cols-1 gap-6">
        {report.narrative.sections.map((section, i) => {
          const sectionCharts = section.charts.map((c) => charts.get(c)).filter(Boolean) as Chart[];
          return (
            <section key={i} className="card">
              <div className="flex min-w-0 gap-4">
                <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-brand-50 text-sm font-bold text-brand-700 dark:bg-brand-500/15 dark:text-brand-200">
                  {i + 1}
                </span>
                <div className="min-w-0 flex-1">
                  <h3 className="text-lg font-semibold tracking-tight">{section.title}</h3>
                  <p className="mt-2 max-w-3xl leading-relaxed text-slate-700 dark:text-slate-300">{section.text}</p>
                  {section.findings.length > 0 && (
                    <p className="mt-3 flex flex-wrap gap-2">
                      {section.findings.map((fid) => (
                        <a
                          key={fid}
                          href={`#${fid}`}
                          title={ids.get(fid)?.summary}
                          className="pill bg-slate-100 font-mono text-slate-700 ring-1 ring-slate-200 transition hover:bg-brand-50 hover:text-brand-700 hover:ring-brand-200 dark:bg-slate-800 dark:text-slate-300 dark:ring-slate-700"
                        >
                          {fid} · {ids.get(fid)?.column}
                        </a>
                      ))}
                    </p>
                  )}
                </div>
              </div>
              {sectionCharts.length > 0 && <ChartGrid charts={sectionCharts} />}
            </section>
          );
        })}
      </div>

      {otherCharts.length > 0 && (
        <Panel id="charts" icon={BarChart3} title="More charts" subtitle="Built from the verified evidence; hover for exact values.">
          <ChartGrid charts={otherCharts} />
        </Panel>
      )}

      {ids.size > 0 && (
        <Panel
          id="evidence"
          icon={TableProperties}
          title="Evidence"
          subtitle="Every finding passed false-discovery control and the effect-size floor. Tentative ones are close to the noise level."
        >
          <div className="-mx-6 overflow-x-auto px-6">
            <table className="w-full min-w-[720px] text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-slate-500">
                <tr className="border-b border-slate-200 dark:border-slate-800">
                  <th className="py-2.5 pr-3 font-medium">ID</th>
                  <th className="py-2.5 pr-3 font-medium">Variable</th>
                  <th className="py-2.5 pr-3 font-medium">Kind</th>
                  <th className="w-40 py-2.5 pr-3 font-medium">Strength (Gini)</th>
                  <th className="py-2.5 pr-3 font-medium">Direction</th>
                  <th className="py-2.5 font-medium">What the data shows</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {[...ids].map(([fid, f]) => (
                  <tr key={fid} id={fid} className="scroll-mt-32 align-top transition target:bg-brand-50 dark:target:bg-brand-500/10">
                    <td className="py-3 pr-3 font-mono text-xs text-slate-500">{fid}</td>
                    <td className="py-3 pr-3 font-medium">
                      <span className="font-mono text-[13px]">{f.column}</span>
                      {f.confidence === "tentative" && (
                        <span className="pill ml-2 bg-amber-100 text-amber-800 dark:bg-amber-500/20 dark:text-amber-200">tentative</span>
                      )}
                    </td>
                    <td className="py-3 pr-3">
                      {KIND_LABEL[f.kind]}
                      {f.related_to && <span className="block text-xs text-slate-500">of {f.related_to}</span>}
                    </td>
                    <td className="py-3 pr-3">
                      <Bar value={f.gini} max={maxGini} label={f.gini.toFixed(2)} />
                    </td>
                    <td className="py-3 pr-3 whitespace-nowrap">{directionLabel(f)}</td>
                    <td className="py-3 text-slate-600 dark:text-slate-400">{f.summary}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      {fintech.variable_strength.length > 0 && (
        <Panel
          id="scorecard"
          icon={Scale}
          title="Scorecard view"
          subtitle="Information Value and KS: the measures credit-risk teams use to rate a variable. Hover a band for what it means."
        >
          <div className="-mx-6 overflow-x-auto px-6">
            <table className="w-full min-w-[560px] text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-slate-500">
                <tr className="border-b border-slate-200 dark:border-slate-800">
                  <th className="py-2.5 pr-3 font-medium">Variable</th>
                  <th className="w-56 py-2.5 pr-3 font-medium">Information Value</th>
                  <th className="py-2.5 pr-3 font-medium">Band</th>
                  <th className="py-2.5 font-medium">KS</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {fintech.variable_strength.slice(0, 12).map((v) => (
                  <tr key={v.column}>
                    <td className="py-3 pr-3">
                      <span className="font-mono text-[13px] font-medium">{v.column}</span>
                      {fintech.lender_decision_columns.includes(v.column) && (
                        <span className="pill ml-2 bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300">lender decision</span>
                      )}
                    </td>
                    <td className="py-3 pr-3">
                      <Bar value={v.iv} max={maxIv} label={v.iv.toFixed(3)} />
                    </td>
                    <td className="py-3 pr-3">
                      <span
                        className={`pill cursor-help whitespace-nowrap ${BAND[v.band] ?? BAND.useless}`}
                        title={bandInfo(v.band).tip}
                      >
                        {v.band === "suspicious" && <AlertTriangle className="h-3 w-3" aria-hidden />}
                        {bandInfo(v.band).label}
                      </span>
                    </td>
                    <td className="py-3 tabular-nums">{v.ks === null ? "–" : v.ks.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      {(fairness.length > 0 || quality.length > 0) && (
        <div id="checks" className="grid min-w-0 scroll-mt-32 grid-cols-1 gap-6 md:grid-cols-2">
          {fairness.length > 0 && (
            <NoteList icon={Scale} title="Fair lending" subtitle="Protected attributes must not drive credit decisions, directly or through proxies." notes={fairness} />
          )}
          {quality.length > 0 && <NoteList icon={ShieldAlert} title="Data quality" subtitle="Problems found and handled before the analysis." notes={quality} />}
        </div>
      )}

      {report.narrative.caveats.length > 0 && (
        <aside className="flex gap-3 rounded-2xl border border-dashed border-slate-300 p-5 text-sm text-slate-600 dark:border-slate-700 dark:text-slate-400">
          <Info className="mt-0.5 h-4 w-4 shrink-0" />
          <div className="grid gap-1">
            {report.narrative.caveats.map((c, i) => (
              <p key={i}>{c}</p>
            ))}
          </div>
        </aside>
      )}
    </article>
  );
}

function ChartGrid({ charts }: { charts: Chart[] }) {
  return (
    <div className="mt-5 grid min-w-0 grid-cols-1 gap-4 md:grid-cols-2">
      {charts.map((c) => (
        <div key={c.id} className={WIDE.has(c.kind) ? "md:col-span-2" : ""}>
          <VegaChart spec={c.spec} title={c.title} tall={WIDE.has(c.kind)} />
        </div>
      ))}
    </div>
  );
}

function Kpi({ icon: Icon, label, value }: { icon: ComponentType<{ className?: string }>; label: string; value: string }) {
  return (
    <div className="rounded-xl bg-slate-50 p-3.5 ring-1 ring-slate-200/70 dark:bg-slate-800/60 dark:ring-slate-700/60">
      <dt className="flex items-center gap-1.5 text-xs font-medium text-slate-500">
        <Icon className="h-3.5 w-3.5" />
        {label}
      </dt>
      <dd className="mt-1 text-lg font-bold tracking-tight tabular-nums">{value}</dd>
    </div>
  );
}

function Bar({ value, max, label }: { value: number; max: number; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
        <div className="h-full rounded-full bg-gradient-to-r from-brand-400 to-brand-600" style={{ width: `${Math.max(4, (value / max) * 100)}%` }} />
      </div>
      <span className="w-12 text-right tabular-nums text-slate-600 dark:text-slate-400">{label}</span>
    </div>
  );
}

function Panel({
  id,
  icon: Icon,
  title,
  subtitle,
  children,
}: {
  id: string;
  icon: ComponentType<{ className?: string }>;
  title: string;
  subtitle: string;
  children: ReactNode;
}) {
  return (
    <section id={id} className="card scroll-mt-32">
      <div className="flex items-start gap-3">
        <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400">
          <Icon className="h-[18px] w-[18px]" />
        </span>
        <div>
          <h3 className="text-lg font-semibold tracking-tight">{title}</h3>
          <p className="text-sm text-slate-500">{subtitle}</p>
        </div>
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function NoteList({
  icon,
  title,
  subtitle,
  notes,
}: {
  icon: ComponentType<{ className?: string }>;
  title: string;
  subtitle: string;
  notes: Report["fintech"]["notes"];
}) {
  return (
    <Panel id={`${title}-panel`} icon={icon} title={title} subtitle={subtitle}>
      <ul className="grid gap-2.5 text-sm">
        {notes.map((n, i) => {
          const s = SEVERITY[n.severity];
          return (
            <li key={i} className="flex gap-3 rounded-xl bg-slate-50 p-3 dark:bg-slate-800/50">
              <s.Icon className={`mt-0.5 h-4 w-4 shrink-0 ${s.icon}`} />
              <span>
                {n.column && <span className="font-mono text-[13px] font-semibold">{n.column}: </span>}
                {n.message}
              </span>
            </li>
          );
        })}
      </ul>
    </Panel>
  );
}
