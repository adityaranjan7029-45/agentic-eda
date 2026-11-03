import { BarChart3, CheckCircle2, FileSearch, Scale, ShieldCheck, Sparkles, Upload } from "lucide-react";
import { redirect } from "next/navigation";

import { auth, signIn } from "@/auth";

const PROOF = [
  { value: "100%", label: "planted risk drivers found", note: "held-out data, 5,000+ rows" },
  { value: "2.4%", label: "false-discovery share", note: "target ≤ 5%" },
  { value: "Every", label: "number checked", note: "by a deterministic critic" },
  { value: "400+", label: "automated tests", note: "real Postgres + Redis in CI" },
];

const FEATURES = [
  {
    icon: BarChart3,
    title: "Statistics find the insights",
    text: "Every column is tested with false-discovery control and an effect-size floor, so noise never becomes a finding.",
  },
  {
    icon: ShieldCheck,
    title: "Every number is verified",
    text: "The AI only writes up verified findings. A checker rejects any number that isn't in the evidence for that variable.",
  },
  {
    icon: Scale,
    title: "Built for credit risk",
    text: "Information Value, KS curves, lender-decision variables and fair-lending proxy checks, out of the box.",
  },
];

const STEPS = [
  { icon: Upload, title: "Upload a CSV", text: "Or try a synthetic credit dataset. Your file is deleted as soon as it's read." },
  { icon: FileSearch, title: "Watch it analyse", text: "Live progress: planning, statistics, writing, checking." },
  { icon: CheckCircle2, title: "Read a verified report", text: "Findings, interactive charts, scorecard metrics and fairness warnings." },
];

function GitHubMark() {
  return (
    <svg aria-hidden viewBox="0 0 16 16" className="h-5 w-5 fill-current">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

function SignIn({ large = false }: { large?: boolean }) {
  return (
    <form
      action={async () => {
        "use server";
        await signIn("github", { redirectTo: "/runs" });
      }}
    >
      <button type="submit" className={`btn-primary ${large ? "px-6 py-3.5 text-base" : ""}`}>
        <GitHubMark />
        Sign in with GitHub
      </button>
    </form>
  );
}

/** A miniature, static preview of a report: an example from a real run on the synthetic dataset, labelled as such. */
function ReportPreview() {
  const bars = [0.29, 0.24, 0.16, 0.16, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03];
  return (
    <figure className="mx-auto w-full max-w-md">
      <figcaption className="mb-3 flex items-center gap-2 text-xs text-slate-400">
        <span className="pill bg-white/10 font-semibold text-slate-200 ring-1 ring-white/15">Example</span>
        From a report on the synthetic credit dataset, not live data
      </figcaption>
      <div className="relative w-full rotate-1 rounded-2xl border border-white/10 bg-white/95 p-5 text-slate-900 shadow-2xl shadow-brand-900/40 dark:bg-slate-900/95 dark:text-slate-100">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-500">sample_credit_applications.csv</span>
          <span className="pill bg-emerald-50 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300">
            <CheckCircle2 className="h-3 w-3" /> Every number verified
          </span>
        </div>
        <p className="mt-3 text-[15px] font-semibold leading-snug">
          Credit score is the strongest applicant signal: 26.2% default in the bottom 20% vs 3.0% in the top 20%.
        </p>
        <div className="mt-4 flex h-24 items-end gap-1.5">
          {bars.map((h, i) => (
            <div key={i} className="flex-1 rounded-t bg-gradient-to-t from-brand-600 to-brand-400" style={{ height: `${(h / 0.3) * 100}%` }} />
          ))}
        </div>
        <div className="mt-2 flex justify-between text-[10px] text-slate-400">
          <span>low credit score</span>
          <span>high credit score</span>
        </div>
        <div className="mt-4 grid grid-cols-3 gap-2 text-center">
          {[
            ["IV", "0.63"],
            ["KS", "0.32"],
            ["Gini", "0.43"],
          ].map(([k, v]) => (
            <div key={k} className="rounded-lg bg-slate-50 py-1.5 dark:bg-slate-800">
              <div className="text-[10px] text-slate-500">{k}</div>
              <div className="text-sm font-bold">{v}</div>
            </div>
          ))}
        </div>
      </div>
    </figure>
  );
}

export default async function Home() {
  if ((await auth())?.user) redirect("/runs");
  return (
    <div>
      <section className="relative overflow-hidden bg-slate-950 text-white">
        <div className="bg-grid absolute inset-0 opacity-40" />
        <div className="absolute -top-40 left-1/2 h-[480px] w-[820px] -translate-x-1/2 rounded-full bg-brand-600/40 blur-[120px]" />
        <div className="relative mx-auto grid max-w-6xl items-center gap-12 px-4 py-20 sm:px-6 lg:grid-cols-2 lg:py-28">
          <div>
            <span className="pill bg-white/10 text-brand-200 ring-1 ring-white/15">
              <Sparkles className="h-3.5 w-3.5" /> Credit-risk EDA with verified AI reports
            </span>
            <h1 className="mt-5 text-4xl font-bold leading-[1.1] tracking-tight sm:text-6xl">
              Exploratory analysis{" "}
              <span className="bg-gradient-to-r from-brand-200 via-white to-brand-200 bg-clip-text text-transparent">you can trust</span>
            </h1>
            <p className="mt-5 max-w-xl text-lg text-slate-300">
              Upload a CSV and get a report on what drives your outcome. Statistics find the insights, the AI only writes them
              up, and a checker verifies every number against the data.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-4">
              <SignIn large />
              <span className="text-sm text-slate-400">Free · no dataset needed to try it</span>
            </div>
          </div>
          <ReportPreview />
        </div>
      </section>

      <section className="relative z-10 mx-auto -mt-10 max-w-6xl px-4 sm:px-6">
        <div className="card grid grid-cols-2 gap-6 lg:grid-cols-4">
          {PROOF.map((p) => (
            <div key={p.label}>
              <div className="text-3xl font-bold tracking-tight text-brand-600 dark:text-brand-400">{p.value}</div>
              <div className="mt-1 text-sm font-medium">{p.label}</div>
              <div className="text-xs text-slate-500">{p.note}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
        <p className="eyebrow text-center">Why it&apos;s different</p>
        <h2 className="mt-2 text-center text-3xl font-bold tracking-tight">AI that can&apos;t make up numbers</h2>
        <div className="mt-10 grid gap-5 md:grid-cols-3">
          {FEATURES.map((f) => (
            <div key={f.title} className="card">
              <span className="grid h-11 w-11 place-items-center rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-500/10 dark:text-brand-400">
                <f.icon className="h-5 w-5" />
              </span>
              <h3 className="mt-4 font-semibold">{f.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">{f.text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="border-y border-slate-200/70 bg-white py-20 dark:border-slate-800 dark:bg-slate-900/40">
        <div className="mx-auto max-w-6xl px-4 sm:px-6">
          <p className="eyebrow text-center">How it works</p>
          <h2 className="mt-2 text-center text-3xl font-bold tracking-tight">From CSV to verified report in seconds</h2>
          <ol className="mt-10 grid gap-8 md:grid-cols-3">
            {STEPS.map((s, i) => (
              <li key={s.title} className="relative text-center">
                <span className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-brand-500 to-indigo-600 text-white shadow-lg shadow-brand-600/30">
                  <s.icon className="h-6 w-6" />
                </span>
                <span className="mt-4 block text-xs font-semibold text-brand-600">STEP {i + 1}</span>
                <h3 className="mt-1 font-semibold">{s.title}</h3>
                <p className="mx-auto mt-2 max-w-xs text-sm text-slate-600 dark:text-slate-400">{s.text}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 py-20 text-center sm:px-6">
        <h2 className="text-3xl font-bold tracking-tight">See it on a real credit dataset</h2>
        <p className="mx-auto mt-3 max-w-xl text-slate-600 dark:text-slate-400">
          Sign in and press &ldquo;Try a sample credit dataset&rdquo;: a full report in under a minute, nothing to upload.
        </p>
        <div className="mt-6 flex justify-center">
          <SignIn />
        </div>
      </section>
    </div>
  );
}
