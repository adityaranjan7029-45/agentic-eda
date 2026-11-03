"use client";

import { FileSpreadsheet, KeyRound, Loader2, Sparkles, UploadCloud, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

import { guessTarget, readHeader } from "@/lib/csv";
import { errorFromResponse } from "@/lib/errors";
import { formatBytes } from "@/lib/report";

/** maxUploadMb comes from the server's MAX_UPLOAD_MB (smaller on the 1 GB server). */
export function UploadCard({ maxUploadMb = 50 }: { maxUploadMb?: number }) {
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [target, setTarget] = useState("");
  const [groqKey, setGroqKey] = useState(""); // page memory only: never stored in the browser
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState<"upload" | "sample" | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function choose(chosen: File | undefined) {
    setError(null);
    if (!chosen) return;
    if (!chosen.name.toLowerCase().endsWith(".csv")) {
      setError("Only .csv files are supported.");
      return;
    }
    if (chosen.size > maxUploadMb * 1024 * 1024) {
      setError(`That file is ${formatBytes(chosen.size)}; the limit on this server is ${maxUploadMb} MB.`);
      return;
    }
    const header = await readHeader(chosen);
    if (header.length < 2) {
      setError("That file doesn't look like a CSV with a header row.");
      return;
    }
    setFile(chosen);
    setColumns(header);
    setTarget(guessTarget(header));
  }

  function clear() {
    setFile(null);
    setColumns([]);
    setTarget("");
    if (input.current) input.current.value = "";
  }

  async function send(path: string, body: BodyInit | undefined, kind: "upload" | "sample") {
    setBusy(kind);
    setError(null);
    try {
      const headers: HeadersInit = groqKey.trim() ? { "x-groq-key": groqKey.trim() } : {};
      const response = await fetch(path, { method: "POST", body, headers });
      if (!response.ok) {
        setError(await errorFromResponse(response));
        return;
      }
      const run = await response.json();
      setGroqKey("");
      router.push(`/runs/${run.id}`);
    } catch {
      setError("Couldn't reach the server. Check your connection and try again.");
    } finally {
      setBusy(null);
    }
  }

  function upload() {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    if (target) form.append("target", target);
    void send("/api/backend/runs", form, "upload");
  }

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
      <section className="card lg:col-span-3" aria-labelledby="upload-title">
        <p className="eyebrow">New analysis</p>
        <h2 id="upload-title" className="mt-1 text-xl font-semibold tracking-tight">
          Upload a dataset
        </h2>
        <p className="mt-1 text-sm text-slate-500">CSV up to {maxUploadMb} MB. The file is deleted as soon as the analysis has read it.</p>

        {file ? (
          <div className="mt-5 flex items-center gap-3 rounded-xl border border-brand-200 bg-brand-50/60 p-4 dark:border-brand-700/40 dark:bg-brand-500/10">
            <span className="grid h-10 w-10 place-items-center rounded-lg bg-white text-brand-600 shadow-sm dark:bg-slate-900">
              <FileSpreadsheet className="h-5 w-5" />
            </span>
            <div className="min-w-0 flex-1">
              <p className="truncate font-medium">{file.name}</p>
              <p className="text-xs text-slate-500">
                {columns.length} columns · {formatBytes(file.size)}
              </p>
            </div>
            <button type="button" className="btn-ghost" onClick={clear} aria-label="Remove file">
              <X className="h-4 w-4" />
            </button>
          </div>
        ) : (
          <div
            role="button"
            tabIndex={0}
            onClick={() => input.current?.click()}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              void choose(e.dataTransfer.files[0]);
            }}
            className={`mt-5 flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-10 text-center transition ${
              dragging
                ? "border-brand-500 bg-brand-50 dark:bg-brand-500/10"
                : "border-slate-300 bg-slate-50/60 hover:border-brand-400 hover:bg-brand-50/40 dark:border-slate-700 dark:bg-slate-900"
            }`}
          >
            <span className="grid h-12 w-12 place-items-center rounded-2xl bg-white text-brand-600 shadow-sm dark:bg-slate-800">
              <UploadCloud className="h-6 w-6" />
            </span>
            <p className="mt-3 font-medium">
              Drop a CSV here, or <span className="text-brand-600 dark:text-brand-400">browse</span>
            </p>
            <p className="mt-1 text-xs text-slate-500">Column names are read in your browser to pick the outcome</p>
          </div>
        )}
        <input ref={input} type="file" accept=".csv,text/csv" className="hidden" onChange={(e) => void choose(e.target.files?.[0])} />

        {file && (
          <div className="mt-5">
            <label htmlFor="target" className="label">
              Outcome column <span className="font-normal text-slate-500">(what you want to explain)</span>
            </label>
            <select id="target" className="input" value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">None: just show how the columns relate to each other</option>
              {columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        )}

        <details className="group mt-5 rounded-xl border border-slate-200 px-4 py-3 text-sm dark:border-slate-800">
          <summary className="flex cursor-pointer list-none items-center gap-2 font-medium">
            <KeyRound className="h-4 w-4 text-slate-400" />
            Use my own Groq API key
            <span className="ml-auto text-xs font-normal text-slate-400 group-open:hidden">optional</span>
          </summary>
          <div className="mt-3">
            <input
              id="groq-key"
              aria-label="Groq API key"
              type="password"
              autoComplete="off"
              spellCheck={false}
              placeholder="gsk_..."
              className="input font-mono"
              value={groqKey}
              onChange={(e) => setGroqKey(e.target.value)}
            />
            <p className="mt-2 text-xs leading-relaxed text-slate-500">
              Used for this run only, then discarded: never saved in your browser, our database or our logs. Only your key is
              used; if it fails, the report is written without AI and we tell you why. Tip: use a key you can revoke.
            </p>
          </div>
        </details>

        {error && (
          <p role="alert" className="mt-4 rounded-xl bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:bg-rose-500/10 dark:text-rose-300">
            {error}
          </p>
        )}

        <button type="button" className="btn-primary mt-5 w-full sm:w-auto" disabled={!file || busy !== null} onClick={upload}>
          {busy === "upload" && <Loader2 className="h-4 w-4 animate-spin" />}
          {busy === "upload" ? "Uploading…" : "Analyse dataset"}
        </button>
      </section>

      <section className="card relative overflow-hidden lg:col-span-2" aria-labelledby="sample-title">
        <div className="absolute -right-16 -top-16 h-48 w-48 rounded-full bg-brand-500/15 blur-3xl" />
        <p className="eyebrow">Quick start</p>
        <h2 id="sample-title" className="mt-1 text-xl font-semibold tracking-tight">
          No dataset? Try ours.
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
          A fresh synthetic credit-application dataset (3,000 applicants, 122 columns) with real-world traps: leakage,
          near-duplicate columns, a lender-priced variable and a fair-lending proxy.
        </p>
        <ul className="mt-4 grid gap-2 text-sm">
          {["Who is most likely to default, and by how much", "Interactive charts and scorecard metrics", "Fair-lending warnings"].map(
            (t) => (
              <li key={t} className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 shrink-0 text-brand-500" />
                {t}
              </li>
            ),
          )}
        </ul>
        <button
          type="button"
          className="btn-secondary mt-6 w-full"
          disabled={busy !== null}
          onClick={() => void send("/api/backend/runs/sample", undefined, "sample")}
        >
          {busy === "sample" && <Loader2 className="h-4 w-4 animate-spin" />}
          {busy === "sample" ? "Starting…" : "Try a sample credit dataset"}
        </button>
      </section>
    </div>
  );
}
