import { redirect } from "next/navigation";

import { auth } from "@/auth";
import { RunHistory } from "@/components/RunHistory";
import { UploadCard } from "@/components/UploadCard";
import { errorMessage } from "@/lib/errors";
import { apiGet } from "@/lib/server-api";
import type { RunSummary } from "@/lib/types";

export const metadata = { title: "Your analyses" };

export default async function RunsPage() {
  const result = await apiGet<{ runs: RunSummary[] }>("/runs?limit=50");
  if (result === null) redirect("/");
  const name = (await auth())?.user?.name?.split(" ")[0];
  return (
    <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">{name ? `Welcome back, ${name}` : "Your analyses"}</h1>
      <p className="mt-1 text-slate-500">Upload a dataset, or explore a sample. Every report is verified number by number.</p>
      <div className="mt-8 grid gap-6">
        <UploadCard />
        {result.ok ? (
          <RunHistory runs={result.data.runs} />
        ) : (
          <p role="alert" className="card text-sm text-rose-700 dark:text-rose-300">
            Couldn&apos;t load your analyses: {errorMessage(result.status)}
          </p>
        )}
      </div>
    </div>
  );
}
