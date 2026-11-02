/**
 * Protected page: your run history, fetched server-side with a freshly signed API token.
 * Step 14 replaces this with the real interface (upload, live progress, report, charts).
 */
import Link from "next/link";
import { redirect } from "next/navigation";

import { currentApiUser } from "@/auth";
import { mintApiToken } from "@/lib/api-token";

type RunSummary = { id: string; status: string; filename: string; created_at: string };

async function loadRuns(): Promise<{ runs: RunSummary[]; error?: string }> {
  const user = await currentApiUser();
  if (!user) redirect("/");
  try {
    const response = await fetch(`${process.env.API_URL ?? "http://localhost:8000"}/runs?limit=20`, {
      headers: { authorization: `Bearer ${await mintApiToken(user, process.env.API_JWT_SECRET)}` },
      cache: "no-store",
    });
    if (!response.ok) return { runs: [], error: `API answered ${response.status}` };
    return { runs: (await response.json()).runs };
  } catch {
    return { runs: [], error: "the analysis service is unavailable" };
  }
}

export default async function RunsPage() {
  const { runs, error } = await loadRuns();
  return (
    <main>
      <p>
        <Link href="/">← Home</Link>
      </p>
      <h1>Your runs</h1>
      {error && <p role="alert">Could not load runs: {error}</p>}
      {!error && runs.length === 0 && <p>No runs yet. Uploading arrives in the next step.</p>}
      <ul>
        {runs.map((run) => (
          <li key={run.id}>
            {run.filename}: {run.status} ({new Date(run.created_at).toLocaleString()})
          </li>
        ))}
      </ul>
    </main>
  );
}
