import Link from "next/link";
import { redirect } from "next/navigation";

import { RunView } from "@/components/RunView";
import { errorMessage } from "@/lib/errors";
import { apiGet } from "@/lib/server-api";
import type { RunDetail } from "@/lib/types";

export const metadata = { title: "Report" };

export default async function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!/^[A-Za-z0-9-]{1,64}$/.test(id)) redirect("/runs");
  const result = await apiGet<RunDetail>(`/runs/${id}`);
  if (result === null) redirect("/");
  if (!result.ok) {
    return (
      <div className="card mx-auto mt-10 max-w-xl">
        <p role="alert">{errorMessage(result.status)}</p>
        <Link href="/runs" className="btn-secondary mt-4">
          ← Your runs
        </Link>
      </div>
    );
  }
  return <RunView key={result.data.id} initial={result.data} />;
}
