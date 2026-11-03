/** Server-side calls to the API (pages render with data already loaded). Never imported by client code. */
import "server-only";

import { currentApiUser } from "@/auth";
import { mintApiToken } from "@/lib/api-token";

export const API_URL = process.env.API_URL ?? "http://localhost:8000";

export type ApiResult<T> = { ok: true; data: T } | { ok: false; status: number };

export async function apiGet<T>(path: string): Promise<ApiResult<T> | null> {
  const user = await currentApiUser();
  if (!user) return null;
  try {
    const response = await fetch(`${API_URL}${path}`, {
      headers: { authorization: `Bearer ${await mintApiToken(user, process.env.API_JWT_SECRET)}` },
      cache: "no-store",
    });
    if (!response.ok) return { ok: false, status: response.status };
    return { ok: true, data: (await response.json()) as T };
  } catch {
    return { ok: false, status: 503 };
  }
}
