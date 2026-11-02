/**
 * Backend-for-frontend proxy: /api/backend/<path> -> API_URL/<path>.
 *
 * The browser talks only to this Next.js server, with its session cookie. Here we
 * check the session, sign a 5-minute token for the API, and forward the request.
 * Responses are streamed straight through, so live progress (SSE) works.
 */
import { NextRequest } from "next/server";

import { currentApiUser } from "@/auth";
import { mintApiToken } from "@/lib/api-token";
import {
  FORWARD_REQUEST_HEADERS,
  FORWARD_RESPONSE_HEADERS,
  backendUrl,
  forwardHeaders,
  isSameOrigin,
} from "@/lib/proxy";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Context = { params: Promise<{ path: string[] }> };

async function handle(request: NextRequest, context: Context): Promise<Response> {
  if (!isSameOrigin(request.method, request.headers.get("origin"), request.headers.get("host"))) {
    return Response.json({ detail: "cross-site request refused" }, { status: 403 });
  }
  const user = await currentApiUser();
  if (!user) return Response.json({ detail: "sign in required" }, { status: 401 });

  let url: string;
  try {
    url = backendUrl(process.env.API_URL ?? "http://localhost:8000", (await context.params).path, request.nextUrl.search);
  } catch {
    return Response.json({ detail: "not found" }, { status: 404 });
  }

  const headers = forwardHeaders(request.headers, FORWARD_REQUEST_HEADERS);
  headers.set("authorization", `Bearer ${await mintApiToken(user, process.env.API_JWT_SECRET)}`);
  const init: RequestInit & { duplex?: "half" } = {
    method: request.method,
    headers,
    cache: "no-store",
    signal: request.signal, // browser closes the stream -> stop reading from the API
  };
  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = request.body;
    init.duplex = "half"; // stream the upload instead of buffering it
  }

  let upstream: Response;
  try {
    upstream = await fetch(url, init);
  } catch {
    return Response.json({ detail: "the analysis service is unavailable" }, { status: 502 });
  }
  return new Response(upstream.body, {
    status: upstream.status,
    headers: forwardHeaders(upstream.headers, FORWARD_RESPONSE_HEADERS),
  });
}

export { handle as GET, handle as POST, handle as DELETE };
