/**
 * Which sign-in options this server offers.
 *
 * GitHub is always on. Google is on only when its OAuth client is configured
 * (AUTH_GOOGLE_ID and AUTH_GOOGLE_SECRET), so local dev and CI keep working without it.
 * When on, Google comes first: most business users have a Google account, fewer have GitHub.
 * The API doesn't care which one was used: users are keyed by (provider, account id).
 */
export type ProviderId = "github" | "google";

export const PROVIDER_LABEL: Record<ProviderId, string> = { github: "GitHub", google: "Google" };

export function enabledProviders(env: Record<string, string | undefined> = process.env): ProviderId[] {
  const configured = (id?: string, secret?: string) => !!id?.trim() && !!secret?.trim();
  return configured(env.AUTH_GOOGLE_ID, env.AUTH_GOOGLE_SECRET) ? ["google", "github"] : ["github"];
}
