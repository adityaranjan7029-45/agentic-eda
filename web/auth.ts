/**
 * Auth.js (NextAuth v5): GitHub login, session kept in an encrypted httpOnly cookie.
 *
 * No database adapter: the API creates the user row in Postgres on the first
 * request, keyed by provider + GitHub account id (emails can change or be private).
 */
import NextAuth from "next-auth";
import GitHub from "next-auth/providers/github";

declare module "next-auth" {
  interface Session {
    user: {
      name?: string | null;
      email?: string | null;
      image?: string | null;
      provider?: string;
      providerAccountId?: string;
    };
  }
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [GitHub],
  session: { strategy: "jwt", maxAge: 7 * 24 * 60 * 60 },
  trustHost: true, // self-hosted behind Caddy (step 16): trust the forwarded host
  callbacks: {
    jwt({ token, account }) {
      if (account) {
        token.provider = account.provider;
        token.providerAccountId = account.providerAccountId;
      }
      return token;
    },
    session({ session, token }) {
      session.user.provider = typeof token.provider === "string" ? token.provider : undefined;
      session.user.providerAccountId =
        typeof token.providerAccountId === "string" ? token.providerAccountId : undefined;
      return session;
    },
  },
});

/** The signed-in user as the API sees them, or null. */
export async function currentApiUser() {
  const session = await auth();
  const user = session?.user;
  if (!user?.provider || !user.providerAccountId) return null;
  return { provider: user.provider, accountId: user.providerAccountId, email: user.email, name: user.name };
}
