import Link from "next/link";

import { auth, signIn, signOut } from "@/auth";

export default async function Home() {
  const session = await auth();
  return (
    <main>
      <h1>Agentic EDA</h1>
      <p>Credit-risk exploratory analysis where statistics find the insights and every number in the report is verified.</p>
      {session?.user ? (
        <>
          <p>
            Signed in as <strong>{session.user.name ?? session.user.email ?? "you"}</strong>.{" "}
            <Link href="/runs">Your runs →</Link>
          </p>
          <form
            action={async () => {
              "use server";
              await signOut({ redirectTo: "/" });
            }}
          >
            <button type="submit">Sign out</button>
          </form>
        </>
      ) : (
        <form
          action={async () => {
            "use server";
            await signIn("github", { redirectTo: "/runs" });
          }}
        >
          <button type="submit">Sign in with GitHub</button>
        </form>
      )}
    </main>
  );
}
