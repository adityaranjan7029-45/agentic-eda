import { LogOut } from "lucide-react";
import Link from "next/link";

import { auth, signOut } from "@/auth";
import { Logo } from "@/components/Logo";

export async function Header() {
  const session = await auth();
  const user = session?.user;
  return (
    <header className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/75 backdrop-blur-xl dark:border-slate-800 dark:bg-slate-950/75">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
        <Link href={user ? "/runs" : "/"} className="flex items-center gap-2.5">
          <Logo />
          <span className="text-[15px] font-semibold tracking-tight">
            Agentic <span className="text-brand-600 dark:text-brand-400">EDA</span>
          </span>
        </Link>
        {user ? (
          <div className="flex items-center gap-2 text-sm">
            <Link href="/runs" className="btn-ghost hidden sm:inline-flex">
              Analyses
            </Link>
            <span className="mx-1 hidden h-6 w-px bg-slate-200 sm:block dark:bg-slate-800" />
            {user.image ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={user.image} alt="" className="h-8 w-8 rounded-full ring-2 ring-white dark:ring-slate-900" />
            ) : (
              <span className="grid h-8 w-8 place-items-center rounded-full bg-brand-100 text-xs font-bold text-brand-700">
                {(user.name ?? user.email ?? "?").slice(0, 1).toUpperCase()}
              </span>
            )}
            <span className="hidden max-w-40 truncate font-medium sm:inline">{user.name ?? user.email}</span>
            <form
              action={async () => {
                "use server";
                await signOut({ redirectTo: "/" });
              }}
            >
              <button type="submit" className="btn-ghost" aria-label="Sign out" title="Sign out">
                <LogOut className="h-4 w-4" />
              </button>
            </form>
          </div>
        ) : (
          <a href="https://github.com/adityaranjan7029-45/agentic-eda" className="btn-ghost text-sm" target="_blank" rel="noreferrer">
            Source on GitHub
          </a>
        )}
      </div>
    </header>
  );
}
