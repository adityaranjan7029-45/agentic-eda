export function Logo({ className = "h-8 w-8" }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={`grid place-items-center rounded-xl bg-gradient-to-br from-brand-500 via-brand-600 to-indigo-700 text-white shadow-md shadow-brand-600/30 ${className}`}
    >
      <svg viewBox="0 0 24 24" className="h-[60%] w-[60%]" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
        <path d="M4 19V11M10 19V5M16 19v-6M22 19H2" />
        <path d="M14.5 7.5l2 2 4-4" strokeWidth="2" />
      </svg>
    </span>
  );
}
