import type { ReactNode } from "react";

export const metadata = {
  title: "Agentic EDA",
  description: "Credit-risk EDA with verified LLM reports",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: "system-ui, sans-serif", maxWidth: 760, margin: "48px auto", padding: "0 16px" }}>
        {children}
      </body>
    </html>
  );
}
