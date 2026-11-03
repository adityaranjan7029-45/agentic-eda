import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import type { ReactNode } from "react";

import { Footer } from "@/components/Footer";
import { Header } from "@/components/Header";

import "./globals.css";

export const metadata = {
  title: { default: "Agentic EDA", template: "%s · Agentic EDA" },
  description: "Credit-risk exploratory analysis where statistics find the insights and every number in the report is verified.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="flex min-h-screen flex-col">
        <Header />
        <div className="flex-1">{children}</div>
        <Footer />
      </body>
    </html>
  );
}
