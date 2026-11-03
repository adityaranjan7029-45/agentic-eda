"use client";

import { useEffect, useRef, useState } from "react";

/** Shared look for every chart: the app's font, brand colours, quiet axes. */
function theme(dark: boolean) {
  const text = dark ? "#cbd5e1" : "#475569";
  const grid = dark ? "#1e293b" : "#eef2f7";
  return {
    background: "transparent",
    font: "Inter Variable, ui-sans-serif, system-ui, sans-serif",
    view: { stroke: null },
    mark: { color: "#3d6bff" },
    line: { color: "#3d6bff" },
    point: { color: "#2650f0" },
    rule: { color: dark ? "#94a3b8" : "#64748b" },
    text: { color: text },
    range: { category: ["#3d6bff", "#e11d48", "#10b981", "#f59e0b", "#8b5cf6", "#06b6d4"] },
    axis: {
      labelColor: text,
      titleColor: text,
      titleFontWeight: 500,
      labelFontSize: 11,
      titleFontSize: 11,
      gridColor: grid,
      domain: false,
      tickColor: grid,
      labelLimit: 140,
    },
    legend: { labelColor: text, titleColor: text, labelFontSize: 11, symbolType: "circle" },
  };
}

/** Renders a Vega-Lite spec from the engine. vega-embed is loaded only in the browser. */
export function VegaChart({ spec, title, tall = false }: { spec: Record<string, unknown>; title: string; tall?: boolean }) {
  const container = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let view: { finalize(): void } | undefined;
    let cancelled = false;
    (async () => {
      try {
        const { default: embed } = await import("vega-embed");
        if (cancelled || !container.current) return;
        const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
        const sized = {
          ...spec,
          title: undefined,
          width: "container",
          height: tall ? 340 : 230,
          autosize: { type: "fit-x", contains: "padding" },
          config: theme(dark),
        };
        const result = await embed(container.current, sized as never, { actions: false, renderer: "svg" });
        view = result.view;
      } catch {
        setFailed(true);
      }
    })();
    return () => {
      cancelled = true;
      view?.finalize();
    };
  }, [spec, tall]);

  return (
    <figure className="min-w-0 rounded-xl border border-slate-200/80 bg-white p-4 dark:border-slate-800 dark:bg-slate-900/60">
      <figcaption className="mb-3 text-sm font-semibold text-slate-700 dark:text-slate-200">{title}</figcaption>
      {failed ? <p className="text-sm text-slate-500">This chart couldn&apos;t be drawn.</p> : <div ref={container} className="w-full" />}
    </figure>
  );
}
