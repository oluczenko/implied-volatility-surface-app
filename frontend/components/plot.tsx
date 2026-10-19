"use client";

import { useEffect, useRef, useState } from "react";
import type { Config, Data, Layout } from "plotly.js";

export default function Plot({ data, layout, label }: { data: Data[]; layout: Partial<Layout>; label: string }) {
  const container = useRef<HTMLDivElement>(null);
  const [error, setError] = useState(false);
  const [ready, setReady] = useState(false);
  const revision = useRef(0);

  useEffect(() => {
    const element = container.current;
    if (!element) return;
    let cancelled = false;
    let observer: ResizeObserver | undefined;
    let plotly: typeof import("plotly.js") | undefined;
    const current = ++revision.current;
    const config: Partial<Config> = { responsive: true, displaylogo: false,
      modeBarButtonsToRemove: ["sendDataToCloud", "lasso2d", "select2d"],
      toImageButtonOptions: { format: "png", filename: "iv-surface", scale: 2 } };
    import("plotly.js-dist-min").then(async (module) => {
      if (cancelled) return;
      plotly = module.default;
      await plotly.react(element, data, { ...layout, autosize: true }, config);
      if (cancelled || current !== revision.current) return;
      setReady(true);
      setError(false);
      observer = new ResizeObserver(() => {
        if (element.isConnected && plotly) void plotly.Plots.resize(element);
      });
      observer.observe(element);
    }).catch(() => { if (!cancelled) setError(true); });
    return () => { cancelled = true; observer?.disconnect(); };
  }, [data, layout]);

  // Purge on unmount only; updates preserve the user's camera via uirevision.
  useEffect(() => {
    const element = container.current;
    return () => {
      if (element) void import("plotly.js-dist-min").then((p) => p.default.purge(element));
    };
  }, []);

  return <div className="plot-shell" role="img" aria-label={label}>
    {!ready && !error && <div className="plot-loading"><span className="spinner" />Preparing interactive chart…</div>}
    {error && <div className="plot-loading" role="alert">Chart unavailable. Try a browser with WebGL enabled.</div>}
    <div ref={container} className="plot-canvas" />
  </div>;
}
