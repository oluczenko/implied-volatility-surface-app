"use client";

import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import type { SurfaceData } from "@/lib/types";
import { interpolate, smilePoints, surfaceGrid } from "@/lib/surface";
import Plot from "./plot";

const colors: [number, string][] = [[0, "#164f59"], [0.25, "#278c8e"], [0.5, "#70bda9"], [0.75, "#c3dca6"], [1, "#f1db83"]];
const font = { family: 'Arial, sans-serif', color: "#65736f", size: 11 };
const axis = { gridcolor: "#e4eae6", zeroline: false, showbackground: true, backgroundcolor: "#fafcfb", linecolor: "#dce4df", tickfont: { size: 10, color: "#64736f" } };

export function SurfaceChart({ dataset, points, axisMode }: { dataset: SurfaceData; points: boolean; axisMode: "strike" | "moneyness" }) {
  const traces = useMemo(() => {
    const { contracts, metrics } = dataset;
    const grid = surfaceGrid(contracts);
    const convert = (strike: number) => axisMode === "strike" ? strike : Math.log(strike / metrics.spot_price);
    const raw: Data = { type: "scatter3d", mode: "markers", name: "Observed quotes",
      x: contracts.map((c) => convert(c.strike)), y: contracts.map((c) => c.T * 365), z: contracts.map((c) => c.iv_pct),
      marker: { size: 2.5, color: contracts.map((c) => c.price_source === "midpoint" ? "#148a78" : "#b77c24"), opacity: 0.8 },
      customdata: contracts.map((c) => c.price_source === "midpoint" ? "Bid/ask midpoint" : "Last-trade estimate"),
      hovertemplate: "%{customdata}<br>Strike axis: %{x:.3f}<br>Days: %{y:.1f}<br>IV: %{z:.2f}%<extra></extra>" };
    const result: Data[] = [];
    if (grid.x.length > 1 && grid.y.length > 1) result.push({ type: "surface", name: "Interpolated IV",
      x: grid.x.map(convert), y: grid.y, z: grid.z, colorscale: colors, opacity: 0.96,
      connectgaps: false, showscale: true,
      colorbar: { title: { text: "IV %", font }, thickness: 9, len: 0.65, outlinewidth: 0, tickfont: font, x: 1.0 },
      hovertemplate: "Interpolated<br>Strike axis: %{x:.3f}<br>Days: %{y:.1f}<br>IV: %{z:.2f}%<extra></extra>" });
    if (points || !result.length) result.push(raw);
    const atm = grid.dates.map((date, i) => ({ y: grid.y[i], z: interpolate(smilePoints(contracts, date), metrics.spot_price) })).filter((p) => p.z !== null);
    if (atm.length) result.push({ type: "scatter3d", mode: "lines", name: "Spot strike",
      x: atm.map(() => convert(metrics.spot_price)), y: atm.map((p) => p.y), z: atm.map((p) => p.z! + 0.05),
      line: { color: "#183f3b", width: 5, dash: "dash" }, hoverinfo: "skip" });
    return result;
  }, [dataset, points, axisMode]);
  const layout = useMemo<Partial<Layout>>(() => ({ paper_bgcolor: "transparent", plot_bgcolor: "transparent", font,
    margin: { l: 0, r: 35, t: 10, b: 0 }, showlegend: false,
    uirevision: `${dataset.generated_at}-${axisMode}`,
    scene: { xaxis: { ...axis, title: { text: axisMode === "strike" ? "Strike ($)" : "Log(K / Spot)", font } },
      yaxis: { ...axis, title: { text: "Days to expiry", font } }, zaxis: { ...axis, title: { text: "IV (%)", font } },
      camera: { eye: { x: 1.6, y: -1.8, z: 0.95 } }, aspectratio: { x: 1.5, y: 1.1, z: 0.7 } },
  }), [dataset.generated_at, axisMode]);
  return <Plot data={traces} layout={layout} label="Interactive 3D implied volatility surface; drag to rotate, scroll to zoom" />;
}

export function SmileChart({ dataset, expiration }: { dataset: SurfaceData; expiration: string }) {
  const traces = useMemo<Data[]>(() => {
    const contracts = dataset.contracts.filter((c) => c.expiration === expiration).sort((a, b) => a.strike - b.strike);
    const points = smilePoints(dataset.contracts, expiration);
    return [{ type: "scatter", mode: "lines", name: "Linear interpolation", x: points.map((p) => p.strike), y: points.map((p) => p.iv),
      line: { color: "#148a78", width: 2.5, shape: "linear" }, hoverinfo: "skip" },
      { type: "scatter", mode: "markers", name: "Observed quotes", x: contracts.map((c) => c.strike), y: contracts.map((c) => c.iv_pct),
        customdata: contracts.map((c) => [c.otype === 1 ? "Call" : "Put", c.midPrice, c.delta, c.gamma, c.vega, c.theta, c.price_source === "midpoint" ? "Bid/ask midpoint" : "Last-trade estimate",
          c.iv_bid == null ? "unavailable" : `${(c.iv_bid * 100).toFixed(2)}%`,
          c.iv_ask == null ? "unavailable" : `${(c.iv_ask * 100).toFixed(2)}%`]),
        error_y: { type: "data", symmetric: false,
          array: contracts.map((c) => c.iv_bid != null && c.iv_ask != null ? Math.max(0, c.iv_ask * 100 - c.iv_pct) : 0),
          arrayminus: contracts.map((c) => c.iv_bid != null && c.iv_ask != null ? Math.max(0, c.iv_pct - c.iv_bid * 100) : 0),
          color: "#81b5a6", thickness: 1, width: 3 },
        marker: { size: 6, color: contracts.map((c) => c.price_source === "midpoint" ? "#148a78" : "#b77c24"), line: { color: "#fff", width: 1 } },
        hovertemplate: "<b>%{customdata[0]} · $%{x:.2f}</b><br>IV: %{y:.2f}%<br>Price: $%{customdata[1]:.2f}<br>Delta: %{customdata[2]:.3f}<br>Gamma: %{customdata[3]:.4f}<br>Vega: %{customdata[4]:.3f}<br>Theta/day: %{customdata[5]:.3f}<br>Source: %{customdata[6]}<br>Bid IV: %{customdata[7]}<br>Ask IV: %{customdata[8]}<extra></extra>" }];
  }, [dataset, expiration]);
  const layout = useMemo<Partial<Layout>>(() => ({ paper_bgcolor: "transparent", plot_bgcolor: "transparent", font,
    margin: { l: 55, r: 20, t: 20, b: 50 }, showlegend: false, hovermode: "closest",
    xaxis: { title: { text: "Strike ($)", font }, gridcolor: "#edf1ed", zeroline: false, fixedrange: false },
    yaxis: { title: { text: "IV (%)", font }, gridcolor: "#edf1ed", zeroline: false },
    shapes: [{ type: "line", x0: dataset.metrics.spot_price, x1: dataset.metrics.spot_price, y0: 0, y1: 1, yref: "paper", line: { color: "#aab8af", width: 1, dash: "dot" } }],
    annotations: [{ x: dataset.metrics.spot_price, y: 1, yref: "paper", text: "SPOT", showarrow: false, yshift: 12, font: { ...font, size: 9 } }],
    uirevision: `${dataset.generated_at}-${expiration}`,
  }), [dataset.generated_at, dataset.metrics.spot_price, expiration]);
  return <Plot data={traces} layout={layout} label={`Volatility smile for ${expiration}; hover over quotes to inspect Greeks`} />;
}
