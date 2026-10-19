"use client";

import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import snapshot from "@/lib/demo.json";
import type { PricingModel, SurfaceData } from "@/lib/types";
import { expirations, parseSurface } from "@/lib/surface";
import { SmileChart, SurfaceChart } from "./charts";

const demo = parseSurface(snapshot);
const modelName = (model: PricingModel) => model === "black_scholes" ? "Black–Scholes" : "Bjerksund–Stensland";
const number = (value: number, decimals = 2) => value.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
const dateLabel = (date: string) => new Date(date).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric", timeZone: "UTC" });
const timeLabel = (date: string) => new Date(date).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "UTC" });

function Mark() {
  return <svg width="30" height="30" viewBox="0 0 30 30" fill="none" aria-hidden="true"><rect width="30" height="30" rx="8" fill="#126b5b" /><path d="M5 20C10 20 8 9 14 9S18 20 25 11M5 24C11 24 10 14 15 14S20 23 25 16M5 15C9 15 9 5 14 5S19 15 25 7" stroke="#d8f0da" strokeWidth="1.3" /></svg>;
}
function Arrow() { return <svg viewBox="0 0 20 20" width="17" height="17" fill="none" aria-hidden="true"><path d="M4 10h12m-5-5 5 5-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>; }

export default function Dashboard() {
  const [dataset, setDataset] = useState<SurfaceData>(demo);
  const [isSnapshot, setIsSnapshot] = useState(true);
  const [symbol, setSymbol] = useState("AAPL");
  const [model, setModel] = useState<PricingModel>(demo.model);
  const [kind, setKind] = useState("otm");
  const [count, setCount] = useState(6);
  const [minStrike, setMinStrike] = useState(70);
  const [maxStrike, setMaxStrike] = useState(130);
  const [allowLast, setAllowLast] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [expiration, setExpiration] = useState(expirations(demo.contracts)[1] ?? demo.contracts[0].expiration);
  const [showPoints, setShowPoints] = useState(false);
  const [includeEstimates, setIncludeEstimates] = useState(true);
  const [axisMode, setAxisMode] = useState<"strike" | "moneyness">("strike");
  const controller = useRef<AbortController | null>(null);
  const requestId = useRef(0);
  useEffect(() => () => controller.current?.abort(), []);

  const visibleDataset = useMemo(() => ({ ...dataset,
    contracts: includeEstimates ? dataset.contracts : dataset.contracts.filter((c) => c.price_source === "midpoint"),
  }), [dataset, includeEstimates]);
  const dates = useMemo(() => expirations(visibleDataset.contracts), [visibleDataset]);
  const activeExpiry = dates.includes(expiration) ? expiration : dates[0] ?? "";
  const lastTrades = dataset.contracts.filter((c) => c.price_source === "last_trade").length;
  const selectedCount = visibleDataset.contracts.filter((c) => c.expiration === activeExpiry).length;
  const oldestTrade = dataset.contracts.filter((c) => c.price_source === "last_trade" && c.lastTradeDate).map((c) => c.lastTradeDate!).sort()[0];

  async function fetchData(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (minStrike > maxStrike) { setError("Minimum strike must not exceed maximum strike."); return; }
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    const id = ++requestId.current;
    const timeout = window.setTimeout(() => abort.abort(), 55000);
    setLoading(true);
    setError("");
    try {
      const query = new URLSearchParams({ symbol: symbol.trim().toUpperCase(), model, option_type: kind,
        max_expirations: String(count), min_strike_pct: String(minStrike / 100),
        max_strike_pct: String(maxStrike / 100), allow_last_price: String(allowLast) });
      const response = await fetch(`/api/surface?${query}`, { signal: abort.signal, cache: "no-store" });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        if (response.status === 404) throw new Error(payload?.detail?.message || "No usable options matched these filters. Try another ticker, a wider strike range, or allow last-trade estimates.");
        if (response.status === 422) throw new Error("Check the ticker and filter values, then try again.");
        throw new Error(payload?.detail?.message || "Market data is unavailable right now. Your previous view has been kept.");
      }
      const result = parseSurface(payload);
      if (id !== requestId.current) return;
      setDataset(result);
      setExpiration(expirations(result.contracts)[0]);
      setIsSnapshot(false);
    } catch (e) {
      if (id !== requestId.current) return;
      setError(abort.signal.aborted ? "The request timed out. Your previous view has been kept. Try fewer expirations." : e instanceof Error ? e.message : "Unable to fetch market data.");
    } finally {
      window.clearTimeout(timeout);
      if (id === requestId.current) setLoading(false);
    }
  }

  function restoreDemo() {
    ++requestId.current;
    controller.current?.abort();
    setDataset(demo); setIsSnapshot(true); setLoading(false); setError(""); setIncludeEstimates(true);
    setSymbol("AAPL"); setModel(demo.model); setKind("otm"); setCount(6);
    setMinStrike(70); setMaxStrike(130); setAllowLast(true);
    setExpiration(expirations(demo.contracts)[1] ?? demo.contracts[0].expiration);
  }

  function exportCsv() {
    const keys = Object.keys(dataset.contracts[0]) as (keyof SurfaceData["contracts"][number])[];
    const rows = [keys.join(","), ...dataset.contracts.map((contract) => keys.map((key) => JSON.stringify(contract[key] ?? "")).join(","))];
    const url = URL.createObjectURL(new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a"); link.href = url;
    link.download = `${dataset.metrics.symbol}-iv-${isSnapshot ? "snapshot" : "fetched"}-${dataset.generated_at.slice(0, 10)}.csv`;
    link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <>
    <a className="skip-link" href="#explorer">Skip to explorer</a>
    <header className="site-header"><div className="header-inner">
      <a href="#" className="brand"><Mark /><span>vol<span className="brand-light">surface</span><span className="brand-dot">.</span></span></a>
      <nav aria-label="Main navigation"><a href="#explorer" className="nav-active">Explorer</a><a href="#methodology">Methodology</a><a href="/docs" target="_blank" rel="noreferrer">API <span aria-hidden="true">↗</span></a></nav>
      <span className="header-caption"><span className="status-dot" />Options analytics</span>
    </div></header>
    <main className="workspace" id="explorer">
      <section className="page-intro"><div><p className="eyebrow">THE SHAPE OF MARKET EXPECTATIONS</p><h1>Implied volatility explorer<span>.</span></h1><p className="intro-copy">Explore how option volatility changes across strikes and time.</p></div>
        <button type="button" className="button button-outline export-button" onClick={exportCsv}><svg width="16" height="16" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M10 3v10m-4-4 4 4 4-4M4 14v3h12v-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" /></svg>Export CSV</button>
      </section>

      <div className="dataset-bar"><div className="dataset-description"><span className={`badge ${isSnapshot ? "badge-snapshot" : "badge-fetched"}`}><span className="status-dot" />{isSnapshot ? "SAVED SNAPSHOT" : "FETCHED DATA"}</span><strong>{dataset.metrics.symbol}</strong><span className="bar-separator" /><span>{modelName(dataset.model)}</span><span className="bar-separator" /><span>{expirations(dataset.contracts).length} expirations</span></div><span className="dataset-date">{isSnapshot ? "Captured" : "Fetched"} {dateLabel(dataset.generated_at)} · {timeLabel(dataset.generated_at)} UTC</span></div>

      <section className="metrics" aria-label="Market inputs"><Metric label="UNDERLYING PRICE" value={`$${number(dataset.metrics.spot_price)}`} note={dataset.metrics.symbol + " · Yahoo Finance"} /><Metric label="RISK-FREE RATE" value={`${number(dataset.metrics.risk_free_rate * 100)}%`} note="10-year Treasury proxy" /><Metric label="DIVIDEND YIELD" value={`${number(dataset.metrics.dividend_yield * 100)}%`} note={dataset.metrics.dividend_source === "assumed_zero" ? "Assumed zero · data unavailable" : `${dataset.metrics.dividend_source === "forward" ? "Forward" : "Trailing annual"} dividend yield`} /><Metric label="OBSERVED CONTRACTS" value={String(dataset.contracts.length)} note={`${dataset.contracts.length - lastTrades} midpoints · ${lastTrades} last-trade estimates`} /></section>

      <section className="quality-summary" aria-label="Data quality summary">
        <p><strong>{dataset.contracts.length - lastTrades} bid/ask midpoints</strong><span> · </span><strong className="estimate-label">{lastTrades} last-trade estimates</strong><span> · {visibleDataset.contracts.length} shown</span></p>
        <label className="checkbox-label"><input type="checkbox" checked={includeEstimates} onChange={(e) => setIncludeEstimates(e.target.checked)} />Include last-trade estimates in charts</label>
        {dataset.quality ? <p className="quality-counts">Received {dataset.quality.received_contracts} · Outside selection {dataset.quality.excluded_by_filters} · Rejected quotes {dataset.quality.rejected_quotes} · Eligible {dataset.quality.eligible_contracts} · No identifiable IV {dataset.quality.invalid_or_unidentifiable_prices} · Outside model domain {dataset.quality.unsupported_model} · Solver unresolved {dataset.quality.unresolved_iv} · Solved {dataset.quality.solved_contracts}</p>
          : <p className="quality-counts">Solver counts are unavailable for this saved dataset.</p>}
        <p className="quality-counts">Bid/ask IV bars show sensitivity to the quoted spread, not a confidence interval. Last trades have no bid/ask interval. Chart filtering does not change the CSV export.</p>
      </section>

      <div className="explorer-grid">
        <aside className="panel controls-panel"><div className="panel-heading"><h2>Market selection</h2><span className="section-number">01</span></div>
          <form onSubmit={fetchData}>
            <fieldset disabled={loading} className="form-fields">
              <label htmlFor="symbol">Ticker symbol</label><div className="ticker-input"><span aria-hidden="true">⌕</span><input id="symbol" name="symbol" value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} placeholder="e.g. AAPL" required maxLength={20} pattern="[A-Za-z0-9^][A-Za-z0-9.^=\-]*" autoComplete="off" spellCheck={false} /></div>
              <label htmlFor="model">Pricing model</label><select id="model" value={model} onChange={(e) => setModel(e.target.value as PricingModel)}><option value="black_scholes">Black–Scholes</option><option value="bjerksund_stensland">Bjerksund–Stensland</option></select><p className="field-help">{model === "black_scholes" ? "European exercise assumption" : "American early-exercise approximation"}</p>
              <label htmlFor="kind">Option selection</label><select id="kind" value={kind} onChange={(e) => setKind(e.target.value)}><option value="otm">OTM puts + calls</option><option value="call">Calls only</option><option value="put">Puts only</option></select>
              <label htmlFor="expirations">Expirations</label><select id="expirations" value={count} onChange={(e) => setCount(Number(e.target.value))}>{[2, 4, 6, 8].map((n) => <option key={n} value={n}>{n} nearest available</option>)}</select>
              <div className="form-divider" /><div className="range-label"><span>Strike range</span><span>% of spot</span></div><div className="range-inputs"><label className="sr-only" htmlFor="min-strike">Minimum strike, percent of spot</label><div><input id="min-strike" type="number" min="1" max="500" value={minStrike} onChange={(e) => setMinStrike(Number(e.target.value))} required /><span>%</span></div><span>—</span><label className="sr-only" htmlFor="max-strike">Maximum strike, percent of spot</label><div><input id="max-strike" type="number" min="1" max="500" value={maxStrike} onChange={(e) => setMaxStrike(Number(e.target.value))} required /><span>%</span></div></div>
              <label className="checkbox-label"><input type="checkbox" checked={allowLast} onChange={(e) => setAllowLast(e.target.checked)} /><span>Allow last-trade estimates</span></label><p className="field-help">Fallback when bid and ask are absent. Trades must be at most 4 days old.</p>
            </fieldset>
            <button type="submit" disabled={loading} className="button button-primary fetch-button">{loading ? <><span className="spinner" />Fetching data…</> : <>Fetch market data<Arrow /></>}</button>
            <p className="request-note">{loading ? "Fetching chains and solving IV. This may take up to 55 seconds." : "Changes apply when you fetch data."}</p>
          </form>
          <button className="restore-button" type="button" onClick={restoreDemo}>↺ Restore AAPL snapshot</button>
          <div className="sidebar-note"><span className="small-dot" /><p>Start with the saved example, or explore another US equity.</p></div>
        </aside>

        <section className="panel surface-panel" aria-busy={loading}>
          <div className="chart-heading"><div><p className="eyebrow">VOLATILITY ACROSS STRIKE & MATURITY</p><h2>Implied volatility surface</h2></div><span className="subtle-tag">3D VIEW</span></div>
          <div className="chart-toolbar"><div className="segmented" role="group" aria-label="Surface horizontal axis"><button type="button" aria-pressed={axisMode === "strike"} className={axisMode === "strike" ? "active" : ""} onClick={() => setAxisMode("strike")}>Strike price</button><button type="button" aria-pressed={axisMode === "moneyness"} className={axisMode === "moneyness" ? "active" : ""} onClick={() => setAxisMode("moneyness")}>Log-moneyness</button></div><label className="checkbox-label points-toggle"><input type="checkbox" checked={showPoints} onChange={(e) => setShowPoints(e.target.checked)} />Show quotes</label></div>
          <div className="surface-chart">{visibleDataset.contracts.length ? <SurfaceChart dataset={visibleDataset} points={showPoints} axisMode={axisMode} /> : <p className="empty-chart" role="status">No bid/ask midpoint observations are available. Include last-trade estimates to view this dataset.</p>}</div>
          {loading && <div className="chart-updating" role="status"><span className="spinner" />Fetching new data · previous view retained</div>}
          <div className="chart-footer"><span><span className="legend-line" />Spot strike</span><span>Drag to rotate <span className="footer-dot">·</span> Scroll to zoom</span></div>
        </section>
      </div>

      {error && <div className="error-banner" role="alert"><strong>Couldn’t update the surface.</strong><span>{error}</span><button type="button" aria-label="Dismiss error" onClick={() => setError("")}>×</button></div>}
      <div className="sr-only" aria-live="polite">{loading ? "Fetching market data" : `Showing ${visibleDataset.contracts.length} of ${dataset.contracts.length} contracts for ${dataset.metrics.symbol}`}</div>

      {(dataset.warnings.length > 0 || isSnapshot) && <details className="quality-note"><summary><span className="info-icon">i</span><span>{lastTrades ? "Last-trade estimates are included" : "About this dataset"}</span><span className="quality-caption">Data quality <span aria-hidden="true">⌄</span></span></summary><div><p>{isSnapshot && `This is a saved Yahoo Finance snapshot captured on ${dateLabel(dataset.generated_at)}. `}Fetched prices may be delayed. Response time is not the timestamp of each quote.</p>{dataset.warnings.map((warning, i) => <p key={i}>{warning}</p>)}{oldestTrade && <p>Oldest fallback trade: {dateLabel(oldestTrade)} · {timeLabel(oldestTrade)} UTC.</p>}</div></details>}

      <section className="panel smile-panel"><div className="chart-heading"><div><p className="eyebrow">A CLOSER LOOK AT ONE EXPIRATION</p><h2>Volatility smile</h2></div><div className="expiry-control"><label htmlFor="smile-expiry">Expiration</label><select id="smile-expiry" disabled={!dates.length} value={activeExpiry} onChange={(e) => setExpiration(e.target.value)}>{dates.map((date) => <option key={date} value={date}>{dateLabel(date)}</option>)}</select></div></div><div className="smile-chart">{visibleDataset.contracts.length ? <SmileChart dataset={visibleDataset} expiration={activeExpiry} /> : <p className="empty-chart">No observations match the chart filter.</p>}</div><div className="chart-footer"><span><span className="legend-point" />Midpoints <span className="legend-point legend-estimate" />Last trades · {selectedCount} contracts</span><span>Hover over a point for price & Greeks</span></div></section>

      <section className="methodology" id="methodology"><div><p className="eyebrow">UNDER THE SURFACE</p><h2>A market view, with its assumptions in view.</h2></div><div className="method-cards"><article><span>01 / PRICES</span><h3>Quotes before estimates</h3><p>Valid bid/ask midpoints take priority. When enabled, recent last trades fill missing quotes and are explicitly flagged.</p></article><article><span>02 / MODELS</span><h3>One engine, two assumptions</h3><p>Black–Scholes assumes European exercise. Bjerksund–Stensland approximates American exercise. Cases outside its supported domain are reported as unavailable. Both use a continuous dividend yield.</p></article><article><span>03 / INTERPOLATION</span><h3>Connect, without extrapolating</h3><p>The display linearly joins IV across strikes and maturities. Gaps outside each expiry’s strike range stay empty. This is not an arbitrage-free calibrated surface.</p></article></div></section>
      <footer className="page-footer"><a href="#" className="footer-brand">volsurface.</a><span>Educational options analytics. Not investment advice.</span><span>Python engine <span className="footer-dot">/</span> Plotly visualization</span></footer>
    </main>
  </>;
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return <article className="metric"><p>{label}</p><strong>{value}</strong><span>{note}</span></article>;
}
