import type { Contract, SurfaceData } from "./types";

export function expirations(contracts: Contract[]): string[] {
  return [...new Set(contracts.map((c) => c.expiration))].sort();
}

// Average duplicate call/put strikes for display only; retain raw points separately.
export function smilePoints(contracts: Contract[], expiration: string) {
  const groups = new Map<number, { total: number; count: number }>();
  for (const c of contracts.filter((c) => c.expiration === expiration)) {
    const group = groups.get(c.strike) ?? { total: 0, count: 0 };
    group.total += c.iv_pct;
    group.count++;
    groups.set(c.strike, group);
  }
  return [...groups].sort(([a], [b]) => a - b).map(([strike, g]) => ({ strike, iv: g.total / g.count }));
}

export function interpolate(points: { strike: number; iv: number }[], strike: number): number | null {
  if (!points.length || strike < points[0].strike || strike > points[points.length - 1].strike) return null;
  for (let i = 0; i < points.length; i++) {
    const right = points[i];
    if (right.strike === strike) return right.iv;
    if (right.strike > strike && i > 0) {
      const left = points[i - 1];
      return left.iv + ((strike - left.strike) / (right.strike - left.strike)) * (right.iv - left.iv);
    }
  }
  return null;
}

export function surfaceGrid(contracts: Contract[], resolution = 45) {
  const dates = expirations(contracts);
  const strikes = contracts.map((c) => c.strike);
  const low = Math.min(...strikes);
  const high = Math.max(...strikes);
  const x = !strikes.length ? [] : low === high ? [low] : Array.from({ length: resolution }, (_, i) => low + (i / (resolution - 1)) * (high - low));
  const y = dates.map((date) => contracts.find((c) => c.expiration === date)!.T * 365);
  const z = dates.map((date) => {
    const points = smilePoints(contracts, date);
    return x.map((strike) => interpolate(points, strike));
  });
  return { x, y, z, dates };
}

export function parseSurface(value: unknown): SurfaceData {
  const d = value as SurfaceData | null;
  if (!d || !d.metrics || typeof d.metrics.symbol !== "string" ||
      !Number.isFinite(d.metrics.spot_price) || d.metrics.spot_price <= 0 ||
      !Number.isFinite(d.metrics.risk_free_rate) || !Number.isFinite(d.metrics.dividend_yield) ||
      !["forward", "trailing", "assumed_zero"].includes(d.metrics.dividend_source) ||
      !["black_scholes", "bjerksund_stensland"].includes(d.model) ||
      typeof d.generated_at !== "string" || !Number.isFinite(Date.parse(d.generated_at)) ||
      !Array.isArray(d.contracts) || !d.contracts.length ||
      !Array.isArray(d.warnings) || !d.warnings.every((w) => typeof w === "string") ||
      !d.contracts.every((c) => c && Number.isFinite(c.strike) && c.strike > 0 &&
        Number.isFinite(c.iv_pct) && c.iv_pct > 0 && Number.isFinite(c.T) && c.T > 0 &&
        typeof c.expiration === "string" && Number.isFinite(Date.parse(c.expiration)) &&
        ["midpoint", "last_trade"].includes(c.price_source))) {
    throw new Error("The server returned an invalid surface. Your previous view has been kept.");
  }
  return d;
}
