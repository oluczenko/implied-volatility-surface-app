export type PricingModel = "black_scholes" | "bjerksund_stensland";
export interface Contract {
  expiration: string;
  dte_days: number;
  T: number;
  strike: number;
  bid: number | null;
  ask: number | null;
  midPrice: number;
  otype: -1 | 1;
  price_source: "midpoint" | "last_trade";
  lastTradeDate: string | null;
  iv: number;
  iv_pct: number;
  delta: number | null;
  gamma: number | null;
  vega: number | null;
  theta: number | null;
  rho: number | null;
}
export interface SurfaceData {
  generated_at: string;
  model: PricingModel;
  metrics: {
    symbol: string;
    spot_price: number;
    risk_free_rate: number;
    dividend_yield: number;
    dividend_source: "forward" | "trailing" | "assumed_zero";
  };
  contracts: Contract[];
  warnings: string[];
}
