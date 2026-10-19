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
  iv_bid: number | null;
  iv_ask: number | null;
  delta: number | null;
  gamma: number | null;
  vega: number | null;
  theta: number | null;
  rho: number | null;
}
export interface QualitySummary {
  received_contracts: number;
  excluded_by_filters: number;
  rejected_quotes: number;
  eligible_contracts: number;
  invalid_or_unidentifiable_prices: number;
  unsupported_model: number;
  unresolved_iv: number;
  solved_contracts: number;
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
  quality: QualitySummary | null;
}
