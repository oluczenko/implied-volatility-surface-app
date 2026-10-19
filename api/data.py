import datetime as dt
import warnings
from concurrent.futures import ThreadPoolExecutor
from threading import local
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from .engine import calculate_all_greeks, iv_price_bounds, solve_iv_robust


class NoOptionsError(ValueError):
    """No contracts match the requested filters or produce a usable IV."""

    def __init__(self, message, quality=None):
        super().__init__(message)
        self.quality = quality


class MarketDataError(ValueError):
    """The market data provider failed or returned missing/invalid inputs."""


def _report_warning(message: str, diagnostics: list[str] | None) -> None:
    if diagnostics is None:
        warnings.warn(message, UserWarning, stacklevel=3)
    else:
        diagnostics.append(message)


def _fetch_option_chains(symbol: str, expirations: list[str]):
    """Return (chain, error) pairs in date order, with at most four concurrent requests.

    Each worker owns a Ticker and initializes its expiration mapping through
    the public options API. Only downloading runs in workers; filtering and
    market-input selection remain on the calling thread.
    """
    if not expirations:
        return []
    worker = local()

    def fetch(expiration):
        try:
            if not hasattr(worker, "ticker"):
                ticker = yf.Ticker(symbol)
                # Populate this Ticker's expiration mapping once per worker.
                ticker.options
                worker.ticker = ticker
            return worker.ticker.option_chain(expiration), None
        except Exception as exc:
            return None, exc

    with ThreadPoolExecutor(
        max_workers=min(4, len(expirations)), thread_name_prefix="option-chain",
    ) as executor:
        return list(executor.map(fetch, expirations))


def fetch_raw_options(
    symbol: str,
    option_type: Literal["call", "put", "otm"] = "otm",
    min_dte_days: int = 2,
    max_expirations: int | None = None,
    min_strike_pct: float = 0.70,
    max_strike_pct: float = 1.30,
    allow_last_price: bool = True,
    max_trade_age_days: float = 4,
    diagnostics: list[str] | None = None,
    max_dte_days: int = 365,
):
    """Fetch quotes and market inputs (rates as decimals, T as ACT/365 years).

    Prefer bid/ask midpoints. Optional last-trade estimates require a timestamp
    no older than max_trade_age_days (calendar days; four covers a long weekend).
    These estimates are not synchronized EOD prices; inspect price_source.
    midPrice is retained for compatibility even when price_source is last_trade.
    Dividend input prefers dividendRate / spot, then trailing yield.
    Missing dividend data assumes q=0 with a warning, not confirmed zero payouts.
    Fetch every available expiration within the inclusive calendar-day horizon
    (up to 365 days), unless an optional legacy max_expirations cap is supplied.
    """
    if option_type not in ("call", "put", "otm"):
        raise ValueError(f"Invalid option_type: '{option_type}'")
    if (not 0 <= min_dte_days <= max_dte_days <= 365
            or max_dte_days < 1
            or (max_expirations is not None and max_expirations <= 0)):
        raise ValueError("Invalid expiration limits")
    if not (0 < min_strike_pct <= max_strike_pct < np.inf):
        raise ValueError("Invalid strike range")
    if not np.isfinite(max_trade_age_days) or max_trade_age_days <= 0:
        raise ValueError("Invalid maximum trade age")
    ticker = yf.Ticker(symbol)
    try:
        expirations = ticker.options
    except Exception as exc:
        raise MarketDataError(f"Failed to fetch expirations for '{symbol}'") from exc
    if not expirations:
        raise NoOptionsError(f"No option chains available for symbol: '{symbol}'")

    # Assume US equity expiry at 16:00 New York, not AM index settlement.
    market_tz = ZoneInfo("America/New_York")
    now = dt.datetime.now(market_tz)
    valid_expirations = []

    for exp_str in sorted(expirations):
        exp_date = dt.date.fromisoformat(exp_str)
        exp_dt = dt.datetime.combine(exp_date, dt.time(16), tzinfo=market_tz)
        dte_days = (exp_date - now.date()).days
        seconds_to_expiry = exp_dt.timestamp() - now.timestamp()
        T_years = seconds_to_expiry / (365.0 * 86400)
        if min_dte_days <= dte_days <= max_dte_days and T_years > 0:
            valid_expirations.append((exp_str, dte_days, T_years))

    valid_expirations = valid_expirations[:max_expirations]

    frames = []
    metrics = None
    chain_errors = []
    failed_expirations = []
    successful_chains = 0
    quality = {"received_contracts": 0, "excluded_by_filters": 0,
               "rejected_quotes": 0, "eligible_contracts": 0,
               "invalid_or_unidentifiable_prices": 0, "unsupported_model": 0, "unresolved_iv": 0,
               "solved_contracts": 0}
    chains = _fetch_option_chains(symbol, [exp for exp, _, _ in valid_expirations])
    for (exp_str, dte_days, T_years), (chain, error) in zip(valid_expirations, chains):
        if error is not None:
            chain_errors.append(error)
            failed_expirations.append(exp_str)
            continue
        successful_chains += 1
        if all(side is None or side.empty for side in (chain.calls, chain.puts)):
            continue

        if metrics is None:
            quote = chain.underlying or {}
            spot_price = quote.get("regularMarketPrice", np.nan)
            # Annual cash amount / spot avoids ambiguous percentage units.
            forward_dividend = quote.get("dividendRate")
            dividend_yield = quote.get("trailingAnnualDividendYield")
            dividend_source = "trailing"
            if forward_dividend is not None:
                if not np.isfinite(forward_dividend) or forward_dividend < 0:
                    raise MarketDataError(f"Invalid forward dividend data for '{symbol}'")
                dividend_source = "forward"
            if spot_price is None or not np.isfinite(spot_price) or spot_price <= 0:
                raise MarketDataError(f"Missing valid spot price in option chain for '{symbol}'")
            if forward_dividend is not None:
                dividend_yield = forward_dividend / spot_price
            elif dividend_yield is None:
                dividend_yield = 0.0
                dividend_source = "assumed_zero"
                _report_warning(f"Missing dividend data for '{symbol}'; assuming q=0.", diagnostics)
            if not np.isfinite(dividend_yield) or dividend_yield < 0:
                raise MarketDataError(f"Missing dividend data for '{symbol}'")

            try:
                treasury_yield = yf.Ticker("^TNX").fast_info.last_price
            except Exception as exc:
                raise MarketDataError("Failed to fetch ^TNX yield") from exc
            if treasury_yield is None or not np.isfinite(treasury_yield) or treasury_yield <= 0:
                raise MarketDataError("Missing valid ^TNX yield")
            metrics = {
                "symbol": symbol.upper(),
                "spot_price": spot_price,
                "risk_free_rate": treasury_yield / 100,
                "dividend_yield": dividend_yield,
                "dividend_source": dividend_source,
            }
            min_strike = spot_price * min_strike_pct
            max_strike = spot_price * max_strike_pct

        columns = ["strike", "bid", "ask", "lastPrice", "lastTradeDate"]
        sides = []
        if option_type in ("call", "otm") and chain.calls is not None:
            sides.append(chain.calls.reindex(columns=columns).assign(otype=1.0))
        if option_type in ("put", "otm") and chain.puts is not None:
            sides.append(chain.puts.reindex(columns=columns).assign(otype=-1.0))
        if not sides:
            continue
        selected = pd.concat(sides, ignore_index=True).sort_values("strike")

        for column in ("strike", "bid", "ask", "lastPrice"):
            selected[column] = pd.to_numeric(selected[column], errors="coerce")
        has_spread = (
            np.isfinite(selected[["bid", "ask"]]).all(axis=1)
            & (selected["bid"] > 0)
            & (selected["ask"] >= selected["bid"])
        )
        trade_time = pd.to_datetime(selected["lastTradeDate"], utc=True, errors="coerce")
        trade_age = (pd.Timestamp(now) - trade_time).dt.total_seconds()
        # Only absent quotes can use last trades; crossed/negative quotes stay invalid.
        missing_quotes = (
            (selected["bid"].isna() | selected["bid"].eq(0))
            & (selected["ask"].isna() | selected["ask"].eq(0))
        )
        use_last = (
            allow_last_price & missing_quotes
            & np.isfinite(selected["lastPrice"]) & (selected["lastPrice"] > 0)
            & trade_age.between(0, max_trade_age_days * 86400)
        )
        selected["lastTradeDate"] = trade_time
        selected["midPrice"] = np.where(
            has_spread, 0.5 * (selected["bid"] + selected["ask"]), selected["lastPrice"]
        )
        selected["price_source"] = np.where(has_spread, "midpoint", "last_trade")
        valid = (
            np.isfinite(selected["strike"])
            & selected["strike"].between(min_strike, max_strike)
        )
        if option_type == "otm":
            valid &= (
                ((selected["otype"] == -1) & (selected["strike"] < spot_price))
                | ((selected["otype"] == 1) & (selected["strike"] >= spot_price))
            )
        quality["received_contracts"] += len(selected)
        quality["excluded_by_filters"] += int((~valid).sum())
        quality["rejected_quotes"] += int((valid & ~(has_spread | use_last)).sum())
        valid &= has_spread | use_last
        quality["eligible_contracts"] += int(valid.sum())
        valid_quotes = selected.loc[valid].copy()
        if valid_quotes.empty:
            continue

        valid_quotes["T"] = T_years
        valid_quotes["dte_days"] = dte_days
        valid_quotes["expiration"] = exp_str

        frames.append(valid_quotes)

    if not frames:
        if chain_errors and successful_chains == 0:
            raise MarketDataError(f"Failed to fetch option chains for '{symbol}'") from chain_errors[-1]
        raise NoOptionsError(f"No usable option prices matched the criteria for symbol: '{symbol}'", quality)

    df = pd.concat(frames, ignore_index=True)[
        ["expiration", "dte_days", "T", "strike", "bid", "ask", "midPrice", "otype",
         "price_source", "lastTradeDate"]
    ]
    if (df["price_source"] == "last_trade").any():
        _report_warning("Using last-trade estimates for some options; prices may be stale "
                        "relative to spot and are not EOD quotes.", diagnostics)
    if chain_errors:
        _report_warning(f"Skipped {len(chain_errors)} failed option-chain requests for '{symbol}' "
                        f"({', '.join(failed_expirations)}).",
                        diagnostics)
    df.attrs["quality"] = quality
    return df, metrics


def build_surface_dataframe(
    symbol: str,
    option_type: Literal["call", "put", "otm"] = "otm",
    model: Literal["black_scholes", "bjerksund_stensland"] = "bjerksund_stensland",
    min_dte_days: int = 2,
    max_expirations: int | None = None,
    min_strike_pct: float = 0.70,
    max_strike_pct: float = 1.30,
    allow_last_price: bool = True,
    max_trade_age_days: float = 4,
    diagnostics: list[str] | None = None,
    max_dte_days: int = 365,
):
    """Fetch quotes, solve IV, and calculate Greeks for the selected model."""
    if model not in ("black_scholes", "bjerksund_stensland"):
        raise ValueError(f"Invalid model: '{model}'")
    df, metrics = fetch_raw_options(
        symbol=symbol,
        option_type=option_type,
        min_dte_days=min_dte_days,
        max_dte_days=max_dte_days,
        max_expirations=max_expirations,
        min_strike_pct=min_strike_pct,
        max_strike_pct=max_strike_pct,
        allow_last_price=allow_last_price,
        max_trade_age_days=max_trade_age_days,
        diagnostics=diagnostics,
    )
    spot = metrics["spot_price"]
    r = metrics["risk_free_rate"]
    q = metrics["dividend_yield"]

    solver_diagnostics = {}
    df["iv"] = solve_iv_robust(
        S=spot,
        K=df["strike"].to_numpy(),
        T=df["T"].to_numpy(),
        r=r,
        market_price=df["midPrice"].to_numpy(),
        otypes=df["otype"].to_numpy(),
        q=q,
        model=model, diagnostics=solver_diagnostics,
    )

    quality = df.attrs["quality"].copy()
    lower, upper = iv_price_bounds(spot, df["strike"].to_numpy(), df["T"].to_numpy(),
                                   r, q, df["otype"].to_numpy(), model)
    identifiable = (df["midPrice"].to_numpy() > lower + 1e-4) & (df["midPrice"].to_numpy() < upper)
    solved = identifiable & np.isfinite(df["iv"]) & (df["iv"] >= .001) & (df["iv"] <= 5.)
    quality["invalid_or_unidentifiable_prices"] = int((~identifiable).sum())
    unsupported = np.asarray(solver_diagnostics.get("unsupported_model", np.zeros(len(df), dtype=bool)))
    quality["unsupported_model"] = int((identifiable & unsupported & ~solved).sum())
    quality["unresolved_iv"] = int((identifiable & ~unsupported & ~solved).sum())
    quality["solved_contracts"] = int(solved.sum())
    dropped = len(df) - quality["solved_contracts"]
    if dropped:
        _report_warning(
            f"IV resolved for {quality['solved_contracts']} of {len(df)} eligible contracts; "
            f"{quality['invalid_or_unidentifiable_prices']} prices outside bounds or without identifiable time value; "
            f"{quality['unsupported_model']} outside the supported approximation domain; "
            f"{quality['unresolved_iv']} unresolved within the supported IV range (0.1%–500%).",
            diagnostics,
        )
    df = df.loc[solved].copy()
    if df.empty:
        raise NoOptionsError(
            f"Failed to resolve implied volatility for any usable contracts of '{symbol}'. "
            f"{quality['invalid_or_unidentifiable_prices']} prices have no identifiable IV; "
            f"{quality['unsupported_model']} contracts are outside the approximation domain; "
            f"{quality['unresolved_iv']} contracts did not converge in the supported range.", quality,
        )

    greeks = calculate_all_greeks(
        S=spot,
        K=df["strike"].to_numpy(),
        T=df["T"].to_numpy(),
        r=r,
        sigma=df["iv"].to_numpy(),
        q=q,
        otype=df["otype"].to_numpy(),
        model=model,
    )

    if any(not np.isfinite(values).all() for values in greeks.values()):
        _report_warning("Some Greeks are unavailable because a finite-difference bump leaves "
                        "the supported approximation domain or produces a nonfinite value.", diagnostics)

    # Bid/ask IVs measure quote-spread sensitivity, not a statistical confidence interval.
    # Last trades have no synchronized spread and must not inherit bid/ask bounds.
    df["iv_bid"] = np.nan
    df["iv_ask"] = np.nan
    midpoint = df["price_source"].eq("midpoint")
    for price_column, iv_column in (("bid", "iv_bid"), ("ask", "iv_ask")):
        if midpoint.any():
            df.loc[midpoint, iv_column] = solve_iv_robust(
                spot, df.loc[midpoint, "strike"].to_numpy(), df.loc[midpoint, "T"].to_numpy(),
                r, df.loc[midpoint, price_column].to_numpy(), df.loc[midpoint, "otype"].to_numpy(),
                q=q, model=model,
            )
    unresolved_bounds = midpoint & (df["iv_bid"].isna() | df["iv_ask"].isna())
    if unresolved_bounds.any():
        _report_warning(f"Bid/ask IV interval unavailable for {int(unresolved_bounds.sum())} midpoint contracts; "
                        "one or both spread endpoints have no identifiable IV in the supported range or approximation domain.", diagnostics)
    result = df.assign(iv_pct=df["iv"] * 100, **greeks).reset_index(drop=True)
    result.attrs["quality"] = quality
    return result, metrics
