import datetime as dt
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pandas as pd
import pytest

from api import data
from api.engine import black_scholes_price, bjerksund_stensland_price, calculate_all_greeks


@pytest.fixture(autouse=True)
def frozen_market(monkeypatch):
    class FrozenDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 19, 12, tzinfo=tz)

    monkeypatch.setattr(data, "dt", SimpleNamespace(
        datetime=FrozenDatetime, date=dt.date, time=dt.time, timezone=dt.timezone,
    ))
    # Never let a test fall through to the network.
    monkeypatch.setattr(data.yf, "Ticker", Mock(side_effect=AssertionError("Unexpected Yahoo request")))


def install_ticker(monkeypatch, ticker, tnx=None):
    if tnx is None:
        tnx = SimpleNamespace(fast_info=SimpleNamespace(last_price=4.2))
    factory = Mock(side_effect=lambda symbol: tnx if symbol == "^TNX" else ticker)
    monkeypatch.setattr(data.yf, "Ticker", factory)
    return factory


@pytest.mark.parametrize("dividend", [0.02, 0])
def test_market_metrics(monkeypatch, dividend):
    ticker = option_ticker(quotes(), quotes(), underlying={
        "regularMarketPrice": 100, "trailingAnnualDividendYield": dividend,
    })
    install_ticker(monkeypatch, ticker)
    _, metrics = data.fetch_raw_options("abc")
    assert metrics["dividend_source"] == "trailing"
    assert {k: v for k, v in metrics.items() if k != "dividend_source"} == pytest.approx({
        "symbol": "ABC", "spot_price": 100,
        "risk_free_rate": 0.042, "dividend_yield": dividend,
    })


@pytest.mark.parametrize("tnx_price", [None, np.nan, np.inf, 0])
def test_missing_treasury_yield_is_an_error(monkeypatch, tnx_price):
    ticker = option_ticker(quotes(), quotes())
    install_ticker(monkeypatch, ticker, SimpleNamespace(fast_info=SimpleNamespace(last_price=tnx_price)))
    with pytest.raises(ValueError, match="TNX yield"):
        data.fetch_raw_options("ABC")


@pytest.mark.parametrize("dividend", [np.nan, np.inf, -0.01])
def test_invalid_dividend_is_an_error(monkeypatch, dividend):
    ticker = option_ticker(quotes(), quotes(), underlying={
        "regularMarketPrice": 100, "trailingAnnualDividendYield": dividend,
    })
    install_ticker(monkeypatch, ticker)
    with pytest.raises(ValueError, match="Missing dividend data"):
        data.fetch_raw_options("ABC")


def quotes(strikes=(90, 100, 110)):
    return pd.DataFrame({"strike": strikes, "bid": 1.0, "ask": 1.2})


def option_ticker(calls=None, puts=None, expirations=("2026-10-21",), underlying=None):
    if underlying is None:
        underlying = {"regularMarketPrice": 100, "trailingAnnualDividendYield": 0}
    chain = SimpleNamespace(calls=calls, puts=puts, underlying=underlying)
    return SimpleNamespace(options=expirations, option_chain=Mock(return_value=chain))


@pytest.mark.parametrize("kind,expected_types,expected_prices", [
    ("otm", [-1, 1, 1], [2.1, 1.1, 1.1]),
    ("call", [1, 1, 1], [1.1, 1.1, 1.1]),
    ("put", [-1, -1, -1], [2.1, 2.1, 2.1]),
])
def test_option_selection(monkeypatch, kind, expected_types, expected_prices):
    calls = quotes()
    puts = quotes().assign(bid=2.0, ask=2.2)
    install_ticker(monkeypatch, option_ticker(calls, puts))
    frame, _ = data.fetch_raw_options("ABC", option_type=kind)
    assert frame["strike"].tolist() == [90, 100, 110]
    assert frame["otype"].tolist() == expected_types
    assert frame["midPrice"].tolist() == pytest.approx(expected_prices)


def test_dte_filtering_and_pricing_time(monkeypatch):
    ticker = option_ticker(quotes(), quotes(), (
        "2026-10-26", "2026-10-20", "2026-10-21",
    ))
    install_ticker(monkeypatch, ticker)
    frame, _ = data.fetch_raw_options("ABC")
    assert sorted(frame["dte_days"].unique()) == [2, 7]
    assert frame.iloc[0]["T"] == pytest.approx((2 + 4 / 24) / 365)
    frame, _ = data.fetch_raw_options("ABC", max_expirations=1)
    assert frame["expiration"].unique().tolist() == ["2026-10-21"]


def test_missing_side_and_bad_quotes(monkeypatch):
    calls = pd.DataFrame({
        "strike": [100, 105, 110, 115, 120, 140],
        "bid": [1, 0, 1, 2, np.nan, 1],
        "ask": [1.2, 1, np.inf, 1, 2, 1.2],
    })
    install_ticker(monkeypatch, option_ticker(calls, None))
    frame, _ = data.fetch_raw_options("ABC")
    assert frame["strike"].tolist() == [100]


@pytest.mark.parametrize("kwargs", [
    {"option_type": "other"},
    {"min_strike_pct": 1.3, "max_strike_pct": 0.7}, {"max_expirations": 0},
])
def test_invalid_inputs_fail_before_fetching(kwargs):
    with pytest.raises(ValueError):
        data.fetch_raw_options("ABC", **kwargs)


@pytest.mark.parametrize("model", ["black_scholes", "bjerksund_stensland"])
def test_build_surface_dataframe_pipeline(monkeypatch, model):
    pricing = black_scholes_price if model == "black_scholes" else bjerksund_stensland_price
    maturity = (7 + 4 / 24) / 365
    strikes = np.array([98, 100, 102])
    frames = []
    for otype in (1, -1):
        prices = pricing(100, strikes, maturity, 0.042, 0.3, q=0.02, otype=otype)
        frames.append(pd.DataFrame({"strike": strikes, "bid": prices - 0.01, "ask": prices + 0.01}))
    ticker = option_ticker(*frames, expirations=("2026-10-26",), underlying={
        "regularMarketPrice": 100, "trailingAnnualDividendYield": 0.02,
    })
    install_ticker(monkeypatch, ticker)
    frame, metrics = data.build_surface_dataframe("ABC", model=model)
    assert len(frame) == 3
    assert np.allclose(frame["iv"], 0.3, atol=1e-3)
    assert np.allclose(frame["iv_pct"], 30, atol=0.1)
    expected_greeks = calculate_all_greeks(
        S=100, K=strikes, T=maturity, r=0.042, sigma=frame["iv"].to_numpy(),
        q=0.02, otype=np.array([-1, 1, 1]), model=model,
    )
    for name, expected in expected_greeks.items():
        assert np.allclose(frame[name], expected), name
    assert metrics["spot_price"] == 100


def test_invalid_solver_results_filtered(monkeypatch):
    install_ticker(monkeypatch, option_ticker(quotes(), quotes()))
    monkeypatch.setattr(data, "solve_iv_robust", Mock(side_effect=[
        np.array([np.nan, .2, .3]), np.array([.15, .25]), np.array([.25, .35]),
    ]))
    frame, _ = data.build_surface_dataframe("ABC", model="black_scholes", diagnostics=[])
    assert frame["strike"].tolist() == [100, 110]
    assert frame["iv"].tolist() == [0.2, 0.3]
    assert frame.index.tolist() == [0, 1]


def test_no_valid_solver_results_is_an_error(monkeypatch):
    install_ticker(monkeypatch, option_ticker(quotes(), quotes()))
    monkeypatch.setattr(data, "solve_iv_robust", lambda **_: np.full(3, np.nan))
    with pytest.warns(UserWarning, match="IV resolved for 0"), pytest.raises(ValueError, match="Failed to resolve"):
        data.build_surface_dataframe("ABC")


def test_chain_metrics_reused_without_extra_ticker_requests(monkeypatch):
    # No .info or .fast_info on this ticker: complete chain data must suffice.
    ticker = option_ticker(quotes(), quotes(), expirations=("2026-10-21", "2026-10-26"), underlying={
        "regularMarketPrice": 100, "trailingAnnualDividendYield": 0.02,
    })
    factory = install_ticker(monkeypatch, ticker)
    frame, metrics = data.fetch_raw_options("ABC")
    assert len(frame) == 6
    assert metrics["spot_price"] == 100
    assert metrics["dividend_yield"] == 0.02
    assert factory.call_args_list == [call("ABC"), call("^TNX")]
    assert ticker.option_chain.call_args_list == [
        call("2026-10-21"), call("2026-10-26"),
    ]


def test_missing_chain_spot_is_an_error(monkeypatch):
    ticker = option_ticker(quotes(), quotes(), underlying={
        "regularMarketPrice": np.nan, "trailingAnnualDividendYield": 0.02,
    })
    install_ticker(monkeypatch, ticker)
    with pytest.raises(ValueError, match="spot price in option chain"):
        data.fetch_raw_options("ABC")


def test_one_failed_chain_does_not_discard_other_expirations(monkeypatch):
    ticker = option_ticker(quotes(), quotes(), expirations=("2026-10-21", "2026-10-26"))
    chain = ticker.option_chain.return_value
    ticker.option_chain.side_effect = [RuntimeError("Yahoo request failed"), chain]
    install_ticker(monkeypatch, ticker)
    with pytest.warns(UserWarning, match="Skipped 1 failed"):
        frame, _ = data.fetch_raw_options("ABC")
    assert frame["expiration"].unique().tolist() == ["2026-10-26"]
    assert ticker.option_chain.call_count == 2


def test_all_failed_chains_raise_an_error(monkeypatch):
    ticker = option_ticker(quotes(), quotes(), expirations=("2026-10-21", "2026-10-26"))
    ticker.option_chain.side_effect = RuntimeError("Yahoo request failed")
    install_ticker(monkeypatch, ticker)
    with pytest.raises(ValueError, match="Failed to fetch option chains"):
        data.fetch_raw_options("ABC")


@pytest.mark.parametrize("field", ["missing", None])
def test_missing_dividend_is_an_explicit_assumption(monkeypatch, field):
    underlying = {"regularMarketPrice": 100}
    if field is None:
        underlying["trailingAnnualDividendYield"] = None
    install_ticker(monkeypatch, option_ticker(quotes(), quotes(), underlying=underlying))
    with pytest.warns(UserWarning, match="assuming q=0"):
        _, metrics = data.fetch_raw_options("ABC")
    assert metrics["dividend_yield"] == 0
    assert metrics["dividend_source"] == "assumed_zero"


def test_forward_cash_dividend_preferred_over_trailing(monkeypatch):
    install_ticker(monkeypatch, option_ticker(quotes(), quotes(), underlying={
        "regularMarketPrice": 100, "dividendRate": 3,
        "trailingAnnualDividendYield": 0.02,
    }))
    _, metrics = data.fetch_raw_options("ABC")
    assert metrics["dividend_yield"] == 0.03
    assert metrics["dividend_source"] == "forward"


def test_empty_chain_does_not_block_next_expiration(monkeypatch):
    ticker = option_ticker(quotes(), quotes(), expirations=("2026-10-21", "2026-10-26"))
    ticker.option_chain.side_effect = [
        SimpleNamespace(calls=None, puts=None, underlying=None), ticker.option_chain.return_value,
    ]
    install_ticker(monkeypatch, ticker)
    frame, _ = data.fetch_raw_options("ABC")
    assert frame["expiration"].unique().tolist() == ["2026-10-26"]


def test_last_trade_fallback_filters_age_and_preserves_source(monkeypatch):
    # Monday noon NY: Friday is usable; old, unknown and future trades are not.
    calls = pd.DataFrame({
        "strike": [100, 101, 102, 103, 104, 105, 106],
        "bid": [0, 0, 0, 0, 0, 0, 2],
        "ask": [0, 0, 0, 0, 0, 0, 1],
        "lastPrice": [1.5, 1.6, 1.7, 1.8, np.inf, -1, 1.9],
        "lastTradeDate": ["2026-10-16T19:00Z", "2026-10-12T19:00Z", None,
                          "2026-10-19T17:00Z", "2026-10-16T19:00Z",
                          "2026-10-16T19:00Z", "2026-10-16T19:00Z"],
    })
    install_ticker(monkeypatch, option_ticker(calls, None))
    with pytest.warns(UserWarning, match="last-trade estimates"):
        frame, _ = data.fetch_raw_options("ABC")
    assert frame["strike"].tolist() == [100]
    assert frame["midPrice"].tolist() == [1.5]
    assert frame["price_source"].tolist() == ["last_trade"]
    assert frame["lastTradeDate"].notna().all()
    with pytest.raises(ValueError, match="No usable option prices"):
        data.fetch_raw_options("ABC", allow_last_price=False)
    with pytest.raises(ValueError, match="No usable option prices"):
        data.fetch_raw_options("ABC", max_trade_age_days=1)


def test_valid_spread_wins_over_last_trade(monkeypatch):
    calls = quotes().assign(lastPrice=50, lastTradeDate="2026-10-16T19:00Z")
    install_ticker(monkeypatch, option_ticker(calls, None))
    frame, _ = data.fetch_raw_options("ABC")
    assert frame["midPrice"].tolist() == pytest.approx([1.1, 1.1])
    assert frame["price_source"].tolist() == ["midpoint", "midpoint"]


@pytest.mark.parametrize("model", ["black_scholes", "bjerksund_stensland"])
def test_last_trade_pipeline(monkeypatch, model):
    maturity = (7 + 4 / 24) / 365
    strikes = np.array([100, 102])
    pricing = black_scholes_price if model == "black_scholes" else bjerksund_stensland_price
    calls = pd.DataFrame({
        "strike": strikes, "bid": 0.0, "ask": 0.0,
        "lastPrice": pricing(100, strikes, maturity, 0.042, 0.3),
        "lastTradeDate": "2026-10-19T15:45Z",
    })
    install_ticker(monkeypatch, option_ticker(calls, None, expirations=("2026-10-26",)))
    with pytest.warns(UserWarning, match="last-trade estimates"):
        frame, _ = data.build_surface_dataframe("ABC", model=model)
    assert np.allclose(frame["iv"], 0.3, atol=1e-3)
    assert frame["price_source"].eq("last_trade").all()


def test_missing_quote_values_can_use_timestamped_trade(monkeypatch):
    calls = pd.DataFrame({"strike": [100], "bid": [None], "ask": [None],
                          "lastPrice": [1.5], "lastTradeDate": ["2026-10-16T19:00Z"]})
    install_ticker(monkeypatch, option_ticker(calls, None))
    with pytest.warns(UserWarning, match="last-trade estimates"):
        frame, _ = data.fetch_raw_options("ABC")
    assert frame["price_source"].tolist() == ["last_trade"]


@pytest.mark.parametrize("age", [0, -1, np.nan, np.inf])
def test_invalid_trade_age_fails_before_network(age):
    with pytest.raises(ValueError, match="maximum trade age"):
        data.fetch_raw_options("ABC", max_trade_age_days=age)


def test_positive_equal_bid_ask_is_a_valid_midpoint(monkeypatch):
    install_ticker(monkeypatch, option_ticker(quotes((100,)).assign(bid=1., ask=1.), None))
    frame, _ = data.fetch_raw_options("ABC", allow_last_price=False)
    assert frame["midPrice"].tolist() == [1.]
    assert frame["price_source"].tolist() == ["midpoint"]


def test_quality_counts_partition_inputs_and_separate_price_from_solver_errors(monkeypatch):
    calls = pd.DataFrame({"strike": [90, 100, 110, 120, 140], "bid": [1, 1, 1, 2, 1], "ask": [1.2, 1.2, 1.2, 1, 1.2]})
    install_ticker(monkeypatch, option_ticker(calls, None))
    monkeypatch.setattr(data, "solve_iv_robust", Mock(side_effect=[
        np.array([np.nan, .3, np.nan]), np.array([.2]), np.array([.4]),
    ]))
    diagnostics = []
    frame, _ = data.build_surface_dataframe("ABC", option_type="call", model="black_scholes", diagnostics=diagnostics)
    assert frame.attrs["quality"] == {
        "received_contracts": 5, "excluded_by_filters": 1, "rejected_quotes": 1,
        "eligible_contracts": 3, "invalid_or_unidentifiable_prices": 1,
        "unsupported_model": 0, "unresolved_iv": 1, "solved_contracts": 1,
    }
    assert any("1 unresolved" in message for message in diagnostics)


@pytest.mark.parametrize("model,pricing", [
    ("black_scholes", black_scholes_price), ("bjerksund_stensland", bjerksund_stensland_price),
])
def test_bid_ask_iv_interval_reprices_spread_and_handles_equal_quotes(monkeypatch, model, pricing):
    T = (7 + 4 / 24) / 365
    strikes = np.array([100., 102.])
    bids = pricing(100, strikes, T, .042, np.array([.25, .3]), 0., 1.)
    asks = pricing(100, strikes, T, .042, np.array([.35, .3]), 0., 1.)
    calls = pd.DataFrame({"strike": strikes, "bid": bids, "ask": asks})
    install_ticker(monkeypatch, option_ticker(calls, None, expirations=("2026-10-26",)))
    frame, _ = data.build_surface_dataframe("ABC", model=model)
    assert np.allclose(frame["iv_bid"], [.25, .3], atol=1e-4)
    assert np.allclose(frame["iv_ask"], [.35, .3], atol=1e-4)
    assert np.all(frame["iv_bid"] <= frame["iv"] + 1e-5)
    assert np.all(frame["iv_ask"] >= frame["iv"] - 1e-5)
    assert frame.attrs["quality"]["solved_contracts"] == 2


def test_last_trades_never_inherit_bid_ask_iv_intervals(monkeypatch):
    calls = pd.DataFrame({"strike": [100], "bid": [0.], "ask": [0.], "lastPrice": [1.5],
                          "lastTradeDate": ["2026-10-19T15:45Z"]})
    install_ticker(monkeypatch, option_ticker(calls, None))
    frame, _ = data.build_surface_dataframe("ABC", diagnostics=[])
    assert frame[["iv_bid", "iv_ask"]].isna().all().all()


def test_unsupported_approximation_has_its_own_quality_bucket(monkeypatch):
    puts = pd.DataFrame({"strike": [99., 100.], "bid": [.29, .004], "ask": [.31, .006]})
    install_ticker(monkeypatch, option_ticker(None, puts, expirations=("2027-10-19",)))
    diagnostics = []
    frame, _ = data.build_surface_dataframe("ABC", option_type="put", diagnostics=diagnostics)
    quality = frame.attrs["quality"]
    assert frame["strike"].tolist() == [99.]
    assert quality["eligible_contracts"] == 2
    assert quality["unsupported_model"] == 1
    assert quality["unresolved_iv"] == 0
    assert quality["invalid_or_unidentifiable_prices"] == 0
    assert quality["solved_contracts"] == 1
    assert any("1 outside the supported approximation domain" in message for message in diagnostics)


def test_all_unsupported_prices_preserve_reason_in_no_options_error(monkeypatch):
    puts = pd.DataFrame({"strike": [100.], "bid": [.004], "ask": [.006]})
    install_ticker(monkeypatch, option_ticker(None, puts, expirations=("2027-10-19",)))
    with pytest.raises(data.NoOptionsError, match="outside the approximation domain") as error:
        data.build_surface_dataframe("ABC", option_type="put", diagnostics=[])
    assert error.value.quality["unsupported_model"] == 1
    assert error.value.quality["unresolved_iv"] == 0
