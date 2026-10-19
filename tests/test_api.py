import datetime as dt
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import data, index


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(data.yf, "Ticker", Mock(side_effect=AssertionError("Unexpected network request")))
    return TestClient(index.app)


@pytest.fixture
def pipeline(monkeypatch):
    frame = pd.DataFrame([{
        "expiration": "2026-11-20", "dte_days": np.int64(32), "T": 32 / 365,
        "strike": np.float64(100), "bid": np.nan, "ask": np.inf,
        "midPrice": 3.0, "otype": 1.0, "price_source": "last_trade",
        "lastTradeDate": pd.Timestamp("2026-10-19T15:45:00Z"),
        "iv": 0.3, "iv_pct": 30, "delta": 0.5, "gamma": 0.03,
        "vega": 0.2, "theta": -0.04, "rho": np.nan,
    }])
    frame.attrs["quality"] = {
        "received_contracts": 5, "excluded_by_filters": 1, "rejected_quotes": 1,
        "eligible_contracts": 3, "invalid_or_unidentifiable_prices": 1,
        "unsupported_model": 0, "unresolved_iv": 1, "solved_contracts": 1,
    }
    metrics = {"symbol": "AAPL", "spot_price": np.float64(100),
               "risk_free_rate": 0.04, "dividend_yield": 0,
               "dividend_source": "assumed_zero"}

    def build(**kwargs):
        kwargs["diagnostics"].append("Last-trade estimate; missing dividend data assumes q=0.")
        return frame, metrics

    mock = Mock(side_effect=build)
    monkeypatch.setattr(index, "build_surface_dataframe", mock)
    return mock


def test_health_without_market_requests(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_surface_json_and_defaults(client, pipeline):
    response = client.get("/api/surface", params={"symbol": " aapl "})
    assert response.status_code == 200, response.text
    payload = response.json()
    contract = payload["contracts"][0]
    assert payload["metrics"]["symbol"] == "AAPL"
    assert payload["model"] == "bjerksund_stensland"
    assert payload["warnings"] == ["Last-trade estimate; missing dividend data assumes q=0."]
    assert dt.datetime.fromisoformat(payload["generated_at"]).utcoffset() == dt.timedelta(0)
    assert contract["bid"] is None and contract["ask"] is None and contract["rho"] is None
    assert contract["lastTradeDate"] == "2026-10-19T15:45:00Z"
    assert contract["expiration"] == "2026-11-20"
    assert contract["iv"] == 0.3
    assert contract["iv_bid"] is None and contract["iv_ask"] is None
    assert payload["quality"]["unresolved_iv"] == 1
    assert payload["quality"]["solved_contracts"] == len(payload["contracts"])
    args = pipeline.call_args.kwargs
    assert args["symbol"] == "AAPL"
    assert args["max_expirations"] == 4
    assert args["allow_last_price"] is True


def test_surface_passes_custom_parameters(client, pipeline):
    params = {"symbol": "BRK-B", "model": "black_scholes", "option_type": "put",
              "max_expirations": 8, "min_dte_days": 7,
              "min_strike_pct": 0.8, "max_strike_pct": 1.1,
              "allow_last_price": "false", "max_trade_age_days": 1.5}
    assert client.get("/api/surface", params=params).status_code == 200
    args = pipeline.call_args.kwargs
    assert args["allow_last_price"] is False
    assert args["model"] == "black_scholes"
    assert args["max_trade_age_days"] == 1.5
    assert args["max_expirations"] == 8


@pytest.mark.parametrize("params", [
    {}, {"symbol": ""}, {"symbol": "A/B"}, {"symbol": "A" * 21},
    {"symbol": "AAPL", "model": "unknown"},
    {"symbol": "AAPL", "option_type": "both"},
    {"symbol": "AAPL", "max_expirations": 0},
    {"symbol": "AAPL", "max_expirations": 9},
    {"symbol": "AAPL", "max_expirations": 1.5},
    {"symbol": "AAPL", "min_dte_days": -1},
    {"symbol": "AAPL", "min_strike_pct": 0},
    {"symbol": "AAPL", "min_strike_pct": 1.4, "max_strike_pct": 1.1},
    {"symbol": "AAPL", "max_strike_pct": "inf"},
    {"symbol": "AAPL", "max_strike_pct": "nan"},
    {"symbol": "AAPL", "max_trade_age_days": 8},
    {"symbol": "AAPL", "unknown": "ignored?"},
])
def test_invalid_parameters_do_not_call_pipeline(client, pipeline, params):
    response = client.get("/api/surface", params=params)
    assert response.status_code == 422, response.text
    pipeline.assert_not_called()


@pytest.mark.parametrize("error,status,code", [
    (data.NoOptionsError("No usable options"), 404, "no_usable_options"),
    (data.MarketDataError("Internal Yahoo details"), 502, "market_data_unavailable"),
])
def test_expected_errors(client, pipeline, error, status, code):
    pipeline.side_effect = error
    response = client.get("/api/surface?symbol=AAPL")
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert "Internal Yahoo details" not in response.text


def test_unexpected_errors_remain_server_errors(monkeypatch):
    monkeypatch.setattr(index, "build_surface_dataframe", Mock(side_effect=RuntimeError("private details")))
    with TestClient(index.app, raise_server_exceptions=False) as client:
        response = client.get("/api/surface?symbol=AAPL")
    assert response.status_code == 500
    assert "private details" not in response.text


def test_openapi_documents_contract(client):
    schema = client.get("/openapi.json").json()
    assert "/api/health" in schema["paths"]
    assert "/api/surface" in schema["paths"]
    assert "SurfaceResponse" in schema["components"]["schemas"]


def test_no_options_error_preserves_quality_diagnostics(client, pipeline):
    counts = {"received_contracts": 2, "excluded_by_filters": 0, "rejected_quotes": 0,
              "eligible_contracts": 2, "invalid_or_unidentifiable_prices": 1,
              "unsupported_model": 0, "unresolved_iv": 1, "solved_contracts": 0}
    pipeline.side_effect = data.NoOptionsError("No resolvable contracts", counts)
    response = client.get("/api/surface?symbol=AAPL")
    assert response.status_code == 404
    assert response.json()["detail"]["quality"] == counts
