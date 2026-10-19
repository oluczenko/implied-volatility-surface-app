"""HTTP interface for the options analytics pipeline."""

import json
import logging
from datetime import date, datetime, timezone
from typing import Annotated, Literal, Self

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .data import MarketDataError, NoOptionsError, build_surface_dataframe

logger = logging.getLogger(__name__)
app = FastAPI(title="Implied Volatility Surface API", version="1.0.0")


class SurfaceQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    symbol: str = Field(min_length=1, max_length=20, pattern=r"^[A-Z0-9^][A-Z0-9.^=\-]*$")
    model: Literal["black_scholes", "bjerksund_stensland"] = "bjerksund_stensland"
    option_type: Literal["call", "put", "otm"] = "otm"
    min_dte_days: int = Field(default=2, ge=0, le=365)
    max_dte_days: int = Field(
        default=365, ge=1, le=365,
        description="Inclusive expiry horizon in calendar days; returns all available dates.",
    )
    max_expirations: int | None = Field(
        default=None, ge=1,
        description="Optional date-count cap for older clients; omitted by the dashboard.",
    )
    min_strike_pct: float = Field(default=0.70, gt=0, le=5)
    max_strike_pct: float = Field(default=1.30, gt=0, le=5)
    allow_last_price: bool = True
    max_trade_age_days: float = Field(default=4, gt=0, le=7)

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.min_strike_pct > self.max_strike_pct:
            raise ValueError("min_strike_pct must not exceed max_strike_pct")
        if self.min_dte_days > self.max_dte_days:
            raise ValueError("min_dte_days must not exceed max_dte_days")
        return self


class MarketMetrics(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    symbol: str
    spot_price: float
    risk_free_rate: float
    dividend_yield: float
    dividend_source: Literal["forward", "trailing", "assumed_zero"]


class Contract(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    expiration: date
    dte_days: int
    T: float
    strike: float
    bid: float | None
    ask: float | None
    midPrice: float
    otype: Literal[-1.0, 1.0]
    price_source: Literal["midpoint", "last_trade"]
    lastTradeDate: datetime | None
    iv: float
    iv_pct: float
    iv_bid: float | None = None
    iv_ask: float | None = None
    delta: float | None
    gamma: float | None
    vega: float | None
    theta: float | None
    rho: float | None


class QualitySummary(BaseModel):
    received_contracts: int = Field(ge=0)
    excluded_by_filters: int = Field(ge=0)
    rejected_quotes: int = Field(ge=0)
    eligible_contracts: int = Field(ge=0)
    invalid_or_unidentifiable_prices: int = Field(ge=0)
    unsupported_model: int = Field(default=0, ge=0)
    unresolved_iv: int = Field(ge=0)
    solved_contracts: int = Field(ge=0)


class SurfaceResponse(BaseModel):
    generated_at: datetime
    model: Literal["black_scholes", "bjerksund_stensland"]
    metrics: MarketMetrics
    contracts: list[Contract]
    warnings: list[str]
    quality: QualitySummary | None = None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/surface", response_model=SurfaceResponse)
def surface(params: Annotated[SurfaceQuery, Query()]) -> SurfaceResponse:
    diagnostics: list[str] = []
    try:
        frame, metrics = build_surface_dataframe(**params.model_dump(), diagnostics=diagnostics)
    except NoOptionsError as exc:
        raise HTTPException(status_code=404, detail={
            "code": "no_usable_options", "message": str(exc), "quality": exc.quality,
        }) from exc
    except MarketDataError as exc:
        logger.warning("Market data unavailable for %s: %s", params.symbol, exc)
        raise HTTPException(status_code=502, detail={
            "code": "market_data_unavailable",
            "message": "Market data is temporarily unavailable. Please try again later.",
        }) from exc

    # Pandas handles NumPy scalars, NaN/Inf/NaT -> null, and ISO timestamps.
    contracts = json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))
    return SurfaceResponse(
        generated_at=datetime.now(timezone.utc), model=params.model,
        metrics=MarketMetrics(**metrics), contracts=contracts, warnings=diagnostics,
        quality=frame.attrs.get("quality"),
    )
