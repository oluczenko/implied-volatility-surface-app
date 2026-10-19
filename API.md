# FastAPI backend

The NumPy/SciPy pipeline is exposed through a synchronous FastAPI endpoint.
The existing Streamlit interface remains available separately; Next.js is not
implemented yet.

## Local development

From the repository root (Python 3.12+):

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn api.index:app --reload
```

Open http://localhost:8000/docs for interactive API documentation.

```sh
curl http://localhost:8000/api/health
curl 'http://localhost:8000/api/surface?symbol=AAPL&max_expirations=2'
pytest -q
```

`/api/health` checks the application only; it does not call Yahoo.

## Surface contract

`GET /api/surface` accepts:

| Parameter | Default | Constraints |
| --- | --- | --- |
| `symbol` | required | Trimmed, uppercased, max 20 characters |
| `model` | `bjerksund_stensland` | `black_scholes`, `bjerksund_stensland` |
| `option_type` | `otm` | `call`, `put`, `otm` |
| `min_dte_days` | `2` | 0–3650 calendar days |
| `max_expirations` | `4` | 1–8 |
| `min_strike_pct` | `0.70` | >0 and <=5; fraction of spot |
| `max_strike_pct` | `1.30` | >= min and <=5 |
| `allow_last_price` | `true` | Set false to require valid bid/ask spreads |
| `max_trade_age_days` | `4` | >0 and <=7 calendar days |

Response fields:

- `generated_at`: UTC response generation time, not the market quote timestamp.
- `model`: the selected pricing model.
- `metrics`: symbol, spot, risk-free rate, dividend yield and dividend source.
- `contracts`: quotes, expiration, IV and Greeks for contracts with a solved IV.
- `warnings`: data quality and skipped-request messages for the UI to display.

Rates and `iv` are decimal fractions; `iv_pct` is a percentage. `T` uses ACT/365
with an assumed US equity expiration at 16:00 New York. Vega/rho are per percentage
point and theta is per calendar day. Non-finite values serialize as `null`; dates
and timestamps use ISO 8601.

`midPrice` is the selected pricing input for backward compatibility. Check
`price_source`: `midpoint` or `last_trade`. Last trades require a valid timestamp
within the age limit. They may be stale relative to spot; they are not synchronized
EOD data. Zero/missing bid AND ask can use the fallback; crossed/negative quotes
are rejected. Even valid spreads can be stale; Yahoo does not provide a quote
freshness guarantee here.

Dividend input prefers the annual forward cash dividend / spot, then trailing
yield. Missing dividend data is explicitly marked `assumed_zero` and warned about.
The Treasury 10-year yield is a single rate proxy, not a maturity-matched curve.

Errors:

- `422`: invalid query parameters (including unknown parameters).
- `404`, `detail.code=no_usable_options`: no option chains, no matching quotes,
  or no converged IVs.
- `502`, `detail.code=market_data_unavailable`: upstream request failure or
  missing/invalid required market inputs. Provider details remain in server logs.
- `500`: unexpected implementation error.

## Vercel deployment

Deploy the repository root as a **FastAPI** project. `pyproject.toml` explicitly
selects `api.index:app` so the legacy Streamlit `app.py` is not the entrypoint.
`requirements.txt` includes only API dependencies, and `vercel.json` sets the
resolved entrypoint's duration to 60 seconds. `.vercelignore` excludes legacy UI,
local environments and internal files.

Select Python 3.12 or later in the project settings. With an authenticated CLI:

```sh
vercel deploy
```

Check the preview deployment before promotion:

```sh
curl https://YOUR-PREVIEW.vercel.app/api/health
curl 'https://YOUR-PREVIEW.vercel.app/api/surface?symbol=AAPL&max_expirations=2'
```

If deployment protection is enabled, authenticate the requests first. Yahoo may
rate-limit or block datacenter IPs; a successful local live request does not verify
hosted access. The platform duration is a hard limit, not an application timeout
response. No durable cache or demonstration snapshot is included in this stage.

For Next.js, use a server-side request or rewrite `/api/*` to the backend so the
browser uses one origin. Direct browser calls across origins need an explicit CORS
configuration once the frontend domain is known.

Official deployment reference: https://vercel.com/docs/frameworks/backend/fastapi

## Legacy Streamlit app

```sh
pip install -r requirements-streamlit.txt
streamlit run app.py
```
