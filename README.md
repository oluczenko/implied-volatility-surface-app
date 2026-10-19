# Volsurface — Implied Volatility Explorer

Next.js + TypeScript + Plotly.js dashboard backed by a NumPy/SciPy FastAPI engine.
Explore a 3D implied volatility surface, 2D volatility smiles and quote-level Greeks.

## Local development

Requires Python 3.12+ and Node.js 22.18+.

First terminal, from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn api.index:app --reload
```

Second terminal:

```sh
cd frontend
npm ci
npm run dev
```

Open http://localhost:3000. Next.js proxies `/api/*` and `/docs` to the local API
on port 8000. Set `BACKEND_URL` when using another local backend address.

The page initially shows a **saved Yahoo Finance snapshot** with its capture time.
Fetching new data replaces it only after a successful response. Use **Restore AAPL
snapshot** to return to the saved example. Refresh the checked-in snapshot with:

```sh
python -m scripts.refresh_demo
```

The snapshot is captured with Black–Scholes; another model selection takes effect
when new market data is fetched. The dataset header always shows the model used
for the displayed results.

## Checks

```sh
pytest -q
npm --prefix frontend run test
npm --prefix frontend run typecheck
npm --prefix frontend run build
```

## One project on Vercel

Import the repository with **Root Directory `./`**, not `frontend/`. The root
`vercel.json` uses [Vercel Services](https://vercel.com/docs/services), currently
beta on all plans, to build the Next.js frontend and FastAPI backend together:

- `/` and frontend assets → `frontend/` (Next.js).
- `/api/*`, `/docs`, `/openapi.json`, `/redoc` → the Python backend.

Keep project-level build/install/output overrides unset; build settings belong
to the individual services. Remove the old standalone FastAPI preset/overrides
if the project was imported before the Services configuration was added. Use the
repository configuration as the source of truth.

No API key or frontend backend-URL environment variable is required. Both apps
share one deployment domain. To run the Services router locally with a current
Vercel CLI:

```sh
vercel dev -L
```

Deployment still needs a hosted smoke test of `/api/health` and `/api/surface`.
Yahoo may rate-limit or block hosting IPs. The backend has a 60-second platform
duration limit; the browser stops waiting after 55 seconds and preserves the
previous chart. There is no durable cache in this version.

## Data and model assumptions

- Bid/ask midpoints take priority. Optional last-trade estimates require a valid
  timestamp at most four calendar days old. Price source and warnings remain visible.
- The snapshot capture time and response time are **not** quote timestamps.
- Black–Scholes assumes European exercise. Bjerksund–Stensland approximates
  American early exercise. Both use a continuous dividend yield.
- Dividend input prefers annual forward cash dividend / spot, then trailing yield.
  Missing dividend data is explicitly marked as an assumed zero.
- The 10-year Treasury yield is a single rate proxy, not a maturity-matched curve.
- Display interpolation is linear in strike and maturity, with no strike
  extrapolation. It is not an arbitrage-free surface calibration. Raw quotes
  can be overlaid; duplicate call/put strikes are averaged only for display.
- Vega/rho are per percentage point; theta is per calendar day. Log-moneyness
  is `ln(K / spot)`. US equity expiration is assumed to be 16:00 New York.

## Structure

- `frontend/`: dashboard, Plotly charts, saved snapshot and interpolation tests.
- `api/engine.py`: pricing, IV solver and Greeks.
- `api/data.py`: market data fetching and filtering.
- `api/index.py`: HTTP endpoints and response schemas.
- `tests/`: numerical, data pipeline and API tests.
- `scripts/refresh_demo.py`: capture the offline example through the same pipeline.

For educational and portfolio use.
