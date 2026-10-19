# Implied Volatility Surface

Options analytics built with Python, NumPy/SciPy and FastAPI. Fetches option
chains from Yahoo Finance, estimates implied volatility using Black–Scholes or
Bjerksund–Stensland, and calculates Greeks.

## Run the API

Python 3.12+:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn api.index:app --reload
```

Interactive documentation: http://localhost:8000/docs

```sh
pytest -q
```

## Structure

- `api/engine.py`: pricing, implied volatility solver and Greeks.
- `api/data.py`: market data fetching and filtering.
- `api/index.py`: HTTP endpoints and response schemas.
- `tests/`: numerical, data pipeline and API tests.

## Next step

A Next.js/Plotly.js frontend with a 3D IV surface and 2D volatility smiles is
planned. The combined frontend/backend Vercel configuration is not implemented
yet; the current configuration deploys only FastAPI.

For educational and portfolio use. Yahoo quotes may be delayed or unavailable;
last-trade estimates can be stale relative to spot. The API reports price sources
and data quality warnings.
