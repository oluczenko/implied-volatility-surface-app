"""Save a dated Yahoo snapshot for the frontend's offline demonstration mode."""

from pathlib import Path

from api.index import SurfaceQuery, surface

if __name__ == "__main__":
    response = surface(SurfaceQuery(symbol="AAPL", model="black_scholes", max_dte_days=365))
    target = Path(__file__).resolve().parents[1] / "frontend" / "lib" / "demo.json"
    target.write_text(response.model_dump_json(indent=2) + "\n")
    print(f"Saved {len(response.contracts)} contracts to {target}")
