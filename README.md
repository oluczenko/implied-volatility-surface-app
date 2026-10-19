# Volsurface — Implied Volatility Explorer

**From market option prices to an interactive view of volatility, skew and risk.**

Volsurface is a quantitative finance application that retrieves option chains
from Yahoo Finance, solves for implied volatility and visualizes how option
pricing changes across strikes and maturities. A Python numerical engine powers
an interactive Next.js dashboard with European and American pricing models.

**[Explore the live application →](https://iv-surface-chi.vercel.app/)**

## Explore the surface

- **Interactive 3D surface:** rotate and zoom through implied volatility across
  strike and time to expiry, with an optional overlay of observed contracts.
- **Volatility smiles and skew:** inspect one expiration at a time and hover over
  individual options to see their prices, IV and Greeks.
- **Two pricing models:** select Black–Scholes for European exercise or the
  Bjerksund–Stensland approximation for American exercise.
- **Flexible market selection:** choose a ticker, calls, puts or OTM puts and
  calls, an expiry horizon of 7, 30, 90, 180 or 365 days, and a strike range relative
  to spot. All available expirations within the horizon are requested; the
  default horizon is one year, with a minimum of two days to expiry.
- **Strike or log-moneyness:** switch the surface's horizontal axis between
  strike and $\ln(K/S)$.
- **Quote quality in view:** distinguish bid/ask midpoints from last-trade
  estimates, inspect rejection counts and filter estimates out of both charts.
- **Bid/ask IV ranges and CSV export:** inspect sensitivity to the quoted spread
  and export contract-level results, including Greeks and spread endpoint IVs.

The application opens with a dated AAPL snapshot, so the explorer can be used
before requesting another dataset. New data replaces the current view after a
successful fetch and validation.

## From Yahoo Finance to implied volatility

The backend uses **yfinance** to retrieve option chains, the underlying price and
market inputs. Each dataset passes through four stages:

1. **Select contracts.** Filter available expirations, strikes and option types.
   Up to four workers retrieve option chains concurrently, each using its own
   ticker object. Results are processed in expiration order; failed chain
   requests are reported with their dates. The OTM selection combines puts
   below spot with calls at or above spot.
2. **Construct prices.** Use the bid/ask midpoint when both quotes are valid:
   $P_{\mathrm{market}}=(\mathrm{bid}+\mathrm{ask})/2$.
   Positive locked quotes, where bid equals ask, are accepted. When enabled,
   timestamped last trades at most four calendar days old can fill absent quotes.
3. **Invert the pricing model.** Find the volatility that reproduces each
   observed price under the selected exercise assumption.
4. **Calculate and visualize.** Compute Greeks, solve IV at bid/ask endpoints
   where available, and pass the results to the interactive charts.

## Pricing models

### Black–Scholes with continuous dividend yield

European call and put prices are calculated using:

$$
\begin{aligned}
C &= Se^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2), \\
P &= Ke^{-rT}\Phi(-d_2)-Se^{-qT}\Phi(-d_1),
\end{aligned}
$$

where

$$
d_1=\frac{\ln(S/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt{T}},
\qquad d_2=d_1-\sigma\sqrt{T}.
$$

$S$ is spot, $K$ is strike, $T$ is time in years, $r$ is the continuously
compounded interest-rate parameter, $q$ is continuous dividend yield and $\sigma$
is annualized volatility. $\Phi$ denotes the standard normal cumulative
distribution function.

The implementation uses NumPy broadcasting for batches of contracts and SciPy's
normal-distribution functions for pricing and sensitivities.

### Bjerksund–Stensland (1993)

American options can be exercised before expiry. The Bjerksund–Stensland model
approximates their value using an analytical exercise boundary and incorporates
the possibility of early exercise. American puts are evaluated through the
model's put–call symmetry transformation.

The implementation uses logarithmic evaluation and stable exponential functions
to reduce numerical overflow and cancellation. Unsupported exercise boundaries
are reported as unavailable, with their own diagnostic count. The application
prices supported cases directly with the analytical approximation.

## How implied volatility is calculated

Implied volatility is obtained by solving an inverse pricing problem:

$$
f(\sigma)=V_{\mathrm{model}}(S,K,T,r,q,\sigma)-P_{\mathrm{market}}=0.
$$

The numerical solver combines:

- **A Brenner–Subrahmanyam-style initial estimate** based on option price, spot
  and time to expiry.
- **Damped, vega-based iteration** for fast refinement across a batch. A
  Black–Scholes vega estimate supplies the local step sensitivity; damping and
  capped updates control the step size.
- **Brent's method** for contracts that remain unresolved. An expanding bracket
  searches for a root without requiring a derivative, helping with low-vega
  options where the initial iteration can stall.
- **Price bounds and model-domain checks** to reject invalid or unsupported
  observations before accepting an IV.

The search covers annualized IV from **0.1% to 500%**, restricted further by the
American approximation's supported domain. Every returned IV must reproduce the
observed price within an absolute model-price tolerance of **0.0001**.

IV therefore describes the volatility implied by an observed option price under
the selected model. It is a pricing measure rather than a direct forecast of
future realized volatility.

## Greeks and spread sensitivity

The engine computes five sensitivities for each resolved contract:

| Greek | Interpretation | Reported units |
| --- | --- | --- |
| Delta | First-order sensitivity to spot | Price change per unit of spot |
| Gamma | Sensitivity of delta to spot | Delta change per unit of spot |
| Vega | Sensitivity to volatility | Price change per 1 percentage point of IV |
| Theta | Sensitivity to the passage of time | Price change per calendar day |
| Rho | Sensitivity to the interest rate | Price change per 1 percentage point of rate |

Black–Scholes Greeks use analytical formulas. American-model Greeks use central
finite differences, repricing the option after small changes in spot,
volatility, time or the interest rate.

For midpoint observations, the engine also solves IV independently at the bid
and ask. Complete endpoint pairs appear as asymmetric bars on the volatility
smile. These ranges show **sensitivity to the quoted spread**, rather than a
statistical confidence interval. Last-trade estimates have no bid/ask IV range.

## Building the displayed surface

Observed IVs are linearly interpolated across strike within each expiration.
Plotly connects the resulting maturity slices into a 3D surface. Each expiry
retains its own strike coverage, with gaps outside that coverage left empty.
The original observations can be overlaid for comparison.

This is a visualization of model-implied quotes, not an arbitrage-free surface
calibration. Interpolation does not enforce strike convexity or calendar
consistency.

## Market assumptions and data quality

- **Rates:** the Yahoo Finance `^TNX` 10-year Treasury yield is used as a single
  rate proxy across maturities, rather than a maturity-matched discount curve.
- **Dividends:** the engine prefers annual forward cash dividend divided by spot,
  then trailing dividend yield. Missing data is explicitly marked as an assumed
  zero. Both models approximate dividends with a continuous yield.
- **Time:** maturities use ACT/365. US equity expiration is assumed to occur at
  16:00 in New York; AM-settled index contracts require different handling.
- **Price sources:** green points represent midpoints; amber points represent
  last-trade estimates. Last trades can be stale relative to spot, and fetching
  a dataset does not synchronize its quotes.
- **Diagnostics:** received contracts are accounted for through selection,
  quote rejection, invalid or unidentifiable prices, unsupported model domains,
  unresolved IVs and successful solutions.

These assumptions remain visible alongside the results so that surface shape
can be interpreted together with the quality of its inputs.

## Technology and validation

| Layer | Technologies |
| --- | --- |
| Numerical engine | Python, NumPy, SciPy |
| Market-data pipeline | yfinance, pandas |
| API and data contracts | FastAPI, Pydantic |
| Interactive dashboard | Next.js, React, TypeScript, Plotly.js |
| Hosting | Vercel |

Regression tests cover pricing benchmarks, put–call parity, Greek sensitivities,
low-vega IV recovery, exercise-boundary failures and quote filtering. API and
frontend tests verify serialization, diagnostic counts, interpolation and data
validation.

The numerical implementation lives in [`api/engine.py`](api/engine.py), the
market-data pipeline in [`api/data.py`](api/data.py), and chart construction in
[`frontend/lib/surface.ts`](frontend/lib/surface.ts).

The API accepts `max_dte_days` from 1 to 365 (default 365) and requires
`min_dte_days <= max_dte_days`. For example,
`/api/surface?symbol=AAPL&max_dte_days=365` requests every available expiry from
two days away through one year. An optional `max_expirations` count cap remains
available for older clients; the dashboard does not send it. Saved snapshots
retain the dates captured when they were generated; changing the horizon takes
effect when fetching a new dataset.
