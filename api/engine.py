import numpy as np
from scipy.special import ndtr, log_ndtr
from scipy.optimize import brentq

def calc_d1_d2(S,K,T,r,sigma,q= 0.0):
    """
    calculate d1 and d2 for Black-Scholes formula
    """
    S= np.asarray(S, dtype=np.float64)
    K= np.asarray(K, dtype=np.float64)
    T= np.asarray(T, dtype=np.float64)
    r= np.asarray(r, dtype=np.float64)
    sigma= np.asarray(sigma, dtype=np.float64)
    q= np.asarray(q, dtype=np.float64)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return d1, d2

def black_scholes_price(S,K,T,r,sigma,q=0.0,otype=1.0):
    """
    calculate Black-Scholes price for European options
    otype: 1.0 for call, -1.0 for put
    """
    d1, d2 = calc_d1_d2(S,K,T,r,sigma,q)
    
    df_q = np.exp(-q * T)
    df_r = np.exp(-r * T)
    otype = np.asarray(otype, dtype=np.float64)
    return otype * (S * df_q * ndtr(otype * d1) - K * df_r * ndtr(otype * d2))


def black_scholes_vega(S,K,T,r,sigma,q=0.0):
    """
    calculate Black-Scholes vega for European options
    """
    d1, _ = calc_d1_d2(S,K,T,r,sigma,q)
    
    T = np.maximum(np.asarray(T, dtype=np.float64), 1e-6)
    pdf_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    
    return np.asarray(S, dtype=np.float64) * np.exp(-q * T) * np.sqrt(T) * pdf_d1

def _bjerksund_stensland_call_scalar(S, K, T, r, q, sigma):
    """Pure 1993 approximation; NaN means its exercise boundary is unsupported."""
    if r < 0 or q < 0:
        return np.nan
    b = r - q
    euro = float(black_scholes_price(S, K, T, r, sigma, q))
    lower = max(euro, S - K, 0.)
    # No early exercise for a non-dividend call at nonnegative interest rates.
    if q <= 0 and r >= 0:
        return lower

    # Negative rates/yields and a collapsed boundary are outside this
    # implementation's supported domain. Do not invent an exercise price.
    if b * T + 2 * sigma * np.sqrt(T) <= 0:
        return np.nan

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        sigma2 = sigma ** 2
        beta = (.5 - b / sigma2) + np.sqrt((b / sigma2 - .5) ** 2 + 2 * r / sigma2)
        B_0 = max(K, r / q * K)
        # Stable limit as beta approaches 1 (the perpetual boundary diverges).
        a = (b * T + 2 * sigma * np.sqrt(T)) * B_0
        if abs(beta - 1) < 1e-8:
            I = B_0 + a
        else:
            gap = beta / (beta - 1) * K - B_0
            I = B_0 + gap * (-np.expm1(-a / gap))
        if not np.isfinite(I) or I <= K:
            return np.nan
        if S >= I:
            approx = lower
        else:
            def phi(gamma, H):
                lam = -r + gamma * b + .5 * gamma * (gamma - 1) * sigma2
                kappa = 2 * b / sigma2 + 2 * gamma - 1
                d = -(np.log(S / H) + (b + (gamma - .5) * sigma2) * T) / (sigma * np.sqrt(T))
                reflected = d - 2 * np.log(I / S) / (sigma * np.sqrt(T))
                # Evaluate the image term in log space: avoid inf * 0.
                image = np.exp(kappa * np.log(I / S) + log_ndtr(reflected))
                return np.exp(lam * T) * (ndtr(d) - image)

            approx = ((I - K) * np.exp(beta * np.log(S / I)) * (1 - phi(beta, I))
                      + S * (phi(1., I) - phi(1., K))
                      - K * (phi(0., I) - phi(0., K)))
        if not np.isfinite(approx):
            return np.nan
        return max(lower, approx)

def bjerksund_stensland_price(S, K, T, r, sigma, q=0.0, otype=1.0):
    """
    Calculate the pure Bjerksund-Stensland (1993) approximation.
    Unsupported/nonfinite exercise boundaries return NaN; there is no alternate
    pricing model. otype: 1.0 for call, -1.0 for put
    """
    S_arr = np.atleast_1d(np.asarray(S, dtype=np.float64))
    K_arr = np.atleast_1d(np.asarray(K, dtype=np.float64))
    T_arr = np.atleast_1d(np.asarray(T, dtype=np.float64))
    r_arr = np.atleast_1d(np.asarray(r, dtype=np.float64))
    sigma_arr = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
    q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
    otype_arr = np.atleast_1d(np.asarray(otype, dtype=np.float64))

    b_S, b_K, b_T, b_r, b_sigma, b_q, b_otype = np.broadcast_arrays(
        S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr, otype_arr
    )

    prices = np.empty_like(b_S, dtype=np.float64)
    flat_S = b_S.ravel()
    flat_K = b_K.ravel()
    flat_T = b_T.ravel()
    flat_r = b_r.ravel()
    flat_sigma = b_sigma.ravel()
    flat_q = b_q.ravel()
    flat_otype = b_otype.ravel()
    flat_out = prices.ravel()

    for i in range(len(flat_S)):
        s_val, k_val = flat_S[i], flat_K[i]
        if (not np.isfinite([flat_S[i], flat_K[i], flat_T[i], flat_r[i], flat_sigma[i], flat_q[i], flat_otype[i]]).all()
                or s_val <= 0 or k_val <= 0 or flat_T[i] < 0 or flat_sigma[i] <= 0
                or flat_otype[i] not in (-1., 1.)):
            flat_out[i] = np.nan
            continue
        if flat_T[i] == 0:
            flat_out[i] = max(0., flat_otype[i] * (s_val - k_val))
            continue
        t_val = max(flat_T[i], 1e-5)
        r_val, q_val = flat_r[i], flat_q[i]
        sig_val = max(flat_sigma[i], 1e-5)

        if flat_otype[i] > 0:
            flat_out[i] = _bjerksund_stensland_call_scalar(s_val, k_val, t_val, r_val,q_val, sig_val)
        else:
            flat_out[i] = _bjerksund_stensland_call_scalar(k_val, s_val, t_val, q_val,r_val, sig_val)

    if all(np.ndim(x) == 0 for x in (S, K, T, r, sigma, q, otype)):
        return float(prices.item())
    return prices

def calculate_all_greeks(S, K, T, r, sigma, q=0.0, otype=1.0, model="black_scholes"):
    """
    Calculate Greeks for the selected model. American Greeks use central finite
    differences. Vega/rho are per percentage point; theta is per calendar day.
    """
    if model not in ("black_scholes", "bjerksund_stensland"):
        raise ValueError(f"Invalid model: '{model}'")
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.maximum(np.asarray(T, dtype=np.float64), 1e-6)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-6)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    otype = np.asarray(otype, dtype=np.float64)

    if model == "bjerksund_stensland":
        def price(s=S, t=T, vol=sigma, rate=r):
            return bjerksund_stensland_price(s, K, t, rate, vol, q=q, otype=otype)

        ds = np.maximum(np.abs(S) * 1e-4, 1e-6)
        dv = np.minimum(1e-4, sigma * 0.5)
        dt = np.minimum(1e-4, T * 0.5)
        dr = 1e-5
        base = price()
        up, down = price(s=S + ds), price(s=S - ds)
        return {
            "delta": (up - down) / (2 * ds),
            "gamma": (up - 2 * base + down) / ds**2,
            "vega": (price(vol=sigma + dv) - price(vol=sigma - dv)) / (2 * dv * 100),
            "theta": (price(t=T - dt) - price(t=T + dt)) / (2 * dt * 365),
            "rho": (price(rate=r + dr) - price(rate=r - dr)) / (2 * dr * 100),
        }

    d1, d2 = calc_d1_d2(S, K, T, r, sigma, q)

    df_q = np.exp(-q * T)
    df_r = np.exp(-r * T)
    sqrt_T = np.sqrt(T)
    pdf_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)

    # 1. Delta
    delta = otype * df_q * ndtr(otype * d1)

    # 2. Gamma 
    gamma = df_q * pdf_d1 / (S * sigma * sqrt_T)

    # 3. Vega 
    vega_pct = (S * df_q * pdf_d1 * sqrt_T) / 100.0

    # 4. Theta 
    term1 = -(S * sigma * df_q * pdf_d1) / (2.0 * sqrt_T)
    term2 = otype * r * K * df_r * ndtr(otype * d2)
    term3 = otype * q * S * df_q * ndtr(otype * d1)
    theta = (term1 - term2 + term3) / 365.0

    # 5. Rho
    rho = (otype * K * T * df_r * ndtr(otype * d2)) / 100.0

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega_pct,
        "theta": theta,
        "rho": rho,
    }


def iv_price_bounds(S, K, T, r, q=0., otypes=1., model="black_scholes"):
    """No-arbitrage bounds for the chosen exercise convention."""
    if model not in ("black_scholes", "bjerksund_stensland"):
        raise ValueError(f"Invalid model: '{model}'")
    S, K, T, r, q, otypes = np.broadcast_arrays(S, K, T, r, q, otypes)
    discounted_S, discounted_K = S * np.exp(-q * T), K * np.exp(-r * T)
    lower = np.maximum(otypes * (discounted_S - discounted_K), 0.)
    if model == "bjerksund_stensland":
        lower = np.maximum(lower, otypes * (S - K))
        upper = np.where(otypes > 0, S * np.maximum(1., np.exp(-q * T)),
                         K * np.maximum(1., np.exp(-r * T)))
    else:
        upper = np.where(otypes > 0, discounted_S, discounted_K)
    return lower, upper


def solve_iv_robust(
    S, K, T, r, market_price, otypes, q=0.0, model="black_scholes",
    max_iter=15, tol=1e-4, diagnostics=None,
):
    """Damped vector iteration with a bracketed Brent fallback on [0.001, 5].

    Every returned IV must reprice within tol. NaN denotes invalid inputs,
    unidentifiable time value, unsupported approximation domain, or no converged
    root inside the supported range. Optional diagnostics receives a boolean
    unsupported_model mask with the same shape as the returned IVs.
    max_iter=0 explicitly disables all refinement (including the fallback).
    """
    if model not in ("black_scholes", "bjerksund_stensland"):
        raise ValueError(f"Invalid model: '{model}'")
    if max_iter < 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("Invalid solver controls")
    S, K, T, r, P_mkt, otypes, q = np.broadcast_arrays(
        *[np.asarray(x, dtype=float) for x in (S, K, T, r, market_price, otypes, q)])
    shape = S.shape
    S, K, T, r, P_mkt, otypes, q = [x.ravel() for x in (S, K, T, r, P_mkt, otypes, q)]
    lower, upper = iv_price_bounds(S, K, T, r, q, otypes, model)
    valid = (np.isfinite(S + K + T + r + q + P_mkt) & (S > 0) & (K > 0) & (T > 0)
             & np.isin(otypes, [-1., 1.]) & (P_mkt > lower + tol) & (P_mkt < upper))
    result = np.full(S.shape, np.nan)
    unsupported = np.zeros(S.shape, dtype=bool)
    minimum_sigma = np.full(S.shape, .001)
    pricing = bjerksund_stensland_price if model == "bjerksund_stensland" else black_scholes_price
    if model == "bjerksund_stensland":
        # Put-call symmetry flips carry. A valid boundary requires sigma above this
        # floor; keep both fast iteration and Brent inside the same model domain.
        boundary_floor = .5 * np.maximum(-otypes * (r - q), 0.) * np.sqrt(np.maximum(T, 0.))
        minimum_sigma = np.maximum(.001, boundary_floor * (1 + 1e-6) + 1e-8)
        unsupported |= valid & ((r < 0) | (q < 0) | (minimum_sigma >= 5.))
        candidates = np.flatnonzero(valid & ~unsupported)
        if candidates.size:
            floor_prices = pricing(S[candidates], K[candidates], T[candidates], r[candidates],
                                   minimum_sigma[candidates], q[candidates], otypes[candidates])
            unsupported[candidates] = ~np.isfinite(floor_prices) | (P_mkt[candidates] < floor_prices - tol)
        valid &= ~unsupported

    def finish():
        if diagnostics is not None:
            diagnostics["unsupported_model"] = unsupported.reshape(shape).copy()
        return float(result.item()) if shape == () else result.reshape(shape)

    indices = np.flatnonzero(valid)
    if not indices.size:
        return finish()
    s, k, t, rate, p, ot, div = [x[indices] for x in (S, K, T, r, P_mkt, otypes, q)]
    min_sigma = minimum_sigma[indices]
    sigma = np.clip(np.sqrt(2 * np.pi / t) * p / s, np.maximum(.1, min_sigma), np.maximum(2.5, min_sigma))
    for _ in range(max_iter):
        diff = pricing(s, k, t, rate, sigma, div, ot) - p
        active = np.isfinite(diff) & (np.abs(diff) > tol)
        if not active.any():
            break
        vega = black_scholes_vega(s, k, t, rate, sigma, div)
        step = diff * vega / (vega ** 2 + 1e-6 + 1e-3 * np.abs(diff))
        # Leave already-converged contracts alone.
        sigma[active] = np.clip(sigma[active] - np.clip(step[active], -.4, .4), min_sigma[active], 5.)
    errors = np.abs(pricing(s, k, t, rate, sigma, div, ot) - p)
    unresolved = ~np.isfinite(errors) | (errors > tol)
    if max_iter > 0:
        for i in np.flatnonzero(unresolved):
            def residual(vol):
                return float(pricing(s[i], k[i], t[i], rate[i], vol, div[i], ot[i])) - p[i]
            # Never evaluate Brent below the approximation's valid sigma floor.
            lo, hi = max(min_sigma[i], sigma[i] / 2), min(5., max(.2, sigma[i] * 2))
            f_lo, f_hi = residual(lo), residual(hi)
            while f_lo > 0 and lo > min_sigma[i]:
                lo = max(min_sigma[i], lo / 2)
                f_lo = residual(lo)
            while f_hi < 0 and hi < 5.:
                hi = min(5., hi * 2)
                f_hi = residual(hi)
            try:
                sigma[i] = brentq(residual, lo, hi, xtol=1e-10, maxiter=100)
            except (ValueError, RuntimeError):
                sigma[i] = np.nan
    final_error = np.abs(pricing(s, k, t, rate, sigma, div, ot) - p)
    result[indices] = np.where(np.isfinite(final_error) & (final_error <= tol), sigma, np.nan)
    return finish()
