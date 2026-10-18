import numpy as np
import pytest
from api.engine import (
    black_scholes_price,
    black_scholes_vega,
    bjerksund_stensland_price,
    calculate_all_greeks,
    solve_iv_robust,
)


def test_black_scholes_benchmark():
    """
    Test Black-Scholes model against Hull's benchmark values for European options.
    """
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    q = 0.0

    call_price = black_scholes_price(S, K, T, r, sigma, q, otype=1.0)
    put_price = black_scholes_price(S, K, T, r, sigma, q, otype=-1.0)

    assert np.isclose(call_price, 10.4506, atol=1e-4)
    assert np.isclose(put_price, 5.5735, atol=1e-4)

def test_put_call_parity_vectorized():
    """
    Test put-call parity for a range of option parameters and working vectorization.
    """
    np.random.seed(42)
    N=100
    S = np.random.uniform(50, 150, N)
    K = np.random.uniform(50, 150, N)
    T = np.random.uniform(0.1, 2.0, N)
    r = np.random.uniform(0.01, 0.1, N)
    sigma = np.random.uniform(0.1, 0.5, N)
    q = np.random.uniform(0.0, 0.05, N)

    calls = black_scholes_price(S, K, T, r, sigma, q, otype=1.0)
    puts = black_scholes_price(S, K, T, r, sigma, q, otype=-1.0)

    lhs = calls - puts
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

    assert np.allclose(lhs, rhs, atol=1e-4)

def test_black_scholes_vega_analytical_vs_numerical():
    """
    Test Black-Scholes vega against numerical approximation using finite differences.
    """
    S = 100.0
    K = 105.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    q = 0.1

    eps = 1e-5
    vega_analytical = black_scholes_vega(S, K, T, r, sigma, q)
    vega_numerical = (black_scholes_price(S, K, T, r, sigma + eps, q) - black_scholes_price(S, K, T, r, sigma - eps, q)) / (2 * eps)

    assert np.isclose(vega_analytical, vega_numerical, atol=1e-4)

def test_bjerksund_stensland_call_vs_black_scholes():
    """
    Test Bjerksund-Stensland model against Black-Scholes for European call options.
    """
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    q = 0.0

    bs_call_price = black_scholes_price(S, K, T, r, sigma, q, otype=1.0)    
    bjerksund_stensland_call_price = bjerksund_stensland_price(S, K, T, r, sigma, q=q, otype=1.0)

    assert np.isclose(bs_call_price, bjerksund_stensland_call_price, atol=1e-4)

def test_bjerksund_stensland_american_greater_than_european():
    """
    Test that Bjerksund-Stensland American call price is greater than or equal to Black-Scholes European call price.
    """
    np.random.seed(42)
    N = 100
    S = np.random.uniform(50, 150, N)
    K = np.random.uniform(50, 150, N)
    T = np.random.uniform(0.1, 2.0, N)
    r = np.random.uniform(0.01, 0.1, N)
    sigma = np.random.uniform(0.1, 0.5, N)
    q = np.random.uniform(0.0, 0.05, N)

    bs_call_prices = black_scholes_price(S, K, T, r, sigma, q, otype=1.0)
    bjerksund_stensland_call_prices = bjerksund_stensland_price(S, K, T, r, sigma, q=q, otype=1.0)

    bs_put_prices = black_scholes_price(S, K, T, r, sigma, q, otype=-1.0)
    bjerksund_stensland_put_prices = bjerksund_stensland_price(S, K, T, r, sigma, q=q, otype=-1.0)

    assert np.all(bjerksund_stensland_call_prices >= bs_call_prices - 1e-4)
    assert np.all(bjerksund_stensland_put_prices >= bs_put_prices - 1e-4)


def test_bjerksund_stensland_deep_itm_put():
    """
    Test Bjerksund-Stensland model for deep in-the-money put options.
    """
    S = 50.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    q = 0.0

    intrinsic = K - S  # 40.0
    p_amer = bjerksund_stensland_price(S, K, T, r, sigma, q=q, otype=-1.0)
    p_euro = black_scholes_price(S, K, T, r, sigma, q=q, otype=-1.0)

    assert p_euro < intrinsic

    assert p_amer >= intrinsic
    assert np.isclose(p_amer, intrinsic, atol=1e-4)

def test_calculate_all_greeks():
    """
    Test analytical Greeks (Delta, Gamma, Vega, Theta, Rho) against theoretical properties and numerical finite differences.
    """
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.02

    g_call = calculate_all_greeks(S, K, T, r, sigma, q=q, otype=1.0)
    g_put = calculate_all_greeks(S, K, T, r, sigma, q=q, otype=-1.0)

    # 1. Własności Delty
    assert 0.0 < g_call["delta"] < 1.0
    assert -1.0 < g_put["delta"] < 0.0
    # Put-Call Delta Parity: Delta_Call - Delta_Put == exp(-q * T)
    assert np.isclose(g_call["delta"] - g_put["delta"], np.exp(-q * T), atol=1e-5)

    # 2. Gamma i Vega są dodatnie i identyczne dla Call i Put
    assert g_call["gamma"] > 0
    assert np.isclose(g_call["gamma"], g_put["gamma"], atol=1e-6)
    assert g_call["vega"] > 0
    assert np.isclose(g_call["vega"], g_put["vega"], atol=1e-6)

    # 3. Theta jest ujemna (upływ czasu pomniejsza wartość opcji)
    assert g_call["theta"] < 0
    assert g_put["theta"] < 0

    # 4. Sprawdzenie analitycznego Rho numeryczną pochodną dPrice / dr (na 1% zmianystopy: / 100)
    eps_r = 1e-4
    p_up = black_scholes_price(S, K, T, r + eps_r, sigma, q=q, otype=1.0)
    p_down = black_scholes_price(S, K, T, r - eps_r, sigma, q=q, otype=1.0)
    rho_numerical = ((p_up - p_down) / (2.0 * eps_r)) / 100.0

    assert np.isclose(g_call["rho"], rho_numerical, atol=1e-4)


def test_solve_iv_black_scholes_vectorized():
    """
    Test vectorized IV recovery across an array of 50 option contracts with known volatilities.
    """
    np.random.seed(42)
    N = 50
    S = np.full(N, 100.0)
    K = np.random.uniform(85.0, 115.0, N)
    T = np.random.uniform(0.2, 1.5, N)
    r = np.full(N, 0.04)
    q = np.full(N, 0.01)
    true_sigmas = np.random.uniform(0.15, 0.50, N)
    otypes = np.random.choice([1.0, -1.0], size=N)

    prices = black_scholes_price(S, K, T, r, true_sigmas, q=q, otype=otypes)
    recovered_sigmas = solve_iv_robust(S, K, T, r, prices, otypes, q=q, model="black_scholes")

    assert np.allclose(recovered_sigmas, true_sigmas, atol=1e-3)


def test_solve_iv_american_bjerksund_stensland():
    """
    Test IV recovery for American Put and Call options using Bjerksund-Stensland model.
    """
    S, K, T, r, q = 100.0, 95.0, 0.5, 0.05, 0.02
    true_sigma_put = 0.22
    true_sigma_call = 0.28

    p_mkt = bjerksund_stensland_price(S, K, T, r, true_sigma_put, q=q, otype=-1.0)
    c_mkt = bjerksund_stensland_price(S, K, T, r, true_sigma_call, q=q, otype=1.0)

    rec_put = solve_iv_robust(S, K, T, r, p_mkt, -1.0, q=q, model="bjerksund_stensland")
    rec_call = solve_iv_robust(S, K, T, r, c_mkt, 1.0, q=q, model="bjerksund_stensland")

    assert np.isclose(rec_put, true_sigma_put, atol=1e-3)
    assert np.isclose(rec_call, true_sigma_call, atol=1e-3)


def test_solve_iv_arbitrage_protection_and_low_vega():
    """
    Verify that:
    1. Low-Vega OTM contract converges smoothly without exploding.
    2. Prices violating no-arbitrage bounds (below intrinsic) safely return NaN.
    """
    # 1. Low-Vega realistic OTM call (K/S = 1.15, T = 0.1)
    S, K, T, r, q, true_sig = 100.0, 115.0, 0.1, 0.05, 0.0, 0.30
    mkt = black_scholes_price(S, K, T, r, true_sig, q=q, otype=1.0)
    rec = solve_iv_robust(S, K, T, r, mkt, 1.0, q=q)
    assert np.isclose(rec, true_sig, atol=1e-3)

    # 2. Arbitrage violation: Call price of 0.01 for S=100, K=70 (intrinsic is ~30$)
    bad_price = 0.01
    rec_bad = solve_iv_robust(100.0, 70.0, 1.0, 0.05, bad_price, 1.0)
    assert np.isnan(rec_bad)