import numpy as np
import pytest
from api.engine import black_scholes_price, black_scholes_vega, bjerksund_stensland_price


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