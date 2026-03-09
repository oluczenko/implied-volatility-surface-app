import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.scipy.stats import norm as jnorm
import numpy as np

# --- JAX MATH LOGIC ---

@jit
def _calc_d1_d2(S, K, T, r, sigma, q):
    """Pomocnicza funkcja wyliczająca parametry d1 i d2."""
    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return d1, d2

@jit
def black_scholes_price(S, K, T, r, sigma, q=0.0, otype=1.0):
    d1, d2 = _calc_d1_d2(S, K, T, r, sigma, q)
    price = (otype * S * jnp.exp(-q * T) * jnorm.cdf(otype * d1) - 
             otype * K * jnp.exp(-r * T) * jnorm.cdf(otype * d2))
    return price

vega_fn = jit(grad(black_scholes_price, argnums=4))
vectorized_price = vmap(black_scholes_price, in_axes=(0, 0, 0, 0, 0, None, 0))
vectorized_vega = vmap(vega_fn, in_axes=(0, 0, 0, 0, 0, None, 0))

@jit
def solve_iv_vectorized(S, K, T, r, market_price, otypes, q=0.0, n_iter=20):
    sigma = jnp.full_like(S, 0.5)
    for _ in range(n_iter):
        p = vectorized_price(S, K, T, r, sigma, q, otypes)
        v = vectorized_vega(S, K, T, r, sigma, q, otypes)
        diff = p - market_price
        v_safe = jnp.where(jnp.abs(v) < 1e-7, 1e-7, v)
        sigma = jnp.clip(sigma - diff / v_safe, 1e-4, 5.0)
    return jnp.clip(sigma, 0.001, 5.0)

@jit
def calculate_greeks(S, K, T, r, sigma, q=0.0, otype=1.0):
    d1, d2 = _calc_d1_d2(S, K, T, r, sigma, q)
    
    # Delta
    delta = otype * jnp.exp(-q * T) * jnorm.cdf(otype * d1)
    
    # Gamma
    gamma = jnp.exp(-q * T) * jnorm.pdf(d1) / (S * sigma * jnp.sqrt(T))
    
    # Vega (usually divided by 100 to get per 1% change)
    vega = S * jnp.exp(-q * T) * jnorm.pdf(d1) * jnp.sqrt(T) / 100.0
    
    # Theta (per day)
    term1 = -(S * sigma * jnp.exp(-q * T) * jnorm.pdf(d1)) / (2 * jnp.sqrt(T))
    term2 = otype * r * K * jnp.exp(-r * T) * jnorm.cdf(otype * d2)
    term3 = otype * q * S * jnp.exp(-q * T) * jnorm.cdf(otype * d1)
    theta = (term1 - term2 + term3) / 365.0
    
    return delta, gamma, vega, theta

vectorized_greeks = vmap(calculate_greeks, in_axes=(0, 0, 0, 0, 0, None, 0))
