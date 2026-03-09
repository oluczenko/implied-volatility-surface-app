import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax_core import solve_iv_vectorized, vectorized_greeks
import streamlit as st

@st.cache_data(ttl=3600)
def get_ticker_metrics(symbol):
    """Fetches real-time Risk Free Rate (^TNX) and Dividend Yield."""
    try:
        tnx = yf.Ticker("^TNX")
        # Wyciągamy rentowność (%) bezpośrednio na wykres bez dzielenia przez 100
        rf_rate_pct = float(tnx.fast_info.last_price)
    except Exception:
        rf_rate_pct = 4.2 # fallback w procentach
        
    try:
        ticker = yf.Ticker(symbol)
        # Yahoo Finance posiada dwa pola:
        # dividendRate (np. 1.00 USD) oraz dividendYield (np. 0.0044 czyli 0.44%)
        div_yield_decimal = ticker.info.get('dividendYield', 0.0)
        
        if div_yield_decimal is None:
             div_yield_pct = 0.0
        else:
             # API Yahoo domyślnie rzuca gotowy wskaźnik jako % (np. ARCC = 10.25, AAPL = 0.4)
             div_yield_pct = float(div_yield_decimal)
             
    except Exception:
        div_yield_pct = 0.0
        
    return rf_rate_pct, div_yield_pct

def fetch_and_calculate_iv(symbol, option_type, period, risk_free_rate_pct, div_yield_pct, min_strike_pct, max_strike_pct):
    # --- Convert inputs BACK to decimals for JAX Math ---
    risk_free_rate = risk_free_rate_pct / 100.0
    div_yield = div_yield_pct / 100.0
    
    try:
        ticker = yf.Ticker(symbol)
        spot_price = ticker.fast_info.last_price
    except Exception as e:
        return None, None, f"Failed to fetch underlying price for {symbol}: {e}"

    exp_dates = ticker.options
    if not exp_dates:
         return None, None, f"No options available for symbol {symbol}."
         
    if period is not None and period > 0:
        exp_dates = exp_dates[:period]

    all_data = []
    now = dt.datetime.now()
    
    # Calculate strike boundaries based on percentage of spot
    min_strike = spot_price * (min_strike_pct / 100.0)
    max_strike = spot_price * (max_strike_pct / 100.0)

    for d in exp_dates:
        try:
            chain = ticker.option_chain(d)
        except Exception:
             continue
        df = chain.calls if option_type == 'call' else chain.puts
        
        # Filter strictly by limits computed from sliders
        df = df[(df['strike'] >= min_strike) & (df['strike'] <= max_strike)]
        
        expiry_dt = dt.datetime.strptime(d, '%Y-%m-%d')
        T_val = (expiry_dt - now).days / 365.0
        
        # INCREASED THRESHOLD: Remove noise from 0DTE options (approx < 25 days)
        if T_val <= 0.07:  
            continue
            
        df['T'] = T_val
        df['otype_val'] = 1.0 if option_type == 'call' else -1.0
        
        # BETTER LIQUIDITY CHECK: use lastPrice if bid/ask is missing 
        bid = df['bid']
        ask = df['ask']
        mid = 0.5 * (bid + ask)
        df['midPrice'] = np.where((bid > 0) & (ask > 0), mid, df['lastPrice'])
        
        df = df[df['midPrice'] > 0.0]
        all_data.append(df[['midPrice', 'strike', 'T', 'otype_val']])

    if not all_data:
        return None, None, f"No valid options for {symbol} after applying strict Expiration & Strike filters."

    df = pd.concat(all_data, ignore_index=True)
    
    N = len(df)
    S_arr = jnp.full(N, spot_price, dtype=jnp.float32)
    r_arr = jnp.full(N, risk_free_rate, dtype=jnp.float32)
    K_arr = jnp.array(df['strike'].values, dtype=jnp.float32)
    T_arr = jnp.array(df['T'].values, dtype=jnp.float32)
    P_arr = jnp.array(df['midPrice'].values, dtype=jnp.float32)
    O_arr = jnp.array(df['otype_val'].values, dtype=jnp.float32)
    
    ivs_jax = solve_iv_vectorized(S_arr, K_arr, T_arr, r_arr, P_arr, O_arr, q=div_yield)
    delta_jax, gamma_jax, vega_jax, theta_jax = vectorized_greeks(S_arr, K_arr, T_arr, r_arr, ivs_jax, div_yield, O_arr)
    
    ivs = np.array(ivs_jax)
    valid = (ivs > 0.001) & (ivs < 4.99)
    
    surface_data = pd.DataFrame({
        "ivs": ivs[valid] * 100, # Display in %
        "strike": np.array(K_arr)[valid],
        "dtes": np.array(T_arr)[valid],
        "delta": np.array(delta_jax)[valid],
        "gamma": np.array(gamma_jax)[valid],
        "vega": np.array(vega_jax)[valid],
        "theta": np.array(theta_jax)[valid],
        "price": np.array(P_arr)[valid]
    })
    
    return surface_data, spot_price, None
