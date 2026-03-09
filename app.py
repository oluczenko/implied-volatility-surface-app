import streamlit as st
import numpy as np
import yfinance as yf
import datetime as dt
import pandas as pd
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.scipy.stats import norm as jnorm
import plotly.graph_objects as go
import time
from scipy.interpolate import griddata

# --- PAGE CONFIG ---
st.set_page_config(page_title="Implied Volatility Surface v2", layout="wide", page_icon="📈")

# --- JAX MATH LOGIC ---
@jit
def black_scholes_price(S, K, T, r, sigma, q=0.0, otype=1.0):
    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
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

# --- CACHED DATA FETCHING AND CALCULATION ---

@st.cache_data(ttl=300)
def fetch_and_calculate_iv(symbol, option_type, period, risk_free_rate, div_yield, min_strike_pct, max_strike_pct):
    try:
        ticker = yf.Ticker(symbol)
        spot_price = ticker.fast_info.last_price
    except Exception as e:
        st.error(f"Failed to fetch underlying price for {symbol}: {e}")
        return None, None

    exp_dates = ticker.options
    if not exp_dates:
         st.error(f"No options available for symbol {symbol}.")
         return None, None
         
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
        st.warning(f"No valid options for {symbol} after applying strict Expiration & Strike filters.")
        return None, None

    df = pd.concat(all_data, ignore_index=True)
    
    N = len(df)
    S_arr = jnp.full(N, spot_price, dtype=jnp.float32)
    r_arr = jnp.full(N, risk_free_rate, dtype=jnp.float32)
    K_arr = jnp.array(df['strike'].values, dtype=jnp.float32)
    T_arr = jnp.array(df['T'].values, dtype=jnp.float32)
    P_arr = jnp.array(df['midPrice'].values, dtype=jnp.float32)
    O_arr = jnp.array(df['otype_val'].values, dtype=jnp.float32)
    
    ivs_jax = solve_iv_vectorized(S_arr, K_arr, T_arr, r_arr, P_arr, O_arr, q=div_yield)
    
    ivs = np.array(ivs_jax)
    valid = (ivs > 0.001) & (ivs < 4.99)
    
    surface_data = pd.DataFrame({
        "ivs": ivs[valid] * 100, # Display in %
        "strike": np.array(K_arr)[valid],
        "dtes": np.array(T_arr)[valid]
    })
    
    return surface_data, spot_price

# --- UI LOGIC ---

st.title("Implied Volatility (IV) Surface - v2 Smooth")
st.markdown("Application combining `JAX` vectorization speed with elegant, interpolated Grid Surface.")

with st.sidebar:
    st.header("Configuration")
    symbol = st.text_input("Ticker Symbol", value="SPY").upper()
    option_type = st.selectbox("Option Type", ["call", "put"]).lower()
    
    y_axis_type = st.selectbox("Y-Axis metric", ["Strike Price", "Log-Moneyness"])
    
    strike_range = st.slider(
        "Strike Range (% of Spot)",
        min_value=20, max_value=200, value=(70, 130), step=5
    )
    
    period = st.slider("Max Expirations to fetch", min_value=1, max_value=25, value=12)
    risk_free_rate = st.number_input("Risk Free Rate", value=0.04, format="%.4f")
    div_yield = st.number_input("Dividend Yield", value=0.0, format="%.4f")
    
    generate_btn = st.button("Generate Smooth Graph", type="primary")

if generate_btn:
    with st.spinner("Fetching data and computing IV using JAX..."):
         start_time = time.time()
         surface_data, spot_price = fetch_and_calculate_iv(
             symbol, option_type, period, risk_free_rate, div_yield, strike_range[0], strike_range[1]
         )
         elapsed = time.time() - start_time
         
         if surface_data is not None and not surface_data.empty:
             st.success(f"Calculated {len(surface_data)} valid IV points in {elapsed:.2f} s")
             
             X = surface_data['dtes'].values
             Z = surface_data['ivs'].values
             
             if y_axis_type == "Log-Moneyness":
                 T = surface_data['dtes'].values
                 F = spot_price * np.exp((risk_free_rate - div_yield) * T)
                 F = np.maximum(F, 1e-12)
                 Y = np.log(surface_data["strike"].values / F)
                 y_label = "Log-moneyness ln(K/F)"
             else:
                 Y = surface_data['strike'].values
                 y_label = "Strike Price ($)"
                 
             if len(np.unique(X)) < 2 or len(np.unique(Y)) < 2:
                 st.error("Not enough variation in data to interpolate a surface. Try expanding the data range.")
                 st.stop()
                 
             # --- THE MAGICAL SMOOTH SURFACE TRICK ---
             xi = np.linspace(X.min(), X.max(), 30)
             yi = np.linspace(Y.min(), Y.max(), 30)
             xi, yi = np.meshgrid(xi, yi)

             zi = griddata((X, Y), Z, (xi, yi), method="linear")
             zi2 = griddata((X, Y), Z, (xi, yi), method="nearest")
             zi = np.where(np.isnan(zi), zi2, zi)
             
             fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=zi, colorscale="Viridis")])

             fig.update_layout(
                 title=f"{symbol} Volatility Surface ({option_type.capitalize()}) - Scipy Interpolated",
                 scene=dict(
                     xaxis_title='Time to Expiration (Years)',
                     yaxis_title=y_label,
                     zaxis_title='Implied Volatility (%)',
                     camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))
                 ),
                 height=800,
                 margin=dict(l=0, r=0, b=0, t=50)
             )
             st.plotly_chart(fig, use_container_width=True)
         else:
             st.warning("Failed to generate chart due to missing or strictly filtered data.")
else:
    st.info("👈 Enter parameters in the sidebar and click 'Generate Smooth Graph'.")
