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

# --- PAGE CONFIG ---
st.set_page_config(page_title="Implied Volatility Surface", layout="wide", page_icon="📈")

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
def fetch_and_calculate_iv(symbol, option_type, period, risk_free_rate, div_yield):
    try:
        ticker = yf.Ticker(symbol)
        spot_price = ticker.fast_info.last_price
    except Exception as e:
        st.error(f"Failed to fetch underlying price for {symbol}: {e}")
        return None

    exp_dates = ticker.options
    if not exp_dates:
         st.error(f"No options available for symbol {symbol}.")
         return None
         
    if period is not None and period > 0:
        exp_dates = exp_dates[:period]

    all_data = []
    now = dt.datetime.now()

    for d in exp_dates:
        try:
            chain = ticker.option_chain(d)
        except Exception:
             continue
        df = chain.calls if option_type == 'call' else chain.puts
        
        expiry_dt = dt.datetime.strptime(d, '%Y-%m-%d')
        T_val = (expiry_dt - now).days / 365.0
        
        if T_val <= 0.001: 
            continue
            
        df['T'] = T_val
        df['otype_val'] = 1.0 if option_type == 'call' else -1.0
        df['midPrice'] = (df['bid'] + df['ask']) / 2.0
        df = df[df['midPrice'] > 0.0]
        
        all_data.append(df[['midPrice', 'strike', 'T', 'otype_val']])

    if not all_data:
        st.warning(f"No valid options for {symbol} in the selected window.")
        return None

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
    moneyness = spot_price / np.array(K_arr)
    dtes = np.array(T_arr)
    
    valid = (ivs > 0.001) & (ivs < 4.99)
    
    surface_data = pd.DataFrame({
        "ivs": ivs[valid],
        "moneyness": moneyness[valid],
        "dtes": dtes[valid]
    })
    
    return surface_data

# --- UI LOGIC ---

st.title("Implied Volatility (IV) Surface")
st.markdown("Application for calculating IV quickly utilizing `JAX` vectorization and `yfinance` market data.")

with st.sidebar:
    st.header("Configuration")
    symbol = st.text_input("Ticker Symbol", value="SPY").upper()
    option_type = st.selectbox("Option Type", ["call", "put"]).lower()
    period = st.slider("Max expirations to fetch", min_value=1, max_value=20, value=6)
    risk_free_rate = st.number_input("Risk Free Rate", value=0.04, format="%.4f")
    div_yield = st.number_input("Dividend Yield", value=0.0, format="%.4f")
    
    generate_btn = st.button("Generate Graph", type="primary")

if generate_btn:
    with st.spinner("Fetching market data and calculating IV..."):
         start_time = time.time()
         surface_data = fetch_and_calculate_iv(symbol, option_type, period, risk_free_rate, div_yield)
         elapsed = time.time() - start_time
         
         if surface_data is not None and not surface_data.empty:
             st.success(f"Calculated {len(surface_data)} option contracts in {elapsed:.2f} s")
             
             # Create Plotly 3D Surface/Scatter Chart
             fig = go.Figure(data=[go.Mesh3d(
                 x=surface_data['moneyness'],
                 y=surface_data['dtes'],
                 z=surface_data['ivs'],
                 opacity=0.8,
                 colorscale='Viridis',
                 intensity=surface_data['ivs'],
             )])

             fig.update_layout(
                 title=f"{symbol} Implied Volatility Surface ({option_type.capitalize()})",
                 scene=dict(
                     xaxis_title='Moneyness (S/K)',
                     yaxis_title='Time to Expiration (Years)',
                     zaxis_title='Implied Volatility (IV)',
                     camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))
                 ),
                 height=800,
                 margin=dict(l=0, r=0, b=0, t=50)
             )
             st.plotly_chart(fig, use_container_width=True)
         else:
             st.warning("Failed to generate the plot due to missing data.")
else:
    st.info("👈 Enter configuration in the sidebar and click 'Generate Graph' to view the Options model.")
