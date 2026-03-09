import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from scipy.interpolate import griddata

from data_fetcher import fetch_and_calculate_iv

# --- PAGE CONFIG ---
st.set_page_config(page_title="Implied Volatility Surface v3 Modular", layout="wide", page_icon="📈")

# --- UI LOGIC ---

st.title("Implied Volatility (IV) Surface - v3 Modular")
st.markdown("Application combining `JAX` vectorization speed with elegant, interpolated Grid Surface and Modular Design.")

# Streamlit Caching wrapper over external logic
@st.cache_data(ttl=300)
def cached_data_fetch(symbol, option_type, period, risk_free_rate, div_yield, min_strike_pct, max_strike_pct):
    return fetch_and_calculate_iv(symbol, option_type, period, risk_free_rate, div_yield, min_strike_pct, max_strike_pct)

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
         
         surface_data, spot_price, err_msg = cached_data_fetch(
             symbol, option_type, period, risk_free_rate, div_yield, strike_range[0], strike_range[1]
         )
         elapsed = time.time() - start_time
         
         if err_msg:
             st.error(err_msg)
         elif surface_data is not None and not surface_data.empty:
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
