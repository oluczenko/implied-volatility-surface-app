import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from scipy.interpolate import griddata

from data_fetcher import fetch_and_calculate_iv, get_ticker_metrics

# --- PAGE CONFIG ---
st.set_page_config(page_title="Implied Volatility Surface v4 Modular (Greeks)", layout="wide", page_icon="📈")

# --- UI LOGIC ---

st.title("Implied Volatility (IV) Surface - v4 Greeks & Auto-Rates")
st.markdown("Application combining `JAX` vectorization speed with elegant, interpolated Grid Surface, Auto Dividend/Rate fetching, and **Interactive Greeks Toolkit**.")

# Streamlit Caching wrapper over external logic
@st.cache_data(ttl=300)
def cached_data_fetch(symbol, option_type, period, risk_free_rate, div_yield, min_strike_pct, max_strike_pct):
    return fetch_and_calculate_iv(symbol, option_type, period, risk_free_rate, div_yield, min_strike_pct, max_strike_pct)

with st.sidebar:
    st.header("Configuration")
    symbol = st.text_input("Ticker Symbol", value="SPY").upper()
    
    # Auto-fetch risk-free rate and dividend yield based on ticker
    auto_rf, auto_div = get_ticker_metrics(symbol)
    
    option_type = st.selectbox("Option Type", ["call", "put"]).lower()
    
    y_axis_type = st.selectbox("Y-Axis metric", ["Strike Price", "Log-Moneyness"])
    
    strike_range = st.slider(
        "Strike Range (% of Spot)",
        min_value=20, max_value=200, value=(70, 130), step=5
    )
    
    period = st.slider("Max Expirations to fetch", min_value=1, max_value=25, value=12)
    
    st.markdown("---")
    st.markdown("**(Auto-fetched parameters)**")
    risk_free_rate = st.number_input("Risk Free Rate (%)", value=float(auto_rf), format="%.4f")
    div_yield = st.number_input("Dividend Yield (%)", value=float(auto_div), format="%.4f")
    
    generate_btn = st.button("Generate Smooth Graph with Greeks", type="primary")

if generate_btn:
    with st.spinner("Fetching data and computing IV & Greeks using JAX..."):
         start_time = time.time()
         
         surface_data, spot_price, err_msg = cached_data_fetch(
             symbol, option_type, period, risk_free_rate, div_yield, strike_range[0], strike_range[1]
         )
         elapsed = time.time() - start_time
         
         if err_msg:
             st.error(err_msg)
         elif surface_data is not None and not surface_data.empty:
             st.session_state['surface_data'] = surface_data
             st.session_state['spot_price'] = spot_price
             st.session_state['calc_time'] = elapsed
             st.session_state['symbol'] = symbol
             st.session_state['option_type'] = option_type
         else:
             st.warning("Failed to generate chart due to missing or strictly filtered data.")

if 'surface_data' in st.session_state:
     surface_data = st.session_state['surface_data']
     spot_price = st.session_state['spot_price']
     elapsed = st.session_state['calc_time']
     calc_symbol = st.session_state['symbol']
     calc_otype = st.session_state['option_type']
     
     st.success(f"Calculated {len(surface_data)} valid IV points & Greeks in {elapsed:.2f} s")
     
     if y_axis_type == "Log-Moneyness":
         T = surface_data['dtes'].values
         F = spot_price * np.exp((risk_free_rate - div_yield) * T)
         F = np.maximum(F, 1e-12)
         surface_data['Y_metric'] = np.log(surface_data["strike"].values / F)
         y_label = "Log-moneyness ln(K/F)"
     else:
         surface_data['Y_metric'] = surface_data['strike'].values
         y_label = "Strike Price ($)"
         
     tab1, tab2 = st.tabs(["3D IV Surface", "2D Volatility Smile"])
     
     with tab1:
         X = surface_data['dtes'].values
         Y = surface_data['Y_metric'].values
         Z = surface_data['ivs'].values
         
         if len(np.unique(X)) < 2 or len(np.unique(Y)) < 2:
             st.error("Not enough variation in data to interpolate a surface. Try expanding the data range.")
         else:
             # --- THE MAGICAL SMOOTH SURFACE TRICK (NOW WITH GREEKS) ---
             xi = np.linspace(X.min(), X.max(), 30)
             yi = np.linspace(Y.min(), Y.max(), 30)
             xi, yi = np.meshgrid(xi, yi)

             def interpolate_grid(val_array):
                 g = griddata((X, Y), val_array, (xi, yi), method="linear")
                 return np.where(np.isnan(g), griddata((X, Y), val_array, (xi, yi), method="nearest"), g)

             zi = interpolate_grid(Z)
             # Interpolate Greeks to match the regular grid layout
             delta_i = interpolate_grid(surface_data['delta'].values)
             gamma_i = interpolate_grid(surface_data['gamma'].values)
             vega_i  = interpolate_grid(surface_data['vega'].values)
             theta_i = interpolate_grid(surface_data['theta'].values)
             price_i = interpolate_grid(surface_data['price'].values)
             
             # Stack for Plotly customdata
             custom_data = np.stack((delta_i, gamma_i, vega_i, theta_i, price_i), axis=-1)
             
             fig = go.Figure(data=[go.Surface(
                 x=xi, y=yi, z=zi, 
                 customdata=custom_data,
                 hovertemplate=(
                     "<b>DTE</b>: %{x:.2f} yrs<br>" +
                     "<b>%{_y_label}</b>: %{y:.2f}<br>" +
                     "<b>IV</b>: %{z:.2f}%<br>" +
                     "<extra>" +
                     "<b>Opt Price</b>: $%{customdata[4]:.2f}<br>" +
                     "<b>Delta</b>: %{customdata[0]:.4f}<br>" +
                     "<b>Gamma</b>: %{customdata[1]:.4f}<br>" +
                     "<b>Vega</b>: %{customdata[2]:.4f}<br>" +
                     "<b>Theta</b>: %{customdata[3]:.4f}" +
                     "</extra>"
                 ).replace('%{_y_label}', 'Strike' if y_axis_type != "Log-Moneyness" else 'Log-Money'),
                 colorscale="Viridis",
                 colorbar=dict(title="IV (%)")
             )])

             fig.update_layout(
                 title=f"{calc_symbol} Volatility Surface ({calc_otype.capitalize()}) - Interactive Greeks",
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
             
     with tab2:
         st.subheader("2D Cross-Section (Volatility Smile)")
         unique_dtes = np.sort(surface_data['dtes'].unique())
         if len(unique_dtes) > 0:
             selected_dte = st.selectbox("Select Expiration (DTE in Years)", unique_dtes, format_func=lambda x: f"{x:.4f} yrs (~{int(x*365)} days)")
             
             smile_data = surface_data[surface_data['dtes'] == selected_dte].sort_values(by='Y_metric')
             
             if not smile_data.empty:
                 custom_data_2d = np.stack((
                     smile_data['delta'].values, 
                     smile_data['gamma'].values, 
                     smile_data['vega'].values, 
                     smile_data['theta'].values, 
                     smile_data['price'].values
                 ), axis=-1)
                 
                 fig2d = go.Figure(data=[go.Scatter(
                     x=smile_data['Y_metric'],
                     y=smile_data['ivs'],
                     mode='lines+markers',
                     marker=dict(size=8, color=smile_data['ivs'], colorscale='Viridis', showscale=True, colorbar=dict(title="IV (%)")),
                     line=dict(width=2, color='rgba(255,255,255,0.2)'),
                     customdata=custom_data_2d,
                     hovertemplate=(
                         "<b>%{_y_label}</b>: %{x:.2f}<br>" +
                         "<b>IV</b>: %{y:.2f}%<br>" +
                         "<extra>" +
                         "<b>Opt Price</b>: $%{customdata[4]:.2f}<br>" +
                         "<b>Delta</b>: %{customdata[0]:.4f}<br>" +
                         "<b>Gamma</b>: %{customdata[1]:.4f}<br>" +
                         "<b>Vega</b>: %{customdata[2]:.4f}<br>" +
                         "<b>Theta</b>: %{customdata[3]:.4f}" +
                         "</extra>"
                     ).replace('%{_y_label}', 'Strike' if y_axis_type != "Log-Moneyness" else 'Log-Money')
                 )])
                 
                 fig2d.update_layout(
                     title=f"{calc_symbol} Volatility Smile ({calc_otype.capitalize()}) for DTE = {selected_dte:.4f}",
                     xaxis_title=y_label,
                     yaxis_title="Implied Volatility (%)",
                     height=600,
                     margin=dict(l=0, r=0, b=0, t=50)
                 )
                 st.plotly_chart(fig2d, use_container_width=True)
             else:
                 st.warning("No data points for selected DTE.")
else:
    st.info("👈 Enter parameters in the sidebar and click 'Generate Smooth Graph with Greeks'.")
