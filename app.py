# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from bs.pricing import calculate_black_scholes_price
from bs.greeks import (
    calculate_delta, calculate_gamma, calculate_vega,
    calculate_theta, calculate_rho
)
from analysis.sensitivity_grid import generate_price_grid

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Black–Scholes Dashboard", layout="wide")
st.title("Black–Scholes Option Dashboard")

# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.header("Option Parameters")

S = st.sidebar.number_input("Spot Price ($)", value=100.0, min_value=0.01)
K = st.sidebar.number_input("Strike Price ($)", value=100.0, min_value=0.01)
T = st.sidebar.number_input("Time to Maturity (Years)", value=1.0, min_value=0.01)
r = st.sidebar.number_input("Risk-free Rate (%)", value=5.0) / 100
sigma = st.sidebar.number_input("Volatility (%)", value=20.0) / 100
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

# ----------------------------
# Single-point valuation
# ----------------------------
st.subheader("Single-Point Valuation")

price = calculate_black_scholes_price(S, K, T, r, sigma, option_type)
delta = calculate_delta(S, K, T, r, sigma, option_type)
gamma = calculate_gamma(S, K, T, r, sigma)
vega = calculate_vega(S, K, T, r, sigma)
theta = calculate_theta(S, K, T, r, sigma, option_type)
rho = calculate_rho(S, K, T, r, sigma, option_type)

# Display Greeks in two columns
col1, col2, col3 = st.columns(3)
col1.metric("Option Price", f"${price:.2f}")
# Greeks in columns
st.markdown("### Greeks")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Delta (Δ)", f"{delta:.4f}", help="Sensitivity to underlying price")
    st.metric("Gamma (Γ)", f"{gamma:.4f}", help="Stability of Delta")
with col2:
    st.metric("Vega (ν)", f"{vega:.4f}", help="Sensitivity to volatility")
    st.metric("Theta (Θ)", f"{theta:.4f}", help="Time decay per day")
with col3:
    st.metric("Rho (ρ)", f"{rho:.4f}", help="Sensitivity to interest rates")

# Optional expandable explanation
with st.expander("What are the Greeks?"):
    st.markdown("""
    - **Delta (Δ):** How much the option price moves if the stock moves $1  
    - **Gamma (Γ):** How fast Delta changes as the stock moves  
    - **Vega (ν):** How sensitive the option is to changes in volatility  
    - **Theta (Θ):** How much value the option loses each day  
    - **Rho (ρ):** How sensitive the option is to changes in interest rates
    """)


# ----------------------------
# Sensitivity controls
# ----------------------------
st.sidebar.header("Sensitivity Heatmap")
spot_pct_min, spot_pct_max = st.sidebar.slider(
    "Spot Price Change (%)", -80, 80, (-50, 50)
)
sigma_min, sigma_max = st.sidebar.slider(
    "Volatility Range (%)", 1, 200, (10, 50)
)
grid_size = st.sidebar.slider("Grid Resolution", 10, 40, 20)

# ----------------------------
# Generate grids
# ----------------------------
spot_prices, volatilities, call_price_grid, put_price_grid = generate_price_grid(
    S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type,
    spot_pct_range=(spot_pct_min / 100, spot_pct_max / 100),
    sigma_range=(sigma_min / 100, sigma_max / 100),
    grid_size=grid_size
)

# ----------------------------
# Parity calculations
# ----------------------------
parity_rhs = np.broadcast_to(
    spot_prices[None, :] - K * np.exp(-r * T),
    call_price_grid.shape
)
parity_error = np.abs(call_price_grid - put_price_grid - parity_rhs)

# ----------------------------
# Select grid to display
# ----------------------------
price_grid = call_price_grid if option_type == "call" else put_price_grid
st.subheader(f"{option_type.capitalize()} Option Price Sensitivity (Spot × Volatility)")

# ----------------------------
# Attach hover data
# ----------------------------
customdata = np.stack([call_price_grid, put_price_grid, parity_rhs, parity_error], axis=-1)

fig = px.imshow(
    price_grid,
    x=np.round(spot_prices, 2),
    y=np.round(volatilities * 100, 1),
    origin="lower",
    color_continuous_scale="YlGnBu",
    labels={"x": "Spot Price ($)", "y": "Volatility (%)", "color": "Option Price"},
)
fig.update_traces(
    customdata=customdata,
    hovertemplate=(
        "Spot: %{x}<br>"
        "Volatility: %{y}%<br>"
        "Displayed Price: %{z:.2f}<br><br>"
        "Call Price: %{customdata[0]:.2f}<br>"
        "Put Price: %{customdata[1]:.2f}<br>"
        "Parity RHS: %{customdata[2]:.2f}<br>"
        "Parity Error: %{customdata[3]:.2e}"
    )
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Heatmap diagnostics
# ----------------------------
st.subheader("Heatmap Diagnostics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Min Price", f"{np.nanmin(price_grid):.2f}")
col2.metric("Max Price", f"{np.nanmax(price_grid):.2f}")
col3.metric("Price Range", f"{(np.nanmax(price_grid) - np.nanmin(price_grid)):.2f}")
col4.metric("Mean Price", f"{np.nanmean(price_grid):.2f}")

# ----------------------------
# Advanced Sensitivity Table
# ----------------------------
st.subheader("Advanced Sensitivity Table")
with st.expander("Show Detailed Table (Spot × Volatility)"):
    # Flatten grids for table
    spot_flat = np.repeat(spot_prices, len(volatilities))
    vol_flat = np.tile(volatilities, len(spot_prices))
    call_flat = call_price_grid.flatten()
    put_flat = put_price_grid.flatten()
    parity_rhs_flat = parity_rhs.flatten()
    parity_err_flat = parity_error.flatten()

    # Optional: compute Delta/Gamma
    delta_flat = np.array([calculate_delta(s, K, T, r, sigma_val, "call")
                           for s, sigma_val in zip(spot_flat, vol_flat)])
    gamma_flat = np.array([calculate_gamma(s, K, T, r, sigma_val)
                           for s, sigma_val in zip(spot_flat, vol_flat)])

    df = pd.DataFrame({
        "Spot": spot_flat,
        "Volatility": vol_flat,
        "Call Price": call_flat,
        "Put Price": put_flat,
        "Parity RHS": parity_rhs_flat,
        "Parity Error": parity_err_flat,
        "Delta": delta_flat,
        "Gamma": gamma_flat
    })

    st.dataframe(df.style.format({
        "Call Price": "{:.2f}",
        "Put Price": "{:.2f}",
        "Parity RHS": "{:.2f}",
        "Parity Error": "{:.2e}",
        "Delta": "{:.4f}",
        "Gamma": "{:.4f}"
    }))
