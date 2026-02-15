# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from bs.pricing import calculate_black_scholes_price
from bs.greeks import (
    calculate_delta, calculate_gamma, calculate_vega,
    calculate_theta, calculate_rho
)
from bs.utils import monte_carlo_pricing
from analysis.sensitivity_grid import generate_price_grid

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Black–Scholes Dashboard", layout="wide")
st.title("Black–Scholes Option Dashboard")

# ----------------------------
# Create tabs for navigation
# ----------------------------
tab_bs, tab_mc = st.tabs(["Black–Scholes", "Monte Carlo"])

# ============================
# BLACK-SCHOLES TAB
# ============================
with tab_bs:
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

# ============================
# MONTE CARLO TAB
# ============================
with tab_mc:
    st.header("Monte Carlo Option Pricing")
    st.markdown("""
    This tab uses Monte Carlo simulation to price European call options by simulating stock price paths 
    using geometric Brownian motion. The method is particularly useful for understanding option value 
    distributions and validating Black-Scholes results.
    """)

    # ----------------------------
    # Input section
    # ----------------------------
    st.subheader("Simulation Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        mc_S0 = st.number_input("Spot Price (S₀)", value=100.0, min_value=0.01, key="mc_spot")
        mc_r = st.number_input("Risk-free Rate (%)", value=5.0, key="mc_rate") / 100
        mc_sigma = st.number_input("Volatility (%)", value=20.0, key="mc_sigma") / 100
    
    with col2:
        mc_T = st.number_input("Time to Maturity (Years)", value=1.0, min_value=0.01, key="mc_T")
        mc_N = st.number_input("Number of Time Steps", value=252, min_value=1, step=10, key="mc_N")
        mc_M = st.number_input("Number of Simulations", value=10000, min_value=100, step=1000, key="mc_M")
    
    with col3:
        mc_strikes_input = st.text_area(
            "Strike Prices (comma-separated, e.g., 90, 100, 110)",
            value="90, 100, 110",
            key="mc_strikes"
        )
    
    # ----------------------------
    # Parse strikes input
    # ----------------------------
    try:
        mc_strikes = np.array([float(x.strip()) for x in mc_strikes_input.split(',')])
        strikes_valid = True
    except ValueError:
        st.error("Invalid strike prices. Please enter numbers separated by commas.")
        strikes_valid = False
    
    # ----------------------------
    # Run simulation button
    # ----------------------------
    run_simulation = st.button("Run Monte Carlo Simulation", key="run_mc")
    
    if run_simulation and strikes_valid:
        # Run Monte Carlo pricing
        with st.spinner("Running simulation..."):
            # Set random seed for reproducibility
            np.random.seed(42)
            mc_results = monte_carlo_pricing(mc_S0, mc_r, mc_sigma, mc_T, mc_strikes, mc_N, mc_M)
        
        st.success("Simulation complete!")
        
        # ----------------------------
        # Results table
        # ----------------------------
        st.subheader("Option Pricing Results")
        results_df = pd.DataFrame({
            "Strike": mc_results['strikes'],
            "MC Price": mc_results['prices'],
            "Std Error": mc_results['std_errors'],
            "95% CI Lower": mc_results['prices'] - 1.96 * mc_results['std_errors'],
            "95% CI Upper": mc_results['prices'] + 1.96 * mc_results['std_errors'],
        })
        
        st.dataframe(
            results_df.style.format({
                "Strike": "{:.2f}",
                "MC Price": "{:.4f}",
                "Std Error": "{:.4f}",
                "95% CI Lower": "{:.4f}",
                "95% CI Upper": "{:.4f}"
            }),
            use_container_width=True
        )
        
        # ----------------------------
        # Comparison with Black-Scholes
        # ----------------------------
        st.subheader("Comparison with Black-Scholes")
        
        # Calculate Black-Scholes prices for the same strikes
        bs_prices = np.array([
            calculate_black_scholes_price(mc_S0, K, mc_T, mc_r, mc_sigma, "call")
            for K in mc_results['strikes']
        ])
        
        comparison_df = pd.DataFrame({
            "Strike": mc_results['strikes'],
            "MC Price": mc_results['prices'],
            "BS Price": bs_prices,
            "Difference": np.abs(mc_results['prices'] - bs_prices),
            "% Difference": 100 * np.abs(mc_results['prices'] - bs_prices) / bs_prices
        })
        
        st.dataframe(
            comparison_df.style.format({
                "Strike": "{:.2f}",
                "MC Price": "{:.4f}",
                "BS Price": "{:.4f}",
                "Difference": "{:.4f}",
                "% Difference": "{:.2f}%"
            }),
            use_container_width=True
        )
        
        # ----------------------------
        # Heatmap: Option prices across strikes
        # ----------------------------
        st.subheader("Option Prices Across Strikes (Heatmap)")
        st.markdown("Visualization of option prices for each strike (single row heatmap for clarity).")
        
        heatmap_data = np.array([mc_results['prices']])
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=np.round(mc_results['strikes'], 2),
            y=["MC Price"],
            colorscale="Viridis",
            hovertemplate="Strike: %{x}<br>Price: %{z:.4f}<extra></extra>"
        ))
        
        fig_heatmap.update_layout(
            title="Monte Carlo Option Prices by Strike",
            xaxis_title="Strike Price ($)",
            yaxis_title="",
            height=300,
            width=1000
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # ----------------------------
        # Visualization: MC vs BS Comparison
        # ----------------------------
        st.subheader("Monte Carlo vs Black-Scholes Comparison")
        
        fig_comparison = go.Figure()
        
        # Add Monte Carlo prices
        fig_comparison.add_trace(go.Scatter(
            x=comparison_df['Strike'],
            y=comparison_df['MC Price'],
            mode='lines+markers',
            name='Monte Carlo',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # Add Black-Scholes prices
        fig_comparison.add_trace(go.Scatter(
            x=comparison_df['Strike'],
            y=comparison_df['BS Price'],
            mode='lines+markers',
            name='Black-Scholes',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        fig_comparison.update_layout(
            title="Option Prices: Monte Carlo vs Black-Scholes",
            xaxis_title="Strike Price ($)",
            yaxis_title="Option Price ($)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # ----------------------------
        # Stock price path distribution
        # ----------------------------
        st.subheader("Simulated Stock Price Paths")
        st.markdown("Distribution of final stock prices from all simulations.")
        
        # Create histogram of final prices
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=mc_results['final_prices'],
            nbinsx=50,
            name='Final Prices',
            marker_color='rgba(0, 100, 200, 0.7)'
        ))
        
        # Add vertical line for initial spot price
        fig_dist.add_vline(
            x=mc_S0,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Initial S₀ = ${mc_S0:.2f}",
            annotation_position="top right"
        )
        
        fig_dist.update_layout(
            title="Distribution of Final Stock Prices (T years)",
            xaxis_title="Stock Price ($)",
            yaxis_title="Frequency",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Display final price statistics
        st.subheader("Final Stock Price Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Mean", f"${np.mean(mc_results['final_prices']):.2f}")
        col2.metric("Std Dev", f"${np.std(mc_results['final_prices']):.2f}")
        col3.metric("Min", f"${np.min(mc_results['final_prices']):.2f}")
        col4.metric("Max", f"${np.max(mc_results['final_prices']):.2f}")
        col5.metric("Initial S₀", f"${mc_S0:.2f}")
    
    elif run_simulation and not strikes_valid:
        st.error("Please fix the strike prices before running the simulation.")
    
    # ----------------------------
    # Information section
    # ----------------------------
    with st.expander("About Monte Carlo Simulation"):
        st.markdown("""
        ### How It Works:
        
        1. **Geometric Brownian Motion (GBM):** Stock prices follow the SDE:
           - dS = μS dt + σS dW
           - Where μ = r (risk-neutral drift), σ = volatility, dW = Wiener process increment
        
        2. **Path Generation:** We simulate M paths over N time steps, each representing a possible future stock price trajectory.
        
        3. **Payoff Calculation:** At maturity (T), compute the call option payoff: max(S_T - K, 0) for each path.
        
        4. **Pricing:** Average payoffs across all paths and discount to present value using e^(-rT).
        
        5. **Error Estimates:** Standard error decreases with O(1/√M), so more simulations = higher accuracy.
        
        ### Advantages:
        - Can handle path-dependent options (American, Asian, Barrier, etc.)
        - Validates Black-Scholes theoretical prices
        - Provides confidence intervals and statistical measures
        
        ### Disadvantages:
        - Computationally expensive for large M or N
        - Monte Carlo error inherent; Black-Scholes is exact for European options
        """)

