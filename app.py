# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import joblib
from pathlib import Path

from bs.pricing import calculate_black_scholes_price
from bs.greeks import (
    calculate_delta, calculate_gamma, calculate_vega,
    calculate_theta, calculate_rho
)
from bs.utils import monte_carlo_pricing
from analysis.sensitivity_grid import generate_price_grid
from rl_hedging.model import DeltaNet


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Black–Scholes Dashboard", layout="wide")
st.title("Options Research Dashboard")

# ----------------------------
# Navigation (single-page render so sidebar stays page-specific)
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Black–Scholes", "Monte Carlo", "RL Hedging (ML)"],
    key="page_nav"
)


@st.cache_resource
def load_delta_model_and_scaler(model_path, scaler_path, device):
    model = DeltaNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_call_delta_batch(model, scaler, device, log_moneyness, time_to_maturity, volatility):
    features = np.column_stack([log_moneyness, time_to_maturity, volatility])
    features_scaled = scaler.transform(features)
    x = torch.tensor(features_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = model(x).cpu().numpy().reshape(-1)
    return np.clip(preds, 0.0, 1.0)


# ============================
# BLACK-SCHOLES PAGE
# ============================
if page == "Black–Scholes":
    st.sidebar.header("Option Parameters")

    S = st.sidebar.number_input("Spot Price ($)", value=100.0, min_value=0.01, key="bs_s")
    K = st.sidebar.number_input("Strike Price ($)", value=100.0, min_value=0.01, key="bs_k")
    T = st.sidebar.number_input("Time to Maturity (Years)", value=1.0, min_value=0.01, key="bs_t")
    r = st.sidebar.number_input("Risk-free Rate (%)", value=5.0, key="bs_r") / 100
    sigma = st.sidebar.number_input("Volatility (%)", value=20.0, key="bs_sigma") / 100
    option_type = st.sidebar.selectbox("Option Type", ["call", "put"], key="bs_option_type")

    st.sidebar.header("Sensitivity Heatmap")
    spot_pct_min, spot_pct_max = st.sidebar.slider(
        "Spot Price Change (%)", -80, 80, (-50, 50), key="bs_spot_range"
    )
    sigma_min, sigma_max = st.sidebar.slider(
        "Volatility Range (%)", 1, 200, (10, 50), key="bs_vol_range"
    )
    grid_size = st.sidebar.slider("Grid Resolution", 10, 40, 20, key="bs_grid")

    st.subheader("Single-Point Valuation")

    price = calculate_black_scholes_price(S, K, T, r, sigma, option_type)
    delta = calculate_delta(S, K, T, r, sigma, option_type)
    gamma = calculate_gamma(S, K, T, r, sigma)
    vega = calculate_vega(S, K, T, r, sigma)
    theta = calculate_theta(S, K, T, r, sigma, option_type)
    rho = calculate_rho(S, K, T, r, sigma, option_type)

    col1, col2, col3 = st.columns(3)
    col1.metric("Option Price", f"${price:.2f}")

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

    with st.expander("What are the Greeks?"):
        st.markdown("""
        - **Delta (Δ):** How much the option price moves if the stock moves $1  
        - **Gamma (Γ):** How fast Delta changes as the stock moves  
        - **Vega (ν):** How sensitive the option is to changes in volatility  
        - **Theta (Θ):** How much value the option loses each day  
        - **Rho (ρ):** How sensitive the option is to changes in interest rates
        """)

    spot_prices, volatilities, call_price_grid, put_price_grid = generate_price_grid(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type=option_type,
        spot_pct_range=(spot_pct_min / 100, spot_pct_max / 100),
        sigma_range=(sigma_min / 100, sigma_max / 100),
        grid_size=grid_size,
    )

    parity_rhs = np.broadcast_to(
        spot_prices[None, :] - K * np.exp(-r * T),
        call_price_grid.shape,
    )
    parity_error = np.abs(call_price_grid - put_price_grid - parity_rhs)

    price_grid = call_price_grid if option_type == "call" else put_price_grid
    st.subheader(f"{option_type.capitalize()} Option Price Sensitivity (Spot × Volatility)")

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
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Heatmap Diagnostics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min Price", f"{np.nanmin(price_grid):.2f}")
    col2.metric("Max Price", f"{np.nanmax(price_grid):.2f}")
    col3.metric("Price Range", f"{(np.nanmax(price_grid) - np.nanmin(price_grid)):.2f}")
    col4.metric("Mean Price", f"{np.nanmean(price_grid):.2f}")

    st.subheader("Advanced Sensitivity Table")
    with st.expander("Show Detailed Table (Spot × Volatility)"):
        spot_flat = np.repeat(spot_prices, len(volatilities))
        vol_flat = np.tile(volatilities, len(spot_prices))
        call_flat = call_price_grid.flatten()
        put_flat = put_price_grid.flatten()
        parity_rhs_flat = parity_rhs.flatten()
        parity_err_flat = parity_error.flatten()

        delta_flat = np.array([
            calculate_delta(s, K, T, r, sigma_val, "call")
            for s, sigma_val in zip(spot_flat, vol_flat)
        ])
        gamma_flat = np.array([
            calculate_gamma(s, K, T, r, sigma_val)
            for s, sigma_val in zip(spot_flat, vol_flat)
        ])

        df = pd.DataFrame(
            {
                "Spot": spot_flat,
                "Volatility": vol_flat,
                "Call Price": call_flat,
                "Put Price": put_flat,
                "Parity RHS": parity_rhs_flat,
                "Parity Error": parity_err_flat,
                "Delta": delta_flat,
                "Gamma": gamma_flat,
            }
        )

        st.dataframe(
            df.style.format(
                {
                    "Call Price": "{:.2f}",
                    "Put Price": "{:.2f}",
                    "Parity RHS": "{:.2f}",
                    "Parity Error": "{:.2e}",
                    "Delta": "{:.4f}",
                    "Gamma": "{:.4f}",
                }
            )
        )


# ============================
# MONTE CARLO PAGE
# ============================
elif page == "Monte Carlo":
    st.header("Monte Carlo Option Pricing")
    st.markdown(
        """
    This page uses Monte Carlo simulation to price European call options by simulating stock price paths
    using geometric Brownian motion. The method is particularly useful for understanding option value
    distributions and validating Black-Scholes results.
    """
    )

    st.sidebar.header("Monte Carlo Parameters")
    mc_S0 = st.sidebar.number_input("Spot Price (S₀)", value=100.0, min_value=0.01, key="mc_spot")
    mc_r = st.sidebar.number_input("Risk-free Rate (%)", value=5.0, key="mc_rate") / 100
    mc_sigma = st.sidebar.number_input("Volatility (%)", value=20.0, key="mc_sigma") / 100
    mc_T = st.sidebar.number_input("Time to Maturity (Years)", value=1.0, min_value=0.01, key="mc_T")
    mc_N = st.sidebar.number_input("Number of Time Steps", value=252, min_value=1, step=10, key="mc_N")
    mc_M = st.sidebar.number_input("Number of Simulations", value=10000, min_value=100, step=1000, key="mc_M")
    mc_strikes_input = st.sidebar.text_area(
        "Strike Prices (comma-separated)",
        value="90, 100, 110",
        key="mc_strikes",
    )

    try:
        mc_strikes = np.array([float(x.strip()) for x in mc_strikes_input.split(",")])
        strikes_valid = True
    except ValueError:
        st.error("Invalid strike prices. Please enter numbers separated by commas.")
        strikes_valid = False

    run_simulation = st.sidebar.button("Run Monte Carlo Simulation", key="run_mc", use_container_width=True)

    if run_simulation and strikes_valid:
        with st.spinner("Running simulation..."):
            np.random.seed(42)
            mc_results = monte_carlo_pricing(mc_S0, mc_r, mc_sigma, mc_T, mc_strikes, mc_N, mc_M)

        st.success("Simulation complete!")

        st.subheader("Option Pricing Results")
        results_df = pd.DataFrame(
            {
                "Strike": mc_results["strikes"],
                "MC Price": mc_results["prices"],
                "Std Error": mc_results["std_errors"],
                "95% CI Lower": mc_results["prices"] - 1.96 * mc_results["std_errors"],
                "95% CI Upper": mc_results["prices"] + 1.96 * mc_results["std_errors"],
            }
        )

        st.dataframe(
            results_df.style.format(
                {
                    "Strike": "{:.2f}",
                    "MC Price": "{:.4f}",
                    "Std Error": "{:.4f}",
                    "95% CI Lower": "{:.4f}",
                    "95% CI Upper": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

        st.subheader("Comparison with Black-Scholes")
        bs_prices = np.array(
            [
                calculate_black_scholes_price(mc_S0, strike, mc_T, mc_r, mc_sigma, "call")
                for strike in mc_results["strikes"]
            ]
        )

        comparison_df = pd.DataFrame(
            {
                "Strike": mc_results["strikes"],
                "MC Price": mc_results["prices"],
                "BS Price": bs_prices,
                "Difference": np.abs(mc_results["prices"] - bs_prices),
                "% Difference": 100 * np.abs(mc_results["prices"] - bs_prices) / bs_prices,
            }
        )

        st.dataframe(
            comparison_df.style.format(
                {
                    "Strike": "{:.2f}",
                    "MC Price": "{:.4f}",
                    "BS Price": "{:.4f}",
                    "Difference": "{:.4f}",
                    "% Difference": "{:.2f}%",
                }
            ),
            use_container_width=True,
        )

        st.subheader("Option Prices Across Strikes (Heatmap)")
        st.markdown("Visualization of option prices for each strike (single row heatmap for clarity).")

        heatmap_data = np.array([mc_results["prices"]])
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap_data,
                x=np.round(mc_results["strikes"], 2),
                y=["MC Price"],
                colorscale="Viridis",
                hovertemplate="Strike: %{x}<br>Price: %{z:.4f}<extra></extra>",
            )
        )
        fig_heatmap.update_layout(
            title="Monte Carlo Option Prices by Strike",
            xaxis_title="Strike Price ($)",
            yaxis_title="",
            height=300,
            width=1000,
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.subheader("Monte Carlo vs Black-Scholes Comparison")
        fig_comparison = go.Figure()
        fig_comparison.add_trace(
            go.Scatter(
                x=comparison_df["Strike"],
                y=comparison_df["MC Price"],
                mode="lines+markers",
                name="Monte Carlo",
                line=dict(color="blue", width=2),
                marker=dict(size=8),
            )
        )
        fig_comparison.add_trace(
            go.Scatter(
                x=comparison_df["Strike"],
                y=comparison_df["BS Price"],
                mode="lines+markers",
                name="Black-Scholes",
                line=dict(color="red", width=2, dash="dash"),
                marker=dict(size=8),
            )
        )
        fig_comparison.update_layout(
            title="Option Prices: Monte Carlo vs Black-Scholes",
            xaxis_title="Strike Price ($)",
            yaxis_title="Option Price ($)",
            height=500,
            hovermode="x unified",
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

        st.subheader("Simulated Stock Price Paths")
        st.markdown("Distribution of final stock prices from all simulations.")

        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=mc_results["final_prices"],
                nbinsx=50,
                name="Final Prices",
                marker_color="rgba(0, 100, 200, 0.7)",
            )
        )
        fig_dist.add_vline(
            x=mc_S0,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Initial S₀ = ${mc_S0:.2f}",
            annotation_position="top right",
        )
        fig_dist.update_layout(
            title="Distribution of Final Stock Prices (T years)",
            xaxis_title="Stock Price ($)",
            yaxis_title="Frequency",
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader("Final Stock Price Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Mean", f"${np.mean(mc_results['final_prices']):.2f}")
        col2.metric("Std Dev", f"${np.std(mc_results['final_prices']):.2f}")
        col3.metric("Min", f"${np.min(mc_results['final_prices']):.2f}")
        col4.metric("Max", f"${np.max(mc_results['final_prices']):.2f}")
        col5.metric("Initial S₀", f"${mc_S0:.2f}")

    elif run_simulation and not strikes_valid:
        st.error("Please fix the strike prices before running the simulation.")

    with st.expander("About Monte Carlo Simulation"):
        st.markdown(
            """
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
        """
        )


# ============================
# RL HEDGING (ML) PAGE
# ============================
else:
    st.header("🤖 AI Hedging Experience Engine")
    st.markdown("*Classical Black–Scholes Delta vs Neural Network Delta Approximation*")

    st.sidebar.header("RL Hedging Parameters")
    st.sidebar.caption("Ranges tuned to the core ML training regime.")
    rl_S0 = st.sidebar.slider(
        "Spot Price ($)", min_value=60.0, max_value=165.0, value=100.0, step=1.0, key="rl_s0"
    )
    rl_K = st.sidebar.slider(
        "Strike Price ($)", min_value=80.0, max_value=120.0, value=100.0, step=1.0, key="rl_k"
    )
    rl_T = st.sidebar.slider(
        "Time Horizon (Years)", min_value=0.02, max_value=1.0, value=0.25, step=0.01, key="rl_t"
    )
    rl_r = st.sidebar.slider(
        "Risk-free Rate (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.1, key="rl_r"
    ) / 100
    rl_sigma = st.sidebar.slider(
        "Volatility (%)", min_value=10.0, max_value=60.0, value=20.0, step=1.0, key="rl_sigma"
    ) / 100
    rl_option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"], key="rl_option_type")

    st.sidebar.header("Strategy")
    rl_strategy_mode = st.sidebar.selectbox(
        "Choose Strategy",
        ["ML Delta"],
        index=0,
        key="rl_strategy_mode",
        help="Current active strategy: neural-network delta approximation.",
    )

    st.sidebar.header("Visualization")
    rl_show_path = st.sidebar.checkbox(
        "📈 Path Trajectory Plot",
        value=True,
        key="rl_show_path",
        help="Show stock price evolution over time",
    )
    rl_show_hedge = st.sidebar.checkbox(
        "🎯 Hedge Ratio Evolution",
        value=True,
        key="rl_show_hedge",
        help="Show how hedge ratios change over time",
    )
    rl_show_perf = st.sidebar.checkbox(
        "📊 Performance Summary",
        value=True,
        key="rl_show_perf",
        help="Display metrics and statistics",
    )

    run_rl_simulation = st.sidebar.button(
        "🚀 Run Hedging Simulation",
        use_container_width=True,
        key="run_rl_simulation",
    )

    st.subheader("Selected Configuration")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    cfg_col1.metric("Spot / Strike", f"${rl_S0:.0f} / ${rl_K:.0f}")
    cfg_col2.metric("T / r", f"{rl_T:.2f}y / {rl_r*100:.1f}%")
    cfg_col3.metric("Vol / Type", f"{rl_sigma*100:.1f}% / {rl_option_type}")

    st.caption(f"Mode: {rl_strategy_mode}")

    if run_rl_simulation:
        model_path = Path("delta_model.pt")
        scaler_path = Path("scaler.pkl")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if not model_path.exists() or not scaler_path.exists():
            st.error(
                "Model artifacts not found. Train first to create `delta_model.pt` and `scaler.pkl`."
            )
        else:
            model, scaler = load_delta_model_and_scaler(str(model_path), str(scaler_path), device)

            log_m = np.log(rl_S0 / rl_K)
            bs_call_delta = calculate_delta(rl_S0, rl_K, rl_T, rl_r, rl_sigma, "call")
            ml_call_delta = predict_call_delta_batch(
                model=model,
                scaler=scaler,
                device=device,
                log_moneyness=np.array([log_m]),
                time_to_maturity=np.array([rl_T]),
                volatility=np.array([rl_sigma]),
            )[0]

            if rl_option_type == "Put":
                bs_delta = bs_call_delta - 1.0
                ml_delta = ml_call_delta - 1.0
            else:
                bs_delta = bs_call_delta
                ml_delta = ml_call_delta

            abs_error = abs(ml_delta - bs_delta)

            st.subheader("Model Inference Snapshot")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            metric_col1.metric("Black-Scholes Delta", f"{bs_delta:.6f}")
            metric_col2.metric("ML Predicted Delta", f"{ml_delta:.6f}")
            metric_col3.metric("Absolute Error", f"{abs_error:.6f}")
            metric_col4.metric("Error (bps)", f"{abs_error * 10000:.2f}")

            spot_grid = np.linspace(0.6 * rl_K, 1.4 * rl_K, 120)
            log_m_grid = np.log(spot_grid / rl_K)
            ttm_grid = np.full_like(spot_grid, rl_T, dtype=float)
            vol_grid = np.full_like(spot_grid, rl_sigma, dtype=float)

            ml_call_grid = predict_call_delta_batch(
                model=model,
                scaler=scaler,
                device=device,
                log_moneyness=log_m_grid,
                time_to_maturity=ttm_grid,
                volatility=vol_grid,
            )
            bs_call_grid = np.array(
                [calculate_delta(s, rl_K, rl_T, rl_r, rl_sigma, "call") for s in spot_grid]
            )

            if rl_option_type == "Put":
                ml_delta_grid = ml_call_grid - 1.0
                bs_delta_grid = bs_call_grid - 1.0
            else:
                ml_delta_grid = ml_call_grid
                bs_delta_grid = bs_call_grid

            if rl_show_path:
                st.markdown("### 📈 Delta Curve vs Spot")
                fig_curve = go.Figure()
                fig_curve.add_trace(
                    go.Scatter(
                        x=spot_grid,
                        y=bs_delta_grid,
                        mode="lines",
                        name="Black-Scholes Delta",
                        line=dict(color="royalblue", width=3),
                    )
                )
                fig_curve.add_trace(
                    go.Scatter(
                        x=spot_grid,
                        y=ml_delta_grid,
                        mode="lines",
                        name="ML Delta",
                        line=dict(color="darkorange", width=2, dash="dash"),
                    )
                )
                fig_curve.update_layout(
                    title=f"{rl_option_type} Delta Approximation (T={rl_T:.2f}, σ={rl_sigma:.2f})",
                    xaxis_title="Spot Price ($)",
                    yaxis_title="Delta",
                    hovermode="x unified",
                    height=420,
                )
                st.plotly_chart(fig_curve, use_container_width=True)

            if rl_show_hedge:
                st.markdown("### 🎯 Delta Error by Spot")
                error_grid = ml_delta_grid - bs_delta_grid
                fig_error = go.Figure(
                    data=go.Scatter(
                        x=spot_grid,
                        y=error_grid,
                        mode="lines",
                        line=dict(color="crimson", width=2),
                        name="ML - BS",
                    )
                )
                fig_error.add_hline(y=0.0, line_dash="dot", line_color="gray")
                fig_error.update_layout(
                    xaxis_title="Spot Price ($)",
                    yaxis_title="Delta Error",
                    height=320,
                    showlegend=False,
                )
                st.plotly_chart(fig_error, use_container_width=True)

            if rl_show_perf:
                st.markdown("### 📊 Performance Summary")
                mae = np.mean(np.abs(ml_delta_grid - bs_delta_grid))
                rmse = np.sqrt(np.mean((ml_delta_grid - bs_delta_grid) ** 2))
                max_err = np.max(np.abs(ml_delta_grid - bs_delta_grid))

                col_perf1, col_perf2, col_perf3 = st.columns(3)
                col_perf1.metric("MAE", f"{mae:.6f}")
                col_perf2.metric("RMSE", f"{rmse:.6f}")
                col_perf3.metric("Max |Error|", f"{max_err:.6f}")
