import time
import numpy as np
from bs.pricing import calculate_black_scholes_price
from bs.greeks import (
    calculate_delta, calculate_gamma, calculate_vega,
    calculate_theta, calculate_rho
)
from analysis.sensitivity_grid import generate_price_grid

# ----------------------------
# Parameters
# ----------------------------
S = 100
K = 100
T = 1.0
r = 0.05
sigma = 0.2
grid_size = 40
spot_pct_range = (-0.5, 0.5)
sigma_range = (0.1, 1.0)
option_type = "call"

# ----------------------------
# Single-point benchmarking
# ----------------------------
start_time = time.time()
price = calculate_black_scholes_price(S, K, T, r, sigma, option_type)
delta = calculate_delta(S, K, T, r, sigma, option_type)
gamma = calculate_gamma(S, K, T, r, sigma)
vega = calculate_vega(S, K, T, r, sigma)
theta = calculate_theta(S, K, T, r, sigma, option_type)
rho = calculate_rho(S, K, T, r, sigma, option_type)
end_time = time.time()

print("Single-point evaluation:")
print(f"  Price: {price:.4f}")
print(f"  Delta: {delta:.4f}, Gamma: {gamma:.4f}, Vega: {vega:.4f}, Theta: {theta:.4f}, Rho: {rho:.4f}")
print(f"  Time taken: {end_time - start_time:.6f} seconds\n")

# ----------------------------
# Grid benchmarking
# ----------------------------
start_time = time.time()
spot_prices, volatilities, call_grid, put_grid = generate_price_grid(
    S=S,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    option_type=option_type,
    spot_pct_range=spot_pct_range,
    sigma_range=sigma_range,
    grid_size=grid_size
)
end_time = time.time()
grid_time = end_time - start_time
print(f"{grid_size}x{grid_size} grid generation time: {grid_time:.6f} seconds")

# ----------------------------
# Parity error check
# ----------------------------
parity_rhs = np.broadcast_to(
    spot_prices[None, :] - K * np.exp(-r * T),
    call_grid.shape
)
parity_error = np.abs(call_grid - put_grid - parity_rhs)
max_error = np.max(parity_error)
mean_error = np.mean(parity_error)

print(f"Maximum put-call parity error: {max_error:.2e}")
print(f"Mean put-call parity error: {mean_error:.2e}\n")

# ----------------------------
# Optional: stress test multiple runs
# ----------------------------
runs = 100
start_time = time.time()
for _ in range(runs):
    generate_price_grid(
        S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type,
        spot_pct_range=spot_pct_range, sigma_range=sigma_range, grid_size=grid_size
    )
end_time = time.time()
print(f"Average time over {runs} runs: {(end_time - start_time)/runs:.6f} seconds per grid")
