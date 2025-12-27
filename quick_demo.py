from bs.pricing import calculate_black_scholes_price
from bs.greeks import calculate_delta, calculate_gamma, calculate_vega, calculate_theta, calculate_rho

# Example inputs
S = 100
K = 100
T = 1
r = 0.05
sigma = 0.2

# Prices
call_price = calculate_black_scholes_price(S, K, T, r, sigma, "call")
put_price = calculate_black_scholes_price(S, K, T, r, sigma, "put")
print("Call price:", call_price)
print("Put price:", put_price)

# Greeks
print("Call Delta:", calculate_delta(S, K, T, r, sigma, "call"))
print("Put Delta:", calculate_delta(S, K, T, r, sigma, "put"))
print("Gamma:", calculate_gamma(S, K, T, r, sigma))
print("Vega:", calculate_vega(S, K, T, r, sigma))
print("Call Theta:", calculate_theta(S, K, T, r, sigma, "call"))
print("Put Theta:", calculate_theta(S, K, T, r, sigma, "put"))
print("Call Rho:", calculate_rho(S, K, T, r, sigma, "call"))
print("Put Rho:", calculate_rho(S, K, T, r, sigma, "put"))


