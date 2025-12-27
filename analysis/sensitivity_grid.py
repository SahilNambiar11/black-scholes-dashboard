# sensitivity_grid.py
import numpy as np
from bs.pricing import calculate_black_scholes_price

def generate_price_grid(
    S, K, T, r, sigma,
    option_type,
    spot_pct_range=(-0.5, 0.5),
    sigma_range=(0.1, 1.0),
    grid_size=40
):
    """
    Generates a grid of option prices for sensitivity analysis.

    Parameters:
        S (float): Base spot price
        K, T, r, sigma (float): Option parameters
        option_type (str): "call" or "put"
        spot_pct_range (tuple): min/max % change for spot
        sigma_range (tuple): min/max volatility
        grid_size (int): resolution of grid

    Returns:
        spot_prices: 1D array of spot prices used
        volatilities: 1D array of volatilities used
        price_grid: 2D array of option prices (vol × spot)
    """
    spot_pct = np.linspace(spot_pct_range[0], spot_pct_range[1], grid_size)
    volatilities = np.linspace(sigma_range[0], sigma_range[1], grid_size)

    spot_prices = S * (1 + spot_pct)
    S_grid, sigma_grid = np.meshgrid(spot_prices, volatilities)

    call_price_grid = calculate_black_scholes_price(S_grid, K, T, r, sigma_grid, "call")
    put_price_grid = calculate_black_scholes_price(S_grid, K, T, r, sigma_grid, "put")

    return spot_prices, volatilities, call_price_grid, put_price_grid
