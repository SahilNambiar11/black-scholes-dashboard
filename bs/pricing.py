# bs/pricing.py
import numpy as np
from scipy.stats import norm


def calculate_black_scholes_price(S, K, T, r, sigma, option_type):
    """
    Vectorized Black–Scholes option pricing formula.

    Parameters
    ----------
    S : float or np.ndarray
        Spot price(s)
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate (decimal)
    sigma : float or np.ndarray
        Volatility (decimal)
    option_type : str
        "call" or "put"
    """

    # Convert inputs to numpy arrays (enables vectorization)
    S = np.asarray(S, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Numerical safety (avoid division by zero)
    sigma = np.maximum(sigma, 1e-10)
    T = max(T, 1e-10)

    # Black–Scholes d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Pricing
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price
