# bs/utils.py
import numpy as np
from scipy.stats import norm

def calculate_d1(S0, K, T, r, sigma):
    """
    Vectorized calculation of d1 for Black-Scholes formula.
    
    S0: Spot price (scalar or array)
    K: Strike price (scalar)
    T: Time to maturity in years (scalar or array)
    r: Risk-free rate (scalar)
    sigma: Volatility (scalar or array)
    """
    S0 = np.array(S0, dtype=float)
    T = np.array(T, dtype=float)
    sigma = np.array(sigma, dtype=float)

    # Prevent division by zero
    sigma_safe = np.where(sigma <= 0, 1e-10, sigma)
    T_safe = np.where(T <= 0, 1e-10, T)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * np.sqrt(T_safe))
    return d1

def calculate_d2(d1, sigma, T):
    """
    Vectorized calculation of d2 for Black-Scholes formula.
    
    d1: Already calculated d1 (scalar or array)
    sigma: Volatility (scalar or array)
    T: Time to maturity in years (scalar or array)
    """
    sigma = np.array(sigma, dtype=float)
    T = np.array(T, dtype=float)

    # Prevent invalid sqrt
    sigma_safe = np.where(sigma <= 0, 1e-10, sigma)
    T_safe = np.where(T <= 0, 1e-10, T)

    d2 = d1 - sigma_safe * np.sqrt(T_safe)
    return d2

def calculate_normcdf(x):
    """
    Normal cumulative distribution function
    """
    return norm.cdf(x)

def calculate_normpdf(x):
    """
    Normal probability density function
    """
    return norm.pdf(x)
