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

def monte_carlo_pricing(S0, r, sigma, T, strikes, N, M):
    """
    Monte Carlo option pricing using geometric Brownian motion.
    
    Parameters
    ----------
    S0 : float
        Initial stock price
    r : float
        Risk-free rate (decimal)
    sigma : float
        Volatility (decimal)
    T : float
        Time to maturity (years)
    strikes : array-like
        Array of strike prices
    N : int
        Number of time steps
    M : int
        Number of simulations (Monte Carlo paths)
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'strikes': input strikes
        - 'prices': array of European call option prices for each strike
        - 'std_errors': standard errors of the estimates
        - 'paths': simulated stock price paths (M x N+1)
        - 'final_prices': final stock prices from all M simulations (M,)
    """
    strikes = np.asarray(strikes, dtype=float)
    
    # Time step
    dt = T / N
    
    # Generate M paths with N+1 time points (including t=0)
    # Shape: (M, N+1)
    dW = np.random.standard_normal(size=(M, N))  # (M, N) random increments
    
    # Initialize paths at t=0
    paths = np.zeros((M, N + 1))
    paths[:, 0] = S0
    
    # Simulate geometric Brownian motion: dS = r*S*dt + sigma*S*dW
    for t in range(N):
        paths[:, t + 1] = paths[:, t] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dW[:, t])
    
    # Final stock prices at maturity
    final_prices = paths[:, -1]  # Shape: (M,)
    
    # Compute European call option prices at maturity: max(S_T - K, 0)
    # Shape: (len(strikes), M)
    payoffs = np.maximum(final_prices[np.newaxis, :] - strikes[:, np.newaxis], 0)
    
    # Discount back to present value
    option_prices = np.exp(-r * T) * np.mean(payoffs, axis=1)  # Shape: (len(strikes),)
    
    # Compute standard errors
    std_errors = np.std(payoffs, axis=1) / np.sqrt(M) * np.exp(-r * T)
    
    return {
        'strikes': strikes,
        'prices': option_prices,
        'std_errors': std_errors,
        'paths': paths,
        'final_prices': final_prices
    }
