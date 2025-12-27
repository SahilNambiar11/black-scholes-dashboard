# bs/greeks.py
import numpy as np
from bs.utils import calculate_d1, calculate_d2, calculate_normcdf, calculate_normpdf

def calculate_delta(S0, K, T, r, sigma, option_type):
    # Edge cases
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S0 > K else 0.0
        else:  # put
            return 0.0 if S0 > K else -1.0

    d1 = calculate_d1(S0, K, T, r, sigma)
    if option_type == "call":
        return calculate_normcdf(d1)
    elif option_type == "put":
        return calculate_normcdf(d1) - 1


def calculate_gamma(S0, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = calculate_d1(S0, K, T, r, sigma)
    return calculate_normpdf(d1) / (S0 * sigma * np.sqrt(T))


def calculate_vega(S0, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = calculate_d1(S0, K, T, r, sigma)
    return S0 * calculate_normpdf(d1) * np.sqrt(T)


def calculate_theta(S0, K, T, r, sigma, option_type):
    if T <= 0:
        # At expiry, option decays instantly
        if option_type == "call":
            return -1.0 if S0 > K else 0.0
        else:  # put
            return -1.0 if S0 < K else 0.0

    d1 = calculate_d1(S0, K, T, r, sigma)
    d2 = calculate_d2(d1, sigma, T)

    first_term = -S0 * calculate_normpdf(d1) * sigma / (2 * np.sqrt(T))

    if option_type == "call":
        return first_term - r * K * np.exp(-r*T) * calculate_normcdf(d2)
    elif option_type == "put":
        return first_term + r * K * np.exp(-r*T) * calculate_normcdf(-d2)


def calculate_rho(S0, K, T, r, sigma, option_type):
    if T <= 0:
        return 0.0

    d1 = calculate_d1(S0, K, T, r, sigma)
    d2 = calculate_d2(d1, sigma, T)

    if option_type == "call":
        return K * T * np.exp(-r*T) * calculate_normcdf(d2)
    elif option_type == "put":
        return -K * T * np.exp(-r*T) * calculate_normcdf(-d2)
