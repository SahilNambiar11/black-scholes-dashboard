# tests/test_greeks.py
import pytest
import numpy as np
from bs.greeks import calculate_delta, calculate_gamma, calculate_vega, calculate_theta, calculate_rho

def test_delta_values():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    delta_call = calculate_delta(S, K, T, r, sigma, "call")
    delta_put = calculate_delta(S, K, T, r, sigma, "put")

    assert 0 < delta_call < 1
    assert -1 < delta_put < 0
    assert np.isclose(delta_call + delta_put, calculate_delta(S, K, T, r, sigma, "call") + calculate_delta(S, K, T, r, sigma, "put"), atol=1e-5)

def test_gamma_positive():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    gamma = calculate_gamma(S, K, T, r, sigma)
    assert gamma > 0

def test_vega_positive():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    vega = calculate_vega(S, K, T, r, sigma)
    assert vega > 0

def test_theta_signs():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    theta_call = calculate_theta(S, K, T, r, sigma, "call")
    theta_put = calculate_theta(S, K, T, r, sigma, "put")
    assert theta_call < 0
    assert theta_put < 0

def test_rho_signs():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    rho_call = calculate_rho(S, K, T, r, sigma, "call")
    rho_put = calculate_rho(S, K, T, r, sigma, "put")
    assert rho_call > 0
    assert rho_put < 0

def test_edge_cases_greeks():
    # Expired options (T=0)
    S, K, T, r, sigma = 100, 100, 0, 0.05, 0.2
    assert calculate_delta(S, K, T, r, sigma, "call") in [0.0, 1.0]
    assert calculate_delta(S, K, T, r, sigma, "put") in [-1.0, 0.0]
    assert calculate_gamma(S, K, T, r, sigma) == 0
    assert calculate_vega(S, K, T, r, sigma) == 0
    assert calculate_theta(S, K, T, r, sigma, "call") in [-1.0, 0.0]
    assert calculate_rho(S, K, T, r, sigma, "call") == 0
