# tests/test_pricing.py
import pytest
import numpy as np
from bs.pricing import calculate_black_scholes_price

def test_basic_call_put_price():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    call_price = calculate_black_scholes_price(S, K, T, r, sigma, "call")
    put_price = calculate_black_scholes_price(S, K, T, r, sigma, "put")

    # Reference values from online calculator
    assert np.isclose(call_price, 10.45, atol=0.1)
    assert np.isclose(put_price, 5.57, atol=0.1)

def test_put_call_parity():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    call_price = calculate_black_scholes_price(S, K, T, r, sigma, "call")
    put_price = calculate_black_scholes_price(S, K, T, r, sigma, "put")

    # Check put-call parity: C - P ≈ S - K * exp(-r*T)
    parity = S - K * np.exp(-r*T)
    assert np.isclose(call_price - put_price, parity, atol=1e-4)

def test_edge_cases():
    # Stock price zero
    assert calculate_black_scholes_price(0, 100, 1, 0.05, 0.2, "call") == 0.0
    assert np.isclose(calculate_black_scholes_price(0, 100, 1, 0.05, 0.2, "put"), 95.1229, atol=1e-3)

    # Strike zero
    assert calculate_black_scholes_price(100, 0, 1, 0.05, 0.2, "call") == 100
    assert calculate_black_scholes_price(100, 0, 1, 0.05, 0.2, "put") == 0.0

    # Expired option
    assert calculate_black_scholes_price(100, 90, 0, 0.05, 0.2, "call") == 10
    assert calculate_black_scholes_price(100, 110, 0, 0.05, 0.2, "put") == 10

    # Zero volatility
    call_price = calculate_black_scholes_price(100, 100, 1, 0.05, 0, "call")
    put_price = calculate_black_scholes_price(100, 100, 1, 0.05, 0, "put")
    assert call_price >= 0
    assert put_price >= 0
