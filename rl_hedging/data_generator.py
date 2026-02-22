#!/usr/bin/env python3
"""
RL Hedging: Training Data Generator

Generates structured Monte Carlo training data for RL agent.
Each path is decomposed into step-by-step states, actions, and rewards.

Key Output:
    DataFrame with columns:
    - path_id: Which MC path (0 to num_paths-1)
    - step: Time step (0 to num_steps-1)
    - spot_price: Stock price at this step
    - time_remaining: Years to maturity
    - bs_delta: Black-Scholes delta (optimal hedge ratio)
    - option_value: BS call option value at this step
    - payoff_at_maturity: Terminal payoff if held to expiry
    - next_spot_price: Stock price next step (for reward calculation)
"""

import numpy as np
import pandas as pd
from bs.pricing import calculate_black_scholes_price
from bs.greeks import calculate_delta


def generate_training_paths(S0, K, T, r, sigma, num_paths=1000, num_steps=20):
    """
    Generate Monte Carlo paths decomposed into step-by-step training data.
    
    Parameters
    ----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate
    sigma : float
        Volatility
    num_paths : int
        Number of MC paths to generate (default: 1000)
    num_steps : int
        Number of time steps per path (default: 20)
    
    Returns
    -------
    pd.DataFrame
        Training data with one row per (path, step) combination
        Columns: [path_id, step, spot_price, time_remaining, bs_delta, 
                  option_value, payoff_at_maturity, next_spot_price]
    """
    
    dt = T / num_steps
    data = []
    
    # Generate all paths
    np.random.seed(42)  # For reproducibility
    
    for path_id in range(num_paths):
        # Initialize spot price
        spot = S0
        
        # Generate path: num_steps+1 prices (including maturity)
        path = [spot]
        
        for step in range(num_steps):
            # GBM: dS = r*S*dt + sigma*S*sqrt(dt)*Z
            Z = np.random.standard_normal()
            dS = r * spot * dt + sigma * spot * np.sqrt(dt) * Z
            spot = spot + dS
            spot = max(spot, 0.01)  # Prevent negative prices
            path.append(spot)
        
        # Extract step-by-step data
        for step in range(num_steps):
            current_spot = path[step]
            next_spot = path[step + 1]
            time_left = T - (step * dt)
            time_left_next = T - ((step + 1) * dt)
            
            # Calculate BS delta and option value at this step
            try:
                bs_delta = calculate_delta(current_spot, K, time_left, r, sigma, "call")
                option_value = calculate_black_scholes_price(current_spot, K, time_left, r, sigma, "call")
            except:
                # Handle edge cases (very small time remaining)
                if current_spot > K:
                    bs_delta = 1.0
                    option_value = current_spot - K * np.exp(-r * time_left)
                else:
                    bs_delta = 0.0
                    option_value = 0.0
            
            # Payoff at maturity
            payoff_at_maturity = max(path[-1] - K, 0)
            
            data.append({
                'path_id': path_id,
                'step': step,
                'spot_price': current_spot,
                'time_remaining': time_left,
                'bs_delta': bs_delta,
                'option_value': option_value,
                'payoff_at_maturity': payoff_at_maturity,
                'next_spot_price': next_spot
            })
    
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # Test: Generate small dataset
    print("\n" + "="*70)
    print("PHASE 1A: DATA GENERATOR TEST")
    print("="*70)
    
    S0, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.20
    
    print(f"\nGenerating training data:")
    print(f"  S0=${S0}, K=${K}, T={T}y, r={r}, σ={sigma}")
    print(f"  Paths: 100, Steps: 20")
    
    df = generate_training_paths(S0, K, T, r, sigma, num_paths=100, num_steps=20)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\nFirst 10 rows:")
    print(df.head(10).to_string())
    
    print("\nLast 10 rows (near maturity):")
    print(df.tail(10).to_string())
    
    print("\nData Summary:")
    print(df[['spot_price', 'time_remaining', 'bs_delta', 'option_value']].describe())
    
    print("\n" + "="*70)
