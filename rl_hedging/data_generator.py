import numpy as np
import pandas as pd

try:
    from bs.greeks import calculate_delta
except ModuleNotFoundError:
    # Allow direct script execution: `python3 rl_hedging/data_generator.py`
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from bs.greeks import calculate_delta


def generate_delta_dataset(
    n_samples=100000,
    log_moneyness_range=(-0.5, 0.5),
    maturity_range=(0.01, 1.0),
    volatility_range=(0.1, 0.5),
    r=0.05,
    K=100,
    seed=42
):
    """
    Generate supervised dataset for delta approximation.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    log_moneyness_range : tuple
        Range for log(S/K)
    maturity_range : tuple
        Range for time to maturity (years)
    volatility_range : tuple
        Range for volatility (sigma)
    r : float
        Risk-free rate
    K : float
        Strike price
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Dataset with features and target delta
    """

    np.random.seed(seed)

    # Sample independent features
    log_moneyness = np.random.uniform(
        log_moneyness_range[0],
        log_moneyness_range[1],
        n_samples
    )

    time_to_maturity = np.random.uniform(
        maturity_range[0],
        maturity_range[1],
        n_samples
    )

    volatility = np.random.uniform(
        volatility_range[0],
        volatility_range[1],
        n_samples
    )

    # Convert log-moneyness back to spot price
    spot_prices = K * np.exp(log_moneyness)

    deltas = []

    for i in range(n_samples):
        S = spot_prices[i]
        T = time_to_maturity[i]
        sigma = volatility[i]

        try:
            delta = calculate_delta(S, K, T, r, sigma, "call")
        except:
            # Handle near-zero maturity edge cases
            delta = 1.0 if S > K else 0.0

        deltas.append(delta)

    df = pd.DataFrame({
        "log_moneyness": log_moneyness,
        "time_to_maturity": time_to_maturity,
        "volatility": volatility,
        "delta": deltas
    })

    return df


if __name__ == "__main__":

    print("\n" + "="*70)
    print("SUPERVISED DELTA DATASET GENERATOR")
    print("="*70)

    df = generate_delta_dataset(n_samples=50000)

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nSummary statistics:")
    print(df.describe())

    # Optional: Save to CSV
    df.to_csv("delta_dataset.csv", index=False)
    print("\nDataset saved to delta_dataset.csv")
    print("="*70)
