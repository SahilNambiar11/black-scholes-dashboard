# Black_Scholes Project

Integrated options research project covering analytics, simulation, and ML-based delta approximation.

## Scope

This repository currently combines three tracks:

1. Analytical Black-Scholes pricing + Greeks
2. Monte Carlo option pricing and diagnostics
3. ML delta approximation (PyTorch) integrated into the dashboard

The app serves as a single interface for all three.

## Project Architecture

```text
.
├── app.py                          # Streamlit dashboard (3 pages)
├── requirements.txt
├── README.md
├── analysis/
│   └── sensitivity_grid.py         # Spot/vol grids for heatmaps
├── bs/
│   ├── pricing.py                  # Black-Scholes option pricing
│   ├── greeks.py                   # Delta/Gamma/Vega/Theta/Rho
│   └── utils.py                    # Shared BS + MC utilities
├── rl_hedging/
│   ├── __init__.py                 # Package exports
│   ├── data_generator.py           # Supervised ML dataset generation
│   ├── model.py                    # DeltaNet architecture
│   └── train.py                    # Training pipeline
├── tests/
│   ├── test_pricing.py             # Pricing tests
│   └── test_greeks.py              # Greeks tests
└── Artifacts (generated)
    ├── delta_dataset.csv
    ├── delta_model.pt
    └── scaler.pkl
```

## End-to-End Flow

1. Use `bs/` functions for analytical pricing and Greeks.
2. Generate synthetic supervised samples in `rl_hedging/data_generator.py`.
3. Train `DeltaNet` in `rl_hedging/train.py`.
4. Save model + scaler artifacts.
5. Load artifacts in `app.py` and compare ML delta vs analytical delta visually.

## Dashboard Pages

Run:

```bash
streamlit run app.py
```

Pages in the sidebar:

1. `Black–Scholes`
   - Single-point pricing
   - Greeks panel
   - Spot/vol sensitivity heatmap
   - Put-call parity diagnostics

2. `Monte Carlo`
   - GBM simulation-driven option pricing
   - Confidence intervals and BS comparison
   - Final price distribution charts

3. `RL Hedging (ML)`
   - Loads `delta_model.pt` and `scaler.pkl`
   - Computes ML delta from `[log(S/K), T, sigma]`
   - Compares ML vs BS deltas
   - Shows error metrics (MAE, RMSE, max error)

## ML Pipeline

### Data Generation

```bash
python3 rl_hedging/data_generator.py
```

Output:

- `delta_dataset.csv`

Features/target:

- `log_moneyness = log(S/K)`
- `time_to_maturity`
- `volatility`
- `delta` (analytical call delta)

### Training

```bash
python3 rl_hedging/train.py
```

Pipeline details:

- train/val/test split (60/20/20)
- feature standardization (`StandardScaler`)
- mini-batch training with `DataLoader`
- MSE loss with Adam optimizer

Saved artifacts:

- `delta_model.pt`
- `scaler.pkl`

## Installation

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Testing

```bash
pytest -q
```

## Notes

- Current ML model is trained on call delta.
- Put delta displayed in app is computed as:
  - `put_delta = call_delta - 1`
- If ML artifacts are missing, run training before opening the ML page in Streamlit.
