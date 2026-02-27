#!/usr/bin/env python3
"""
Delta Approximator Model

PyTorch MLP to approximate Black-Scholes delta:
    delta = f(log_moneyness, time_to_maturity, volatility)

Input features:
    - log_moneyness
    - time_to_maturity
    - volatility

Output:
    - predicted delta (scalar)
"""

import torch
import torch.nn as nn


class DeltaNet(nn.Module):
    """
    Simple feedforward neural network for delta regression.
    """

    def __init__(self, input_dim=3, hidden_dim=64):
        super(DeltaNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1)  # Output layer
        )

    def forward(self, x):
        return self.model(x)


def get_model(device="cpu"):
    """
    Initialize model and move to device.
    """
    model = DeltaNet()
    return model.to(device)


if __name__ == "__main__":
    # Quick sanity check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(device)

    print("=" * 60)
    print("MODEL INITIALIZATION TEST")
    print("=" * 60)

    sample_input = torch.randn(5, 3).to(device)
    output = model(sample_input)

    print("Input shape:", sample_input.shape)
    print("Output shape:", output.shape)
    print("Sample output:", output.detach().cpu().numpy())