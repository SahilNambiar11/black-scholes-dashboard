"""
RL Hedging Module

Reinforcement Learning agent for option hedging.

Components:
- data_generator: Generate MC training data
- data_validator: Validate data quality
- classical_hedger: Baseline delta-hedging strategy
- agent: Neural network agent
- environment: Hedging simulation environment
- trainer: Training loop
"""

from .data_generator import generate_training_paths
from .data_validator import validate_training_data, print_validation_report

__all__ = [
    'generate_training_paths',
    'validate_training_data',
    'print_validation_report'
]
