"""
RL Hedging Module

Reinforcement Learning agent for option hedging.

Components:
- data_generator: Generate supervised delta training data
- data_validator: Validate data quality
- classical_hedger: Baseline delta-hedging strategy
- agent: Neural network agent
- environment: Hedging simulation environment
- trainer: Training loop
"""

__all__ = ["generate_delta_dataset"]

def generate_delta_dataset(*args, **kwargs):
    from .data_generator import generate_delta_dataset as _generate_delta_dataset

    return _generate_delta_dataset(*args, **kwargs)


# Optional exports: only available when validator module exists.
def validate_training_data(*args, **kwargs):
    from .data_validator import validate_training_data as _validate_training_data

    return _validate_training_data(*args, **kwargs)


def print_validation_report(*args, **kwargs):
    from .data_validator import print_validation_report as _print_validation_report

    return _print_validation_report(*args, **kwargs)


try:
    from . import data_validator as _data_validator  # noqa: F401
except ImportError:
    pass
else:
    __all__.extend(["validate_training_data", "print_validation_report"])
