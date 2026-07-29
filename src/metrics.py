"""Fidelity metrics."""

import numpy as np


def calculate_mse(original, restored):
    """Mean Squared Error between the original and reconstructed images.

    Inputs are cast to float64 before subtraction to avoid uint16 wrap-around
    when computing the difference.
    """
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)
    return np.mean((original - restored) ** 2)
