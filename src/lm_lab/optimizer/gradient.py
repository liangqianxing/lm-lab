"""
Gradient clipping utilities.

Provides functions to clip gradients by norm or value to prevent exploding gradients.
"""

import numpy as np
from typing import List
import math


def clip_grad_norm(
    gradients: List[np.ndarray],
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradients by global norm.

    Scales gradients so that their global norm does not exceed max_norm.

    Args:
        gradients: List of gradient arrays
        max_norm: Maximum allowed norm
        norm_type: Type of norm (default 2.0 for L2 norm)

    Returns:
        Total norm before clipping
    """
    if norm_type == float('inf'):
        # Infinity norm: max absolute value
        total_norm = max(np.abs(g).max() for g in gradients if g is not None)
    else:
        # L-p norm
        total_norm = 0.0
        for g in gradients:
            if g is not None:
                param_norm = np.linalg.norm(g.ravel(), ord=norm_type)
                total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)

    # Clip if needed
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in gradients:
            if g is not None:
                g *= clip_coef

    return total_norm


def clip_grad_value(
    gradients: List[np.ndarray],
    clip_value: float
) -> None:
    """
    Clip gradients by value.

    Clips each gradient element to [-clip_value, clip_value].

    Args:
        gradients: List of gradient arrays
        clip_value: Maximum absolute value
    """
    for g in gradients:
        if g is not None:
            np.clip(g, -clip_value, clip_value, out=g)
