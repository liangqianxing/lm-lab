"""
Numerical stability utilities for deep learning.

Provides functions to ensure numerical stability during training:
- Check for NaN/Inf values
- Numerically stable softmax and log-softmax
- Safe division and logarithm
"""

import math
from typing import Any

import torch
import torch.nn.functional as F


def check_finite(tensor: torch.Tensor, name: str = "tensor") -> None:
    """
    Check if tensor contains only finite values (no NaN or Inf).

    Args:
        tensor: Tensor to check
        name: Name for error message

    Raises:
        ValueError: If tensor contains NaN or Inf
    """
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        raise ValueError(
            f"{name} contains non-finite values: "
            f"{nan_count} NaN, {inf_count} Inf"
        )


def safe_softmax(
    x: torch.Tensor,
    dim: int = -1,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Numerically stable softmax.

    Subtracts max before exp to avoid overflow.

    Args:
        x: Input tensor
        dim: Dimension to apply softmax
        temperature: Temperature scaling (default 1.0)

    Returns:
        Softmax probabilities
    """
    if temperature != 1.0:
        x = x / temperature

    # Subtract max for numerical stability
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_shifted = x - x_max

    return F.softmax(x_shifted, dim=dim)


def stable_log_softmax(
    x: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Numerically stable log-softmax.

    Uses PyTorch's built-in implementation which is more stable
    than log(softmax(x)).

    Args:
        x: Input tensor
        dim: Dimension to apply log-softmax

    Returns:
        Log probabilities
    """
    return F.log_softmax(x, dim=dim)


def safe_log(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Safe logarithm that avoids log(0).

    Args:
        x: Input tensor
        eps: Small constant to add

    Returns:
        log(x + eps)
    """
    return torch.log(x + eps)


def safe_div(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Safe division that avoids division by zero.

    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        eps: Small constant to add to denominator

    Returns:
        numerator / (denominator + eps)
    """
    return numerator / (denominator + eps)


def check_gradients(
    parameters: Any,
    max_norm: float = 1000.0
) -> dict[str, float]:
    """
    Check gradient statistics for debugging.

    Args:
        parameters: Model parameters (iterable)
        max_norm: Maximum acceptable gradient norm

    Returns:
        Dictionary with gradient statistics
    """
    total_norm = 0.0
    num_params = 0
    num_nan = 0
    num_inf = 0

    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            num_params += 1

            if torch.isnan(p.grad).any():
                num_nan += 1
            if torch.isinf(p.grad).any():
                num_inf += 1

    total_norm = math.sqrt(total_norm)

    stats = {
        'total_norm': total_norm,
        'num_params': num_params,
        'num_nan': num_nan,
        'num_inf': num_inf,
        'is_stable': total_norm < max_norm and num_nan == 0 and num_inf == 0,
    }

    return stats
