"""
Layer normalization implementation.

Normalizes inputs across the feature dimension for numerical stability
and improved training dynamics.
"""

import numpy as np
from typing import Tuple


class LayerNorm:
    """
    Layer normalization.

    Normalizes the input across the feature dimension and applies
    learned affine transformation.

    Args:
        d_model: Model dimensionality
        eps: Small constant for numerical stability
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of layer normalization.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized output of same shape
        """
        # Compute mean and variance along last dimension
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)

        # Normalize
        x_normalized = (x - mean) / np.sqrt(var + self.eps)

        # Apply affine transformation
        output = self.gamma * x_normalized + self.beta

        # Cache for backward
        self.cache = {
            'x': x,
            'mean': mean,
            'var': var,
            'x_normalized': x_normalized
        }

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of layer normalization.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient w.r.t. input
        """
        x = self.cache['x']
        mean = self.cache['mean']
        var = self.cache['var']
        x_normalized = self.cache['x_normalized']

        # Gradient of affine parameters
        self.grad_gamma = (grad_output * x_normalized).sum(axis=tuple(range(grad_output.ndim - 1)))
        self.grad_beta = grad_output.sum(axis=tuple(range(grad_output.ndim - 1)))

        # Gradient w.r.t. normalized input
        grad_x_normalized = grad_output * self.gamma

        # Gradient w.r.t. variance
        std = np.sqrt(var + self.eps)
        N = x.shape[-1]

        # Gradient through normalization
        grad_var = (grad_x_normalized * (x - mean) * -0.5 * (var + self.eps) ** (-1.5)).sum(
            axis=-1, keepdims=True
        )

        # Gradient through mean
        grad_mean = (grad_x_normalized * -1.0 / std).sum(axis=-1, keepdims=True)
        grad_mean += grad_var * (-2.0 / N) * (x - mean).sum(axis=-1, keepdims=True)

        # Gradient w.r.t. input
        grad_x = grad_x_normalized / std
        grad_x += grad_var * 2.0 * (x - mean) / N
        grad_x += grad_mean / N

        return grad_x

    def parameters(self):
        """Return trainable parameters."""
        return [self.gamma, self.beta]

    def gradients(self):
        """Return parameter gradients."""
        return [self.grad_gamma, self.grad_beta]
