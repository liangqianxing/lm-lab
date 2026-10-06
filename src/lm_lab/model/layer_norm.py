"""
Layer normalization implementation using pure NumPy.

Normalizes inputs across the feature dimension for numerical stability
and improved training dynamics.
"""

import numpy as np


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
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)

        # Cache for backward pass
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of layer normalization.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized output of same shape
        """
        # Compute mean and variance along last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # Normalize
        x_normalized = (x - mean) / np.sqrt(var + self.eps)

        # Apply affine transformation
        output = self.gamma * x_normalized + self.beta

        # Cache for backward pass
        self.cache = (x, mean, var, x_normalized)

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of layer normalization.

        Args:
            grad_output: Gradient of loss w.r.t. output, shape (..., d_model)

        Returns:
            Gradient of loss w.r.t. input, same shape as input
        """
        if self.cache is None:
            raise RuntimeError("backward() called before forward()")

        x, mean, var, x_normalized = self.cache

        # Gradient w.r.t. gamma and beta
        self.grad_gamma = np.sum(grad_output * x_normalized, axis=tuple(range(grad_output.ndim - 1)))
        self.grad_beta = np.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))

        # Gradient w.r.t. normalized input
        grad_x_normalized = grad_output * self.gamma

        # Gradient w.r.t. variance
        std_inv = 1.0 / np.sqrt(var + self.eps)
        grad_var = np.sum(grad_x_normalized * (x - mean) * (-0.5) * std_inv**3, axis=-1, keepdims=True)

        # Gradient w.r.t. mean
        grad_mean = (
            np.sum(grad_x_normalized * (-std_inv), axis=-1, keepdims=True) +
            grad_var * np.mean(-2.0 * (x - mean), axis=-1, keepdims=True)
        )

        # Gradient w.r.t. input
        N = x.shape[-1]
        grad_x = (
            grad_x_normalized * std_inv +
            grad_var * 2.0 * (x - mean) / N +
            grad_mean / N
        )

        return grad_x
