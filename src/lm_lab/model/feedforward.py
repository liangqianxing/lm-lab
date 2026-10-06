"""
Feed-forward network implementation for Transformer.

Implements the position-wise feed-forward network with GELU activation.
"""

import numpy as np
import math
from typing import Tuple


class FeedForward:
    """
    Position-wise feed-forward network.

    Applies two linear transformations with GELU activation in between.

    Args:
        d_model: Model dimensionality
        d_ff: Hidden layer dimensionality (typically 4 * d_model)
        dropout: Dropout probability (not implemented in basic version)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        # Initialize weights
        scale1 = 1.0 / math.sqrt(d_model)
        scale2 = 1.0 / math.sqrt(d_ff)

        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear transformation
        hidden = x @ self.W1 + self.b1

        # GELU activation
        hidden_activated = self._gelu(hidden)

        # Second linear transformation
        output = hidden_activated @ self.W2 + self.b2

        # Cache for backward
        self.cache = {
            'x': x,
            'hidden': hidden,
            'hidden_activated': hidden_activated
        }

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad_output: Gradient of loss w.r.t. output

        Returns:
            Gradient w.r.t. input
        """
        x = self.cache['x']
        hidden = self.cache['hidden']
        hidden_activated = self.cache['hidden_activated']

        # Gradient of second linear layer
        grad_hidden_activated = grad_output @ self.W2.T
        self.grad_W2 = hidden_activated.reshape(-1, self.d_ff).T @ grad_output.reshape(-1, self.d_model)
        self.grad_b2 = grad_output.sum(axis=(0, 1))

        # Gradient through GELU
        grad_hidden = grad_hidden_activated * self._gelu_backward(hidden)

        # Gradient of first linear layer
        grad_x = grad_hidden @ self.W1.T
        self.grad_W1 = x.reshape(-1, self.d_model).T @ grad_hidden.reshape(-1, self.d_ff)
        self.grad_b1 = grad_hidden.sum(axis=(0, 1))

        return grad_x

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        """
        GELU activation function.

        GELU(x) = x * Φ(x) where Φ is the cumulative distribution function
        of the standard normal distribution.

        Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        """
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

    @staticmethod
    def _gelu_backward(x: np.ndarray) -> np.ndarray:
        """Gradient of GELU activation."""
        sqrt_2_pi = math.sqrt(2.0 / math.pi)
        x_cubed = x ** 3
        tanh_arg = sqrt_2_pi * (x + 0.044715 * x_cubed)
        tanh_val = np.tanh(tanh_arg)
        sech2_val = 1 - tanh_val ** 2

        grad = 0.5 * (1.0 + tanh_val)
        grad += 0.5 * x * sech2_val * sqrt_2_pi * (1 + 3 * 0.044715 * x ** 2)

        return grad

    def parameters(self):
        """Return trainable parameters."""
        return [self.W1, self.b1, self.W2, self.b2]

    def gradients(self):
        """Return parameter gradients."""
        return [self.grad_W1, self.grad_b1, self.grad_W2, self.grad_b2]
