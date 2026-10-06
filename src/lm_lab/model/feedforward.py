"""
Feed-forward network implementation for Transformer using NumPy.

Implements the position-wise feed-forward network with GELU activation.
"""

import numpy as np


class FeedForward:
    """
    Position-wise feed-forward network.

    Applies two linear transformations with GELU activation in between.

    Args:
        d_model: Model dimensionality
        d_ff: Hidden layer dimensionality (typically 4 * d_model)
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        # Initialize weights with Xavier/Glorot initialization
        self.w1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / (d_model + d_ff))
        self.b1 = np.zeros(d_ff)
        self.w2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / (d_ff + d_model))
        self.b2 = np.zeros(d_model)

        # Cache for backward pass
        self.cache = {}

    def gelu(self, x: np.ndarray) -> np.ndarray:
        """
        GELU activation function.

        GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function
        of the standard normal distribution.

        Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        """
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def gelu_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of GELU activation.
        """
        sqrt_2_pi = np.sqrt(2.0 / np.pi)
        tanh_arg = sqrt_2_pi * (x + 0.044715 * x**3)
        tanh_out = np.tanh(tanh_arg)
        sech2 = 1 - tanh_out**2

        dtanh = sqrt_2_pi * (1 + 0.044715 * 3 * x**2) * sech2
        return 0.5 * (1 + tanh_out) + 0.5 * x * dtanh

    def dropout_mask(self, shape: tuple, training: bool = True) -> np.ndarray:
        """
        Generate dropout mask.

        During training, randomly zero elements with probability dropout.
        During inference, return all ones.
        """
        if not training or self.dropout == 0.0:
            return np.ones(shape)
        mask = (np.random.rand(*shape) > self.dropout).astype(np.float32)
        return mask / (1.0 - self.dropout)  # Inverted dropout scaling

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            training: Whether in training mode (for dropout)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear transformation: (batch, seq_len, d_model) @ (d_model, d_ff)
        z1 = x @ self.w1 + self.b1

        # GELU activation
        a1 = self.gelu(z1)

        # Dropout
        dropout_mask = self.dropout_mask(a1.shape, training)
        a1_dropped = a1 * dropout_mask

        # Second linear transformation: (batch, seq_len, d_ff) @ (d_ff, d_model)
        z2 = a1_dropped @ self.w2 + self.b2

        # Cache for backward pass
        self.cache = {
            'x': x,
            'z1': z1,
            'a1': a1,
            'dropout_mask': dropout_mask,
            'a1_dropped': a1_dropped
        }

        return z2

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            dout: Gradient of loss w.r.t. output, shape (batch_size, seq_len, d_model)

        Returns:
            Gradient of loss w.r.t. input, shape (batch_size, seq_len, d_model)
        """
        x = self.cache['x']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        dropout_mask = self.cache['dropout_mask']
        a1_dropped = self.cache['a1_dropped']

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Gradient w.r.t. second linear layer
        # dout: (batch, seq_len, d_model)
        # a1_dropped: (batch, seq_len, d_ff)
        dw2 = np.einsum('bsi,bsj->ij', a1_dropped, dout) / batch_size
        db2 = dout.sum(axis=(0, 1)) / batch_size

        # Gradient w.r.t. a1_dropped
        da1_dropped = dout @ self.w2.T

        # Gradient through dropout
        da1 = da1_dropped * dropout_mask

        # Gradient through GELU
        dz1 = da1 * self.gelu_derivative(z1)

        # Gradient w.r.t. first linear layer
        dw1 = np.einsum('bsi,bsj->ij', x, dz1) / batch_size
        db1 = dz1.sum(axis=(0, 1)) / batch_size

        # Gradient w.r.t. input
        dx = dz1 @ self.w1.T

        # Store gradients for optimizer
        self.gradients = {
            'w1': dw1,
            'b1': db1,
            'w2': dw2,
            'b2': db2
        }

        return dx
