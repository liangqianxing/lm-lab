"""
Position encoding implementations for Transformer models.

Provides both learned and sinusoidal positional encodings.
"""

import math
import numpy as np
from typing import Optional


class PositionalEncoding:
    """
    Sinusoidal positional encoding.

    Uses sine and cosine functions of different frequencies to encode
    position information, as described in "Attention Is All You Need".

    Args:
        d_model: Model dimensionality
        max_len: Maximum sequence length
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.max_len = max_len

        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Input with positional encoding added
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass (gradient flows through unchanged).

        Args:
            grad_output: Gradient from upstream

        Returns:
            Gradient w.r.t. input
        """
        return grad_output

    def parameters(self):
        """No trainable parameters."""
        return []

    def gradients(self):
        """No gradients."""
        return []


class LearnedPositionalEncoding:
    """
    Learned positional embeddings.

    Uses trainable embeddings for each position.

    Args:
        d_model: Model dimensionality
        max_len: Maximum sequence length
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.max_len = max_len

        # Initialize learned position embeddings
        scale = 1.0 / math.sqrt(d_model)
        self.position_embeddings = np.random.randn(max_len, d_model) * scale

        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add learned positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Input with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape

        # Get position embeddings for this sequence
        pos_emb = self.position_embeddings[:seq_len, :]

        # Cache for backward
        self.cache = {'seq_len': seq_len}

        return x + pos_emb

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad_output: Gradient from upstream (batch_size, seq_len, d_model)

        Returns:
            Gradient w.r.t. input
        """
        seq_len = self.cache['seq_len']

        # Gradient w.r.t. position embeddings
        self.grad_position_embeddings = np.zeros_like(self.position_embeddings)
        self.grad_position_embeddings[:seq_len, :] = grad_output.sum(axis=0)

        # Gradient w.r.t. input (flows through unchanged)
        return grad_output

    def parameters(self):
        """Return trainable parameters."""
        return [self.position_embeddings]

    def gradients(self):
        """Return parameter gradients."""
        return [self.grad_position_embeddings]
