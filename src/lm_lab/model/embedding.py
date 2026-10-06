"""
Position encoding implementations for Transformer models using NumPy.

Provides both learned and sinusoidal positional encodings.
"""

import math
import numpy as np


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
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Store as instance attribute (not trainable)
        self.pe = pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input.

        Args:
            x: Input array of shape (batch_size, seq_len, d_model)

        Returns:
            Input with positional encoding added
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for positional encoding.

        Since positional encoding is just addition, gradient passes through unchanged.

        Args:
            grad_output: Gradient from downstream, shape (batch_size, seq_len, d_model)

        Returns:
            Gradient with respect to input
        """
        return grad_output


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

        # Learnable position embeddings (initialized with small random values)
        self.position_embeddings = np.random.randn(max_len, d_model).astype(np.float32) * 0.02

        # Cache for backward pass
        self.last_seq_len = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add learned positional encoding to input.

        Args:
            x: Input array of shape (batch_size, seq_len, d_model)

        Returns:
            Input with positional encoding added
        """
        seq_len = x.shape[1]
        self.last_seq_len = seq_len
        return x + self.position_embeddings[:seq_len, :]

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Backward pass for learned positional encoding.

        Args:
            grad_output: Gradient from downstream, shape (batch_size, seq_len, d_model)

        Returns:
            Tuple of (gradient w.r.t. input, gradient w.r.t. position_embeddings)
        """
        # Gradient w.r.t. input passes through unchanged
        grad_input = grad_output

        # Gradient w.r.t. position embeddings
        # Sum over batch dimension since position embeddings are shared
        grad_position_embeddings = np.zeros_like(self.position_embeddings)
        grad_position_embeddings[:self.last_seq_len, :] = np.sum(grad_output, axis=0)

        return grad_input, grad_position_embeddings
