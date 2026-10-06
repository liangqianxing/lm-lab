"""
Causal self-attention implementation using NumPy.

Implements multi-head causal self-attention for autoregressive language modeling.
"""

import numpy as np
import math
from typing import Tuple, Optional


class CausalSelfAttention:
    """
    Multi-head causal self-attention.

    Implements scaled dot-product attention with causal masking to prevent
    attending to future tokens.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        d_model: Model dimensionality
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        dropout: Dropout rate (not implemented in basic version)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0
    ):
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        # Initialize weights for Q, K, V projections (combined)
        scale = 1.0 / math.sqrt(d_model)
        self.W_qkv = np.random.randn(d_model, 3 * d_model) * scale
        self.b_qkv = np.zeros(3 * d_model)

        # Output projection
        self.W_out = np.random.randn(d_model, d_model) * scale
        self.b_out = np.zeros(d_model)

        # Create causal mask (lower triangular)
        # mask[i, j] = 1 if i >= j (can attend), 0 if i < j (cannot attend)
        self.causal_mask = np.tril(np.ones((max_seq_len, max_seq_len)))

        # Cache for backward pass
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model
        assert seq_len <= self.max_seq_len

        # Project to Q, K, V
        qkv = x @ self.W_qkv + self.b_qkv  # (batch, seq_len, 3*d_model)

        # Split into Q, K, V
        q, k, v = np.split(qkv, 3, axis=-1)  # Each: (batch, seq_len, d_model)

        # Reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        # scores = QK^T / sqrt(d_k)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        # scores: (batch, num_heads, seq_len, seq_len)

        # Apply causal mask
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = np.where(mask == 1, scores, -1e10)

        # Softmax
        # Numerically stable softmax
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        # attn_weights: (batch, num_heads, seq_len, seq_len)

        # Replace NaN with 0 (can happen with all -inf in a row)
        attn_weights = np.where(np.isnan(attn_weights), 0, attn_weights)

        # Apply attention to values
        out = attn_weights @ v  # (batch, num_heads, seq_len, d_k)

        # Concatenate heads
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

        # Output projection
        out = out @ self.W_out + self.b_out  # (batch, seq_len, d_model)

        # Cache for backward
        self.cache = {
            'x': x,
            'q': q,
            'k': k,
            'v': v,
            'attn_weights': attn_weights,
            'out_pre_proj': out.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3),
        }

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of causal self-attention.

        Args:
            grad_output: Gradient of loss w.r.t. output (batch, seq_len, d_model)

        Returns:
            Gradient w.r.t. input (batch, seq_len, d_model)
        """
        x = self.cache['x']
        q = self.cache['q']
        k = self.cache['k']
        v = self.cache['v']
        attn_weights = self.cache['attn_weights']
        out_pre_proj = self.cache['out_pre_proj']

        batch_size, seq_len, d_model = grad_output.shape

        # Gradient of output projection
        grad_out_pre_proj = grad_output @ self.W_out.T
        self.grad_W_out = grad_output.reshape(-1, d_model).T @ out_pre_proj.transpose(0, 2, 1, 3).reshape(-1, d_model)
        self.grad_b_out = grad_output.sum(axis=(0, 1))

        # Reshape gradient back to multi-head format
        grad_out_pre_proj = grad_out_pre_proj.reshape(
            batch_size, seq_len, self.num_heads, self.d_k
        ).transpose(0, 2, 1, 3)

        # Gradient through attention application (attn_weights @ v)
        grad_attn_weights = grad_out_pre_proj @ v.transpose(0, 1, 3, 2)
        grad_v = attn_weights.transpose(0, 1, 3, 2) @ grad_out_pre_proj

        # Gradient through softmax
        grad_scores = attn_weights * (
            grad_attn_weights - np.sum(grad_attn_weights * attn_weights, axis=-1, keepdims=True)
        )

        # Apply causal mask to gradient
        mask = self.causal_mask[:seq_len, :seq_len]
        grad_scores = np.where(mask == 1, grad_scores, 0)

        # Gradient through scaling
        grad_scores = grad_scores / math.sqrt(self.d_k)

        # Gradient through QK^T
        grad_q = grad_scores @ k
        grad_k = grad_scores.transpose(0, 1, 3, 2) @ q

        # Reshape gradients back to (batch, seq_len, d_model)
        grad_q = grad_q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        grad_k = grad_k.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        grad_v = grad_v.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

        # Concatenate Q, K, V gradients
        grad_qkv = np.concatenate([grad_q, grad_k, grad_v], axis=-1)

        # Gradient through QKV projection
        grad_x = grad_qkv @ self.W_qkv.T
        self.grad_W_qkv = x.reshape(-1, d_model).T @ grad_qkv.reshape(-1, 3 * d_model)
        self.grad_b_qkv = grad_qkv.sum(axis=(0, 1))

        return grad_x

    def parameters(self):
        """Return trainable parameters."""
        return [self.W_qkv, self.b_qkv, self.W_out, self.b_out]

    def gradients(self):
        """Return parameter gradients."""
        return [self.grad_W_qkv, self.grad_b_qkv, self.grad_W_out, self.grad_b_out]
