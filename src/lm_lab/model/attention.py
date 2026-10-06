"""
Causal self-attention implementation using pure NumPy.

Implements multi-head causal self-attention for autoregressive language modeling.
"""

import numpy as np
from typing import Optional


class CausalSelfAttention:
    """
    Multi-head causal self-attention.

    Implements scaled dot-product attention with causal masking to prevent
    attending to future tokens.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        d_model: Model dimensionality
        num_heads: Number of attention heads
        max_len: Maximum sequence length
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_len: int = 2048,
        dropout: float = 0.0
    ):
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_len = max_len
        self.dropout = dropout

        # Initialize weights for QKV projection
        # Xavier/Glorot uniform initialization
        limit = np.sqrt(6.0 / (d_model + 3 * d_model))
        self.qkv_weight = np.random.uniform(
            -limit, limit, (d_model, 3 * d_model)
        ).astype(np.float32)
        self.qkv_bias = np.zeros(3 * d_model, dtype=np.float32)

        # Initialize weights for output projection
        limit = np.sqrt(6.0 / (d_model + d_model))
        self.out_weight = np.random.uniform(
            -limit, limit, (d_model, d_model)
        ).astype(np.float32)
        self.out_bias = np.zeros(d_model, dtype=np.float32)

        # Create causal mask
        # mask[i, j] = 1 if i >= j (can attend), 0 if i < j (cannot attend)
        self.causal_mask = np.tril(np.ones((max_len, max_len), dtype=np.float32))
        self.causal_mask = self.causal_mask.reshape(1, 1, max_len, max_len)

        # Cache for backward pass
        self.cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass of causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            training: Whether in training mode (for dropout)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, f"Expected d_model={self.d_model}, got {d_model}"
        assert seq_len <= self.max_len, f"seq_len {seq_len} exceeds max_len {self.max_len}"

        # Cache input for backward
        self.cache['x'] = x
        self.cache['training'] = training

        # Project to Q, K, V: (batch, seq_len, d_model) @ (d_model, 3*d_model)
        qkv = x @ self.qkv_weight + self.qkv_bias  # (batch, seq_len, 3*d_model)
        self.cache['qkv'] = qkv

        # Split into Q, K, V and reshape for multi-head attention
        # (batch, seq_len, 3*d_model) -> (batch, seq_len, 3, num_heads, d_k)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.d_k)

        # (batch, seq_len, 3, num_heads, d_k) -> (3, batch, num_heads, seq_len, d_k)
        qkv = np.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (batch, num_heads, seq_len, d_k)
        self.cache['q'] = q
        self.cache['k'] = k
        self.cache['v'] = v

        # Scaled dot-product attention
        # scores = QK^T / sqrt(d_k)
        scores = (q @ np.transpose(k, (0, 1, 3, 2))) / np.sqrt(self.d_k)
        # scores: (batch, num_heads, seq_len, seq_len)
        self.cache['scores_pre_mask'] = scores

        # Apply causal mask
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = np.where(mask == 0, -1e9, scores)
        self.cache['scores_masked'] = scores

        # Softmax
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        # attn_weights: (batch, num_heads, seq_len, seq_len)

        # Handle NaN from numerical issues
        attn_weights = np.nan_to_num(attn_weights, nan=0.0)
        self.cache['attn_weights_pre_dropout'] = attn_weights

        # Apply dropout to attention weights
        if training and self.dropout > 0:
            dropout_mask = (np.random.rand(*attn_weights.shape) > self.dropout).astype(np.float32)
            attn_weights = attn_weights * dropout_mask / (1.0 - self.dropout)
            self.cache['dropout_mask'] = dropout_mask

        self.cache['attn_weights'] = attn_weights

        # Apply attention to values
        out = attn_weights @ v  # (batch, num_heads, seq_len, d_k)
        self.cache['attn_out'] = out

        # Concatenate heads
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
        out = np.transpose(out, (0, 2, 1, 3))
        # -> (batch, seq_len, d_model)
        out = out.reshape(batch_size, seq_len, d_model)
        self.cache['concat_out'] = out

        # Output projection
        out = out @ self.out_weight + self.out_bias
        self.cache['output'] = out

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass of causal self-attention.

        Args:
            dout: Gradient of loss w.r.t. output, shape (batch_size, seq_len, d_model)

        Returns:
            Gradient of loss w.r.t. input, shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = dout.shape

        # Gradient through output projection
        dout_flat = dout.reshape(-1, d_model)
        concat_out_flat = self.cache['concat_out'].reshape(-1, d_model)

        self.dout_weight = concat_out_flat.T @ dout_flat
        self.dout_bias = np.sum(dout_flat, axis=0)
        d_concat = dout_flat @ self.out_weight.T
        d_concat = d_concat.reshape(batch_size, seq_len, d_model)

        # Gradient through head concatenation
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        d_concat = d_concat.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        # -> (batch, num_heads, seq_len, d_k)
        d_attn_out = np.transpose(d_concat, (0, 2, 1, 3))

        # Gradient through attention application: out = attn_weights @ v
        attn_weights = self.cache['attn_weights']
        v = self.cache['v']

        d_attn_weights = d_attn_out @ np.transpose(v, (0, 1, 3, 2))
        d_v = np.transpose(attn_weights, (0, 1, 3, 2)) @ d_attn_out

        # Gradient through dropout
        if self.cache['training'] and self.dropout > 0:
            dropout_mask = self.cache['dropout_mask']
            d_attn_weights = d_attn_weights * dropout_mask / (1.0 - self.dropout)

        # Gradient through softmax
        attn_weights_pre = self.cache['attn_weights_pre_dropout']
        d_scores = attn_weights_pre * (
            d_attn_weights - np.sum(d_attn_weights * attn_weights_pre, axis=-1, keepdims=True)
        )

        # Gradient through causal mask (masked positions get zero gradient)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        d_scores = d_scores * mask

        # Gradient through scaled dot-product: scores = (q @ k.T) / sqrt(d_k)
        scale = 1.0 / np.sqrt(self.d_k)
        d_scores_scaled = d_scores * scale

        k = self.cache['k']
        q = self.cache['q']

        d_q = d_scores_scaled @ k
        d_k = np.transpose(d_scores_scaled, (0, 1, 3, 2)) @ q

        # Gradient through split and reshape
        # Stack gradients: (3, batch, num_heads, seq_len, d_k)
        d_qkv = np.stack([d_q, d_k, d_v], axis=0)
        # -> (batch, seq_len, 3, num_heads, d_k)
        d_qkv = np.transpose(d_qkv, (1, 3, 0, 2, 4))
        # -> (batch, seq_len, 3*d_model)
        d_qkv = d_qkv.reshape(batch_size, seq_len, 3 * self.d_model)

        # Gradient through QKV projection
        x = self.cache['x']
        x_flat = x.reshape(-1, self.d_model)
        d_qkv_flat = d_qkv.reshape(-1, 3 * self.d_model)

        self.dqkv_weight = x_flat.T @ d_qkv_flat
        self.dqkv_bias = np.sum(d_qkv_flat, axis=0)
        dx = d_qkv_flat @ self.qkv_weight.T
        dx = dx.reshape(batch_size, seq_len, self.d_model)

        return dx


# Alias for compatibility
MultiHeadAttention = CausalSelfAttention
