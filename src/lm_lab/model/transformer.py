"""
Complete Transformer block and language model implementation using pure NumPy.
"""

import numpy as np
from typing import Optional
from .attention import CausalSelfAttention
from .layer_norm import LayerNorm
from .feedforward import FeedForward
from .embedding import LearnedPositionalEncoding


class TransformerBlock:
    """
    Single Transformer decoder block with:
    - Causal self-attention
    - Layer normalization (pre-norm architecture)
    - Position-wise feed-forward network
    - Residual connections

    Args:
        d_model: Model dimensionality
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension (default: 4 * d_model)
        max_len: Maximum sequence length
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        max_len: int = 2048,
        dropout: float = 0.0
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        if d_ff is None:
            d_ff = 4 * d_model

        # Pre-norm architecture
        self.ln1 = LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads, max_len, dropout)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # Gradient caches for backward pass
        self.cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass with pre-norm and residual connections.

        Args:
            x: Input array of shape (batch_size, seq_len, d_model)
            training: Whether in training mode (affects dropout)

        Returns:
            Output array of same shape
        """
        # Self-attention block with residual
        # x: (batch, seq_len, d_model)
        attn_input = self.ln1.forward(x)
        attn_output = self.attention.forward(attn_input, training=training)
        x = x + attn_output

        # Feed-forward block with residual
        ffn_input = self.ln2.forward(x)
        ffn_output = self.ffn.forward(ffn_input, training=training)
        x = x + ffn_output

        # Cache for backward pass
        self.cache['attn_input'] = attn_input
        self.cache['attn_output'] = attn_output
        self.cache['ffn_input'] = ffn_input
        self.cache['ffn_output'] = ffn_output

        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass through the transformer block.

        Args:
            dout: Gradient of loss w.r.t. output, shape (batch_size, seq_len, d_model)

        Returns:
            Gradient of loss w.r.t. input, same shape as dout
        """
        # Backprop through second residual connection
        dffn_output = dout
        dx2 = dout

        # Backprop through FFN
        dffn_input = self.ffn.backward(dffn_output)

        # Backprop through second layer norm
        dx2_ln = self.ln2.backward(dffn_input)
        dx2 = dx2 + dx2_ln

        # Backprop through first residual connection
        dattn_output = dx2
        dx1 = dx2

        # Backprop through attention
        dattn_input = self.attention.backward(dattn_output)

        # Backprop through first layer norm
        dx1_ln = self.ln1.backward(dattn_input)
        dx1 = dx1 + dx1_ln

        return dx1


class TransformerLM:
    """
    Complete Transformer language model.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimensionality
        num_heads: Number of attention heads
        n_layers: Number of Transformer blocks
        d_ff: Feed-forward hidden dimension (default: 4 * d_model)
        max_len: Maximum sequence length
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        n_layers: int,
        d_ff: Optional[int] = None,
        max_len: int = 2048,
        dropout: float = 0.0
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.max_len = max_len
        self.dropout = dropout

        if d_ff is None:
            d_ff = 4 * d_model

        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, d_model) * 0.02

        # Positional encoding
        self.pos_encoding = LearnedPositionalEncoding(d_model, max_len)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, max_len, dropout)
            for _ in range(n_layers)
        ]

        # Final layer norm
        self.ln_f = LayerNorm(d_model)

        # Dropout probability
        self.dropout_p = dropout

        # Training mode flag
        self.training = True

        # Gradient caches
        self.cache = {}
        self.grads = {}

    def _apply_dropout(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply dropout during training."""
        if not training or self.dropout_p == 0.0:
            return x
        mask = (np.random.rand(*x.shape) > self.dropout_p).astype(x.dtype)
        return x * mask / (1.0 - self.dropout_p)

    def forward(self, input_ids: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            training: Whether in training mode

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        self.training = training

        # Token embeddings: (batch, seq_len, d_model)
        x = self.token_embeddings[input_ids]

        # Add positional encoding
        x = self.pos_encoding.forward(x)

        # Apply dropout
        x = self._apply_dropout(x, training=training)

        # Cache for backward
        self.cache['input_ids'] = input_ids
        self.cache['embed_output'] = x

        # Apply Transformer blocks
        for i, block in enumerate(self.blocks):
            x = block.forward(x, training=training)
            self.cache[f'block_{i}_output'] = x

        # Final layer norm
        x = self.ln_f.forward(x)
        self.cache['ln_f_output'] = x

        # Project to vocabulary (tied embeddings)
        # (batch, seq_len, d_model) @ (d_model, vocab_size) -> (batch, seq_len, vocab_size)
        logits = x @ self.token_embeddings.T

        return logits

    def backward(self, dlogits: np.ndarray) -> None:
        """
        Backward pass.

        Args:
            dlogits: Gradient of loss w.r.t. logits, shape (batch_size, seq_len, vocab_size)
        """
        # Backprop through output projection
        x = self.cache['ln_f_output']
        dx = dlogits @ self.token_embeddings  # (batch, seq_len, d_model)
        dW_embed = x.reshape(-1, self.d_model).T @ dlogits.reshape(-1, self.vocab_size)

        # Backprop through final layer norm
        dx = self.ln_f.backward(dx)

        # Backprop through transformer blocks (reverse order)
        for i in reversed(range(self.n_layers)):
            dx = self.blocks[i].backward(dx)

        # Backprop through positional encoding
        dx = self.pos_encoding.backward(dx)

        # Accumulate embedding gradients
        input_ids = self.cache['input_ids']
        np.add.at(dW_embed.T, input_ids.ravel(), dx.reshape(-1, self.d_model))

        # Store gradients
        self.grads['token_embeddings'] = dW_embed.T

    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Starting token IDs of shape (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Get predictions for current sequence
            logits = self.forward(input_ids, training=False)  # (batch, seq_len, vocab_size)

            # Focus on last position
            logits = logits[:, -1, :] / temperature  # (batch, vocab_size)

            # Optional top-k filtering
            if top_k is not None:
                top_k_actual = min(top_k, logits.shape[-1])
                # Get top-k threshold for each batch
                top_k_values = np.partition(logits, -top_k_actual, axis=-1)[:, -top_k_actual:]
                threshold = top_k_values.min(axis=1, keepdims=True)
                logits = np.where(logits >= threshold, logits, -np.inf)

            # Sample from distribution
            # Numerical stability: subtract max before exp
            logits_stable = logits - np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits_stable)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # Sample next token for each batch
            next_tokens = []
            for batch_probs in probs:
                next_token = np.random.choice(len(batch_probs), p=batch_probs)
                next_tokens.append(next_token)
            next_tokens = np.array(next_tokens).reshape(-1, 1)

            # Append to sequence
            input_ids = np.concatenate([input_ids, next_tokens], axis=1)

        return input_ids

    def eval(self):
        """Set model to evaluation mode."""
        self.training = False

    def train(self):
        """Set model to training mode."""
        self.training = True
