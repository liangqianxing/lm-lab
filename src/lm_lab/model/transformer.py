"""
Complete Transformer block and language model implementation.
"""

import math
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
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 2048, dropout: float = 0.0):
        self.d_model = d_model

        # Components
        self.ln1 = LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads, max_seq_len, dropout)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        """
        Forward pass with pre-norm and residual connections.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of same shape
        """
        # Self-attention block with residual
        attn_input = self.ln1.forward(x)
        attn_output = self.attention.forward(attn_input)
        x = x + attn_output

        # Feed-forward block with residual
        ffn_input = self.ln2.forward(x)
        ffn_output = self.ffn.forward(ffn_input)
        x = x + ffn_output

        self.cache = {'x_pre_attn': attn_input, 'x_post_attn': x - ffn_output,
                      'x_pre_ffn': ffn_input}

        return x

    def backward(self, grad_output):
        """
        Backward pass.

        Args:
            grad_output: Gradient from next layer

        Returns:
            Gradient with respect to input
        """
        # Backward through FFN residual
        grad_ffn_output = grad_output
        grad_x_post_attn = grad_output

        # Backward through FFN
        grad_ffn_input = self.ffn.backward(grad_ffn_output)
        grad_x_pre_ffn = self.ln2.backward(grad_ffn_input)

        # Add residual gradient
        grad_x_post_attn = grad_x_post_attn + grad_x_pre_ffn

        # Backward through attention residual
        grad_attn_output = grad_x_post_attn
        grad_x_pre_attn_residual = grad_x_post_attn

        # Backward through attention
        grad_attn_input = self.attention.backward(grad_attn_output)
        grad_x = self.ln1.backward(grad_attn_input)

        # Add residual gradient
        grad_x = grad_x + grad_x_pre_attn_residual

        return grad_x

    def parameters(self):
        """Return all parameters."""
        params = []
        params.extend(self.ln1.parameters())
        params.extend(self.attention.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.ffn.parameters())
        return params

    def gradients(self):
        """Return all gradients."""
        grads = []
        grads.extend(self.ln1.gradients())
        grads.extend(self.attention.gradients())
        grads.extend(self.ln2.gradients())
        grads.extend(self.ffn.gradients())
        return grads


class TransformerLM:
    """
    Complete Transformer language model.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimensionality
        num_heads: Number of attention heads
        num_layers: Number of Transformer blocks
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int,
                 d_ff: int, max_seq_len: int = 2048, dropout: float = 0.0):
        import numpy as np

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        # Token embeddings
        scale = 1.0 / math.sqrt(d_model)
        self.token_embeddings = np.random.randn(vocab_size, d_model) * scale
        self.grad_token_embeddings = None

        # Positional encoding
        self.pos_encoding = LearnedPositionalEncoding(d_model, max_seq_len)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, dropout)
            for _ in range(num_layers)
        ]

        # Final layer norm
        self.ln_f = LayerNorm(d_model)

        # Output projection (tied with input embeddings)
        self.use_tied_embeddings = True

    def forward(self, input_ids):
        """
        Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        import numpy as np

        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embeddings[input_ids]  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoding.forward(x)

        # Apply Transformer blocks
        for block in self.blocks:
            x = block.forward(x)

        # Final layer norm
        x = self.ln_f.forward(x)

        # Project to vocabulary (tied embeddings)
        if self.use_tied_embeddings:
            logits = x @ self.token_embeddings.T  # (batch, seq_len, vocab_size)
        else:
            raise NotImplementedError("Untied embeddings not implemented")

        self.cache = {
            'input_ids': input_ids,
            'x_final': x
        }

        return logits

    def backward(self, grad_logits):
        """
        Backward pass.

        Args:
            grad_logits: Gradient of loss w.r.t. logits

        Returns:
            None (gradients stored in parameters)
        """
        import numpy as np

        input_ids = self.cache['input_ids']
        x_final = self.cache['x_final']

        # Gradient through output projection
        if self.use_tied_embeddings:
            grad_x_final = grad_logits @ self.token_embeddings
            grad_token_embeddings_output = grad_logits.transpose(0, 2, 1) @ x_final
            grad_token_embeddings_output = grad_token_embeddings_output.sum(axis=0)

        # Backward through final layer norm
        grad_x = self.ln_f.backward(grad_x_final)

        # Backward through Transformer blocks
        for block in reversed(self.blocks):
            grad_x = block.backward(grad_x)

        # Backward through positional encoding
        grad_x = self.pos_encoding.backward(grad_x)

        # Gradient for token embeddings (input side)
        self.grad_token_embeddings = np.zeros_like(self.token_embeddings)
        batch_size, seq_len, d_model = grad_x.shape

        for b in range(batch_size):
            for s in range(seq_len):
                token_id = input_ids[b, s]
                self.grad_token_embeddings[token_id] += grad_x[b, s]

        # Add gradient from output projection
        if self.use_tied_embeddings:
            self.grad_token_embeddings += grad_token_embeddings_output

    def parameters(self):
        """Return all parameters."""
        params = [self.token_embeddings]
        params.extend(self.pos_encoding.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.ln_f.parameters())
        return params

    def gradients(self):
        """Return all gradients."""
        grads = [self.grad_token_embeddings]
        grads.extend(self.pos_encoding.gradients())
        for block in self.blocks:
            grads.extend(block.gradients())
        grads.extend(self.ln_f.gradients())
        return grads

