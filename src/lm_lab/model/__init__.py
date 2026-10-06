"""
Transformer model components.

This module implements a Transformer language model from scratch, including:
- Multi-head causal self-attention
- Position encodings (learned and sinusoidal)
- Layer normalization
- Feed-forward networks
- Complete Transformer blocks and model
"""

from .attention import CausalSelfAttention
from .embedding import PositionalEncoding, LearnedPositionalEncoding
from .layer_norm import LayerNorm
from .feedforward import FeedForward
from .transformer import TransformerBlock, TransformerLM

# Alias for compatibility
MultiHeadAttention = CausalSelfAttention

__all__ = [
    'CausalSelfAttention',
    'MultiHeadAttention',
    'PositionalEncoding',
    'LearnedPositionalEncoding',
    'LayerNorm',
    'FeedForward',
    'TransformerBlock',
    'TransformerLM',
]
