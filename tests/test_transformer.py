"""
Tests for B2: Transformer components.

Verifies shapes, causal masking, gradients, and numerical stability.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lm_lab.model import (
    LayerNorm,
    FeedForward,
    MultiHeadAttention,
    CausalSelfAttention,
    TransformerBlock,
    TransformerLM
)


def test_layer_norm_shape():
    """Test LayerNorm preserves input shape."""
    ln = LayerNorm(d_model=512)
    x = torch.randn(32, 128, 512)
    out = ln(x)
    assert out.shape == x.shape


def test_layer_norm_normalization():
    """Test LayerNorm normalizes correctly."""
    ln = LayerNorm(d_model=512)
    x = torch.randn(32, 128, 512)
    out = ln(x)

    # Check mean and variance along feature dimension
    mean = out.mean(dim=-1)
    var = out.var(dim=-1, unbiased=False)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


def test_feedforward_shape():
    """Test FeedForward preserves shape."""
    ff = FeedForward(d_model=512, d_ff=2048)
    x = torch.randn(32, 128, 512)
    out = ff(x)
    assert out.shape == x.shape


def test_feedforward_gradient():
    """Test FeedForward has valid gradients."""
    ff = FeedForward(d_model=512, d_ff=2048)
    x = torch.randn(32, 128, 512, requires_grad=True)
    out = ff(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


def test_attention_shape():
    """Test MultiHeadAttention preserves shape."""
    attn = MultiHeadAttention(d_model=512, n_heads=8)
    x = torch.randn(32, 128, 512)
    out = attn(x)
    assert out.shape == x.shape


def test_causal_masking():
    """Test CausalSelfAttention prevents attending to future."""
    torch.manual_seed(42)
    attn = CausalSelfAttention(d_model=64, n_heads=4)

    # Create input where each position has a unique pattern
    batch_size, seq_len = 1, 10
    x = torch.zeros(batch_size, seq_len, 64)
    for i in range(seq_len):
        x[:, i, :] = float(i + 1)

    x.requires_grad = True
    out = attn(x)

    # Compute gradient of position 5 w.r.t. all positions
    out[:, 5, :].sum().backward()

    # Gradient should be zero for future positions (6, 7, 8, 9)
    # but non-zero for current and past positions (0-5)
    grad = x.grad[0, :, 0]

    # Past and current should have non-zero gradient
    assert grad[:6].abs().max() > 1e-6

    # Future should have near-zero gradient
    assert grad[6:].abs().max() < 1e-6


def test_transformer_block_shape():
    """Test TransformerBlock preserves shape."""
    block = TransformerBlock(d_model=512, n_heads=8)
    x = torch.randn(32, 128, 512)
    out = block(x)
    assert out.shape == x.shape


def test_transformer_lm_forward():
    """Test TransformerLM forward pass."""
    model = TransformerLM(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_layers=2,
        max_len=128
    )

    tokens = torch.randint(0, 1000, (8, 64))
    logits = model(tokens)

    assert logits.shape == (8, 64, 1000)
    assert not torch.isnan(logits).any()


def test_transformer_lm_generation():
    """Test TransformerLM can generate text."""
    torch.manual_seed(42)
    model = TransformerLM(
        vocab_size=100,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_len=64
    )

    start_tokens = torch.tensor([[1, 2, 3]])
    generated = model.generate(start_tokens, max_new_tokens=10)

    assert generated.shape == (1, 13)  # 3 + 10
    assert (generated[0, :3] == start_tokens[0]).all()


def test_numerical_stability():
    """Test model handles extreme inputs without NaN/Inf."""
    model = TransformerLM(
        vocab_size=100,
        d_model=128,
        n_heads=4,
        n_layers=2
    )

    # Test with random inputs
    tokens = torch.randint(0, 100, (4, 32))
    logits = model(tokens)

    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_gradient_flow():
    """Test gradients flow through entire model."""
    model = TransformerLM(
        vocab_size=100,
        d_model=128,
        n_heads=4,
        n_layers=2
    )

    tokens = torch.randint(0, 100, (4, 32))
    targets = torch.randint(0, 100, (4, 32))

    logits = model(tokens)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, 100),
        targets.view(-1)
    )

    loss.backward()

    # Check all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
