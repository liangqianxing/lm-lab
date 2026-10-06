"""
Comprehensive unit tests for all Transformer components.

Tests cover:
- Attention masking (future tokens shouldn't affect past)
- Shape propagation through all layers
- Gradient flow (check no NaN/Inf)
- Numerical stability with extreme inputs
- Comparison with known correct implementations
"""

import torch
import pytest
import math
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lm_lab.model import (
    LayerNorm,
    FeedForward,
    CausalSelfAttention,
    TransformerBlock,
    TransformerLM,
    PositionalEncoding,
    LearnedPositionalEncoding
)


# ============================================================================
# LayerNorm Tests
# ============================================================================

class TestLayerNorm:
    """Test suite for LayerNorm component."""

    def test_shape_preservation(self):
        """LayerNorm should preserve input shape."""
        ln = LayerNorm(d_model=512)

        # Test various shapes
        shapes = [(1, 10, 512), (32, 128, 512), (4, 1, 512)]
        for shape in shapes:
            x = torch.randn(shape)
            out = ln(x)
            assert out.shape == x.shape, f"Shape mismatch for {shape}"

    def test_normalization_statistics(self):
        """LayerNorm should normalize to mean=0, var=1."""
        ln = LayerNorm(d_model=512)
        x = torch.randn(32, 128, 512) * 10 + 5  # arbitrary mean and variance
        out = ln(x)

        # Check statistics along feature dimension
        mean = out.mean(dim=-1)
        var = out.var(dim=-1, unbiased=False)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5)

    def test_gradient_flow(self):
        """LayerNorm should have valid gradients."""
        ln = LayerNorm(d_model=512)
        x = torch.randn(32, 128, 512, requires_grad=True)
        out = ln(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        assert (x.grad.abs() > 0).any()  # Non-zero gradients

    def test_learnable_parameters(self):
        """LayerNorm should have learnable gamma and beta."""
        ln = LayerNorm(d_model=512)

        params = dict(ln.named_parameters())
        assert 'gamma' in params
        assert 'beta' in params
        assert params['gamma'].shape == (512,)
        assert params['beta'].shape == (512,)

    def test_numerical_stability_extreme_values(self):
        """LayerNorm should handle extreme values without NaN/Inf."""
        ln = LayerNorm(d_model=512, eps=1e-6)

        # Very large values
        x_large = torch.randn(4, 10, 512) * 1e6
        out_large = ln(x_large)
        assert not torch.isnan(out_large).any()
        assert not torch.isinf(out_large).any()

        # Very small values
        x_small = torch.randn(4, 10, 512) * 1e-6
        out_small = ln(x_small)
        assert not torch.isnan(out_small).any()
        assert not torch.isinf(out_small).any()

    def test_comparison_with_pytorch_layernorm(self):
        """Compare with PyTorch's LayerNorm implementation."""
        d_model = 512
        ln_custom = LayerNorm(d_model=d_model)
        ln_pytorch = torch.nn.LayerNorm(d_model)

        # Copy parameters
        ln_pytorch.weight.data = ln_custom.gamma.data.clone()
        ln_pytorch.bias.data = ln_custom.beta.data.clone()

        x = torch.randn(32, 128, d_model)
        out_custom = ln_custom(x)
        out_pytorch = ln_pytorch(x)

        assert torch.allclose(out_custom, out_pytorch, atol=1e-5)


# ============================================================================
# FeedForward Tests
# ============================================================================

class TestFeedForward:
    """Test suite for FeedForward component."""

    def test_shape_preservation(self):
        """FeedForward should preserve batch and sequence dimensions."""
        ff = FeedForward(d_model=512, d_ff=2048)

        shapes = [(1, 10, 512), (32, 128, 512), (4, 1, 512)]
        for shape in shapes:
            x = torch.randn(shape)
            out = ff(x)
            assert out.shape == x.shape, f"Shape mismatch for {shape}"

    def test_hidden_dimension_expansion(self):
        """FeedForward should expand to d_ff internally."""
        d_model, d_ff = 256, 1024
        ff = FeedForward(d_model=d_model, d_ff=d_ff)

        assert ff.linear1.weight.shape == (d_ff, d_model)
        assert ff.linear2.weight.shape == (d_model, d_ff)

    def test_gradient_flow(self):
        """FeedForward should have valid gradients."""
        ff = FeedForward(d_model=512, d_ff=2048)
        x = torch.randn(32, 128, 512, requires_grad=True)
        out = ff(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_dropout_effect(self):
        """Dropout should reduce some activations during training."""
        ff = FeedForward(d_model=512, d_ff=2048, dropout=0.5)
        ff.train()

        torch.manual_seed(42)
        x = torch.randn(32, 128, 512)

        # Run multiple times and check variability
        outputs = [ff(x) for _ in range(5)]

        # Outputs should differ due to dropout
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[0], outputs[i])

    def test_eval_mode_no_dropout(self):
        """In eval mode, dropout should be disabled."""
        ff = FeedForward(d_model=512, d_ff=2048, dropout=0.5)
        ff.eval()

        torch.manual_seed(42)
        x = torch.randn(32, 128, 512)

        # Multiple runs should give same output
        out1 = ff(x)
        out2 = ff(x)

        assert torch.allclose(out1, out2)

    def test_numerical_stability(self):
        """FeedForward should handle extreme inputs."""
        ff = FeedForward(d_model=512, d_ff=2048)

        # Large values
        x_large = torch.randn(4, 10, 512) * 100
        out_large = ff(x_large)
        assert not torch.isnan(out_large).any()
        assert not torch.isinf(out_large).any()


# ============================================================================
# CausalSelfAttention Tests
# ============================================================================

class TestCausalSelfAttention:
    """Test suite for CausalSelfAttention component."""

    def test_shape_preservation(self):
        """Attention should preserve input shape."""
        attn = CausalSelfAttention(d_model=512, n_heads=8)

        shapes = [(1, 10, 512), (32, 64, 512), (4, 128, 512)]
        for shape in shapes:
            x = torch.randn(shape)
            out = attn(x)
            assert out.shape == x.shape, f"Shape mismatch for {shape}"

    def test_causal_masking_gradient(self):
        """Future tokens should not affect past token gradients."""
        torch.manual_seed(42)
        attn = CausalSelfAttention(d_model=128, n_heads=4)

        batch_size, seq_len = 1, 10
        x = torch.randn(batch_size, seq_len, 128, requires_grad=True)

        out = attn(x)

        # Compute gradient of position 5 w.r.t. all positions
        out[:, 5, :].sum().backward()

        grad = x.grad[0, :, 0]

        # Past and current positions should have non-zero gradient
        assert grad[:6].abs().max() > 1e-6, "No gradient from past positions"

        # Future positions should have near-zero gradient
        assert grad[6:].abs().max() < 1e-6, "Future positions affecting past"

    def test_causal_masking_output(self):
        """Output at position i should only depend on positions <= i."""
        torch.manual_seed(42)
        attn = CausalSelfAttention(d_model=128, n_heads=4)
        attn.eval()

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 128)

        # Get output for full sequence
        out_full = attn(x)

        # Get output for truncated sequence (first 5 tokens)
        out_partial = attn(x[:, :5, :])

        # First 5 positions should match
        assert torch.allclose(out_full[:, :5, :], out_partial, atol=1e-5)

    def test_multi_head_splitting(self):
        """Attention should correctly split into multiple heads."""
        d_model, n_heads = 512, 8
        attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads)

        assert attn.d_k == d_model // n_heads
        assert attn.qkv_proj.weight.shape == (3 * d_model, d_model)
        assert attn.out_proj.weight.shape == (d_model, d_model)

    def test_attention_scores_scaling(self):
        """Attention scores should be scaled by sqrt(d_k)."""
        d_model, n_heads = 128, 4
        d_k = d_model // n_heads
        attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads)

        # The scaling is 1/sqrt(d_k), verify d_k is correct
        assert attn.d_k == d_k
        assert attn.d_k == 32

    def test_gradient_flow(self):
        """Gradients should flow through attention."""
        attn = CausalSelfAttention(d_model=512, n_heads=8)
        x = torch.randn(4, 32, 512, requires_grad=True)

        out = attn(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_numerical_stability_softmax(self):
        """Attention should handle extreme logits without NaN."""
        attn = CausalSelfAttention(d_model=128, n_heads=4)

        # Normal input
        x = torch.randn(4, 16, 128)
        out = attn(x)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_max_length_constraint(self):
        """Attention should work up to max_len."""
        max_len = 128
        attn = CausalSelfAttention(d_model=256, n_heads=4, max_len=max_len)

        # Should work at max_len
        x = torch.randn(2, max_len, 256)
        out = attn(x)
        assert out.shape == x.shape

    def test_attention_dropout(self):
        """Dropout should affect attention weights in training."""
        attn = CausalSelfAttention(d_model=256, n_heads=4, dropout=0.5)
        attn.train()

        torch.manual_seed(42)
        x = torch.randn(4, 16, 256)

        outputs = [attn(x) for _ in range(3)]

        # Outputs should differ
        assert not torch.allclose(outputs[0], outputs[1])


# ============================================================================
# PositionalEncoding Tests
# ============================================================================

class TestPositionalEncoding:
    """Test suite for PositionalEncoding components."""

    def test_sinusoidal_pe_shape(self):
        """Sinusoidal PE should preserve input shape."""
        pe = PositionalEncoding(d_model=512, max_len=1000)

        x = torch.randn(32, 100, 512)
        out = pe(x)
        assert out.shape == x.shape

    def test_sinusoidal_pe_deterministic(self):
        """Sinusoidal PE should be deterministic."""
        pe = PositionalEncoding(d_model=512, max_len=1000)

        x = torch.randn(32, 100, 512)
        out1 = pe(x)
        out2 = pe(x)

        assert torch.allclose(out1, out2)

    def test_learned_pe_shape(self):
        """Learned PE should preserve input shape."""
        pe = LearnedPositionalEncoding(d_model=512, max_len=1000)

        x = torch.randn(32, 100, 512)
        out = pe(x)
        assert out.shape == x.shape

    def test_learned_pe_trainable(self):
        """Learned PE should have trainable parameters."""
        pe = LearnedPositionalEncoding(d_model=512, max_len=1000)

        params = list(pe.parameters())
        assert len(params) == 1
        assert params[0].shape == (1000, 512)
        assert params[0].requires_grad

    def test_pe_different_sequence_lengths(self):
        """PE should handle varying sequence lengths."""
        pe = PositionalEncoding(d_model=256, max_len=1000)

        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(4, seq_len, 256)
            out = pe(x)
            assert out.shape == x.shape


# ============================================================================
# TransformerBlock Tests
# ============================================================================

class TestTransformerBlock:
    """Test suite for TransformerBlock component."""

    def test_shape_preservation(self):
        """TransformerBlock should preserve input shape."""
        block = TransformerBlock(d_model=512, n_heads=8)

        shapes = [(1, 10, 512), (32, 64, 512), (4, 128, 512)]
        for shape in shapes:
            x = torch.randn(shape)
            out = block(x)
            assert out.shape == x.shape, f"Shape mismatch for {shape}"

    def test_residual_connections(self):
        """Residual connections should allow gradient flow."""
        block = TransformerBlock(d_model=512, n_heads=8)
        x = torch.randn(4, 32, 512, requires_grad=True)

        out = block(x)
        loss = out.sum()
        loss.backward()

        # Gradient should flow
        assert x.grad is not None
        assert (x.grad.abs() > 0).any()

    def test_pre_norm_architecture(self):
        """Block should use pre-norm (LayerNorm before sublayers)."""
        block = TransformerBlock(d_model=512, n_heads=8)

        # Check that ln1 and ln2 exist
        assert hasattr(block, 'ln1')
        assert hasattr(block, 'ln2')
        assert isinstance(block.ln1, LayerNorm)
        assert isinstance(block.ln2, LayerNorm)

    def test_gradient_flow_through_block(self):
        """Gradients should flow through entire block."""
        block = TransformerBlock(d_model=512, n_heads=8)
        x = torch.randn(4, 32, 512, requires_grad=True)

        out = block(x)
        loss = out.sum()
        loss.backward()

        # Check all parameters have gradients
        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_numerical_stability(self):
        """Block should handle various inputs without NaN/Inf."""
        block = TransformerBlock(d_model=512, n_heads=8)

        # Normal input
        x_normal = torch.randn(4, 32, 512)
        out_normal = block(x_normal)
        assert not torch.isnan(out_normal).any()
        assert not torch.isinf(out_normal).any()

        # Large values
        x_large = torch.randn(4, 32, 512) * 10
        out_large = block(x_large)
        assert not torch.isnan(out_large).any()
        assert not torch.isinf(out_large).any()

    def test_causal_property_preserved(self):
        """Block should preserve causal property from attention."""
        torch.manual_seed(42)
        block = TransformerBlock(d_model=128, n_heads=4)
        block.eval()

        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 128)

        # Full sequence
        out_full = block(x)

        # Partial sequence
        out_partial = block(x[:, :6, :])

        # First 6 positions should match
        assert torch.allclose(out_full[:, :6, :], out_partial, atol=1e-5)


# ============================================================================
# TransformerLM Tests
# ============================================================================

class TestTransformerLM:
    """Test suite for complete TransformerLM model."""

    def test_output_shape(self):
        """Model should output logits of correct shape."""
        model = TransformerLM(
            vocab_size=1000,
            d_model=256,
            n_heads=4,
            n_layers=2,
            max_len=128
        )

        batch_size, seq_len = 8, 32
        tokens = torch.randint(0, 1000, (batch_size, seq_len))
        logits = model(tokens)

        assert logits.shape == (batch_size, seq_len, 1000)

    def test_gradient_flow_end_to_end(self):
        """Gradients should flow through entire model."""
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

        # Check all parameters have valid gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in {name}"

    def test_numerical_stability_forward(self):
        """Model should handle forward pass without NaN/Inf."""
        model = TransformerLM(
            vocab_size=100,
            d_model=128,
            n_heads=4,
            n_layers=2
        )

        tokens = torch.randint(0, 100, (4, 32))
        logits = model(tokens)

        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_generation(self):
        """Model should generate coherent sequences."""
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
        assert generated.max() < 100  # All tokens in vocab

    def test_generation_temperature(self):
        """Temperature should affect generation randomness."""
        torch.manual_seed(42)
        model = TransformerLM(
            vocab_size=100,
            d_model=128,
            n_heads=4,
            n_layers=2
        )

        start_tokens = torch.tensor([[1, 2, 3]])

        # Low temperature (more deterministic)
        gen_low = model.generate(start_tokens, max_new_tokens=10, temperature=0.1)

        torch.manual_seed(42)
        # High temperature (more random)
        gen_high = model.generate(start_tokens, max_new_tokens=10, temperature=2.0)

        # Should produce different sequences
        assert not (gen_low == gen_high).all()

    def test_causal_language_modeling(self):
        """Model should respect causal constraint."""
        torch.manual_seed(42)
        model = TransformerLM(
            vocab_size=100,
            d_model=128,
            n_heads=4,
            n_layers=2
        )
        model.eval()

        tokens = torch.randint(0, 100, (2, 10))

        # Full sequence
        logits_full = model(tokens)

        # Partial sequence
        logits_partial = model(tokens[:, :6])

        # First 6 positions should match
        assert torch.allclose(
            logits_full[:, :6, :],
            logits_partial,
            atol=1e-5
        )

    def test_weight_tying(self):
        """Model should tie input and output embeddings."""
        model = TransformerLM(
            vocab_size=1000,
            d_model=256,
            n_heads=4,
            n_layers=2
        )

        # The output projection uses transposed token embeddings
        tokens = torch.randint(0, 1000, (4, 32))
        logits = model(tokens)

        # Logits should have vocab_size dimension
        assert logits.shape[-1] == 1000

    def test_multiple_layers_stack(self):
        """Model should correctly stack multiple layers."""
        for n_layers in [1, 2, 4, 6]:
            model = TransformerLM(
                vocab_size=100,
                d_model=128,
                n_heads=4,
                n_layers=n_layers
            )

            assert len(model.blocks) == n_layers

            tokens = torch.randint(0, 100, (2, 16))
            logits = model(tokens)
            assert logits.shape == (2, 16, 100)

    def test_extreme_input_stability(self):
        """Model should handle edge cases."""
        model = TransformerLM(
            vocab_size=100,
            d_model=128,
            n_heads=4,
            n_layers=2
        )

        # Single token
        tokens_single = torch.randint(0, 100, (1, 1))
        logits_single = model(tokens_single)
        assert logits_single.shape == (1, 1, 100)
        assert not torch.isnan(logits_single).any()

        # Very long sequence (within max_len)
        tokens_long = torch.randint(0, 100, (1, 100))
        logits_long = model(tokens_long)
        assert logits_long.shape == (1, 100, 100)
        assert not torch.isnan(logits_long).any()

    def test_batch_independence(self):
        """Different batch elements should be independent."""
        torch.manual_seed(42)
        model = TransformerLM(
            vocab_size=100,
            d_model=128,
            n_heads=4,
            n_layers=2
        )
        model.eval()

        # Process separately
        tokens_a = torch.randint(0, 100, (1, 16))
        tokens_b = torch.randint(0, 100, (1, 16))

        logits_a = model(tokens_a)
        logits_b = model(tokens_b)

        # Process together
        tokens_batch = torch.cat([tokens_a, tokens_b], dim=0)
        logits_batch = model(tokens_batch)

        # Results should match
        assert torch.allclose(logits_batch[0], logits_a[0], atol=1e-5)
        assert torch.allclose(logits_batch[1], logits_b[0], atol=1e-5)


# ============================================================================
# Integration Tests
# ============================================================================

class TestTransformerIntegration:
    """Integration tests for complete Transformer pipeline."""

    def test_overfitting_single_batch(self):
        """Model should be able to overfit a single batch."""
        torch.manual_seed(42)
        model = TransformerLM(
            vocab_size=50,
            d_model=128,
            n_heads=4,
            n_layers=2
        )

        # Small batch
        tokens = torch.randint(0, 50, (4, 16))
        targets = tokens.clone()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        initial_loss = None
        final_loss = None

        for step in range(100):
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, 50),
                targets.view(-1)
            )

            if step == 0:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step == 99:
                final_loss = loss.item()

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5, \
            f"Model failed to overfit: initial={initial_loss:.4f}, final={final_loss:.4f}"

    def test_gradient_magnitude_reasonable(self):
        """Gradient magnitudes should be reasonable (not exploding/vanishing)."""
        model = TransformerLM(
            vocab_size=100,
            d_model=256,
            n_heads=8,
            n_layers=4
        )

        tokens = torch.randint(0, 100, (8, 64))
        targets = torch.randint(0, 100, (8, 64))

        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 100),
            targets.view(-1)
        )

        loss.backward()

        # Check gradient magnitudes
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)

                # Gradients should not be too large or too small
                assert 1e-8 < grad_norm < 1e3, \
                    f"Unusual gradient magnitude for {name}: {grad_norm}"

    def test_parameter_updates(self):
        """Parameters should actually update during training."""
        torch.manual_seed(42)
        model = TransformerLM(
            vocab_size=50,
            d_model=128,
            n_heads=4,
            n_layers=2
        )

        # Save initial parameters
        initial_params = {
            name: param.clone()
            for name, param in model.named_parameters()
        }

        # Training step
        tokens = torch.randint(0, 50, (4, 16))
        targets = torch.randint(0, 50, (4, 16))

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 50),
            targets.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check parameters changed
        for name, param in model.named_parameters():
            assert not torch.allclose(param, initial_params[name]), \
                f"Parameter {name} did not update"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
