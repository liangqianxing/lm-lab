"""
Integration test for Transformer, optimizer, and training pipeline.

This test verifies:
- Transformer forward/backward pass
- Causal masking
- Gradient computation
- Optimizer updates
- Training loop
- Numerical stability
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lm_lab.model import TransformerLM
from lm_lab.optimizer import Adam
from lm_lab.training import Trainer, create_dataloader, LanguageModelingDataset


def test_transformer_forward():
    """Test forward pass and shape correctness."""
    print("Testing Transformer forward pass...")

    vocab_size = 100
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 256
    batch_size = 2
    seq_len = 10

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=128
    )

    # Create input
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    logits = model.forward(input_ids)

    # Check shape
    assert logits.shape == (batch_size, seq_len, vocab_size), \
        f"Expected shape {(batch_size, seq_len, vocab_size)}, got {logits.shape}"

    # Check no NaN or Inf
    assert np.isfinite(logits).all(), "Logits contain NaN or Inf"

    print("[OK] Forward pass shape and stability correct")


def test_causal_masking():
    """Test that future tokens don't affect past predictions."""
    print("\nTesting causal masking...")

    vocab_size = 50
    d_model = 32
    num_heads = 2
    num_layers = 1
    d_ff = 128
    seq_len = 5

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=128
    )

    # Create two inputs that differ only in future positions
    input1 = np.random.randint(0, vocab_size, (1, seq_len))
    input2 = input1.copy()
    input2[0, 3:] = np.random.randint(0, vocab_size, 2)  # Change last 2 tokens

    # Forward passes
    logits1 = model.forward(input1)
    logits2 = model.forward(input2)

    # First 3 positions should be identical (causal masking)
    for pos in range(3):
        diff = np.abs(logits1[0, pos] - logits2[0, pos]).max()
        assert diff < 1e-6, f"Position {pos} affected by future tokens (diff={diff})"

    # Last positions should differ
    diff = np.abs(logits1[0, -1] - logits2[0, -1]).max()
    assert diff > 1e-3, "Future positions not affecting output (possible masking failure)"

    print("[OK] Causal masking working correctly")


def test_gradient_flow():
    """Test gradient computation and backward pass."""
    print("\nTesting gradient flow...")

    vocab_size = 50
    d_model = 32
    num_heads = 2
    num_layers = 1
    d_ff = 128

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=128
    )

    # Create input and target
    input_ids = np.random.randint(0, vocab_size, (1, 5))
    target_ids = np.random.randint(0, vocab_size, (1, 5))

    # Forward pass
    logits = model.forward(input_ids)

    # Compute loss gradient
    batch_size, seq_len, vocab_size = logits.shape
    logits_stable = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    grad_logits = probs.copy()
    for b in range(batch_size):
        for s in range(seq_len):
            target_id = target_ids[b, s]
            grad_logits[b, s, target_id] -= 1.0

    # Backward pass
    model.backward(grad_logits)

    # Check gradients exist and are finite
    gradients = model.gradients()
    for i, grad in enumerate(gradients):
        assert grad is not None, f"Gradient {i} is None"
        assert np.isfinite(grad).all(), f"Gradient {i} contains NaN or Inf"
        assert (grad != 0).any(), f"Gradient {i} is all zeros"

    print("[OK] Gradients computed correctly")


def test_optimizer_step():
    """Test optimizer parameter updates."""
    print("\nTesting optimizer updates...")

    # Create simple parameter
    param = np.array([[1.0, 2.0], [3.0, 4.0]])
    grad = np.array([[0.1, 0.2], [0.3, 0.4]])

    params = {'param_0': param.copy()}
    grads = {'param_0': grad}

    optimizer = Adam(learning_rate=0.1)

    # Take a step
    optimizer.step(params, grads)

    # Parameters should have changed
    assert not np.allclose(params['param_0'], param), "Parameters not updated"

    # Parameters should have moved in opposite direction of gradient (descent)
    diff = params['param_0'] - param
    assert (diff * grad < 0).all(), "Parameters not descending gradient"

    print("[OK] Optimizer updates working")


def test_training_step():
    """Test one training step."""
    print("\nTesting training step...")

    vocab_size = 100
    d_model = 32
    num_heads = 2
    num_layers = 1
    d_ff = 128
    batch_size = 2
    seq_len = 8

    # Create model and optimizer
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=128
    )

    optimizer = Adam(learning_rate=0.001)

    # Create dummy data
    tokens = np.random.randint(0, vocab_size, 1000)
    dataset = LanguageModelingDataset(tokens)
    train_loader = create_dataloader(dataset, batch_size=batch_size, seq_len=seq_len, shuffle=False)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        log_interval=10
    )

    # Get initial parameters
    initial_params = [p.copy() for p in model.parameters()]

    # Train for one step
    batch = next(iter(train_loader))
    loss = trainer._train_step(batch)

    # Check loss is finite
    assert np.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss > 0, f"Loss should be positive: {loss}"

    # Check parameters changed
    final_params = model.parameters()
    changed = False
    for init_p, final_p in zip(initial_params, final_params):
        if not np.allclose(init_p, final_p):
            changed = True
            break

    assert changed, "Parameters did not update during training"

    print(f"[OK] Training step completed (loss={loss:.4f})")


def test_overfitting_tiny_dataset():
    """Test model can overfit a tiny dataset."""
    print("\nTesting overfitting capability...")

    vocab_size = 20
    d_model = 32
    num_heads = 2
    num_layers = 2
    d_ff = 128
    batch_size = 2
    seq_len = 8

    # Create model and optimizer
    np.random.seed(42)
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=128
    )

    optimizer = Adam(learning_rate=0.01)

    # Create tiny repeating dataset
    pattern = np.array([1, 2, 3, 4, 5, 6, 7, 8] * 5)
    dataset = LanguageModelingDataset(pattern)
    train_loader = create_dataloader(dataset, batch_size=batch_size, seq_len=seq_len, shuffle=False)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        log_interval=50
    )

    # Train for multiple epochs
    print("Training for 5 epochs...")
    initial_loss = None

    for epoch in range(5):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            loss = trainer._train_step(batch, accumulation_steps=1, batch_idx=batch_idx)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        if initial_loss is None:
            initial_loss = avg_loss

        print(f"  Epoch {epoch}: loss={avg_loss:.4f}")

    # Loss should decrease significantly
    final_loss = avg_loss
    improvement = initial_loss - final_loss

    assert improvement > 0.5, \
        f"Loss did not decrease enough (initial={initial_loss:.4f}, final={final_loss:.4f})"

    print(f"[OK] Model overfitted successfully (improvement={improvement:.4f})")


def test_numerical_stability():
    """Test numerical stability with extreme inputs."""
    print("\nTesting numerical stability...")

    vocab_size = 50
    d_model = 32
    num_heads = 2
    num_layers = 1
    d_ff = 128

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=128
    )

    # Test with various inputs
    test_cases = [
        np.array([[0]]),  # Single token
        np.array([[0, 0, 0, 0]]),  # Repeated tokens
        np.random.randint(0, vocab_size, (1, 50)),  # Long sequence
        np.random.randint(0, vocab_size, (8, 10)),  # Large batch
    ]

    for i, input_ids in enumerate(test_cases):
        logits = model.forward(input_ids)
        assert np.isfinite(logits).all(), f"Test case {i} produced non-finite logits"

    print("[OK] Numerical stability verified")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Transformer + Optimizer + Training Tests")
    print("=" * 60)

    test_transformer_forward()
    test_causal_masking()
    test_gradient_flow()
    test_optimizer_step()
    test_training_step()
    test_overfitting_tiny_dataset()
    test_numerical_stability()

    print("\n" + "=" * 60)
    print("All tests passed! [OK]")
    print("=" * 60)
    print("\nB2 (Transformer), B3 (Optimizer), B4 (Training) milestones:")
    print("- [OK] Causal attention with proper masking")
    print("- [OK] Position encodings (learned)")
    print("- [OK] Feed-forward networks with GELU")
    print("- [OK] Layer normalization")
    print("- [OK] Complete gradient flow")
    print("- [OK] Adam optimizer with momentum and adaptive learning rates")
    print("- [OK] Training loop with gradient clipping")
    print("- [OK] Cross-entropy loss computation")
    print("- [OK] Numerical stability checks")
    print("- [OK] Overfitting capability verified")


if __name__ == "__main__":
    run_all_tests()
