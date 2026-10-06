"""
Simple integration test for Transformer + Optimizer + Training.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lm_lab.model import TransformerLM
from lm_lab.optimizer import Adam

print("=" * 60)
print("Testing Transformer + Optimizer + Training Integration")
print("=" * 60)

# Test 1: Forward pass
print("\n1. Testing forward pass...")
vocab_size = 100
d_model = 64
num_heads = 4
num_layers = 2
d_ff = 256

model = TransformerLM(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_seq_len=128
)

input_ids = np.random.randint(0, vocab_size, (2, 10))
logits = model.forward(input_ids)
assert logits.shape == (2, 10, vocab_size)
assert np.isfinite(logits).all()
print("   [OK] Forward pass works")

# Test 2: Causal masking
print("\n2. Testing causal masking...")
input1 = np.random.randint(0, vocab_size, (1, 5))
input2 = input1.copy()
input2[0, 3:] = np.random.randint(0, vocab_size, 2)

logits1 = model.forward(input1)
logits2 = model.forward(input2)

for pos in range(3):
    diff = np.abs(logits1[0, pos] - logits2[0, pos]).max()
    assert diff < 1e-6
print("   [OK] Causal masking works")

# Test 3: Backward pass
print("\n3. Testing backward pass...")
target_ids = np.random.randint(0, vocab_size, (2, 10))
logits = model.forward(input_ids)

batch_size, seq_len, vocab_size = logits.shape
logits_stable = logits - np.max(logits, axis=-1, keepdims=True)
exp_logits = np.exp(logits_stable)
probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

grad_logits = probs.copy()
for b in range(batch_size):
    for s in range(seq_len):
        target_id = target_ids[b, s]
        grad_logits[b, s, target_id] -= 1.0

model.backward(grad_logits)

gradients = model.gradients()
assert all(g is not None for g in gradients)
assert all(np.isfinite(g).all() for g in gradients)
print("   [OK] Backward pass works")

# Test 4: Optimizer step
print("\n4. Testing optimizer step...")
optimizer = Adam(learning_rate=0.01)

params_before = [p.copy() for p in model.parameters()]

# Create param/grad dicts
params_dict = {f"p{i}": p for i, p in enumerate(model.parameters())}
grads_dict = {f"p{i}": g for i, g in enumerate(model.gradients())}

optimizer.step(params_dict, grads_dict)

# Update model parameters
for i, p in enumerate(model.parameters()):
    p[:] = params_dict[f"p{i}"]

changed = any(not np.allclose(p1, p2) for p1, p2 in zip(params_before, model.parameters()))
assert changed
print("   [OK] Optimizer updates parameters")

# Test 5: Training for multiple steps
print("\n5. Testing multiple training steps...")
losses = []

for step in range(10):
    # Forward
    input_ids = np.random.randint(0, vocab_size, (2, 8))
    target_ids = np.random.randint(0, vocab_size, (2, 8))
    logits = model.forward(input_ids)

    # Loss
    batch_size, seq_len, vocab_size = logits.shape
    logits_stable = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    loss = 0.0
    for b in range(batch_size):
        for s in range(seq_len):
            target_id = target_ids[b, s]
            loss -= np.log(probs[b, s, target_id] + 1e-10)
    loss = loss / (batch_size * seq_len)
    losses.append(loss)

    # Backward
    grad_logits = probs.copy()
    for b in range(batch_size):
        for s in range(seq_len):
            target_id = target_ids[b, s]
            grad_logits[b, s, target_id] -= 1.0
    grad_logits = grad_logits / (batch_size * seq_len)

    model.backward(grad_logits)

    # Optimizer step
    params_dict = {f"p{i}": p for i, p in enumerate(model.parameters())}
    grads_dict = {f"p{i}": g for i, g in enumerate(model.gradients())}
    optimizer.step(params_dict, grads_dict)
    for i, p in enumerate(model.parameters()):
        p[:] = params_dict[f"p{i}"]

assert all(np.isfinite(l) for l in losses)
print(f"   [OK] Training completed (loss: {losses[0]:.4f} -> {losses[-1]:.4f})")

# Test 6: Overfit tiny dataset
print("\n6. Testing overfitting on tiny dataset...")
np.random.seed(42)

small_model = TransformerLM(
    vocab_size=20,
    d_model=32,
    num_heads=2,
    num_layers=2,
    d_ff=128,
    max_seq_len=128
)

optimizer = Adam(learning_rate=0.01)

# Fixed tiny dataset
fixed_input = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
fixed_target = np.array([[2, 3, 4, 5, 6, 7, 8, 9]])

initial_loss = None
for epoch in range(20):
    logits = small_model.forward(fixed_input)

    # Compute loss
    batch_size, seq_len, vocab_size = logits.shape
    logits_stable = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    loss = 0.0
    for b in range(batch_size):
        for s in range(seq_len):
            target_id = fixed_target[b, s]
            loss -= np.log(probs[b, s, target_id] + 1e-10)
    loss = loss / (batch_size * seq_len)

    if initial_loss is None:
        initial_loss = loss

    # Backward
    grad_logits = probs.copy()
    for b in range(batch_size):
        for s in range(seq_len):
            target_id = fixed_target[b, s]
            grad_logits[b, s, target_id] -= 1.0
    grad_logits = grad_logits / (batch_size * seq_len)

    small_model.backward(grad_logits)

    # Optimizer step
    params_dict = {f"p{i}": p for i, p in enumerate(small_model.parameters())}
    grads_dict = {f"p{i}": g for i, g in enumerate(small_model.gradients())}
    optimizer.step(params_dict, grads_dict)
    for i, p in enumerate(small_model.parameters()):
        p[:] = params_dict[f"p{i}"]

final_loss = loss
improvement = initial_loss - final_loss
assert improvement > 0.5
print(f"   [OK] Overfitted successfully (loss: {initial_loss:.4f} -> {final_loss:.4f})")

print("\n" + "=" * 60)
print("All Tests Passed!")
print("=" * 60)
print("\nImplemented components:")
print("- Transformer with causal self-attention")
print("- Multi-head attention with proper masking")
print("- Layer normalization")
print("- Feed-forward networks with GELU")
print("- Learned positional encodings")
print("- Complete gradient flow")
print("- Adam optimizer")
print("- Training loop with loss computation")
print("- Numerical stability throughout")
print("\nMilestones completed:")
print("- B2 (Transformer): COMPLETE")
print("- B3 (Optimizer): COMPLETE")
print("- B4 (Training): COMPLETE")
