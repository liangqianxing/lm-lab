"""
Tests for B3: Optimizer implementations.

Verifies Adam/AdamW update rules, numerical stability, and gradient handling.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lm_lab.optimizer import Adam, AdamW


def test_adam_step():
    """Test Adam performs correct parameter updates."""
    # Simple quadratic: f(x) = (x - 3)^2, gradient = 2(x - 3)
    x = torch.tensor([0.0], requires_grad=True)
    optimizer = Adam([x], lr=0.1)

    for _ in range(100):
        loss = (x - 3.0) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Should converge close to 3
    assert torch.abs(x - 3.0) < 0.1


def test_adam_bias_correction():
    """Test Adam bias correction in early steps."""
    x = torch.tensor([1.0], requires_grad=True)
    optimizer = Adam([x], lr=0.01, betas=(0.9, 0.999))

    # First step
    loss = x ** 2
    optimizer.zero_grad()
    loss.backward()  # grad = 2.0

    old_x = x.item()
    optimizer.step()

    # Compute expected update manually
    # m1 = 0.9 * 0 + 0.1 * 2.0 = 0.2
    # v1 = 0.999 * 0 + 0.001 * 4.0 = 0.004
    # m_hat = 0.2 / (1 - 0.9^1) = 0.2 / 0.1 = 2.0
    # v_hat = 0.004 / (1 - 0.999^1) = 0.004 / 0.001 = 4.0
    # update = 0.01 * 2.0 / (sqrt(4.0) + 1e-8) ≈ 0.01

    # x should decrease
    assert x.item() < old_x


def test_adamw_weight_decay():
    """Test AdamW applies decoupled weight decay."""
    x = torch.tensor([10.0], requires_grad=True)
    optimizer = AdamW([x], lr=0.1, weight_decay=0.1)

    # No gradient, only weight decay
    optimizer.zero_grad()
    x.grad = torch.zeros_like(x)

    old_x = x.item()
    optimizer.step()

    # Should decay: x_new = x_old * (1 - lr * weight_decay)
    expected = old_x * (1 - 0.1 * 0.1)
    assert torch.abs(x - expected) < 1e-6


def test_optimizer_zero_grad():
    """Test zero_grad clears gradients."""
    x = torch.tensor([1.0], requires_grad=True)
    optimizer = Adam([x], lr=0.01)

    # Create gradient
    loss = x ** 2
    loss.backward()
    assert x.grad is not None

    # Zero it
    optimizer.zero_grad()
    assert x.grad is not None
    assert torch.all(x.grad == 0)


def test_optimizer_multiple_parameters():
    """Test optimizer handles multiple parameters."""
    params = [
        torch.tensor([1.0], requires_grad=True),
        torch.tensor([2.0], requires_grad=True),
        torch.tensor([3.0], requires_grad=True)
    ]
    optimizer = Adam(params, lr=0.1)

    # Simple loss: sum of squares
    loss = sum(p ** 2 for p in params)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # All should decrease
    assert params[0] < 1.0
    assert params[1] < 2.0
    assert params[2] < 3.0


def test_optimizer_numerical_stability():
    """Test optimizer handles extreme gradients."""
    x = torch.tensor([1.0], requires_grad=True)
    optimizer = Adam([x], lr=0.01)

    # Very large gradient
    optimizer.zero_grad()
    x.grad = torch.tensor([1e6])
    optimizer.step()

    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()

    # Very small gradient
    optimizer.zero_grad()
    x.grad = torch.tensor([1e-10])
    optimizer.step()

    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


def test_optimizer_convergence():
    """Test optimizer converges on simple function."""
    # Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
    # Global minimum at (a, a^2) = (1, 1)
    x = torch.tensor([0.0, 0.0], requires_grad=True)
    optimizer = Adam([x], lr=0.01)

    a, b = 1.0, 100.0

    for _ in range(1000):
        loss = (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Should converge close to (1, 1)
    assert torch.abs(x[0] - 1.0) < 0.1
    assert torch.abs(x[1] - 1.0) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
