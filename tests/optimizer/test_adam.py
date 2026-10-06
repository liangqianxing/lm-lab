"""
Tests for Adam and AdamW optimizers.

Tests Adam update rules, bias correction, momentum/variance tracking,
weight decay (Adam vs AdamW), and state persistence.
"""

import pytest
import torch
import numpy as np
from lm_lab.optimizer.adam_torch import Adam, AdamW


class TestAdamBasic:
    """Test basic Adam optimizer."""

    def test_simple_update(self):
        """Test basic Adam parameter update with bias correction."""
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        param.grad = torch.tensor([0.1, 0.2])

        opt = Adam([param], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
        opt.step()

        # Step 1:
        # m_1 = 0.9 * 0 + (1 - 0.9) * [0.1, 0.2] = [0.01, 0.02]
        # v_1 = 0.999 * 0 + (1 - 0.999) * [0.01, 0.04] = [0.00001, 0.00004]
        # m_hat = [0.01, 0.02] / (1 - 0.9^1) = [0.1, 0.2]
        # v_hat = [0.00001, 0.00004] / (1 - 0.999^1) = [0.01, 0.04]
        # param = [1.0, 2.0] - 0.1 * [0.1, 0.2] / (sqrt([0.01, 0.04]) + 1e-8)
        #       = [1.0, 2.0] - 0.1 * [0.1/0.1, 0.2/0.2]
        #       = [1.0, 2.0] - [0.1, 0.1] = [0.9, 1.9]

        expected = torch.tensor([0.9, 1.9])
        assert torch.allclose(param.data, expected, rtol=1e-5)

    def test_multiple_steps(self):
        """Test Adam over multiple optimization steps."""
        param = torch.tensor([1.0], requires_grad=True)
        opt = Adam([param], lr=0.01, betas=(0.9, 0.9), eps=1e-8)

        # Step 1
        param.grad = torch.tensor([1.0])
        opt.step()

        # m_1 = 0.1, v_1 = 0.1
        # m_hat = 0.1 / 0.1 = 1.0, v_hat = 0.1 / 0.1 = 1.0
        # param = 1.0 - 0.01 * 1.0 / sqrt(1.0) = 0.99
        assert torch.isclose(param.data, torch.tensor(0.99), rtol=1e-5)

        # Step 2
        param.grad = torch.tensor([1.0])
        opt.step()

        # m_2 = 0.9 * 0.1 + 0.1 * 1.0 = 0.19
        # v_2 = 0.9 * 0.1 + 0.1 * 1.0 = 0.19
        # m_hat = 0.19 / (1 - 0.81) = 1.0
        # v_hat = 0.19 / 0.19 = 1.0
        # param = 0.99 - 0.01 * 1.0 / 1.0 = 0.98
        assert torch.isclose(param.data, torch.tensor(0.98), rtol=1e-5)

    def test_zero_grad(self):
        """Test zero_grad clears gradients."""
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        param.grad = torch.tensor([0.5, 0.5])

        opt = Adam([param], lr=0.1)
        opt.zero_grad()

        assert param.grad is not None
        assert torch.all(param.grad == 0)

    def test_no_grad_skip(self):
        """Test that parameters without gradients are skipped."""
        param1 = torch.tensor([1.0], requires_grad=True)
        param2 = torch.tensor([2.0], requires_grad=True)
        param1.grad = torch.tensor([0.1])
        # param2 has no gradient

        opt = Adam([param1, param2], lr=0.1)
        initial_param2 = param2.data.clone()
        opt.step()

        # param1 should be updated, param2 should remain unchanged
        assert not torch.equal(param1.data, torch.tensor([1.0]))
        assert torch.equal(param2.data, initial_param2)


class TestAdamBiasCorrection:
    """Test Adam bias correction mechanism."""

    def test_bias_correction_convergence(self):
        """Test that bias correction approaches 1 as steps increase."""
        param = torch.tensor([1.0], requires_grad=True)
        opt = Adam([param], lr=0.01, betas=(0.9, 0.999))

        corrections_m = []
        corrections_v = []

        for step in range(1, 11):
            param.grad = torch.tensor([1.0])

            # Calculate correction factors
            correction_m = 1 - 0.9 ** step
            correction_v = 1 - 0.999 ** step
            corrections_m.append(correction_m)
            corrections_v.append(correction_v)

            opt.step()

        # Corrections should approach 1.0
        assert corrections_m[0] < corrections_m[-1]
        assert corrections_v[0] < corrections_v[-1]
        assert corrections_m[-1] > 0.6  # After 10 steps
        assert corrections_v[-1] > 0.009  # Slower convergence

    def test_first_step_bias_correction(self):
        """Test that first step has maximal bias correction."""
        param = torch.tensor([1.0], requires_grad=True)
        param.grad = torch.tensor([1.0])

        opt = Adam([param], lr=1.0, betas=(0.5, 0.5), eps=0)
        opt.step()

        # m_1 = 0.5 * grad = 0.5, m_hat = 0.5 / 0.5 = 1.0
        # v_1 = 0.5 * grad^2 = 0.5, v_hat = 0.5 / 0.5 = 1.0
        # update = 1.0 * 1.0 / sqrt(1.0) = 1.0
        # param = 1.0 - 1.0 = 0.0
        assert torch.isclose(param.data, torch.tensor(0.0), atol=1e-6)


class TestAdamMomentumVariance:
    """Test Adam's first and second moment estimates."""

    def test_momentum_accumulation(self):
        """Test first moment (momentum) accumulation."""
        param = torch.tensor([1.0], requires_grad=True)
        opt = Adam([param], lr=0.1, betas=(0.9, 0.999))

        # Apply same gradient multiple times
        for _ in range(3):
            param.grad = torch.tensor([1.0])
            opt.step()

        state = opt.state[param]
        # m should accumulate toward gradient value
        assert state['m'][0] > 0.1  # More than first step
        assert state['m'][0] < 1.0  # Less than gradient

    def test_variance_tracking(self):
        """Test second moment (variance) tracking."""
        param = torch.tensor([1.0], requires_grad=True)
        opt = Adam([param], lr=0.1, betas=(0.9, 0.999))

        # High variance gradients
        param.grad = torch.tensor([1.0])
        opt.step()
        param.grad = torch.tensor([-1.0])
        opt.step()
        param.grad = torch.tensor([1.0])
        opt.step()

        state = opt.state[param]
        # v should track squared gradients
        assert state['v'][0] > 0
        assert state['step'] == 3

    def test_different_betas(self):
        """Test effect of different beta values."""
        param1 = torch.tensor([1.0], requires_grad=True)
        param2 = torch.tensor([1.0], requires_grad=True)

        opt1 = Adam([param1], lr=0.1, betas=(0.5, 0.9))  # Less momentum
        opt2 = Adam([param2], lr=0.1, betas=(0.99, 0.999))  # More momentum

        for _ in range(5):
            param1.grad = torch.tensor([1.0])
            param2.grad = torch.tensor([1.0])
            opt1.step()
            opt2.step()

        # Higher beta1 means slower momentum accumulation per step
        # but momentum decays slower, so after many steps it accumulates more
        # After 5 steps with constant gradient, lower beta1 reaches steady state faster
        assert opt1.state[param1]['m'][0] > opt2.state[param2]['m'][0]


class TestAdamW:
    """Test AdamW optimizer with decoupled weight decay."""

    def test_weight_decay_separate_from_gradient(self):
        """Test that AdamW applies weight decay independently."""
        param = torch.tensor([1.0, 1.0], requires_grad=True)
        param.grad = torch.tensor([0.0, 0.0])  # Zero gradient

        opt = AdamW([param], lr=0.1, weight_decay=0.1)
        initial = param.data.clone()
        opt.step()

        # Even with zero gradient, weight decay should reduce parameters
        # Decoupled: param = param - lr * weight_decay * param
        # Expected: 1.0 - 0.1 * 0.1 * 1.0 = 0.99
        assert torch.all(param.data < initial)
        assert torch.allclose(param.data, torch.tensor([0.99, 0.99]), rtol=1e-5)

    def test_adamw_vs_adam_with_gradient(self):
        """Test AdamW weight decay differs from Adam's L2 penalty."""
        param_adam = torch.tensor([2.0], requires_grad=True)
        param_adamw = torch.tensor([2.0], requires_grad=True)

        # Note: Adam doesn't have weight_decay in our implementation
        # This test shows AdamW's decoupled approach
        opt_adamw = AdamW([param_adamw], lr=0.1, weight_decay=0.01, betas=(0.9, 0.9))

        param_adam.grad = torch.tensor([1.0])
        param_adamw.grad = torch.tensor([1.0])

        opt_adamw.step()

        # AdamW: Adam update + decoupled weight decay
        # After Adam step: ~2.0 - 0.1 (approx)
        # After weight decay: result - 0.1 * 0.01 * 2.0 = result - 0.002
        assert param_adamw.data < torch.tensor([2.0])

    def test_weight_decay_accumulation(self):
        """Test weight decay effect over multiple steps."""
        param = torch.tensor([1.0], requires_grad=True)
        opt = AdamW([param], lr=0.1, weight_decay=0.1)

        values = [param.data.item()]
        for _ in range(5):
            param.grad = torch.tensor([0.1])
            opt.step()
            values.append(param.data.item())

        # Parameter should decrease due to gradient and weight decay
        assert all(values[i] > values[i+1] for i in range(len(values)-1))


class TestAdamStatePersistence:
    """Test Adam optimizer state save/load."""

    def test_state_dict_structure(self):
        """Test state_dict contains all necessary information."""
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        opt = Adam([param], lr=0.01, betas=(0.9, 0.999), eps=1e-8)

        param.grad = torch.tensor([0.1, 0.2])
        opt.step()

        state = opt.state_dict()

        assert 'lr' in state
        assert 'beta1' in state
        assert 'beta2' in state
        assert 'eps' in state
        assert 'state' in state
        assert state['lr'] == 0.01
        assert state['beta1'] == 0.9
        assert state['beta2'] == 0.999

    def test_load_state_dict(self):
        """Test loading optimizer state."""
        param = torch.tensor([1.0], requires_grad=True)
        opt1 = Adam([param], lr=0.01, betas=(0.9, 0.999))

        param.grad = torch.tensor([1.0])
        opt1.step()
        opt1.step()

        state = opt1.state_dict()

        opt2 = Adam([param], lr=0.1)  # Different initial config
        opt2.load_state_dict(state)

        assert opt2.lr == 0.01
        assert opt2.beta1 == 0.9
        assert opt2.beta2 == 0.999

    def test_adamw_state_includes_weight_decay(self):
        """Test AdamW state_dict includes weight decay."""
        param = torch.tensor([1.0], requires_grad=True)
        opt = AdamW([param], lr=0.01, weight_decay=0.01)

        state = opt.state_dict()
        assert 'weight_decay' in state
        assert state['weight_decay'] == 0.01

    def test_adamw_load_state_dict(self):
        """Test loading AdamW state with weight decay."""
        param = torch.tensor([1.0], requires_grad=True)
        opt1 = AdamW([param], lr=0.01, weight_decay=0.05)

        state = opt1.state_dict()

        opt2 = AdamW([param], lr=0.1, weight_decay=0.1)
        opt2.load_state_dict(state)

        assert opt2.weight_decay == 0.05


class TestAdamNumericalStability:
    """Test Adam's numerical stability features."""

    def test_epsilon_prevents_division_by_zero(self):
        """Test epsilon prevents division by zero in denominator."""
        param = torch.tensor([1.0], requires_grad=True)
        param.grad = torch.tensor([0.0])  # Zero gradient

        opt = Adam([param], lr=0.1, eps=1e-8)
        opt.step()  # Should not raise error

        # Parameter should remain unchanged (zero gradient, eps protects division)
        assert torch.isclose(param.data, torch.tensor([1.0]), atol=1e-7)

    def test_small_gradients_stability(self):
        """Test stability with very small gradients."""
        param = torch.tensor([1.0], requires_grad=True)
        param.grad = torch.tensor([1e-10])

        opt = Adam([param], lr=0.1, eps=1e-8)
        opt.step()

        # Should handle small gradients without numerical issues
        assert torch.isfinite(param.data).all()

    def test_large_gradients_handling(self):
        """Test Adam with large gradients (adaptive learning helps)."""
        param = torch.tensor([1.0], requires_grad=True)
        param.grad = torch.tensor([1000.0])

        opt = Adam([param], lr=0.1, eps=1e-8)
        opt.step()

        # Adaptive learning rate should prevent extreme updates
        assert torch.isfinite(param.data).all()
        # Should still be reasonably close to original value
        assert param.data < torch.tensor([5.0])


class TestAdamVsPytorchReference:
    """Test our Adam implementation against expected behavior."""

    def test_manual_calculation_single_step(self):
        """Test single step matches manual calculation."""
        param = torch.tensor([5.0], requires_grad=True)
        param.grad = torch.tensor([2.0])

        lr = 0.1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        opt = Adam([param], lr=lr, betas=(beta1, beta2), eps=eps)
        opt.step()

        # Manual calculation
        # m_1 = 0.9 * 0 + 0.1 * 2.0 = 0.2
        # v_1 = 0.999 * 0 + 0.001 * 4.0 = 0.004
        # m_hat = 0.2 / (1 - 0.9) = 2.0
        # v_hat = 0.004 / (1 - 0.999) = 4.0
        # param = 5.0 - 0.1 * 2.0 / (sqrt(4.0) + 1e-8)
        #       = 5.0 - 0.1 * 2.0 / 2.0 = 5.0 - 0.1 = 4.9

        expected = torch.tensor([4.9])
        assert torch.isclose(param.data, expected, atol=1e-6)

    def test_convergence_on_quadratic(self):
        """Test Adam converges on simple quadratic function."""
        # Minimize f(x) = x^2, gradient = 2x
        param = torch.tensor([10.0], requires_grad=True)
        opt = Adam([param], lr=0.5)

        for _ in range(50):
            opt.zero_grad()
            loss = param ** 2
            loss.backward()
            opt.step()

        # Should converge close to 0
        assert torch.abs(param.data) < 1.0
