"""
Tests for SGD optimizer.

Tests SGD update rules, momentum accumulation, weight decay,
Nesterov momentum, and state persistence.
"""

import pytest
import numpy as np
from lm_lab.optimizer.sgd import SGD, SGDWithClipping


class TestSGDBasic:
    """Test basic SGD without momentum."""

    def test_simple_update(self):
        """Test basic SGD parameter update."""
        opt = SGD(learning_rate=0.1)
        params = {'w': np.array([1.0, 2.0, 3.0])}
        grads = {'w': np.array([0.1, 0.2, 0.3])}

        opt.step(params, grads)

        # Expected: w_new = w_old - lr * grad
        expected = np.array([1.0 - 0.1 * 0.1, 2.0 - 0.1 * 0.2, 3.0 - 0.1 * 0.3])
        np.testing.assert_allclose(params['w'], expected)

    def test_multiple_params(self):
        """Test SGD with multiple parameters."""
        opt = SGD(learning_rate=0.01)
        params = {
            'w1': np.array([[1.0, 2.0], [3.0, 4.0]]),
            'w2': np.array([5.0, 6.0])
        }
        grads = {
            'w1': np.array([[0.1, 0.2], [0.3, 0.4]]),
            'w2': np.array([0.5, 0.6])
        }

        opt.step(params, grads)

        expected_w1 = np.array([[1.0 - 0.01 * 0.1, 2.0 - 0.01 * 0.2],
                                [3.0 - 0.01 * 0.3, 4.0 - 0.01 * 0.4]])
        expected_w2 = np.array([5.0 - 0.01 * 0.5, 6.0 - 0.01 * 0.6])

        np.testing.assert_allclose(params['w1'], expected_w1)
        np.testing.assert_allclose(params['w2'], expected_w2)

    def test_invalid_learning_rate(self):
        """Test that negative learning rate raises error."""
        with pytest.raises(ValueError, match="Invalid learning rate"):
            SGD(learning_rate=-0.1)

    def test_nan_gradient(self):
        """Test that NaN gradients raise error."""
        opt = SGD(learning_rate=0.1)
        params = {'w': np.array([1.0, 2.0])}
        grads = {'w': np.array([np.nan, 0.1])}

        with pytest.raises(ValueError, match="contains NaN or Inf"):
            opt.step(params, grads)

    def test_inf_gradient(self):
        """Test that Inf gradients raise error."""
        opt = SGD(learning_rate=0.1)
        params = {'w': np.array([1.0, 2.0])}
        grads = {'w': np.array([np.inf, 0.1])}

        with pytest.raises(ValueError, match="contains NaN or Inf"):
            opt.step(params, grads)


class TestSGDMomentum:
    """Test SGD with momentum."""

    def test_momentum_accumulation(self):
        """Test momentum buffer accumulation over multiple steps."""
        opt = SGD(learning_rate=0.1, momentum=0.9)
        params = {'w': np.array([1.0])}

        # Step 1: v_0 = 0, g_1 = 1.0
        # v_1 = g_1 = 1.0 (first step initializes to gradient)
        # w_1 = 1.0 - 0.1 * 1.0 = 0.9
        grads = {'w': np.array([1.0])}
        opt.step(params, grads)
        assert np.isclose(params['w'][0], 0.9)

        # Step 2: g_2 = 1.0
        # v_2 = 0.9 * 1.0 + (1 - 0) * 1.0 = 1.9
        # w_2 = 0.9 - 0.1 * 1.9 = 0.71
        grads = {'w': np.array([1.0])}
        opt.step(params, grads)
        assert np.isclose(params['w'][0], 0.71)

        # Step 3: g_3 = 1.0
        # v_3 = 0.9 * 1.9 + 1.0 = 2.71
        # w_3 = 0.71 - 0.1 * 2.71 = 0.439
        grads = {'w': np.array([1.0])}
        opt.step(params, grads)
        assert np.isclose(params['w'][0], 0.439)

    def test_momentum_with_dampening(self):
        """Test momentum with dampening."""
        opt = SGD(learning_rate=0.1, momentum=0.9, dampening=0.5)
        params = {'w': np.array([1.0])}

        # Step 1: v_1 = g_1 = 0.5 (initial)
        grads = {'w': np.array([0.5])}
        opt.step(params, grads)
        w1 = 1.0 - 0.1 * 0.5
        assert np.isclose(params['w'][0], w1)

        # Step 2: v_2 = 0.9 * 0.5 + (1 - 0.5) * 0.5 = 0.45 + 0.25 = 0.7
        grads = {'w': np.array([0.5])}
        opt.step(params, grads)
        w2 = w1 - 0.1 * 0.7
        assert np.isclose(params['w'][0], w2)

    def test_nesterov_momentum(self):
        """Test Nesterov accelerated gradient."""
        opt = SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
        params = {'w': np.array([1.0])}

        # Step 1: v_1 = g_1 = 1.0
        # Nesterov: grad = g_1 + momentum * v_1 = 1.0 + 0.9 * 1.0 = 1.9
        # w_1 = 1.0 - 0.1 * 1.9 = 0.81
        grads = {'w': np.array([1.0])}
        opt.step(params, grads)
        assert np.isclose(params['w'][0], 0.81)

        # Step 2: v_2 = 0.9 * 1.0 + 1.0 = 1.9
        # grad = 1.0 + 0.9 * 1.9 = 2.71
        # w_2 = 0.81 - 0.1 * 2.71 = 0.539
        grads = {'w': np.array([1.0])}
        opt.step(params, grads)
        assert np.isclose(params['w'][0], 0.539)

    def test_nesterov_validation(self):
        """Test Nesterov momentum validation."""
        # Nesterov requires momentum > 0
        with pytest.raises(ValueError, match="Nesterov momentum requires"):
            SGD(learning_rate=0.1, momentum=0.0, nesterov=True)

        # Nesterov requires dampening = 0
        with pytest.raises(ValueError, match="Nesterov momentum requires"):
            SGD(learning_rate=0.1, momentum=0.9, dampening=0.1, nesterov=True)


class TestSGDWeightDecay:
    """Test SGD with weight decay (L2 regularization)."""

    def test_weight_decay(self):
        """Test L2 weight decay application."""
        opt = SGD(learning_rate=0.1, weight_decay=0.01)
        params = {'w': np.array([1.0, 2.0])}
        grads = {'w': np.array([0.1, 0.2])}

        opt.step(params, grads)

        # With weight decay: grad_eff = grad + weight_decay * param
        # grad_eff = [0.1 + 0.01 * 1.0, 0.2 + 0.01 * 2.0] = [0.11, 0.22]
        # w_new = [1.0 - 0.1 * 0.11, 2.0 - 0.1 * 0.22] = [0.989, 1.978]
        expected = np.array([1.0 - 0.1 * 0.11, 2.0 - 0.1 * 0.22])
        np.testing.assert_allclose(params['w'], expected)

    def test_weight_decay_with_momentum(self):
        """Test weight decay combined with momentum."""
        opt = SGD(learning_rate=0.1, momentum=0.9, weight_decay=0.01)
        params = {'w': np.array([1.0])}

        # Step 1
        grads = {'w': np.array([0.5])}
        opt.step(params, grads)
        # grad_eff = 0.5 + 0.01 * 1.0 = 0.51
        # v_1 = 0.51
        # w_1 = 1.0 - 0.1 * 0.51 = 0.949
        assert np.isclose(params['w'][0], 0.949)


class TestSGDGradientClipping:
    """Test SGD with gradient clipping."""

    def test_clip_by_value(self):
        """Test gradient clipping by value."""
        opt = SGDWithClipping(learning_rate=0.1, clip_value=0.5)
        params = {'w': np.array([1.0, 2.0, 3.0])}
        grads = {'w': np.array([2.0, -1.0, 0.3])}

        opt.step(params, grads)

        # Gradients clipped to [-0.5, 0.5]
        # clipped_grads = [0.5, -0.5, 0.3]
        expected = np.array([1.0 - 0.1 * 0.5, 2.0 - 0.1 * (-0.5), 3.0 - 0.1 * 0.3])
        np.testing.assert_allclose(params['w'], expected)

    def test_clip_by_norm(self):
        """Test gradient clipping by global norm."""
        opt = SGDWithClipping(learning_rate=0.1, clip_norm=1.0)
        params = {
            'w1': np.array([1.0]),
            'w2': np.array([1.0])
        }
        grads = {
            'w1': np.array([3.0]),  # norm = 3
            'w2': np.array([4.0])   # norm = 4, total = 5
        }

        opt.step(params, grads)

        # Total norm = sqrt(3^2 + 4^2) = 5.0
        # Clip coefficient = 1.0 / 5.0 = 0.2
        # Clipped grads: w1 = 0.6, w2 = 0.8
        expected_w1 = np.array([1.0 - 0.1 * 0.6])
        expected_w2 = np.array([1.0 - 0.1 * 0.8])
        np.testing.assert_allclose(params['w1'], expected_w1)
        np.testing.assert_allclose(params['w2'], expected_w2)

    def test_no_clipping_when_below_threshold(self):
        """Test that small gradients are not clipped."""
        opt = SGDWithClipping(learning_rate=0.1, clip_norm=10.0)
        params = {'w': np.array([1.0, 2.0])}
        grads = {'w': np.array([0.1, 0.2])}

        opt.step(params, grads)

        # Norm = sqrt(0.1^2 + 0.2^2) ≈ 0.224, no clipping
        expected = np.array([1.0 - 0.1 * 0.1, 2.0 - 0.1 * 0.2])
        np.testing.assert_allclose(params['w'], expected)


class TestSGDStatePersistence:
    """Test optimizer state save/load."""

    def test_state_dict_basic(self):
        """Test state_dict captures optimizer configuration."""
        opt = SGD(learning_rate=0.01, momentum=0.9, weight_decay=0.001)
        state = opt.state_dict()

        assert state['learning_rate'] == 0.01
        assert state['momentum'] == 0.9
        assert state['weight_decay'] == 0.001
        assert state['t'] == 0

    def test_state_dict_with_velocity(self):
        """Test state_dict captures velocity buffers."""
        opt = SGD(learning_rate=0.1, momentum=0.9)
        params = {'w': np.array([1.0, 2.0])}
        grads = {'w': np.array([0.1, 0.2])}

        opt.step(params, grads)

        state = opt.state_dict()
        assert 'velocity' in state
        assert 'w' in state['velocity']
        np.testing.assert_allclose(state['velocity']['w'], np.array([0.1, 0.2]))
        assert state['t'] == 1

    def test_load_state_dict(self):
        """Test loading optimizer state."""
        opt1 = SGD(learning_rate=0.1, momentum=0.9)
        params = {'w': np.array([1.0, 2.0])}
        grads = {'w': np.array([0.5, 0.5])}

        # Run a few steps
        opt1.step(params, grads)
        opt1.step(params, grads)

        # Save state
        state = opt1.state_dict()

        # Create new optimizer and load state
        opt2 = SGD(learning_rate=0.01)  # Different initial lr
        opt2.load_state_dict(state)

        assert opt2.learning_rate == 0.1
        assert opt2.momentum == 0.9
        assert opt2.t == 2
        assert 'w' in opt2.velocity
        np.testing.assert_array_equal(opt2.velocity['w'], opt1.velocity['w'])

    def test_state_persistence_continues_training(self):
        """Test that loaded state continues training correctly."""
        params1 = {'w': np.array([1.0])}
        params2 = {'w': np.array([1.0])}
        grads = {'w': np.array([0.5])}

        # Run 3 steps continuously
        opt_continuous = SGD(learning_rate=0.1, momentum=0.9)
        opt_continuous.step(params1, grads)
        opt_continuous.step(params1, grads)
        opt_continuous.step(params1, grads)

        # Run 2 steps, save, load, run 1 more
        opt_saved = SGD(learning_rate=0.1, momentum=0.9)
        opt_saved.step(params2, grads)
        opt_saved.step(params2, grads)

        state = opt_saved.state_dict()
        opt_loaded = SGD(learning_rate=0.1, momentum=0.9)
        opt_loaded.load_state_dict(state)
        opt_loaded.step(params2, grads)

        # Both should reach the same final state
        np.testing.assert_allclose(params1['w'], params2['w'], rtol=1e-10)

    def test_clipping_state_persistence(self):
        """Test state persistence for SGDWithClipping."""
        opt = SGDWithClipping(learning_rate=0.1, clip_norm=1.0, clip_value=0.5)
        state = opt.state_dict()

        assert state['clip_norm'] == 1.0
        assert state['clip_value'] == 0.5

        opt2 = SGDWithClipping(learning_rate=0.01)
        opt2.load_state_dict(state)
        assert opt2.clip_norm == 1.0
        assert opt2.clip_value == 0.5


class TestSGDLearningRateScheduling:
    """Test learning rate get/set for scheduling."""

    def test_get_lr(self):
        """Test getting current learning rate."""
        opt = SGD(learning_rate=0.01)
        assert opt.get_lr() == 0.01

    def test_set_lr(self):
        """Test setting learning rate."""
        opt = SGD(learning_rate=0.01)
        opt.set_lr(0.001)
        assert opt.get_lr() == 0.001

    def test_set_lr_invalid(self):
        """Test that negative learning rate raises error."""
        opt = SGD(learning_rate=0.01)
        with pytest.raises(ValueError, match="Invalid learning rate"):
            opt.set_lr(-0.1)

    def test_lr_scheduling_affects_updates(self):
        """Test that changing learning rate affects parameter updates."""
        params1 = {'w': np.array([1.0])}
        params2 = {'w': np.array([1.0])}
        grads = {'w': np.array([1.0])}

        opt1 = SGD(learning_rate=0.1)
        opt1.step(params1, grads)

        opt2 = SGD(learning_rate=0.1)
        opt2.set_lr(0.05)
        opt2.step(params2, grads)

        # opt1: 1.0 - 0.1 * 1.0 = 0.9
        # opt2: 1.0 - 0.05 * 1.0 = 0.95
        assert np.isclose(params1['w'][0], 0.9)
        assert np.isclose(params2['w'][0], 0.95)
