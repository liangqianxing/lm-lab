"""
Tests for gradient clipping utilities.

Tests clip_grad_norm and clip_grad_value functions with various
norm types and edge cases.
"""

import pytest
import numpy as np
from lm_lab.optimizer.gradient import clip_grad_norm, clip_grad_value


class TestClipGradNorm:
    """Test gradient clipping by global norm."""

    def test_l2_norm_clipping(self):
        """Test L2 norm gradient clipping."""
        grad1 = np.array([3.0, 4.0])
        grad2 = np.array([0.0, 0.0])
        gradients = [grad1, grad2]

        # Total L2 norm = sqrt(3^2 + 4^2) = 5.0
        total_norm = clip_grad_norm(gradients, max_norm=2.0, norm_type=2.0)

        assert np.isclose(total_norm, 5.0)
        # Clip coefficient = 2.0 / 5.0 = 0.4
        expected_grad1 = np.array([3.0 * 0.4, 4.0 * 0.4])
        np.testing.assert_allclose(grad1, expected_grad1, rtol=1e-6)

    def test_no_clipping_when_below_threshold(self):
        """Test gradients are not clipped when below max_norm."""
        grad1 = np.array([0.1, 0.2])
        grad2 = np.array([0.1, 0.1])
        gradients = [grad1, grad2]

        original_grad1 = grad1.copy()
        original_grad2 = grad2.copy()

        # Total norm ~ 0.283, max_norm = 10.0
        total_norm = clip_grad_norm(gradients, max_norm=10.0, norm_type=2.0)

        # Gradients should remain unchanged
        np.testing.assert_allclose(grad1, original_grad1)
        np.testing.assert_allclose(grad2, original_grad2)
        assert total_norm < 10.0

    def test_infinity_norm_clipping(self):
        """Test infinity norm (max absolute value) clipping."""
        grad1 = np.array([1.0, 5.0, 2.0])
        grad2 = np.array([-3.0, 0.5])
        gradients = [grad1, grad2]

        # Infinity norm = max(|5.0|, |-3.0|, ...) = 5.0
        total_norm = clip_grad_norm(gradients, max_norm=2.0, norm_type=float('inf'))

        assert np.isclose(total_norm, 5.0)
        # Clip coefficient = 2.0 / 5.0 = 0.4
        expected_grad1 = np.array([1.0 * 0.4, 5.0 * 0.4, 2.0 * 0.4])
        expected_grad2 = np.array([-3.0 * 0.4, 0.5 * 0.4])
        np.testing.assert_allclose(grad1, expected_grad1, rtol=1e-6)
        np.testing.assert_allclose(grad2, expected_grad2, rtol=1e-6)

    def test_l1_norm_clipping(self):
        """Test L1 norm gradient clipping."""
        grad1 = np.array([1.0, 2.0, 3.0])
        gradients = [grad1]

        # L1 norm = |1| + |2| + |3| = 6.0
        total_norm = clip_grad_norm(gradients, max_norm=3.0, norm_type=1.0)

        assert np.isclose(total_norm, 6.0)
        # Clip coefficient = 3.0 / 6.0 = 0.5
        expected = np.array([0.5, 1.0, 1.5])
        np.testing.assert_allclose(grad1, expected, rtol=1e-6)

    def test_none_gradients_skipped(self):
        """Test that None gradients are skipped."""
        grad1 = np.array([3.0, 4.0])
        grad2 = None
        grad3 = np.array([0.0, 0.0])
        gradients = [grad1, grad2, grad3]

        # Should only consider grad1 and grad3
        total_norm = clip_grad_norm(gradients, max_norm=2.0, norm_type=2.0)

        assert np.isclose(total_norm, 5.0)
        expected_grad1 = np.array([1.2, 1.6])  # 3.0 * 0.4, 4.0 * 0.4
        np.testing.assert_allclose(grad1, expected_grad1, rtol=1e-6)

    def test_multidimensional_gradients(self):
        """Test clipping with multi-dimensional gradient arrays."""
        grad1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        grad2 = np.array([1.0, 1.0, 1.0])
        gradients = [grad1, grad2]

        # Total L2 norm = sqrt(1 + 4 + 9 + 16 + 1 + 1 + 1) = sqrt(33) ≈ 5.745
        total_norm = clip_grad_norm(gradients, max_norm=3.0, norm_type=2.0)

        assert np.isclose(total_norm, np.sqrt(33))
        clip_coef = 3.0 / np.sqrt(33)
        expected_grad1 = np.array([[1.0, 2.0], [3.0, 4.0]]) * clip_coef
        np.testing.assert_allclose(grad1, expected_grad1, rtol=1e-6)

    def test_zero_gradients(self):
        """Test clipping with all-zero gradients."""
        grad1 = np.array([0.0, 0.0])
        grad2 = np.array([0.0])
        gradients = [grad1, grad2]

        total_norm = clip_grad_norm(gradients, max_norm=1.0, norm_type=2.0)

        assert np.isclose(total_norm, 0.0)
        # Zero gradients should remain zero
        np.testing.assert_allclose(grad1, np.array([0.0, 0.0]))
        np.testing.assert_allclose(grad2, np.array([0.0]))

    def test_single_gradient(self):
        """Test clipping with single gradient."""
        grad = np.array([5.0, 12.0])  # Norm = 13.0
        gradients = [grad]

        total_norm = clip_grad_norm(gradients, max_norm=1.0, norm_type=2.0)

        assert np.isclose(total_norm, 13.0)
        expected = np.array([5.0 / 13.0, 12.0 / 13.0])
        np.testing.assert_allclose(grad, expected)

    def test_return_value_is_original_norm(self):
        """Test that return value is the norm before clipping."""
        grad = np.array([3.0, 4.0])
        gradients = [grad]

        original_norm = np.linalg.norm(grad)
        returned_norm = clip_grad_norm(gradients, max_norm=100.0, norm_type=2.0)

        assert np.isclose(returned_norm, original_norm)
        assert np.isclose(returned_norm, 5.0)


class TestClipGradValue:
    """Test gradient clipping by absolute value."""

    def test_basic_clipping(self):
        """Test basic value clipping."""
        grad = np.array([5.0, -3.0, 0.5, -0.2])
        gradients = [grad]

        clip_grad_value(gradients, clip_value=2.0)

        expected = np.array([2.0, -2.0, 0.5, -0.2])
        np.testing.assert_allclose(grad, expected)

    def test_symmetric_clipping(self):
        """Test clipping is symmetric around zero."""
        grad1 = np.array([10.0])
        grad2 = np.array([-10.0])
        gradients1 = [grad1]
        gradients2 = [grad2]

        clip_grad_value(gradients1, clip_value=3.0)
        clip_grad_value(gradients2, clip_value=3.0)

        assert np.isclose(grad1[0], 3.0)
        assert np.isclose(grad2[0], -3.0)

    def test_no_clipping_within_range(self):
        """Test values within range are unchanged."""
        grad = np.array([0.5, -0.3, 1.0, -1.5])
        original = grad.copy()
        gradients = [grad]

        clip_grad_value(gradients, clip_value=2.0)

        np.testing.assert_allclose(grad, original)

    def test_multiple_gradients(self):
        """Test clipping multiple gradient arrays."""
        grad1 = np.array([10.0, -5.0])
        grad2 = np.array([[3.0, -8.0], [0.5, 15.0]])
        gradients = [grad1, grad2]

        clip_grad_value(gradients, clip_value=4.0)

        expected_grad1 = np.array([4.0, -4.0])
        expected_grad2 = np.array([[3.0, -4.0], [0.5, 4.0]])
        np.testing.assert_allclose(grad1, expected_grad1, rtol=1e-6)
        np.testing.assert_allclose(grad2, expected_grad2, rtol=1e-6)

    def test_none_gradients_skipped(self):
        """Test that None gradients are skipped."""
        grad1 = np.array([5.0, -5.0])
        grad2 = None
        grad3 = np.array([3.0])
        gradients = [grad1, grad2, grad3]

        clip_grad_value(gradients, clip_value=2.0)

        expected_grad1 = np.array([2.0, -2.0])
        expected_grad3 = np.array([2.0])
        np.testing.assert_allclose(grad1, expected_grad1, rtol=1e-6)
        np.testing.assert_allclose(grad3, expected_grad3)

    def test_zero_clip_value(self):
        """Test clipping to zero."""
        grad = np.array([1.0, -2.0, 3.0])
        gradients = [grad]

        clip_grad_value(gradients, clip_value=0.0)

        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(grad, expected)

    def test_large_clip_value(self):
        """Test that large clip_value doesn't affect reasonable gradients."""
        grad = np.array([1.0, -5.0, 10.0, -20.0])
        original = grad.copy()
        gradients = [grad]

        clip_grad_value(gradients, clip_value=1000.0)

        np.testing.assert_allclose(grad, original)

    def test_inplace_modification(self):
        """Test that clipping modifies arrays in-place."""
        grad = np.array([10.0, -10.0])
        gradients = [grad]
        original_id = id(grad)

        clip_grad_value(gradients, clip_value=5.0)

        # Should modify the same array, not create a new one
        assert id(grad) == original_id
        np.testing.assert_allclose(grad, np.array([5.0, -5.0]))


class TestClippingEdgeCases:
    """Test edge cases for gradient clipping."""

    def test_empty_gradient_list(self):
        """Test clipping with empty gradient list."""
        gradients = []

        # clip_grad_value should handle empty list
        clip_grad_value(gradients, clip_value=1.0)

        # clip_grad_norm may have issues with empty list
        # depending on implementation

    def test_all_none_gradients(self):
        """Test clipping when all gradients are None."""
        gradients = [None, None, None]

        clip_grad_value(gradients, clip_value=1.0)
        # Should not raise error

    def test_very_small_norm(self):
        """Test norm clipping with very small gradients."""
        grad = np.array([1e-10, 1e-10])
        gradients = [grad]

        total_norm = clip_grad_norm(gradients, max_norm=1.0, norm_type=2.0)

        # Should handle small norms without numerical issues
        assert total_norm < 1e-9
        assert np.all(np.isfinite(grad))

    def test_mixed_dimensions(self):
        """Test clipping with mixed-dimension gradients."""
        grad1 = np.array([1.0])  # 1D
        grad2 = np.array([[2.0, 3.0]])  # 2D
        grad3 = np.array([[[4.0, 5.0]]])  # 3D
        gradients = [grad1, grad2, grad3]

        total_norm = clip_grad_norm(gradients, max_norm=3.0, norm_type=2.0)

        # Total norm = sqrt(1 + 4 + 9 + 16 + 25) = sqrt(55) ≈ 7.416
        assert np.isclose(total_norm, np.sqrt(55))
        assert np.all(np.isfinite(grad1))
        assert np.all(np.isfinite(grad2))
        assert np.all(np.isfinite(grad3))

    def test_negative_zero_handling(self):
        """Test that -0.0 and 0.0 are handled correctly."""
        grad = np.array([0.0, -0.0, 0.0])
        gradients = [grad]

        clip_grad_value(gradients, clip_value=1.0)

        # Should remain zero (both positive and negative zero)
        assert np.all(grad == 0.0)


class TestClippingIntegration:
    """Test gradient clipping in realistic scenarios."""

    def test_exploding_gradients_scenario(self):
        """Test clipping prevents exploding gradients."""
        # Simulate exploding gradients
        grads = [
            np.array([1000.0, 2000.0]),
            np.array([3000.0, 4000.0]),
            np.array([5000.0])
        ]

        total_norm = clip_grad_norm(grads, max_norm=1.0, norm_type=2.0)

        # Original norm should be very large
        assert total_norm > 7000

        # After clipping, total norm should be approximately 1.0
        clipped_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        assert np.isclose(clipped_norm, 1.0, rtol=1e-3)

    def test_vanishing_gradients_preserved(self):
        """Test that small gradients are preserved when appropriate."""
        grads = [
            np.array([1e-5, 2e-5]),
            np.array([3e-5])
        ]
        original_grads = [g.copy() for g in grads]

        clip_grad_norm(grads, max_norm=1.0, norm_type=2.0)

        # Small gradients should not be clipped
        for original, clipped in zip(original_grads, grads):
            np.testing.assert_allclose(original, clipped)

    def test_combining_norm_and_value_clipping(self):
        """Test applying both norm and value clipping sequentially."""
        grad = np.array([10.0, 20.0, 30.0])
        gradients = [grad]

        # First clip by value
        clip_grad_value(gradients, clip_value=15.0)
        assert np.max(np.abs(grad)) <= 15.0

        # Then clip by norm
        total_norm = clip_grad_norm(gradients, max_norm=10.0, norm_type=2.0)
        clipped_norm = np.linalg.norm(grad)
        assert clipped_norm <= 10.0 + 1e-6
