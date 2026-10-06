"""
SGD optimizer with momentum and weight decay.

Stochastic Gradient Descent with optional momentum (standard and Nesterov)
and weight decay for training neural networks.
"""

import numpy as np
from typing import Dict, Optional


class SGD:
    """
    Stochastic Gradient Descent optimizer with momentum.

    Supports standard momentum and Nesterov accelerated gradient (NAG).
    Weight decay can be applied as L2 regularization.

    Args:
        learning_rate: Learning rate (step size)
        momentum: Momentum factor (default: 0.0, no momentum)
        weight_decay: Weight decay (L2 penalty) coefficient (default: 0.0)
        nesterov: Whether to use Nesterov momentum (default: False)
        dampening: Dampening for momentum (default: 0.0)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        dampening: float = 0.0
    ):
        if learning_rate < 0.0:
            raise ValueError(f"Invalid learning rate: {learning_rate}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening = 0")

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.dampening = dampening

        # State: velocity (momentum buffer)
        self.velocity: Dict[str, np.ndarray] = {}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Perform a single optimization step.

        Args:
            params: Dictionary of parameters to update (modified in-place)
            grads: Dictionary of gradients corresponding to params
        """
        self.t += 1

        for name, param in params.items():
            if name not in grads:
                continue

            grad = grads[name]

            # Check for NaN or Inf in gradients
            if not np.isfinite(grad).all():
                raise ValueError(f"Gradient for {name} contains NaN or Inf values")

            # Apply weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param

            # Apply momentum
            if self.momentum != 0:
                if name not in self.velocity:
                    # Initialize velocity buffer
                    self.velocity[name] = np.zeros_like(param)
                    buf = self.velocity[name]
                    buf[:] = grad
                else:
                    buf = self.velocity[name]
                    # v_t = momentum * v_{t-1} + (1 - dampening) * g_t
                    buf[:] = self.momentum * buf + (1 - self.dampening) * grad

                if self.nesterov:
                    # Nesterov momentum: theta = theta - lr * (momentum * v_t + g_t)
                    grad = grad + self.momentum * buf
                else:
                    # Standard momentum: theta = theta - lr * v_t
                    grad = buf

            # Update parameters
            param -= self.learning_rate * grad

    def zero_grad(self) -> None:
        """Reset gradient accumulation (not needed in this implementation)."""
        pass

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        return {
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'nesterov': self.nesterov,
            'dampening': self.dampening,
            't': self.t,
            'velocity': {k: v.copy() for k, v in self.velocity.items()}
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state from checkpoint."""
        self.learning_rate = state_dict['learning_rate']
        self.momentum = state_dict['momentum']
        self.weight_decay = state_dict['weight_decay']
        self.nesterov = state_dict['nesterov']
        self.dampening = state_dict['dampening']
        self.t = state_dict['t']
        self.velocity = {k: v.copy() for k, v in state_dict['velocity'].items()}

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.learning_rate

    def set_lr(self, learning_rate: float) -> None:
        """Set learning rate (for learning rate scheduling)."""
        if learning_rate < 0.0:
            raise ValueError(f"Invalid learning rate: {learning_rate}")
        self.learning_rate = learning_rate


class SGDWithClipping(SGD):
    """
    SGD optimizer with gradient clipping.

    Adds gradient clipping capability to prevent exploding gradients.

    Args:
        learning_rate: Learning rate (step size)
        momentum: Momentum factor (default: 0.0)
        weight_decay: Weight decay coefficient (default: 0.0)
        nesterov: Whether to use Nesterov momentum (default: False)
        dampening: Dampening for momentum (default: 0.0)
        clip_norm: Maximum gradient norm for clipping (default: None, no clipping)
        clip_value: Maximum absolute value for gradient clipping (default: None)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        dampening: float = 0.0,
        clip_norm: Optional[float] = None,
        clip_value: Optional[float] = None
    ):
        super().__init__(learning_rate, momentum, weight_decay, nesterov, dampening)
        self.clip_norm = clip_norm
        self.clip_value = clip_value

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Perform a single optimization step with gradient clipping.

        Args:
            params: Dictionary of parameters to update (modified in-place)
            grads: Dictionary of gradients corresponding to params
        """
        # Apply gradient clipping
        if self.clip_value is not None:
            for name in grads:
                grads[name] = np.clip(grads[name], -self.clip_value, self.clip_value)

        if self.clip_norm is not None:
            # Compute total gradient norm
            total_norm = 0.0
            for grad in grads.values():
                total_norm += np.sum(grad ** 2)
            total_norm = np.sqrt(total_norm)

            # Clip gradients by norm
            if total_norm > self.clip_norm:
                clip_coef = self.clip_norm / (total_norm + 1e-8)
                for name in grads:
                    grads[name] = grads[name] * clip_coef

        # Call parent step
        super().step(params, grads)

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        state = super().state_dict()
        state['clip_norm'] = self.clip_norm
        state['clip_value'] = self.clip_value
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state from checkpoint."""
        super().load_state_dict(state_dict)
        self.clip_norm = state_dict.get('clip_norm')
        self.clip_value = state_dict.get('clip_value')
