"""
Adam and AdamW optimizer implementations.

Adam: Adaptive Moment Estimation
AdamW: Adam with decoupled weight decay
"""

import numpy as np
from typing import Dict, Optional


class Adam:
    """
    Adam optimizer.

    Combines momentum (first moment) and RMSprop (second moment) for
    adaptive learning rates per parameter.

    Args:
        learning_rate: Learning rate (alpha)
        beta1: Exponential decay rate for first moment (default: 0.9)
        beta2: Exponential decay rate for second moment (default: 0.999)
        eps: Small constant for numerical stability (default: 1e-8)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # State: first moment (m), second moment (v), timestep
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
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

            # Initialize moments if first time
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)

            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad

            # Update biased second moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Reset gradient accumulation (not needed in this implementation)."""
        pass

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            't': self.t,
            'm': {k: v.copy() for k, v in self.m.items()},
            'v': {k: v.copy() for k, v in self.v.items()}
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state from checkpoint."""
        self.learning_rate = state_dict['learning_rate']
        self.beta1 = state_dict['beta1']
        self.beta2 = state_dict['beta2']
        self.eps = state_dict['eps']
        self.t = state_dict['t']
        self.m = {k: v.copy() for k, v in state_dict['m'].items()}
        self.v = {k: v.copy() for k, v in state_dict['v'].items()}


class AdamW(Adam):
    """
    AdamW optimizer with decoupled weight decay.

    Separates weight decay from gradient-based update for better
    regularization, especially with adaptive optimizers.

    Args:
        learning_rate: Learning rate (alpha)
        beta1: Exponential decay rate for first moment (default: 0.9)
        beta2: Exponential decay rate for second moment (default: 0.999)
        eps: Small constant for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        super().__init__(learning_rate, beta1, beta2, eps)
        self.weight_decay = weight_decay

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Perform a single optimization step with decoupled weight decay.

        Args:
            params: Dictionary of parameters to update (modified in-place)
            grads: Dictionary of gradients corresponding to params
        """
        self.t += 1

        for name, param in params.items():
            if name not in grads:
                continue

            grad = grads[name]

            # Initialize moments if first time
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)

            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad

            # Update biased second moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Update with Adam step
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

            # Decoupled weight decay
            param -= self.learning_rate * self.weight_decay * param

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        state = super().state_dict()
        state['weight_decay'] = self.weight_decay
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state from checkpoint."""
        super().load_state_dict(state_dict)
        self.weight_decay = state_dict.get('weight_decay', 0.01)
