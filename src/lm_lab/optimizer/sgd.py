"""
SGD optimizer with momentum.

Implements stochastic gradient descent with optional momentum and weight decay.
"""

import numpy as np
from typing import List


class SGD:
    """
    Stochastic Gradient Descent optimizer with momentum.

    Args:
        learning_rate: Learning rate (default 0.01)
        momentum: Momentum factor (default 0.0, no momentum)
        weight_decay: L2 penalty coefficient (default 0.0)
        nesterov: Whether to use Nesterov momentum (default False)

    Example:
        >>> optimizer = SGD(learning_rate=0.01, momentum=0.9)
        >>> optimizer.step(parameters, gradients)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # Velocity for momentum
        self.velocity = {}

    def step(self, parameters: List[np.ndarray], gradients: List[np.ndarray]) -> None:
        """
        Perform a single optimization step.

        Args:
            parameters: List of parameter arrays
            gradients: List of gradient arrays
        """
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            if grad is None:
                continue

            # Apply weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            # Initialize velocity on first step
            if i not in self.velocity:
                self.velocity[i] = np.zeros_like(param)

            # Update velocity
            if self.momentum > 0:
                self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad

                # Nesterov momentum
                if self.nesterov:
                    param += self.momentum * self.velocity[i] - self.learning_rate * grad
                else:
                    param += self.velocity[i]
            else:
                # Standard SGD without momentum
                param -= self.learning_rate * grad

    def zero_grad(self) -> None:
        """Reset optimizer state."""
        self.velocity = {}

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.learning_rate

    def set_lr(self, lr: float) -> None:
        """Set learning rate."""
        self.learning_rate = lr
