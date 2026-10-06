"""
Adam and AdamW optimizer implementations using PyTorch.

Adam: Adaptive Moment Estimation
AdamW: Adam with decoupled weight decay
"""

import torch
from typing import List, Iterable


class Adam:
    """
    Adam optimizer.

    Combines momentum (first moment) and RMSprop (second moment) for
    adaptive learning rates per parameter.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8
    ):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        # Convert params to list
        self.params = list(params)

        # State: first moment (m), second moment (v), timestep
        self.state = {}
        for param in self.params:
            self.state[param] = {
                'm': torch.zeros_like(param),
                'v': torch.zeros_like(param),
                'step': 0
            }

    def zero_grad(self):
        """Zero out the gradients of all parameters."""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        """
        Perform a single optimization step.
        """
        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.data
            state = self.state[param]

            # Increment step
            state['step'] += 1

            # Update biased first moment estimate
            state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * grad

            # Update biased second moment estimate
            state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = state['m'] / (1 - self.beta1 ** state['step'])
            v_hat = state['v'] / (1 - self.beta2 ** state['step'])

            # Update parameters
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def state_dict(self):
        """Return optimizer state for checkpointing."""
        return {
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'state': {
                id(p): {
                    'step': s['step'],
                    'm': s['m'].clone(),
                    'v': s['v'].clone()
                } for p, s in self.state.items()
            }
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state from checkpoint."""
        self.lr = state_dict['lr']
        self.beta1 = state_dict['beta1']
        self.beta2 = state_dict['beta2']
        self.eps = state_dict['eps']
        # Note: This is simplified; full implementation would need param ID mapping


class AdamW(Adam):
    """
    AdamW optimizer with decoupled weight decay.

    Separates weight decay from gradient-based update for better
    regularization, especially with adaptive optimizers.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 1e-2)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2
    ):
        super().__init__(params, lr, betas, eps)
        self.weight_decay = weight_decay

    def step(self):
        """
        Perform a single optimization step with decoupled weight decay.
        """
        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.data
            state = self.state[param]

            # Increment step
            state['step'] += 1

            # Update biased first moment estimate
            state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * grad

            # Update biased second moment estimate
            state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = state['m'] / (1 - self.beta1 ** state['step'])
            v_hat = state['v'] / (1 - self.beta2 ** state['step'])

            # Update with Adam step
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

            # Decoupled weight decay
            param.data -= self.lr * self.weight_decay * param.data

    def state_dict(self):
        """Return optimizer state for checkpointing."""
        state = super().state_dict()
        state['weight_decay'] = self.weight_decay
        return state

    def load_state_dict(self, state_dict):
        """Load optimizer state from checkpoint."""
        super().load_state_dict(state_dict)
        self.weight_decay = state_dict.get('weight_decay', 1e-2)
