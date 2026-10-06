"""
Evaluation utilities for language models.
"""

import numpy as np
import torch
from typing import Optional, Tuple
from .dataloader import TextDataLoader


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute cross-entropy loss with numerical stability.

    Args:
        logits: Model output (batch_size, seq_len, vocab_size)
        targets: Target token IDs (batch_size, seq_len)

    Returns:
        Tuple of (loss, gradient)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Numerical stability: subtract max
    logits_max = np.max(logits, axis=-1, keepdims=True)
    logits_stable = logits - logits_max

    # Softmax
    exp_logits = np.exp(logits_stable)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Cross-entropy loss
    loss = 0.0
    for b in range(batch_size):
        for s in range(seq_len):
            target_id = targets[b, s]
            loss -= np.log(probs[b, s, target_id] + 1e-10)

    loss = loss / (batch_size * seq_len)

    # Gradient: dL/d(logits) = (probs - one_hot(targets)) / (batch_size * seq_len)
    grad = probs.copy()
    for b in range(batch_size):
        for s in range(seq_len):
            target_id = targets[b, s]
            grad[b, s, target_id] -= 1.0

    grad = grad / (batch_size * seq_len)

    return loss, grad


class Evaluator:
    """
    Evaluate language model performance.

    Args:
        model: TransformerLM instance
        data_loader: Data loader for evaluation
    """

    def __init__(self, model, data_loader: TextDataLoader):
        self.model = model
        self.data_loader = data_loader

    def evaluate(self) -> dict:
        """
        Evaluate model on entire dataset.

        Returns:
            Dictionary with metrics (loss, perplexity, num_tokens)
        """
        total_loss = 0.0
        total_tokens = 0

        for batch in self.data_loader:
            # Convert to tensors
            input_ids = torch.from_numpy(batch.input_ids).long()
            target_ids = torch.from_numpy(batch.target_ids).long()

            # Forward pass
            with torch.no_grad():
                logits = self.model.forward(input_ids)

            # Compute loss
            batch_loss = self._compute_loss(logits, target_ids)
            batch_size, seq_len = target_ids.shape

            total_loss += batch_loss * (batch_size * seq_len)
            total_tokens += batch_size * seq_len

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        ppl = np.exp(avg_loss) if avg_loss < 100 else float('inf')

        return {
            'loss': float(avg_loss),
            'perplexity': float(ppl),
            'num_tokens': int(total_tokens)
        }

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model output (batch_size, seq_len, vocab_size)
            targets: Target token IDs (batch_size, seq_len)

        Returns:
            Average loss per token
        """
        # Reshape for cross entropy
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
        return loss.item()


def perplexity(model, data_loader: TextDataLoader) -> float:
    """
    Compute perplexity on a dataset.

    Args:
        model: TransformerLM instance
        data_loader: Data loader

    Returns:
        Perplexity value
    """
    evaluator = Evaluator(model, data_loader)
    metrics = evaluator.evaluate()
    return metrics['perplexity']
