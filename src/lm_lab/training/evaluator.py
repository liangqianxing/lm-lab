"""
Evaluation utilities for language models.
"""

import numpy as np
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
            Dictionary with metrics (loss, perplexity)
        """
        total_loss = 0.0
        total_tokens = 0

        for batch in self.data_loader:
            # Forward pass
            logits = self.model.forward(batch.input_ids)

            # Compute loss
            batch_loss = self._compute_loss(logits, batch.target_ids)
            batch_size, seq_len = batch.target_ids.shape

            total_loss += batch_loss * (batch_size * seq_len)
            total_tokens += batch_size * seq_len

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        ppl = np.exp(avg_loss) if avg_loss < 100 else float('inf')

        return {
            'loss': avg_loss,
            'perplexity': ppl,
            'num_tokens': total_tokens
        }

    def _compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model output (batch_size, seq_len, vocab_size)
            targets: Target token IDs (batch_size, seq_len)

        Returns:
            Average loss per token
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Numerical stability
        logits_max = np.max(logits, axis=-1, keepdims=True)
        logits_stable = logits - logits_max

        # Softmax
        exp_logits = np.exp(logits_stable)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Cross-entropy
        loss = 0.0
        for b in range(batch_size):
            for s in range(seq_len):
                target_id = targets[b, s]
                loss -= np.log(probs[b, s, target_id] + 1e-10)

        return loss / (batch_size * seq_len)


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
