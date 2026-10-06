"""
Main training loop for language models.

Provides the Trainer class that orchestrates model training, evaluation,
checkpointing, and generation.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import time

from ..model.transformer import TransformerLM
from ..optimizer.adam import AdamW
from .dataloader import TextDataLoader, Batch
from .checkpoint import CheckpointManager
from .evaluator import cross_entropy_loss
from .sampler import generate


class LanguageModelingDataset:
    """
    Simple dataset wrapper for tokenized text.

    Args:
        tokens: Array of token IDs
    """

    def __init__(self, tokens: np.ndarray):
        self.tokens = tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> int:
        return self.tokens[idx]


def create_dataloader(
    tokens: np.ndarray,
    batch_size: int,
    seq_len: int,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> TextDataLoader:
    """
    Create a dataloader from tokens.

    Args:
        tokens: Array of token IDs
        batch_size: Batch size
        seq_len: Sequence length
        shuffle: Whether to shuffle
        seed: Random seed

    Returns:
        TextDataLoader instance
    """
    return TextDataLoader(tokens, batch_size, seq_len, shuffle, seed)


class Trainer:
    """
    Trainer for language models.

    Handles the complete training pipeline including:
    - Training loop with gradient computation
    - Checkpointing
    - Evaluation
    - Text generation

    Args:
        model: TransformerLM model
        optimizer: Optimizer (e.g., AdamW)
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
        checkpoint_dir: Directory for checkpoints
        log_interval: Steps between logging
        eval_interval: Steps between evaluation
        save_interval: Steps between checkpoints
    """

    def __init__(
        self,
        model: TransformerLM,
        optimizer: AdamW,
        train_dataloader: TextDataLoader,
        val_dataloader: Optional[TextDataLoader] = None,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
        eval_interval: int = 100,
        save_interval: int = 1000
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        self.step = 0
        self.epoch = 0

    def train(
        self,
        num_steps: Optional[int] = None,
        num_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            num_steps: Number of training steps (exclusive with num_epochs)
            num_epochs: Number of epochs (exclusive with num_steps)

        Returns:
            Training history dictionary
        """
        if num_steps is None and num_epochs is None:
            raise ValueError("Must specify either num_steps or num_epochs")

        if num_steps is not None and num_epochs is not None:
            raise ValueError("Cannot specify both num_steps and num_epochs")

        history = {
            'train_loss': [],
            'val_loss': [],
            'steps': [],
            'epochs': []
        }

        if num_epochs is not None:
            for epoch in range(num_epochs):
                self.epoch = epoch
                epoch_history = self._train_epoch()
                history['train_loss'].extend(epoch_history['train_loss'])
                history['val_loss'].extend(epoch_history['val_loss'])
                history['steps'].extend(epoch_history['steps'])
                history['epochs'].extend(epoch_history['epochs'])
        else:
            # Train for fixed number of steps
            while self.step < num_steps:
                batch_history = self._train_step_loop(num_steps)
                history['train_loss'].extend(batch_history['train_loss'])
                history['val_loss'].extend(batch_history['val_loss'])
                history['steps'].extend(batch_history['steps'])
                history['epochs'].extend(batch_history['epochs'])

        return history

    def _train_epoch(self) -> Dict[str, Any]:
        """Train for one epoch."""
        history = {
            'train_loss': [],
            'val_loss': [],
            'steps': [],
            'epochs': []
        }

        for batch in self.train_dataloader:
            loss = self._train_step(batch)

            if self.step % self.log_interval == 0:
                history['train_loss'].append(loss)
                history['steps'].append(self.step)
                history['epochs'].append(self.epoch)
                print(f"Step {self.step}, Epoch {self.epoch}, Loss: {loss:.4f}")

            if self.val_dataloader and self.step % self.eval_interval == 0:
                val_loss = self._evaluate()
                history['val_loss'].append(val_loss)
                print(f"Step {self.step}, Val Loss: {val_loss:.4f}")

            if self.step % self.save_interval == 0:
                self._save_checkpoint(loss)

            self.step += 1

        return history

    def _train_step_loop(self, max_steps: int) -> Dict[str, Any]:
        """Train until reaching max_steps."""
        history = {
            'train_loss': [],
            'val_loss': [],
            'steps': [],
            'epochs': []
        }

        for batch in self.train_dataloader:
            if self.step >= max_steps:
                break

            loss = self._train_step(batch)

            if self.step % self.log_interval == 0:
                history['train_loss'].append(loss)
                history['steps'].append(self.step)
                history['epochs'].append(self.epoch)
                print(f"Step {self.step}, Loss: {loss:.4f}")

            if self.val_dataloader and self.step % self.eval_interval == 0:
                val_loss = self._evaluate()
                history['val_loss'].append(val_loss)
                print(f"Step {self.step}, Val Loss: {val_loss:.4f}")

            if self.step % self.save_interval == 0:
                self._save_checkpoint(loss)

            self.step += 1

        return history

    def _train_step(self, batch: Batch) -> float:
        """
        Perform single training step.

        Args:
            batch: Training batch

        Returns:
            Loss value
        """
        # Forward pass
        logits = self.model.forward(batch.input_ids)

        # Compute loss
        loss = cross_entropy_loss(logits, batch.target_ids)

        # Backward pass
        grad_logits = self._loss_backward(logits, batch.target_ids)
        grads = self.model.backward(grad_logits)

        # Check for non-finite gradients
        if not all(np.isfinite(g).all() for g in grads.values()):
            print(f"Warning: Non-finite gradients at step {self.step}")
            return loss

        # Update parameters
        params = self.model.parameters()
        self.optimizer.step(params, grads)

        return loss

    def _loss_backward(
        self,
        logits: np.ndarray,
        targets: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss w.r.t. logits.

        Args:
            logits: Model logits (batch_size, seq_len, vocab_size)
            targets: Target token IDs (batch_size, seq_len)

        Returns:
            Gradient w.r.t. logits
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Compute softmax probabilities
        logits_flat = logits.reshape(-1, vocab_size)
        probs = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)

        # Create one-hot targets
        targets_flat = targets.reshape(-1)
        targets_one_hot = np.zeros_like(probs)
        targets_one_hot[np.arange(len(targets_flat)), targets_flat] = 1.0

        # Gradient: (probs - targets) / batch_size
        grad = (probs - targets_one_hot) / (batch_size * seq_len)

        return grad.reshape(batch_size, seq_len, vocab_size)

    def _evaluate(self) -> float:
        """Evaluate on validation set."""
        if self.val_dataloader is None:
            return 0.0

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            logits = self.model.forward(batch.input_ids)
            loss = cross_entropy_loss(logits, batch.target_ids)
            total_loss += loss
            num_batches += 1

            # Limit evaluation batches
            if num_batches >= 10:
                break

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, loss: float) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_manager.save(
            step=self.step,
            model_params=self.model.parameters(),
            optimizer_state=self.optimizer.state_dict(),
            metadata={
                'loss': float(loss),
                'epoch': self.epoch
            }
        )
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        data = self.checkpoint_manager.load(checkpoint_path)

        # Load model parameters
        model_params = self.model.parameters()
        for name, param in data['model_params'].items():
            if name in model_params:
                model_params[name][:] = param

        # Load optimizer state
        self.optimizer.load_state_dict(data['optimizer_state'])

        # Load training state
        self.step = data['metadata'].get('step', 0)
        self.epoch = data['metadata'].get('epoch', 0)

        print(f"Loaded checkpoint from step {self.step}, epoch {self.epoch}")

    def generate_text(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate text from prompt.

        Args:
            prompt_ids: Initial token IDs (1, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering

        Returns:
            Generated token IDs
        """
        return generate(
            self.model,
            prompt_ids,
            max_new_tokens,
            temperature,
            top_k,
            top_p
        )
