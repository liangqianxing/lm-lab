"""
Main training loop for language models.

Provides the Trainer class that orchestrates model training, evaluation,
checkpointing, and generation.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

from ..model.transformer import TransformerLM
from ..optimizer.adam import Adam, AdamW
from .dataloader import TextDataLoader, Batch
from .checkpoint import CheckpointManager
from .evaluator import cross_entropy_loss, Evaluator
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
    - Training loop with forward/backward passes
    - Loss computation with numerical stability
    - Optimizer updates
    - Evaluation on validation set
    - Checkpointing
    - Text generation

    Args:
        model: TransformerLM model
        optimizer: Optimizer (Adam or AdamW)
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
        checkpoint_dir: Directory for checkpoints
        log_interval: Steps between logging
        eval_interval: Steps between evaluation
        save_interval: Steps between checkpoints
        gradient_clip_norm: Maximum gradient norm (None = no clipping)
    """

    def __init__(
        self,
        model: TransformerLM,
        optimizer,
        train_dataloader: TextDataLoader,
        val_dataloader: Optional[TextDataLoader] = None,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
        eval_interval: int = 100,
        save_interval: int = 1000,
        gradient_clip_norm: Optional[float] = 1.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.gradient_clip_norm = gradient_clip_norm

        self.step = 0
        self.epoch = 0

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'steps': [],
            'epochs': [],
            'learning_rates': []
        }

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

        print(f"Starting training: {num_steps or 'N/A'} steps, {num_epochs or 'N/A'} epochs")
        start_time = time.time()

        if num_epochs is not None:
            self._train_epochs(num_epochs)
        else:
            self._train_steps(num_steps)

        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f}s")

        return self.history

    def _train_epochs(self, num_epochs: int) -> None:
        """Train for a fixed number of epochs."""
        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            epoch_loss = 0.0
            num_batches = 0

            for batch in self.train_dataloader:
                loss = self._train_step(batch)
                epoch_loss += loss
                num_batches += 1

                # Logging
                if self.step % self.log_interval == 0:
                    self._log_training(loss)

                # Evaluation
                if self.val_dataloader and self.step % self.eval_interval == 0:
                    self._run_evaluation()

                # Checkpointing
                if self.step % self.save_interval == 0:
                    self._save_checkpoint(loss)

                self.step += 1

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

    def _train_steps(self, num_steps: int) -> None:
        """Train for a fixed number of steps."""
        while self.step < num_steps:
            for batch in self.train_dataloader:
                if self.step >= num_steps:
                    break

                loss = self._train_step(batch)

                # Logging
                if self.step % self.log_interval == 0:
                    self._log_training(loss)

                # Evaluation
                if self.val_dataloader and self.step % self.eval_interval == 0:
                    self._run_evaluation()

                # Checkpointing
                if self.step % self.save_interval == 0:
                    self._save_checkpoint(loss)

                self.step += 1

            self.epoch += 1

    def _train_step(self, batch: Batch) -> float:
        """
        Perform single training step.

        1. Forward pass: compute logits
        2. Compute loss and gradients
        3. Backward pass: compute parameter gradients
        4. Gradient clipping (if enabled)
        5. Optimizer step: update parameters

        Args:
            batch: Training batch with input_ids and target_ids

        Returns:
            Loss value (scalar)
        """
        # Convert numpy arrays to torch tensors
        input_ids = torch.from_numpy(batch.input_ids).long()
        target_ids = torch.from_numpy(batch.target_ids).long()

        # Zero gradients
        self.model.zero_grad()

        # Forward pass: (batch_size, seq_len) -> (batch_size, seq_len, vocab_size)
        logits = self.model.forward(input_ids)

        # Compute loss using PyTorch's cross entropy
        # Reshape: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_ids.view(-1)

        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)

        # Check for NaN/Inf in loss
        loss_value = loss.item()
        if not np.isfinite(loss_value):
            print(f"Warning: Non-finite loss at step {self.step}: {loss_value}")
            return loss_value

        # Backward pass: compute gradients using autograd
        loss.backward()

        # Get parameters and gradients as numpy arrays for optimizer
        params_list = []
        grads_list = []

        for param in self.model.parameters():
            if param.grad is not None:
                params_list.append(param.data.cpu().numpy())
                grads_list.append(param.grad.cpu().numpy())
            else:
                params_list.append(param.data.cpu().numpy())
                grads_list.append(None)

        # Check for non-finite gradients
        grad_is_finite = all(
            grad is None or np.isfinite(grad).all()
            for grad in grads_list
        )

        if not grad_is_finite:
            print(f"Warning: Non-finite gradients at step {self.step}")
            return loss_value

        # Gradient clipping (optional)
        if self.gradient_clip_norm is not None:
            grads_list = self._clip_gradients(grads_list, self.gradient_clip_norm)

        # Create parameter and gradient dictionaries for optimizer
        param_dict = {f"param_{i}": p for i, p in enumerate(params_list)}
        grad_dict = {f"param_{i}": g for i, g in enumerate(grads_list) if g is not None}

        # Optimizer step: update parameters in-place
        self.optimizer.step(param_dict, grad_dict)

        # Copy updated parameters back to model
        param_idx = 0
        for param in self.model.parameters():
            updated_values = param_dict[f"param_{param_idx}"]
            param.data.copy_(torch.from_numpy(updated_values))
            param_idx += 1

        return loss_value

    def _clip_gradients(self, grads: List[np.ndarray], max_norm: float) -> List[np.ndarray]:
        """
        Clip gradients by global norm.

        Args:
            grads: List of gradient arrays
            max_norm: Maximum gradient norm

        Returns:
            Clipped gradients
        """
        # Compute total norm
        total_norm = 0.0
        for grad in grads:
            if grad is not None:
                total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)

        # Clip if necessary
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            clipped_grads = []
            for grad in grads:
                if grad is not None:
                    clipped_grads.append(grad * clip_coef)
                else:
                    clipped_grads.append(None)
            return clipped_grads

        return grads

    def _log_training(self, loss: float) -> None:
        """Log training progress."""
        self.history['train_loss'].append(loss)
        self.history['steps'].append(self.step)
        self.history['epochs'].append(self.epoch)
        self.history['learning_rates'].append(self.optimizer.learning_rate)

        print(f"Step {self.step:6d} | Epoch {self.epoch:3d} | Loss: {loss:.4f}")

    def _run_evaluation(self) -> None:
        """Run evaluation on validation set."""
        evaluator = Evaluator(self.model, self.val_dataloader)
        metrics = evaluator.evaluate()

        val_loss = metrics['loss']
        val_ppl = metrics['perplexity']

        self.history['val_loss'].append(val_loss)

        print(f"Step {self.step:6d} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

    def _save_checkpoint(self, loss: float) -> None:
        """Save model checkpoint."""
        filename = self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
            epoch=self.epoch,
            metadata={
                'loss': float(loss),
                'learning_rate': self.optimizer.learning_rate
            }
        )
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        metadata = self.checkpoint_manager.load(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_path=checkpoint_path
        )

        self.step = metadata.get('step', 0)
        self.epoch = metadata.get('epoch', 0)

        print(f"Loaded checkpoint: step {self.step}, epoch {self.epoch}")

    def generate_text(
        self,
        prompt_ids: np.ndarray,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_token_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate text autoregressively from a prompt.

        Args:
            prompt_ids: Initial token IDs, shape (seq_len,)
            max_length: Maximum total length (prompt + generated)
            temperature: Sampling temperature (higher = more random)
            top_k: If set, use top-k sampling
            top_p: If set, use nucleus (top-p) sampling
            stop_token_id: Token ID to stop generation

        Returns:
            Generated token IDs, shape (total_length,)
        """
        return generate(
            model=self.model,
            prompt_ids=prompt_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token_id=stop_token_id
        )

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Returns:
            Dictionary with metrics (loss, perplexity, num_tokens)
        """
        if self.val_dataloader is None:
            raise ValueError("No validation dataloader provided")

        evaluator = Evaluator(self.model, self.val_dataloader)
        return evaluator.evaluate()

    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.learning_rate

    def set_learning_rate(self, lr: float) -> None:
        """Set learning rate."""
        self.optimizer.learning_rate = lr
