"""
PyTorch-based training utilities for language models.

Provides Dataset, DataLoader creation, and Trainer class.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, Any
import time


class LanguageModelingDataset(Dataset):
    """
    Dataset for language modeling with input-target pairs.

    Creates sequences where target is input shifted by 1.

    Args:
        token_ids: List or tensor of token IDs
        seq_len: Sequence length
    """

    def __init__(self, token_ids, seq_len: int):
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        elif not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)

        self.token_ids = token_ids
        self.seq_len = seq_len

        # Calculate number of sequences
        # Each sequence needs seq_len tokens for input and seq_len for target (shifted by 1)
        # With 100 tokens and seq_len=10: [0:10]->[1:11], [10:20]->[11:21], etc = 10 sequences
        self.num_sequences = len(token_ids) // seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len

        input_ids = self.token_ids[start:end]
        target_ids = self.token_ids[start + 1:end + 1]

        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }


def create_dataloader(
    token_ids,
    batch_size: int,
    seq_len: int,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader from token IDs.

    Args:
        token_ids: List or array of token IDs
        batch_size: Batch size
        seq_len: Sequence length
        shuffle: Whether to shuffle

    Returns:
        DataLoader instance
    """
    dataset = LanguageModelingDataset(token_ids, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class Trainer:
    """
    Trainer for language models.

    Handles training loop, loss computation, and checkpointing.

    Args:
        model: TransformerLM model
        optimizer: PyTorch optimizer
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (optional)
        checkpoint_dir: Directory for checkpoints
        log_interval: Steps between logging
        eval_interval: Steps between evaluation
        device: Device to train on
    """

    def __init__(
        self,
        model,
        optimizer,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
        eval_interval: int = 100,
        device: str = "cpu"
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.device = device

        self.model.to(device)

        self.step = 0
        self.epoch = 0

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'steps': [],
        }

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model logits of shape (batch, seq_len, vocab_size)
            targets: Target token IDs of shape (batch, seq_len)

        Returns:
            Scalar loss
        """
        # Reshape for cross_entropy: (batch*seq_len, vocab_size) and (batch*seq_len,)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)

        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
        return loss

    def train(self, num_steps: Optional[int] = None, num_epochs: Optional[int] = None):
        """
        Train the model.

        Args:
            num_steps: Number of training steps
            num_epochs: Number of epochs

        Returns:
            Training history
        """
        if num_steps is None and num_epochs is None:
            raise ValueError("Must specify either num_steps or num_epochs")

        if num_steps is not None and num_epochs is not None:
            raise ValueError("Cannot specify both num_steps and num_epochs")

        if num_steps is not None:
            return self._train_steps(num_steps)
        else:
            return self._train_epochs(num_epochs)

    def _train_steps(self, num_steps: int):
        """Train for a fixed number of steps."""
        self.model.train()
        start_time = time.time()

        while self.step < num_steps:
            if self.train_loader is None:
                raise ValueError("No training loader provided")

            for batch in self.train_loader:
                if self.step >= num_steps:
                    break

                loss = self._train_step(batch)

                # Logging
                if self.step % self.log_interval == 0:
                    print(f"Step {self.step:6d} | Loss: {loss:.4f}")

                # Evaluation
                if self.val_loader and self.step % self.eval_interval == 0:
                    val_loss = self._evaluate()
                    print(f"Step {self.step:6d} | Val Loss: {val_loss:.4f}")
                    self.model.train()

                self.step += 1

            self.epoch += 1

        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.2f}s")
        return self.history

    def _train_epochs(self, num_epochs: int):
        """Train for a fixed number of epochs."""
        self.model.train()

        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            epoch_loss = 0.0
            num_batches = 0

            for batch in self.train_loader:
                loss = self._train_step(batch)
                epoch_loss += loss
                num_batches += 1

                if self.step % self.log_interval == 0:
                    print(f"Step {self.step:6d} | Loss: {loss:.4f}")

                self.step += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        return self.history

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform single training step.

        Args:
            batch: Batch dictionary with input_ids and target_ids

        Returns:
            Loss value
        """
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)

        # Forward pass
        logits = self.model(input_ids)

        # Compute loss
        loss = self.compute_loss(logits, target_ids)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Record history
        loss_val = loss.item()
        self.history['train_loss'].append(loss_val)
        self.history['steps'].append(self.step)

        return loss_val

    def _evaluate(self) -> float:
        """
        Evaluate on validation set.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)

                logits = self.model(input_ids)
                loss = self.compute_loss(logits, target_ids)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.history['val_loss'].append(avg_loss)

        return avg_loss

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)

        print(f"Loaded checkpoint from step {self.step}, epoch {self.epoch}")
