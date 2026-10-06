"""
Checkpoint management for model training.

Provides CheckpointManager for saving/loading model parameters, optimizer state,
and training metadata with automatic rotation policy.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


class CheckpointManager:
    """
    Manages model checkpoints with automatic rotation.

    Saves model parameters, optimizer state, and training metadata.
    Implements a rotation policy to limit disk usage while preserving
    important checkpoints (best metric, latest, periodic milestones).

    Args:
        checkpoint_dir: Directory for saving checkpoints
        keep_last_n: Number of most recent checkpoints to keep (default: 3)
        keep_best_n: Number of best checkpoints to keep (default: 1)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
        keep_best_n: int = 1
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n

        self.checkpoints: List[Dict[str, Any]] = []

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Any,
        step: int,
        epoch: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            step: Current training step
            epoch: Current epoch
            metadata: Additional metadata to save

        Returns:
            Checkpoint filename
        """
        filename = f"checkpoint_step{step}_epoch{epoch}.pt"
        filepath = self.checkpoint_dir / filename

        # Build checkpoint dictionary
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self._get_optimizer_state(optimizer),
            'metadata': metadata or {}
        }

        # Save checkpoint
        torch.save(checkpoint, filepath)

        # Update checkpoint list
        self.checkpoints.append({
            'filepath': str(filepath),
            'step': step,
            'epoch': epoch,
            'loss': metadata.get('loss', float('inf')) if metadata else float('inf')
        })

        # Apply rotation policy
        self._rotate_checkpoints()

        return filename

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Any,
        checkpoint_path: str
    ) -> Dict[str, Any]:
        """
        Load model from checkpoint.

        Args:
            model: Model to load into
            optimizer: Optimizer to load into
            checkpoint_path: Path to checkpoint file

        Returns:
            Metadata dictionary
        """
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self._set_optimizer_state(optimizer, checkpoint['optimizer_state_dict'])

        # Return metadata
        metadata = {
            'step': checkpoint.get('step', 0),
            'epoch': checkpoint.get('epoch', 0),
        }
        metadata.update(checkpoint.get('metadata', {}))

        return metadata

    def _get_optimizer_state(self, optimizer: Any) -> Dict[str, Any]:
        """Extract optimizer state for saving."""
        if hasattr(optimizer, 'state_dict'):
            return optimizer.state_dict()

        # For custom optimizers without state_dict
        state = {
            'learning_rate': optimizer.learning_rate,
        }

        # Save Adam/AdamW state
        if hasattr(optimizer, 'm') and hasattr(optimizer, 'v'):
            state['m'] = optimizer.m
            state['v'] = optimizer.v
            state['t'] = optimizer.t

        return state

    def _set_optimizer_state(self, optimizer: Any, state: Dict[str, Any]) -> None:
        """Restore optimizer state from checkpoint."""
        if hasattr(optimizer, 'load_state_dict'):
            optimizer.load_state_dict(state)
            return

        # For custom optimizers
        if 'learning_rate' in state:
            optimizer.learning_rate = state['learning_rate']

        if 'm' in state and hasattr(optimizer, 'm'):
            optimizer.m = state['m']
        if 'v' in state and hasattr(optimizer, 'v'):
            optimizer.v = state['v']
        if 't' in state and hasattr(optimizer, 't'):
            optimizer.t = state['t']

    def _rotate_checkpoints(self) -> None:
        """Apply rotation policy to limit number of checkpoints."""
        if len(self.checkpoints) <= self.keep_last_n:
            return

        # Sort by step
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x['step'])

        # Keep last N
        keep_last = set(c['filepath'] for c in sorted_checkpoints[-self.keep_last_n:])

        # Keep best N by loss
        sorted_by_loss = sorted(self.checkpoints, key=lambda x: x['loss'])
        keep_best = set(c['filepath'] for c in sorted_by_loss[:self.keep_best_n])

        # Combine keep sets
        keep_set = keep_last | keep_best

        # Remove old checkpoints
        new_checkpoints = []
        for checkpoint in self.checkpoints:
            if checkpoint['filepath'] in keep_set:
                new_checkpoints.append(checkpoint)
            else:
                # Delete file
                filepath = Path(checkpoint['filepath'])
                if filepath.exists():
                    filepath.unlink()

        self.checkpoints = new_checkpoints

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return sorted(self.checkpoints, key=lambda x: x['step'])

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to most recent checkpoint."""
        if not self.checkpoints:
            return None
        latest = max(self.checkpoints, key=lambda x: x['step'])
        return latest['filepath']

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to checkpoint with best (lowest) loss."""
        if not self.checkpoints:
            return None
        best = min(self.checkpoints, key=lambda x: x['loss'])
        return best['filepath']
