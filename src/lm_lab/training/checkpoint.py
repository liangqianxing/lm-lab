"""
Checkpoint management for training.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional


class CheckpointManager:
    """
    Manage model checkpoints during training.

    Args:
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep (None = unlimited)
    """

    def __init__(self, checkpoint_dir: str, max_checkpoints: Optional[int] = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = []

    def save(
        self,
        model,
        optimizer,
        step: int,
        epoch: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a checkpoint.

        Args:
            model: Model instance
            optimizer: Optimizer instance
            step: Current training step
            epoch: Current epoch
            metadata: Additional metadata to save

        Returns:
            Checkpoint filename
        """
        filename = f"checkpoint_step_{step}.npz"
        checkpoint_path = self.checkpoint_dir / filename

        # Collect model parameters
        params = {f"param_{i}": p for i, p in enumerate(model.parameters())}

        # Optimizer state
        optimizer_state = optimizer.state_dict()

        # Metadata
        meta = {
            'step': step,
            'epoch': epoch,
            'vocab_size': model.vocab_size,
            'd_model': model.d_model,
            'num_layers': model.num_layers
        }
        if metadata:
            meta.update(metadata)

        # Save
        np.savez(
            checkpoint_path,
            metadata=json.dumps(meta),
            optimizer_state=json.dumps(optimizer_state),
            **params
        )

        self.checkpoints.append(str(checkpoint_path))

        # Remove old checkpoints if limit exceeded
        if self.max_checkpoints and len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            Path(old_checkpoint).unlink(missing_ok=True)

        return filename

    def load(self, model, optimizer, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            model: Model instance
            optimizer: Optimizer instance
            checkpoint_path: Path to checkpoint file

        Returns:
            Metadata dictionary
        """
        path = Path(checkpoint_path)
        if not path.exists():
            path = self.checkpoint_dir / checkpoint_path

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        data = np.load(path, allow_pickle=True)

        # Load metadata
        metadata = json.loads(str(data['metadata']))

        # Load model parameters
        params = model.parameters()
        for i in range(len(params)):
            key = f"param_{i}"
            if key in data:
                params[i][:] = data[key]

        # Load optimizer state
        optimizer_state = json.loads(str(data['optimizer_state']))
        optimizer.load_state_dict(optimizer_state)

        return metadata

    def list_checkpoints(self):
        """List all available checkpoints."""
        return sorted(self.checkpoint_dir.glob("checkpoint_*.npz"))

    def get_latest(self) -> Optional[Path]:
        """Get the most recent checkpoint."""
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None
