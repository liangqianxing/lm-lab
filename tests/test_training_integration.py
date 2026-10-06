"""
Integration tests for the complete training pipeline.

Tests:
- Overfitting on tiny dataset (verify model can learn)
- Checkpoint save/resume
- Text generation
- Gradient accumulation
- Loss decreases over iterations
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import shutil
import tempfile

from lm_lab.model.transformer import TransformerLM
from lm_lab.optimizer.adam import Adam, AdamW
from lm_lab.training.trainer import Trainer, create_dataloader
from lm_lab.training.checkpoint import CheckpointManager


class TestTrainingIntegration:
    """Integration tests for training pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for checkpoints."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def tiny_dataset(self):
        """Create a tiny dataset for overfitting tests."""
        # Repeating sequence: [1, 2, 3, 4, 1, 2, 3, 4, ...]
        tokens = np.array([1, 2, 3, 4] * 50, dtype=np.int64)
        return tokens

    @pytest.fixture
    def small_model(self):
        """Create a small model for fast testing."""
        model = TransformerLM(
            vocab_size=10,
            d_model=64,
            n_heads=2,
            n_layers=2,
            max_len=128,
            dropout=0.0
        )
        return model

    def test_overfit_tiny_dataset(self, small_model, tiny_dataset):
        """Test that model can overfit on a tiny repeating dataset."""
        # Create dataloader
        train_loader = create_dataloader(
            tokens=tiny_dataset,
            batch_size=4,
            seq_len=16,
            shuffle=False,
            seed=42
        )

        # Create optimizer
        optimizer = Adam(learning_rate=0.001)

        # Create trainer
        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            log_interval=10,
            gradient_clip_norm=1.0
        )

        # Train for a few steps
        history = trainer.train(num_steps=50)

        # Check that loss decreased (use first and last available)
        train_losses = history['train_loss']
        assert len(train_losses) > 0, "Should have some training losses"

        initial_loss = train_losses[0]
        final_loss = train_losses[-1]

        assert final_loss < initial_loss, (
            f"Loss should decrease: initial={initial_loss:.4f}, "
            f"final={final_loss:.4f}"
        )

        # Loss should be reasonable (not NaN, not too high)
        assert np.isfinite(final_loss), "Final loss should be finite"
        assert final_loss < 10.0, f"Final loss too high: {final_loss:.4f}"

    def test_checkpoint_save_and_resume(self, small_model, tiny_dataset, temp_dir):
        """Test checkpoint saving and resuming training."""
        # Create dataloader
        train_loader = create_dataloader(
            tokens=tiny_dataset,
            batch_size=4,
            seq_len=16,
            shuffle=False,
            seed=42
        )

        # Create optimizer
        optimizer = Adam(learning_rate=0.001)

        # Create trainer with checkpoint directory
        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            checkpoint_dir=temp_dir,
            log_interval=5,
            save_interval=20
        )

        # Train for 25 steps (should save at step 20)
        trainer.train(num_steps=25)

        # Check checkpoint was saved
        checkpoint_files = list(Path(temp_dir).glob("checkpoint_step20*.pt"))
        assert len(checkpoint_files) > 0, "Checkpoint at step 20 should be saved"

        # Get the checkpoint file
        checkpoint_path = str(checkpoint_files[0])

        # Create new model and optimizer
        new_model = TransformerLM(
            vocab_size=10,
            d_model=64,
            n_heads=2,
            n_layers=2,
            max_len=128,
            dropout=0.0
        )
        new_optimizer = Adam(learning_rate=0.001)

        # Create new trainer
        new_trainer = Trainer(
            model=new_model,
            optimizer=new_optimizer,
            train_dataloader=train_loader,
            checkpoint_dir=temp_dir
        )

        # Load checkpoint
        new_trainer.load_checkpoint(checkpoint_path)

        # Check step was restored
        assert new_trainer.step == 20, f"Step should be 20, got {new_trainer.step}"

        # Continue training
        new_trainer.train(num_steps=30)
        assert new_trainer.step == 30, "Training should continue from checkpoint"

    def test_generation_produces_tokens(self, small_model, tiny_dataset):
        """Test that text generation produces valid tokens."""
        # Train model briefly
        train_loader = create_dataloader(
            tokens=tiny_dataset,
            batch_size=4,
            seq_len=16,
            shuffle=False,
            seed=42
        )

        optimizer = Adam(learning_rate=0.001)
        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            train_dataloader=train_loader
        )

        trainer.train(num_steps=20)

        # Generate text
        prompt_ids = np.array([1, 2], dtype=np.int64)
        generated = trainer.generate_text(
            prompt_ids=prompt_ids,
            max_length=20,
            temperature=1.0
        )

        # Check output shape and validity
        assert len(generated) == 20, f"Should generate 20 tokens, got {len(generated)}"
        assert np.all(generated >= 0), "All token IDs should be non-negative"
        assert np.all(generated < small_model.vocab_size), (
            f"All token IDs should be < vocab_size={small_model.vocab_size}"
        )

        # Check prompt is preserved
        assert np.array_equal(generated[:2], prompt_ids), "Prompt should be preserved"

    def test_generation_with_temperature(self, small_model, tiny_dataset):
        """Test generation with different temperatures."""
        # Train model briefly
        train_loader = create_dataloader(
            tokens=tiny_dataset,
            batch_size=4,
            seq_len=16,
            shuffle=False,
            seed=42
        )

        optimizer = Adam(learning_rate=0.001)
        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            train_dataloader=train_loader
        )

        trainer.train(num_steps=20)

        prompt_ids = np.array([1, 2], dtype=np.int64)

        # Generate with low temperature (should be more deterministic)
        gen_low = trainer.generate_text(
            prompt_ids=prompt_ids,
            max_length=15,
            temperature=0.1
        )

        # Generate with high temperature (should be more random)
        gen_high = trainer.generate_text(
            prompt_ids=prompt_ids,
            max_length=15,
            temperature=2.0
        )

        # Both should be valid
        assert len(gen_low) == 15
        assert len(gen_high) == 15
        assert np.all(gen_low >= 0) and np.all(gen_low < 10)
        assert np.all(gen_high >= 0) and np.all(gen_high < 10)

    def test_gradient_clipping(self, small_model, tiny_dataset):
        """Test gradient clipping prevents exploding gradients."""
        train_loader = create_dataloader(
            tokens=tiny_dataset,
            batch_size=4,
            seq_len=16,
            shuffle=False,
            seed=42
        )

        # Train with gradient clipping
        optimizer_clipped = Adam(learning_rate=0.01)  # Higher LR
        trainer_clipped = Trainer(
            model=small_model,
            optimizer=optimizer_clipped,
            train_dataloader=train_loader,
            gradient_clip_norm=1.0
        )

        history = trainer_clipped.train(num_steps=30)

        # All losses should be finite
        assert all(np.isfinite(loss) for loss in history['train_loss']), (
            "All losses should be finite with gradient clipping"
        )

    def test_loss_decreases_monotonically(self, small_model, tiny_dataset):
        """Test that loss generally decreases over training."""
        train_loader = create_dataloader(
            tokens=tiny_dataset,
            batch_size=4,
            seq_len=16,
            shuffle=False,
            seed=42
        )

        optimizer = Adam(learning_rate=0.001)
        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            log_interval=1  # Log every step
        )

        history = trainer.train(num_steps=50)

        # Compute moving average of loss
        losses = history['train_loss']
        window = 10

        if len(losses) >= 2 * window:
            early_avg = np.mean(losses[:window])
            late_avg = np.mean(losses[-window:])

            assert late_avg < early_avg, (
                f"Loss should decrease: early={early_avg:.4f}, late={late_avg:.4f}"
            )

    def test_evaluation_metrics(self, small_model, tiny_dataset):
        """Test evaluation computes correct metrics."""
        # Split into train and val
        split_idx = len(tiny_dataset) * 3 // 4
        train_tokens = tiny_dataset[:split_idx]
        val_tokens = tiny_dataset[split_idx:]

        train_loader = create_dataloader(
            tokens=train_tokens,
            batch_size=4,
            seq_len=16,
            shuffle=False,
            seed=42
        )

        # Use smaller batch/seq for validation to fit the smaller dataset
        val_loader = create_dataloader(
            tokens=val_tokens,
            batch_size=2,
            seq_len=8,
            shuffle=False,
            seed=42
        )

        optimizer = Adam(learning_rate=0.001)
        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            eval_interval=10
        )

        trainer.train(num_steps=20)

        # Evaluate
        metrics = trainer.evaluate()

        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert 'num_tokens' in metrics

        assert np.isfinite(metrics['loss'])
        assert np.isfinite(metrics['perplexity'])
        assert metrics['perplexity'] > 0
        assert metrics['num_tokens'] > 0

    def test_learning_rate_getter_setter(self, small_model, tiny_dataset):
        """Test learning rate can be adjusted during training."""
        train_loader = create_dataloader(
            tokens=tiny_dataset,
            batch_size=4,
            seq_len=16,
            shuffle=False,
            seed=42
        )

        optimizer = Adam(learning_rate=0.001)
        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            train_dataloader=train_loader
        )

        # Check initial LR
        assert trainer.get_learning_rate() == 0.001

        # Change LR
        trainer.set_learning_rate(0.0001)
        assert trainer.get_learning_rate() == 0.0001

        # Train with new LR
        trainer.train(num_steps=10)

        # LR should be recorded in history
        assert all(lr == 0.0001 for lr in trainer.history['learning_rates'])

    def test_adamw_optimizer(self, small_model, tiny_dataset):
        """Test training with AdamW optimizer."""
        train_loader = create_dataloader(
            tokens=tiny_dataset,
            batch_size=4,
            seq_len=16,
            shuffle=False,
            seed=42
        )

        # Use AdamW with weight decay
        optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
        trainer = Trainer(
            model=small_model,
            optimizer=optimizer,
            train_dataloader=train_loader
        )

        history = trainer.train(num_steps=30)

        # Should still converge
        train_losses = history['train_loss']
        assert len(train_losses) > 0, "Should have some training losses"

        initial_loss = train_losses[0]
        final_loss = train_losses[-1]

        assert final_loss < initial_loss, "Loss should decrease with AdamW"
        assert np.isfinite(final_loss), "Loss should be finite"
