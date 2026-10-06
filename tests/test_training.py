"""
Tests for B4: Training pipeline.

Verifies training loop, loss computation, checkpointing, and overfitting capability.
"""

import torch
import pytest
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lm_lab.model import TransformerLM
from lm_lab.optimizer import AdamW
from lm_lab.training import Trainer, create_dataloader, LanguageModelingDataset


def test_dataset_creation():
    """Test LanguageModelingDataset creates correct input-target pairs."""
    token_ids = list(range(100))
    dataset = LanguageModelingDataset(token_ids, seq_len=10)

    assert len(dataset) == 10

    sample = dataset[0]
    assert 'input_ids' in sample
    assert 'target_ids' in sample
    assert sample['input_ids'].shape == (10,)
    assert sample['target_ids'].shape == (10,)

    # Target should be input shifted by 1
    assert (sample['target_ids'] == sample['input_ids'] + 1).all()


def test_dataloader_creation():
    """Test create_dataloader works correctly."""
    token_ids = list(range(1000))
    loader = create_dataloader(token_ids, batch_size=8, seq_len=32, shuffle=False)

    batch = next(iter(loader))
    assert batch['input_ids'].shape == (8, 32)
    assert batch['target_ids'].shape == (8, 32)


def test_loss_computation():
    """Test loss computation is correct."""
    model = TransformerLM(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_len=32
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            optimizer=AdamW(model.parameters()),
            train_loader=None,
            checkpoint_dir=tmpdir
        )

        logits = torch.randn(4, 16, 100)
        targets = torch.randint(0, 100, (4, 16))

        loss = trainer.compute_loss(logits, targets)

        assert loss.ndim == 0  # Scalar
        assert loss > 0
        assert not torch.isnan(loss)


def test_overfitting_small_batch():
    """Test model can overfit a small batch (verification of learning)."""
    torch.manual_seed(42)

    # Create tiny dataset
    token_ids = [i % 20 for i in range(100)]

    model = TransformerLM(
        vocab_size=20,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_len=32
    )

    optimizer = AdamW(model.parameters(), lr=1e-3)

    train_loader = create_dataloader(
        token_ids,
        batch_size=2,
        seq_len=8,
        shuffle=False
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            checkpoint_dir=tmpdir,
            eval_interval=1000  # No eval
        )

        # Get initial loss
        batch = next(iter(train_loader))
        logits = model(batch['input_ids'])
        initial_loss = trainer.compute_loss(logits, batch['target_ids']).item()

        # Train for a few steps
        trainer.train(num_steps=50)

        # Get final loss
        model.eval()
        with torch.no_grad():
            logits = model(batch['input_ids'])
            final_loss = trainer.compute_loss(logits, batch['target_ids']).item()

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    model = TransformerLM(
        vocab_size=50,
        d_model=64,
        n_heads=4,
        n_layers=2
    )

    optimizer = AdamW(model.parameters())

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=None,
            checkpoint_dir=tmpdir
        )

        # Save checkpoint
        trainer.step = 100
        trainer.epoch = 5
        trainer.save_checkpoint('test_checkpoint.pt')

        # Verify file exists
        checkpoint_path = Path(tmpdir) / 'test_checkpoint.pt'
        assert checkpoint_path.exists()

        # Create new trainer and load
        model2 = TransformerLM(
            vocab_size=50,
            d_model=64,
            n_heads=4,
            n_layers=2
        )
        optimizer2 = AdamW(model2.parameters())

        trainer2 = Trainer(
            model=model2,
            optimizer=optimizer2,
            train_loader=None,
            checkpoint_dir=tmpdir
        )

        trainer2.load_checkpoint('test_checkpoint.pt')

        # Verify state restored
        assert trainer2.step == 100
        assert trainer2.epoch == 5

        # Verify model weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


def test_generation_after_training():
    """Test model can generate text after training."""
    torch.manual_seed(42)

    # Create simple repeating pattern
    token_ids = [0, 1, 2, 0, 1, 2, 0, 1, 2] * 20

    model = TransformerLM(
        vocab_size=10,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_len=32
    )

    optimizer = AdamW(model.parameters(), lr=1e-3)

    train_loader = create_dataloader(
        token_ids,
        batch_size=4,
        seq_len=8,
        shuffle=True
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            checkpoint_dir=tmpdir,
            eval_interval=1000
        )

        # Train briefly
        trainer.train(num_steps=100)

    # Test generation
    model.eval()
    start_tokens = torch.tensor([[0, 1]])
    generated = model.generate(start_tokens, max_new_tokens=5, temperature=0.1)

    assert generated.shape == (1, 7)
    assert not torch.isnan(generated).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
