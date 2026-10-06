"""
Example training script for a small Transformer language model.

This script demonstrates how to train a small model on a simple dataset
to verify the B2-B4 milestone implementations.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lm_lab.model import TransformerLM
from lm_lab.optimizer import AdamW
from lm_lab.training import Trainer, create_dataloader


def create_toy_dataset(vocab_size: int = 100, num_tokens: int = 10000):
    """Create a simple toy dataset for testing."""
    # Create synthetic data with some pattern
    # Sequence: 0, 1, 2, ..., 9, 0, 1, 2, ..., 9, ...
    token_ids = [i % vocab_size for i in range(num_tokens)]
    return token_ids


def main():
    """Main training function."""

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Configuration
    config = {
        'vocab_size': 100,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 512,
        'max_len': 64,
        'dropout': 0.1,
        'batch_size': 8,
        'seq_len': 32,
        'num_steps': 1000,
        'lr': 3e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("=" * 60)

    # Create dataset
    print("\nCreating dataset...")
    token_ids = create_toy_dataset(config['vocab_size'], num_tokens=10000)

    # Split into train/val
    split_idx = int(0.9 * len(token_ids))
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    print(f"Train tokens: {len(train_ids)}")
    print(f"Val tokens: {len(val_ids)}")

    # Create dataloaders
    train_loader = create_dataloader(
        train_ids,
        batch_size=config['batch_size'],
        seq_len=config['seq_len'],
        shuffle=True
    )

    val_loader = create_dataloader(
        val_ids,
        batch_size=config['batch_size'],
        seq_len=config['seq_len'],
        shuffle=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout']
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=0.01
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        checkpoint_dir='runs/toy_model',
        eval_interval=50,
        save_interval=500
    )

    print("\nStarting training...")
    print("=" * 60)

    # Train
    trainer.train(num_steps=config['num_steps'])

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    # Test generation
    print("\nTesting generation...")
    model.eval()
    start_tokens = torch.tensor([[0, 1, 2]], device=config['device'])
    generated = model.generate(
        start_tokens,
        max_new_tokens=20,
        temperature=0.8
    )
    print(f"Input: {start_tokens[0].tolist()}")
    print(f"Generated: {generated[0].tolist()}")


if __name__ == '__main__':
    main()
