"""
Simple training script demonstrating the complete pipeline.

This script:
1. Creates a tiny synthetic dataset
2. Builds a small Transformer model
3. Trains with Adam optimizer
4. Saves checkpoints
5. Generates sample text
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lm_lab.model.transformer import TransformerLM
from lm_lab.optimizer.adam import Adam
from lm_lab.training.trainer import Trainer, create_dataloader


def create_synthetic_dataset(vocab_size=100, num_tokens=10000, seed=42):
    """
    Create a synthetic dataset with patterns.

    Args:
        vocab_size: Size of vocabulary
        num_tokens: Total number of tokens
        seed: Random seed

    Returns:
        Token array
    """
    np.random.seed(seed)

    # Create repeating patterns to make learning easier
    # Pattern 1: Sequential [1, 2, 3, ..., 10]
    pattern1 = np.arange(1, 11, dtype=np.int64)

    # Pattern 2: Reversed [10, 9, 8, ..., 1]
    pattern2 = np.arange(10, 0, -1, dtype=np.int64)

    # Pattern 3: Random but with structure
    pattern3 = np.random.randint(11, vocab_size, size=20, dtype=np.int64)

    # Combine patterns
    patterns = [pattern1, pattern2, pattern3]

    # Build dataset by repeating patterns
    tokens = []
    while len(tokens) < num_tokens:
        pattern = patterns[np.random.randint(len(patterns))]
        tokens.extend(pattern)

    tokens = np.array(tokens[:num_tokens], dtype=np.int64)

    return tokens


def main():
    """Main training script."""
    print("=" * 60)
    print("Simple Transformer Language Model Training")
    print("=" * 60)

    # Hyperparameters
    vocab_size = 100
    d_model = 128
    n_heads = 4
    n_layers = 3
    max_len = 256
    dropout = 0.1

    batch_size = 8
    seq_len = 32
    num_steps = 500
    learning_rate = 0.001

    checkpoint_dir = "runs/simple_train"

    print("\nConfiguration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Attention heads: {n_heads}")
    print(f"  Layers: {n_layers}")
    print(f"  Max sequence length: {max_len}")
    print(f"  Dropout: {dropout}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Training steps: {num_steps}")
    print(f"  Learning rate: {learning_rate}")

    # Create synthetic dataset
    print("\nCreating synthetic dataset...")
    tokens = create_synthetic_dataset(vocab_size=vocab_size, num_tokens=20000, seed=42)
    print(f"  Total tokens: {len(tokens)}")

    # Split into train/val
    split_idx = int(len(tokens) * 0.9)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    print(f"  Train tokens: {len(train_tokens)}")
    print(f"  Val tokens: {len(val_tokens)}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(
        tokens=train_tokens,
        batch_size=batch_size,
        seq_len=seq_len,
        shuffle=True,
        seed=42
    )

    val_loader = create_dataloader(
        tokens=val_tokens,
        batch_size=batch_size,
        seq_len=seq_len,
        shuffle=False,
        seed=42
    )

    # Create model
    print("\nBuilding Transformer model...")
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=max_len,
        dropout=dropout
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Create optimizer
    print("\nCreating Adam optimizer...")
    optimizer = Adam(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        checkpoint_dir=checkpoint_dir,
        log_interval=50,
        eval_interval=100,
        save_interval=200,
        gradient_clip_norm=1.0
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    history = trainer.train(num_steps=num_steps)

    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    train_losses = history['train_loss']
    if len(train_losses) > 0:
        print(f"Initial loss: {train_losses[0]:.4f}")
        print(f"Final loss: {train_losses[-1]:.4f}")
        print(f"Loss reduction: {train_losses[0] - train_losses[-1]:.4f}")

    val_losses = history['val_loss']
    if len(val_losses) > 0:
        print(f"Final validation loss: {val_losses[-1]:.4f}")

    # Generate sample text
    print("\n" + "=" * 60)
    print("Sample Text Generation")
    print("=" * 60)

    # Generate from different prompts
    prompts = [
        np.array([1, 2], dtype=np.int64),
        np.array([10, 9], dtype=np.int64),
        np.array([5], dtype=np.int64),
    ]

    for i, prompt_ids in enumerate(prompts, 1):
        print(f"\nSample {i}:")
        print(f"  Prompt: {prompt_ids.tolist()}")

        # Generate with different temperatures
        for temp in [0.5, 1.0, 1.5]:
            generated = trainer.generate_text(
                prompt_ids=prompt_ids,
                max_length=20,
                temperature=temp
            )
            print(f"  Temperature {temp}: {generated.tolist()}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    metrics = trainer.evaluate()
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Tokens evaluated: {metrics['num_tokens']}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
