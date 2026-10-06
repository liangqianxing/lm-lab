"""
Data loading and batching for language model training.

Provides efficient batch creation from tokenized text data.
"""

import numpy as np
from typing import List, Iterator, Optional, NamedTuple


class Batch(NamedTuple):
    """A training batch with input and target sequences."""
    input_ids: np.ndarray  # (batch_size, seq_len)
    target_ids: np.ndarray  # (batch_size, seq_len)


class TextDataLoader:
    """
    Data loader for language modeling.

    Creates batches from tokenized text by sliding a window
    over the token sequence.

    Args:
        tokens: Array of token IDs
        batch_size: Number of sequences per batch
        seq_len: Length of each sequence
        shuffle: Whether to shuffle data (default: True)
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        tokens: np.ndarray,
        batch_size: int,
        seq_len: int,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        self.tokens = tokens
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle

        if seed is not None:
            np.random.seed(seed)

        # Calculate number of complete sequences
        # Each sequence needs seq_len + 1 tokens (input + target)
        self.num_sequences = (len(tokens) - 1) // seq_len
        self.num_batches = self.num_sequences // batch_size

        # Truncate to fit exact number of batches
        self.total_tokens = self.num_batches * batch_size * seq_len

    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_batches

    def __iter__(self) -> Iterator[Batch]:
        """Iterate over batches."""
        # Create sequence indices
        indices = np.arange(self.num_sequences)

        if self.shuffle:
            np.random.shuffle(indices)

        # Generate batches
        for i in range(self.num_batches):
            batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]

            input_ids = np.zeros((self.batch_size, self.seq_len), dtype=np.int64)
            target_ids = np.zeros((self.batch_size, self.seq_len), dtype=np.int64)

            for j, idx in enumerate(batch_indices):
                start = idx * self.seq_len
                end = start + self.seq_len

                input_ids[j] = self.tokens[start:end]
                target_ids[j] = self.tokens[start + 1:end + 1]

            yield Batch(input_ids=input_ids, target_ids=target_ids)

    def reset(self) -> None:
        """Reset random state (useful for deterministic iteration)."""
        pass


def create_dataloader(
    tokens: np.ndarray,
    batch_size: int,
    seq_len: int,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> TextDataLoader:
    """
    Create a data loader.

    Args:
        tokens: Array of token IDs
        batch_size: Number of sequences per batch
        seq_len: Length of each sequence
        shuffle: Whether to shuffle data
        seed: Random seed for shuffling

    Returns:
        TextDataLoader instance
    """
    return TextDataLoader(tokens, batch_size, seq_len, shuffle, seed)
