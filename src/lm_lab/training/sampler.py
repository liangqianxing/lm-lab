"""
Text generation and sampling utilities.
"""

import numpy as np
from typing import Optional, Callable


def sample(logits: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample from logits with temperature.

    Args:
        logits: Logits array of shape (vocab_size,)
        temperature: Sampling temperature (higher = more random)

    Returns:
        Sampled token ID
    """
    if temperature == 0.0:
        return int(np.argmax(logits))

    # Apply temperature
    logits = logits / temperature

    # Numerical stability
    logits = logits - np.max(logits)

    # Softmax
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)

    # Sample
    return int(np.random.choice(len(logits), p=probs))


def sample_top_k(logits: np.ndarray, k: int, temperature: float = 1.0) -> int:
    """
    Sample from top-k most likely tokens.

    Args:
        logits: Logits array of shape (vocab_size,)
        k: Number of top tokens to consider
        temperature: Sampling temperature

    Returns:
        Sampled token ID
    """
    # Get top k indices
    top_k_indices = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_indices]

    # Apply temperature
    if temperature != 1.0:
        top_k_logits = top_k_logits / temperature

    # Numerical stability
    top_k_logits = top_k_logits - np.max(top_k_logits)

    # Softmax
    exp_logits = np.exp(top_k_logits)
    probs = exp_logits / np.sum(exp_logits)

    # Sample
    sampled_idx = np.random.choice(len(top_k_indices), p=probs)
    return int(top_k_indices[sampled_idx])


def sample_top_p(logits: np.ndarray, p: float, temperature: float = 1.0) -> int:
    """
    Nucleus sampling (top-p).

    Sample from smallest set of tokens whose cumulative probability exceeds p.

    Args:
        logits: Logits array of shape (vocab_size,)
        p: Cumulative probability threshold
        temperature: Sampling temperature

    Returns:
        Sampled token ID
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Numerical stability
    logits = logits - np.max(logits)

    # Softmax
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)

    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # Compute cumulative probabilities
    cumsum_probs = np.cumsum(sorted_probs)

    # Find cutoff index
    cutoff_idx = np.searchsorted(cumsum_probs, p) + 1
    cutoff_idx = min(cutoff_idx, len(sorted_indices))

    # Select top-p tokens
    top_p_indices = sorted_indices[:cutoff_idx]
    top_p_probs = sorted_probs[:cutoff_idx]

    # Renormalize
    top_p_probs = top_p_probs / np.sum(top_p_probs)

    # Sample
    sampled_idx = np.random.choice(len(top_p_indices), p=top_p_probs)
    return int(top_p_indices[sampled_idx])


def sample_temperature(logits: np.ndarray, temperature: float) -> int:
    """
    Sample with temperature (alias for sample).

    Args:
        logits: Logits array of shape (vocab_size,)
        temperature: Sampling temperature

    Returns:
        Sampled token ID
    """
    return sample(logits, temperature)


def generate(
    model,
    prompt_ids: np.ndarray,
    max_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    stop_token_id: Optional[int] = None
) -> np.ndarray:
    """
    Generate text autoregressively.

    Args:
        model: TransformerLM instance
        prompt_ids: Initial token IDs, shape (seq_len,)
        max_length: Maximum total length
        temperature: Sampling temperature
        top_k: If set, use top-k sampling
        top_p: If set, use nucleus sampling
        stop_token_id: Token ID to stop generation

    Returns:
        Generated token IDs
    """
    generated = list(prompt_ids)

    for _ in range(max_length - len(prompt_ids)):
        # Prepare input (add batch dimension)
        input_ids = np.array([generated])

        # Forward pass
        logits = model.forward(input_ids)

        # Get logits for last position
        next_token_logits = logits[0, -1, :]

        # Sample next token
        if top_k is not None:
            next_token = sample_top_k(next_token_logits, top_k, temperature)
        elif top_p is not None:
            next_token = sample_top_p(next_token_logits, top_p, temperature)
        else:
            next_token = sample(next_token_logits, temperature)

        generated.append(next_token)

        # Check stop condition
        if stop_token_id is not None and next_token == stop_token_id:
            break

    return np.array(generated)
