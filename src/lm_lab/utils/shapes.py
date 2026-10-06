"""
Shape assertion and debugging utilities.

Provides utilities for verifying tensor shapes during development:
- Shape assertions with clear error messages
- Shape debugging helpers
"""

from typing import Sequence, Optional
import torch


def assert_shape(
    tensor: torch.Tensor,
    expected_shape: Sequence[Optional[int]],
    name: str = "tensor"
) -> None:
    """
    Assert that tensor has expected shape.

    Use None for dimensions that can be any size.

    Args:
        tensor: Tensor to check
        expected_shape: Expected shape (use None for any size)
        name: Name for error message

    Raises:
        AssertionError: If shape doesn't match

    Example:
        >>> x = torch.randn(32, 128, 512)
        >>> assert_shape(x, [32, None, 512], "embeddings")
    """
    actual_shape = tuple(tensor.shape)

    if len(actual_shape) != len(expected_shape):
        raise AssertionError(
            f"{name} has {len(actual_shape)} dimensions, "
            f"expected {len(expected_shape)}\n"
            f"  Actual: {actual_shape}\n"
            f"  Expected: {expected_shape}"
        )

    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected is not None and actual != expected:
            raise AssertionError(
                f"{name} dimension {i} is {actual}, expected {expected}\n"
                f"  Actual shape: {actual_shape}\n"
                f"  Expected shape: {expected_shape}"
            )


def debug_shapes(**tensors: torch.Tensor) -> str:
    """
    Format tensor shapes for debugging.

    Args:
        **tensors: Named tensors to display

    Returns:
        Formatted string with shapes

    Example:
        >>> print(debug_shapes(q=q, k=k, v=v, out=out))
        q: torch.Size([32, 8, 128, 64])
        k: torch.Size([32, 8, 128, 64])
        v: torch.Size([32, 8, 128, 64])
        out: torch.Size([32, 128, 512])
    """
    lines = []
    for name, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            lines.append(f"{name}: {tensor.shape}")
        else:
            lines.append(f"{name}: {type(tensor).__name__}")

    return "\n".join(lines)


def shapes_match(tensor1: torch.Tensor, tensor2: torch.Tensor) -> bool:
    """Check if two tensors have the same shape."""
    return tensor1.shape == tensor2.shape


def broadcastable(shape1: Sequence[int], shape2: Sequence[int]) -> bool:
    """
    Check if two shapes are broadcastable.

    Args:
        shape1: First shape
        shape2: Second shape

    Returns:
        True if shapes are broadcastable
    """
    # Pad shorter shape with 1s on the left
    if len(shape1) < len(shape2):
        shape1 = (1,) * (len(shape2) - len(shape1)) + tuple(shape1)
    elif len(shape2) < len(shape1):
        shape2 = (1,) * (len(shape1) - len(shape2)) + tuple(shape2)

    # Check each dimension
    for d1, d2 in zip(shape1, shape2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            return False

    return True
