"""
Utility functions for numerical stability and debugging.

This module provides:
- Numerical stability checks
- Shape assertion helpers
- Gradient checking utilities
- Logging and monitoring
"""

from .numerical import check_finite, safe_softmax, stable_log_softmax
from .shapes import assert_shape, debug_shapes
from .seed import set_seed

__all__ = [
    'check_finite',
    'safe_softmax',
    'stable_log_softmax',
    'assert_shape',
    'debug_shapes',
    'set_seed',
]
