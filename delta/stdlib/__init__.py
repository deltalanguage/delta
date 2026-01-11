"""
Delta Standard Library.

Provides built-in functionality for:
- Tensor operations (std.tensor)
- Neural network layers (std.nn)
- Optimizers (std.optim)
- Constraints (std.constraints)
- Distributions (std.dist)
- Data loading (std.data)
- Debugging (std.debug)
"""

from delta.stdlib import tensor
from delta.stdlib import nn
from delta.stdlib import optim
from delta.stdlib import constraints
from delta.stdlib import dist
from delta.stdlib import data
from delta.stdlib import debug

__all__ = [
    "tensor",
    "nn",
    "optim",
    "constraints",
    "dist",
    "data",
    "debug",
]
