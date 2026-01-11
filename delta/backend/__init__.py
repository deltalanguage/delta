"""
Delta backend: PyTorch FX lowering and code generation.

The backend lowers SIR to PyTorch FX graphs that can be:
- Executed directly
- Further optimized by PyTorch
- Compiled with torch.compile
"""

from delta.backend.fx_lowering import FXLowering, lower_to_fx
from delta.backend.fx_optimize import FXOptimizer, optimize_fx
from delta.backend.codegen import CodeGenerator, generate_python

__all__ = [
    "FXLowering", "lower_to_fx",
    "FXOptimizer", "optimize_fx",
    "CodeGenerator", "generate_python",
]
