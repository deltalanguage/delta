"""
Delta: A Differentiable, Constraint-Oriented Programming Language

Delta is a staged compiler that takes learning intent (constraints, roles,
uncertainty, differentiability) and compiles it into an optimized tensor
program executed by PyTorch.

Key invariant: Delta never executes tensor math itself. It only rewrites,
simplifies, specializes, and erases work before execution.

Usage:
    import delta
    
    model = delta.run("model.delta", train_loader, val_loader, epochs=10)
    predictions = model.predict(test_data)
"""

from delta.compiler import Compiler, CompileOptions, CompileResult
from delta.source import SourceFile, SourceLocation
from delta.errors import DeltaError, CompileError, TypeError, ModeError
from delta.debug import introspect, diagnose_optimization, print_sir, print_fx
from delta.run import compile, run, DeltaModel

__version__ = "0.1.0"
__all__ = [
    # High-level API
    "compile",
    "run",
    "DeltaModel",
    # Core compiler
    "Compiler",
    "CompileOptions",
    "CompileResult",
    # Source handling
    "SourceFile",
    "SourceLocation",
    # Errors
    "DeltaError",
    "CompileError",
    "TypeError",
    "ModeError",
    # Debugging
    "introspect",
    "diagnose_optimization",
    "print_sir",
    "print_fx",
]
