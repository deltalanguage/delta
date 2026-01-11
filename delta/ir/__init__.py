"""
Delta Semantic IR (SIR).

SIR is the core intermediate representation where Delta's
differentiable semantics are fully expressed. It represents:

- Tensor operations
- Gates (soft comparisons)
- Mixes (soft conditionals)
- Constraints
- Role provenance
- Gradient requirements
- Mode validity
"""

from delta.ir.sir import (
    SIRNode, SIRProperty,
    TensorOp, GateOp, MixOp, ConstraintOp,
    ParamRef, ObsRef, Const, StopGrad, Harden,
    RandomVar, Observe, GradBoundary, ModeSwitch,
    BinaryTensorOp, UnaryTensorOp, ReduceOp, ShapeOp,
    SIRModule, SIRFunction, SIRBlock,
)
from delta.ir.sir_builder import SIRBuilder
from delta.ir.sir_printer import SIRPrinter, format_sir

__all__ = [
    # Nodes
    "SIRNode", "SIRProperty",
    "TensorOp", "GateOp", "MixOp", "ConstraintOp",
    "ParamRef", "ObsRef", "Const", "StopGrad", "Harden",
    "RandomVar", "Observe", "GradBoundary", "ModeSwitch",
    "BinaryTensorOp", "UnaryTensorOp", "ReduceOp", "ShapeOp",
    "SIRModule", "SIRFunction", "SIRBlock",
    # Builder
    "SIRBuilder",
    # Printer
    "SIRPrinter", "format_sir",
]
