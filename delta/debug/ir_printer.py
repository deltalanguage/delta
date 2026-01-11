"""
IR Pretty Printing for Delta.

Provides human-readable visualization of SIR and FX graphs.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, TextIO
import sys
from io import StringIO
import torch
from torch import fx

from delta.ir.sir import (
    SIRNode, SIRModule, SIRFunction, SIRBlock,
    TensorOp, GateOp, MixOp, ConstraintOp,
    ParamRef, ObsRef, Const, StopGrad, Harden,
    RandomVar, Observe, GradBoundary, ModeSwitch,
    BinaryTensorOp, UnaryTensorOp,
)


# ============================================================
# SIR Pretty Printing
# ============================================================

def format_sir(
    module: SIRModule,
    show_types: bool = True,
    show_effects: bool = True,
    show_roles: bool = True,
    indent: int = 2
) -> str:
    """Format a SIR module as a string."""
    output = StringIO()
    _print_sir_impl(module, output, show_types, show_effects, show_roles, indent)
    return output.getvalue()


def print_sir(
    module: SIRModule,
    show_types: bool = True,
    show_effects: bool = True,
    show_roles: bool = True,
    indent: int = 2,
    file: TextIO = sys.stdout
) -> None:
    """Print a SIR module."""
    _print_sir_impl(module, file, show_types, show_effects, show_roles, indent)


def _print_sir_impl(
    module: SIRModule,
    output: TextIO,
    show_types: bool,
    show_effects: bool,
    show_roles: bool,
    indent: int
) -> None:
    """Implementation of SIR printing."""
    sp = " " * indent
    
    # Header
    output.write("=" * 60 + "\n")
    output.write(f"SIR Module: {module.name}\n")
    output.write("=" * 60 + "\n\n")
    
    # Parameters
    if module.params:
        output.write("Parameters:\n")
        for name, param in module.params.items():
            output.write(f"{sp}{name}: {_format_props(param, show_types, show_roles)}\n")
        output.write("\n")
    
    # Constraints
    if module.constraints:
        output.write("Constraints:\n")
        for constraint in module.constraints:
            output.write(f"{sp}{_format_node(constraint)}\n")
        output.write("\n")
    
    # Functions
    for name, func in module.functions.items():
        _print_function(func, output, show_types, show_effects, show_roles, indent)
    
    output.write("=" * 60 + "\n")


def _print_function(
    func: SIRFunction,
    output: TextIO,
    show_types: bool,
    show_effects: bool,
    show_roles: bool,
    indent: int
) -> None:
    """Print a SIR function."""
    sp = " " * indent
    
    # Function signature
    params = ", ".join(f"{name}" for name, _ in func.params)
    output.write(f"fn {func.name}({params}) -> {func.return_type}:\n")
    
    # Body
    _print_block(func.body, output, show_types, show_effects, show_roles, indent)
    output.write("\n")


def _print_block(
    block: SIRBlock,
    output: TextIO,
    show_types: bool,
    show_effects: bool,
    show_roles: bool,
    indent: int
) -> None:
    """Print a SIR block."""
    sp = " " * indent
    
    for node in block.nodes:
        output.write(f"{sp}{_format_node(node)}\n")
    
    if block.result:
        output.write(f"{sp}return {_format_node(block.result)}\n")


def _format_props(node: SIRNode, show_types: bool, show_roles: bool) -> str:
    """Format node properties."""
    parts = []
    if show_types and hasattr(node, '_props') and node._props:
        parts.append(str(node._props.dtype))
    if show_roles and hasattr(node, '_props') and node._props and node._props.role:
        parts.append(str(node._props.role))
    return ", ".join(parts) if parts else ""


def _format_node(node: SIRNode) -> str:
    """Format a SIR node."""
    if isinstance(node, BinaryTensorOp):
        return f"{node.op.name.lower()}({_format_node(node.left)}, {_format_node(node.right)})"
    
    elif isinstance(node, UnaryTensorOp):
        return f"{node.op.name.lower()}({_format_node(node.operand)})"
    
    elif isinstance(node, TensorOp):
        args = ", ".join(_format_node(op) for op in node.operands)
        return f"{node.op.name.lower()}({args})"
    
    elif isinstance(node, GateOp):
        return f"gate({_format_node(node.lhs)} {node.compare.name} {_format_node(node.rhs)}, temp={_format_node(node.temperature)})"
    
    elif isinstance(node, MixOp):
        return f"mix({_format_node(node.gate)}, {_format_node(node.then_value)}, {_format_node(node.else_value)})"
    
    elif isinstance(node, ConstraintOp):
        return f"constraint({_format_node(node.lhs)}, weight={_format_node(node.weight)})"
    
    elif isinstance(node, ParamRef):
        return f"param({node.name})"
    
    elif isinstance(node, ObsRef):
        return f"obs({node.name})"
    
    elif isinstance(node, Const):
        return repr(node.value)
    
    elif isinstance(node, StopGrad):
        return f"stop_grad({_format_node(node.operand)})"
    
    elif isinstance(node, Harden):
        return f"harden({_format_node(node.operand)})"
    
    elif isinstance(node, RandomVar):
        params = ", ".join(_format_node(p) for p in node.params)
        return f"rand({node.distribution}({params}))"
    
    elif isinstance(node, Observe):
        params = ", ".join(_format_node(p) for p in node.params)
        return f"observe({_format_node(node.value)}, {node.distribution}({params}))"
    
    elif isinstance(node, GradBoundary):
        return f"grad_boundary({_format_node(node.operand)})"
    
    elif isinstance(node, ModeSwitch):
        return f"mode_switch({node.mode})"
    
    else:
        return f"<{type(node).__name__}>"


# ============================================================
# FX Pretty Printing
# ============================================================

def format_fx(
    module: fx.GraphModule,
    show_shapes: bool = True,
    indent: int = 2
) -> str:
    """Format a PyTorch FX graph module as a string."""
    output = StringIO()
    _print_fx_impl(module, output, show_shapes, indent)
    return output.getvalue()


def print_fx(
    module: fx.GraphModule,
    show_shapes: bool = True,
    indent: int = 2,
    file: TextIO = sys.stdout
) -> None:
    """Print a PyTorch FX graph module."""
    _print_fx_impl(module, file, show_shapes, indent)


def _print_fx_impl(
    module: fx.GraphModule,
    output: TextIO,
    show_shapes: bool,
    indent: int
) -> None:
    """Implementation of FX printing."""
    sp = " " * indent
    graph = module.graph
    
    output.write("=" * 60 + "\n")
    output.write("FX Graph Module\n")
    output.write("=" * 60 + "\n\n")
    
    for node in graph.nodes:
        output.write(f"{sp}{_format_fx_node(node)}\n")
    
    output.write("\n" + "=" * 60 + "\n")


def _format_fx_node(node: fx.Node) -> str:
    """Format a single FX node."""
    if node.op == "placeholder":
        return f"{node.name} = input()"
    
    elif node.op == "get_attr":
        return f"{node.name} = {node.target}"
    
    elif node.op == "call_function":
        name = getattr(node.target, "__name__", str(node.target))
        args = ", ".join(_format_fx_arg(a) for a in node.args)
        return f"{node.name} = {name}({args})"
    
    elif node.op == "call_method":
        self_arg = _format_fx_arg(node.args[0]) if node.args else "?"
        rest = ", ".join(_format_fx_arg(a) for a in node.args[1:])
        return f"{node.name} = {self_arg}.{node.target}({rest})"
    
    elif node.op == "call_module":
        args = ", ".join(_format_fx_arg(a) for a in node.args)
        return f"{node.name} = {node.target}({args})"
    
    elif node.op == "output":
        return f"return {_format_fx_arg(node.args[0])}"
    
    return f"{node.name} = {node.op}(...)"


def _format_fx_arg(arg: Any) -> str:
    """Format an FX argument."""
    if isinstance(arg, fx.Node):
        return arg.name
    elif isinstance(arg, (list, tuple)):
        inner = ", ".join(_format_fx_arg(a) for a in arg)
        return f"[{inner}]"
    return repr(arg)


def compare_graphs(sir: SIRModule, fx_module: fx.GraphModule) -> str:
    """Compare SIR and FX graphs side by side."""
    sir_str = format_sir(sir, show_types=False, show_effects=False, show_roles=False)
    fx_str = format_fx(fx_module, show_shapes=False)
    
    lines = ["=" * 80, "SIR vs FX Comparison", "=" * 80, ""]
    lines.append("--- SIR ---")
    lines.append(sir_str)
    lines.append("\n--- FX ---")
    lines.append(fx_str)
    lines.append("=" * 80)
    
    return "\n".join(lines)
