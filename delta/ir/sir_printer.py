"""
SIR pretty printer for debugging and visualization.

Provides human-readable output of SIR graphs, including:
- Node types and IDs
- Properties (dtype, shape, grad)
- Graph structure
"""

from __future__ import annotations
from typing import TextIO
import sys

from delta.ir.sir import (
    SIRNode, SIRModule, SIRFunction, SIRBlock,
    TensorOp, GateOp, MixOp, ConstraintOp,
    ParamRef, ObsRef, Const, StopGrad, Harden, GradBoundary,
    RandomVar, Observe, ModeSwitch,
    walk_sir,
)


class SIRPrinter:
    """
    Pretty printer for SIR.
    
    Outputs human-readable representation of SIR graphs
    for debugging and inspection.
    """
    
    def __init__(self, out: TextIO = sys.stdout) -> None:
        self.out = out
        self.indent_level = 0
        self.printed_nodes: set[int] = set()
    
    def print_module(self, module: SIRModule) -> None:
        """Print an entire SIR module."""
        self._write(f"// SIR Module: {module.name}\n")
        self._write("\n")
        
        # Print parameters
        if module.params:
            self._write("// Parameters\n")
            for name, param in module.params.items():
                self._write(f"param {name}: {param.dtype}\n")
            self._write("\n")
        
        # Print constants
        if module.constants:
            self._write("// Constants\n")
            for name, const in module.constants.items():
                self._write(f"const {name} = {const.value}\n")
            self._write("\n")
        
        # Print functions
        for name, func in module.functions.items():
            self.print_function(func)
            self._write("\n")
        
        # Print constraints
        if module.constraints:
            self._write("// Constraints\n")
            for constraint in module.constraints:
                self.print_node(constraint)
            self._write("\n")
    
    def print_function(self, func: SIRFunction) -> None:
        """Print a SIR function."""
        params_str = ", ".join(f"{name}: {props.dtype}" for name, props in func.params)
        effects_str = f" / {func.effects}" if not func.effects.is_pure() else ""
        
        self._write(f"fn {func.name}({params_str}) -> {func.return_type}{effects_str} {{\n")
        self.indent_level += 1
        
        self.print_block(func.body)
        
        self.indent_level -= 1
        self._write("}\n")
    
    def print_block(self, block: SIRBlock) -> None:
        """Print a SIR block."""
        for node in block.nodes:
            self.print_node(node)
        
        if block.result:
            self._indent()
            self._write(f"return %{block.result.id}\n")
    
    def print_node(self, node: SIRNode) -> None:
        """Print a single SIR node."""
        if node.id in self.printed_nodes:
            return
        
        # Print dependencies first
        for inp in node.inputs():
            if inp.id not in self.printed_nodes:
                self.print_node(inp)
        
        self.printed_nodes.add(node.id)
        self._indent()
        
        # Node assignment
        grad_marker = " [grad]" if node.requires_grad else ""
        self._write(f"%{node.id}: {node.dtype}{grad_marker} = ")
        
        # Node content
        if isinstance(node, TensorOp):
            args = ", ".join(f"%{op.id}" for op in node.operands)
            attrs = ""
            if node.attrs:
                attrs = ", " + ", ".join(f"{k}={v}" for k, v in node.attrs.items())
            self._write(f"{node.op.name}({args}{attrs})\n")
        
        elif isinstance(node, GateOp):
            self._write(f"Gate({node.compare.name}, %{node.lhs.id}, %{node.rhs.id}, temp=%{node.temperature.id})\n")
        
        elif isinstance(node, MixOp):
            self._write(f"Mix(%{node.gate.id}, %{node.then_value.id}, %{node.else_value.id})\n")
        
        elif isinstance(node, ConstraintOp):
            rhs = f", %{node.rhs.id}" if node.rhs else ""
            self._write(f"Constraint({node.kind.name}, %{node.lhs.id}{rhs}, weight=%{node.weight.id})\n")
        
        elif isinstance(node, ParamRef):
            self._write(f"Param({node.name})\n")
        
        elif isinstance(node, ObsRef):
            self._write(f"Obs({node.name})\n")
        
        elif isinstance(node, Const):
            value_str = repr(node.value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            self._write(f"Const({value_str})\n")
        
        elif isinstance(node, StopGrad):
            self._write(f"StopGrad(%{node.operand.id})\n")
        
        elif isinstance(node, Harden):
            self._write(f"Harden(%{node.operand.id}, threshold={node.threshold})\n")
        
        elif isinstance(node, GradBoundary):
            self._write(f"GradBoundary(%{node.operand.id}, {node.kind})\n")
        
        elif isinstance(node, RandomVar):
            params = ", ".join(f"%{p.id}" for p in node.params)
            self._write(f"Rand({node.distribution}({params}))\n")
        
        elif isinstance(node, Observe):
            params = ", ".join(f"%{p.id}" for p in node.params)
            self._write(f"Observe(%{node.value.id}, {node.distribution}({params}))\n")
        
        elif isinstance(node, ModeSwitch):
            self._write(f"ModeSwitch(train=%{node.train_value.id}, infer=%{node.infer_value.id})\n")
        
        else:
            self._write(f"Unknown({type(node).__name__})\n")
    
    def _indent(self) -> None:
        """Write indentation."""
        self._write("  " * self.indent_level)
    
    def _write(self, s: str) -> None:
        """Write to output."""
        self.out.write(s)


def format_sir(module: SIRModule) -> str:
    """Format a SIR module as a string."""
    from io import StringIO
    out = StringIO()
    printer = SIRPrinter(out)
    printer.print_module(module)
    return out.getvalue()


def format_node(node: SIRNode) -> str:
    """Format a single SIR node and its dependencies."""
    from io import StringIO
    out = StringIO()
    printer = SIRPrinter(out)
    printer.print_node(node)
    return out.getvalue()


def format_graph(root: SIRNode) -> str:
    """Format an entire SIR graph from a root node."""
    from io import StringIO
    out = StringIO()
    printer = SIRPrinter(out)
    
    nodes = walk_sir(root)
    for node in nodes:
        printer.print_node(node)
    
    return out.getvalue()
