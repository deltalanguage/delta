"""
Compiler-level introspection for Delta.

Provides deep inspection of IR, type information, and execution traces.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import torch
from torch import Tensor
from torch import fx

from delta.ir.sir import SIRNode, SIRModule
from delta.types.types import Type
from delta.types.effects import Effect
from delta.types.roles import Role


class IntrospectionMode(Enum):
    """Level of introspection detail."""
    MINIMAL = auto()     # Just structure
    STANDARD = auto()    # Structure + types
    DETAILED = auto()    # Structure + types + effects + roles
    FULL = auto()        # Everything including internal state


@dataclass
class NodeInfo:
    """Introspection info for a single SIR node."""
    node_id: str
    kind: str
    inputs: List[str]
    outputs: List[str]
    type_info: Optional[Type] = None
    effect_info: Optional[Effect] = None
    role_info: Optional[Role] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "type": str(self.type_info) if self.type_info else None,
            "effect": str(self.effect_info) if self.effect_info else None,
            "role": str(self.role_info) if self.role_info else None,
            "metadata": self.metadata,
        }


@dataclass
class GraphInfo:
    """Introspection info for a graph."""
    name: str
    nodes: List[NodeInfo]
    inputs: List[str]
    outputs: List[str]
    node_count: int
    edge_count: int
    constraint_count: int
    has_stochastic: bool
    has_gradients: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "inputs": self.inputs,
            "outputs": self.outputs,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "constraint_count": self.constraint_count,
            "has_stochastic": self.has_stochastic,
            "has_gradients": self.has_gradients,
        }


@dataclass
class IntrospectionResult:
    """Result of introspecting a compiled program."""
    source_info: Dict[str, Any]
    ast_info: Dict[str, Any]
    sir_info: GraphInfo
    fx_info: Optional[Dict[str, Any]]
    type_environment: Dict[str, str]
    effect_summary: Dict[str, str]
    role_summary: Dict[str, str]
    warnings: List[str]
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = ["=" * 60]
        lines.append("INTROSPECTION RESULT")
        lines.append("=" * 60)
        
        lines.append(f"\nGraph: {self.sir_info.name}")
        lines.append(f"  Nodes: {self.sir_info.node_count}")
        lines.append(f"  Edges: {self.sir_info.edge_count}")
        lines.append(f"  Constraints: {self.sir_info.constraint_count}")
        lines.append(f"  Has stochastic: {self.sir_info.has_stochastic}")
        lines.append(f"  Has gradients: {self.sir_info.has_gradients}")
        
        if self.type_environment:
            lines.append("\nType Environment:")
            for name, typ in self.type_environment.items():
                lines.append(f"  {name}: {typ}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


class Introspector:
    """Deep introspection of Delta compiled programs."""
    
    def __init__(self, mode: IntrospectionMode = IntrospectionMode.STANDARD) -> None:
        self.mode = mode
        self._type_cache: Dict[str, Type] = {}
        self._effect_cache: Dict[str, Effect] = {}
        self._role_cache: Dict[str, Role] = {}
    
    def introspect_sir(self, graph: SIRModule) -> GraphInfo:
        """Introspect a SIR graph."""
        nodes = []
        edge_count = 0
        constraint_count = 0
        has_stochastic = False
        has_gradients = False
        
        for node in graph.nodes:
            node_info = self._introspect_node(node)
            nodes.append(node_info)
            
            edge_count += len(node.inputs)
            
            if node.kind.value == "constraint_op":
                constraint_count += 1
            if node.kind.value == "random_var":
                has_stochastic = True
            if node.kind.value == "grad_boundary":
                has_gradients = True
        
        return GraphInfo(
            name=graph.name,
            nodes=nodes,
            inputs=[n.name for n in graph.inputs],
            outputs=[n.name for n in graph.outputs],
            node_count=len(nodes),
            edge_count=edge_count,
            constraint_count=constraint_count,
            has_stochastic=has_stochastic,
            has_gradients=has_gradients,
        )
    
    def _introspect_node(self, node: SIRNode) -> NodeInfo:
        """Introspect a single SIR node."""
        info = NodeInfo(
            node_id=node.name,
            kind=node.kind.value,
            inputs=[inp.name for inp in node.inputs],
            outputs=[out.name for out in node.outputs],
        )
        
        if self.mode in (IntrospectionMode.STANDARD, 
                         IntrospectionMode.DETAILED, 
                         IntrospectionMode.FULL):
            info.type_info = getattr(node, 'dtype', None)
        
        if self.mode in (IntrospectionMode.DETAILED, IntrospectionMode.FULL):
            info.effect_info = getattr(node, 'effect', None)
            info.role_info = getattr(node, 'role', None)
        
        if self.mode == IntrospectionMode.FULL:
            info.metadata = {
                "shape": getattr(node, 'shape', None),
                "requires_grad": getattr(node, 'requires_grad', None),
                "temperature": getattr(node, 'temperature', None),
            }
        
        return info
    
    def introspect_fx(self, module: fx.GraphModule) -> Dict[str, Any]:
        """Introspect a PyTorch FX graph module."""
        graph = module.graph
        
        nodes = []
        for node in graph.nodes:
            nodes.append({
                "name": node.name,
                "op": node.op,
                "target": str(node.target) if node.target else None,
                "args": [str(a) for a in node.args],
                "kwargs": {k: str(v) for k, v in node.kwargs.items()},
            })
        
        return {
            "node_count": len(nodes),
            "nodes": nodes,
        }
    
    def introspect_types(
        self,
        type_env: Dict[str, Type]
    ) -> Dict[str, str]:
        """Introspect type environment."""
        return {name: str(typ) for name, typ in type_env.items()}
    
    def introspect_effects(
        self,
        effect_env: Dict[str, Effect]
    ) -> Dict[str, str]:
        """Introspect effect environment."""
        return {name: str(eff) for name, eff in effect_env.items()}
    
    def introspect_roles(
        self,
        role_env: Dict[str, Role]
    ) -> Dict[str, str]:
        """Introspect role environment."""
        return {name: str(role) for name, role in role_env.items()}


def introspect(
    graph: SIRModule,
    mode: IntrospectionMode = IntrospectionMode.STANDARD,
    fx_module: Optional[fx.GraphModule] = None,
    type_env: Optional[Dict[str, Type]] = None,
    effect_env: Optional[Dict[str, Effect]] = None,
    role_env: Optional[Dict[str, Role]] = None,
) -> IntrospectionResult:
    """Introspect a compiled Delta program."""
    inspector = Introspector(mode)
    
    sir_info = inspector.introspect_sir(graph)
    
    fx_info = None
    if fx_module is not None:
        fx_info = inspector.introspect_fx(fx_module)
    
    type_summary = {}
    if type_env is not None:
        type_summary = inspector.introspect_types(type_env)
    
    effect_summary = {}
    if effect_env is not None:
        effect_summary = inspector.introspect_effects(effect_env)
    
    role_summary = {}
    if role_env is not None:
        role_summary = inspector.introspect_roles(role_env)
    
    warnings = []
    if sir_info.has_stochastic and not sir_info.has_gradients:
        warnings.append("Stochastic nodes without gradient boundaries")
    
    return IntrospectionResult(
        source_info={},
        ast_info={},
        sir_info=sir_info,
        fx_info=fx_info,
        type_environment=type_summary,
        effect_summary=effect_summary,
        role_summary=role_summary,
        warnings=warnings,
    )


# ============================================================
# Execution Tracing
# ============================================================

@dataclass
class ExecutionTrace:
    """Trace of program execution."""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    tensors: Dict[str, List[Tensor]] = field(default_factory=dict)
    
    def record(
        self,
        step: int,
        node_name: str,
        value: Tensor,
        grad: Optional[Tensor] = None
    ) -> None:
        """Record an execution step."""
        self.steps.append({
            "step": step,
            "node": node_name,
            "shape": tuple(value.shape),
            "mean": value.mean().item(),
            "std": value.std().item() if value.numel() > 1 else 0.0,
            "grad_norm": grad.norm().item() if grad is not None else None,
        })
        
        if node_name not in self.tensors:
            self.tensors[node_name] = []
        self.tensors[node_name].append(value.detach().clone())
    
    def get_history(self, node_name: str) -> List[Tensor]:
        """Get value history for a node."""
        return self.tensors.get(node_name, [])
    
    def summary(self) -> str:
        """Generate trace summary."""
        lines = ["Execution Trace:"]
        lines.append(f"  Total steps: {len(self.steps)}")
        lines.append(f"  Traced nodes: {len(self.tensors)}")
        
        for step in self.steps[:10]:  # First 10 steps
            lines.append(
                f"  [{step['step']}] {step['node']}: "
                f"shape={step['shape']}, mean={step['mean']:.4f}"
            )
        
        if len(self.steps) > 10:
            lines.append(f"  ... and {len(self.steps) - 10} more steps")
        
        return "\n".join(lines)


class ExecutionTracer:
    """Traces program execution for debugging."""
    
    def __init__(self) -> None:
        self._trace: Optional[ExecutionTrace] = None
        self._step = 0
        self._hooks: List[Any] = []
    
    def start(self) -> ExecutionTrace:
        """Start tracing."""
        self._trace = ExecutionTrace()
        self._step = 0
        return self._trace
    
    def stop(self) -> ExecutionTrace:
        """Stop tracing and return trace."""
        trace = self._trace
        self._trace = None
        return trace
    
    def is_tracing(self) -> bool:
        """Check if currently tracing."""
        return self._trace is not None
    
    def record(
        self,
        node_name: str,
        value: Tensor,
        grad: Optional[Tensor] = None
    ) -> None:
        """Record a value during execution."""
        if self._trace is not None:
            self._trace.record(self._step, node_name, value, grad)
            self._step += 1
    
    def add_hook(
        self,
        module: torch.nn.Module,
        node_name: str
    ) -> None:
        """Add a forward hook to trace a module."""
        def hook(mod, inp, out):
            if isinstance(out, Tensor):
                self.record(node_name, out)
        
        handle = module.register_forward_hook(hook)
        self._hooks.append(handle)
    
    def remove_hooks(self) -> None:
        """Remove all hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []


# Global tracer
_tracer = ExecutionTracer()


def start_trace() -> ExecutionTrace:
    """Start global execution trace."""
    return _tracer.start()


def stop_trace() -> ExecutionTrace:
    """Stop global execution trace."""
    return _tracer.stop()


def trace_value(node_name: str, value: Tensor, grad: Optional[Tensor] = None) -> None:
    """Trace a value in global tracer."""
    _tracer.record(node_name, value, grad)
