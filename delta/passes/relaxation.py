"""
Relaxation pass for differentiability.

Transforms hard control flow into soft, differentiable alternatives:
- Comparisons -> Gates (sigmoid of difference)
- If expressions -> Mix (weighted combination)
- Boolean logic -> Soft logic (multiplication, etc.)
- While loops -> (forbidden unless in non_diff)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from delta.ir.sir import (
    SIRNode, SIRModule, SIRFunction, SIRBlock, SIRProperty,
    TensorOp, TensorOpKind, GateOp, MixOp, ConstraintOp,
    ParamRef, ObsRef, Const, StopGrad, Harden,
    RandomVar, Observe, ModeSwitch,
    BinaryTensorOp, UnaryTensorOp,
    walk_sir, replace_inputs,
)
from delta.types.types import FloatType, BoolType


@dataclass
class RelaxationConfig:
    """Configuration for relaxation pass."""
    default_temperature: float = 1.0
    min_temperature: float = 0.01
    anneal_temperature: bool = False
    anneal_schedule: str = "linear"  # linear, exponential, cosine


class RelaxationPass:
    """
    Relaxes hard operations to differentiable alternatives.
    
    Key transformations:
    1. Comparison ops -> GateOp (sigmoid)
    2. Hard select -> MixOp (weighted average)
    3. Boolean AND -> multiplication
    4. Boolean OR -> soft OR (1 - (1-a)(1-b))
    5. Boolean NOT -> 1 - x
    """
    
    def __init__(self, config: Optional[RelaxationConfig] = None) -> None:
        self.config = config or RelaxationConfig()
        self.node_map: dict[int, SIRNode] = {}
    
    def relax(self, module: SIRModule) -> SIRModule:
        """Relax a SIR module."""
        new_module = SIRModule(name=module.name)
        
        # Copy params
        for name, param in module.params.items():
            new_module.params[name] = param
        
        # Copy constants
        for name, const in module.constants.items():
            new_module.constants[name] = const
        
        # Relax functions
        for name, func in module.functions.items():
            new_func = self._relax_function(func)
            new_module.functions[name] = new_func
        
        # Relax constraints
        for constraint in module.constraints:
            new_constraint = self._relax_node(constraint)
            if isinstance(new_constraint, ConstraintOp):
                new_module.constraints.append(new_constraint)
        
        return new_module
    
    def _relax_function(self, func: SIRFunction) -> SIRFunction:
        """Relax a function."""
        new_body = self._relax_block(func.body)
        
        return SIRFunction(
            name=func.name,
            params=func.params,
            body=new_body,
            return_type=func.return_type,
            effects=func.effects
        )
    
    def _relax_block(self, block: SIRBlock) -> SIRBlock:
        """Relax a block."""
        new_nodes = [self._relax_node(node) for node in block.nodes]
        new_result = self._relax_node(block.result) if block.result else None
        
        return SIRBlock(nodes=new_nodes, result=new_result)
    
    def _relax_node(self, node: SIRNode) -> SIRNode:
        """Relax a single node."""
        if node.id in self.node_map:
            return self.node_map[node.id]
        
        # First relax all inputs
        new_inputs = [self._relax_node(inp) for inp in node.inputs()]
        
        # Then handle specific node types
        if isinstance(node, TensorOp):
            relaxed = self._relax_tensor_op(node, new_inputs)
        elif isinstance(node, GateOp):
            relaxed = self._relax_gate_op(node, new_inputs)
        elif isinstance(node, MixOp):
            relaxed = self._relax_mix_op(node, new_inputs)
        elif isinstance(node, ConstraintOp):
            relaxed = self._relax_constraint_op(node, new_inputs)
        elif isinstance(node, StopGrad):
            relaxed = StopGrad(operand=new_inputs[0], _props=node.props)
        elif isinstance(node, Harden):
            # In relaxation mode, keep gates soft
            relaxed = new_inputs[0]  # Just pass through without hardening
        else:
            # Clone with new inputs
            if new_inputs != node.inputs():
                relaxed = node.clone_with_inputs(new_inputs)
            else:
                relaxed = node
        
        self.node_map[node.id] = relaxed
        return relaxed
    
    def _relax_tensor_op(self, node: TensorOp, new_inputs: list[SIRNode]) -> SIRNode:
        """Relax a tensor operation."""
        # Handle comparison operators - convert to gates
        if node.op in (TensorOpKind.LT, TensorOpKind.LE, 
                       TensorOpKind.GT, TensorOpKind.GE,
                       TensorOpKind.EQ, TensorOpKind.NE):
            
            if len(new_inputs) >= 2:
                temp = Const(value=self.config.default_temperature)
                return GateOp(
                    compare=node.op,
                    lhs=new_inputs[0],
                    rhs=new_inputs[1],
                    temperature=temp,
                    _props=SIRProperty(dtype=FloatType(), requires_grad=True)
                )
        
        # Clone with new inputs
        return node.clone_with_inputs(new_inputs)
    
    def _relax_gate_op(self, node: GateOp, new_inputs: list[SIRNode]) -> SIRNode:
        """Relax a gate operation (already soft, just update inputs)."""
        return GateOp(
            compare=node.compare,
            lhs=new_inputs[0],
            rhs=new_inputs[1],
            temperature=new_inputs[2],
            _props=node.props
        )
    
    def _relax_mix_op(self, node: MixOp, new_inputs: list[SIRNode]) -> SIRNode:
        """Relax a mix operation (already soft, just update inputs)."""
        return MixOp(
            gate=new_inputs[0],
            then_value=new_inputs[1],
            else_value=new_inputs[2],
            _props=node.props
        )
    
    def _relax_constraint_op(self, node: ConstraintOp, new_inputs: list[SIRNode]) -> ConstraintOp:
        """Relax a constraint operation."""
        # Constraints are already in penalty form, just update inputs
        idx = 0
        lhs = new_inputs[idx]; idx += 1
        weight = new_inputs[idx]; idx += 1
        rhs = new_inputs[idx] if node.rhs and idx < len(new_inputs) else None
        if node.rhs:
            idx += 1
        slack = new_inputs[idx] if node.slack and idx < len(new_inputs) else None
        
        return ConstraintOp(
            kind=node.kind,
            lhs=lhs,
            rhs=rhs,
            weight=weight,
            slack=slack,
            _props=node.props
        )


def relax_sir(module: SIRModule, config: Optional[RelaxationConfig] = None) -> SIRModule:
    """Convenience function to relax a SIR module."""
    relaxation = RelaxationPass(config)
    return relaxation.relax(module)
