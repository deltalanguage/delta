"""
Mode specialization pass.

Specializes SIR for a specific execution mode:
- train: Soft gates, gradients enabled, stochastic allowed
- infer: Hard gates, gradients disabled, no stochastic
- analyze: Like train plus tracing
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from delta.ir.sir import (
    SIRNode, SIRModule, SIRFunction, SIRBlock, SIRProperty,
    TensorOp, TensorOpKind, GateOp, MixOp, ConstraintOp,
    ParamRef, ObsRef, Const, StopGrad, Harden, GradBoundary,
    RandomVar, Observe, ModeSwitch, Mode,
    walk_sir,
)
from delta.types.types import FloatType, BoolType


class SpecializationMode(Enum):
    """Modes for specialization."""
    TRAIN = "train"
    INFER = "infer"
    ANALYZE = "analyze"


@dataclass
class SpecializationConfig:
    """Configuration for mode specialization."""
    remove_constraints_in_infer: bool = True
    harden_gates_in_infer: bool = True
    disable_gradients_in_infer: bool = True
    keep_random_in_analyze: bool = True


class ModeSpecializer:
    """
    Specializes SIR for a specific execution mode.
    
    Transformations:
    - TRAIN: Keep everything soft and differentiable
    - INFER: Harden gates, remove gradients, eliminate stochastic
    - ANALYZE: Like train but with additional tracing hooks
    """
    
    def __init__(
        self,
        mode: SpecializationMode,
        config: Optional[SpecializationConfig] = None
    ) -> None:
        self.mode = mode
        self.config = config or SpecializationConfig()
        self.node_map: dict[int, SIRNode] = {}
    
    def specialize(self, module: SIRModule) -> SIRModule:
        """Specialize a SIR module for the given mode."""
        new_module = SIRModule(name=module.name)
        
        # Copy and specialize params
        for name, param in module.params.items():
            new_param = self._specialize_param(param)
            new_module.params[name] = new_param
        
        # Copy constants
        for name, const in module.constants.items():
            new_module.constants[name] = const
        
        # Specialize functions
        for name, func in module.functions.items():
            new_func = self._specialize_function(func)
            new_module.functions[name] = new_func
        
        # Handle constraints based on mode
        if self.mode == SpecializationMode.INFER and self.config.remove_constraints_in_infer:
            # Remove constraints in infer mode
            pass
        else:
            for constraint in module.constraints:
                new_constraint = self._specialize_node(constraint)
                if isinstance(new_constraint, ConstraintOp):
                    new_module.constraints.append(new_constraint)
        
        return new_module
    
    def _specialize_param(self, param: ParamRef) -> ParamRef:
        """Specialize a parameter reference."""
        if self.mode == SpecializationMode.INFER:
            # Disable gradients in infer mode
            new_props = param.props.with_grad(False)
            return ParamRef(name=param.name, _props=new_props)
        return param
    
    def _specialize_function(self, func: SIRFunction) -> SIRFunction:
        """Specialize a function."""
        new_body = self._specialize_block(func.body)
        
        return SIRFunction(
            name=func.name,
            params=func.params,
            body=new_body,
            return_type=func.return_type,
            effects=func.effects
        )
    
    def _specialize_block(self, block: SIRBlock) -> SIRBlock:
        """Specialize a block."""
        new_nodes = []
        for node in block.nodes:
            new_node = self._specialize_node(node)
            if new_node is not None:
                new_nodes.append(new_node)
        
        new_result = None
        if block.result:
            new_result = self._specialize_node(block.result)
        
        return SIRBlock(nodes=new_nodes, result=new_result)
    
    def _specialize_node(self, node: SIRNode) -> Optional[SIRNode]:
        """Specialize a single node for the mode."""
        if node.id in self.node_map:
            return self.node_map[node.id]
        
        # First specialize all inputs
        new_inputs = []
        for inp in node.inputs():
            new_inp = self._specialize_node(inp)
            if new_inp is not None:
                new_inputs.append(new_inp)
        
        # Handle specific node types based on mode
        result: Optional[SIRNode]
        
        if isinstance(node, GateOp):
            result = self._specialize_gate(node, new_inputs)
        elif isinstance(node, MixOp):
            result = self._specialize_mix(node, new_inputs)
        elif isinstance(node, RandomVar):
            result = self._specialize_random(node, new_inputs)
        elif isinstance(node, ModeSwitch):
            result = self._specialize_mode_switch(node, new_inputs)
        elif isinstance(node, ConstraintOp):
            result = self._specialize_constraint(node, new_inputs)
        else:
            # Default: clone with new inputs
            if new_inputs != node.inputs():
                result = node.clone_with_inputs(new_inputs)
            else:
                result = node
            
            # Disable gradients in infer mode
            if self.mode == SpecializationMode.INFER and self.config.disable_gradients_in_infer:
                if result and result.requires_grad:
                    result = StopGrad(operand=result, _props=result.props.with_grad(False))
        
        if result:
            self.node_map[node.id] = result
        return result
    
    def _specialize_gate(self, node: GateOp, new_inputs: list[SIRNode]) -> SIRNode:
        """Specialize a gate operation."""
        if self.mode == SpecializationMode.INFER and self.config.harden_gates_in_infer:
            # Harden the gate to a boolean decision
            soft_gate = GateOp(
                compare=node.compare,
                lhs=new_inputs[0],
                rhs=new_inputs[1],
                temperature=Const(value=0.001),  # Very low temp = almost hard
                _props=node.props.with_grad(False)
            )
            return Harden(operand=soft_gate, _props=SIRProperty(dtype=BoolType()))
        else:
            # Keep soft gate
            return GateOp(
                compare=node.compare,
                lhs=new_inputs[0],
                rhs=new_inputs[1],
                temperature=new_inputs[2],
                _props=node.props
            )
    
    def _specialize_mix(self, node: MixOp, new_inputs: list[SIRNode]) -> SIRNode:
        """Specialize a mix operation."""
        if self.mode == SpecializationMode.INFER:
            # In infer mode with hardened gates, mix becomes select
            # The gate should already be hardened
            return MixOp(
                gate=new_inputs[0],
                then_value=new_inputs[1],
                else_value=new_inputs[2],
                _props=node.props.with_grad(False)
            )
        else:
            return MixOp(
                gate=new_inputs[0],
                then_value=new_inputs[1],
                else_value=new_inputs[2],
                _props=node.props
            )
    
    def _specialize_random(self, node: RandomVar, new_inputs: list[SIRNode]) -> Optional[SIRNode]:
        """Specialize a random variable."""
        if self.mode == SpecializationMode.INFER:
            # In infer mode, use mean of distribution instead of sampling
            # For Normal(mu, sigma), return mu
            if node.distribution == "Normal" and len(new_inputs) >= 1:
                return new_inputs[0]  # Return mean
            elif node.distribution == "Bernoulli" and len(new_inputs) >= 1:
                return new_inputs[0]  # Return probability
            else:
                # Default: return first param (usually mean/mode)
                return new_inputs[0] if new_inputs else Const(value=0.0)
        else:
            # Keep random sampling
            return RandomVar(
                distribution=node.distribution,
                params=new_inputs,
                _props=node.props
            )
    
    def _specialize_mode_switch(self, node: ModeSwitch, new_inputs: list[SIRNode]) -> SIRNode:
        """Specialize a mode switch - select the appropriate branch."""
        if self.mode == SpecializationMode.TRAIN:
            return new_inputs[0]  # train_value
        elif self.mode == SpecializationMode.INFER:
            return new_inputs[1]  # infer_value
        else:  # ANALYZE
            return new_inputs[2] if len(new_inputs) > 2 else new_inputs[0]
    
    def _specialize_constraint(self, node: ConstraintOp, new_inputs: list[SIRNode]) -> Optional[ConstraintOp]:
        """Specialize a constraint."""
        if self.mode == SpecializationMode.INFER and self.config.remove_constraints_in_infer:
            return None
        
        # Reconstruct constraint with new inputs
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


def specialize_mode(
    module: SIRModule,
    mode: str,
    config: Optional[SpecializationConfig] = None
) -> SIRModule:
    """Convenience function to specialize for a mode."""
    spec_mode = SpecializationMode(mode)
    specializer = ModeSpecializer(spec_mode, config)
    return specializer.specialize(module)
