"""
Dead gradient elimination pass.

Removes unnecessary gradient computations:
- Gradients that don't flow to any loss
- Gradients for nodes that don't affect parameters
- Redundant gradient checkpoints
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from delta.ir.sir import (
    SIRNode, SIRModule, SIRFunction, SIRBlock, SIRProperty,
    TensorOp, GateOp, MixOp, ConstraintOp,
    ParamRef, ObsRef, Const, StopGrad, Harden, GradBoundary,
    RandomVar, Observe,
    walk_sir,
)
from delta.passes.gradient_analysis import GradientAnalyzer, GradientInfo


@dataclass
class EliminationStats:
    """Statistics from dead gradient elimination."""
    nodes_processed: int = 0
    gradients_eliminated: int = 0
    stop_grads_inserted: int = 0


class DeadGradientEliminator:
    """
    Eliminates dead gradient computations.
    
    A gradient computation is dead if:
    1. It doesn't depend on any parameter, OR
    2. It doesn't flow into any loss/constraint
    
    Dead gradients are eliminated by inserting StopGrad nodes.
    """
    
    def __init__(self, gradient_info: dict[int, GradientInfo]) -> None:
        self.gradient_info = gradient_info
        self.node_map: dict[int, SIRNode] = {}
        self.stats = EliminationStats()
    
    def eliminate(self, module: SIRModule) -> SIRModule:
        """Eliminate dead gradients from a module."""
        new_module = SIRModule(name=module.name)
        
        # Copy params (they always need gradients)
        for name, param in module.params.items():
            new_module.params[name] = param
        
        # Copy constants
        for name, const in module.constants.items():
            new_module.constants[name] = const
        
        # Process functions
        for name, func in module.functions.items():
            new_func = self._process_function(func)
            new_module.functions[name] = new_func
        
        # Process constraints
        for constraint in module.constraints:
            new_constraint = self._process_node(constraint)
            if isinstance(new_constraint, ConstraintOp):
                new_module.constraints.append(new_constraint)
        
        return new_module
    
    def _process_function(self, func: SIRFunction) -> SIRFunction:
        """Process a function."""
        new_body = self._process_block(func.body)
        
        return SIRFunction(
            name=func.name,
            params=func.params,
            body=new_body,
            return_type=func.return_type,
            effects=func.effects
        )
    
    def _process_block(self, block: SIRBlock) -> SIRBlock:
        """Process a block."""
        new_nodes = [self._process_node(node) for node in block.nodes]
        new_result = self._process_node(block.result) if block.result else None
        
        return SIRBlock(nodes=new_nodes, result=new_result)
    
    def _process_node(self, node: SIRNode) -> SIRNode:
        """Process a node, eliminating dead gradients."""
        if node.id in self.node_map:
            return self.node_map[node.id]
        
        self.stats.nodes_processed += 1
        
        # First process all inputs
        new_inputs = [self._process_node(inp) for inp in node.inputs()]
        
        # Check if this node has dead gradients
        info = self.gradient_info.get(node.id, GradientInfo())
        
        if self._should_eliminate_gradient(node, info):
            # Insert StopGrad to eliminate gradient
            self.stats.gradients_eliminated += 1
            
            # Clone with new inputs first
            if new_inputs != node.inputs():
                new_node = node.clone_with_inputs(new_inputs)
            else:
                new_node = node
            
            # Wrap in StopGrad
            self.stats.stop_grads_inserted += 1
            result = StopGrad(
                operand=new_node,
                _props=new_node.props.with_grad(False)
            )
        else:
            # Just clone with new inputs
            if new_inputs != node.inputs():
                result = node.clone_with_inputs(new_inputs)
            else:
                result = node
        
        self.node_map[node.id] = result
        return result
    
    def _should_eliminate_gradient(self, node: SIRNode, info: GradientInfo) -> bool:
        """Determine if gradient for this node should be eliminated."""
        # Don't eliminate gradients for:
        # - Parameters (they are gradient sources)
        # - Nodes that are already stop_grad
        # - Constants (no gradients anyway)
        
        if isinstance(node, (ParamRef, Const, StopGrad, Harden)):
            return False
        
        if isinstance(node, ObsRef):
            return False  # Already no gradient
        
        # Eliminate if:
        # 1. Node requires grad but gradient is never used (dead)
        # 2. Node is marked requires_grad but has no path to loss
        
        if info.requires_grad and info.gradient_path_count == 0:
            return True
        
        return False


def eliminate_dead_gradients(
    module: SIRModule,
    gradient_info: dict[int, GradientInfo]
) -> tuple[SIRModule, EliminationStats]:
    """Convenience function to eliminate dead gradients."""
    eliminator = DeadGradientEliminator(gradient_info)
    new_module = eliminator.eliminate(module)
    return new_module, eliminator.stats
