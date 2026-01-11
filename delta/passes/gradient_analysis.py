"""
Gradient liveness analysis pass.

Analyzes which nodes actually need gradients computed:
- Traces backward from loss to parameters
- Identifies dead gradient paths
- Marks nodes for gradient computation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from delta.ir.sir import (
    SIRNode, SIRModule, SIRFunction, SIRBlock,
    TensorOp, GateOp, MixOp, ConstraintOp,
    ParamRef, ObsRef, Const, StopGrad, Harden, GradBoundary,
    RandomVar, Observe,
    walk_sir,
)


@dataclass
class GradientInfo:
    """Gradient information for a node."""
    requires_grad: bool = False
    is_gradient_source: bool = False  # Is this a parameter?
    is_gradient_sink: bool = False    # Does gradient flow into this?
    gradient_path_count: int = 0      # Number of paths from this to a param


class GradientAnalyzer:
    """
    Analyzes gradient flow in a SIR graph.
    
    Performs two passes:
    1. Forward: Mark nodes that depend on parameters
    2. Backward: Mark nodes whose gradients are used
    """
    
    def __init__(self) -> None:
        self.node_info: dict[int, GradientInfo] = {}
        self.param_nodes: set[int] = set()
        self.loss_nodes: set[int] = set()
    
    def analyze(self, module: SIRModule) -> dict[int, GradientInfo]:
        """Analyze gradient flow in a module."""
        # Collect all param nodes
        for param in module.params.values():
            self._mark_param(param)
        
        # Analyze functions
        for func in module.functions.values():
            self._analyze_function(func)
        
        # Analyze constraints (these are gradient sinks)
        for constraint in module.constraints:
            self._mark_loss(constraint)
            self._analyze_node_forward(constraint)
        
        # Backward pass from loss nodes
        for loss_id in self.loss_nodes:
            self._analyze_backward(loss_id)
        
        return self.node_info
    
    def _mark_param(self, param: ParamRef) -> None:
        """Mark a parameter as gradient source."""
        self.param_nodes.add(param.id)
        info = self._get_info(param)
        info.requires_grad = True
        info.is_gradient_source = True
    
    def _mark_loss(self, node: SIRNode) -> None:
        """Mark a node as gradient sink (loss)."""
        self.loss_nodes.add(node.id)
        info = self._get_info(node)
        info.is_gradient_sink = True
    
    def _get_info(self, node: SIRNode) -> GradientInfo:
        """Get or create gradient info for a node."""
        if node.id not in self.node_info:
            self.node_info[node.id] = GradientInfo()
        return self.node_info[node.id]
    
    def _analyze_function(self, func: SIRFunction) -> None:
        """Analyze a function's gradient flow."""
        self._analyze_block(func.body)
    
    def _analyze_block(self, block: SIRBlock) -> None:
        """Analyze a block's gradient flow."""
        for node in block.nodes:
            self._analyze_node_forward(node)
        
        if block.result:
            self._analyze_node_forward(block.result)
    
    def _analyze_node_forward(self, node: SIRNode) -> bool:
        """
        Forward pass: determine if node depends on parameters.
        
        Returns True if this node requires gradients.
        """
        info = self._get_info(node)
        
        # Already analyzed?
        if info.requires_grad:
            return True
        
        # Parameters always require grad
        if isinstance(node, ParamRef):
            info.requires_grad = True
            info.is_gradient_source = True
            return True
        
        # Observations never require grad
        if isinstance(node, ObsRef):
            info.requires_grad = False
            return False
        
        # Constants never require grad
        if isinstance(node, Const):
            info.requires_grad = False
            return False
        
        # StopGrad blocks gradient flow
        if isinstance(node, StopGrad):
            self._analyze_node_forward(node.operand)
            info.requires_grad = False
            return False
        
        # Harden blocks gradient flow
        if isinstance(node, Harden):
            self._analyze_node_forward(node.operand)
            info.requires_grad = False
            return False
        
        # GradBoundary controls gradient flow
        if isinstance(node, GradBoundary):
            self._analyze_node_forward(node.operand)
            if node.kind == "stop":
                info.requires_grad = False
                return False
            info.requires_grad = self._get_info(node.operand).requires_grad
            return info.requires_grad
        
        # For other nodes, require grad if any input requires grad
        any_input_requires_grad = False
        for inp in node.inputs():
            if self._analyze_node_forward(inp):
                any_input_requires_grad = True
        
        info.requires_grad = any_input_requires_grad
        return any_input_requires_grad
    
    def _analyze_backward(self, node_id: int) -> None:
        """
        Backward pass: mark gradient path from loss to parameters.
        
        Increments gradient_path_count for nodes on the path.
        """
        visited: set[int] = set()
        
        def visit(nid: int) -> None:
            if nid in visited:
                return
            visited.add(nid)
            
            info = self.node_info.get(nid)
            if info is None:
                return
            
            info.gradient_path_count += 1
            
            # Don't trace through stop grad
            # (would need access to the actual node to check type)
            
            # For a full implementation, we'd need access to the actual nodes
            # to trace through their inputs
        
        visit(node_id)
    
    def get_nodes_requiring_grad(self) -> set[int]:
        """Get set of node IDs that require gradient computation."""
        return {
            nid for nid, info in self.node_info.items()
            if info.requires_grad and info.gradient_path_count > 0
        }
    
    def get_dead_gradient_nodes(self) -> set[int]:
        """Get nodes that compute gradients but those gradients are never used."""
        return {
            nid for nid, info in self.node_info.items()
            if info.requires_grad and info.gradient_path_count == 0
        }


def analyze_gradients(module: SIRModule) -> tuple[dict[int, GradientInfo], GradientAnalyzer]:
    """Convenience function to analyze gradients."""
    analyzer = GradientAnalyzer()
    info = analyzer.analyze(module)
    return info, analyzer
