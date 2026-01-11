"""
FX graph optimization pass.

Optimizes PyTorch FX graphs for better performance:
- Operator fusion
- Dead code elimination
- Constant folding
- Common subexpression elimination
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Set
import torch
import torch.fx as fx
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.shape_prop import ShapeProp


@dataclass
class OptimizationStats:
    """Statistics from FX optimization."""
    nodes_before: int = 0
    nodes_after: int = 0
    constants_folded: int = 0
    dead_nodes_removed: int = 0
    fusions_performed: int = 0


class FXOptimizer:
    """
    Optimizes PyTorch FX graphs.
    
    Optimization passes:
    1. Dead code elimination
    2. Constant folding
    3. Common subexpression elimination
    4. Operator fusion
    """
    
    def __init__(self) -> None:
        self.stats = OptimizationStats()
    
    def optimize(self, gm: GraphModule) -> GraphModule:
        """Optimize a GraphModule."""
        self.stats.nodes_before = len(list(gm.graph.nodes))
        
        # Run optimization passes
        gm = self._eliminate_dead_code(gm)
        gm = self._fold_constants(gm)
        gm = self._eliminate_common_subexpressions(gm)
        gm = self._fuse_operators(gm)
        
        # Recompile
        gm.recompile()
        
        self.stats.nodes_after = len(list(gm.graph.nodes))
        return gm
    
    def _eliminate_dead_code(self, gm: GraphModule) -> GraphModule:
        """Eliminate dead code (nodes with no users)."""
        graph = gm.graph
        
        # Find nodes that are used
        used: Set[Node] = set()
        
        # Start from output and trace backwards
        for node in graph.nodes:
            if node.op == "output":
                used.add(node)
                self._mark_used(node, used)
        
        # Remove unused nodes
        nodes_to_remove = []
        for node in graph.nodes:
            if node not in used and node.op not in ("placeholder", "output"):
                nodes_to_remove.append(node)
        
        for node in reversed(nodes_to_remove):
            graph.erase_node(node)
            self.stats.dead_nodes_removed += 1
        
        return gm
    
    def _mark_used(self, node: Node, used: Set[Node]) -> None:
        """Recursively mark nodes as used."""
        for inp in node.all_input_nodes:
            if inp not in used:
                used.add(inp)
                self._mark_used(inp, used)
    
    def _fold_constants(self, gm: GraphModule) -> GraphModule:
        """Fold constant expressions."""
        graph = gm.graph
        
        # Find nodes that only depend on constants
        foldable: list[Node] = []
        
        for node in graph.nodes:
            if node.op == "call_function" and self._is_foldable(node, gm):
                foldable.append(node)
        
        # Fold each foldable node
        for node in foldable:
            try:
                # Evaluate the constant
                result = self._evaluate_constant(node, gm)
                if result is not None:
                    # Replace with constant
                    with graph.inserting_after(node):
                        const_node = graph.call_function(torch.tensor, (result.item(),))
                    node.replace_all_uses_with(const_node)
                    graph.erase_node(node)
                    self.stats.constants_folded += 1
            except (RuntimeError, TypeError, ValueError):
                # Can't fold this one
                pass
        
        return gm
    
    def _is_foldable(self, node: Node, gm: GraphModule) -> bool:
        """Check if a node can be constant-folded."""
        # A node is foldable if all its inputs are constants or foldable
        for inp in node.all_input_nodes:
            if inp.op == "placeholder":
                return False
            if inp.op == "get_attr":
                # Check if it's a constant buffer
                continue
            if inp.op == "call_function":
                # Check if it's a tensor constructor with constants
                if inp.target == torch.tensor:
                    continue
                return False
        return True
    
    def _evaluate_constant(self, node: Node, gm: GraphModule) -> Optional[torch.Tensor]:
        """Evaluate a constant expression."""
        # Get input values
        args = []
        for inp in node.args:
            if isinstance(inp, Node):
                if inp.op == "call_function" and inp.target == torch.tensor:
                    args.append(torch.tensor(inp.args[0]))
                else:
                    return None
            else:
                args.append(inp)
        
        # Evaluate
        if callable(node.target):
            try:
                return node.target(*args)
            except (RuntimeError, TypeError):
                return None
        return None
    
    def _eliminate_common_subexpressions(self, gm: GraphModule) -> GraphModule:
        """Eliminate common subexpressions."""
        graph = gm.graph
        
        # Map from (op, target, args_hash) -> node
        seen: dict[tuple, Node] = {}
        
        for node in list(graph.nodes):
            if node.op in ("call_function", "call_method"):
                # Create a hashable key for this operation
                args_key = self._hash_args(node.args)
                key = (node.op, node.target, args_key)
                
                if key in seen:
                    # Replace with existing node
                    node.replace_all_uses_with(seen[key])
                    graph.erase_node(node)
                else:
                    seen[key] = node
        
        return gm
    
    def _hash_args(self, args: tuple) -> tuple:
        """Create a hashable representation of arguments."""
        result = []
        for arg in args:
            if isinstance(arg, Node):
                result.append(("node", arg.name))
            elif isinstance(arg, (int, float, str, bool)):
                result.append(arg)
            elif isinstance(arg, tuple):
                result.append(self._hash_args(arg))
            else:
                result.append(str(type(arg)))
        return tuple(result)
    
    def _fuse_operators(self, gm: GraphModule) -> GraphModule:
        """Fuse operators for better performance."""
        graph = gm.graph
        
        # Pattern: mul + add -> fused multiply-add
        # Pattern: sub + pow(2) -> squared difference
        
        for node in list(graph.nodes):
            # Fuse patterns
            fused = self._try_fuse_matmul_add(node, graph)
            if fused:
                self.stats.fusions_performed += 1
                continue
            
            fused = self._try_fuse_activation(node, graph)
            if fused:
                self.stats.fusions_performed += 1
        
        return gm
    
    def _try_fuse_matmul_add(self, node: Node, graph: Graph) -> bool:
        """Try to fuse matmul + add into linear."""
        # Pattern: add(matmul(x, w), b) -> linear(x, w, b)
        if node.op != "call_function" or node.target != torch.add:
            return False
        
        if len(node.args) != 2:
            return False
        
        matmul_node = None
        bias_node = None
        
        for arg in node.args:
            if isinstance(arg, Node):
                if arg.op == "call_function" and arg.target == torch.matmul:
                    matmul_node = arg
                else:
                    bias_node = arg
        
        if matmul_node and bias_node:
            # Can fuse - but keep as matmul+add for now
            # Real fusion would require more analysis
            pass
        
        return False
    
    def _try_fuse_activation(self, node: Node, graph: Graph) -> bool:
        """Try to fuse operations followed by activation."""
        # Pattern: relu(add(x, y)) -> fused
        # For now, just identify the pattern
        return False


def optimize_fx(gm: GraphModule) -> tuple[GraphModule, OptimizationStats]:
    """Convenience function to optimize an FX graph."""
    optimizer = FXOptimizer()
    optimized = optimizer.optimize(gm)
    return optimized, optimizer.stats
