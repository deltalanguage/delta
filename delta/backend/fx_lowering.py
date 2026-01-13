"""
PyTorch FX lowering for Delta SIR.

Lowers Semantic IR to PyTorch FX graphs that can be executed
or further optimized. Key mappings:

| SIR Node    | FX Representation          |
|-------------|----------------------------|
| TensorOp    | torch.op                   |
| Mix         | g*a + (1-g)*b              |
| Gate        | torch.sigmoid((a-b)/temp)  |
| Constraint  | penalty tensor             |
| ParamRef    | torch.nn.Parameter         |
| StopGrad    | tensor.detach()            |
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
import torch
import torch.fx as fx
from torch.fx import Graph, GraphModule, Node

from delta.ir.sir import (
    SIRNode, SIRModule, SIRFunction, SIRBlock, SIRProperty,
    TensorOp, TensorOpKind, GateOp, MixOp, ConstraintOp, ConstraintKind,
    ParamRef, ObsRef, Const, StopGrad, Harden, GradBoundary,
    RandomVar, Observe, ModeSwitch, Layer, TupleIndex,
    walk_sir,
)
import operator
from delta.types.types import FloatType, IntType, BoolType


# Identity function for passthrough
def _identity(x):
    return x


# Mapping from SIR tensor ops to PyTorch ops
TENSOR_OP_MAP: dict[TensorOpKind, tuple[str, Callable[..., Any]]] = {
    # Identity (passthrough)
    TensorOpKind.IDENTITY: ("call_function", _identity),
    
    # Binary ops
    TensorOpKind.ADD: ("call_function", torch.add),
    TensorOpKind.SUB: ("call_function", torch.sub),
    TensorOpKind.MUL: ("call_function", torch.mul),
    TensorOpKind.DIV: ("call_function", torch.div),
    TensorOpKind.POW: ("call_function", torch.pow),
    TensorOpKind.MATMUL: ("call_function", torch.matmul),
    
    # Unary ops
    TensorOpKind.NEG: ("call_function", torch.neg),
    TensorOpKind.EXP: ("call_function", torch.exp),
    TensorOpKind.LOG: ("call_function", torch.log),
    TensorOpKind.SIN: ("call_function", torch.sin),
    TensorOpKind.COS: ("call_function", torch.cos),
    TensorOpKind.TANH: ("call_function", torch.tanh),
    TensorOpKind.SIGMOID: ("call_function", torch.sigmoid),
    TensorOpKind.RELU: ("call_function", torch.relu),
    # TensorOpKind.SOFTMAX handled specially below
    TensorOpKind.SQRT: ("call_function", torch.sqrt),
    TensorOpKind.ABS: ("call_function", torch.abs),
    TensorOpKind.GELU: ("call_function", torch.nn.functional.gelu),
    
    # Reductions
    TensorOpKind.SUM: ("call_method", "sum"),
    TensorOpKind.MEAN: ("call_method", "mean"),
    TensorOpKind.MAX: ("call_method", "max"),
    TensorOpKind.MIN: ("call_method", "min"),
    TensorOpKind.PROD: ("call_method", "prod"),
    TensorOpKind.ARGMAX: ("call_method", "argmax"),
    TensorOpKind.ARGMIN: ("call_method", "argmin"),
    
    # Shape ops
    # TensorOpKind.RESHAPE handled specially below
    # TensorOpKind.TRANSPOSE handled specially below
    TensorOpKind.SQUEEZE: ("call_method", "squeeze"),
    TensorOpKind.UNSQUEEZE: ("call_method", "unsqueeze"),
    TensorOpKind.FLATTEN: ("call_method", "flatten"),
    
    # Comparisons
    TensorOpKind.LT: ("call_function", torch.lt),
    TensorOpKind.LE: ("call_function", torch.le),
    TensorOpKind.GT: ("call_function", torch.gt),
    TensorOpKind.GE: ("call_function", torch.ge),
    TensorOpKind.EQ: ("call_function", torch.eq),
    TensorOpKind.NE: ("call_function", torch.ne),
}


@dataclass
class FXContext:
    """Context for FX lowering."""
    graph: Graph
    node_map: dict[int, Node] = field(default_factory=dict)
    param_nodes: dict[str, Node] = field(default_factory=dict)
    input_nodes: dict[str, Node] = field(default_factory=dict)


class FXLowering:
    """
    Lowers Delta SIR to PyTorch FX graphs.
    
    Produces executable FX GraphModules that can be:
    - Directly executed
    - Further optimized by torch.compile
    - Exported to other formats
    """
    
    def __init__(self) -> None:
        self.contexts: dict[str, FXContext] = {}
        self.layers: dict[str, torch.nn.Module] = {} # layer_id -> module
    
    def lower(self, module: SIRModule) -> dict[str, GraphModule]:
        """Lower a SIR module to FX GraphModules."""
        result: dict[str, GraphModule] = {}
        
        for name, func in module.functions.items():
            graph_module = self._lower_function(func, module)
            result[name] = graph_module
        
        return result
    
    def lower_function(self, func: SIRFunction, module: SIRModule) -> GraphModule:
        """Lower a single function to an FX GraphModule."""
        return self._lower_function(func, module)
    
    def _lower_function(self, func: SIRFunction, module: SIRModule) -> GraphModule:
        """Internal function lowering."""
        graph = Graph()
        ctx = FXContext(graph=graph)
        self.contexts[func.name] = ctx
        
        # Create input placeholders for function parameters
        for param_name, param_props in func.params:
            placeholder = graph.placeholder(param_name)
            ctx.input_nodes[param_name] = placeholder
            ctx.node_map[id(param_name)] = placeholder
        
        # Create nodes for module params
        for param_name, param_ref in module.params.items():
            # Module params are accessed via get_attr
            param_node = graph.get_attr(f"params.{param_name}")
            ctx.param_nodes[param_name] = param_node
        
        # Lower the function body
        result_node = self._lower_block(func.body, ctx)
        
        # Add output node
        if result_node:
            graph.output(result_node)
        else:
            # Return None if no result
            none_node = graph.call_function(lambda: None, ())
            graph.output(none_node)
        
        # Create the GraphModule
        # Note: In real usage, we'd pass a proper root module with parameters
        class DeltaModule(torch.nn.Module):
            def __init__(self, params: dict[str, torch.nn.Parameter], layers: dict[str, torch.nn.Module]) -> None:
                super().__init__()
                self.params = torch.nn.ParameterDict(params)
                for name, layer in layers.items():
                    self.add_module(name, layer)
        
        # Create empty params for now - real params come from runtime
        dummy_params = {
            name: torch.nn.Parameter(torch.zeros(1))
            for name in module.params
        }
        root = DeltaModule(dummy_params, self.layers)
        
        return GraphModule(root, graph)
    
    def _lower_block(self, block: SIRBlock, ctx: FXContext) -> Optional[Node]:
        """Lower a SIR block to FX nodes."""
        for node in block.nodes:
            self._lower_node(node, ctx)
        
        if block.result:
            return self._lower_node(block.result, ctx)
        return None
    
    def _lower_node(self, node: SIRNode, ctx: FXContext) -> Node:
        """Lower a SIR node to an FX node."""
        # Check if already lowered
        if node.id in ctx.node_map:
            return ctx.node_map[node.id]
        
        # Lower based on node type
        fx_node: Node
        
        if isinstance(node, TensorOp):
            fx_node = self._lower_tensor_op(node, ctx)
        elif isinstance(node, GateOp):
            fx_node = self._lower_gate_op(node, ctx)
        elif isinstance(node, MixOp):
            fx_node = self._lower_mix_op(node, ctx)
        elif isinstance(node, ConstraintOp):
            fx_node = self._lower_constraint_op(node, ctx)
        elif isinstance(node, ParamRef):
            fx_node = self._lower_param_ref(node, ctx)
        elif isinstance(node, ObsRef):
            fx_node = self._lower_obs_ref(node, ctx)
        elif isinstance(node, Const):
            fx_node = self._lower_const(node, ctx)
        elif isinstance(node, StopGrad):
            fx_node = self._lower_stop_grad(node, ctx)
        elif isinstance(node, Harden):
            fx_node = self._lower_harden(node, ctx)
        elif isinstance(node, RandomVar):
            fx_node = self._lower_random_var(node, ctx)
        elif isinstance(node, Observe):
            fx_node = self._lower_observe(node, ctx)
        elif isinstance(node, ModeSwitch):
            # Mode switch should be resolved by specialization
            fx_node = self._lower_node(node.train_value, ctx)
        elif isinstance(node, Layer):
            fx_node = self._lower_layer(node, ctx)
        elif isinstance(node, TupleIndex):
            fx_node = self._lower_tuple_index(node, ctx)
        else:
            # Unknown node type - create a placeholder
            fx_node = ctx.graph.call_function(lambda: None, ())
        
        ctx.node_map[node.id] = fx_node
        return fx_node
    
    def _lower_tensor_op(self, node: TensorOp, ctx: FXContext) -> Node:
        """Lower a tensor operation."""
        # Handle CALL_LAYER
        if node.op == TensorOpKind.CALL_LAYER:
            layer_node = node.operands[0]
            if not isinstance(layer_node, Layer):
                return ctx.graph.call_function(lambda: None, ())
            
            layer_name = f"layer_{layer_node.id}"
            args = [self._lower_node(op, ctx) for op in node.operands[1:]]
            return ctx.graph.call_module(layer_name, tuple(args))

        # Handle IDENTITY with no operands as an input placeholder
        if node.op == TensorOpKind.IDENTITY and not node.operands:
            # ... (rest of identity handling)
            name = getattr(node, '_name', None)
            if name:
                if name in ctx.input_nodes:
                    return ctx.input_nodes[name]
                if name in ctx.node_map:
                    return ctx.node_map[name]
                placeholder = ctx.graph.placeholder(name)
                ctx.node_map[name] = placeholder
                ctx.input_nodes[name] = placeholder
                return placeholder
            return ctx.graph.placeholder('input')
        
        # Special handling for constant args in shape operations
        if node.op in (TensorOpKind.RESHAPE, TensorOpKind.TRANSPOSE, TensorOpKind.SQUEEZE, TensorOpKind.UNSQUEEZE):
            operand_nodes = []
            for i, op in enumerate(node.operands):
                if i > 0 and isinstance(op, Const):
                    # For shape args, use the value directly if it's a Const
                    operand_nodes.append(op.value)
                else:
                    operand_nodes.append(self._lower_node(op, ctx))
        else:
            # Lower all operands normally
            operand_nodes = [self._lower_node(op, ctx) for op in node.operands]
        
        # Handle IDENTITY with operands (passthrough)
        if node.op == TensorOpKind.IDENTITY and operand_nodes:
            return operand_nodes[0]
        
        if node.op in (TensorOpKind.ARGMAX, TensorOpKind.ARGMIN):
            target = TENSOR_OP_MAP[node.op][1]
            args = [operand_nodes[0]]
            # If we have a dimension argument
            if len(operand_nodes) > 1:
                dim_node = node.operands[1]
                if isinstance(dim_node, Const) and isinstance(dim_node.value, int):
                    args.append(dim_node.value)
                else:
                    # Fallback to passing the node (might fail if torch expects int)
                    args.append(operand_nodes[1])
                # Add any remaining args
                args.extend(operand_nodes[2:])
            
            return ctx.graph.call_method(target, tuple(args))
            
        if node.op in TENSOR_OP_MAP:
            op_type, op_target = TENSOR_OP_MAP[node.op]
            
            # Special case: POW with one operand is a square operation
            if node.op == TensorOpKind.POW and len(operand_nodes) == 1:
                return ctx.graph.call_function(torch.pow, (operand_nodes[0], 2))
            
            if op_type == "call_function":
                return ctx.graph.call_function(op_target, tuple(operand_nodes))
            elif op_type == "call_method":
                if operand_nodes:
                    return ctx.graph.call_method(op_target, (operand_nodes[0],) + tuple(operand_nodes[1:]))
                else:
                    return ctx.graph.call_function(lambda: None, ())
        
        if node.op == TensorOpKind.ZEROS:
            return ctx.graph.call_function(torch.zeros, (tuple(operand_nodes),))
        
        if node.op == TensorOpKind.ONES:
            return ctx.graph.call_function(torch.ones, (tuple(operand_nodes),))
        
        if node.op == TensorOpKind.RAND:
            return ctx.graph.call_function(torch.rand, (tuple(operand_nodes),))
        
        if node.op == TensorOpKind.RANDN:
            return ctx.graph.call_function(torch.randn, (tuple(operand_nodes),))
        
        if node.op == TensorOpKind.FULL:
            # torch.full(size, fill_value)
            # size is all operands except the last one
            if len(operand_nodes) >= 2:
                return ctx.graph.call_function(torch.full, (tuple(operand_nodes[:-1]), operand_nodes[-1]))
            return ctx.graph.call_function(lambda: None, ())
        
        if node.op == TensorOpKind.EYE:
            # torch.eye(n)
            return ctx.graph.call_function(torch.eye, tuple(operand_nodes))
        
        # Handle special cases
        if node.op == TensorOpKind.SLICE:
            # Indexing operation
            if len(operand_nodes) >= 2:
                return ctx.graph.call_function(
                    lambda x, *idx: x[idx] if len(idx) > 1 else x[idx[0]],
                    tuple(operand_nodes)
                )
        
        if node.op == TensorOpKind.CONCAT:
            # Concatenation
            dim = node.attrs.get("dim", 0)
            return ctx.graph.call_function(
                torch.cat,
                (operand_nodes, dim)
            )
        
        if node.op == TensorOpKind.EMBEDDING:
            # torch.nn.functional.embedding(input, weight)
            return ctx.graph.call_function(
                torch.nn.functional.embedding,
                tuple(operand_nodes)
            )
        
        if node.op == TensorOpKind.STACK:
            # Stack tensors
            dim = node.attrs.get("dim", 0)
            return ctx.graph.call_function(
                torch.stack,
                (operand_nodes, dim)
            )
        
        if node.op == TensorOpKind.CROSS_ENTROPY:
            # torch.nn.functional.cross_entropy(input, target)
            return ctx.graph.call_function(
                torch.nn.functional.cross_entropy,
                tuple(operand_nodes)
            )
        
        if node.op == TensorOpKind.MSE_LOSS:
            # torch.nn.functional.mse_loss(input, target)
            return ctx.graph.call_function(
                torch.nn.functional.mse_loss,
                tuple(operand_nodes)
            )
        
        if node.op == TensorOpKind.CAUSAL_MASK:
            # Create causal attention mask
            def causal_mask(x):
                T = x.size(-1)
                mask = torch.triu(torch.ones(T, T, device=x.device, dtype=x.dtype), diagonal=1)
                return mask * float('-inf')
            return ctx.graph.call_function(causal_mask, tuple(operand_nodes))
        
        if node.op == TensorOpKind.SHAPE:
            # Get tensor shape
            if len(operand_nodes) > 1:
                return ctx.graph.call_method("size", tuple(operand_nodes))
            return ctx.graph.call_method("size", (operand_nodes[0],))
        
        if node.op == TensorOpKind.WHERE:
            # torch.where(condition, x, y)
            return ctx.graph.call_function(torch.where, tuple(operand_nodes))
        
        if node.op == TensorOpKind.TRANSPOSE:
            # Handle transpose - swap last two dimensions by default
            if len(operand_nodes) == 1:
                # Just tensor - transpose last two dims
                return ctx.graph.call_method("transpose", (operand_nodes[0], -2, -1))
            elif len(operand_nodes) >= 3:
                # tensor, dim0, dim1
                return ctx.graph.call_method("transpose", tuple(operand_nodes))
            else:
                # tensor, dim0 (assume dim1=-1)
                return ctx.graph.call_method("transpose", (operand_nodes[0], operand_nodes[1], -1))
        
        if node.op == TensorOpKind.SOFTMAX:
            # softmax needs a dim argument - default to last dim
            return ctx.graph.call_method("softmax", (operand_nodes[0], -1))
        
        if node.op == TensorOpKind.RESHAPE:
            # reshape(tensor, *shape)
            if len(operand_nodes) >= 2:
                return ctx.graph.call_method("reshape", tuple(operand_nodes))
            return operand_nodes[0]
        
        if node.op == TensorOpKind.TRANSPOSE:
            if len(operand_nodes) >= 3:
                return ctx.graph.call_method("transpose", tuple(operand_nodes))
            return operand_nodes[0]
        
        # Fallback: return first operand
        return operand_nodes[0] if operand_nodes else ctx.graph.call_function(lambda: None, ())
    
    def _lower_gate_op(self, node: GateOp, ctx: FXContext) -> Node:
        """
        Lower a gate operation.
        
        gate(a, b, temp) = sigmoid((a - b) / temp)
        
        For comparisons:
        - LT: sigmoid((b - a) / temp)  # a < b
        - GT: sigmoid((a - b) / temp)  # a > b
        - LE: sigmoid((b - a + eps) / temp)
        - GE: sigmoid((a - b + eps) / temp)
        """
        lhs = self._lower_node(node.lhs, ctx)
        rhs = self._lower_node(node.rhs, ctx)
        temp = self._lower_node(node.temperature, ctx)
        
        # Ensure float for subtraction (comparisons return bool)
        lhs_f = ctx.graph.call_method("float", (lhs,))
        rhs_f = ctx.graph.call_method("float", (rhs,))
        
        # Compute difference based on comparison type
        if node.compare in (TensorOpKind.LT, TensorOpKind.LE):
            # a < b: want high value when b > a, so (b - a)
            diff = ctx.graph.call_function(torch.sub, (rhs_f, lhs_f))
        else:
            # a > b or a >= b: want high value when a > b, so (a - b)
            diff = ctx.graph.call_function(torch.sub, (lhs_f, rhs_f))
        
        # Divide by temperature
        scaled = ctx.graph.call_function(torch.div, (diff, temp))
        
        # Apply sigmoid
        return ctx.graph.call_function(torch.sigmoid, (scaled,))
    
    def _lower_mix_op(self, node: MixOp, ctx: FXContext) -> Node:
        """
        Lower a mix operation.
        
        mix(gate, a, b) = gate * a + (1 - gate) * b
        """
        gate = self._lower_node(node.gate, ctx)
        then_val = self._lower_node(node.then_value, ctx)
        else_val = self._lower_node(node.else_value, ctx)
        
        # gate * then_value
        weighted_then = ctx.graph.call_function(torch.mul, (gate, then_val))
        
        # 1 - gate
        one = ctx.graph.call_function(torch.tensor, (1.0,))
        one_minus_gate = ctx.graph.call_function(torch.sub, (one, gate))
        
        # (1 - gate) * else_value
        weighted_else = ctx.graph.call_function(torch.mul, (one_minus_gate, else_val))
        
        # Sum
        return ctx.graph.call_function(torch.add, (weighted_then, weighted_else))
    
    def _lower_constraint_op(self, node: ConstraintOp, ctx: FXContext) -> Node:
        """Lower a constraint operation to a penalty term."""
        lhs = self._lower_node(node.lhs, ctx)
        weight = self._lower_node(node.weight, ctx)
        
        if node.kind == ConstraintKind.EQUALITY:
            # Squared penalty: weight * (lhs - rhs)^2
            if node.rhs:
                rhs = self._lower_node(node.rhs, ctx)
                diff = ctx.graph.call_function(torch.sub, (lhs, rhs))
            else:
                diff = lhs
            
            squared = ctx.graph.call_function(torch.pow, (diff, 2))
            mean_sq = ctx.graph.call_method("mean", (squared,))
            return ctx.graph.call_function(torch.mul, (weight, mean_sq))
        
        elif node.kind == ConstraintKind.INEQUALITY:
            # Softplus penalty: weight * softplus(lhs - rhs)
            if node.rhs:
                rhs = self._lower_node(node.rhs, ctx)
                diff = ctx.graph.call_function(torch.sub, (lhs, rhs))
            else:
                diff = lhs
            
            softplus = ctx.graph.call_function(torch.nn.functional.softplus, (diff,))
            mean_sp = ctx.graph.call_method("mean", (softplus,))
            return ctx.graph.call_function(torch.mul, (weight, mean_sp))
        
        elif node.kind == ConstraintKind.BOOLEAN:
            # Boolean penalty: weight * (1 - lhs)
            one = ctx.graph.call_function(torch.tensor, (1.0,))
            violation = ctx.graph.call_function(torch.sub, (one, lhs))
            mean_v = ctx.graph.call_method("mean", (violation,))
            return ctx.graph.call_function(torch.mul, (weight, mean_v))
        
        else:
            # Likelihood: weight * (-lhs)  # lhs is log prob
            neg_one = ctx.graph.call_function(torch.tensor, (-1.0,))
            neg_log_prob = ctx.graph.call_function(torch.mul, (neg_one, lhs))
            mean_nlp = ctx.graph.call_method("mean", (neg_log_prob,))
            return ctx.graph.call_function(torch.mul, (weight, mean_nlp))
    
    def _lower_layer(self, node: Layer, ctx: FXContext) -> Node:
        """Lower a layer instantiation."""
        layer_id = f"layer_{node.id}"
        if layer_id not in self.layers:
            # Instantiate the actual torch module
            if node.kind == 'Linear':
                self.layers[layer_id] = torch.nn.Linear(*node.args, **node.kwargs)
            elif node.kind == 'Conv1d':
                self.layers[layer_id] = torch.nn.Conv1d(*node.args, **node.kwargs)
            elif node.kind == 'Conv2d':
                self.layers[layer_id] = torch.nn.Conv2d(*node.args, **node.kwargs)
            elif node.kind == 'Embedding':
                self.layers[layer_id] = torch.nn.Embedding(*node.args, **node.kwargs)
            elif node.kind == 'LayerNorm':
                self.layers[layer_id] = torch.nn.LayerNorm(*node.args, **node.kwargs)
            elif node.kind == 'Dropout':
                self.layers[layer_id] = torch.nn.Dropout(*node.args, **node.kwargs)
            elif node.kind == 'LSTM':
                self.layers[layer_id] = torch.nn.LSTM(*node.args, **node.kwargs)
            else:
                # Generic fallback if we don't know the exact class
                # This might fail if the kind is not in torch.nn
                cls = getattr(torch.nn, node.kind, None)
                if cls:
                    self.layers[layer_id] = cls(*node.args, **node.kwargs)
                else:
                    self.layers[layer_id] = torch.nn.Identity()
        
        # Layer nodes themselves don't produce an FX node in the graph,
        # they are just registered. CALL_LAYER uses them.
        return ctx.graph.call_function(lambda: None, ())
    
    def _lower_param_ref(self, node: ParamRef, ctx: FXContext) -> Node:
        """Lower a parameter reference."""
        if node.name in ctx.param_nodes:
            return ctx.param_nodes[node.name]
        if node.name in ctx.input_nodes:
            return ctx.input_nodes[node.name]
        
        # Create a get_attr node for the parameter
        param_node = ctx.graph.get_attr(f"params.{node.name}")
        ctx.param_nodes[node.name] = param_node
        return param_node
    
    def _lower_obs_ref(self, node: ObsRef, ctx: FXContext) -> Node:
        """Lower an observation reference."""
        if node.name in ctx.input_nodes:
            return ctx.input_nodes[node.name]
        
        # Observations are typically function inputs
        # Create a placeholder if not already present
        placeholder = ctx.graph.placeholder(node.name)
        ctx.input_nodes[node.name] = placeholder
        return placeholder
    
    def _lower_const(self, node: Const, ctx: FXContext) -> Node:
        """Lower a constant."""
        value = node.value
        
        if value is None:
            return ctx.graph.call_function(lambda: None, ())
        
        if isinstance(value, (int, float)):
            return ctx.graph.call_function(torch.tensor, (value,))
        
        if isinstance(value, bool):
            return ctx.graph.call_function(torch.tensor, (value,))
        
        if isinstance(value, list):
            return ctx.graph.call_function(torch.tensor, (value,))
        
        # For other types, try to create a tensor
        try:
            return ctx.graph.call_function(torch.tensor, (value,))
        except (TypeError, ValueError):
            return ctx.graph.call_function(lambda: value, ())
    
    def _lower_stop_grad(self, node: StopGrad, ctx: FXContext) -> Node:
        """Lower a stop gradient operation."""
        operand = self._lower_node(node.operand, ctx)
        return ctx.graph.call_method("detach", (operand,))
    
    def _lower_harden(self, node: Harden, ctx: FXContext) -> Node:
        """Lower a harden operation (threshold to boolean)."""
        operand = self._lower_node(node.operand, ctx)
        threshold = ctx.graph.call_function(torch.tensor, (node.threshold,))
        bool_result = ctx.graph.call_function(torch.gt, (operand, threshold))
        return ctx.graph.call_method("float", (bool_result,))
    
    def _lower_tuple_index(self, node: TupleIndex, ctx: FXContext) -> Node:
        """Lower a tuple index operation."""
        operand = self._lower_node(node.operand, ctx)
        return ctx.graph.call_function(operator.getitem, (operand, node.index))

    def _get_dist_param(self, node: SIRNode, lowered_params: list[Node], lowered_kwargs: dict[str, Node], name: str, index: int) -> Optional[Node]:
        """Helper to get a distribution parameter by name or index."""
        if name in lowered_kwargs:
            return lowered_kwargs[name]
        if index < len(lowered_params):
            return lowered_params[index]
        return None

    def _lower_random_var(self, node: RandomVar, ctx: FXContext) -> Node:
        """Lower a random variable (sampling)."""
        params = [self._lower_node(p, ctx) for p in node.params]
        kwargs = {k: self._lower_node(v, ctx) for k, v in node.kwargs.items()}
        
        if node.distribution == "Normal":
            # Normal distribution with reparameterization
            mean = self._get_dist_param(node, params, kwargs, "loc", 0)
            std = self._get_dist_param(node, params, kwargs, "scale", 1)
            
            if mean is not None:
                if std is not None:
                    # Sample: mean + std * eps, where eps ~ N(0, 1)
                    eps = ctx.graph.call_function(torch.randn_like, (mean,))
                    scaled_eps = ctx.graph.call_function(torch.mul, (std, eps))
                    return ctx.graph.call_function(torch.add, (mean, scaled_eps))
                else:
                    eps = ctx.graph.call_function(torch.randn_like, (mean,))
                    return ctx.graph.call_function(torch.add, (mean, eps))
        
        elif node.distribution == "Bernoulli":
            probs = self._get_dist_param(node, params, kwargs, "probs", 0)
            logits = self._get_dist_param(node, params, kwargs, "logits", 0) # Overlaps with probs but OK
            
            if probs is not None:
                dist = ctx.graph.call_function(torch.distributions.Bernoulli, (), {'probs': probs})
                return ctx.graph.call_method("rsample", (dist,))
            elif logits is not None:
                dist = ctx.graph.call_function(torch.distributions.Bernoulli, (), {'logits': logits})
                return ctx.graph.call_method("rsample", (dist,))
        
        # Default: return first param as mean or 0.0
        return params[0] if params else ctx.graph.call_function(torch.tensor, (0.0,))
    
    def _lower_observe(self, node: Observe, ctx: FXContext) -> Node:
        """Lower an observe operation (log probability)."""
        value = self._lower_node(node.value, ctx)
        params = [self._lower_node(p, ctx) for p in node.params]
        kwargs = {k: self._lower_node(v, ctx) for k, v in node.kwargs.items()}
        
        if node.distribution == "Normal":
            mean = self._get_dist_param(node, params, kwargs, "loc", 0)
            std = self._get_dist_param(node, params, kwargs, "scale", 1)
            
            if mean is not None and std is not None:
                # Log probability of Normal distribution
                # log_prob = -0.5 * ((x - mean) / std)^2 - log(std) - 0.5 * log(2π)
                diff = ctx.graph.call_function(torch.sub, (value, mean))
                normalized = ctx.graph.call_function(torch.div, (diff, std))
                sq = ctx.graph.call_function(torch.pow, (normalized, 2))
                neg_half = ctx.graph.call_function(torch.tensor, (-0.5,))
                term1 = ctx.graph.call_function(torch.mul, (neg_half, sq))
                log_std = ctx.graph.call_function(torch.log, (std,))
                term2 = ctx.graph.call_function(torch.neg, (log_std,))
                return ctx.graph.call_function(torch.add, (term1, term2))
        
        elif node.distribution == "Bernoulli":
            probs = self._get_dist_param(node, params, kwargs, "probs", 0)
            if probs is not None:
                dist = ctx.graph.call_function(torch.distributions.Bernoulli, (), {'probs': probs})
                return ctx.graph.call_method("log_prob", (dist, value))
        
        # Default: return 0 (log prob of 1)
        return ctx.graph.call_function(torch.tensor, (0.0,))


def lower_to_fx(module: SIRModule) -> dict[str, GraphModule]:
    """Convenience function to lower SIR to FX."""
    lowering = FXLowering()
    return lowering.lower(module)
