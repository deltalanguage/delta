"""
Semantic IR (SIR) node definitions for Delta.

SIR is where Delta's differentiable semantics are fully expressed.
Every node carries properties about types, shapes, roles, effects,
and gradient requirements.

Key SIR node types:
- TensorOp: Tensor math operations
- GateOp: Soft comparisons (sigmoid of comparison)
- MixOp: Soft conditionals (gate * a + (1-gate) * b)
- ConstraintOp: Constraint representations
- ParamRef/ObsRef: Role-annotated references
- StopGrad/Harden: Gradient control
- RandomVar/Observe: Probabilistic primitives
- ModeSwitch: Mode-specific behavior
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Sequence
from enum import Enum, auto
import itertools

from delta.types.types import Type, TensorType, FloatType, BoolType, ShapeDim
from delta.types.effects import EffectSet
from delta.types.roles import RoleInfo, RoleSet
from delta.source import SourceLocation


# Node ID counter
_node_id_counter = itertools.count()


def fresh_node_id() -> int:
    """Generate a fresh node ID."""
    return next(_node_id_counter)


class Mode(Enum):
    """Execution modes."""
    TRAIN = auto()
    INFER = auto()
    ANALYZE = auto()
    
    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class SIRProperty:
    """
    Properties attached to every SIR node.
    
    Captures all semantic information needed for:
    - Type checking
    - Gradient analysis
    - Mode specialization
    - Backend lowering
    """
    dtype: Type = field(default_factory=FloatType)
    shape: Optional[tuple[ShapeDim, ...]] = None
    role: RoleInfo = field(default_factory=lambda: RoleInfo.const())
    requires_grad: bool = False
    effects: EffectSet = field(default_factory=EffectSet.pure)
    mode_valid: frozenset[Mode] = field(default_factory=lambda: frozenset({Mode.TRAIN, Mode.INFER, Mode.ANALYZE}))
    cost_estimate: Optional[float] = None
    location: Optional[SourceLocation] = None
    
    def with_dtype(self, dtype: Type) -> SIRProperty:
        """Create a copy with different dtype."""
        return SIRProperty(
            dtype=dtype,
            shape=self.shape,
            role=self.role,
            requires_grad=self.requires_grad,
            effects=self.effects,
            mode_valid=self.mode_valid,
            cost_estimate=self.cost_estimate,
            location=self.location
        )
    
    def with_grad(self, requires_grad: bool) -> SIRProperty:
        """Create a copy with different gradient requirement."""
        return SIRProperty(
            dtype=self.dtype,
            shape=self.shape,
            role=self.role,
            requires_grad=requires_grad,
            effects=self.effects,
            mode_valid=self.mode_valid,
            cost_estimate=self.cost_estimate,
            location=self.location
        )


class SIRNode(ABC):
    """
    Base class for all SIR nodes.
    
    SIR nodes are immutable and form a directed acyclic graph.
    Each node has a unique ID and properties.
    """
    
    def __init__(self, props: Optional[SIRProperty] = None) -> None:
        self.id = fresh_node_id()
        self.props = props or SIRProperty()
    
    @abstractmethod
    def inputs(self) -> list[SIRNode]:
        """Get input nodes."""
        pass
    
    @abstractmethod
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> SIRNode:
        """Create a copy with different inputs."""
        pass
    
    @property
    def dtype(self) -> Type:
        return self.props.dtype
    
    @property
    def shape(self) -> Optional[tuple[ShapeDim, ...]]:
        return self.props.shape
    
    @property
    def requires_grad(self) -> bool:
        return self.props.requires_grad
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SIRNode):
            return False
        return self.id == other.id


@dataclass
class Layer(SIRNode):
    """
    A neural network layer or module.
    
    Layers are stateful but represented as callable nodes in SIR.
    """
    kind: str  # 'Linear', 'Conv2d', etc.
    args: list[Any]
    kwargs: dict[str, Any] = field(default_factory=dict)
    _props: SIRProperty = field(default_factory=SIRProperty)
    
    def __post_init__(self) -> None:
        self.id = fresh_node_id()
        self.props = self._props

    def inputs(self) -> list[SIRNode]:
        return []
    
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> Layer:
        return Layer(kind=self.kind, args=self.args, kwargs=self.kwargs, _props=self.props)


# Tensor Operations


class TensorOpKind(Enum):
    """Kinds of tensor operations."""    
    # Identity (passthrough)
    IDENTITY = auto()
    
    # Elementwise binary
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()
    
    # Matrix operations
    MATMUL = auto()
    
    # Elementwise unary
    NEG = auto()
    ZEROS = auto()
    ONES = auto()
    RAND = auto()
    RANDN = auto()
    FULL = auto()
    EYE = auto()
    EXP = auto()
    LOG = auto()
    SIN = auto()
    COS = auto()
    TANH = auto()
    SIGMOID = auto()
    RELU = auto()
    SOFTMAX = auto()
    SQRT = auto()
    ABS = auto()
    GELU = auto()
    
    # Reductions
    SUM = auto()
    MEAN = auto()
    MAX = auto()
    MIN = auto()
    PROD = auto()
    ARGMAX = auto()
    ARGMIN = auto()
    
    # Shape operations
    RESHAPE = auto()
    TRANSPOSE = auto()
    SQUEEZE = auto()
    UNSQUEEZE = auto()
    CONCAT = auto()
    STACK = auto()
    SLICE = auto()
    FLATTEN = auto()
    SHAPE = auto()
    
    # Neural network ops
    EMBEDDING = auto()
    CROSS_ENTROPY = auto()
    MSE_LOSS = auto()
    CAUSAL_MASK = auto()
    WHERE = auto()
    CALL_LAYER = auto()
    
    # Comparisons (hard - for non_diff)
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()


@dataclass
class TensorOp(SIRNode):
    """
    General tensor operation.
    
    Represents math operations on tensors that will be lowered
    to PyTorch operations.
    """
    op: TensorOpKind
    operands: list[SIRNode]
    attrs: dict[str, Any] = field(default_factory=dict)
    _props: SIRProperty = field(default_factory=SIRProperty)
    
    def __post_init__(self) -> None:
        self.id = fresh_node_id()
        self.props = self._props
    
    def inputs(self) -> list[SIRNode]:
        return self.operands
    
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> TensorOp:
        return TensorOp(
            op=self.op,
            operands=new_inputs,
            attrs=self.attrs,
            _props=self.props
        )
    
    def __str__(self) -> str:
        return f"{self.op.name}({', '.join(str(o.id) for o in self.operands)})"


def BinaryTensorOp(op: TensorOpKind, left: SIRNode, right: SIRNode, props: Optional[SIRProperty] = None) -> TensorOp:
    """Create a binary tensor operation."""
    if props is None:
        props = SIRProperty(
            dtype=left.dtype,
            shape=left.shape,
            requires_grad=left.requires_grad or right.requires_grad
        )
    return TensorOp(op=op, operands=[left, right], _props=props)


def UnaryTensorOp(op: TensorOpKind, operand: SIRNode, props: Optional[SIRProperty] = None) -> TensorOp:
    """Create a unary tensor operation."""
    if props is None:
        props = SIRProperty(
            dtype=operand.dtype,
            shape=operand.shape,
            requires_grad=operand.requires_grad
        )
    return TensorOp(op=op, operands=[operand], _props=props)


def ReduceOp(op: TensorOpKind, operand: SIRNode, axis: Optional[int] = None, props: Optional[SIRProperty] = None) -> TensorOp:
    """Create a reduction operation."""
    if props is None:
        props = SIRProperty(
            dtype=operand.dtype,
            requires_grad=operand.requires_grad
        )
    attrs = {"axis": axis} if axis is not None else {}
    return TensorOp(op=op, operands=[operand], attrs=attrs, _props=props)


def ShapeOp(op: TensorOpKind, operand: SIRNode, shape: Optional[tuple[int, ...]] = None, props: Optional[SIRProperty] = None) -> TensorOp:
    """Create a shape operation."""
    if props is None:
        props = SIRProperty(
            dtype=operand.dtype,
            requires_grad=operand.requires_grad
        )
    attrs = {"shape": shape} if shape is not None else {}
    return TensorOp(op=op, operands=[operand], attrs=attrs, _props=props)


@dataclass
class GateOp(SIRNode):
    """
    Soft comparison gate.
    
    gate(a, b, temp) = sigmoid((a - b) / temp)
    
    Used to make comparisons differentiable. At temp → 0,
    approaches a hard comparison.
    """
    compare: TensorOpKind  # LT, LE, GT, GE, EQ
    lhs: SIRNode
    rhs: SIRNode
    temperature: SIRNode  # Temperature for softening
    _props: SIRProperty = field(default_factory=SIRProperty)
    
    def __post_init__(self) -> None:
        self.id = fresh_node_id()
        self.props = self._props
    
    def inputs(self) -> list[SIRNode]:
        return [self.lhs, self.rhs, self.temperature]
    
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> GateOp:
        return GateOp(
            compare=self.compare,
            lhs=new_inputs[0],
            rhs=new_inputs[1],
            temperature=new_inputs[2],
            _props=self.props
        )
    
    def __str__(self) -> str:
        return f"Gate({self.compare.name}, {self.lhs.id}, {self.rhs.id}, temp={self.temperature.id})"


@dataclass
class MixOp(SIRNode):
    """
    Soft conditional mixing.
    
    mix(gate, a, b) = gate * a + (1 - gate) * b
    
    Used to make if-expressions differentiable.
    """
    gate: SIRNode
    then_value: SIRNode
    else_value: SIRNode
    _props: SIRProperty = field(default_factory=SIRProperty)
    
    def __post_init__(self) -> None:
        self.id = fresh_node_id()
        self.props = self._props
    
    def inputs(self) -> list[SIRNode]:
        return [self.gate, self.then_value, self.else_value]
    
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> MixOp:
        return MixOp(
            gate=new_inputs[0],
            then_value=new_inputs[1],
            else_value=new_inputs[2],
            _props=self.props
        )
    
    def __str__(self) -> str:
        return f"Mix({self.gate.id}, {self.then_value.id}, {self.else_value.id})"


class ConstraintKind(Enum):
    """Kinds of constraints."""
    EQUALITY = auto()      # a == b (robust L2)
    INEQUALITY = auto()    # a <= b (softplus)
    BOOLEAN = auto()       # expr is true (1 - gate)
    LIKELIHOOD = auto()    # log p(x | params)


@dataclass
class ConstraintOp(SIRNode):
    """
    Constraint representation.
    
    Constraints are compiled to scalar penalty terms that
    are added to the objective.
    """
    kind: ConstraintKind
    lhs: SIRNode
    rhs: Optional[SIRNode]
    weight: SIRNode
    slack: Optional[SIRNode] = None
    _props: SIRProperty = field(default_factory=SIRProperty)
    
    def __post_init__(self) -> None:
        self.id = fresh_node_id()
        self.props = self._props
    
    def inputs(self) -> list[SIRNode]:
        result = [self.lhs, self.weight]
        if self.rhs:
            result.append(self.rhs)
        if self.slack:
            result.append(self.slack)
        return result
    
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> ConstraintOp:
        idx = 0
        lhs = new_inputs[idx]; idx += 1
        weight = new_inputs[idx]; idx += 1
        rhs = new_inputs[idx] if self.rhs and idx < len(new_inputs) else None; idx += 1 if self.rhs else 0
        slack = new_inputs[idx] if self.slack and idx < len(new_inputs) else None
        
        return ConstraintOp(
            kind=self.kind,
            lhs=lhs,
            rhs=rhs,
            weight=weight,
            slack=slack,
            _props=self.props
        )
    
    def __str__(self) -> str:
        rhs_str = f", {self.rhs.id}" if self.rhs else ""
        return f"Constraint({self.kind.name}, {self.lhs.id}{rhs_str}, weight={self.weight.id})"


# Reference Nodes


@dataclass
class ParamRef(SIRNode):
    """
    Reference to a learnable parameter.
    
    Parameters are gradient sinks - they can be optimized.
    """
    name: str
    _props: SIRProperty = field(default_factory=lambda: SIRProperty(
        role=RoleInfo.param(),
        requires_grad=True
    ))
    
    def __post_init__(self) -> None:
        self.id = fresh_node_id()
        self.props = self._props
    
    def inputs(self) -> list[SIRNode]:
        return []
    
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> ParamRef:
        return ParamRef(name=self.name, _props=self.props)
    
    def __str__(self) -> str:
        return f"Param({self.name})"


@dataclass
class ObsRef(SIRNode):
    """
    Reference to an observed value.
    
    Observations are fixed - gradients cannot flow into them.
    """
    name: str
    _props: SIRProperty = field(default_factory=lambda: SIRProperty(
        role=RoleInfo.obs(),
        requires_grad=False
    ))
    
    def __post_init__(self) -> None:
        self.id = fresh_node_id()
        self.props = self._props
    
    def inputs(self) -> list[SIRNode]:
        return []
    
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> ObsRef:
        return ObsRef(name=self.name, _props=self.props)
    
    def __str__(self) -> str:
        return f"Obs({self.name})"


@dataclass
class Const(SIRNode):
    """
    Constant value.
    
    Constants are known at compile time.
    """
    value: Any
    _props: SIRProperty = field(default_factory=lambda: SIRProperty(
        role=RoleInfo.const(),
        requires_grad=False
    ))
    
    def __post_init__(self) -> None:
        self.id = fresh_node_id()
        self.props = self._props
    
    def inputs(self) -> list[SIRNode]:
        return []
    
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> Const:
        return Const(value=self.value, _props=self.props)
    
    def __str__(self) -> str:
        return f"Const({self.value})"


# Gradient Control


@dataclass
class StopGrad(SIRNode):
    """
    Stop gradient flow.
    
    The value passes through but gradients do not.
    Equivalent to PyTorch's detach().
    """
    operand: SIRNode
    _props: SIRProperty = field(default_factory=SIRProperty)
    
    def __post_init__(self) -> None:
        self.id = fresh_node_id()
        self.props = self._props.with_grad(False)
    
    def inputs(self) -> list[SIRNode]:
        return [self.operand]
    
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> StopGrad:
        return StopGrad(operand=new_inputs[0], _props=self.props)
    
    def __str__(self) -> str:
        return f"StopGrad({self.operand.id})"


@dataclass
class Harden(SIRNode):
    """
    Harden a soft value to discrete.
    
    Used in infer mode to convert gates to boolean decisions.
    """
    operand: SIRNode
    threshold: float = 0.5
    _props: SIRProperty = field(default_factory=lambda: SIRProperty(
        dtype=BoolType(),
        requires_grad=False
    ))
    
    def __post_init__(self) -> None:
        self.id = fresh_node_id()
        self.props = self._props
    
    def inputs(self) -> list[SIRNode]:
        return [self.operand]
    
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> Harden:
        return Harden(operand=new_inputs[0], threshold=self.threshold, _props=self.props)
    
    def __str__(self) -> str:
        return f"Harden({self.operand.id}, threshold={self.threshold})"


@dataclass
class TupleIndex(SIRNode):
    """
    Extract an element from a tuple-returning node.
    """
    operand: SIRNode
    index: int
    _props: SIRProperty = field(default_factory=SIRProperty)
    
    def __post_init__(self) -> None:
        self.id = fresh_node_id()
        self.props = self._props
    
    def inputs(self) -> list[SIRNode]:
        return [self.operand]
    
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> TupleIndex:
        return TupleIndex(operand=new_inputs[0], index=self.index, _props=self.props)
    
    def __str__(self) -> str:
        return f"TupleIndex({self.operand.id}, {self.index})"


@dataclass
class GradBoundary(SIRNode):
    """
    Gradient boundary marker.
    
    Marks where gradient computation should stop or be modified.
    """
    operand: SIRNode
    kind: str = "stop"  # "stop", "checkpoint", "custom"
    _props: SIRProperty = field(default_factory=SIRProperty)
    
    def __post_init__(self) -> None:
        self.id = fresh_node_id()
        self.props = self._props
    
    def inputs(self) -> list[SIRNode]:
        return [self.operand]
    
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> GradBoundary:
        return GradBoundary(operand=new_inputs[0], kind=self.kind, _props=self.props)
    
    def __str__(self) -> str:
        return f"GradBoundary({self.operand.id}, {self.kind})"


# Probabilistic Nodes


@dataclass
class RandomVar(SIRNode):
    """
    Random variable (sample from distribution).
    
    Adds stochastic effect and uses reparameterization
    when possible for gradient flow.
    """
    distribution: str  # "Normal", "Bernoulli", etc.
    params: list[SIRNode]
    kwargs: dict[str, SIRNode] = field(default_factory=dict)
    _props: SIRProperty = field(default_factory=lambda: SIRProperty(
        effects=EffectSet.stochastic()
    ))
    
    def __post_init__(self) -> None:
        self.id = fresh_node_id()
        self.props = self._props
    
    def inputs(self) -> list[SIRNode]:
        return self.params + list(self.kwargs.values())
    
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> RandomVar:
        num_params = len(self.params)
        new_params = new_inputs[:num_params]
        new_kwargs = {}
        for i, key in enumerate(self.kwargs.keys()):
            new_kwargs[key] = new_inputs[num_params + i]
        return RandomVar(distribution=self.distribution, params=new_params, kwargs=new_kwargs, _props=self.props)
    
    def __str__(self) -> str:
        params_str = ", ".join(str(p.id) for p in self.params)
        return f"Rand({self.distribution}({params_str}))"


@dataclass
class Observe(SIRNode):
    """
    Probabilistic observation.
    
    Returns log probability of observing the value under the distribution.
    """
    value: SIRNode
    distribution: str
    params: list[SIRNode]
    kwargs: dict[str, SIRNode] = field(default_factory=dict)
    _props: SIRProperty = field(default_factory=SIRProperty)
    
    def __post_init__(self) -> None:
        self.id = fresh_node_id()
        self.props = self._props
    
    def inputs(self) -> list[SIRNode]:
        return [self.value] + self.params + list(self.kwargs.values())
    
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> Observe:
        num_params = len(self.params)
        new_value = new_inputs[0]
        new_params = new_inputs[1:1+num_params]
        new_kwargs = {}
        for i, key in enumerate(self.kwargs.keys()):
            new_kwargs[key] = new_inputs[1 + num_params + i]
        return Observe(
            value=new_value,
            distribution=self.distribution,
            params=new_params,
            kwargs=new_kwargs,
            _props=self.props
        )
    
    def __str__(self) -> str:
        params_str = ", ".join(str(p.id) for p in self.params)
        return f"Observe({self.value.id}, {self.distribution}({params_str}))"


# Mode-Specific Nodes


@dataclass
class ModeSwitch(SIRNode):
    """
    Mode-specific value selection.
    
    Returns different values depending on execution mode.
    """
    train_value: SIRNode
    infer_value: SIRNode
    analyze_value: Optional[SIRNode] = None
    _props: SIRProperty = field(default_factory=SIRProperty)
    
    def __post_init__(self) -> None:
        self.id = fresh_node_id()
        self.props = self._props
    
    def inputs(self) -> list[SIRNode]:
        result = [self.train_value, self.infer_value]
        if self.analyze_value:
            result.append(self.analyze_value)
        return result
    
    def clone_with_inputs(self, new_inputs: list[SIRNode]) -> ModeSwitch:
        return ModeSwitch(
            train_value=new_inputs[0],
            infer_value=new_inputs[1],
            analyze_value=new_inputs[2] if len(new_inputs) > 2 else None,
            _props=self.props
        )
    
    def __str__(self) -> str:
        return f"ModeSwitch(train={self.train_value.id}, infer={self.infer_value.id})"


# Control Flow


@dataclass
class SIRBlock:
    """
    A block of SIR nodes with a result.
    """
    nodes: list[SIRNode]
    result: Optional[SIRNode] = None
    
    def __iter__(self):
        return iter(self.nodes)
    
    def __len__(self) -> int:
        return len(self.nodes)


@dataclass
class SIRFunction:
    """
    A function in SIR.
    """
    name: str
    params: list[tuple[str, SIRProperty]]
    body: SIRBlock
    return_type: Type
    effects: EffectSet = field(default_factory=EffectSet.pure)
    
    @property
    def param_names(self) -> list[str]:
        return [name for name, _ in self.params]


@dataclass
class SIRLearnConfig:
    """Configuration for a learn block."""
    mode: str = "train"
    optimizer: Optional[Any] = None  # AST Expression or simplified spec
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    location: Optional[SourceLocation] = None

@dataclass
class SIRModule:
    """
    A complete SIR module.
    """
    name: str
    functions: dict[str, SIRFunction] = field(default_factory=dict)
    params: dict[str, ParamRef] = field(default_factory=dict)
    constants: dict[str, Const] = field(default_factory=dict)
    constraints: list[ConstraintOp] = field(default_factory=list)
    learn_configs: list[SIRLearnConfig] = field(default_factory=list)
    
    def add_function(self, func: SIRFunction) -> None:
        self.functions[func.name] = func
    
    def add_param(self, name: str, param: ParamRef) -> None:
        self.params[name] = param
    
    def add_constraint(self, constraint: ConstraintOp) -> None:
        self.constraints.append(constraint)
        
    def add_learn_config(self, config: SIRLearnConfig) -> None:
        self.learn_configs.append(config)


# Utilities


def walk_sir(node: SIRNode) -> list[SIRNode]:
    """Walk a SIR graph in topological order."""
    visited: set[int] = set()
    result: list[SIRNode] = []
    
    def visit(n: SIRNode) -> None:
        if n.id in visited:
            return
        visited.add(n.id)
        for inp in n.inputs():
            visit(inp)
        result.append(n)
    
    visit(node)
    return result


def replace_inputs(node: SIRNode, replacements: dict[int, SIRNode]) -> SIRNode:
    """Replace inputs in a node according to a mapping."""
    new_inputs = []
    changed = False
    
    for inp in node.inputs():
        if inp.id in replacements:
            new_inputs.append(replacements[inp.id])
            changed = True
        else:
            new_inputs.append(inp)
    
    if changed:
        return node.clone_with_inputs(new_inputs)
    return node
