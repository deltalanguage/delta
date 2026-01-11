"""
Type inference for Delta.

Implements Hindley-Milner type inference extended with:
- Effect inference
- Role propagation
- Mode constraints
- Tensor shape inference
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
from copy import deepcopy

from delta.frontend.ast import (
    ASTNode, Module, Statement, Expression,
    FunctionDef, StructDef, LetStmt, AssignStmt, ExprStmt, ReturnStmt,
    IfStmt, WhileStmt, ForStmt, LearnBlock, ConstraintStmt,
    ParamDecl, ObsDecl, ImportStmt,
    Block, IfExpr, BinaryOp, UnaryOp, Call, MethodCall,
    Index, FieldAccess, Tensor, Param, Obs, Identifier, Literal,
    Lambda, RandExpr, ObserveExpr, NonDiffBlock,
    BinaryOperator, UnaryOperator, ConstraintKind,
    TypeAnnotation, SimpleType, TensorType as ASTTensorType, 
    FunctionType as ASTFunctionType, GenericType as ASTGenericType,
    Parameter, IdentifierPattern, TuplePattern, Pattern,
)
from delta.types.types import (
    Type, TypeVar, UnitType, BoolType, IntType, FloatType, StringType,
    TensorType, FunctionType, TupleType, StructType, GenericType,
    UnionType, NeverType, AnyType,
    Substitution, unify, apply_substitution, UnificationError,
    fresh_type_var, TypeScheme, instantiate, generalize,
    ConcreteDim, SymbolicDim, DynamicDim, ShapeDim,
)
from delta.types.effects import EffectSet, Effect, EffectVar, fresh_effect_var
from delta.types.roles import Role, RoleSet, RoleInfo
from delta.source import SourceLocation
from delta.errors import TypeError, ErrorCode


@dataclass
class TypedExpr:
    """
    Type information attached to an expression.
    
    Contains the inferred type, effects, and role information.
    """
    type: Type
    effects: EffectSet = field(default_factory=EffectSet.pure)
    role: RoleInfo = field(default_factory=lambda: RoleInfo.const())
    requires_grad: bool = False
    
    def __str__(self) -> str:
        return f"{self.type} / {self.effects} / {self.role}"


@dataclass
class Binding:
    """A binding in the type environment."""
    scheme: TypeScheme
    role: RoleInfo
    mutable: bool = False
    location: Optional[SourceLocation] = None
    
    @property
    def type(self) -> Type:
        """Get the underlying type (instantiate if polymorphic)."""
        return instantiate(self.scheme)


class TypeEnvironment:
    """
    Type environment for type inference.
    
    Maps names to type schemes and role information.
    Supports lexical scoping with parent environments.
    """
    
    def __init__(self, parent: Optional[TypeEnvironment] = None) -> None:
        self.bindings: dict[str, Binding] = {}
        self.parent = parent
        self.type_defs: dict[str, Type] = {}
        # For tracking return types within function bodies
        self.expected_return_type: Optional[Type] = None
        self.return_types: list[tuple[Type, SourceLocation]] = []
    
    def lookup(self, name: str) -> Optional[Binding]:
        """Look up a binding by name."""
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        return None
    
    def define(
        self, 
        name: str, 
        typ: Type,
        role: RoleInfo,
        mutable: bool = False,
        location: Optional[SourceLocation] = None
    ) -> None:
        """Define a new binding."""
        scheme = TypeScheme(type_vars=frozenset(), body=typ)
        self.bindings[name] = Binding(
            scheme=scheme,
            role=role,
            mutable=mutable,
            location=location
        )
    
    def define_scheme(
        self,
        name: str,
        scheme: TypeScheme,
        role: RoleInfo,
        mutable: bool = False,
        location: Optional[SourceLocation] = None
    ) -> None:
        """Define a binding with a type scheme."""
        self.bindings[name] = Binding(
            scheme=scheme,
            role=role,
            mutable=mutable,
            location=location
        )
    
    def define_type(self, name: str, typ: Type) -> None:
        """Define a type alias or struct type."""
        self.type_defs[name] = typ
    
    def lookup_type(self, name: str) -> Optional[Type]:
        """Look up a type definition."""
        if name in self.type_defs:
            return self.type_defs[name]
        if self.parent:
            return self.parent.lookup_type(name)
        return None
    
    def child_scope(self) -> TypeEnvironment:
        """Create a child scope."""
        return TypeEnvironment(parent=self)
    
    def free_vars(self) -> set[TypeVar]:
        """Get all free type variables in the environment."""
        result: set[TypeVar] = set()
        for binding in self.bindings.values():
            result |= binding.scheme.free_vars()
        if self.parent:
            result |= self.parent.free_vars()
        return result
    
    def generalize(self, typ: Type) -> TypeScheme:
        """Generalize a type to a scheme in this environment."""
        return generalize(typ, self.free_vars())


class TypeInference:
    """
    Type inference engine for Delta.
    
    Implements Algorithm W (Hindley-Milner) extended with
    effect inference and role propagation.
    """
    
    def __init__(self) -> None:
        self.substitution = Substitution()
        self.errors: list[TypeError] = []
        self.current_mode = "train"
        self.in_learn_block = False
    
    def infer_module(self, module: Module, env: Optional[TypeEnvironment] = None) -> TypeEnvironment:
        """Infer types for an entire module."""
        if env is None:
            env = self._create_builtin_env()
        
        for item in module.items:
            self._infer_statement(item, env)
        
        return env
    
    def infer_expression(self, expr: Expression, env: TypeEnvironment) -> TypedExpr:
        """Infer the type of an expression."""
        return self._infer_expr(expr, env)
    
    def _create_builtin_env(self) -> TypeEnvironment:
        """Create environment with built-in types and functions."""
        env = TypeEnvironment()
        
        # Built-in types
        env.define_type("Unit", UnitType())
        env.define_type("Bool", BoolType())
        env.define_type("Int", IntType())
        env.define_type("Float", FloatType())
        env.define_type("String", StringType())
        
        # Common type aliases
        tensor_t = TensorType(FloatType())
        any_t = AnyType()  # For flexible argument types
        
        # ═══════════════════════════════════════════════════════════════════
        # Tensor Creation (variadic - accept any number of dim args)
        # ═══════════════════════════════════════════════════════════════════
        
        env.define("zeros", FunctionType(
            param_types=(), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("ones", FunctionType(
            param_types=(), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("randn", FunctionType(
            param_types=(), return_type=tensor_t, variadic=True, min_args=1,
            effects=frozenset({"stoch"})
        ), RoleInfo.const())
        
        env.define("rand", FunctionType(
            param_types=(), return_type=tensor_t, variadic=True, min_args=1,
            effects=frozenset({"stoch"})
        ), RoleInfo.const())
        
        env.define("full", FunctionType(
            param_types=(), return_type=tensor_t, variadic=True, min_args=2
        ), RoleInfo.const())
        
        env.define("arange", FunctionType(
            param_types=(), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("linspace", FunctionType(
            param_types=(), return_type=tensor_t, variadic=True, min_args=3
        ), RoleInfo.const())
        
        env.define("eye", FunctionType(
            param_types=(), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("range", FunctionType(
            param_types=(), return_type=AnyType(), variadic=True, min_args=1
        ), RoleInfo.const())
        
        # ═══════════════════════════════════════════════════════════════════
        # Math Functions (tensor -> tensor, with optional kwargs)
        # ═══════════════════════════════════════════════════════════════════
        
        # Unary math functions - accept tensor or scalar
        for fn in ["exp", "log", "sin", "cos", "tan", "tanh", "sigmoid", 
                   "relu", "gelu", "leaky_relu", "elu", "selu", "swish", "mish",
                   "sqrt", "abs", "neg", "sign", "floor", "ceil", "round"]:
            env.define(fn, FunctionType(
                param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
            ), RoleInfo.const())
        
        # Softmax and log_softmax - tensor + optional dim
        for fn in ["softmax", "log_softmax"]:
            env.define(fn, FunctionType(
                param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
            ), RoleInfo.const())
        
        # Binary math - pow, etc. (accept any numeric types)
        env.define("pow", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=2
        ), RoleInfo.const())
        
        env.define("clamp", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        # ═══════════════════════════════════════════════════════════════════
        # Reduction Functions (variadic for dim/keepdim kwargs)
        # ═══════════════════════════════════════════════════════════════════
        
        for fn in ["sum", "mean", "prod", "std", "var", "norm"]:
            env.define(fn, FunctionType(
                param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
            ), RoleInfo.const())
        
        for fn in ["max", "min"]:
            env.define(fn, FunctionType(
                param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
            ), RoleInfo.const())
        
        for fn in ["argmax", "argmin"]:
            env.define(fn, FunctionType(
                param_types=(tensor_t,), return_type=TensorType(IntType()), variadic=True, min_args=1
            ), RoleInfo.const())
        
        # ═══════════════════════════════════════════════════════════════════
        # Shape Operations (most are variadic)
        # ═══════════════════════════════════════════════════════════════════
        
        env.define("reshape", FunctionType(
            param_types=(tensor_t, any_t), return_type=tensor_t, variadic=True, min_args=2
        ), RoleInfo.const())
        
        env.define("view", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("transpose", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("permute", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("squeeze", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("unsqueeze", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=2
        ), RoleInfo.const())
        
        env.define("flatten", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("expand", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("repeat", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("cat", FunctionType(
            param_types=(AnyType(),), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("stack", FunctionType(
            param_types=(AnyType(),), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("split", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("chunk", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=2
        ), RoleInfo.const())
        
        # shape(x) or shape(x, dim)
        env.define("shape", FunctionType(
            param_types=(tensor_t,), return_type=IntType(), variadic=True, min_args=1
        ), RoleInfo.const())
        
        # slice(x, dim, start, end)
        env.define("slice", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("narrow", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=4
        ), RoleInfo.const())
        
        env.define("index_select", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=3
        ), RoleInfo.const())
        
        env.define("gather", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=3
        ), RoleInfo.const())
        
        # ═══════════════════════════════════════════════════════════════════
        # Linear Algebra
        # ═══════════════════════════════════════════════════════════════════
        
        env.define("matmul", FunctionType(
            param_types=(tensor_t, tensor_t), return_type=tensor_t
        ), RoleInfo.const())
        
        env.define("mm", FunctionType(
            param_types=(tensor_t, tensor_t), return_type=tensor_t
        ), RoleInfo.const())
        
        env.define("bmm", FunctionType(
            param_types=(tensor_t, tensor_t), return_type=tensor_t
        ), RoleInfo.const())
        
        env.define("mv", FunctionType(
            param_types=(tensor_t, tensor_t), return_type=tensor_t
        ), RoleInfo.const())
        
        env.define("dot", FunctionType(
            param_types=(tensor_t, tensor_t), return_type=tensor_t
        ), RoleInfo.const())
        
        env.define("outer", FunctionType(
            param_types=(tensor_t, tensor_t), return_type=tensor_t
        ), RoleInfo.const())
        
        env.define("einsum", FunctionType(
            param_types=(StringType(),), return_type=tensor_t, variadic=True, min_args=2
        ), RoleInfo.const())
        
        env.define("inv", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t
        ), RoleInfo.const())
        
        env.define("det", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t
        ), RoleInfo.const())
        
        env.define("solve", FunctionType(
            param_types=(tensor_t, tensor_t), return_type=tensor_t
        ), RoleInfo.const())
        
        # ═══════════════════════════════════════════════════════════════════
        # Matrix Operations
        # ═══════════════════════════════════════════════════════════════════
        
        env.define("tril", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("triu", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("diag", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("causal_mask", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t
        ), RoleInfo.const())
        
        # ═══════════════════════════════════════════════════════════════════
        # Comparison Operations
        # ═══════════════════════════════════════════════════════════════════
        
        for fn in ["eq", "ne", "lt", "le", "gt", "ge"]:
            env.define(fn, FunctionType(
                param_types=(tensor_t, tensor_t), return_type=tensor_t
            ), RoleInfo.const())
        
        for fn in ["isnan", "isinf", "isfinite"]:
            env.define(fn, FunctionType(
                param_types=(tensor_t,), return_type=tensor_t
            ), RoleInfo.const())
        
        env.define("where", FunctionType(
            param_types=(tensor_t, tensor_t, tensor_t), return_type=tensor_t
        ), RoleInfo.const())
        
        # ═══════════════════════════════════════════════════════════════════
        # Neural Network Layers
        # ═══════════════════════════════════════════════════════════════════
        
        env.define("embedding", FunctionType(
            param_types=(tensor_t, tensor_t), return_type=tensor_t
        ), RoleInfo.const())
        
        env.define("linear", FunctionType(
            param_types=(tensor_t, tensor_t), return_type=tensor_t, variadic=True, min_args=2
        ), RoleInfo.const())
        
        # Capitalized Layer Constructors (alias for now, or could map to specific types)
        env.define("Linear", FunctionType(
            param_types=(IntType(), IntType()), return_type=AnyType(), variadic=True
        ), RoleInfo.const())
        
        env.define("Embedding", FunctionType(
            param_types=(IntType(), IntType()), return_type=AnyType(), variadic=True
        ), RoleInfo.const())

        for fn in ["conv1d", "conv2d"]:
            env.define(fn, FunctionType(
                param_types=(tensor_t, tensor_t), return_type=tensor_t, variadic=True, min_args=2
            ), RoleInfo.const())
        
        for fn in ["Conv1d", "Conv2d"]:
            env.define(fn, FunctionType(
                param_types=(IntType(), IntType(), IntType()), return_type=AnyType(), variadic=True
            ), RoleInfo.const())

        for fn in ["batch_norm", "layer_norm"]:
            env.define(fn, FunctionType(
                param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
            ), RoleInfo.const())
            
        for fn in ["BatchNorm1d", "BatchNorm2d", "LayerNorm"]:
            env.define(fn, FunctionType(
                param_types=(IntType(),), return_type=AnyType(), variadic=True
            ), RoleInfo.const())

        env.define("dropout", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("Dropout", FunctionType(
            param_types=(FloatType(),), return_type=AnyType(), variadic=True
        ), RoleInfo.const())
        
        for fn in ["LSTM", "GRU", "RNN"]:
            env.define(fn, FunctionType(
                param_types=(IntType(), IntType()), return_type=AnyType(), variadic=True
            ), RoleInfo.const())
        
        env.define("MultiheadAttention", FunctionType(
            param_types=(IntType(), IntType()), return_type=AnyType(), variadic=True
        ), RoleInfo.const())
        
        for fn in ["max_pool1d", "max_pool2d", "avg_pool1d", "avg_pool2d"]:
            env.define(fn, FunctionType(
                param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=2
            ), RoleInfo.const())
        
        env.define("attention", FunctionType(
            param_types=(tensor_t, tensor_t, tensor_t), return_type=tensor_t, variadic=True, min_args=3
        ), RoleInfo.const())
        
        # ═══════════════════════════════════════════════════════════════════
        # Loss Functions
        # ═══════════════════════════════════════════════════════════════════
        
        for fn in ["mse_loss", "l1_loss", "smooth_l1_loss", "cross_entropy", 
                   "nll_loss", "binary_cross_entropy", "kl_div"]:
            # Labels/targets can be any_t (e.g. Tensor[Int] for cross_entropy)
            env.define(fn, FunctionType(
                param_types=(tensor_t, any_t), return_type=tensor_t, variadic=True, min_args=2
            ), RoleInfo.const())
        
        env.define("cosine_similarity", FunctionType(
            param_types=(tensor_t, tensor_t), return_type=tensor_t, variadic=True, min_args=2
        ), RoleInfo.const())
        
        env.define("triplet_margin_loss", FunctionType(
            param_types=(tensor_t, tensor_t, tensor_t), return_type=tensor_t, variadic=True, min_args=3
        ), RoleInfo.const())
        
        # ═══════════════════════════════════════════════════════════════════
        # Distributions
        # ═══════════════════════════════════════════════════════════════════
        
        env.define("Normal", FunctionType(
            param_types=(), return_type=GenericType("Distribution", (FloatType(),)),
            variadic=True, min_args=2
        ), RoleInfo.const())
        
        env.define("Bernoulli", FunctionType(
            param_types=(), return_type=GenericType("Distribution", (BoolType(),)),
            variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("Categorical", FunctionType(
            param_types=(tensor_t,), return_type=GenericType("Distribution", (IntType(),))
        ), RoleInfo.const())
        
        env.define("Uniform", FunctionType(
            param_types=(), return_type=GenericType("Distribution", (FloatType(),)),
            variadic=True, min_args=2
        ), RoleInfo.const())
        
        env.define("Beta", FunctionType(
            param_types=(), return_type=GenericType("Distribution", (FloatType(),)),
            variadic=True, min_args=2
        ), RoleInfo.const())
        
        env.define("Gamma", FunctionType(
            param_types=(), return_type=GenericType("Distribution", (FloatType(),)),
            variadic=True, min_args=2
        ), RoleInfo.const())
        
        env.define("Exponential", FunctionType(
            param_types=(), return_type=GenericType("Distribution", (FloatType(),)),
            variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("Poisson", FunctionType(
            param_types=(), return_type=GenericType("Distribution", (IntType(),)),
            variadic=True, min_args=1
        ), RoleInfo.const())
        
        env.define("Dirichlet", FunctionType(
            param_types=(tensor_t,), return_type=GenericType("Distribution", (tensor_t,))
        ), RoleInfo.const())
        
        env.define("MultivariateNormal", FunctionType(
            param_types=(tensor_t,), return_type=GenericType("Distribution", (tensor_t,)),
            variadic=True, min_args=1
        ), RoleInfo.const())
        
        # ═══════════════════════════════════════════════════════════════════
        # Optimizers (all accept kwargs)
        # ═══════════════════════════════════════════════════════════════════
        
        for opt in ["Adam", "AdamW", "SGD", "RMSprop", "Adagrad", "Adadelta"]:
            env.define(opt, FunctionType(
                param_types=(), return_type=GenericType("Optimizer", ()), variadic=True
            ), RoleInfo.const())
        
        # ═══════════════════════════════════════════════════════════════════
        # Gradient Control
        # ═══════════════════════════════════════════════════════════════════
        
        # StopGrad(x) - stops gradient flow, returns same type as input
        env.define("StopGrad", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t
        ), RoleInfo.const())
        
        # Harden(gate) - converts soft value to discrete, returns Bool (as 0.0 or 1.0 floats)
        env.define("Harden", FunctionType(
            param_types=(tensor_t,), return_type=tensor_t, variadic=True, min_args=1
        ), RoleInfo.const())
        
        return env
    
    def _infer_statement(self, stmt: Statement, env: TypeEnvironment) -> None:
        """Infer types for a statement."""
        if isinstance(stmt, FunctionDef):
            self._infer_function_def(stmt, env)
        elif isinstance(stmt, StructDef):
            self._infer_struct_def(stmt, env)
        elif isinstance(stmt, LetStmt):
            self._infer_let_stmt(stmt, env)
        elif isinstance(stmt, AssignStmt):
            self._infer_assign_stmt(stmt, env)
        elif isinstance(stmt, ExprStmt):
            self._infer_expr(stmt.expr, env)
        elif isinstance(stmt, ReturnStmt):
            if stmt.value:
                self._infer_expr(stmt.value, env)
        elif isinstance(stmt, IfStmt):
            self._infer_if_stmt(stmt, env)
        elif isinstance(stmt, WhileStmt):
            self._infer_while_stmt(stmt, env)
        elif isinstance(stmt, ForStmt):
            self._infer_for_stmt(stmt, env)
        elif isinstance(stmt, LearnBlock):
            self._infer_learn_block(stmt, env)
        elif isinstance(stmt, ConstraintStmt):
            self._infer_constraint_stmt(stmt, env)
        elif isinstance(stmt, ParamDecl):
            self._infer_param_decl(stmt, env)
        elif isinstance(stmt, ObsDecl):
            self._infer_obs_decl(stmt, env)
        elif isinstance(stmt, ImportStmt):
            pass  # Handled by module system
    
    def _infer_function_def(self, func: FunctionDef, env: TypeEnvironment) -> None:
        """Infer types for a function definition."""
        # Create parameter types
        param_types: list[Type] = []
        body_env = env.child_scope()
        
        for param in func.params:
            if param.type_annotation:
                param_type = self._resolve_type_annotation(param.type_annotation, env)
            else:
                param_type = fresh_type_var("P")
            
            param_types.append(param_type)
            
            # Determine role from annotation
            role = RoleInfo.const()
            if param.role == "param":
                role = RoleInfo.param()
            elif param.role == "obs":
                role = RoleInfo.obs()
            
            body_env.define(param.name, param_type, role, location=param.location)
        
        # Infer return type
        if func.return_type:
            return_type = self._resolve_type_annotation(func.return_type, env)
        else:
            return_type = fresh_type_var("R")
        
        # Store expected return type in environment for return statement checking
        body_env.expected_return_type = return_type
        
        # Infer body and collect return statement types
        body_result = self._infer_block_with_returns(func.body, body_env)
        
        # If there are explicit return statements, they define the return type
        # Otherwise use the trailing expression or Unit
        if body_env.return_types:
            # Unify all return types with expected return type
            for ret_type, ret_loc in body_env.return_types:
                try:
                    subst = unify(ret_type, return_type)
                    self.substitution = self.substitution.compose(subst)
                except UnificationError:
                    self.errors.append(TypeError(
                        message=f"Return type mismatch: expected {return_type}, got {ret_type}",
                        location=ret_loc,
                        code=ErrorCode.E301_TYPE_MISMATCH
                    ))
        elif body_result.type != UnitType():
            # No return statements, but there's a trailing expression
            try:
                subst = unify(body_result.type, return_type)
                self.substitution = self.substitution.compose(subst)
            except UnificationError:
                self.errors.append(TypeError(
                    message=f"Return type mismatch: expected {return_type}, got {body_result.type}",
                    location=func.location,
                    code=ErrorCode.E301_TYPE_MISMATCH
                ))
        
        # Create function type
        func_type = FunctionType(
            param_types=tuple(apply_substitution(t, self.substitution) for t in param_types),
            return_type=apply_substitution(return_type, self.substitution),
            effects=frozenset(func.effects) if func.effects else frozenset()
        )
        
        # Generalize and bind
        scheme = env.generalize(func_type)
        env.define_scheme(func.name, scheme, RoleInfo.const(), location=func.location)
    
    def _infer_struct_def(self, struct: StructDef, env: TypeEnvironment) -> None:
        """Infer types for a struct definition."""
        fields: list[tuple[str, Type]] = []
        
        for f in struct.fields:
            field_type = self._resolve_type_annotation(f.type_annotation, env)
            fields.append((f.name, field_type))
        
        struct_type = StructType(
            name=struct.name,
            fields=tuple(fields)
        )
        
        env.define_type(struct.name, struct_type)
        
        # Also define constructor function
        constructor_type = FunctionType(
            param_types=tuple(t for _, t in fields),
            return_type=struct_type
        )
        env.define(struct.name, constructor_type, RoleInfo.const())
    
    def _infer_let_stmt(self, stmt: LetStmt, env: TypeEnvironment) -> None:
        """Infer types for a let statement."""
        value_typed = self._infer_expr(stmt.value, env)
        
        # Check type annotation if present
        if stmt.type_annotation:
            expected_type = self._resolve_type_annotation(stmt.type_annotation, env)
            try:
                subst = unify(value_typed.type, expected_type)
                self.substitution = self.substitution.compose(subst)
            except UnificationError:
                self.errors.append(TypeError(
                    message=f"Type mismatch: expected {expected_type}, got {value_typed.type}",
                    location=stmt.location,
                    code=ErrorCode.E301_TYPE_MISMATCH
                ))
        
        # Bind the pattern
        self._bind_pattern(stmt.pattern, value_typed.type, value_typed.role, stmt.mutable, env)
    
    def _infer_assign_stmt(self, stmt: AssignStmt, env: TypeEnvironment) -> None:
        """Infer types for an assignment statement."""
        target_typed = self._infer_expr(stmt.target, env)
        value_typed = self._infer_expr(stmt.value, env)
        
        try:
            subst = unify(target_typed.type, value_typed.type)
            self.substitution = self.substitution.compose(subst)
        except UnificationError:
            self.errors.append(TypeError(
                message=f"Cannot assign {value_typed.type} to {target_typed.type}",
                location=stmt.location,
                code=ErrorCode.E301_TYPE_MISMATCH
            ))
    
    def _infer_if_stmt(self, stmt: IfStmt, env: TypeEnvironment) -> None:
        """Infer types for an if statement."""
        cond_typed = self._infer_expr(stmt.condition, env)
        
        # Condition must be boolean
        try:
            subst = unify(cond_typed.type, BoolType())
            self.substitution = self.substitution.compose(subst)
        except UnificationError:
            self.errors.append(TypeError(
                message=f"Condition must be Bool, got {cond_typed.type}",
                location=stmt.condition.location,
                code=ErrorCode.E301_TYPE_MISMATCH
            ))
        
        # Infer branches
        self._infer_block(stmt.then_block, env.child_scope())
        
        if stmt.else_block:
            if isinstance(stmt.else_block, Block):
                self._infer_block(stmt.else_block, env.child_scope())
            else:  # IfStmt (elif chain)
                self._infer_if_stmt(stmt.else_block, env)
    
    def _infer_while_stmt(self, stmt: WhileStmt, env: TypeEnvironment) -> None:
        """Infer types for a while statement."""
        cond_typed = self._infer_expr(stmt.condition, env)
        
        try:
            subst = unify(cond_typed.type, BoolType())
            self.substitution = self.substitution.compose(subst)
        except UnificationError:
            self.errors.append(TypeError(
                message=f"Condition must be Bool, got {cond_typed.type}",
                location=stmt.condition.location,
                code=ErrorCode.E301_TYPE_MISMATCH
            ))
        
        self._infer_block(stmt.body, env.child_scope())
    
    def _infer_for_stmt(self, stmt: ForStmt, env: TypeEnvironment) -> None:
        """Infer types for a for statement."""
        iter_typed = self._infer_expr(stmt.iterable, env)
        
        # Create element type variable
        elem_type = fresh_type_var("E")
        
        # Bind loop variable
        body_env = env.child_scope()
        self._bind_pattern(stmt.pattern, elem_type, RoleInfo.const(), False, body_env)
        
        self._infer_block(stmt.body, body_env)
    
    def _infer_learn_block(self, stmt: LearnBlock, env: TypeEnvironment) -> None:
        """Infer types for a learn block."""
        old_mode = self.current_mode
        old_in_learn = self.in_learn_block
        
        self.current_mode = stmt.mode
        self.in_learn_block = True
        
        try:
            self._infer_block(stmt.body, env.child_scope())
            
            if stmt.optimizer:
                self._infer_expr(stmt.optimizer, env)
            if stmt.epochs:
                epochs_typed = self._infer_expr(stmt.epochs, env)
                try:
                    subst = unify(epochs_typed.type, IntType())
                    self.substitution = self.substitution.compose(subst)
                except UnificationError:
                    self.errors.append(TypeError(
                        message=f"Epochs must be Int, got {epochs_typed.type}",
                        location=stmt.epochs.location,
                        code=ErrorCode.E301_TYPE_MISMATCH
                    ))
        finally:
            self.current_mode = old_mode
            self.in_learn_block = old_in_learn
    
    def _infer_constraint_stmt(self, stmt: ConstraintStmt, env: TypeEnvironment) -> None:
        """Infer types for a constraint statement."""
        expr_typed = self._infer_expr(stmt.expr, env)
        
        # Constraint expression should be boolean or numeric
        # (will be converted to penalty)
        
        if stmt.weight:
            weight_typed = self._infer_expr(stmt.weight, env)
            try:
                subst = unify(weight_typed.type, FloatType())
                self.substitution = self.substitution.compose(subst)
            except UnificationError:
                pass  # Allow Int too
        
        if stmt.slack:
            slack_typed = self._infer_expr(stmt.slack, env)
    
    def _infer_param_decl(self, stmt: ParamDecl, env: TypeEnvironment) -> None:
        """Infer types for a param declaration."""
        if stmt.type_annotation:
            param_type = self._resolve_type_annotation(stmt.type_annotation, env)
        elif stmt.initializer:
            init_typed = self._infer_expr(stmt.initializer, env)
            param_type = init_typed.type
        else:
            param_type = TensorType(FloatType())  # Default
        
        env.define(stmt.name, param_type, RoleInfo.param(), location=stmt.location)
    
    def _infer_obs_decl(self, stmt: ObsDecl, env: TypeEnvironment) -> None:
        """Infer types for an obs declaration."""
        if stmt.type_annotation:
            obs_type = self._resolve_type_annotation(stmt.type_annotation, env)
        else:
            obs_type = TensorType(FloatType())  # Default
        
        env.define(stmt.name, obs_type, RoleInfo.obs(), location=stmt.location)
    
    def _infer_block(self, block: Block, env: TypeEnvironment) -> TypedExpr:
        """Infer types for a block."""
        for stmt in block.statements:
            self._infer_statement(stmt, env)
        
        if block.result:
            return self._infer_expr(block.result, env)
        
        return TypedExpr(type=UnitType())
    
    def _infer_block_with_returns(self, block: Block, env: TypeEnvironment) -> TypedExpr:
        """Infer types for a function body block, tracking return statements."""
        for stmt in block.statements:
            self._infer_statement_with_returns(stmt, env)
        
        if block.result:
            return self._infer_expr(block.result, env)
        
        return TypedExpr(type=UnitType())
    
    def _infer_statement_with_returns(self, stmt: Statement, env: TypeEnvironment) -> None:
        """Infer statement types while tracking return statement types."""
        if isinstance(stmt, ReturnStmt):
            if stmt.value:
                ret_typed = self._infer_expr(stmt.value, env)
                env.return_types.append((ret_typed.type, stmt.location))
            else:
                env.return_types.append((UnitType(), stmt.location))
        elif isinstance(stmt, IfStmt):
            # Need to recursively track returns in if branches
            self._infer_expr(stmt.condition, env)
            self._infer_block_with_returns(stmt.then_block, env)
            if stmt.else_block:
                self._infer_block_with_returns(stmt.else_block, env)
        elif isinstance(stmt, WhileStmt):
            self._infer_expr(stmt.condition, env)
            self._infer_block_with_returns(stmt.body, env)
        elif isinstance(stmt, ForStmt):
            child_env = env.child_scope()
            iter_typed = self._infer_expr(stmt.iterable, child_env)
            # Bind loop variable - ForStmt has a pattern
            if isinstance(stmt.pattern, IdentifierPattern):
                child_env.define(stmt.pattern.name, fresh_type_var("I"), RoleInfo.const(), location=stmt.location)
            self._infer_block_with_returns(stmt.body, child_env)
        else:
            # Delegate to regular statement inference
            self._infer_statement(stmt, env)
    
    def _infer_expr(self, expr: Expression, env: TypeEnvironment) -> TypedExpr:
        """Infer the type of an expression."""
        if isinstance(expr, Literal):
            return self._infer_literal(expr)
        elif isinstance(expr, Identifier):
            return self._infer_identifier(expr, env)
        elif isinstance(expr, BinaryOp):
            return self._infer_binary_op(expr, env)
        elif isinstance(expr, UnaryOp):
            return self._infer_unary_op(expr, env)
        elif isinstance(expr, Call):
            return self._infer_call(expr, env)
        elif isinstance(expr, MethodCall):
            return self._infer_method_call(expr, env)
        elif isinstance(expr, Index):
            return self._infer_index(expr, env)
        elif isinstance(expr, FieldAccess):
            return self._infer_field_access(expr, env)
        elif isinstance(expr, IfExpr):
            return self._infer_if_expr(expr, env)
        elif isinstance(expr, Block):
            return self._infer_block(expr, env.child_scope())
        elif isinstance(expr, Lambda):
            return self._infer_lambda(expr, env)
        elif isinstance(expr, Tensor):
            return self._infer_tensor_literal(expr, env)
        elif isinstance(expr, Param):
            return self._infer_param_expr(expr, env)
        elif isinstance(expr, Obs):
            return self._infer_obs_expr(expr, env)
        elif isinstance(expr, RandExpr):
            return self._infer_rand_expr(expr, env)
        elif isinstance(expr, ObserveExpr):
            return self._infer_observe_expr(expr, env)
        elif isinstance(expr, NonDiffBlock):
            return self._infer_non_diff_block(expr, env)
        else:
            return TypedExpr(type=AnyType())
    
    def _infer_literal(self, lit: Literal) -> TypedExpr:
        """Infer type of a literal."""
        if lit.kind == 'int':
            return TypedExpr(type=IntType())
        elif lit.kind == 'float':
            return TypedExpr(type=FloatType())
        elif lit.kind == 'string':
            return TypedExpr(type=StringType())
        elif lit.kind == 'bool':
            return TypedExpr(type=BoolType())
        elif lit.kind == 'none' or lit.kind == 'unit':
            return TypedExpr(type=UnitType())
        else:
            return TypedExpr(type=AnyType())
    
    def _infer_identifier(self, ident: Identifier, env: TypeEnvironment) -> TypedExpr:
        """Infer type of an identifier."""
        binding = env.lookup(ident.name)
        
        if binding is None:
            self.errors.append(TypeError(
                message=f"Undefined variable: {ident.name}",
                location=ident.location,
                code=ErrorCode.E201_UNDEFINED_VARIABLE
            ))
            return TypedExpr(type=AnyType())
        
        # Instantiate the type scheme
        typ = instantiate(binding.scheme)
        
        return TypedExpr(
            type=typ,
            role=binding.role,
            requires_grad=binding.role.requires_grad
        )
    
    def _infer_binary_op(self, op: BinaryOp, env: TypeEnvironment) -> TypedExpr:
        """Infer type of a binary operation."""
        left = self._infer_expr(op.left, env)
        right = self._infer_expr(op.right, env)
        
        # Propagate roles
        result_role = RoleInfo.computed(left.role, right.role)
        effects = left.effects.union(right.effects)
        
        # Determine result type based on operator
        if op.op in (BinaryOperator.ADD, BinaryOperator.SUB, 
                     BinaryOperator.MUL, BinaryOperator.DIV, BinaryOperator.POW):
            # Numeric operations
            try:
                subst = unify(left.type, right.type)
                self.substitution = self.substitution.compose(subst)
                result_type = apply_substitution(left.type, self.substitution)
            except UnificationError:
                # Allow numeric coercion
                if isinstance(left.type, (IntType, FloatType)) and isinstance(right.type, (IntType, FloatType)):
                    result_type = FloatType()
                else:
                    result_type = left.type
            
            return TypedExpr(type=result_type, effects=effects, role=result_role)
        
        elif op.op == BinaryOperator.MATMUL:
            # Matrix multiplication - result is tensor
            return TypedExpr(
                type=TensorType(FloatType()),
                effects=effects,
                role=result_role
            )
        
        elif op.op in (BinaryOperator.EQ, BinaryOperator.NE, BinaryOperator.LT,
                       BinaryOperator.LE, BinaryOperator.GT, BinaryOperator.GE):
            # Comparison - result is bool
            return TypedExpr(
                type=BoolType(),
                effects=effects,
                role=result_role
            )
        
        elif op.op in (BinaryOperator.AND, BinaryOperator.OR):
            # Logical - result is bool
            return TypedExpr(
                type=BoolType(),
                effects=effects,
                role=result_role
            )
        
        else:
            return TypedExpr(type=left.type, effects=effects, role=result_role)
    
    def _infer_unary_op(self, op: UnaryOp, env: TypeEnvironment) -> TypedExpr:
        """Infer type of a unary operation."""
        operand = self._infer_expr(op.operand, env)
        
        if op.op == UnaryOperator.NOT:
            return TypedExpr(
                type=BoolType(),
                effects=operand.effects,
                role=operand.role
            )
        elif op.op == UnaryOperator.NEG:
            return operand
        else:
            return operand
    
    def _infer_call(self, call: Call, env: TypeEnvironment) -> TypedExpr:
        """Infer type of a function call."""
        func_typed = self._infer_expr(call.func, env)
        
        if not isinstance(func_typed.type, FunctionType):
            # Could be a type variable - create constraints
            arg_types = [self._infer_expr(arg, env).type for arg in call.args]
            result_type = fresh_type_var("R")
            
            expected_func_type = FunctionType(
                param_types=tuple(arg_types),
                return_type=result_type
            )
            
            try:
                subst = unify(func_typed.type, expected_func_type)
                self.substitution = self.substitution.compose(subst)
            except UnificationError as e:
                self.errors.append(TypeError(
                    message=f"Not callable: {func_typed.type}",
                    location=call.location,
                    code=ErrorCode.E303_NOT_CALLABLE
                ))
            
            return TypedExpr(type=apply_substitution(result_type, self.substitution))
        
        func_type = func_typed.type
        
        # Check argument count (variadic functions accept any count >= min_args)
        num_args = len(call.args) + len(call.kwargs)
        if func_type.variadic:
            if num_args < func_type.min_args:
                self.errors.append(TypeError(
                    message=f"Expected at least {func_type.min_args} arguments, got {num_args}",
                    location=call.location,
                    code=ErrorCode.E302_ARITY_MISMATCH
                ))
        else:
            if num_args != len(func_type.param_types):
                self.errors.append(TypeError(
                    message=f"Expected {len(func_type.param_types)} arguments, got {num_args}",
                    location=call.location,
                    code=ErrorCode.E302_ARITY_MISMATCH
                ))
        
        # Type check arguments
        effects = EffectSet.pure()
        roles: list[RoleInfo] = []
        
        for i, arg in enumerate(call.args):
            arg_typed = self._infer_expr(arg, env)
            
            if i < len(func_type.param_types):
                param_type = func_type.param_types[i]
            elif func_type.variadic:
                if func_type.param_types:
                    # For variadic functions, we repeat the last parameter type
                    param_type = func_type.param_types[-1]
                else:
                    param_type = fresh_type_var("V")
            else:
                # Extra arg beyond expected - skip type checking
                effects = effects.union(arg_typed.effects)
                roles.append(arg_typed.role)
                continue
            
            # Try unification with numeric coercion
            if not self._try_unify_with_coercion(arg_typed.type, param_type, arg.location):
                pass  # Error already added
            
            effects = effects.union(arg_typed.effects)
            roles.append(arg_typed.role)
        
        # Add function effects
        for eff in func_type.effects:
            if eff == "stoch":
                effects = effects.union(EffectSet.stochastic())
            elif eff == "non_diff":
                effects = effects.union(EffectSet.non_diff())
        
        return TypedExpr(
            type=apply_substitution(func_type.return_type, self.substitution),
            effects=effects,
            role=RoleInfo.computed(*roles) if roles else RoleInfo.const()
        )
    
    def _try_unify_with_coercion(self, arg_type: Type, param_type: Type, location: SourceLocation) -> bool:
        """Try to unify types, allowing numeric coercion (Int -> Float -> Tensor)."""
        try:
            subst = unify(arg_type, param_type)
            self.substitution = self.substitution.compose(subst)
            return True
        except UnificationError:
            pass
        
        # Try numeric coercion: Int can become Float, both can become Tensor
        if self._is_numeric_coercible(arg_type, param_type):
            return True
        
        # If param is Tensor and arg is numeric, allow coercion
        if isinstance(param_type, TensorType) and isinstance(arg_type, (IntType, FloatType)):
            return True
        
        # If param is Float and arg is Int, allow coercion
        if isinstance(param_type, FloatType) and isinstance(arg_type, IntType):
            return True
        
        # If param is a type variable, try unifying with the coerced type
        if isinstance(param_type, TypeVar):
            try:
                subst = unify(arg_type, param_type)
                self.substitution = self.substitution.compose(subst)
                return True
            except UnificationError:
                pass
        
        self.errors.append(TypeError(
            message=f"Argument type mismatch: expected {param_type}, got {arg_type}",
            location=location,
            code=ErrorCode.E301_TYPE_MISMATCH
        ))
        return False
    
    def _is_numeric_coercible(self, from_type: Type, to_type: Type) -> bool:
        """Check if from_type can be coerced to to_type."""
        # Int -> Float
        if isinstance(from_type, IntType) and isinstance(to_type, FloatType):
            return True
        # Int/Float -> Tensor
        if isinstance(from_type, (IntType, FloatType)) and isinstance(to_type, TensorType):
            return True
        # Float -> Tensor
        if isinstance(from_type, FloatType) and isinstance(to_type, TensorType):
            return True
        return False
    
    def _infer_method_call(self, call: MethodCall, env: TypeEnvironment) -> TypedExpr:
        """Infer type of a method call."""
        receiver = self._infer_expr(call.receiver, env)
        
        # For now, treat method calls as field access + call
        # In a full implementation, would look up methods on the type
        
        arg_types = [self._infer_expr(arg, env) for arg in call.args]
        
        return TypedExpr(
            type=fresh_type_var("M"),
            role=RoleInfo.computed(receiver.role, *[a.role for a in arg_types])
        )
    
    def _infer_index(self, index: Index, env: TypeEnvironment) -> TypedExpr:
        """Infer type of an indexing operation."""
        base = self._infer_expr(index.base, env)
        
        for idx in index.indices:
            idx_typed = self._infer_expr(idx, env)
        
        # For tensors, indexing reduces rank or returns scalar
        if isinstance(base.type, TensorType):
            if len(index.indices) == len(base.type.shape or []):
                # Full indexing - returns element type
                return TypedExpr(type=base.type.element_type, role=base.role)
            else:
                # Partial indexing - returns tensor with fewer dims
                return TypedExpr(type=base.type, role=base.role)
        
        return TypedExpr(type=fresh_type_var("I"), role=base.role)
    
    def _infer_field_access(self, access: FieldAccess, env: TypeEnvironment) -> TypedExpr:
        """Infer type of field access."""
        base = self._infer_expr(access.base, env)
        
        if isinstance(base.type, StructType):
            field_type = base.type.get_field(access.field)
            if field_type:
                return TypedExpr(type=field_type, role=base.role)
            else:
                self.errors.append(TypeError(
                    message=f"No field '{access.field}' on type {base.type}",
                    location=access.location,
                    code=ErrorCode.E305_MISSING_FIELD
                ))
        
        return TypedExpr(type=fresh_type_var("F"), role=base.role)
    
    def _infer_if_expr(self, expr: IfExpr, env: TypeEnvironment) -> TypedExpr:
        """Infer type of an if expression."""
        cond = self._infer_expr(expr.condition, env)
        then_branch = self._infer_expr(expr.then_expr, env)
        else_branch = self._infer_expr(expr.else_expr, env)
        
        # Check condition is bool
        try:
            subst = unify(cond.type, BoolType())
            self.substitution = self.substitution.compose(subst)
        except UnificationError:
            self.errors.append(TypeError(
                message=f"Condition must be Bool, got {cond.type}",
                location=expr.condition.location,
                code=ErrorCode.E301_TYPE_MISMATCH
            ))
        
        # Unify branches
        try:
            subst = unify(then_branch.type, else_branch.type)
            self.substitution = self.substitution.compose(subst)
        except UnificationError:
            self.errors.append(TypeError(
                message=f"Branch type mismatch: {then_branch.type} vs {else_branch.type}",
                location=expr.location,
                code=ErrorCode.E301_TYPE_MISMATCH
            ))
        
        effects = cond.effects.union(then_branch.effects).union(else_branch.effects)
        role = RoleInfo.computed(cond.role, then_branch.role, else_branch.role)
        
        return TypedExpr(
            type=apply_substitution(then_branch.type, self.substitution),
            effects=effects,
            role=role
        )
    
    def _infer_lambda(self, lam: Lambda, env: TypeEnvironment) -> TypedExpr:
        """Infer type of a lambda expression."""
        body_env = env.child_scope()
        param_types: list[Type] = []
        
        for param in lam.params:
            if param.type_annotation:
                param_type = self._resolve_type_annotation(param.type_annotation, env)
            else:
                param_type = fresh_type_var("P")
            
            param_types.append(param_type)
            body_env.define(param.name, param_type, RoleInfo.const())
        
        body_typed = self._infer_expr(lam.body, body_env)
        
        return TypedExpr(
            type=FunctionType(
                param_types=tuple(param_types),
                return_type=body_typed.type
            )
        )
    
    def _infer_tensor_literal(self, tensor: Tensor, env: TypeEnvironment) -> TypedExpr:
        """Infer type of a tensor literal."""
        if not tensor.elements:
            return TypedExpr(type=TensorType(FloatType(), shape=(ConcreteDim(0),)))
        
        elem_types = [self._infer_expr(e, env) for e in tensor.elements]
        
        # Unify all element types
        elem_type = elem_types[0].type
        for et in elem_types[1:]:
            try:
                subst = unify(elem_type, et.type)
                self.substitution = self.substitution.compose(subst)
            except UnificationError:
                pass
        
        return TypedExpr(
            type=TensorType(
                element_type=apply_substitution(elem_type, self.substitution),
                shape=(ConcreteDim(len(tensor.elements)),)
            ),
            role=RoleInfo.computed(*[e.role for e in elem_types])
        )
    
    def _infer_param_expr(self, param: Param, env: TypeEnvironment) -> TypedExpr:
        """Infer type of a param expression."""
        init = self._infer_expr(param.initializer, env)
        return TypedExpr(
            type=init.type,
            role=RoleInfo.param(),
            requires_grad=True
        )
    
    def _infer_obs_expr(self, obs: Obs, env: TypeEnvironment) -> TypedExpr:
        """Infer type of an obs expression."""
        value = self._infer_expr(obs.value, env)
        return TypedExpr(
            type=value.type,
            role=RoleInfo.obs(),
            requires_grad=False
        )
    
    def _infer_rand_expr(self, rand: RandExpr, env: TypeEnvironment) -> TypedExpr:
        """Infer type of a rand expression."""
        dist = self._infer_expr(rand.distribution, env)
        
        # Extract sample type from distribution
        if isinstance(dist.type, GenericType) and dist.type.name == "Distribution":
            sample_type = dist.type.type_args[0] if dist.type.type_args else FloatType()
        else:
            sample_type = FloatType()
        
        return TypedExpr(
            type=sample_type,
            effects=EffectSet.stochastic(),
            role=RoleInfo.computed(dist.role)
        )
    
    def _infer_observe_expr(self, obs: ObserveExpr, env: TypeEnvironment) -> TypedExpr:
        """Infer type of an observe expression."""
        value = self._infer_expr(obs.value, env)
        dist = self._infer_expr(obs.distribution, env)
        
        return TypedExpr(
            type=FloatType(),  # Log probability
            effects=EffectSet.stochastic(),
            role=RoleInfo.computed(value.role, dist.role)
        )
    
    def _infer_non_diff_block(self, block: NonDiffBlock, env: TypeEnvironment) -> TypedExpr:
        """Infer type of a non_diff block."""
        result = self._infer_block(block.body, env.child_scope())
        return TypedExpr(
            type=result.type,
            effects=result.effects.union(EffectSet.non_diff()),
            role=result.role,
            requires_grad=False
        )
    
    def _bind_pattern(
        self, 
        pattern: Pattern, 
        typ: Type, 
        role: RoleInfo,
        mutable: bool,
        env: TypeEnvironment
    ) -> None:
        """Bind a pattern to a type in the environment."""
        if isinstance(pattern, IdentifierPattern):
            env.define(pattern.name, typ, role, mutable, pattern.location)
        elif isinstance(pattern, TuplePattern):
            if isinstance(typ, TupleType):
                for pat, elem_type in zip(pattern.elements, typ.element_types):
                    self._bind_pattern(pat, elem_type, role, mutable, env)
            else:
                # Create fresh type variables for tuple elements
                for i, pat in enumerate(pattern.elements):
                    self._bind_pattern(pat, fresh_type_var(f"T{i}"), role, mutable, env)
    
    def _resolve_type_annotation(self, ann: TypeAnnotation, env: TypeEnvironment) -> Type:
        """Resolve a type annotation to a Type."""
        if isinstance(ann, SimpleType):
            # Look up type
            typ = env.lookup_type(ann.name)
            if typ:
                return typ
            
            # Primitive types
            primitives = {
                "Unit": UnitType(),
                "Bool": BoolType(),
                "Int": IntType(),
                "Float": FloatType(),
                "String": StringType(),
            }
            if ann.name in primitives:
                return primitives[ann.name]
            
            # Unknown type - create type variable
            return fresh_type_var(ann.name)
        
        elif isinstance(ann, ASTTensorType):
            elem_type = self._resolve_type_annotation(ann.element_type, env)
            
            shape: Optional[tuple[ShapeDim, ...]] = None
            if ann.shape:
                dims: list[ShapeDim] = []
                for s in ann.shape:
                    if isinstance(s, Literal) and s.kind == 'int':
                        dims.append(ConcreteDim(s.value))
                    elif isinstance(s, Identifier):
                        dims.append(SymbolicDim(s.name))
                    else:
                        dims.append(DynamicDim())
                shape = tuple(dims)
            
            return TensorType(element_type=elem_type, shape=shape)
        
        elif isinstance(ann, ASTFunctionType):
            param_types = tuple(self._resolve_type_annotation(p, env) for p in ann.param_types)
            return_type = self._resolve_type_annotation(ann.return_type, env)
            return FunctionType(param_types=param_types, return_type=return_type)
        
        elif isinstance(ann, ASTGenericType):
            type_args = tuple(self._resolve_type_annotation(a, env) for a in ann.type_args)
            return GenericType(name=ann.name, type_args=type_args)
        
        else:
            return AnyType()
