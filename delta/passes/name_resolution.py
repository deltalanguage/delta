"""
Name resolution pass.

Resolves all identifier references to their definitions,
building scope chains and detecting undefined variables.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from delta.frontend.ast import (
    Module, Statement, Expression,
    FunctionDef, StructDef, LetStmt, AssignStmt, ExprStmt, ReturnStmt,
    IfStmt, WhileStmt, ForStmt, LearnBlock, ConstraintStmt,
    ParamDecl, ObsDecl, ImportStmt,
    Block, IfExpr, BinaryOp, UnaryOp, Call, MethodCall,
    Index, FieldAccess, Tensor, Param, Obs, Identifier, Literal,
    Lambda, RandExpr, ObserveExpr, NonDiffBlock,
    IdentifierPattern, TuplePattern, Pattern,
    DefaultASTVisitor,
)
from delta.source import SourceLocation
from delta.errors import NameError, ErrorCode, ErrorCollector


@dataclass
class Symbol:
    """A symbol in the symbol table."""
    name: str
    kind: str  # 'variable', 'function', 'param', 'obs', 'type', 'import'
    location: SourceLocation
    mutable: bool = False
    exported: bool = False


@dataclass
class Scope:
    """A lexical scope."""
    parent: Optional[Scope] = None
    symbols: dict[str, Symbol] = field(default_factory=dict)
    
    def define(self, symbol: Symbol) -> Optional[Symbol]:
        """Define a symbol, returning any previous definition."""
        prev = self.symbols.get(symbol.name)
        self.symbols[symbol.name] = symbol
        return prev
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a symbol in this scope or parents."""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None
    
    def lookup_local(self, name: str) -> Optional[Symbol]:
        """Look up a symbol only in this scope."""
        return self.symbols.get(name)
    
    def child(self) -> Scope:
        """Create a child scope."""
        return Scope(parent=self)


class NameResolver(DefaultASTVisitor):
    """
    Resolves names in a Delta AST.
    
    Builds symbol tables and validates that all references
    are to defined symbols.
    """
    
    def __init__(self) -> None:
        self.scope = Scope()
        self.errors = ErrorCollector()
        self.resolutions: dict[int, Symbol] = {}  # AST node id -> symbol
        self._setup_builtins()
    
    def _setup_builtins(self) -> None:
        """Add built-in symbols."""
        builtins = [
            # Tensor creation
            "zeros", "ones", "randn", "rand", "full", "arange", "linspace", "eye",
            # Math functions
            "exp", "log", "sin", "cos", "tanh", "sigmoid", "relu", "softmax",
            "pow", "sqrt", "abs", "neg", "sign", "floor", "ceil", "clamp",
            "gelu", "leaky_relu", "elu", "selu", "swish", "mish", "log_softmax",
            # Reductions
            "sum", "mean", "max", "min", "prod", "std", "var", "norm", "argmax", "argmin",
            # Shape operations
            "matmul", "reshape", "transpose", "squeeze", "unsqueeze", "flatten",
            "permute", "expand", "repeat", "cat", "stack", "split", "chunk",
            "shape", "slice", "view",
            # Linear algebra
            "mm", "bmm", "mv", "dot", "outer", "einsum", "inv", "det", "svd", "eig", "solve",
            # Matrix operations
            "tril", "triu", "diag", "causal_mask",
            # Comparison
            "eq", "ne", "lt", "le", "gt", "ge", "isnan", "isinf", "isfinite", "where",
            # Neural network
            "embedding", "linear", "conv1d", "conv2d", "batch_norm", "layer_norm",
            "dropout", "max_pool1d", "max_pool2d", "avg_pool1d", "avg_pool2d",
            "attention",
            # Layer Classes (Constructors)
            "Linear", "Conv1d", "Conv2d", "BatchNorm1d", "BatchNorm2d", "LayerNorm",
            "Dropout", "Embedding", "LSTM", "GRU", "RNN", "MultiheadAttention",
            # Loss functions
            "mse_loss", "l1_loss", "smooth_l1_loss", "cross_entropy", "nll_loss",
            "binary_cross_entropy", "kl_div", "cosine_similarity", "triplet_margin_loss",
            # Distributions
            "Normal", "Bernoulli", "Categorical", "Uniform", "Beta", "Gamma",
            "Exponential", "Poisson", "Dirichlet", "MultivariateNormal",
            # Optimizers
            "Adam", "SGD", "AdamW", "RMSprop",
            # Types
            "Tensor", "Scalar", "Int", "Float", "Bool", "String",
            # Gradient control
            "StopGrad", "Harden",
            # Builtins
            "range",
        ]
        
        for name in builtins:
            self.scope.define(Symbol(
                name=name,
                kind="builtin",
                location=SourceLocation.builtin()
            ))
    
    def resolve(self, module: Module) -> dict[int, Symbol]:
        """Resolve all names in a module."""
        self.visit_module(module)
        return self.resolutions
    
    def visit_module(self, node: Module) -> None:
        # First pass: collect all top-level definitions
        for item in node.items:
            if isinstance(item, FunctionDef):
                self._define_function(item)
            elif isinstance(item, StructDef):
                self._define_struct(item)
            elif isinstance(item, ParamDecl):
                self._define_param(item)
            elif isinstance(item, ObsDecl):
                self._define_obs(item)
            elif isinstance(item, ImportStmt):
                self._process_import(item)
        
        # Second pass: resolve bodies
        for item in node.items:
            self._resolve_statement(item)
    
    def _define_function(self, func: FunctionDef) -> None:
        """Define a function."""
        prev = self.scope.define(Symbol(
            name=func.name,
            kind="function",
            location=func.location,
            exported=func.is_pub
        ))
        
        if prev and prev.kind != "builtin":
            self.errors.add(NameError(
                message=f"Redefinition of '{func.name}'",
                location=func.location,
                code=ErrorCode.E204_DUPLICATE_DEFINITION
            ).with_related(f"Previously defined here", prev.location))
    
    def _define_struct(self, struct: StructDef) -> None:
        """Define a struct type."""
        prev = self.scope.define(Symbol(
            name=struct.name,
            kind="type",
            location=struct.location,
            exported=struct.is_pub
        ))
        
        if prev and prev.kind != "builtin":
            self.errors.add(NameError(
                message=f"Redefinition of type '{struct.name}'",
                location=struct.location,
                code=ErrorCode.E204_DUPLICATE_DEFINITION
            ))
    
    def _define_param(self, decl: ParamDecl) -> None:
        """Define a param."""
        self.scope.define(Symbol(
            name=decl.name,
            kind="param",
            location=decl.location
        ))
    
    def _define_obs(self, decl: ObsDecl) -> None:
        """Define an obs."""
        self.scope.define(Symbol(
            name=decl.name,
            kind="obs",
            location=decl.location
        ))
    
    def _process_import(self, imp: ImportStmt) -> None:
        """Process an import statement."""
        if imp.items:
            for name, alias in imp.items:
                imported_name = alias or name
                self.scope.define(Symbol(
                    name=imported_name,
                    kind="import",
                    location=imp.location
                ))
        elif imp.alias:
            self.scope.define(Symbol(
                name=imp.alias,
                kind="import",
                location=imp.location
            ))
        else:
            # Import whole module
            module_name = imp.module_path[-1]
            self.scope.define(Symbol(
                name=module_name,
                kind="import",
                location=imp.location
            ))
    
    def _resolve_statement(self, stmt: Statement) -> None:
        """Resolve names in a statement."""
        if isinstance(stmt, FunctionDef):
            self._resolve_function(stmt)
        elif isinstance(stmt, LetStmt):
            self._resolve_let(stmt)
        elif isinstance(stmt, AssignStmt):
            self._resolve_expr(stmt.target)
            self._resolve_expr(stmt.value)
        elif isinstance(stmt, ExprStmt):
            self._resolve_expr(stmt.expr)
        elif isinstance(stmt, ReturnStmt):
            if stmt.value:
                self._resolve_expr(stmt.value)
        elif isinstance(stmt, IfStmt):
            self._resolve_if_stmt(stmt)
        elif isinstance(stmt, WhileStmt):
            self._resolve_expr(stmt.condition)
            self._resolve_block(stmt.body)
        elif isinstance(stmt, ForStmt):
            self._resolve_for(stmt)
        elif isinstance(stmt, LearnBlock):
            self._resolve_learn_block(stmt)
        elif isinstance(stmt, ConstraintStmt):
            self._resolve_expr(stmt.expr)
            if stmt.weight:
                self._resolve_expr(stmt.weight)
            if stmt.slack:
                self._resolve_expr(stmt.slack)
        elif isinstance(stmt, ParamDecl):
            if stmt.initializer:
                self._resolve_expr(stmt.initializer)
        # ObsDecl, ImportStmt, StructDef don't need body resolution
    
    def _resolve_function(self, func: FunctionDef) -> None:
        """Resolve names in a function."""
        old_scope = self.scope
        self.scope = self.scope.child()
        
        # Add parameters to scope
        for param in func.params:
            kind = "param" if param.role == "param" else "obs" if param.role == "obs" else "variable"
            self.scope.define(Symbol(
                name=param.name,
                kind=kind,
                location=param.location
            ))
        
        # Resolve body
        self._resolve_block(func.body)
        
        self.scope = old_scope
    
    def _resolve_let(self, stmt: LetStmt) -> None:
        """Resolve a let statement."""
        # Resolve value first
        self._resolve_expr(stmt.value)
        
        # Then bind pattern
        self._bind_pattern(stmt.pattern, stmt.mutable)
    
    def _resolve_for(self, stmt: ForStmt) -> None:
        """Resolve a for statement."""
        self._resolve_expr(stmt.iterable)
        
        old_scope = self.scope
        self.scope = self.scope.child()
        
        self._bind_pattern(stmt.pattern, mutable=False)
        self._resolve_block(stmt.body)
        
        self.scope = old_scope
    
    def _resolve_if_stmt(self, stmt: IfStmt) -> None:
        """Resolve an if statement."""
        self._resolve_expr(stmt.condition)
        self._resolve_block(stmt.then_block)
        
        if stmt.else_block:
            if isinstance(stmt.else_block, Block):
                self._resolve_block(stmt.else_block)
            else:
                self._resolve_if_stmt(stmt.else_block)
    
    def _resolve_learn_block(self, stmt: LearnBlock) -> None:
        """Resolve a learn block."""
        old_scope = self.scope
        self.scope = self.scope.child()
        
        self._resolve_block(stmt.body)
        
        if stmt.optimizer:
            self._resolve_expr(stmt.optimizer)
        if stmt.epochs:
            self._resolve_expr(stmt.epochs)
        if stmt.batch_size:
            self._resolve_expr(stmt.batch_size)
        
        self.scope = old_scope
    
    def _resolve_block(self, block: Block) -> None:
        """Resolve names in a block."""
        old_scope = self.scope
        self.scope = self.scope.child()
        
        for stmt in block.statements:
            self._resolve_statement(stmt)
        
        if block.result:
            self._resolve_expr(block.result)
        
        self.scope = old_scope
    
    def _resolve_expr(self, expr: Expression) -> None:
        """Resolve names in an expression."""
        if isinstance(expr, Identifier):
            symbol = self.scope.lookup(expr.name)
            if symbol:
                self.resolutions[id(expr)] = symbol
            else:
                self.errors.add(NameError(
                    message=f"Undefined variable: '{expr.name}'",
                    location=expr.location,
                    code=ErrorCode.E201_UNDEFINED_VARIABLE
                ))
        
        elif isinstance(expr, BinaryOp):
            self._resolve_expr(expr.left)
            self._resolve_expr(expr.right)
        
        elif isinstance(expr, UnaryOp):
            self._resolve_expr(expr.operand)
        
        elif isinstance(expr, Call):
            self._resolve_expr(expr.func)
            for arg in expr.args:
                self._resolve_expr(arg)
            for _, arg in expr.kwargs:
                self._resolve_expr(arg)
        
        elif isinstance(expr, MethodCall):
            self._resolve_expr(expr.receiver)
            for arg in expr.args:
                self._resolve_expr(arg)
            for _, arg in expr.kwargs:
                self._resolve_expr(arg)
        
        elif isinstance(expr, Index):
            self._resolve_expr(expr.base)
            for idx in expr.indices:
                self._resolve_expr(idx)
        
        elif isinstance(expr, FieldAccess):
            self._resolve_expr(expr.base)
        
        elif isinstance(expr, IfExpr):
            self._resolve_expr(expr.condition)
            self._resolve_expr(expr.then_expr)
            self._resolve_expr(expr.else_expr)
            if expr.temperature:
                self._resolve_expr(expr.temperature)
        
        elif isinstance(expr, Block):
            self._resolve_block(expr)
        
        elif isinstance(expr, Lambda):
            old_scope = self.scope
            self.scope = self.scope.child()
            
            for param in expr.params:
                self.scope.define(Symbol(
                    name=param.name,
                    kind="variable",
                    location=param.location
                ))
            
            self._resolve_expr(expr.body)
            self.scope = old_scope
        
        elif isinstance(expr, Tensor):
            for elem in expr.elements:
                self._resolve_expr(elem)
        
        elif isinstance(expr, Param):
            self._resolve_expr(expr.initializer)
        
        elif isinstance(expr, Obs):
            self._resolve_expr(expr.value)
        
        elif isinstance(expr, RandExpr):
            self._resolve_expr(expr.distribution)
        
        elif isinstance(expr, ObserveExpr):
            self._resolve_expr(expr.value)
            self._resolve_expr(expr.distribution)
        
        elif isinstance(expr, NonDiffBlock):
            self._resolve_block(expr.body)
    
    def _bind_pattern(self, pattern: Pattern, mutable: bool) -> None:
        """Bind a pattern to the current scope."""
        if isinstance(pattern, IdentifierPattern):
            self.scope.define(Symbol(
                name=pattern.name,
                kind="variable",
                location=pattern.location,
                mutable=mutable or pattern.mutable
            ))
        elif isinstance(pattern, TuplePattern):
            for elem in pattern.elements:
                self._bind_pattern(elem, mutable)


def resolve_names(module: Module) -> tuple[dict[int, Symbol], ErrorCollector]:
    """Convenience function to resolve names."""
    resolver = NameResolver()
    resolutions = resolver.resolve(module)
    return resolutions, resolver.errors
