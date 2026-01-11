"""
Abstract Syntax Tree definitions for Delta.

The AST represents the syntactic structure of Delta programs before
type checking and lowering to SIR. Every node carries source location
information for accurate error reporting.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, TypeVar, Generic
from enum import Enum, auto

from delta.source import SourceLocation


# AST Node Base Classes


class ASTNode(ABC):
    """
    Base class for all AST nodes.
    
    Every node has a source location and optional type annotation
    that gets filled in during type inference.
    """
    location: SourceLocation
    
    @abstractmethod
    def accept(self, visitor: ASTVisitor[T]) -> T:
        """Accept a visitor for traversal."""
        pass
    
    def children(self) -> list[ASTNode]:
        """Get all child nodes for traversal."""
        return []


T = TypeVar('T')


class ASTVisitor(ABC, Generic[T]):
    """Visitor interface for AST traversal."""
    
    def visit(self, node: ASTNode) -> T:
        """Visit any AST node."""
        return node.accept(self)
    
    @abstractmethod
    def visit_module(self, node: Module) -> T: ...
    @abstractmethod
    def visit_function_def(self, node: FunctionDef) -> T: ...
    @abstractmethod
    def visit_struct_def(self, node: StructDef) -> T: ...
    @abstractmethod
    def visit_let_stmt(self, node: LetStmt) -> T: ...
    @abstractmethod
    def visit_assign_stmt(self, node: AssignStmt) -> T: ...
    @abstractmethod
    def visit_expr_stmt(self, node: ExprStmt) -> T: ...
    @abstractmethod
    def visit_return_stmt(self, node: ReturnStmt) -> T: ...
    @abstractmethod
    def visit_if_stmt(self, node: IfStmt) -> T: ...
    @abstractmethod
    def visit_while_stmt(self, node: WhileStmt) -> T: ...
    @abstractmethod
    def visit_for_stmt(self, node: ForStmt) -> T: ...
    @abstractmethod
    def visit_learn_block(self, node: LearnBlock) -> T: ...
    @abstractmethod
    def visit_constraint_stmt(self, node: ConstraintStmt) -> T: ...
    @abstractmethod
    def visit_param_decl(self, node: ParamDecl) -> T: ...
    @abstractmethod
    def visit_obs_decl(self, node: ObsDecl) -> T: ...
    @abstractmethod
    def visit_import_stmt(self, node: ImportStmt) -> T: ...
    @abstractmethod
    def visit_if_expr(self, node: IfExpr) -> T: ...
    @abstractmethod
    def visit_binary_op(self, node: BinaryOp) -> T: ...
    @abstractmethod
    def visit_unary_op(self, node: UnaryOp) -> T: ...
    @abstractmethod
    def visit_call(self, node: Call) -> T: ...
    @abstractmethod
    def visit_method_call(self, node: MethodCall) -> T: ...
    @abstractmethod
    def visit_index(self, node: Index) -> T: ...
    @abstractmethod
    def visit_field_access(self, node: FieldAccess) -> T: ...
    @abstractmethod
    def visit_tensor(self, node: Tensor) -> T: ...
    @abstractmethod
    def visit_param(self, node: Param) -> T: ...
    @abstractmethod
    def visit_obs(self, node: Obs) -> T: ...
    @abstractmethod
    def visit_identifier(self, node: Identifier) -> T: ...
    @abstractmethod
    def visit_literal(self, node: Literal) -> T: ...
    @abstractmethod
    def visit_lambda(self, node: Lambda) -> T: ...
    @abstractmethod
    def visit_rand_expr(self, node: RandExpr) -> T: ...
    @abstractmethod
    def visit_observe_expr(self, node: ObserveExpr) -> T: ...
    @abstractmethod
    def visit_non_diff_block(self, node: NonDiffBlock) -> T: ...
    @abstractmethod
    def visit_block(self, node: Block) -> T: ...


# Operator Enums


class BinaryOperator(Enum):
    """Binary operators in Delta."""
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    POW = auto()
    MATMUL = auto()
    
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    
    AND = auto()
    OR = auto()
    
    BIT_AND = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    OBSERVE = auto()  # ~ operator (x ~ dist)


class UnaryOperator(Enum):
    """Unary operators in Delta."""
    NEG = auto()
    NOT = auto()
    BIT_NOT = auto()


class ConstraintKind(Enum):
    """Kinds of constraints in Delta."""
    REQUIRE = auto()   # Hard constraint
    PREFER = auto()    # Soft constraint
    OBSERVE = auto()   # Probabilistic observation


# Type Annotations


@dataclass
class TypeAnnotation:
    """Type annotation in source code."""
    location: SourceLocation
    
    def accept(self, visitor: TypeAnnotationVisitor[T]) -> T:
        pass


class TypeAnnotationVisitor(ABC, Generic[T]):
    """Visitor for type annotations."""
    pass


@dataclass
class SimpleType(TypeAnnotation):
    """Simple named type like Int, Float, Bool."""
    name: str


@dataclass
class TensorType(TypeAnnotation):
    """Tensor type with element type and shape."""
    element_type: TypeAnnotation
    shape: Optional[list[Expression]] = None
    

@dataclass
class FunctionType(TypeAnnotation):
    """Function type annotation."""
    param_types: list[TypeAnnotation]
    return_type: TypeAnnotation


@dataclass
class GenericType(TypeAnnotation):
    """Generic type with type parameters."""
    name: str
    type_args: list[TypeAnnotation]


# Patterns (for let bindings and function parameters)


@dataclass
class Pattern:
    """Base class for patterns."""
    location: SourceLocation


@dataclass  
class IdentifierPattern(Pattern):
    """Simple identifier pattern."""
    name: str
    type_annotation: Optional[TypeAnnotation] = None
    mutable: bool = False


@dataclass
class TuplePattern(Pattern):
    """Tuple destructuring pattern."""
    elements: list[Pattern]


@dataclass
class StructPattern(Pattern):
    """Struct destructuring pattern."""
    type_name: str
    fields: list[tuple[str, Pattern]]


# Top-Level Items


@dataclass
class Module(ASTNode):
    """A Delta module (source file)."""
    location: SourceLocation
    name: str
    items: list[Statement]
    doc: Optional[str] = None
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_module(self)
    
    def children(self) -> list[ASTNode]:
        return list(self.items)


@dataclass
class FunctionDef(ASTNode):
    """Function definition."""
    location: SourceLocation
    name: str
    params: list[Parameter]
    return_type: Optional[TypeAnnotation]
    body: Block
    type_params: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    doc: Optional[str] = None
    is_pub: bool = False
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_function_def(self)
    
    def children(self) -> list[ASTNode]:
        return [self.body]


@dataclass
class Parameter:
    """Function parameter."""
    location: SourceLocation
    name: str
    type_annotation: Optional[TypeAnnotation]
    default: Optional[Expression] = None
    role: Optional[str] = None  # param, obs, or None


@dataclass
class StructDef(ASTNode):
    """Struct type definition."""
    location: SourceLocation
    name: str
    fields: list[StructField]
    type_params: list[str] = field(default_factory=list)
    doc: Optional[str] = None
    is_pub: bool = False
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_struct_def(self)


@dataclass
class StructField:
    """A field in a struct definition."""
    location: SourceLocation
    name: str
    type_annotation: TypeAnnotation
    default: Optional[Expression] = None


# Statements


@dataclass
class Statement(ASTNode):
    """Base class for statements."""
    pass


@dataclass
class LetStmt(Statement):
    """Variable binding statement."""
    location: SourceLocation
    pattern: Pattern
    type_annotation: Optional[TypeAnnotation]
    value: Expression
    mutable: bool = False
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_let_stmt(self)
    
    def children(self) -> list[ASTNode]:
        return [self.value]


@dataclass
class AssignStmt(Statement):
    """Assignment statement."""
    location: SourceLocation
    target: Expression
    value: Expression
    op: Optional[BinaryOperator] = None  # For +=, -=, etc.
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_assign_stmt(self)
    
    def children(self) -> list[ASTNode]:
        return [self.target, self.value]


@dataclass
class ExprStmt(Statement):
    """Expression statement."""
    location: SourceLocation
    expr: Expression
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_expr_stmt(self)
    
    def children(self) -> list[ASTNode]:
        return [self.expr]


@dataclass
class ReturnStmt(Statement):
    """Return statement."""
    location: SourceLocation
    value: Optional[Expression]
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_return_stmt(self)
    
    def children(self) -> list[ASTNode]:
        return [self.value] if self.value else []


@dataclass
class IfStmt(Statement):
    """If statement (statement form, not expression)."""
    location: SourceLocation
    condition: Expression
    then_block: Block
    else_block: Optional[Block | IfStmt] = None
    temperature: Optional[Expression] = None
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_if_stmt(self)
    
    def children(self) -> list[ASTNode]:
        result: list[ASTNode] = [self.condition, self.then_block]
        if self.else_block:
            result.append(self.else_block)
        if self.temperature:
            result.append(self.temperature)
        return result


@dataclass
class WhileStmt(Statement):
    """While loop statement."""
    location: SourceLocation
    condition: Expression
    body: Block
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_while_stmt(self)
    
    def children(self) -> list[ASTNode]:
        return [self.condition, self.body]


@dataclass
class ForStmt(Statement):
    """For loop statement."""
    location: SourceLocation
    pattern: Pattern
    iterable: Expression
    body: Block
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_for_stmt(self)
    
    def children(self) -> list[ASTNode]:
        return [self.iterable, self.body]


@dataclass
class LearnBlock(Statement):
    """
    Learning block - the core training construct.
    
    ```delta
    learn {
        param w: Tensor[32, 10]
        obs x: Tensor[batch, 32]
        obs y: Tensor[batch, 10]
        
        let pred = x @ w
        require pred ~= y
    } with optimizer = Adam(lr=0.001)
    ```
    """
    location: SourceLocation
    body: Block
    optimizer: Optional[Expression] = None
    epochs: Optional[Expression] = None
    batch_size: Optional[Expression] = None
    mode: str = "train"  # train, infer, analyze
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_learn_block(self)
    
    def children(self) -> list[ASTNode]:
        result: list[ASTNode] = [self.body]
        if self.optimizer:
            result.append(self.optimizer)
        if self.epochs:
            result.append(self.epochs)
        if self.batch_size:
            result.append(self.batch_size)
        return result


@dataclass
class ConstraintStmt(Statement):
    """
    Constraint statement.
    
    ```delta
    require loss < 0.1              // Hard constraint
    prefer accuracy > 0.9 weight 2  // Soft constraint
    ```
    """
    location: SourceLocation
    kind: ConstraintKind
    expr: Expression
    weight: Optional[Expression] = None
    slack: Optional[Expression] = None
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_constraint_stmt(self)
    
    def children(self) -> list[ASTNode]:
        result: list[ASTNode] = [self.expr]
        if self.weight:
            result.append(self.weight)
        if self.slack:
            result.append(self.slack)
        return result


@dataclass
class ParamDecl(Statement):
    """
    Parameter declaration.
    
    ```delta
    param w: Tensor[32, 10] = zeros()
    ```
    """
    location: SourceLocation
    name: str
    type_annotation: Optional[TypeAnnotation]
    initializer: Optional[Expression] = None
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_param_decl(self)
    
    def children(self) -> list[ASTNode]:
        return [self.initializer] if self.initializer else []


@dataclass
class ObsDecl(Statement):
    """
    Observation declaration.
    
    ```delta
    obs x: Tensor[batch, 32]
    ```
    """
    location: SourceLocation
    name: str
    type_annotation: Optional[TypeAnnotation]
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_obs_decl(self)


@dataclass
class ImportStmt(Statement):
    """Import statement."""
    location: SourceLocation
    module_path: list[str]
    items: Optional[list[tuple[str, Optional[str]]]] = None  # (name, alias)
    alias: Optional[str] = None  # For 'import x as y'
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_import_stmt(self)


# Expressions


@dataclass  
class Expression(ASTNode):
    """Base class for expressions."""
    pass


@dataclass
class Block(Expression):
    """Block expression containing statements."""
    location: SourceLocation
    statements: list[Statement]
    result: Optional[Expression] = None
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_block(self)
    
    def children(self) -> list[ASTNode]:
        result: list[ASTNode] = list(self.statements)
        if self.result:
            result.append(self.result)
        return result


@dataclass
class IfExpr(Expression):
    """
    Conditional expression with optional temperature.
    
    ```delta
    if x > 0 temperature 0.1 { a } else { b }
    ```
    """
    location: SourceLocation
    condition: Expression
    then_expr: Expression
    else_expr: Expression
    temperature: Optional[Expression] = None
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_if_expr(self)
    
    def children(self) -> list[ASTNode]:
        result: list[ASTNode] = [self.condition, self.then_expr, self.else_expr]
        if self.temperature:
            result.append(self.temperature)
        return result


@dataclass
class BinaryOp(Expression):
    """Binary operation."""
    location: SourceLocation
    op: BinaryOperator
    left: Expression
    right: Expression
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_binary_op(self)
    
    def children(self) -> list[ASTNode]:
        return [self.left, self.right]


@dataclass
class UnaryOp(Expression):
    """Unary operation."""
    location: SourceLocation
    op: UnaryOperator
    operand: Expression
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_unary_op(self)
    
    def children(self) -> list[ASTNode]:
        return [self.operand]


@dataclass
class Call(Expression):
    """Function call."""
    location: SourceLocation
    func: Expression
    args: list[Expression]
    kwargs: list[tuple[str, Expression]] = field(default_factory=list)
    type_args: list[TypeAnnotation] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_call(self)
    
    def children(self) -> list[ASTNode]:
        result: list[ASTNode] = [self.func] + self.args
        result.extend(v for _, v in self.kwargs)
        return result


@dataclass
class MethodCall(Expression):
    """Method call."""
    location: SourceLocation
    receiver: Expression
    method: str
    args: list[Expression]
    kwargs: list[tuple[str, Expression]] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_method_call(self)
    
    def children(self) -> list[ASTNode]:
        result: list[ASTNode] = [self.receiver] + self.args
        result.extend(v for _, v in self.kwargs)
        return result


@dataclass
class Index(Expression):
    """Indexing operation."""
    location: SourceLocation
    base: Expression
    indices: list[Expression]
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_index(self)
    
    def children(self) -> list[ASTNode]:
        return [self.base] + self.indices


@dataclass
class FieldAccess(Expression):
    """Field access."""
    location: SourceLocation
    base: Expression
    field: str
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_field_access(self)
    
    def children(self) -> list[ASTNode]:
        return [self.base]


@dataclass
class Tensor(Expression):
    """Tensor literal."""
    location: SourceLocation
    elements: list[Expression]
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_tensor(self)
    
    def children(self) -> list[ASTNode]:
        return list(self.elements)


@dataclass
class Param(Expression):
    """
    Inline parameter expression.
    
    ```delta
    param(zeros(32, 10))
    ```
    """
    location: SourceLocation
    initializer: Expression
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_param(self)
    
    def children(self) -> list[ASTNode]:
        return [self.initializer]


@dataclass
class Obs(Expression):
    """
    Inline observation expression.
    
    ```delta
    obs(data)
    ```
    """
    location: SourceLocation
    value: Expression
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_obs(self)
    
    def children(self) -> list[ASTNode]:
        return [self.value]


@dataclass
class Identifier(Expression):
    """Identifier reference."""
    location: SourceLocation
    name: str
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_identifier(self)


@dataclass
class Literal(Expression):
    """Literal value."""
    location: SourceLocation
    value: Any
    kind: str  # 'int', 'float', 'string', 'bool', 'none'
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_literal(self)


@dataclass
class Lambda(Expression):
    """Lambda expression."""
    location: SourceLocation
    params: list[Parameter]
    body: Expression
    return_type: Optional[TypeAnnotation] = None
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_lambda(self)
    
    def children(self) -> list[ASTNode]:
        return [self.body]


@dataclass
class RandExpr(Expression):
    """
    Random variable expression.
    
    ```delta
    rand Normal(0, 1)
    ```
    """
    location: SourceLocation
    distribution: Expression
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_rand_expr(self)
    
    def children(self) -> list[ASTNode]:
        return [self.distribution]


@dataclass
class ObserveExpr(Expression):
    """
    Probabilistic observation.
    
    ```delta
    observe(y, Normal(pred, sigma))
    ```
    """
    location: SourceLocation
    value: Expression
    distribution: Expression
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_observe_expr(self)
    
    def children(self) -> list[ASTNode]:
        return [self.value, self.distribution]


@dataclass
class NonDiffBlock(Expression):
    """
    Non-differentiable block.
    
    ```delta
    non_diff {
        // Hard decisions here
    }
    ```
    """
    location: SourceLocation
    body: Block
    
    def accept(self, visitor: ASTVisitor[T]) -> T:
        return visitor.visit_non_diff_block(self)
    
    def children(self) -> list[ASTNode]:
        return [self.body]


# AST Utilities


class DefaultASTVisitor(ASTVisitor[None]):
    """Default visitor that traverses all nodes."""
    
    def visit_children(self, node: ASTNode) -> None:
        for child in node.children():
            self.visit(child)
    
    def visit_module(self, node: Module) -> None:
        self.visit_children(node)
    
    def visit_function_def(self, node: FunctionDef) -> None:
        self.visit_children(node)
    
    def visit_struct_def(self, node: StructDef) -> None:
        self.visit_children(node)
    
    def visit_let_stmt(self, node: LetStmt) -> None:
        self.visit_children(node)
    
    def visit_assign_stmt(self, node: AssignStmt) -> None:
        self.visit_children(node)
    
    def visit_expr_stmt(self, node: ExprStmt) -> None:
        self.visit_children(node)
    
    def visit_return_stmt(self, node: ReturnStmt) -> None:
        self.visit_children(node)
    
    def visit_if_stmt(self, node: IfStmt) -> None:
        self.visit_children(node)
    
    def visit_while_stmt(self, node: WhileStmt) -> None:
        self.visit_children(node)
    
    def visit_for_stmt(self, node: ForStmt) -> None:
        self.visit_children(node)
    
    def visit_learn_block(self, node: LearnBlock) -> None:
        self.visit_children(node)
    
    def visit_constraint_stmt(self, node: ConstraintStmt) -> None:
        self.visit_children(node)
    
    def visit_param_decl(self, node: ParamDecl) -> None:
        self.visit_children(node)
    
    def visit_obs_decl(self, node: ObsDecl) -> None:
        self.visit_children(node)
    
    def visit_import_stmt(self, node: ImportStmt) -> None:
        self.visit_children(node)
    
    def visit_if_expr(self, node: IfExpr) -> None:
        self.visit_children(node)
    
    def visit_binary_op(self, node: BinaryOp) -> None:
        self.visit_children(node)
    
    def visit_unary_op(self, node: UnaryOp) -> None:
        self.visit_children(node)
    
    def visit_call(self, node: Call) -> None:
        self.visit_children(node)
    
    def visit_method_call(self, node: MethodCall) -> None:
        self.visit_children(node)
    
    def visit_index(self, node: Index) -> None:
        self.visit_children(node)
    
    def visit_field_access(self, node: FieldAccess) -> None:
        self.visit_children(node)
    
    def visit_tensor(self, node: Tensor) -> None:
        self.visit_children(node)
    
    def visit_param(self, node: Param) -> None:
        self.visit_children(node)
    
    def visit_obs(self, node: Obs) -> None:
        self.visit_children(node)
    
    def visit_identifier(self, node: Identifier) -> None:
        pass
    
    def visit_literal(self, node: Literal) -> None:
        pass
    
    def visit_lambda(self, node: Lambda) -> None:
        self.visit_children(node)
    
    def visit_rand_expr(self, node: RandExpr) -> None:
        self.visit_children(node)
    
    def visit_observe_expr(self, node: ObserveExpr) -> None:
        self.visit_children(node)
    
    def visit_non_diff_block(self, node: NonDiffBlock) -> None:
        self.visit_children(node)
    
    def visit_block(self, node: Block) -> None:
        self.visit_children(node)


def walk_ast(node: ASTNode) -> list[ASTNode]:
    """
    Walk the AST and collect all nodes in pre-order.
    """
    result: list[ASTNode] = [node]
    for child in node.children():
        result.extend(walk_ast(child))
    return result
