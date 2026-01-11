"""
Desugaring pass for Delta AST.

Converts syntactic sugar into core language constructs:
- if expressions with temperature -> Mix(Gate(...), then, else)
- for loops -> while loops with iterators
- compound assignments -> binary ops + assignment
- method syntax -> function calls
"""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

from delta.frontend.ast import (
    ASTNode, Module, Statement, Expression,
    FunctionDef, StructDef, LetStmt, AssignStmt, ExprStmt, ReturnStmt,
    IfStmt, WhileStmt, ForStmt, LearnBlock, ConstraintStmt,
    ParamDecl, ObsDecl, ImportStmt,
    Block, IfExpr, BinaryOp, UnaryOp, Call, MethodCall,
    Index, FieldAccess, Tensor, Param, Obs, Identifier, Literal,
    Lambda, RandExpr, ObserveExpr, NonDiffBlock,
    BinaryOperator, UnaryOperator, ConstraintKind,
    ASTVisitor, DefaultASTVisitor, Parameter,
    IdentifierPattern,
    T,
)
from delta.source import SourceLocation


class Desugarer(DefaultASTVisitor):
    """
    Desugars AST constructs into simpler forms.
    
    Key transformations:
    1. if expr with temperature -> GatedIf (for later SIR lowering)
    2. Compound assignments (+=, etc.) -> expanded form
    3. for loops -> while with iterator
    """
    
    def __init__(self) -> None:
        self._temp_counter = 0
    
    def fresh_name(self, prefix: str = "_tmp") -> str:
        """Generate a fresh temporary name."""
        self._temp_counter += 1
        return f"{prefix}_{self._temp_counter}"
    
    def desugar(self, module: Module) -> Module:
        """Desugar an entire module."""
        return self.desugar_module(module)
    
    def desugar_module(self, node: Module) -> Module:
        """Desugar a module."""
        return Module(
            location=node.location,
            name=node.name,
            items=[self.desugar_statement(item) for item in node.items],
            doc=node.doc
        )
    
    def desugar_statement(self, stmt: Statement) -> Statement:
        """Desugar a statement."""
        if isinstance(stmt, FunctionDef):
            return self.desugar_function_def(stmt)
        elif isinstance(stmt, LetStmt):
            return self.desugar_let_stmt(stmt)
        elif isinstance(stmt, AssignStmt):
            return self.desugar_assign_stmt(stmt)
        elif isinstance(stmt, ExprStmt):
            return ExprStmt(
                location=stmt.location,
                expr=self.desugar_expr(stmt.expr)
            )
        elif isinstance(stmt, ReturnStmt):
            return ReturnStmt(
                location=stmt.location,
                value=self.desugar_expr(stmt.value) if stmt.value else None
            )
        elif isinstance(stmt, IfStmt):
            return self.desugar_if_stmt(stmt)
        elif isinstance(stmt, WhileStmt):
            return self.desugar_while_stmt(stmt)
        elif isinstance(stmt, ForStmt):
            return self.desugar_for_stmt(stmt)
        elif isinstance(stmt, LearnBlock):
            return self.desugar_learn_block(stmt)
        elif isinstance(stmt, ConstraintStmt):
            return self.desugar_constraint_stmt(stmt)
        elif isinstance(stmt, ParamDecl):
            return ParamDecl(
                location=stmt.location,
                name=stmt.name,
                type_annotation=stmt.type_annotation,
                initializer=self.desugar_expr(stmt.initializer) if stmt.initializer else None
            )
        elif isinstance(stmt, ObsDecl):
            return stmt
        elif isinstance(stmt, ImportStmt):
            return stmt
        elif isinstance(stmt, StructDef):
            return stmt
        else:
            return stmt
    
    def desugar_function_def(self, node: FunctionDef) -> FunctionDef:
        """Desugar a function definition."""
        return FunctionDef(
            location=node.location,
            name=node.name,
            params=node.params,
            return_type=node.return_type,
            body=self.desugar_block(node.body),
            type_params=node.type_params,
            effects=node.effects,
            doc=node.doc,
            is_pub=node.is_pub
        )
    
    def desugar_let_stmt(self, node: LetStmt) -> LetStmt:
        """Desugar a let statement."""
        return LetStmt(
            location=node.location,
            pattern=node.pattern,
            type_annotation=node.type_annotation,
            value=self.desugar_expr(node.value),
            mutable=node.mutable
        )
    
    def desugar_assign_stmt(self, node: AssignStmt) -> AssignStmt:
        """
        Desugar compound assignment.
        
        x += y  ->  x = x + y
        """
        if node.op is not None:
            # Compound assignment: x op= y -> x = x op y
            expanded_value = BinaryOp(
                location=node.location,
                op=node.op,
                left=node.target,
                right=node.value
            )
            return AssignStmt(
                location=node.location,
                target=self.desugar_expr(node.target),
                value=self.desugar_expr(expanded_value),
                op=None
            )
        return AssignStmt(
            location=node.location,
            target=self.desugar_expr(node.target),
            value=self.desugar_expr(node.value),
            op=None
        )
    
    def desugar_if_stmt(self, node: IfStmt) -> IfStmt:
        """Desugar an if statement."""
        else_block: Optional[Block | IfStmt] = None
        if isinstance(node.else_block, Block):
            else_block = self.desugar_block(node.else_block)
        elif isinstance(node.else_block, IfStmt):
            else_block = self.desugar_if_stmt(node.else_block)
        
        return IfStmt(
            location=node.location,
            condition=self.desugar_expr(node.condition),
            then_block=self.desugar_block(node.then_block),
            else_block=else_block,
            temperature=self.desugar_expr(node.temperature) if node.temperature else None
        )
    
    def desugar_while_stmt(self, node: WhileStmt) -> WhileStmt:
        """Desugar a while statement."""
        return WhileStmt(
            location=node.location,
            condition=self.desugar_expr(node.condition),
            body=self.desugar_block(node.body)
        )
    
    def desugar_for_stmt(self, node: ForStmt) -> ForStmt:
        """
        Desugar a for loop.
        
        For now, keep as ForStmt - full desugaring to while + iterator
        would require more complex transformations.
        """
        return ForStmt(
            location=node.location,
            pattern=node.pattern,
            iterable=self.desugar_expr(node.iterable),
            body=self.desugar_block(node.body)
        )
    
    def desugar_learn_block(self, node: LearnBlock) -> LearnBlock:
        """Desugar a learn block."""
        return LearnBlock(
            location=node.location,
            body=self.desugar_block(node.body),
            optimizer=self.desugar_expr(node.optimizer) if node.optimizer else None,
            epochs=self.desugar_expr(node.epochs) if node.epochs else None,
            batch_size=self.desugar_expr(node.batch_size) if node.batch_size else None,
            mode=node.mode
        )
    
    def desugar_constraint_stmt(self, node: ConstraintStmt) -> ConstraintStmt:
        """Desugar a constraint statement."""
        return ConstraintStmt(
            location=node.location,
            kind=node.kind,
            expr=self.desugar_expr(node.expr),
            weight=self.desugar_expr(node.weight) if node.weight else None,
            slack=self.desugar_expr(node.slack) if node.slack else None
        )
    
    def desugar_block(self, node: Block) -> Block:
        """Desugar a block."""
        return Block(
            location=node.location,
            statements=[self.desugar_statement(s) for s in node.statements],
            result=self.desugar_expr(node.result) if node.result else None
        )
    
    def desugar_expr(self, expr: Optional[Expression]) -> Optional[Expression]:
        """Desugar an expression."""
        if expr is None:
            return None
        
        if isinstance(expr, Block):
            return self.desugar_block(expr)
        elif isinstance(expr, IfExpr):
            return self.desugar_if_expr(expr)
        elif isinstance(expr, BinaryOp):
            return BinaryOp(
                location=expr.location,
                op=expr.op,
                left=self.desugar_expr(expr.left),
                right=self.desugar_expr(expr.right)
            )
        elif isinstance(expr, UnaryOp):
            return UnaryOp(
                location=expr.location,
                op=expr.op,
                operand=self.desugar_expr(expr.operand)
            )
        elif isinstance(expr, Call):
            return Call(
                location=expr.location,
                func=self.desugar_expr(expr.func),
                args=[self.desugar_expr(a) for a in expr.args],
                kwargs=[(k, self.desugar_expr(v)) for k, v in expr.kwargs],
                type_args=expr.type_args
            )
        elif isinstance(expr, MethodCall):
            # Desugar method call: x.method(args) -> method(x, args)
            # Keep as MethodCall for now - could convert to Call later
            return MethodCall(
                location=expr.location,
                receiver=self.desugar_expr(expr.receiver),
                method=expr.method,
                args=[self.desugar_expr(a) for a in expr.args],
                kwargs=[(k, self.desugar_expr(v)) for k, v in expr.kwargs]
            )
        elif isinstance(expr, Index):
            return Index(
                location=expr.location,
                base=self.desugar_expr(expr.base),
                indices=[self.desugar_expr(i) for i in expr.indices]
            )
        elif isinstance(expr, FieldAccess):
            return FieldAccess(
                location=expr.location,
                base=self.desugar_expr(expr.base),
                field=expr.field
            )
        elif isinstance(expr, Tensor):
            return Tensor(
                location=expr.location,
                elements=[self.desugar_expr(e) for e in expr.elements]
            )
        elif isinstance(expr, Param):
            return Param(
                location=expr.location,
                initializer=self.desugar_expr(expr.initializer)
            )
        elif isinstance(expr, Obs):
            return Obs(
                location=expr.location,
                value=self.desugar_expr(expr.value)
            )
        elif isinstance(expr, Lambda):
            return Lambda(
                location=expr.location,
                params=expr.params,
                body=self.desugar_expr(expr.body),
                return_type=expr.return_type
            )
        elif isinstance(expr, RandExpr):
            return RandExpr(
                location=expr.location,
                distribution=self.desugar_expr(expr.distribution)
            )
        elif isinstance(expr, ObserveExpr):
            return ObserveExpr(
                location=expr.location,
                value=self.desugar_expr(expr.value),
                distribution=self.desugar_expr(expr.distribution)
            )
        elif isinstance(expr, NonDiffBlock):
            return NonDiffBlock(
                location=expr.location,
                body=self.desugar_block(expr.body)
            )
        elif isinstance(expr, (Identifier, Literal)):
            return expr
        else:
            return expr
    
    def desugar_if_expr(self, node: IfExpr) -> IfExpr:
        """
        Desugar an if expression.
        
        If temperature is specified, marks for later Gate/Mix lowering.
        Without temperature, remains a standard conditional.
        """
        return IfExpr(
            location=node.location,
            condition=self.desugar_expr(node.condition),
            then_expr=self.desugar_expr(node.then_expr),
            else_expr=self.desugar_expr(node.else_expr),
            temperature=self.desugar_expr(node.temperature) if node.temperature else None
        )


def desugar_module(module: Module) -> Module:
    """Convenience function to desugar a module."""
    return Desugarer().desugar(module)
