"""
Role assignment pass.

Assigns roles (param, obs, const, computed) to all expressions
and tracks role provenance through computations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from delta.frontend.ast import (
    Module, Statement, Expression,
    FunctionDef, LetStmt, LearnBlock, ConstraintStmt,
    ParamDecl, ObsDecl,
    Block, IfExpr, BinaryOp, UnaryOp, Call, MethodCall,
    Index, FieldAccess, Tensor, Param, Obs, Identifier, Literal,
    Lambda, RandExpr, ObserveExpr, NonDiffBlock,
    IdentifierPattern,
)
from delta.types.roles import Role, RoleSet, RoleInfo, propagate_roles
from delta.source import SourceLocation


@dataclass
class RoleContext:
    """Context for role assignment."""
    roles: dict[str, RoleInfo] = field(default_factory=dict)
    parent: Optional[RoleContext] = None
    
    def get(self, name: str) -> Optional[RoleInfo]:
        """Get role for a name."""
        if name in self.roles:
            return self.roles[name]
        if self.parent:
            return self.parent.get(name)
        return None
    
    def set(self, name: str, role: RoleInfo) -> None:
        """Set role for a name."""
        self.roles[name] = role
    
    def child(self) -> RoleContext:
        """Create child context."""
        return RoleContext(parent=self)


class RoleAssigner:
    """
    Assigns roles to expressions in a Delta AST.
    
    Tracks role propagation to ensure:
    - Param-derived values are marked for gradient
    - Obs values don't receive gradients
    - Role violations are detected
    """
    
    def __init__(self) -> None:
        self.context = RoleContext()
        self.expr_roles: dict[int, RoleInfo] = {}  # expr id -> role
    
    def assign(self, module: Module) -> dict[int, RoleInfo]:
        """Assign roles to all expressions in a module."""
        for item in module.items:
            self._assign_statement(item)
        return self.expr_roles
    
    def _assign_statement(self, stmt: Statement) -> None:
        """Assign roles for a statement."""
        if isinstance(stmt, FunctionDef):
            self._assign_function(stmt)
        elif isinstance(stmt, LetStmt):
            role = self._assign_expr(stmt.value)
            if isinstance(stmt.pattern, IdentifierPattern):
                self.context.set(stmt.pattern.name, role)
        elif isinstance(stmt, ParamDecl):
            self.context.set(stmt.name, RoleInfo.param())
        elif isinstance(stmt, ObsDecl):
            self.context.set(stmt.name, RoleInfo.obs())
        elif isinstance(stmt, LearnBlock):
            self._assign_learn_block(stmt)
        elif isinstance(stmt, ConstraintStmt):
            self._assign_expr(stmt.expr)
    
    def _assign_function(self, func: FunctionDef) -> None:
        """Assign roles in a function."""
        old_context = self.context
        self.context = self.context.child()
        
        for param in func.params:
            if param.role == "param":
                self.context.set(param.name, RoleInfo.param())
            elif param.role == "obs":
                self.context.set(param.name, RoleInfo.obs())
            else:
                self.context.set(param.name, RoleInfo.const())
        
        self._assign_block(func.body)
        self.context = old_context
    
    def _assign_learn_block(self, stmt: LearnBlock) -> None:
        """Assign roles in a learn block."""
        old_context = self.context
        self.context = self.context.child()
        
        self._assign_block(stmt.body)
        
        self.context = old_context
    
    def _assign_block(self, block: Block) -> RoleInfo:
        """Assign roles in a block."""
        for s in block.statements:
            self._assign_statement(s)
        
        if block.result:
            return self._assign_expr(block.result)
        return RoleInfo.const()
    
    def _assign_expr(self, expr: Expression) -> RoleInfo:
        """Assign role to an expression."""
        role: RoleInfo
        
        if isinstance(expr, Literal):
            role = RoleInfo.const()
        
        elif isinstance(expr, Identifier):
            existing = self.context.get(expr.name)
            role = existing if existing else RoleInfo.const()
        
        elif isinstance(expr, BinaryOp):
            left_role = self._assign_expr(expr.left)
            right_role = self._assign_expr(expr.right)
            role = RoleInfo.computed(left_role, right_role)
        
        elif isinstance(expr, UnaryOp):
            operand_role = self._assign_expr(expr.operand)
            role = RoleInfo.computed(operand_role)
        
        elif isinstance(expr, Call):
            func_role = self._assign_expr(expr.func)
            arg_roles = [self._assign_expr(arg) for arg in expr.args]
            for _, kwarg in expr.kwargs:
                arg_roles.append(self._assign_expr(kwarg))
            role = RoleInfo.computed(func_role, *arg_roles)
        
        elif isinstance(expr, MethodCall):
            receiver_role = self._assign_expr(expr.receiver)
            arg_roles = [self._assign_expr(arg) for arg in expr.args]
            role = RoleInfo.computed(receiver_role, *arg_roles)
        
        elif isinstance(expr, Index):
            base_role = self._assign_expr(expr.base)
            for idx in expr.indices:
                self._assign_expr(idx)
            role = RoleInfo.computed(base_role)
        
        elif isinstance(expr, FieldAccess):
            base_role = self._assign_expr(expr.base)
            role = RoleInfo.computed(base_role)
        
        elif isinstance(expr, IfExpr):
            cond_role = self._assign_expr(expr.condition)
            then_role = self._assign_expr(expr.then_expr)
            else_role = self._assign_expr(expr.else_expr)
            if expr.temperature:
                self._assign_expr(expr.temperature)
            role = RoleInfo.computed(cond_role, then_role, else_role)
        
        elif isinstance(expr, Block):
            role = self._assign_block(expr)
        
        elif isinstance(expr, Lambda):
            old_context = self.context
            self.context = self.context.child()
            for param in expr.params:
                self.context.set(param.name, RoleInfo.const())
            body_role = self._assign_expr(expr.body)
            self.context = old_context
            role = body_role
        
        elif isinstance(expr, Tensor):
            elem_roles = [self._assign_expr(e) for e in expr.elements]
            role = RoleInfo.computed(*elem_roles) if elem_roles else RoleInfo.const()
        
        elif isinstance(expr, Param):
            self._assign_expr(expr.initializer)
            role = RoleInfo.param()
        
        elif isinstance(expr, Obs):
            self._assign_expr(expr.value)
            role = RoleInfo.obs()
        
        elif isinstance(expr, RandExpr):
            self._assign_expr(expr.distribution)
            role = RoleInfo.computed()  # Stochastic but can be reparameterized
        
        elif isinstance(expr, ObserveExpr):
            value_role = self._assign_expr(expr.value)
            self._assign_expr(expr.distribution)
            role = RoleInfo.computed(value_role)
        
        elif isinstance(expr, NonDiffBlock):
            block_role = self._assign_block(expr.body)
            # Non-diff blocks don't propagate gradients
            role = RoleInfo(
                primary_role=Role.COMPUTED,
                contributing_roles=block_role.contributing_roles,
                requires_grad=False
            )
        
        else:
            role = RoleInfo.const()
        
        self.expr_roles[id(expr)] = role
        return role


def assign_roles(module: Module) -> dict[int, RoleInfo]:
    """Convenience function to assign roles."""
    assigner = RoleAssigner()
    return assigner.assign(module)
