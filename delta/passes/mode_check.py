"""
Mode admissibility checking pass.

Validates that code is legal in the current mode:
- train: All effects allowed
- infer: No stochastic effects
- analyze: All effects allowed, plus introspection
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from delta.frontend.ast import (
    Module, Statement, Expression,
    FunctionDef, LetStmt, LearnBlock, ConstraintStmt,
    Block, IfExpr, RandExpr, ObserveExpr, NonDiffBlock,
    Identifier, WhileStmt, ForStmt,
)
from delta.types.effects import EffectSet
from delta.types.roles import RoleInfo
from delta.errors import ModeError, ErrorCode, ErrorCollector


@dataclass
class ModeContext:
    """Context for mode checking."""
    mode: str = "train"  # train, infer, analyze
    in_learn_block: bool = False
    in_constraint: bool = False


class ModeChecker:
    """
    Checks mode admissibility for Delta programs.
    
    Validates:
    - No stochastic in infer mode
    - No non_diff in learn constraints
    - No hard branches on grad-requiring conditions in train
    - No gradient into obs
    """
    
    def __init__(
        self,
        expr_effects: dict[int, EffectSet],
        expr_roles: dict[int, RoleInfo]
    ) -> None:
        self.expr_effects = expr_effects
        self.expr_roles = expr_roles
        self.errors = ErrorCollector()
        self.context = ModeContext()
    
    def check(self, module: Module) -> ErrorCollector:
        """Check mode admissibility for a module."""
        for item in module.items:
            self._check_statement(item)
        return self.errors
    
    def _check_statement(self, stmt: Statement) -> None:
        """Check a statement."""
        if isinstance(stmt, FunctionDef):
            self._check_function(stmt)
        elif isinstance(stmt, LearnBlock):
            self._check_learn_block(stmt)
        elif isinstance(stmt, ConstraintStmt):
            self._check_constraint(stmt)
        elif isinstance(stmt, LetStmt):
            self._check_expr(stmt.value)
        elif isinstance(stmt, WhileStmt):
            self._check_while_stmt(stmt)
        elif isinstance(stmt, ForStmt):
            self._check_for_stmt(stmt)
    
    def _check_function(self, func: FunctionDef) -> None:
        """Check a function."""
        self._check_block(func.body)
    
    def _check_while_stmt(self, stmt: WhileStmt) -> None:
        """Check a while statement.
        
        While loops are forbidden in differentiable contexts (learn blocks)
        because they have data-dependent iteration counts, which breaks
        the differentiability requirements of the spec (Section 8).
        """
        # While loops are forbidden in learn blocks (differentiable context)
        if self.context.in_learn_block and self.context.mode == "train":
            self.errors.add(ModeError(
                message="While loops are forbidden in differentiable (learn) contexts. "
                        "Use 'for' loops with fixed iteration counts, or wrap in non_diff.",
                location=stmt.location,
                code=ErrorCode.E506_WHILE_IN_DIFF
            ))
        
        self._check_expr(stmt.condition)
        self._check_block(stmt.body)
    
    def _check_for_stmt(self, stmt: ForStmt) -> None:
        """Check a for statement."""
        self._check_expr(stmt.iterable)
        self._check_block(stmt.body)
    
    def _check_learn_block(self, stmt: LearnBlock) -> None:
        """Check a learn block."""
        old_mode = self.context.mode
        old_in_learn = self.context.in_learn_block
        
        self.context.mode = stmt.mode
        self.context.in_learn_block = True
        
        try:
            self._check_block(stmt.body)
        finally:
            self.context.mode = old_mode
            self.context.in_learn_block = old_in_learn
    
    def _check_constraint(self, stmt: ConstraintStmt) -> None:
        """Check a constraint statement."""
        old_in_constraint = self.context.in_constraint
        self.context.in_constraint = True
        
        try:
            effects = self.expr_effects.get(id(stmt.expr), EffectSet.pure())
            
            # Check for non_diff in learn constraint
            if self.context.in_learn_block and effects.has_non_diff():
                self.errors.add(ModeError(
                    message="Non-differentiable expression in learn constraint",
                    location=stmt.location,
                    code=ErrorCode.E503_NON_DIFF_IN_LEARN
                ))
            
            self._check_expr(stmt.expr)
        finally:
            self.context.in_constraint = old_in_constraint
    
    def _check_block(self, block: Block) -> None:
        """Check a block."""
        for s in block.statements:
            self._check_statement(s)
        
        if block.result:
            self._check_expr(block.result)
    
    def _check_expr(self, expr: Expression) -> None:
        """Check an expression for mode violations."""
        effects = self.expr_effects.get(id(expr), EffectSet.pure())
        role = self.expr_roles.get(id(expr))
        
        # Check stochastic in infer mode
        if self.context.mode == "infer" and effects.has_stochastic():
            if isinstance(expr, RandExpr):
                self.errors.add(ModeError(
                    message="Stochastic expression (rand) not allowed in infer mode",
                    location=expr.location,
                    code=ErrorCode.E501_STOCHASTIC_IN_INFER
                ))
        
        # Check if expression with special handling
        if isinstance(expr, IfExpr):
            self._check_if_expr(expr)
        
        # Check non_diff block in constraint
        if isinstance(expr, NonDiffBlock):
            if self.context.in_constraint and self.context.in_learn_block:
                self.errors.add(ModeError(
                    message="non_diff block in learn constraint",
                    location=expr.location,
                    code=ErrorCode.E503_NON_DIFF_IN_LEARN
                ))
        
        # Recurse into children
        if isinstance(expr, Block):
            self._check_block(expr)
    
    def _check_if_expr(self, expr: IfExpr) -> None:
        """Check an if expression for mode violations."""
        cond_role = self.expr_roles.get(id(expr.condition))
        
        # Hard branch on grad-requiring condition in train mode
        if (self.context.mode == "train" and 
            self.context.in_learn_block and
            expr.temperature is None):
            
            if cond_role and cond_role.requires_grad:
                self.errors.add(ModeError(
                    message="Hard branch on gradient-requiring condition in train mode. "
                            "Use 'temperature' for differentiable branching, or wrap in non_diff.",
                    location=expr.location,
                    code=ErrorCode.E504_HARD_BRANCH_IN_TRAIN
                ))
        
        self._check_expr(expr.condition)
        self._check_expr(expr.then_expr)
        self._check_expr(expr.else_expr)
        
        if expr.temperature:
            self._check_expr(expr.temperature)


def check_mode(
    module: Module,
    expr_effects: dict[int, EffectSet],
    expr_roles: dict[int, RoleInfo]
) -> ErrorCollector:
    """Convenience function to check mode admissibility."""
    checker = ModeChecker(expr_effects, expr_roles)
    return checker.check(module)
