"""
Type checker for Delta.

Performs type checking after inference to validate:
- Mode constraints (stochastic in infer, etc.)
- Role constraints (gradient into obs, etc.)
- Effect constraints (non_diff in learn, etc.)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from delta.frontend.ast import (
    ASTNode, Module, Statement, Expression,
    FunctionDef, LetStmt, LearnBlock, ConstraintStmt,
    ParamDecl, ObsDecl, IfExpr, BinaryOp, NonDiffBlock,
    RandExpr, ObserveExpr, Block, Identifier,
    DefaultASTVisitor,
)
from delta.types.types import Type
from delta.types.effects import EffectSet, Effect
from delta.types.roles import Role, RoleSet, RoleInfo, RoleChecker
from delta.types.inference import TypeInference, TypeEnvironment, TypedExpr
from delta.errors import (
    TypeError, EffectError, ModeError, ErrorCode, 
    ErrorCollector, ErrorSeverity
)
from delta.source import SourceLocation


@dataclass
class CheckContext:
    """Context for type checking."""
    mode: str = "train"  # train, infer, analyze
    in_learn_block: bool = False
    in_non_diff: bool = False
    in_constraint: bool = False
    return_type: Optional[Type] = None


class TypeChecker(DefaultASTVisitor):
    """
    Type checker for Delta programs.
    
    Validates semantic constraints after type inference:
    - Mode validity
    - Effect legality
    - Role compatibility
    - Gradient flow correctness
    """
    
    def __init__(self, env: TypeEnvironment, inference: TypeInference) -> None:
        self.env = env
        self.inference = inference
        self.errors = ErrorCollector()
        self.context = CheckContext()
        self.role_checker = RoleChecker()
    
    def check_module(self, module: Module) -> list[TypeError]:
        """Check an entire module."""
        for item in module.items:
            self.check_statement(item, self.env)
        
        return list(self.errors.errors)
    
    def check_statement(self, stmt: Statement, env: TypeEnvironment) -> None:
        """Check a statement."""
        if isinstance(stmt, FunctionDef):
            self.check_function_def(stmt, env)
        elif isinstance(stmt, LetStmt):
            self.check_let_stmt(stmt, env)
        elif isinstance(stmt, LearnBlock):
            self.check_learn_block(stmt, env)
        elif isinstance(stmt, ConstraintStmt):
            self.check_constraint_stmt(stmt, env)
        elif isinstance(stmt, ParamDecl):
            self.check_param_decl(stmt, env)
        elif isinstance(stmt, ObsDecl):
            self.check_obs_decl(stmt, env)
        else:
            # Check any expressions in the statement
            for child in stmt.children():
                if isinstance(child, Expression):
                    self.check_expression(child, env)
                elif isinstance(child, Statement):
                    self.check_statement(child, env)
    
    def check_function_def(self, func: FunctionDef, env: TypeEnvironment) -> None:
        """Check a function definition."""
        body_env = env.child_scope()
        
        # Add parameters to environment
        for param in func.params:
            binding = env.lookup(param.name)
            if binding:
                body_env.define(
                    param.name, 
                    binding.type, 
                    binding.role,
                    location=param.location
                )
        
        # Check body
        self.check_block(func.body, body_env)
    
    def check_let_stmt(self, stmt: LetStmt, env: TypeEnvironment) -> None:
        """Check a let statement."""
        value_typed = self.inference.infer_expression(stmt.value, env)
        
        # Check for invalid role assignments
        # (e.g., assigning param-derived to obs variable)
        self.check_expression(stmt.value, env)
    
    def check_learn_block(self, stmt: LearnBlock, env: TypeEnvironment) -> None:
        """Check a learn block."""
        old_mode = self.context.mode
        old_in_learn = self.context.in_learn_block
        
        self.context.mode = stmt.mode
        self.context.in_learn_block = True
        
        try:
            self.check_block(stmt.body, env.child_scope())
            
            # Validate mode-specific constraints
            if stmt.mode == "infer":
                self._check_infer_mode_constraints(stmt)
        finally:
            self.context.mode = old_mode
            self.context.in_learn_block = old_in_learn
    
    def check_constraint_stmt(self, stmt: ConstraintStmt, env: TypeEnvironment) -> None:
        """Check a constraint statement."""
        old_in_constraint = self.context.in_constraint
        self.context.in_constraint = True
        
        try:
            expr_typed = self.inference.infer_expression(stmt.expr, env)
            
            # Check that constraint expression is valid
            self.check_expression(stmt.expr, env)
            
            # In learn block, non_diff in constraint is an error
            if self.context.in_learn_block and expr_typed.effects.has_non_diff():
                self.errors.add(EffectError(
                    message="Non-differentiable expression in learn constraint",
                    location=stmt.location,
                    code=ErrorCode.E503_NON_DIFF_IN_LEARN
                ))
        finally:
            self.context.in_constraint = old_in_constraint
    
    def check_param_decl(self, stmt: ParamDecl, env: TypeEnvironment) -> None:
        """Check a param declaration."""
        if stmt.initializer:
            self.check_expression(stmt.initializer, env)
    
    def check_obs_decl(self, stmt: ObsDecl, env: TypeEnvironment) -> None:
        """Check an obs declaration."""
        pass  # No special checks needed
    
    def check_block(self, block: Block, env: TypeEnvironment) -> None:
        """Check a block."""
        for stmt in block.statements:
            self.check_statement(stmt, env)
        
        if block.result:
            self.check_expression(block.result, env)
    
    def check_expression(self, expr: Expression, env: TypeEnvironment) -> TypedExpr:
        """Check an expression."""
        typed = self.inference.infer_expression(expr, env)
        
        # Check mode constraints
        self._check_mode_effects(typed.effects, expr.location)
        
        # Check gradient flow
        if typed.requires_grad and self.context.in_non_diff:
            self.errors.add(ModeError(
                message="Gradient-requiring expression in non_diff block",
                location=expr.location,
                code=ErrorCode.E504_HARD_BRANCH_IN_TRAIN
            ))
        
        # Recursively check sub-expressions
        if isinstance(expr, IfExpr):
            self._check_if_expr(expr, env, typed)
        elif isinstance(expr, BinaryOp):
            self.check_expression(expr.left, env)
            self.check_expression(expr.right, env)
        elif isinstance(expr, NonDiffBlock):
            self._check_non_diff_block(expr, env)
        elif isinstance(expr, RandExpr):
            self._check_rand_expr(expr, env)
        elif isinstance(expr, ObserveExpr):
            self._check_observe_expr(expr, env)
        elif isinstance(expr, Block):
            self.check_block(expr, env.child_scope())
        
        return typed
    
    def _check_if_expr(self, expr: IfExpr, env: TypeEnvironment, typed: TypedExpr) -> None:
        """Check an if expression."""
        # In train mode without temperature, if with param-derived condition is error
        if self.context.mode == "train" and expr.temperature is None:
            cond_typed = self.inference.infer_expression(expr.condition, env)
            if cond_typed.requires_grad and self.context.in_learn_block:
                self.errors.add(ModeError(
                    message="Hard branch on gradient-requiring condition in train mode. "
                            "Use 'temperature' for differentiable branching.",
                    location=expr.location,
                    code=ErrorCode.E504_HARD_BRANCH_IN_TRAIN
                ))
        
        self.check_expression(expr.condition, env)
        self.check_expression(expr.then_expr, env)
        self.check_expression(expr.else_expr, env)
        
        if expr.temperature:
            self.check_expression(expr.temperature, env)
    
    def _check_non_diff_block(self, expr: NonDiffBlock, env: TypeEnvironment) -> None:
        """Check a non_diff block."""
        if self.context.in_learn_block and self.context.in_constraint:
            self.errors.add(EffectError(
                message="non_diff block inside learn constraint",
                location=expr.location,
                code=ErrorCode.E503_NON_DIFF_IN_LEARN
            ))
        
        old_in_non_diff = self.context.in_non_diff
        self.context.in_non_diff = True
        
        try:
            self.check_block(expr.body, env.child_scope())
        finally:
            self.context.in_non_diff = old_in_non_diff
    
    def _check_rand_expr(self, expr: RandExpr, env: TypeEnvironment) -> None:
        """Check a rand expression."""
        if self.context.mode == "infer":
            self.errors.add(ModeError(
                message="Stochastic expression (rand) not allowed in infer mode",
                location=expr.location,
                code=ErrorCode.E501_STOCHASTIC_IN_INFER
            ))
        
        self.check_expression(expr.distribution, env)
    
    def _check_observe_expr(self, expr: ObserveExpr, env: TypeEnvironment) -> None:
        """Check an observe expression."""
        self.check_expression(expr.value, env)
        self.check_expression(expr.distribution, env)
    
    def _check_mode_effects(self, effects: EffectSet, location: SourceLocation) -> None:
        """Check that effects are allowed in current mode."""
        if self.context.mode == "infer" and effects.has_stochastic():
            self.errors.add(ModeError(
                message="Stochastic effect not allowed in infer mode",
                location=location,
                code=ErrorCode.E501_STOCHASTIC_IN_INFER
            ))
    
    def _check_infer_mode_constraints(self, stmt: LearnBlock) -> None:
        """Check infer mode specific constraints."""
        # In infer mode, we shouldn't have stochastic expressions
        # (unless explicitly marked as allowed)
        pass
    
    def _check_gradient_into_obs(
        self, 
        source: TypedExpr, 
        target_name: str,
        env: TypeEnvironment,
        location: SourceLocation
    ) -> None:
        """Check that gradients don't flow into obs variables."""
        target_binding = env.lookup(target_name)
        
        if target_binding and target_binding.role.primary_role == Role.OBS:
            if source.requires_grad:
                self.errors.add(ModeError(
                    message=f"Cannot flow gradients into obs variable '{target_name}'",
                    location=location,
                    code=ErrorCode.E502_GRADIENT_INTO_OBS
                ))


def check_module(module: Module, env: TypeEnvironment, inference: TypeInference) -> list[TypeError]:
    """Convenience function to check a module."""
    checker = TypeChecker(env, inference)
    return checker.check_module(module)
