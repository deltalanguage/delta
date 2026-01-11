"""
Effect inference pass.

Infers effects (stochastic, non_diff, io) for all expressions
and validates effect constraints.
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
from delta.types.effects import EffectSet, Effect, effect_union


@dataclass
class EffectContext:
    """Context for effect inference."""
    function_effects: dict[str, EffectSet] = field(default_factory=dict)
    current_effects: EffectSet = field(default_factory=EffectSet.pure)


class EffectInferrer:
    """
    Infers effects for expressions in a Delta AST.
    
    Effects include:
    - stochastic: Random sampling
    - non_diff: Non-differentiable operations
    - io: Input/output
    """
    
    def __init__(self) -> None:
        self.context = EffectContext()
        self.expr_effects: dict[int, EffectSet] = {}
        self._setup_builtin_effects()
    
    def _setup_builtin_effects(self) -> None:
        """Define effects for built-in functions."""
        # Stochastic functions
        self.context.function_effects["randn"] = EffectSet.stochastic()
        self.context.function_effects["rand"] = EffectSet.stochastic()
        
        # Pure functions (no effects)
        pure_functions = [
            "zeros", "ones", "exp", "log", "sin", "cos", "tanh",
            "sigmoid", "relu", "softmax", "sum", "mean", "max", "min",
            "matmul", "reshape", "transpose"
        ]
        for f in pure_functions:
            self.context.function_effects[f] = EffectSet.pure()
    
    def infer(self, module: Module) -> dict[int, EffectSet]:
        """Infer effects for all expressions in a module."""
        # First pass: collect function signatures
        for item in module.items:
            if isinstance(item, FunctionDef):
                self._collect_function_effects(item)
        
        # Second pass: infer effects
        for item in module.items:
            self._infer_statement(item)
        
        return self.expr_effects
    
    def _collect_function_effects(self, func: FunctionDef) -> None:
        """Collect declared effects for a function."""
        effects = EffectSet.pure()
        for eff in func.effects:
            if eff == "stoch":
                effects = effects.union(EffectSet.stochastic())
            elif eff == "non_diff":
                effects = effects.union(EffectSet.non_diff())
            elif eff == "io":
                effects = effects.union(EffectSet.io())
        
        self.context.function_effects[func.name] = effects
    
    def _infer_statement(self, stmt: Statement) -> EffectSet:
        """Infer effects for a statement."""
        if isinstance(stmt, FunctionDef):
            return self._infer_function(stmt)
        elif isinstance(stmt, LetStmt):
            return self._infer_expr(stmt.value)
        elif isinstance(stmt, LearnBlock):
            return self._infer_learn_block(stmt)
        elif isinstance(stmt, ConstraintStmt):
            return self._infer_expr(stmt.expr)
        elif isinstance(stmt, ParamDecl):
            if stmt.initializer:
                return self._infer_expr(stmt.initializer)
            return EffectSet.pure()
        else:
            return EffectSet.pure()
    
    def _infer_function(self, func: FunctionDef) -> EffectSet:
        """Infer effects in a function body."""
        body_effects = self._infer_block(func.body)
        
        # Update function's declared effects if needed
        declared = self.context.function_effects.get(func.name, EffectSet.pure())
        actual = declared.union(body_effects)
        self.context.function_effects[func.name] = actual
        
        return actual
    
    def _infer_learn_block(self, stmt: LearnBlock) -> EffectSet:
        """Infer effects in a learn block."""
        return self._infer_block(stmt.body)
    
    def _infer_block(self, block: Block) -> EffectSet:
        """Infer effects for a block."""
        effects = EffectSet.pure()
        
        for s in block.statements:
            effects = effects.union(self._infer_statement(s))
        
        if block.result:
            effects = effects.union(self._infer_expr(block.result))
        
        return effects
    
    def _infer_expr(self, expr: Expression) -> EffectSet:
        """Infer effects for an expression."""
        effects: EffectSet
        
        if isinstance(expr, Literal):
            effects = EffectSet.pure()
        
        elif isinstance(expr, Identifier):
            effects = EffectSet.pure()
        
        elif isinstance(expr, BinaryOp):
            left = self._infer_expr(expr.left)
            right = self._infer_expr(expr.right)
            effects = left.union(right)
        
        elif isinstance(expr, UnaryOp):
            effects = self._infer_expr(expr.operand)
        
        elif isinstance(expr, Call):
            # Get function effects
            func_effects = EffectSet.pure()
            if isinstance(expr.func, Identifier):
                func_effects = self.context.function_effects.get(
                    expr.func.name, EffectSet.pure()
                )
            
            # Combine with argument effects
            arg_effects = EffectSet.pure()
            for arg in expr.args:
                arg_effects = arg_effects.union(self._infer_expr(arg))
            for _, kwarg in expr.kwargs:
                arg_effects = arg_effects.union(self._infer_expr(kwarg))
            
            effects = func_effects.union(arg_effects)
        
        elif isinstance(expr, MethodCall):
            receiver = self._infer_expr(expr.receiver)
            arg_effects = EffectSet.pure()
            for arg in expr.args:
                arg_effects = arg_effects.union(self._infer_expr(arg))
            effects = receiver.union(arg_effects)
        
        elif isinstance(expr, Index):
            effects = self._infer_expr(expr.base)
            for idx in expr.indices:
                effects = effects.union(self._infer_expr(idx))
        
        elif isinstance(expr, FieldAccess):
            effects = self._infer_expr(expr.base)
        
        elif isinstance(expr, IfExpr):
            cond = self._infer_expr(expr.condition)
            then_eff = self._infer_expr(expr.then_expr)
            else_eff = self._infer_expr(expr.else_expr)
            effects = cond.union(then_eff).union(else_eff)
            if expr.temperature:
                effects = effects.union(self._infer_expr(expr.temperature))
        
        elif isinstance(expr, Block):
            effects = self._infer_block(expr)
        
        elif isinstance(expr, Lambda):
            effects = self._infer_expr(expr.body)
        
        elif isinstance(expr, Tensor):
            effects = EffectSet.pure()
            for elem in expr.elements:
                effects = effects.union(self._infer_expr(elem))
        
        elif isinstance(expr, Param):
            effects = self._infer_expr(expr.initializer)
        
        elif isinstance(expr, Obs):
            effects = self._infer_expr(expr.value)
        
        elif isinstance(expr, RandExpr):
            # Rand always has stochastic effect
            dist_effects = self._infer_expr(expr.distribution)
            effects = EffectSet.stochastic().union(dist_effects)
        
        elif isinstance(expr, ObserveExpr):
            value_effects = self._infer_expr(expr.value)
            dist_effects = self._infer_expr(expr.distribution)
            effects = EffectSet.stochastic().union(value_effects).union(dist_effects)
        
        elif isinstance(expr, NonDiffBlock):
            body_effects = self._infer_block(expr.body)
            effects = EffectSet.non_diff().union(body_effects)
        
        else:
            effects = EffectSet.pure()
        
        self.expr_effects[id(expr)] = effects
        return effects


def infer_effects(module: Module) -> dict[int, EffectSet]:
    """Convenience function to infer effects."""
    inferrer = EffectInferrer()
    return inferrer.infer(module)
