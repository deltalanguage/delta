"""
Constraint compilation pass.

Compiles constraints into penalty terms that can be optimized:
- Equality constraints -> Robust L2 loss
- Inequality constraints -> Softplus penalty
- Boolean constraints -> 1 - gate
- Likelihood constraints -> -log p
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from delta.ir.sir import (
    SIRNode, SIRModule, SIRFunction, SIRBlock, SIRProperty,
    TensorOp, TensorOpKind, GateOp, MixOp, ConstraintOp, ConstraintKind,
    Const, StopGrad,
    BinaryTensorOp, UnaryTensorOp,
    walk_sir,
)
from delta.types.types import FloatType


@dataclass
class ConstraintConfig:
    """Configuration for constraint compilation."""
    equality_eps: float = 1e-6  # For numerical stability
    softplus_beta: float = 1.0  # Softplus sharpness
    log_barrier_eps: float = 1e-8  # Log barrier minimum
    default_weight: float = 1.0


class ConstraintCompiler:
    """
    Compiles constraints into differentiable penalty terms.
    
    Compilation strategies:
    1. Equality (a == b): weight * huber_loss(a - b) or weight * (a - b)^2
    2. Inequality (a <= b): weight * softplus(a - b)
    3. Boolean (expr): weight * (1 - sigmoid(expr))
    4. Likelihood: -log_prob
    """
    
    def __init__(self, config: Optional[ConstraintConfig] = None) -> None:
        self.config = config or ConstraintConfig()
    
    def compile(self, module: SIRModule) -> tuple[SIRModule, SIRNode]:
        """
        Compile all constraints into a single objective node.
        
        Returns the modified module and the total penalty node.
        """
        penalties: list[SIRNode] = []
        
        for constraint in module.constraints:
            penalty = self._compile_constraint(constraint)
            penalties.append(penalty)
        
        # Sum all penalties into single objective
        if not penalties:
            total_penalty = Const(value=0.0, _props=SIRProperty(dtype=FloatType()))
        elif len(penalties) == 1:
            total_penalty = penalties[0]
        else:
            total_penalty = penalties[0]
            for p in penalties[1:]:
                total_penalty = BinaryTensorOp(TensorOpKind.ADD, total_penalty, p)
        
        return module, total_penalty
    
    def _compile_constraint(self, constraint: ConstraintOp) -> SIRNode:
        """Compile a single constraint to a penalty term."""
        if constraint.kind == ConstraintKind.EQUALITY:
            return self._compile_equality(constraint)
        elif constraint.kind == ConstraintKind.INEQUALITY:
            return self._compile_inequality(constraint)
        elif constraint.kind == ConstraintKind.BOOLEAN:
            return self._compile_boolean(constraint)
        elif constraint.kind == ConstraintKind.LIKELIHOOD:
            return self._compile_likelihood(constraint)
        else:
            # Default to equality
            return self._compile_equality(constraint)
    
    def _compile_equality(self, constraint: ConstraintOp) -> SIRNode:
        """
        Compile equality constraint: a == b
        
        Penalty: weight * (a - b)^2
        For robustness, can use Huber loss instead.
        """
        lhs = constraint.lhs
        
        if constraint.rhs:
            # a == b -> (a - b)^2
            diff = BinaryTensorOp(TensorOpKind.SUB, lhs, constraint.rhs)
        else:
            # a == 0 -> a^2
            diff = lhs
        
        # Square the difference
        squared = BinaryTensorOp(TensorOpKind.MUL, diff, diff)
        
        # Apply weight
        weighted = BinaryTensorOp(TensorOpKind.MUL, constraint.weight, squared)
        
        # Reduce to scalar if needed
        penalty = UnaryTensorOp(
            TensorOpKind.MEAN,
            weighted,
            props=SIRProperty(dtype=FloatType(), requires_grad=True)
        )
        
        return penalty
    
    def _compile_inequality(self, constraint: ConstraintOp) -> SIRNode:
        """
        Compile inequality constraint: a <= b
        
        Penalty: weight * softplus(a - b)
        Where softplus(x) = log(1 + exp(x))
        """
        lhs = constraint.lhs
        
        if constraint.rhs:
            # a <= b -> softplus(a - b)
            diff = BinaryTensorOp(TensorOpKind.SUB, lhs, constraint.rhs)
        else:
            # a <= 0 -> softplus(a)
            diff = lhs
        
        # Softplus: log(1 + exp(x))
        # For numerical stability: x + log(1 + exp(-x)) when x > 0
        exp_diff = UnaryTensorOp(TensorOpKind.EXP, diff)
        one = Const(value=1.0)
        one_plus_exp = BinaryTensorOp(TensorOpKind.ADD, one, exp_diff)
        softplus = UnaryTensorOp(TensorOpKind.LOG, one_plus_exp)
        
        # Apply weight
        weighted = BinaryTensorOp(TensorOpKind.MUL, constraint.weight, softplus)
        
        # Reduce to scalar
        penalty = UnaryTensorOp(
            TensorOpKind.MEAN,
            weighted,
            props=SIRProperty(dtype=FloatType(), requires_grad=True)
        )
        
        return penalty
    
    def _compile_boolean(self, constraint: ConstraintOp) -> SIRNode:
        """
        Compile boolean constraint: expr should be true
        
        Penalty: weight * (1 - sigmoid(expr))
        For soft gates, just: weight * (1 - gate)
        """
        lhs = constraint.lhs
        
        # Penalty is 1 - value (assuming value is already in [0, 1])
        one = Const(value=1.0)
        violation = BinaryTensorOp(TensorOpKind.SUB, one, lhs)
        
        # Apply weight
        weighted = BinaryTensorOp(TensorOpKind.MUL, constraint.weight, violation)
        
        # Reduce to scalar
        penalty = UnaryTensorOp(
            TensorOpKind.MEAN,
            weighted,
            props=SIRProperty(dtype=FloatType(), requires_grad=True)
        )
        
        return penalty
    
    def _compile_likelihood(self, constraint: ConstraintOp) -> SIRNode:
        """
        Compile likelihood constraint.
        
        The lhs should already be a log probability.
        Penalty: -log_prob * weight
        """
        # Negate the log probability
        neg_one = Const(value=-1.0)
        neg_log_prob = BinaryTensorOp(TensorOpKind.MUL, neg_one, constraint.lhs)
        
        # Apply weight
        weighted = BinaryTensorOp(TensorOpKind.MUL, constraint.weight, neg_log_prob)
        
        # Reduce to scalar
        penalty = UnaryTensorOp(
            TensorOpKind.MEAN,
            weighted,
            props=SIRProperty(dtype=FloatType(), requires_grad=True)
        )
        
        return penalty


def compile_constraints(
    module: SIRModule,
    config: Optional[ConstraintConfig] = None
) -> tuple[SIRModule, SIRNode]:
    """Convenience function to compile constraints."""
    compiler = ConstraintCompiler(config)
    return compiler.compile(module)
