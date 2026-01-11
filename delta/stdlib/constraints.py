"""
std.constraints - Constraint definitions for Delta.

Provides constraint types and violation functions
for use in Delta learn blocks.
"""

from __future__ import annotations
from typing import Optional, Union, Callable
from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn.functional as F


# ============================================================
# Constraint Types
# ============================================================

@dataclass
class Constraint:
    """Base class for constraints."""
    weight: float = 1.0
    slack: float = 0.0
    name: Optional[str] = None
    
    def violation(self, *args) -> Tensor:
        """Compute violation (0 = satisfied, positive = violated)."""
        raise NotImplementedError
    
    def penalty(self, *args) -> Tensor:
        """Compute penalty for optimization."""
        v = self.violation(*args)
        return self.weight * v


# ============================================================
# Equality Constraints
# ============================================================

@dataclass
class Equal(Constraint):
    """Equality constraint: a == b."""
    
    def violation(self, a: Tensor, b: Tensor) -> Tensor:
        """Squared difference."""
        return ((a - b) ** 2).mean()


@dataclass
class EqualTo(Constraint):
    """Equality constraint to a constant: a == value."""
    value: float = 0.0
    
    def violation(self, a: Tensor) -> Tensor:
        """Squared difference from value."""
        return ((a - self.value) ** 2).mean()


@dataclass
class ApproxEqual(Constraint):
    """Approximate equality: |a - b| < epsilon."""
    epsilon: float = 1e-3
    
    def violation(self, a: Tensor, b: Tensor) -> Tensor:
        """Violation if difference exceeds epsilon."""
        diff = torch.abs(a - b) - self.epsilon
        return F.relu(diff).mean()


# ============================================================
# Inequality Constraints
# ============================================================

@dataclass
class LessThan(Constraint):
    """Inequality constraint: a < b."""
    
    def violation(self, a: Tensor, b: Tensor) -> Tensor:
        """Softplus of (a - b)."""
        return F.softplus(a - b).mean()


@dataclass
class LessEqual(Constraint):
    """Inequality constraint: a <= b."""
    
    def violation(self, a: Tensor, b: Tensor) -> Tensor:
        """Softplus of (a - b)."""
        return F.softplus(a - b).mean()


@dataclass
class GreaterThan(Constraint):
    """Inequality constraint: a > b."""
    
    def violation(self, a: Tensor, b: Tensor) -> Tensor:
        """Softplus of (b - a)."""
        return F.softplus(b - a).mean()


@dataclass
class GreaterEqual(Constraint):
    """Inequality constraint: a >= b."""
    
    def violation(self, a: Tensor, b: Tensor) -> Tensor:
        """Softplus of (b - a)."""
        return F.softplus(b - a).mean()


@dataclass
class InRange(Constraint):
    """Range constraint: lower <= a <= upper."""
    lower: float = 0.0
    upper: float = 1.0
    
    def violation(self, a: Tensor) -> Tensor:
        """Violation if outside range."""
        below = F.relu(self.lower - a)
        above = F.relu(a - self.upper)
        return (below + above).mean()


@dataclass
class Positive(Constraint):
    """Positivity constraint: a > 0."""
    
    def violation(self, a: Tensor) -> Tensor:
        """Softplus of -a."""
        return F.softplus(-a).mean()


@dataclass
class NonNegative(Constraint):
    """Non-negativity constraint: a >= 0."""
    
    def violation(self, a: Tensor) -> Tensor:
        """Softplus of -a."""
        return F.softplus(-a).mean()


# ============================================================
# Norm Constraints
# ============================================================

@dataclass
class L1Norm(Constraint):
    """L1 norm constraint: ||a||_1 <= max_norm."""
    max_norm: float = 1.0
    
    def violation(self, a: Tensor) -> Tensor:
        """Softplus of (norm - max_norm)."""
        norm = torch.abs(a).sum()
        return F.softplus(norm - self.max_norm)


@dataclass
class L2Norm(Constraint):
    """L2 norm constraint: ||a||_2 <= max_norm."""
    max_norm: float = 1.0
    
    def violation(self, a: Tensor) -> Tensor:
        """Softplus of (norm - max_norm)."""
        norm = torch.norm(a, p=2)
        return F.softplus(norm - self.max_norm)


@dataclass
class UnitNorm(Constraint):
    """Unit norm constraint: ||a||_2 == 1."""
    
    def violation(self, a: Tensor) -> Tensor:
        """Squared deviation from unit norm."""
        norm = torch.norm(a, p=2)
        return (norm - 1.0) ** 2


# ============================================================
# Probabilistic Constraints
# ============================================================

@dataclass
class Simplex(Constraint):
    """Simplex constraint: a >= 0, sum(a) == 1."""
    
    def violation(self, a: Tensor) -> Tensor:
        """Combined violation."""
        # Non-negativity
        neg_violation = F.relu(-a).sum()
        # Sum to 1
        sum_violation = (a.sum() - 1.0) ** 2
        return neg_violation + sum_violation


@dataclass
class Probability(Constraint):
    """Probability constraint: 0 <= a <= 1."""
    
    def violation(self, a: Tensor) -> Tensor:
        """Violation if outside [0, 1]."""
        below = F.relu(-a)
        above = F.relu(a - 1.0)
        return (below + above).mean()


# ============================================================
# Matrix Constraints
# ============================================================

@dataclass
class PositiveDefinite(Constraint):
    """Positive definite matrix constraint."""
    
    def violation(self, a: Tensor) -> Tensor:
        """Violation based on minimum eigenvalue."""
        eigenvalues = torch.linalg.eigvalsh(a)
        min_eigenvalue = eigenvalues.min()
        return F.relu(-min_eigenvalue + 1e-6)


@dataclass
class Symmetric(Constraint):
    """Symmetric matrix constraint: A == A^T."""
    
    def violation(self, a: Tensor) -> Tensor:
        """Frobenius norm of (A - A^T)."""
        return ((a - a.T) ** 2).mean()


@dataclass
class Orthogonal(Constraint):
    """Orthogonal matrix constraint: A^T A == I."""
    
    def violation(self, a: Tensor) -> Tensor:
        """Frobenius norm of (A^T A - I)."""
        ata = torch.matmul(a.T, a)
        identity = torch.eye(ata.size(0), device=a.device)
        return ((ata - identity) ** 2).mean()


# ============================================================
# Constraint Utilities
# ============================================================

def combine_constraints(constraints: list[Constraint], *args) -> Tensor:
    """Combine multiple constraints into a single penalty."""
    total = torch.tensor(0.0)
    for c in constraints:
        total = total + c.penalty(*args)
    return total


def check_satisfied(constraint: Constraint, *args, threshold: float = 1e-4) -> bool:
    """Check if a constraint is approximately satisfied."""
    violation = constraint.violation(*args)
    return violation.item() < threshold


# ============================================================
# Soft Versions of Operations
# ============================================================

def soft_eq(a: Tensor, b: Tensor, temperature: float = 1.0) -> Tensor:
    """Soft equality: exp(-|a - b|^2 / temp)."""
    diff_sq = (a - b) ** 2
    return torch.exp(-diff_sq / temperature)


def soft_lt(a: Tensor, b: Tensor, temperature: float = 1.0) -> Tensor:
    """Soft less-than: sigmoid((b - a) / temp)."""
    return torch.sigmoid((b - a) / temperature)


def soft_gt(a: Tensor, b: Tensor, temperature: float = 1.0) -> Tensor:
    """Soft greater-than: sigmoid((a - b) / temp)."""
    return torch.sigmoid((a - b) / temperature)


def soft_and(a: Tensor, b: Tensor) -> Tensor:
    """Soft AND (product)."""
    return a * b


def soft_or(a: Tensor, b: Tensor) -> Tensor:
    """Soft OR: 1 - (1-a)(1-b)."""
    return 1 - (1 - a) * (1 - b)


def soft_not(a: Tensor) -> Tensor:
    """Soft NOT: 1 - a."""
    return 1 - a


def soft_if(condition: Tensor, then_val: Tensor, else_val: Tensor) -> Tensor:
    """Soft if-then-else: condition * then + (1-condition) * else."""
    return condition * then_val + (1 - condition) * else_val


# ============================================================
# Violation Functions (for testing compatibility)
# ============================================================

def equality_violation(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise equality violation (squared difference)."""
    return (a - b) ** 2


def greater_than_violation(a: Tensor, threshold: float) -> Tensor:
    """Element-wise violation for a > threshold."""
    return F.relu(threshold - a)


def less_than_violation(a: Tensor, threshold: float) -> Tensor:
    """Element-wise violation for a < threshold."""
    return F.relu(a - threshold)


def bound_violation(a: Tensor, low: float = 0.0, high: float = 1.0) -> Tensor:
    """Element-wise violation for low <= a <= high."""
    below = F.relu(low - a)
    above = F.relu(a - high)
    return below + above


def weight(violation: Tensor, w: float = 1.0) -> Tensor:
    """Apply weight to a violation tensor."""
    return w * violation
