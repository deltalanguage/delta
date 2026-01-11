"""
Effect system for Delta.

Tracks computational effects including:
- Stochastic effects (random sampling)
- Non-differentiable effects
- IO effects
- Effect polymorphism with effect variables
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, FrozenSet
from enum import Enum, auto
import itertools


# Effect variable counter
_effect_var_counter = itertools.count()


def fresh_effect_var(prefix: str = "E") -> EffectVar:
    """Create a fresh effect variable."""
    return EffectVar(name=f"{prefix}{next(_effect_var_counter)}")


class Effect(Enum):
    """
    Primitive effects in Delta.
    
    Effects track computational side effects that constrain
    where expressions can be used.
    """
    PURE = auto()       # No effects
    STOCHASTIC = auto() # Random sampling (rand)
    NON_DIFF = auto()   # Non-differentiable operations
    IO = auto()         # Input/output operations
    DIVERGE = auto()    # Potential non-termination
    
    def __str__(self) -> str:
        return self.name.lower()


# Singleton effects for convenience
PURE = frozenset[Effect]()
STOCHASTIC = frozenset({Effect.STOCHASTIC})
NON_DIFF = frozenset({Effect.NON_DIFF})
IO = frozenset({Effect.IO})


@dataclass(frozen=True)
class EffectVar:
    """
    Effect variable for effect polymorphism.
    
    Allows functions to be polymorphic in their effects:
    fn map[E](f: fn(A) -> B / E, xs: List[A]) -> List[B] / E
    """
    name: str
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, EffectVar) and self.name == other.name
    
    def __hash__(self) -> int:
        return hash(("EffectVar", self.name))


@dataclass(frozen=True)
class EffectSet:
    """
    A set of effects, possibly with effect variables.
    
    Represents the effect annotation on expressions and functions.
    """
    effects: frozenset[Effect] = field(default_factory=frozenset)
    effect_vars: frozenset[EffectVar] = field(default_factory=frozenset)
    
    def __str__(self) -> str:
        parts = [str(e) for e in sorted(self.effects, key=lambda e: e.name)]
        parts.extend(str(v) for v in sorted(self.effect_vars, key=lambda v: v.name))
        
        if not parts:
            return "pure"
        return " + ".join(parts)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EffectSet):
            return False
        return self.effects == other.effects and self.effect_vars == other.effect_vars
    
    def __hash__(self) -> int:
        return hash((self.effects, self.effect_vars))
    
    def is_pure(self) -> bool:
        """Check if this effect set is pure (no effects)."""
        return not self.effects and not self.effect_vars
    
    def has_stochastic(self) -> bool:
        """Check if this includes stochastic effects."""
        return Effect.STOCHASTIC in self.effects
    
    def has_non_diff(self) -> bool:
        """Check if this includes non-differentiable effects."""
        return Effect.NON_DIFF in self.effects
    
    def has_io(self) -> bool:
        """Check if this includes IO effects."""
        return Effect.IO in self.effects
    
    def union(self, other: EffectSet) -> EffectSet:
        """Union of two effect sets."""
        return EffectSet(
            effects=self.effects | other.effects,
            effect_vars=self.effect_vars | other.effect_vars
        )
    
    def without(self, effects: frozenset[Effect]) -> EffectSet:
        """Remove specific effects."""
        return EffectSet(
            effects=self.effects - effects,
            effect_vars=self.effect_vars
        )
    
    def substitute(self, var: EffectVar, effects: EffectSet) -> EffectSet:
        """Substitute an effect variable with an effect set."""
        if var not in self.effect_vars:
            return self
        
        new_vars = self.effect_vars - {var}
        return EffectSet(
            effects=self.effects | effects.effects,
            effect_vars=new_vars | effects.effect_vars
        )
    
    def free_vars(self) -> frozenset[EffectVar]:
        """Get free effect variables."""
        return self.effect_vars
    
    @classmethod
    def pure(cls) -> EffectSet:
        """Create a pure (no effects) set."""
        return cls()
    
    @classmethod
    def stochastic(cls) -> EffectSet:
        """Create a stochastic effect set."""
        return cls(effects=frozenset({Effect.STOCHASTIC}))
    
    @classmethod
    def non_diff(cls) -> EffectSet:
        """Create a non-differentiable effect set."""
        return cls(effects=frozenset({Effect.NON_DIFF}))
    
    @classmethod
    def io(cls) -> EffectSet:
        """Create an IO effect set."""
        return cls(effects=frozenset({Effect.IO}))
    
    @classmethod
    def from_var(cls, var: EffectVar) -> EffectSet:
        """Create an effect set with a single effect variable."""
        return cls(effect_vars=frozenset({var}))


def effect_union(*effect_sets: EffectSet) -> EffectSet:
    """Union multiple effect sets."""
    result = EffectSet.pure()
    for es in effect_sets:
        result = result.union(es)
    return result


def effect_subset(sub: EffectSet, sup: EffectSet) -> bool:
    """
    Check if one effect set is a subset of another.
    
    Note: This is a conservative check - effect variables make
    subset checking undecidable in general.
    """
    # Concrete effects must be subsets
    if not sub.effects.issubset(sup.effects):
        return False
    
    # If sup has no effect vars, sub must also have none
    if not sup.effect_vars and sub.effect_vars:
        return False
    
    # If sup has effect vars, they could represent anything
    return True


class EffectChecker:
    """
    Checks effect-related constraints for Delta programs.
    """
    
    def __init__(self) -> None:
        self.errors: list[str] = []
    
    def check_mode_effects(
        self, 
        effects: EffectSet, 
        mode: str,
        context: str = ""
    ) -> bool:
        """
        Check if effects are allowed in a given mode.
        
        Rules:
        - train: all effects allowed
        - infer: no stochastic effects (unless explicit)
        - analyze: all effects allowed
        """
        if mode == "infer" and effects.has_stochastic():
            self.errors.append(
                f"Stochastic effect not allowed in infer mode{context}"
            )
            return False
        
        return True
    
    def check_learn_effects(self, effects: EffectSet, context: str = "") -> bool:
        """
        Check effects are allowed in a learn block.
        
        Rules:
        - non_diff is forbidden in learn objectives
        """
        if effects.has_non_diff():
            self.errors.append(
                f"Non-differentiable effect in learn block{context}"
            )
            return False
        
        return True
    
    def check_gradient_effects(
        self, 
        effects: EffectSet,
        requires_grad: bool,
        context: str = ""
    ) -> bool:
        """
        Check effect compatibility with gradient flow.
        
        Rules:
        - non_diff blocks cannot have gradients flowing through them
        """
        if effects.has_non_diff() and requires_grad:
            self.errors.append(
                f"Gradient flow through non-differentiable block{context}"
            )
            return False
        
        return True
