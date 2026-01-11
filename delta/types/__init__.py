"""
Delta type system.

Implements a Hindley-Milner style type inference extended with:
- Effect rows for tracking stochastic/non-diff effects
- Role propagation (param, obs)
- Mode constraints (train, infer, analyze)
- Tensor shape inference
"""

from delta.types.types import (
    Type, TypeVar, UnitType, BoolType, IntType, FloatType, StringType,
    TensorType, FunctionType, TupleType, StructType, GenericType,
    UnionType, NeverType, AnyType,
    Substitution, unify, occurs_check, apply_substitution,
    fresh_type_var, instantiate, generalize,
)
from delta.types.effects import (
    Effect, EffectSet, EffectVar,
    PURE, STOCHASTIC, NON_DIFF, IO,
    effect_union, effect_subset,
)
from delta.types.roles import (
    Role, RoleSet, PARAM, OBS, CONST,
    check_role_compatibility,
)
from delta.types.inference import TypeInference, TypeEnvironment
from delta.types.checker import TypeChecker

__all__ = [
    # Types
    "Type", "TypeVar", "UnitType", "BoolType", "IntType", "FloatType", "StringType",
    "TensorType", "FunctionType", "TupleType", "StructType", "GenericType",
    "UnionType", "NeverType", "AnyType",
    "Substitution", "unify", "occurs_check", "apply_substitution",
    "fresh_type_var", "instantiate", "generalize",
    # Effects
    "Effect", "EffectSet", "EffectVar",
    "PURE", "STOCHASTIC", "NON_DIFF", "IO",
    "effect_union", "effect_subset",
    # Roles
    "Role", "RoleSet", "PARAM", "OBS", "CONST",
    "check_role_compatibility",
    # Inference
    "TypeInference", "TypeEnvironment",
    "TypeChecker",
]
