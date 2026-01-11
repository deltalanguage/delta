"""
Type definitions for Delta.

Implements the core type system with support for:
- Primitive types (Bool, Int, Float, String)
- Tensor types with shape information
- Function types
- Polymorphic types with type variables
- Type unification for inference
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, TypeVar as PyTypeVar, Generic
from enum import Enum, auto
import itertools


# Type variable counter for fresh names
_type_var_counter = itertools.count()


def fresh_type_var(prefix: str = "T") -> TypeVar:
    """Create a fresh type variable."""
    return TypeVar(name=f"{prefix}{next(_type_var_counter)}")


class Type(ABC):
    """
    Base class for all types in Delta.
    
    Types are immutable and support structural equality.
    """
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        pass
    
    @abstractmethod
    def free_vars(self) -> set[TypeVar]:
        """Get all free type variables in this type."""
        pass
    
    @abstractmethod
    def substitute(self, subst: Substitution) -> Type:
        """Apply a substitution to this type."""
        pass
    
    def contains(self, var: TypeVar) -> bool:
        """Check if this type contains a type variable."""
        return var in self.free_vars()


@dataclass(frozen=True)
class TypeVar(Type):
    """
    Type variable for polymorphism.
    
    Type variables are unified during type inference and can be
    substituted with concrete types.
    """
    name: str
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, TypeVar) and self.name == other.name
    
    def __hash__(self) -> int:
        return hash(("TypeVar", self.name))
    
    def free_vars(self) -> set[TypeVar]:
        return {self}
    
    def substitute(self, subst: Substitution) -> Type:
        if self in subst:
            return subst[self]
        return self


@dataclass(frozen=True)
class UnitType(Type):
    """Unit type (void/none)."""
    
    def __str__(self) -> str:
        return "Unit"
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnitType)
    
    def __hash__(self) -> int:
        return hash("UnitType")
    
    def free_vars(self) -> set[TypeVar]:
        return set()
    
    def substitute(self, subst: Substitution) -> Type:
        return self


@dataclass(frozen=True)
class BoolType(Type):
    """Boolean type."""
    
    def __str__(self) -> str:
        return "Bool"
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, BoolType)
    
    def __hash__(self) -> int:
        return hash("BoolType")
    
    def free_vars(self) -> set[TypeVar]:
        return set()
    
    def substitute(self, subst: Substitution) -> Type:
        return self


@dataclass(frozen=True)
class IntType(Type):
    """Integer type."""
    
    def __str__(self) -> str:
        return "Int"
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, IntType)
    
    def __hash__(self) -> int:
        return hash("IntType")
    
    def free_vars(self) -> set[TypeVar]:
        return set()
    
    def substitute(self, subst: Substitution) -> Type:
        return self


@dataclass(frozen=True)
class FloatType(Type):
    """Floating-point type."""
    
    def __str__(self) -> str:
        return "Float"
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, FloatType)
    
    def __hash__(self) -> int:
        return hash("FloatType")
    
    def free_vars(self) -> set[TypeVar]:
        return set()
    
    def substitute(self, subst: Substitution) -> Type:
        return self


@dataclass(frozen=True)
class StringType(Type):
    """String type."""
    
    def __str__(self) -> str:
        return "String"
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, StringType)
    
    def __hash__(self) -> int:
        return hash("StringType")
    
    def free_vars(self) -> set[TypeVar]:
        return set()
    
    def substitute(self, subst: Substitution) -> Type:
        return self


class ShapeDim:
    """
    Represents a dimension in a tensor shape.
    
    Can be:
    - A concrete integer
    - A symbolic dimension variable
    - Dynamic (unknown at compile time)
    """
    pass


@dataclass(frozen=True)
class ConcreteDim(ShapeDim):
    """A concrete integer dimension."""
    value: int
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class SymbolicDim(ShapeDim):
    """A symbolic dimension variable."""
    name: str
    
    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class DynamicDim(ShapeDim):
    """A dynamic dimension (unknown at compile time)."""
    
    def __str__(self) -> str:
        return "?"


@dataclass(frozen=True)
class TensorType(Type):
    """
    Tensor type with element type and optional shape.
    
    Examples:
    - Tensor[Float]           # Unknown shape
    - Tensor[Float, 32, 10]   # Concrete shape
    - Tensor[Float, batch, 10]  # Symbolic shape
    """
    element_type: Type
    shape: Optional[tuple[ShapeDim, ...]] = None
    
    def __str__(self) -> str:
        if self.shape is None:
            return f"Tensor[{self.element_type}]"
        shape_str = ", ".join(str(d) for d in self.shape)
        return f"Tensor[{self.element_type}, {shape_str}]"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorType):
            return False
        return self.element_type == other.element_type and self.shape == other.shape
    
    def __hash__(self) -> int:
        return hash(("TensorType", self.element_type, self.shape))
    
    def free_vars(self) -> set[TypeVar]:
        return self.element_type.free_vars()
    
    def substitute(self, subst: Substitution) -> Type:
        return TensorType(
            element_type=self.element_type.substitute(subst),
            shape=self.shape
        )
    
    @property
    def rank(self) -> Optional[int]:
        """Get the rank (number of dimensions) if known."""
        return len(self.shape) if self.shape else None
    
    def compatible_with(self, other: TensorType) -> bool:
        """Check if shapes are compatible for broadcasting."""
        if self.shape is None or other.shape is None:
            return True
        
        # Check element type
        if self.element_type != other.element_type:
            return False
        
        # Broadcasting rules
        for d1, d2 in zip(reversed(self.shape), reversed(other.shape)):
            if isinstance(d1, ConcreteDim) and isinstance(d2, ConcreteDim):
                if d1.value != d2.value and d1.value != 1 and d2.value != 1:
                    return False
        
        return True


@dataclass(frozen=True)
class FunctionType(Type):
    """
    Function type.
    
    Includes parameter types, return type, and effect signature.
    
    If variadic=True, the function accepts any number of arguments
    (all args are checked against the first param_type if provided).
    """
    param_types: tuple[Type, ...]
    return_type: Type
    effects: frozenset[str] = field(default_factory=frozenset)
    variadic: bool = False  # Accept any number of arguments
    min_args: int = 0  # Minimum required arguments for variadic functions
    
    def __str__(self) -> str:
        params = ", ".join(str(p) for p in self.param_types)
        effect_str = ""
        if self.effects:
            effect_str = f" / {', '.join(sorted(self.effects))}"
        return f"fn({params}) -> {self.return_type}{effect_str}"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionType):
            return False
        return (
            self.param_types == other.param_types and
            self.return_type == other.return_type and
            self.effects == other.effects
        )
    
    def __hash__(self) -> int:
        return hash(("FunctionType", self.param_types, self.return_type, self.effects, self.variadic))
    
    def free_vars(self) -> set[TypeVar]:
        result: set[TypeVar] = set()
        for p in self.param_types:
            result |= p.free_vars()
        result |= self.return_type.free_vars()
        return result
    
    def substitute(self, subst: Substitution) -> Type:
        return FunctionType(
            param_types=tuple(p.substitute(subst) for p in self.param_types),
            return_type=self.return_type.substitute(subst),
            effects=self.effects,
            variadic=self.variadic,
            min_args=self.min_args
        )


@dataclass(frozen=True)
class TupleType(Type):
    """Tuple type."""
    element_types: tuple[Type, ...]
    
    def __str__(self) -> str:
        if not self.element_types:
            return "()"
        elements = ", ".join(str(t) for t in self.element_types)
        return f"({elements})"
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, TupleType) and self.element_types == other.element_types
    
    def __hash__(self) -> int:
        return hash(("TupleType", self.element_types))
    
    def free_vars(self) -> set[TypeVar]:
        result: set[TypeVar] = set()
        for t in self.element_types:
            result |= t.free_vars()
        return result
    
    def substitute(self, subst: Substitution) -> Type:
        return TupleType(
            element_types=tuple(t.substitute(subst) for t in self.element_types)
        )


@dataclass(frozen=True)
class StructType(Type):
    """Struct (record) type."""
    name: str
    fields: tuple[tuple[str, Type], ...]
    type_params: tuple[Type, ...] = ()
    
    def __str__(self) -> str:
        if self.type_params:
            params = ", ".join(str(t) for t in self.type_params)
            return f"{self.name}[{params}]"
        return self.name
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StructType):
            return False
        return self.name == other.name and self.type_params == other.type_params
    
    def __hash__(self) -> int:
        return hash(("StructType", self.name, self.type_params))
    
    def free_vars(self) -> set[TypeVar]:
        result: set[TypeVar] = set()
        for _, t in self.fields:
            result |= t.free_vars()
        for t in self.type_params:
            result |= t.free_vars()
        return result
    
    def substitute(self, subst: Substitution) -> Type:
        return StructType(
            name=self.name,
            fields=tuple((n, t.substitute(subst)) for n, t in self.fields),
            type_params=tuple(t.substitute(subst) for t in self.type_params)
        )
    
    def get_field(self, name: str) -> Optional[Type]:
        """Get the type of a field by name."""
        for fname, ftype in self.fields:
            if fname == name:
                return ftype
        return None


@dataclass(frozen=True)
class GenericType(Type):
    """Generic type with type arguments."""
    name: str
    type_args: tuple[Type, ...]
    
    def __str__(self) -> str:
        if not self.type_args:
            return self.name
        args = ", ".join(str(t) for t in self.type_args)
        return f"{self.name}[{args}]"
    
    def __eq__(self, other: object) -> bool:
        return (isinstance(other, GenericType) and 
                self.name == other.name and 
                self.type_args == other.type_args)
    
    def __hash__(self) -> int:
        return hash(("GenericType", self.name, self.type_args))
    
    def free_vars(self) -> set[TypeVar]:
        result: set[TypeVar] = set()
        for t in self.type_args:
            result |= t.free_vars()
        return result
    
    def substitute(self, subst: Substitution) -> Type:
        return GenericType(
            name=self.name,
            type_args=tuple(t.substitute(subst) for t in self.type_args)
        )


@dataclass(frozen=True)
class UnionType(Type):
    """Union type (sum type)."""
    types: frozenset[Type]
    
    def __str__(self) -> str:
        return " | ".join(str(t) for t in sorted(self.types, key=str))
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnionType) and self.types == other.types
    
    def __hash__(self) -> int:
        return hash(("UnionType", self.types))
    
    def free_vars(self) -> set[TypeVar]:
        result: set[TypeVar] = set()
        for t in self.types:
            result |= t.free_vars()
        return result
    
    def substitute(self, subst: Substitution) -> Type:
        return UnionType(
            types=frozenset(t.substitute(subst) for t in self.types)
        )


@dataclass(frozen=True)
class NeverType(Type):
    """Bottom type (no values)."""
    
    def __str__(self) -> str:
        return "Never"
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, NeverType)
    
    def __hash__(self) -> int:
        return hash("NeverType")
    
    def free_vars(self) -> set[TypeVar]:
        return set()
    
    def substitute(self, subst: Substitution) -> Type:
        return self


@dataclass(frozen=True)
class AnyType(Type):
    """Top type (any value)."""
    
    def __str__(self) -> str:
        return "Any"
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, AnyType)
    
    def __hash__(self) -> int:
        return hash("AnyType")
    
    def free_vars(self) -> set[TypeVar]:
        return set()
    
    def substitute(self, subst: Substitution) -> Type:
        return self


# Type Substitution


class Substitution(dict[TypeVar, Type]):
    """
    A mapping from type variables to types.
    
    Used during unification to track type variable bindings.
    """
    
    def compose(self, other: Substitution) -> Substitution:
        """
        Compose two substitutions.
        
        (self . other)(x) = self(other(x))
        """
        result = Substitution()
        
        # Apply self to all bindings in other
        for var, typ in other.items():
            result[var] = apply_substitution(typ, self)
        
        # Add bindings from self that aren't in other
        for var, typ in self.items():
            if var not in result:
                result[var] = typ
        
        return result
    
    def __str__(self) -> str:
        bindings = ", ".join(f"{k} -> {v}" for k, v in self.items())
        return f"{{{bindings}}}"


def apply_substitution(typ: Type, subst: Substitution) -> Type:
    """Apply a substitution to a type."""
    return typ.substitute(subst)


def occurs_check(var: TypeVar, typ: Type) -> bool:
    """
    Check if a type variable occurs in a type.
    
    Used to prevent infinite types during unification.
    """
    return var in typ.free_vars()


class UnificationError(Exception):
    """Error during type unification."""
    def __init__(self, t1: Type, t2: Type, message: str = ""):
        self.t1 = t1
        self.t2 = t2
        self.message = message or f"Cannot unify {t1} with {t2}"
        super().__init__(self.message)


def unify(t1: Type, t2: Type) -> Substitution:
    """
    Unify two types, returning a substitution.
    
    Raises UnificationError if the types cannot be unified.
    """
    # Same types
    if t1 == t2:
        return Substitution()
    
    # Type variables
    if isinstance(t1, TypeVar):
        if occurs_check(t1, t2):
            raise UnificationError(t1, t2, f"Infinite type: {t1} occurs in {t2}")
        return Substitution({t1: t2})
    
    if isinstance(t2, TypeVar):
        if occurs_check(t2, t1):
            raise UnificationError(t1, t2, f"Infinite type: {t2} occurs in {t1}")
        return Substitution({t2: t1})
    
    # Any type unifies with anything
    if isinstance(t1, AnyType) or isinstance(t2, AnyType):
        return Substitution()
    
    # Never type unifies with anything (subtype of all types)
    if isinstance(t1, NeverType):
        return Substitution()
    if isinstance(t2, NeverType):
        return Substitution()
    
    # Tensor types
    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return unify_tensors(t1, t2)
    
    # Function types
    if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
        return unify_functions(t1, t2)
    
    # Tuple types
    if isinstance(t1, TupleType) and isinstance(t2, TupleType):
        return unify_tuples(t1, t2)
    
    # Struct types
    if isinstance(t1, StructType) and isinstance(t2, StructType):
        if t1.name != t2.name:
            raise UnificationError(t1, t2)
        return unify_type_args(list(t1.type_params), list(t2.type_params))
    
    # Generic types
    if isinstance(t1, GenericType) and isinstance(t2, GenericType):
        if t1.name != t2.name:
            raise UnificationError(t1, t2)
        return unify_type_args(list(t1.type_args), list(t2.type_args))
    
    raise UnificationError(t1, t2)


def unify_tensors(t1: TensorType, t2: TensorType) -> Substitution:
    """Unify two tensor types."""
    subst = unify(t1.element_type, t2.element_type)
    
    # Shape unification (simplified - just check compatibility)
    if t1.shape is not None and t2.shape is not None:
        if len(t1.shape) != len(t2.shape):
            raise UnificationError(t1, t2, "Incompatible tensor ranks")
        
        for d1, d2 in zip(t1.shape, t2.shape):
            if isinstance(d1, ConcreteDim) and isinstance(d2, ConcreteDim):
                if d1.value != d2.value:
                    raise UnificationError(t1, t2, f"Incompatible dimensions: {d1} vs {d2}")
    
    return subst


def unify_functions(t1: FunctionType, t2: FunctionType) -> Substitution:
    """Unify two function types."""
    if len(t1.param_types) != len(t2.param_types):
        raise UnificationError(t1, t2, "Different number of parameters")
    
    subst = Substitution()
    
    # Unify parameter types
    for p1, p2 in zip(t1.param_types, t2.param_types):
        s = unify(apply_substitution(p1, subst), apply_substitution(p2, subst))
        subst = subst.compose(s)
    
    # Unify return types
    s = unify(
        apply_substitution(t1.return_type, subst),
        apply_substitution(t2.return_type, subst)
    )
    subst = subst.compose(s)
    
    return subst


def unify_tuples(t1: TupleType, t2: TupleType) -> Substitution:
    """Unify two tuple types."""
    if len(t1.element_types) != len(t2.element_types):
        raise UnificationError(t1, t2, "Different tuple lengths")
    
    return unify_type_args(list(t1.element_types), list(t2.element_types))


def unify_type_args(args1: list[Type], args2: list[Type]) -> Substitution:
    """Unify two lists of type arguments."""
    if len(args1) != len(args2):
        raise UnificationError(
            TupleType(tuple(args1)), 
            TupleType(tuple(args2)),
            "Different number of type arguments"
        )
    
    subst = Substitution()
    for a1, a2 in zip(args1, args2):
        s = unify(apply_substitution(a1, subst), apply_substitution(a2, subst))
        subst = subst.compose(s)
    
    return subst


# Type Schemes (Polymorphic Types)


@dataclass
class TypeScheme:
    """
    A polymorphic type scheme: ∀ α₁...αₙ. τ
    
    Used for let-polymorphism in Hindley-Milner.
    """
    type_vars: frozenset[TypeVar]
    body: Type
    
    def __str__(self) -> str:
        if not self.type_vars:
            return str(self.body)
        vars_str = ", ".join(str(v) for v in self.type_vars)
        return f"∀{vars_str}. {self.body}"
    
    def free_vars(self) -> set[TypeVar]:
        """Free type variables (not quantified)."""
        return self.body.free_vars() - set(self.type_vars)


def instantiate(scheme: TypeScheme) -> Type:
    """
    Instantiate a type scheme with fresh type variables.
    """
    subst = Substitution()
    for var in scheme.type_vars:
        subst[var] = fresh_type_var(var.name)
    return apply_substitution(scheme.body, subst)


def generalize(typ: Type, env_free_vars: set[TypeVar]) -> TypeScheme:
    """
    Generalize a type to a type scheme.
    
    Quantifies over all type variables not free in the environment.
    """
    free = typ.free_vars() - env_free_vars
    return TypeScheme(type_vars=frozenset(free), body=typ)
