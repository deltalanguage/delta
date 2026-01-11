"""
Role system for Delta.

Roles track the learning semantics of values:
- param: Learnable parameters (gradient sink)
- obs: Observed data (gradient boundary)
- const: Compile-time constants
- computed: Derived values

Role propagation ensures gradients flow correctly and
prevents invalid operations like gradient into obs.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, FrozenSet
from enum import Enum, auto


class Role(Enum):
    """
    Roles for values in Delta.
    
    Roles track learning semantics and gradient flow.
    """
    PARAM = auto()      # Learnable parameter
    OBS = auto()        # Observed/fixed value
    CONST = auto()      # Compile-time constant
    COMPUTED = auto()   # Computed from other values
    UNKNOWN = auto()    # Not yet determined
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @property
    def requires_grad(self) -> bool:
        """Check if this role requires gradients."""
        return self == Role.PARAM
    
    @property
    def is_fixed(self) -> bool:
        """Check if this role is fixed (not learnable)."""
        return self in (Role.OBS, Role.CONST)


# Convenience constants
PARAM = Role.PARAM
OBS = Role.OBS
CONST = Role.CONST
COMPUTED = Role.COMPUTED


@dataclass(frozen=True)
class RoleSet:
    """
    A set of roles that contribute to a value.
    
    Used to track role provenance through computations.
    """
    roles: frozenset[Role]
    
    def __str__(self) -> str:
        if not self.roles:
            return "none"
        return " + ".join(r.name.lower() for r in sorted(self.roles, key=lambda r: r.name))
    
    def __eq__(self, other: object) -> bool:
        return isinstance(other, RoleSet) and self.roles == other.roles
    
    def __hash__(self) -> int:
        return hash(self.roles)
    
    def has_param(self) -> bool:
        """Check if this includes param role."""
        return Role.PARAM in self.roles
    
    def has_obs(self) -> bool:
        """Check if this includes obs role."""
        return Role.OBS in self.roles
    
    def has_const(self) -> bool:
        """Check if this includes const role."""
        return Role.CONST in self.roles
    
    def requires_grad(self) -> bool:
        """Check if gradients should be tracked."""
        return self.has_param()
    
    def union(self, other: RoleSet) -> RoleSet:
        """Union of two role sets."""
        return RoleSet(roles=self.roles | other.roles)
    
    @classmethod
    def empty(cls) -> RoleSet:
        """Create an empty role set."""
        return cls(roles=frozenset())
    
    @classmethod
    def param(cls) -> RoleSet:
        """Create a param role set."""
        return cls(roles=frozenset({Role.PARAM}))
    
    @classmethod
    def obs(cls) -> RoleSet:
        """Create an obs role set."""
        return cls(roles=frozenset({Role.OBS}))
    
    @classmethod
    def const(cls) -> RoleSet:
        """Create a const role set."""
        return cls(roles=frozenset({Role.CONST}))
    
    @classmethod
    def computed(cls) -> RoleSet:
        """Create a computed role set."""
        return cls(roles=frozenset({Role.COMPUTED}))


def check_role_compatibility(
    source_role: RoleSet,
    target_role: Role,
    context: str = ""
) -> tuple[bool, Optional[str]]:
    """
    Check if a role assignment is valid.
    
    Rules:
    - Cannot assign param to obs variable
    - Cannot flow gradients into obs
    """
    # Cannot assign param-derived value to obs variable
    if source_role.has_param() and target_role == Role.OBS:
        return False, f"Cannot assign param-derived value to obs{context}"
    
    return True, None


def propagate_roles(*role_sets: RoleSet) -> RoleSet:
    """
    Propagate roles through a computation.
    
    The result has the union of all input roles.
    """
    result = RoleSet.empty()
    for rs in role_sets:
        result = result.union(rs)
    return result


class RoleChecker:
    """
    Checks role-related constraints for Delta programs.
    """
    
    def __init__(self) -> None:
        self.errors: list[str] = []
    
    def check_gradient_target(
        self,
        role: RoleSet,
        context: str = ""
    ) -> bool:
        """
        Check if a value can be a gradient target.
        
        Rules:
        - obs values cannot be gradient sinks
        """
        if role.has_obs() and not role.has_param():
            self.errors.append(
                f"Cannot compute gradients with respect to obs value{context}"
            )
            return False
        
        return True
    
    def check_gradient_source(
        self,
        source_role: RoleSet,
        target_role: RoleSet,
        context: str = ""
    ) -> bool:
        """
        Check if gradient flow between roles is valid.
        
        Rules:
        - Gradients from param cannot flow into obs
        """
        if source_role.has_param() and target_role.has_obs():
            self.errors.append(
                f"Gradient flow from param to obs{context}"
            )
            return False
        
        return True
    
    def check_learn_block_roles(
        self,
        roles: RoleSet,
        context: str = ""
    ) -> bool:
        """
        Check roles are valid in a learn block context.
        
        Rules:
        - Must have at least one param to learn
        """
        if not roles.has_param():
            self.errors.append(
                f"Learn block has no learnable parameters{context}"
            )
            return False
        
        return True


@dataclass
class RoleInfo:
    """
    Complete role information for a value.
    
    Tracks:
    - The primary role
    - Contributing roles (for computed values)
    - Whether gradients should be tracked
    """
    primary_role: Role
    contributing_roles: RoleSet
    requires_grad: bool
    gradient_checkpointed: bool = False
    
    def __str__(self) -> str:
        return f"{self.primary_role}(grad={self.requires_grad})"
    
    @classmethod
    def param(cls) -> RoleInfo:
        """Create role info for a parameter."""
        return cls(
            primary_role=Role.PARAM,
            contributing_roles=RoleSet.param(),
            requires_grad=True
        )
    
    @classmethod
    def obs(cls) -> RoleInfo:
        """Create role info for an observation."""
        return cls(
            primary_role=Role.OBS,
            contributing_roles=RoleSet.obs(),
            requires_grad=False
        )
    
    @classmethod
    def const(cls) -> RoleInfo:
        """Create role info for a constant."""
        return cls(
            primary_role=Role.CONST,
            contributing_roles=RoleSet.const(),
            requires_grad=False
        )
    
    @classmethod
    def computed(cls, *inputs: RoleInfo) -> RoleInfo:
        """Create role info for a computed value."""
        contributing = RoleSet.empty()
        requires_grad = False
        
        for inp in inputs:
            contributing = contributing.union(inp.contributing_roles)
            requires_grad = requires_grad or inp.requires_grad
        
        return cls(
            primary_role=Role.COMPUTED,
            contributing_roles=contributing,
            requires_grad=requires_grad
        )
