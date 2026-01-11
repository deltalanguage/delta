"""
Compiler passes for Delta.

Implements the compiler pipeline passes:
1. Name resolution
2. Role assignment
3. Effect inference
4. Mode admissibility check
5. Relaxation (differentiability)
6. Constraint compilation
7. Mode specialization
8. Gradient liveness analysis
9. Dead gradient elimination
"""

from delta.passes.name_resolution import NameResolver, resolve_names
from delta.passes.role_assignment import RoleAssigner, assign_roles
from delta.passes.effect_inference import EffectInferrer, infer_effects
from delta.passes.mode_check import ModeChecker, check_mode
from delta.passes.relaxation import RelaxationPass, relax_sir
from delta.passes.constraint_compile import ConstraintCompiler, compile_constraints
from delta.passes.mode_specialize import ModeSpecializer, specialize_mode
from delta.passes.gradient_analysis import GradientAnalyzer, analyze_gradients
from delta.passes.dead_gradient_elim import DeadGradientEliminator, eliminate_dead_gradients

__all__ = [
    "NameResolver", "resolve_names",
    "RoleAssigner", "assign_roles",
    "EffectInferrer", "infer_effects",
    "ModeChecker", "check_mode",
    "RelaxationPass", "relax_sir",
    "ConstraintCompiler", "compile_constraints",
    "ModeSpecializer", "specialize_mode",
    "GradientAnalyzer", "analyze_gradients",
    "DeadGradientEliminator", "eliminate_dead_gradients",
]
