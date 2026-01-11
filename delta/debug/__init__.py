"""
Delta Debugging and Introspection System.

Provides compiler-level debugging capabilities beyond the stdlib.debug module.
"""

from delta.debug.introspection import (
    Introspector,
    IntrospectionMode,
    IntrospectionResult,
    introspect,
)
from delta.debug.why_system import (
    WhyEngine,
    ConstraintDiagnosis,
    GradientDiagnosis,
    OptimizationDiagnosis,
    diagnose_optimization,
)
from delta.debug.ir_printer import (
    print_sir,
    print_fx,
    format_sir,
    format_fx,
)

__all__ = [
    # Introspection
    "Introspector",
    "IntrospectionMode",
    "IntrospectionResult",
    "introspect",
    # Why system
    "WhyEngine",
    "ConstraintDiagnosis",
    "GradientDiagnosis",
    "OptimizationDiagnosis",
    "diagnose_optimization",
    # IR printing
    "print_sir",
    "print_fx",
    "format_sir",
    "format_fx",
]
