"""
Delta runtime: execution model and caching.

The runtime handles:
- Graph execution
- Parameter management
- Compilation caching
- Optimizer integration
"""

from delta.runtime.executor import Executor, ExecutionContext
from delta.runtime.cache import CompilationCache, CacheKey
from delta.runtime.optimizer import OptimizerWrapper, create_optimizer
from delta.runtime.context import DeltaContext, TrainingContext

__all__ = [
    "Executor", "ExecutionContext",
    "CompilationCache", "CacheKey",
    "OptimizerWrapper", "create_optimizer",
    "DeltaContext", "TrainingContext",
]
