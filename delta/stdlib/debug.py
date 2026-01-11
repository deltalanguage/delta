"""
std.debug - Debugging, introspection, and the "why" system for Delta.

Provides tools for understanding constraint violations, gradient flow,
and diagnosing optimization issues.
"""

from __future__ import annotations
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union, NamedTuple
)
from dataclasses import dataclass, field
from enum import Enum, auto
from contextlib import contextmanager
import torch
from torch import Tensor
import weakref


# ============================================================
# Trace Recording
# ============================================================

class TraceKind(Enum):
    """Type of trace event."""
    FORWARD = auto()
    BACKWARD = auto()
    CONSTRAINT = auto()
    GRADIENT = auto()
    MODE_SWITCH = auto()
    RELAXATION = auto()


@dataclass
class TraceEvent:
    """A single trace event."""
    kind: TraceKind
    name: str
    value: Optional[Tensor]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    
    def __repr__(self) -> str:
        shape = None
        if self.value is not None and isinstance(self.value, Tensor):
            shape = tuple(self.value.shape)
        return f"TraceEvent({self.kind.name}, {self.name}, shape={shape})"


class TraceRecorder:
    """Records execution traces for debugging."""
    
    _instance: Optional['TraceRecorder'] = None
    
    def __init__(self) -> None:
        self.events: List[TraceEvent] = []
        self.enabled = False
        self._counter = 0
    
    @classmethod
    def get(cls) -> 'TraceRecorder':
        if cls._instance is None:
            cls._instance = TraceRecorder()
        return cls._instance
    
    def start(self) -> None:
        """Start recording traces."""
        self.enabled = True
        self.events = []
        self._counter = 0
    
    def stop(self) -> List[TraceEvent]:
        """Stop recording and return events."""
        self.enabled = False
        return self.events
    
    def record(
        self,
        kind: TraceKind,
        name: str,
        value: Optional[Tensor] = None,
        **metadata: Any
    ) -> None:
        """Record a trace event."""
        if not self.enabled:
            return
        
        self.events.append(TraceEvent(
            kind=kind,
            name=name,
            value=value.detach().clone() if value is not None else None,
            metadata=metadata,
            timestamp=self._counter
        ))
        self._counter += 1
    
    def clear(self) -> None:
        """Clear recorded events."""
        self.events = []
        self._counter = 0


@contextmanager
def trace():
    """Context manager for tracing execution."""
    recorder = TraceRecorder.get()
    recorder.start()
    try:
        yield recorder
    finally:
        recorder.stop()


# ============================================================
# Constraint Attribution (the "why" system)
# ============================================================

@dataclass
class ConstraintViolation:
    """Information about a constraint violation."""
    name: str
    violation: float
    weight: float
    contribution: float  # weighted violation
    location: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    gradient_magnitude: float = 0.0


@dataclass
class WhyReport:
    """Report explaining optimization state."""
    total_loss: float
    constraint_violations: List[ConstraintViolation]
    gradient_norms: Dict[str, float]
    saturation_warnings: List[str]
    recommendations: List[str]
    
    def __str__(self) -> str:
        lines = ["=" * 60]
        lines.append("WHY REPORT")
        lines.append("=" * 60)
        lines.append(f"\nTotal Loss: {self.total_loss:.6f}\n")
        
        if self.constraint_violations:
            lines.append("CONSTRAINT VIOLATIONS:")
            lines.append("-" * 40)
            for cv in sorted(self.constraint_violations, 
                           key=lambda x: -x.contribution):
                lines.append(
                    f"  {cv.name}: violation={cv.violation:.4f}, "
                    f"weight={cv.weight:.2f}, contribution={cv.contribution:.4f}"
                )
                if cv.gradient_magnitude > 0:
                    lines.append(f"    gradient_norm={cv.gradient_magnitude:.6f}")
        
        if self.gradient_norms:
            lines.append("\nGRADIENT NORMS:")
            lines.append("-" * 40)
            for name, norm in sorted(self.gradient_norms.items(),
                                    key=lambda x: -x[1]):
                lines.append(f"  {name}: {norm:.6f}")
        
        if self.saturation_warnings:
            lines.append("\nSATURATION WARNINGS:")
            lines.append("-" * 40)
            for warning in self.saturation_warnings:
                lines.append(f"  ⚠ {warning}")
        
        if self.recommendations:
            lines.append("\nRECOMMENDATIONS:")
            lines.append("-" * 40)
            for rec in self.recommendations:
                lines.append(f"  → {rec}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


class WhyAnalyzer:
    """Analyzes optimization state and provides explanations."""
    
    def __init__(self) -> None:
        self._constraint_history: List[Dict[str, float]] = []
        self._loss_history: List[float] = []
        self._gradient_history: List[Dict[str, float]] = []
    
    def record_step(
        self,
        loss: float,
        constraints: Dict[str, Tuple[float, float]],  # name -> (violation, weight)
        gradients: Dict[str, Tensor]
    ) -> None:
        """Record a training step for analysis."""
        self._loss_history.append(loss)
        
        constraint_contributions = {}
        for name, (violation, weight) in constraints.items():
            constraint_contributions[name] = violation * weight
        self._constraint_history.append(constraint_contributions)
        
        grad_norms = {}
        for name, grad in gradients.items():
            if grad is not None:
                grad_norms[name] = grad.norm().item()
        self._gradient_history.append(grad_norms)
    
    def why(
        self,
        constraints: Dict[str, Tuple[float, float, Tensor]],  # name -> (violation, weight, tensor)
        gradients: Dict[str, Tensor],
        loss: Optional[float] = None
    ) -> WhyReport:
        """Generate a why report for current state."""
        violations = []
        
        for name, (violation_val, weight, tensor) in constraints.items():
            grad_mag = 0.0
            if tensor.grad is not None:
                grad_mag = tensor.grad.norm().item()
            
            violations.append(ConstraintViolation(
                name=name,
                violation=violation_val,
                weight=weight,
                contribution=violation_val * weight,
                gradient_magnitude=grad_mag
            ))
        
        # Compute gradient norms
        grad_norms = {}
        for name, grad in gradients.items():
            if grad is not None:
                grad_norms[name] = grad.norm().item()
        
        # Detect saturation
        saturation_warnings = self._detect_saturation(gradients)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            violations, grad_norms, saturation_warnings
        )
        
        return WhyReport(
            total_loss=loss if loss is not None else 0.0,
            constraint_violations=violations,
            gradient_norms=grad_norms,
            saturation_warnings=saturation_warnings,
            recommendations=recommendations
        )
    
    def _detect_saturation(
        self,
        gradients: Dict[str, Tensor]
    ) -> List[str]:
        """Detect saturated gradients."""
        warnings = []
        
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            grad_norm = grad.norm().item()
            
            # Near-zero gradient
            if grad_norm < 1e-7:
                warnings.append(f"{name}: gradient near zero (saturation)")
            
            # Exploding gradient
            if grad_norm > 1e6:
                warnings.append(f"{name}: gradient exploding")
            
            # Check for NaN/Inf
            if torch.isnan(grad).any():
                warnings.append(f"{name}: NaN in gradient")
            if torch.isinf(grad).any():
                warnings.append(f"{name}: Inf in gradient")
        
        return warnings
    
    def _generate_recommendations(
        self,
        violations: List[ConstraintViolation],
        grad_norms: Dict[str, float],
        saturation_warnings: List[str]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Check for dominating constraints
        if violations:
            total = sum(v.contribution for v in violations)
            if total > 0:
                for v in violations:
                    ratio = v.contribution / total
                    if ratio > 0.8:
                        recommendations.append(
                            f"Constraint '{v.name}' dominates loss ({ratio:.0%}). "
                            f"Consider reducing its weight."
                        )
        
        # Check for vanishing gradients
        if saturation_warnings:
            if any("near zero" in w for w in saturation_warnings):
                recommendations.append(
                    "Vanishing gradients detected. Consider:\n"
                    "  - Reducing relaxation temperature\n"
                    "  - Using gradient clipping\n"
                    "  - Checking for over-constrained problem"
                )
        
        # Check for gradient imbalance
        if grad_norms:
            norms = list(grad_norms.values())
            if norms:
                max_norm = max(norms)
                min_norm = min(n for n in norms if n > 0) if any(n > 0 for n in norms) else 1
                if max_norm / (min_norm + 1e-8) > 1000:
                    recommendations.append(
                        "Severe gradient imbalance detected. "
                        "Consider per-parameter learning rates."
                    )
        
        # Check loss history for stagnation
        if len(self._loss_history) >= 10:
            recent = self._loss_history[-10:]
            if max(recent) - min(recent) < 1e-6:
                recommendations.append(
                    "Loss appears stagnant. Consider:\n"
                    "  - Increasing learning rate\n"
                    "  - Relaxing constraints\n"
                    "  - Checking for local minimum"
                )
        
        return recommendations
    
    def clear_history(self) -> None:
        """Clear recorded history."""
        self._constraint_history = []
        self._loss_history = []
        self._gradient_history = []


# Global analyzer instance
_analyzer = WhyAnalyzer()


def why(
    constraints: Dict[str, Tuple[float, float, Tensor]],
    gradients: Dict[str, Tensor],
    loss: Optional[float] = None
) -> WhyReport:
    """Generate a why report explaining the current optimization state."""
    return _analyzer.why(constraints, gradients, loss)


def record_step(
    loss: float,
    constraints: Dict[str, Tuple[float, float]],
    gradients: Dict[str, Tensor]
) -> None:
    """Record a training step for later analysis."""
    _analyzer.record_step(loss, constraints, gradients)


# ============================================================
# Gradient Flow Visualization
# ============================================================

@dataclass
class GradientFlowNode:
    """Node in gradient flow graph."""
    name: str
    tensor: Optional[Tensor]
    grad_norm: float
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)


class GradientFlowTracer:
    """Traces gradient flow through computation graph."""
    
    def __init__(self) -> None:
        self._nodes: Dict[str, GradientFlowNode] = {}
        self._hooks: List[Any] = []
    
    def trace(
        self,
        loss: Tensor,
        named_tensors: Dict[str, Tensor]
    ) -> Dict[str, GradientFlowNode]:
        """Trace gradient flow from loss to named tensors."""
        # Run backward to populate gradients
        loss.backward(retain_graph=True)
        
        # Collect gradient information
        nodes = {}
        for name, tensor in named_tensors.items():
            grad_norm = 0.0
            if tensor.grad is not None:
                grad_norm = tensor.grad.norm().item()
            
            nodes[name] = GradientFlowNode(
                name=name,
                tensor=tensor,
                grad_norm=grad_norm
            )
        
        return nodes
    
    def visualize_text(
        self,
        nodes: Dict[str, GradientFlowNode]
    ) -> str:
        """Create text visualization of gradient flow."""
        lines = ["Gradient Flow:"]
        lines.append("-" * 40)
        
        sorted_nodes = sorted(
            nodes.values(),
            key=lambda n: -n.grad_norm
        )
        
        max_norm = max(n.grad_norm for n in sorted_nodes) if sorted_nodes else 1.0
        
        for node in sorted_nodes:
            bar_len = int(30 * node.grad_norm / (max_norm + 1e-8))
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(f"{node.name:20} |{bar}| {node.grad_norm:.2e}")
        
        return "\n".join(lines)


# ============================================================
# Tensor Inspection
# ============================================================

@dataclass
class TensorStats:
    """Statistics about a tensor."""
    shape: Tuple[int, ...]
    dtype: str
    device: str
    min: float
    max: float
    mean: float
    std: float
    nan_count: int
    inf_count: int
    zero_count: int
    grad_norm: Optional[float]
    
    def __str__(self) -> str:
        lines = [
            f"Shape: {self.shape}",
            f"Dtype: {self.dtype}",
            f"Device: {self.device}",
            f"Range: [{self.min:.4f}, {self.max:.4f}]",
            f"Mean: {self.mean:.4f}, Std: {self.std:.4f}",
        ]
        
        if self.nan_count > 0:
            lines.append(f"⚠ NaN count: {self.nan_count}")
        if self.inf_count > 0:
            lines.append(f"⚠ Inf count: {self.inf_count}")
        if self.grad_norm is not None:
            lines.append(f"Gradient norm: {self.grad_norm:.4f}")
        
        return "\n".join(lines)


def inspect(tensor: Tensor, name: str = "tensor") -> TensorStats:
    """Inspect a tensor and return statistics."""
    with torch.no_grad():
        t = tensor.float()
        stats = TensorStats(
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            min=t.min().item(),
            max=t.max().item(),
            mean=t.mean().item(),
            std=t.std().item() if t.numel() > 1 else 0.0,
            nan_count=torch.isnan(t).sum().item(),
            inf_count=torch.isinf(t).sum().item(),
            zero_count=(t == 0).sum().item(),
            grad_norm=tensor.grad.norm().item() if tensor.grad is not None else None
        )
    
    print(f"\n=== {name} ===")
    print(stats)
    
    return stats


def check_health(
    tensors: Dict[str, Tensor],
    gradients: bool = True
) -> List[str]:
    """Check health of tensors and their gradients."""
    issues = []
    
    for name, tensor in tensors.items():
        # Check for NaN
        if torch.isnan(tensor).any():
            issues.append(f"{name}: contains NaN values")
        
        # Check for Inf
        if torch.isinf(tensor).any():
            issues.append(f"{name}: contains Inf values")
        
        # Check gradient health
        if gradients and tensor.grad is not None:
            if torch.isnan(tensor.grad).any():
                issues.append(f"{name}.grad: contains NaN values")
            if torch.isinf(tensor.grad).any():
                issues.append(f"{name}.grad: contains Inf values")
            
            grad_norm = tensor.grad.norm().item()
            if grad_norm < 1e-10:
                issues.append(f"{name}.grad: near-zero gradient")
            if grad_norm > 1e8:
                issues.append(f"{name}.grad: exploding gradient")
    
    return issues


# ============================================================
# Breakpoints and Watches
# ============================================================

class Watch:
    """Watch a tensor value during training."""
    
    def __init__(self, name: str, tensor: Tensor) -> None:
        self.name = name
        self._tensor_ref = weakref.ref(tensor)
        self._history: List[float] = []
        self._grad_history: List[float] = []
    
    def update(self) -> None:
        """Record current value."""
        tensor = self._tensor_ref()
        if tensor is not None:
            self._history.append(tensor.mean().item())
            if tensor.grad is not None:
                self._grad_history.append(tensor.grad.norm().item())
    
    def summary(self) -> str:
        """Get watch summary."""
        if not self._history:
            return f"{self.name}: no data"
        
        return (
            f"{self.name}: "
            f"current={self._history[-1]:.4f}, "
            f"min={min(self._history):.4f}, "
            f"max={max(self._history):.4f}"
        )


class Watchlist:
    """Collection of tensor watches."""
    
    def __init__(self) -> None:
        self._watches: Dict[str, Watch] = {}
    
    def add(self, name: str, tensor: Tensor) -> None:
        """Add a tensor to watch."""
        self._watches[name] = Watch(name, tensor)
    
    def update_all(self) -> None:
        """Update all watches."""
        for watch in self._watches.values():
            watch.update()
    
    def summary(self) -> str:
        """Get summary of all watches."""
        lines = ["Watchlist:"]
        for watch in self._watches.values():
            lines.append(f"  {watch.summary()}")
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all watches."""
        self._watches = {}


# Global watchlist
_watchlist = Watchlist()


def watch(name: str, tensor: Tensor) -> None:
    """Add a tensor to the global watchlist."""
    _watchlist.add(name, tensor)


def update_watches() -> None:
    """Update all global watches."""
    _watchlist.update_all()


def watch_summary() -> str:
    """Get summary of global watches."""
    return _watchlist.summary()


# ============================================================
# Assertion Helpers
# ============================================================

def assert_finite(tensor: Tensor, name: str = "tensor") -> None:
    """Assert tensor contains no NaN or Inf."""
    if torch.isnan(tensor).any():
        raise AssertionError(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        raise AssertionError(f"{name} contains Inf values")


def assert_shape(tensor: Tensor, expected: Tuple[int, ...], name: str = "tensor") -> None:
    """Assert tensor has expected shape."""
    if tuple(tensor.shape) != expected:
        raise AssertionError(
            f"{name} has shape {tuple(tensor.shape)}, expected {expected}"
        )


def assert_grad_exists(tensor: Tensor, name: str = "tensor") -> None:
    """Assert tensor has gradient."""
    if tensor.grad is None:
        raise AssertionError(f"{name} has no gradient")


def assert_no_grad_issues(tensor: Tensor, name: str = "tensor") -> None:
    """Assert gradient is healthy."""
    assert_grad_exists(tensor, name)
    assert_finite(tensor.grad, f"{name}.grad")
    
    grad_norm = tensor.grad.norm().item()
    if grad_norm < 1e-12:
        raise AssertionError(f"{name}.grad is essentially zero")
