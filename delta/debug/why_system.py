"""
The "Why" System for Delta.

Provides deep diagnostics for understanding optimization behavior,
constraint satisfaction, and gradient flow.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import torch
from torch import Tensor
import math


# ============================================================
# Diagnostic Data Structures
# ============================================================

@dataclass
class ConstraintDiagnosis:
    """Diagnosis for a single constraint."""
    name: str
    violation_value: float
    weight: float
    weighted_contribution: float
    gradient_norm: float
    is_satisfied: bool
    slack: float
    trend: str  # "improving", "stagnant", "worsening"
    recommendations: List[str] = field(default_factory=list)
    
    def severity(self) -> str:
        """Get severity level."""
        if self.is_satisfied:
            return "ok"
        if self.weighted_contribution < 0.1:
            return "low"
        if self.weighted_contribution < 1.0:
            return "medium"
        return "high"
    
    def __str__(self) -> str:
        status = "✓" if self.is_satisfied else "✗"
        return (
            f"{status} {self.name}: "
            f"violation={self.violation_value:.4f}, "
            f"weight={self.weight:.2f}, "
            f"trend={self.trend}"
        )


@dataclass
class GradientDiagnosis:
    """Diagnosis for gradient flow."""
    parameter_name: str
    gradient_norm: float
    gradient_mean: float
    gradient_std: float
    is_vanishing: bool
    is_exploding: bool
    has_nan: bool
    has_inf: bool
    sparsity: float  # fraction of zeros
    recommendations: List[str] = field(default_factory=list)
    
    def health(self) -> str:
        """Get gradient health status."""
        if self.has_nan or self.has_inf:
            return "critical"
        if self.is_exploding:
            return "exploding"
        if self.is_vanishing:
            return "vanishing"
        return "healthy"
    
    def __str__(self) -> str:
        health = self.health()
        icon = {"critical": "💀", "exploding": "🔥", "vanishing": "❄️", "healthy": "✓"}
        return (
            f"{icon.get(health, '?')} {self.parameter_name}: "
            f"norm={self.gradient_norm:.4e}, "
            f"health={health}"
        )


@dataclass
class OptimizationDiagnosis:
    """Complete diagnosis of optimization state."""
    step: int
    loss: float
    constraint_diagnoses: List[ConstraintDiagnosis]
    gradient_diagnoses: List[GradientDiagnosis]
    loss_trend: str
    convergence_estimate: Optional[int]  # estimated steps to converge
    bottlenecks: List[str]
    recommendations: List[str]
    
    def __str__(self) -> str:
        lines = ["=" * 60]
        lines.append(f"OPTIMIZATION DIAGNOSIS (Step {self.step})")
        lines.append("=" * 60)
        
        lines.append(f"\nLoss: {self.loss:.6f} ({self.loss_trend})")
        if self.convergence_estimate:
            lines.append(f"Estimated convergence: ~{self.convergence_estimate} steps")
        
        if self.constraint_diagnoses:
            lines.append("\nCONSTRAINTS:")
            lines.append("-" * 40)
            for cd in sorted(self.constraint_diagnoses, 
                           key=lambda x: -x.weighted_contribution):
                lines.append(f"  {cd}")
        
        if self.gradient_diagnoses:
            lines.append("\nGRADIENTS:")
            lines.append("-" * 40)
            unhealthy = [g for g in self.gradient_diagnoses if g.health() != "healthy"]
            if unhealthy:
                for gd in unhealthy:
                    lines.append(f"  {gd}")
            else:
                lines.append("  All gradients healthy")
        
        if self.bottlenecks:
            lines.append("\nBOTTLENECKS:")
            lines.append("-" * 40)
            for bottleneck in self.bottlenecks:
                lines.append(f"  ⚠ {bottleneck}")
        
        if self.recommendations:
            lines.append("\nRECOMMENDATIONS:")
            lines.append("-" * 40)
            for rec in self.recommendations:
                lines.append(f"  → {rec}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ============================================================
# Why Engine
# ============================================================

class WhyEngine:
    """
    The Why Engine provides deep analysis of optimization state.
    
    It tracks history to detect trends, identifies bottlenecks,
    and generates actionable recommendations.
    """
    
    def __init__(self, history_size: int = 100) -> None:
        self._history_size = history_size
        self._loss_history: List[float] = []
        self._constraint_history: Dict[str, List[float]] = {}
        self._gradient_history: Dict[str, List[float]] = {}
        self._step = 0
    
    def record_step(
        self,
        loss: float,
        constraints: Dict[str, Tuple[float, float]],  # name -> (violation, weight)
        gradients: Dict[str, Tensor]
    ) -> None:
        """Record a training step."""
        self._step += 1
        
        # Loss history
        self._loss_history.append(loss)
        if len(self._loss_history) > self._history_size:
            self._loss_history.pop(0)
        
        # Constraint history
        for name, (violation, _) in constraints.items():
            if name not in self._constraint_history:
                self._constraint_history[name] = []
            self._constraint_history[name].append(violation)
            if len(self._constraint_history[name]) > self._history_size:
                self._constraint_history[name].pop(0)
        
        # Gradient history
        for name, grad in gradients.items():
            if grad is not None:
                norm = grad.norm().item()
                if name not in self._gradient_history:
                    self._gradient_history[name] = []
                self._gradient_history[name].append(norm)
                if len(self._gradient_history[name]) > self._history_size:
                    self._gradient_history[name].pop(0)
    
    def diagnose(
        self,
        loss: float,
        constraints: Dict[str, Tuple[float, float, Optional[Tensor]]],
        gradients: Dict[str, Tensor],
        satisfaction_threshold: float = 1e-4
    ) -> OptimizationDiagnosis:
        """Generate complete diagnosis of current optimization state."""
        # Diagnose constraints
        constraint_diagnoses = []
        for name, (violation, weight, tensor) in constraints.items():
            grad_norm = 0.0
            if tensor is not None and tensor.grad is not None:
                grad_norm = tensor.grad.norm().item()
            
            trend = self._compute_trend(
                self._constraint_history.get(name, [])
            )
            
            is_satisfied = violation < satisfaction_threshold
            
            diagnosis = ConstraintDiagnosis(
                name=name,
                violation_value=violation,
                weight=weight,
                weighted_contribution=violation * weight,
                gradient_norm=grad_norm,
                is_satisfied=is_satisfied,
                slack=max(0, satisfaction_threshold - violation),
                trend=trend,
            )
            
            # Add recommendations
            if not is_satisfied:
                if trend == "worsening":
                    diagnosis.recommendations.append(
                        f"Constraint '{name}' is getting worse. "
                        f"Consider increasing weight or relaxing other constraints."
                    )
                if grad_norm < 1e-8:
                    diagnosis.recommendations.append(
                        f"Constraint '{name}' has near-zero gradient. "
                        f"Check if it's being optimized."
                    )
            
            constraint_diagnoses.append(diagnosis)
        
        # Diagnose gradients
        gradient_diagnoses = []
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            diagnosis = self._diagnose_gradient(name, grad)
            gradient_diagnoses.append(diagnosis)
        
        # Compute loss trend
        loss_trend = self._compute_trend(self._loss_history)
        
        # Estimate convergence
        convergence_estimate = self._estimate_convergence()
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(
            constraint_diagnoses, gradient_diagnoses
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            loss_trend, constraint_diagnoses, gradient_diagnoses, bottlenecks
        )
        
        return OptimizationDiagnosis(
            step=self._step,
            loss=loss,
            constraint_diagnoses=constraint_diagnoses,
            gradient_diagnoses=gradient_diagnoses,
            loss_trend=loss_trend,
            convergence_estimate=convergence_estimate,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )
    
    def _diagnose_gradient(self, name: str, grad: Tensor) -> GradientDiagnosis:
        """Diagnose a single gradient."""
        with torch.no_grad():
            grad_flat = grad.flatten().float()
            
            norm = grad.norm().item()
            mean = grad_flat.mean().item()
            std = grad_flat.std().item() if grad_flat.numel() > 1 else 0.0
            
            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()
            
            sparsity = (grad_flat == 0).float().mean().item()
            
            is_vanishing = norm < 1e-8
            is_exploding = norm > 1e6
        
        diagnosis = GradientDiagnosis(
            parameter_name=name,
            gradient_norm=norm,
            gradient_mean=mean,
            gradient_std=std,
            is_vanishing=is_vanishing,
            is_exploding=is_exploding,
            has_nan=has_nan,
            has_inf=has_inf,
            sparsity=sparsity,
        )
        
        # Add recommendations
        if has_nan:
            diagnosis.recommendations.append(
                f"NaN detected in {name}. Check for numerical instability."
            )
        if has_inf:
            diagnosis.recommendations.append(
                f"Inf detected in {name}. Check for overflow."
            )
        if is_vanishing:
            diagnosis.recommendations.append(
                f"Vanishing gradient in {name}. Consider gradient clipping or "
                f"different initialization."
            )
        if is_exploding:
            diagnosis.recommendations.append(
                f"Exploding gradient in {name}. Use gradient clipping."
            )
        if sparsity > 0.9:
            diagnosis.recommendations.append(
                f"Highly sparse gradient in {name} ({sparsity:.0%} zeros). "
                f"This may indicate dead units."
            )
        
        return diagnosis
    
    def _compute_trend(self, history: List[float]) -> str:
        """Compute trend from history."""
        if len(history) < 5:
            return "insufficient_data"
        
        recent = history[-10:]
        older = history[-20:-10] if len(history) >= 20 else history[:len(history)//2]
        
        if not older:
            return "insufficient_data"
        
        recent_mean = sum(recent) / len(recent)
        older_mean = sum(older) / len(older)
        
        if recent_mean < older_mean * 0.95:
            return "improving"
        if recent_mean > older_mean * 1.05:
            return "worsening"
        return "stagnant"
    
    def _estimate_convergence(self) -> Optional[int]:
        """Estimate steps to convergence."""
        if len(self._loss_history) < 20:
            return None
        
        recent = self._loss_history[-10:]
        older = self._loss_history[-20:-10]
        
        recent_mean = sum(recent) / len(recent)
        older_mean = sum(older) / len(older)
        
        if recent_mean >= older_mean:
            return None  # Not converging
        
        rate = (older_mean - recent_mean) / 10  # improvement per step
        if rate <= 0:
            return None
        
        # Rough estimate assuming linear convergence
        remaining = recent_mean / rate
        return int(remaining)
    
    def _identify_bottlenecks(
        self,
        constraints: List[ConstraintDiagnosis],
        gradients: List[GradientDiagnosis]
    ) -> List[str]:
        """Identify optimization bottlenecks."""
        bottlenecks = []
        
        # Check for dominating constraints
        if constraints:
            total = sum(c.weighted_contribution for c in constraints)
            if total > 0:
                for c in constraints:
                    if c.weighted_contribution / total > 0.8:
                        bottlenecks.append(
                            f"Constraint '{c.name}' dominates loss "
                            f"({c.weighted_contribution/total:.0%})"
                        )
        
        # Check for gradient issues
        unhealthy = [g for g in gradients if g.health() != "healthy"]
        if len(unhealthy) > len(gradients) / 2:
            bottlenecks.append(
                f"Many unhealthy gradients ({len(unhealthy)}/{len(gradients)})"
            )
        
        # Check for stagnating constraints
        stagnating = [c for c in constraints 
                     if not c.is_satisfied and c.trend == "stagnant"]
        if stagnating:
            names = ", ".join(c.name for c in stagnating[:3])
            bottlenecks.append(f"Stagnating constraints: {names}")
        
        return bottlenecks
    
    def _generate_recommendations(
        self,
        loss_trend: str,
        constraints: List[ConstraintDiagnosis],
        gradients: List[GradientDiagnosis],
        bottlenecks: List[str]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if loss_trend == "stagnant":
            recommendations.append(
                "Loss is stagnant. Consider:\n"
                "  - Increasing learning rate\n"
                "  - Adjusting constraint weights\n"
                "  - Adding momentum"
            )
        
        if loss_trend == "worsening":
            recommendations.append(
                "Loss is increasing! Consider:\n"
                "  - Reducing learning rate\n"
                "  - Checking for numerical instability\n"
                "  - Verifying constraint formulation"
            )
        
        # Collect all sub-recommendations
        for c in constraints:
            recommendations.extend(c.recommendations)
        for g in gradients:
            recommendations.extend(g.recommendations)
        
        # Deduplicate
        seen = set()
        unique = []
        for r in recommendations:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        
        return unique[:10]  # Top 10
    
    def clear_history(self) -> None:
        """Clear recorded history."""
        self._loss_history = []
        self._constraint_history = {}
        self._gradient_history = {}
        self._step = 0


# ============================================================
# Convenience Functions
# ============================================================

# Global engine
_engine = WhyEngine()


def diagnose_optimization(
    loss: float,
    constraints: Dict[str, Tuple[float, float, Optional[Tensor]]],
    gradients: Dict[str, Tensor],
    record: bool = True
) -> OptimizationDiagnosis:
    """
    Diagnose current optimization state.
    
    Args:
        loss: Current loss value
        constraints: Dict mapping name -> (violation, weight, tensor)
        gradients: Dict mapping parameter name -> gradient tensor
        record: Whether to record this step in history
    
    Returns:
        Complete diagnosis with recommendations
    """
    if record:
        simple_constraints = {
            name: (v, w) for name, (v, w, _) in constraints.items()
        }
        _engine.record_step(loss, simple_constraints, gradients)
    
    return _engine.diagnose(loss, constraints, gradients)


def why(
    loss: float,
    constraints: Dict[str, Tuple[float, float, Optional[Tensor]]],
    gradients: Dict[str, Tensor]
) -> str:
    """
    Get a quick explanation of current optimization state.
    
    This is the main entry point for the "why" system.
    """
    diagnosis = diagnose_optimization(loss, constraints, gradients)
    return str(diagnosis)


def clear_why_history() -> None:
    """Clear the why engine's history."""
    _engine.clear_history()
