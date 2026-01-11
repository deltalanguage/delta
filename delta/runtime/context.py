"""
Runtime context for Delta.

Manages the execution environment:
- Mode (train/infer/analyze)
- Device placement
- Random seeds
- Tracing and debugging
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, ContextManager
from contextlib import contextmanager
import torch
import torch.nn as nn


@dataclass
class DeltaContext:
    """
    Global context for Delta execution.
    
    Manages:
    - Execution mode
    - Device placement
    - Random state
    - Debug flags
    """
    mode: str = "train"
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float32
    seed: Optional[int] = None
    debug: bool = False
    trace_gradients: bool = False
    trace_activations: bool = False
    
    def __post_init__(self) -> None:
        if self.seed is not None:
            self.set_seed(self.seed)
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor to the context's device."""
        return tensor.to(device=self.device, dtype=self.dtype)
    
    @contextmanager
    def train_mode(self):
        """Context manager for training mode."""
        old_mode = self.mode
        self.mode = "train"
        try:
            with torch.enable_grad():
                yield
        finally:
            self.mode = old_mode
    
    @contextmanager
    def infer_mode(self):
        """Context manager for inference mode."""
        old_mode = self.mode
        self.mode = "infer"
        try:
            with torch.no_grad():
                yield
        finally:
            self.mode = old_mode
    
    @contextmanager
    def analyze_mode(self):
        """Context manager for analysis mode."""
        old_mode = self.mode
        old_trace = self.trace_gradients
        self.mode = "analyze"
        self.trace_gradients = True
        try:
            with torch.enable_grad():
                yield
        finally:
            self.mode = old_mode
            self.trace_gradients = old_trace


@dataclass
class TrainingContext:
    """
    Context for a training session.
    
    Tracks:
    - Training history
    - Best metrics
    - Checkpoints
    """
    epochs: int = 0
    steps: int = 0
    best_loss: float = float("inf")
    best_metric: float = float("-inf")
    history: List[Dict[str, float]] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    early_stop_patience: int = 10
    early_stop_counter: int = 0
    
    def update(
        self,
        loss: float,
        metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Update training context with new results.
        
        Returns True if training should continue, False if early stopping.
        """
        self.steps += 1
        
        record = {"step": self.steps, "loss": loss}
        if metrics:
            record.update(metrics)
        self.history.append(record)
        
        # Check for improvement
        improved = False
        if loss < self.best_loss:
            self.best_loss = loss
            improved = True
        
        if metrics and "accuracy" in metrics:
            if metrics["accuracy"] > self.best_metric:
                self.best_metric = metrics["accuracy"]
                improved = True
        
        # Early stopping
        if improved:
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
        
        return self.early_stop_counter < self.early_stop_patience
    
    def end_epoch(self) -> None:
        """Mark the end of an epoch."""
        self.epochs += 1
    
    def should_checkpoint(self, every_n_epochs: int = 1) -> bool:
        """Check if we should save a checkpoint."""
        return self.epochs % every_n_epochs == 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of training."""
        return {
            "epochs": self.epochs,
            "steps": self.steps,
            "best_loss": self.best_loss,
            "best_metric": self.best_metric,
            "final_loss": self.history[-1]["loss"] if self.history else None,
        }


class GradientTracer:
    """
    Traces gradient flow for debugging.
    
    Records:
    - Gradient magnitudes
    - Gradient statistics
    - Vanishing/exploding gradients
    """
    
    def __init__(self) -> None:
        self.gradient_history: List[Dict[str, float]] = []
        self.hooks: List[Any] = []
    
    def attach(self, module: nn.Module) -> None:
        """Attach gradient hooks to a module."""
        for name, param in module.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, n=name: self._record_gradient(n, grad)
                )
                self.hooks.append(hook)
    
    def detach(self) -> None:
        """Remove all gradient hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _record_gradient(self, name: str, grad: torch.Tensor) -> None:
        """Record gradient statistics."""
        if grad is None:
            return
        
        self.gradient_history.append({
            "name": name,
            "mean": grad.abs().mean().item(),
            "max": grad.abs().max().item(),
            "min": grad.abs().min().item(),
            "std": grad.std().item(),
            "norm": grad.norm().item(),
        })
    
    def get_vanishing_gradients(self, threshold: float = 1e-7) -> List[str]:
        """Find parameters with vanishing gradients."""
        if not self.gradient_history:
            return []
        
        vanishing = set()
        for record in self.gradient_history:
            if record["max"] < threshold:
                vanishing.add(record["name"])
        
        return list(vanishing)
    
    def get_exploding_gradients(self, threshold: float = 1e3) -> List[str]:
        """Find parameters with exploding gradients."""
        if not self.gradient_history:
            return []
        
        exploding = set()
        for record in self.gradient_history:
            if record["max"] > threshold:
                exploding.add(record["name"])
        
        return list(exploding)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of gradient flow."""
        if not self.gradient_history:
            return {"status": "no gradients recorded"}
        
        all_norms = [r["norm"] for r in self.gradient_history]
        
        return {
            "num_parameters": len(set(r["name"] for r in self.gradient_history)),
            "mean_gradient_norm": sum(all_norms) / len(all_norms),
            "max_gradient_norm": max(all_norms),
            "min_gradient_norm": min(all_norms),
            "vanishing": self.get_vanishing_gradients(),
            "exploding": self.get_exploding_gradients(),
        }


# Global context
_global_context: Optional[DeltaContext] = None


def get_context() -> DeltaContext:
    """Get the global Delta context."""
    global _global_context
    if _global_context is None:
        _global_context = DeltaContext()
    return _global_context


def set_context(ctx: DeltaContext) -> None:
    """Set the global Delta context."""
    global _global_context
    _global_context = ctx


@contextmanager
def delta_context(**kwargs):
    """Context manager for temporary Delta context."""
    old_ctx = get_context()
    new_ctx = DeltaContext(**kwargs)
    set_context(new_ctx)
    try:
        yield new_ctx
    finally:
        set_context(old_ctx)
