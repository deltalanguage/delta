"""
Optimizer integration for Delta.

Wraps PyTorch optimizers with Delta-specific functionality:
- Constraint-aware optimization
- Learning rate scheduling
- Gradient clipping
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Iterator
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class OptimizerConfig:
    """Configuration for Delta optimizer."""
    optimizer_type: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0.9  # For SGD
    betas: tuple[float, float] = (0.9, 0.999)  # For Adam
    eps: float = 1e-8
    grad_clip_norm: Optional[float] = None
    grad_clip_value: Optional[float] = None


class OptimizerWrapper:
    """
    Wraps a PyTorch optimizer with Delta functionality.
    
    Features:
    - Gradient clipping
    - Learning rate scheduling
    - Training statistics
    - Constraint weighting
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        config: Optional[OptimizerConfig] = None,
        scheduler: Optional[_LRScheduler] = None
    ) -> None:
        self.optimizer = optimizer
        self.config = config or OptimizerConfig()
        self.scheduler = scheduler
        
        self.step_count = 0
        self.history: List[Dict[str, float]] = []
    
    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform an optimization step.
        
        Includes gradient clipping if configured.
        """
        # Gradient clipping
        if self.config.grad_clip_norm is not None:
            params = []
            for group in self.optimizer.param_groups:
                params.extend(group["params"])
            torch.nn.utils.clip_grad_norm_(params, self.config.grad_clip_norm)
        
        if self.config.grad_clip_value is not None:
            params = []
            for group in self.optimizer.param_groups:
                params.extend(group["params"])
            torch.nn.utils.clip_grad_value_(params, self.config.grad_clip_value)
        
        # Optimization step
        loss = self.optimizer.step(closure)
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.step_count += 1
        
        # Record history
        self.history.append({
            "step": self.step_count,
            "lr": self.get_lr(),
        })
        
        return loss
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
    
    def set_lr(self, lr: float) -> None:
        """Set learning rate for all param groups."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "step_count": self.step_count,
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load optimizer state."""
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler and state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        self.step_count = state.get("step_count", 0)


def create_optimizer(
    parameters: Iterator[nn.Parameter],
    config: Optional[OptimizerConfig] = None,
    scheduler_type: Optional[str] = None,
    scheduler_config: Optional[Dict[str, Any]] = None
) -> OptimizerWrapper:
    """
    Create an optimizer from configuration.
    
    Args:
        parameters: Model parameters to optimize
        config: Optimizer configuration
        scheduler_type: Type of LR scheduler (step, cosine, exponential)
        scheduler_config: Scheduler configuration
    
    Returns:
        OptimizerWrapper instance
    """
    config = config or OptimizerConfig()
    params_list = list(parameters)
    
    # Create base optimizer
    if config.optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            params_list,
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            params_list,
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            params_list,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    elif config.optimizer_type.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(
            params_list,
            lr=config.learning_rate,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")
    
    # Create scheduler if specified
    scheduler = None
    if scheduler_type:
        scheduler_config = scheduler_config or {}
        
        if scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get("step_size", 10),
                gamma=scheduler_config.get("gamma", 0.1)
            )
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get("T_max", 100),
                eta_min=scheduler_config.get("eta_min", 0)
            )
        elif scheduler_type == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=scheduler_config.get("gamma", 0.9)
            )
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get("mode", "min"),
                factor=scheduler_config.get("factor", 0.1),
                patience=scheduler_config.get("patience", 10)
            )
    
    return OptimizerWrapper(optimizer, config, scheduler)


class ConstraintAwareOptimizer(OptimizerWrapper):
    """
    Optimizer with constraint-aware optimization.
    
    Supports:
    - Adaptive constraint weighting
    - Augmented Lagrangian method
    - Penalty scheduling
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        config: Optional[OptimizerConfig] = None,
        constraint_weights: Optional[Dict[str, float]] = None,
        lagrange_multipliers: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        super().__init__(optimizer, config)
        self.constraint_weights = constraint_weights or {}
        self.lagrange_multipliers = lagrange_multipliers or {}
        self.constraint_history: List[Dict[str, float]] = []
    
    def update_constraint_weights(
        self,
        constraint_values: Dict[str, float],
        method: str = "adaptive"
    ) -> None:
        """
        Update constraint weights based on violation.
        
        Methods:
        - adaptive: Increase weight if constraint violated
        - fixed: Keep weights fixed
        """
        if method == "adaptive":
            for name, value in constraint_values.items():
                current_weight = self.constraint_weights.get(name, 1.0)
                
                # Increase weight if constraint is violated
                if value > 0.1:  # Violation threshold
                    self.constraint_weights[name] = min(current_weight * 1.5, 1000.0)
                elif value < 0.01:  # Well satisfied
                    self.constraint_weights[name] = max(current_weight * 0.9, 0.1)
    
    def update_lagrange_multipliers(
        self,
        constraint_values: Dict[str, torch.Tensor],
        step_size: float = 0.01
    ) -> None:
        """Update Lagrange multipliers for augmented Lagrangian method."""
        for name, value in constraint_values.items():
            if name not in self.lagrange_multipliers:
                self.lagrange_multipliers[name] = torch.zeros_like(value)
            
            # Gradient ascent on dual
            self.lagrange_multipliers[name] = self.lagrange_multipliers[name] + step_size * value
    
    def get_constraint_loss(
        self,
        constraint_values: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute total constraint loss with current weights and multipliers."""
        total_loss = torch.tensor(0.0)
        
        for name, value in constraint_values.items():
            weight = self.constraint_weights.get(name, 1.0)
            multiplier = self.lagrange_multipliers.get(name, 0.0)
            
            # Augmented Lagrangian: λ * g(x) + (ρ/2) * g(x)^2
            if isinstance(multiplier, torch.Tensor):
                total_loss = total_loss + (multiplier * value).sum()
            else:
                total_loss = total_loss + multiplier * value.sum()
            
            total_loss = total_loss + (weight / 2) * (value ** 2).sum()
        
        return total_loss
