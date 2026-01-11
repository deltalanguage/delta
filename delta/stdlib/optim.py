"""
std.optim - Optimizers for Delta.

Provides optimizer definitions and learning rate schedulers
for use in Delta learn blocks.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Iterator, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# ============================================================
# Optimizer Factories
# ============================================================

@dataclass
class OptimizerSpec:
    """Specification for creating an optimizer."""
    type: str
    lr: float = 0.001
    weight_decay: float = 0.0
    kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


def Adam(
    params=None,
    lr: float = 0.001,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False
) -> "torch.optim.Adam | OptimizerSpec":
    """Create Adam optimizer or specification.
    
    If params is provided, returns a torch.optim.Adam instance.
    Otherwise, returns an OptimizerSpec for use in Delta learn blocks.
    """
    if params is not None:
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, 
                                 weight_decay=weight_decay, amsgrad=amsgrad)
    return OptimizerSpec(
        type="adam",
        lr=lr,
        weight_decay=weight_decay,
        kwargs={"betas": betas, "eps": eps, "amsgrad": amsgrad}
    )


def AdamW(
    lr: float = 0.001,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01
) -> OptimizerSpec:
    """Create AdamW optimizer specification."""
    return OptimizerSpec(
        type="adamw",
        lr=lr,
        weight_decay=weight_decay,
        kwargs={"betas": betas, "eps": eps}
    )


def SGD(
    params=None,
    lr: float = 0.01,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    dampening: float = 0.0,
    nesterov: bool = False
) -> "torch.optim.SGD | OptimizerSpec":
    """Create SGD optimizer or specification.
    
    If params is provided, returns a torch.optim.SGD instance.
    Otherwise, returns an OptimizerSpec for use in Delta learn blocks.
    """
    if params is not None:
        return torch.optim.SGD(params, lr=lr, momentum=momentum, 
                                weight_decay=weight_decay, dampening=dampening, 
                                nesterov=nesterov)
    return OptimizerSpec(
        type="sgd",
        lr=lr,
        weight_decay=weight_decay,
        kwargs={"momentum": momentum, "dampening": dampening, "nesterov": nesterov}
    )


def RMSprop(
    lr: float = 0.01,
    alpha: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    momentum: float = 0.0
) -> OptimizerSpec:
    """Create RMSprop optimizer specification."""
    return OptimizerSpec(
        type="rmsprop",
        lr=lr,
        weight_decay=weight_decay,
        kwargs={"alpha": alpha, "eps": eps, "momentum": momentum}
    )


def Adagrad(
    lr: float = 0.01,
    lr_decay: float = 0.0,
    weight_decay: float = 0.0,
    eps: float = 1e-10
) -> OptimizerSpec:
    """Create Adagrad optimizer specification."""
    return OptimizerSpec(
        type="adagrad",
        lr=lr,
        weight_decay=weight_decay,
        kwargs={"lr_decay": lr_decay, "eps": eps}
    )


def LBFGS(
    lr: float = 1.0,
    max_iter: int = 20,
    history_size: int = 100,
    line_search_fn: Optional[str] = "strong_wolfe"
) -> OptimizerSpec:
    """Create L-BFGS optimizer specification."""
    return OptimizerSpec(
        type="lbfgs",
        lr=lr,
        kwargs={"max_iter": max_iter, "history_size": history_size,
                "line_search_fn": line_search_fn}
    )


# ============================================================
# Optimizer Creation
# ============================================================

def create_optimizer(params: Iterator[nn.Parameter], spec: OptimizerSpec) -> Optimizer:
    """Create a PyTorch optimizer from a specification."""
    params_list = list(params)
    
    if spec.type == "adam":
        return torch.optim.Adam(
            params_list,
            lr=spec.lr,
            weight_decay=spec.weight_decay,
            **spec.kwargs
        )
    elif spec.type == "adamw":
        return torch.optim.AdamW(
            params_list,
            lr=spec.lr,
            weight_decay=spec.weight_decay,
            **spec.kwargs
        )
    elif spec.type == "sgd":
        return torch.optim.SGD(
            params_list,
            lr=spec.lr,
            weight_decay=spec.weight_decay,
            **spec.kwargs
        )
    elif spec.type == "rmsprop":
        return torch.optim.RMSprop(
            params_list,
            lr=spec.lr,
            weight_decay=spec.weight_decay,
            **spec.kwargs
        )
    elif spec.type == "adagrad":
        return torch.optim.Adagrad(
            params_list,
            lr=spec.lr,
            weight_decay=spec.weight_decay,
            **spec.kwargs
        )
    elif spec.type == "lbfgs":
        return torch.optim.LBFGS(
            params_list,
            lr=spec.lr,
            **spec.kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer type: {spec.type}")


# ============================================================
# Learning Rate Schedulers
# ============================================================

@dataclass
class SchedulerSpec:
    """Specification for creating a scheduler."""
    type: str
    kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


def StepLR(optimizer=None, step_size: int = 1, gamma: float = 0.1) -> "torch.optim.lr_scheduler.StepLR | SchedulerSpec":
    """Step learning rate decay.
    
    If optimizer is provided, returns a torch.optim.lr_scheduler.StepLR instance.
    Otherwise, returns a SchedulerSpec for use in Delta learn blocks.
    """
    if optimizer is not None:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return SchedulerSpec(type="step", kwargs={"step_size": step_size, "gamma": gamma})


def MultiStepLR(milestones: list[int], gamma: float = 0.1) -> SchedulerSpec:
    """Multi-step learning rate decay."""
    return SchedulerSpec(type="multistep", kwargs={"milestones": milestones, "gamma": gamma})


def ExponentialLR(gamma: float) -> SchedulerSpec:
    """Exponential learning rate decay."""
    return SchedulerSpec(type="exponential", kwargs={"gamma": gamma})


def CosineAnnealingLR(T_max: int, eta_min: float = 0.0) -> SchedulerSpec:
    """Cosine annealing learning rate."""
    return SchedulerSpec(type="cosine", kwargs={"T_max": T_max, "eta_min": eta_min})


def ReduceLROnPlateau(
    mode: str = "min",
    factor: float = 0.1,
    patience: int = 10,
    threshold: float = 1e-4,
    min_lr: float = 0.0
) -> SchedulerSpec:
    """Reduce LR on plateau."""
    return SchedulerSpec(
        type="plateau",
        kwargs={"mode": mode, "factor": factor, "patience": patience,
                "threshold": threshold, "min_lr": min_lr}
    )


def OneCycleLR(
    max_lr: float,
    total_steps: int,
    pct_start: float = 0.3,
    anneal_strategy: str = "cos"
) -> SchedulerSpec:
    """One cycle learning rate policy."""
    return SchedulerSpec(
        type="onecycle",
        kwargs={"max_lr": max_lr, "total_steps": total_steps,
                "pct_start": pct_start, "anneal_strategy": anneal_strategy}
    )


def WarmupLR(warmup_steps: int, warmup_factor: float = 0.1) -> SchedulerSpec:
    """Linear warmup scheduler."""
    return SchedulerSpec(
        type="warmup",
        kwargs={"warmup_steps": warmup_steps, "warmup_factor": warmup_factor}
    )


def create_scheduler(optimizer: Optimizer, spec: SchedulerSpec) -> _LRScheduler:
    """Create a PyTorch scheduler from a specification."""
    if spec.type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **spec.kwargs)
    elif spec.type == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **spec.kwargs)
    elif spec.type == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **spec.kwargs)
    elif spec.type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **spec.kwargs)
    elif spec.type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **spec.kwargs)
    elif spec.type == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, **spec.kwargs)
    elif spec.type == "warmup":
        # Custom warmup scheduler
        warmup_steps = spec.kwargs["warmup_steps"]
        warmup_factor = spec.kwargs.get("warmup_factor", 0.1)
        
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return warmup_factor + (1 - warmup_factor) * step / warmup_steps
            return 1.0
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler type: {spec.type}")


# ============================================================
# Gradient Utilities
# ============================================================

def clip_grad_norm(params: Iterator[nn.Parameter], max_norm: float, norm_type: float = 2.0) -> float:
    """Clip gradient norm."""
    params_list = list(params)
    return torch.nn.utils.clip_grad_norm_(params_list, max_norm, norm_type)


def clip_grad_value(params: Iterator[nn.Parameter], clip_value: float) -> None:
    """Clip gradient values."""
    params_list = list(params)
    torch.nn.utils.clip_grad_value_(params_list, clip_value)


def get_grad_norm(params: Iterator[nn.Parameter], norm_type: float = 2.0) -> float:
    """Get total gradient norm."""
    params_list = list(p for p in params if p.grad is not None)
    if not params_list:
        return 0.0
    
    total_norm = 0.0
    for p in params_list:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    
    return total_norm ** (1.0 / norm_type)
