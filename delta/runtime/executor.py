"""
Graph executor for Delta.

Executes compiled FX graphs with proper parameter and gradient management.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, Callable
import torch
import torch.nn as nn
from torch.fx import GraphModule


@dataclass
class ExecutionContext:
    """Context for graph execution."""
    mode: str = "train"  # train, infer, analyze
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float32
    grad_enabled: bool = True
    trace: bool = False


class Executor:
    """
    Executes compiled Delta graphs.
    
    Manages:
    - Parameter tensors
    - Graph execution
    - Mode switching
    - Device placement
    """
    
    def __init__(
        self,
        graph_modules: Dict[str, GraphModule],
        parameters: Dict[str, nn.Parameter],
        context: Optional[ExecutionContext] = None
    ) -> None:
        self.graph_modules = graph_modules
        self.parameters = nn.ParameterDict(parameters)
        self.context = context or ExecutionContext()
        self._compiled_forward: Optional[Callable] = None
    
    def to(self, device: torch.device) -> Executor:
        """Move executor to a device."""
        self.context.device = device
        self.parameters = nn.ParameterDict({
            name: nn.Parameter(p.to(device))
            for name, p in self.parameters.items()
        })
        return self
    
    def train(self) -> Executor:
        """Set to training mode."""
        self.context.mode = "train"
        self.context.grad_enabled = True
        return self
    
    def eval(self) -> Executor:
        """Set to inference mode."""
        self.context.mode = "infer"
        self.context.grad_enabled = False
        return self
    
    def analyze(self) -> Executor:
        """Set to analyze mode."""
        self.context.mode = "analyze"
        self.context.grad_enabled = True
        self.context.trace = True
        return self
    
    def __call__(self, func_name: str, **inputs: torch.Tensor) -> torch.Tensor:
        """Execute a function by name."""
        return self.execute(func_name, **inputs)
    
    def execute(self, func_name: str, **inputs: torch.Tensor) -> torch.Tensor:
        """Execute a named function with the given inputs."""
        if func_name not in self.graph_modules:
            raise ValueError(f"Unknown function: {func_name}")
        
        graph_module = self.graph_modules[func_name]
        
        # Move inputs to device
        device_inputs = {
            k: v.to(self.context.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        
        # Inject parameters
        for name, param in self.parameters.items():
            if hasattr(graph_module, "params") and hasattr(graph_module.params, name):
                setattr(graph_module.params, name, param)
        
        # Execute with appropriate gradient context
        if self.context.grad_enabled:
            return graph_module(**device_inputs)
        else:
            with torch.no_grad():
                return graph_module(**device_inputs)
    
    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        """Execute the main forward pass (first function)."""
        if not self.graph_modules:
            raise ValueError("No functions to execute")
        
        first_func = next(iter(self.graph_modules.keys()))
        return self.execute(first_func, **inputs)
    
    def compute_loss(
        self,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        targets: torch.Tensor,
        **inputs: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss given a loss function and targets."""
        output = self.forward(**inputs)
        return loss_fn(output, targets)
    
    def step(
        self,
        optimizer: torch.optim.Optimizer,
        **inputs: torch.Tensor
    ) -> Dict[str, Any]:
        """Execute one training step."""
        if self.context.mode != "train":
            raise RuntimeError("Cannot step in non-training mode")
        
        optimizer.zero_grad()
        
        # Execute forward pass
        output = self.forward(**inputs)
        
        # Assuming output is the loss
        if output.requires_grad:
            output.backward()
            optimizer.step()
        
        return {
            "loss": output.detach(),
            "output": output.detach()
        }
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all parameter values."""
        return {name: p.data for name, p in self.parameters.items()}
    
    def set_parameters(self, values: Dict[str, torch.Tensor]) -> None:
        """Set parameter values."""
        for name, value in values.items():
            if name in self.parameters:
                self.parameters[name].data = value.to(self.context.device)
    
    def compile(self, backend: str = "inductor") -> None:
        """Compile the executor with torch.compile."""
        try:
            for name, gm in self.graph_modules.items():
                self.graph_modules[name] = torch.compile(gm, backend=backend)
        except Exception as e:
            # Compilation failed, continue without
            print(f"Warning: torch.compile failed: {e}")


class BatchExecutor(Executor):
    """
    Executor optimized for batch processing.
    
    Supports:
    - Data loader integration
    - Automatic batching
    - Progress tracking
    """
    
    def __init__(
        self,
        graph_modules: Dict[str, GraphModule],
        parameters: Dict[str, nn.Parameter],
        context: Optional[ExecutionContext] = None
    ) -> None:
        super().__init__(graph_modules, parameters, context)
        self.history: list[Dict[str, Any]] = []
    
    def train_epoch(
        self,
        data_loader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    inputs, targets = batch
                    if isinstance(inputs, dict):
                        batch_inputs = inputs
                    else:
                        batch_inputs = {"x": inputs}
                    batch_inputs["targets"] = targets
                else:
                    batch_inputs = {"x": batch[0]}
            elif isinstance(batch, dict):
                batch_inputs = batch
            else:
                batch_inputs = {"x": batch}
            
            # Move to device
            batch_inputs = {
                k: v.to(self.context.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_inputs.items()
            }
            
            result = self.step(optimizer, **batch_inputs)
            total_loss += result["loss"].item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.history.append({"epoch_loss": avg_loss})
        
        return {"loss": avg_loss, "num_batches": num_batches}
    
    def evaluate(self, data_loader, metric_fn: Optional[Callable] = None) -> Dict[str, float]:
        """Evaluate on a dataset."""
        self.eval()
        
        total_metric = 0.0
        num_batches = 0
        
        for batch in data_loader:
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch
                if isinstance(inputs, dict):
                    batch_inputs = inputs
                else:
                    batch_inputs = {"x": inputs}
            elif isinstance(batch, dict):
                batch_inputs = batch
            else:
                batch_inputs = {"x": batch}
                targets = None
            
            # Move to device
            batch_inputs = {
                k: v.to(self.context.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_inputs.items()
            }
            
            output = self.forward(**batch_inputs)
            
            if metric_fn and targets is not None:
                metric = metric_fn(output, targets.to(self.context.device))
                total_metric += metric.item()
            
            num_batches += 1
        
        return {"metric": total_metric / max(num_batches, 1)}
