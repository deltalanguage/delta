"""
High-level Delta execution API.

Usage:
    import delta
    
    # One-liner training
    model = delta.run("model.delta", train_loader, epochs=100)
    
    # Or step by step
    model = delta.compile("model.delta")
    model.fit(train_loader, epochs=100)
    output = model.generate("First Citizen:", max_tokens=100)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Callable, List
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from delta.compiler import Compiler, CompileOptions, CompileResult
from delta.source import SourceFile
from delta.types.types import TensorType, ConcreteDim
# Note: We import frontend AST types here for initializer evaluation.
# Ideally this metadata would be extracted during compilation, but
# for now we use the AST directly since CompileResult exposes it.
from delta.frontend.ast import ParamDecl, Call, Identifier, Literal, BinaryOp


@dataclass 
class DeltaModel:
    """
    A compiled Delta model ready for training/inference.
    
    Provides a simple interface:
        model = delta.compile("model.delta")
        model.fit(train_data, epochs=10, lr=3e-3)
        text = model.generate("Hello", max_tokens=100)
    """
    _module: nn.Module
    _compile_result: CompileResult
    _device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    _vocab: Optional[Dict[str, int]] = None
    _vocab_inv: Optional[Dict[int, str]] = None
    
    def __init__(self, compile_result: CompileResult):
        self._compile_result = compile_result
        # Prefer 'forward' module if available (standard for training), otherwise take the first one
        if compile_result.graph_modules and "forward" in compile_result.graph_modules:
            self._module = compile_result.graph_modules["forward"]
        else:
            self._module = list(compile_result.graph_modules.values())[0]
        self._device = torch.device("cpu")
        self._init_params_from_sir()
    
    def _init_params_from_sir(self):
        """Initialize parameters from SIR param declarations, respecting user's initializers."""
        if not self._compile_result.sir:
            return
        
        # Extract initializers from CompileResult metadata if available,
        # otherwise fall back to extracting from AST directly
        if self._compile_result.param_initializers is not None:
            param_initializers = self._compile_result.param_initializers
        else:
            # Fallback: extract from AST if metadata not available
            param_initializers = {}
            if self._compile_result.ast:
                for item in self._compile_result.ast.items:
                    if isinstance(item, ParamDecl) and item.initializer:
                        param_initializers[item.name] = item.initializer
        
        sir = self._compile_result.sir
        for name, param_node in sir.params.items():
            if hasattr(self._module, 'params') and hasattr(self._module.params, name):
                # Parse shape from the param name or use defaults
                shape = self._parse_param_shape(name, param_node)
                if shape:
                    # Use the user's initializer if available
                    if name in param_initializers:
                        data = self._eval_initializer(param_initializers[name], shape)
                    else:
                        # Fallback: use name-based heuristics
                        data = self._init_from_name(name, shape)
                    setattr(self._module.params, name, nn.Parameter(data))
    
    def _eval_initializer(self, initializer_expr: Any, shape: tuple) -> torch.Tensor:
        """Evaluate an initializer expression (e.g., randn(10, 5) or zeros(5))."""
        # Handle Call expressions like randn(10, 5) or zeros(5)
        if isinstance(initializer_expr, Call) and isinstance(initializer_expr.func, Identifier):
            func_name = initializer_expr.func.name
            
            # Extract shape arguments from call args
            args = initializer_expr.args
            if func_name == "randn":
                # randn(10, 5) - use provided shape args, or fallback to shape
                if args and all(isinstance(a, Literal) and a.kind == 'int' for a in args):
                    init_shape = tuple(a.value for a in args)
                    return torch.randn(*init_shape)
                else:
                    return torch.randn(*shape)
            elif func_name == "zeros":
                # zeros(5) - use provided shape args, or fallback to shape
                if args and all(isinstance(a, Literal) and a.kind == 'int' for a in args):
                    init_shape = tuple(a.value for a in args)
                    return torch.zeros(*init_shape)
                else:
                    return torch.zeros(*shape)
            elif func_name == "ones":
                # ones(5) - use provided shape args, or fallback to shape
                if args and all(isinstance(a, Literal) and a.kind == 'int' for a in args):
                    init_shape = tuple(a.value for a in args)
                    return torch.ones(*init_shape)
                else:
                    return torch.ones(*shape)
            elif func_name == "rand":
                # rand(5) - use provided shape args, or fallback to shape
                if args and all(isinstance(a, Literal) and a.kind == 'int' for a in args):
                    init_shape = tuple(a.value for a in args)
                    return torch.rand(*init_shape)
                else:
                    return torch.rand(*shape)
            elif func_name == "full":
                # full(5, 3, 2.5) -> size=(5, 3), fill_value=2.5
                if len(args) >= 2:
                    shape_args = args[:-1]
                    fill_value = args[-1].value if isinstance(args[-1], Literal) else 0.0
                    if all(isinstance(a, Literal) and a.kind == 'int' for a in shape_args):
                        init_shape = tuple(a.value for a in shape_args)
                        return torch.full(init_shape, fill_value)
                return torch.full(shape, args[0].value if args and isinstance(args[0], Literal) else 0.0)
        
        # Fallback: use shape with small random init
        return torch.randn(*shape) * 0.01
    
    def _init_from_name(self, name: str, shape: tuple) -> torch.Tensor:
        """Fallback initialization based on parameter name pattern."""
        name_lower = name.lower()
        # Weight matrices: use small random initialization
        if any(x in name_lower for x in ['w', 'weight', 'emb']) and not name_lower.startswith('b'):
            return torch.randn(*shape) * 0.02
        # Bias terms: use zeros
        elif name_lower.startswith('b') or 'bias' in name_lower:
            return torch.zeros(*shape)
        # Gamma/scale parameters: use ones
        elif any(x in name_lower for x in ['_g', 'gamma', 'ln', 'scale']):
            return torch.ones(*shape)
        # Default: use small random initialization (prevents dead neurons)
        else:
            return torch.randn(*shape) * 0.01
    
    def _extract_shape_from_initializer(self, initializer_expr: Any) -> Optional[tuple]:
        """Extract shape from an initializer expression like randn(3, 2)."""
        from delta.frontend.ast import Call, Identifier, Literal, Tensor, BinaryOp
        
        # Handle scaling: randn(...) * 0.01
        if isinstance(initializer_expr, BinaryOp):
            # Recursively check operands
            shape = self._extract_shape_from_initializer(initializer_expr.left)
            if shape: return shape
            return self._extract_shape_from_initializer(initializer_expr.right)

        if isinstance(initializer_expr, Call) and isinstance(initializer_expr.func, Identifier):
            func_name = initializer_expr.func.name
            args = initializer_expr.args
            
            # Shape-defining initializers: randn, zeros, ones, rand, full
            if func_name in ('randn', 'zeros', 'ones', 'rand', 'full'):
                if not args:
                    return None
                
                # Case 1: Variadic args randn(10, 5)
                if all(isinstance(a, Literal) and a.kind == 'int' for a in args):
                    return tuple(a.value for a in args)
                
                # Special handling for full(5, 3, 2.5)
                if func_name == 'full' and len(args) >= 2:
                    shape_args = args[:-1]
                    if all(isinstance(a, Literal) and a.kind == 'int' for a in shape_args):
                        return tuple(a.value for a in shape_args)
                
                # Case 2: List arg randn([10, 5])
                if len(args) == 1 and isinstance(args[0], Tensor):
                    elements = args[0].elements
                    if all(isinstance(e, Literal) and e.kind == 'int' for e in elements):
                        return tuple(e.value for e in elements)
            
            # Special case: eye(N) -> (N, N)
            if func_name == 'eye' and len(args) == 1:
                if isinstance(args[0], Literal) and args[0].kind == 'int':
                    n = args[0].value
                    return (n, n)
                    
        return None
    
    def _parse_param_shape(self, name: str, node: Any) -> Optional[tuple]:
        """Parse parameter shape from SIR node's type or infer from name."""
        # First, try to extract shape from the parameter's type
        if hasattr(node, 'props') and hasattr(node.props, 'dtype'):
            dtype = node.props.dtype
            if isinstance(dtype, TensorType) and dtype.shape:
                # Extract concrete dimensions
                shape = tuple(
                    dim.value for dim in dtype.shape
                    if isinstance(dim, ConcreteDim)
                )
                if shape and all(isinstance(d, int) for d in shape):
                    return shape
        
        # Second, try to extract shape from initializer in param_initializers
        if self._compile_result.param_initializers:
            initializer = self._compile_result.param_initializers.get(name)
            if initializer:
                shape = self._extract_shape_from_initializer(initializer)
                if shape:
                    return shape
        
        # Third, check AST directly if param_initializers not available
        if self._compile_result.ast:
            for item in self._compile_result.ast.items:
                if isinstance(item, ParamDecl) and item.name == name and item.initializer:
                    shape = self._extract_shape_from_initializer(item.initializer)
                    if shape:
                        return shape
        
        # Fallback: use hardcoded shapes for common parameter names
        shapes = {
            # Embeddings
            "tok_emb": (46, 64), "pos_emb": (32, 64),
            # Attention
            "W_q": (64, 64), "W_k": (64, 64), "W_v": (64, 64), "W_o": (64, 64),
            # Layer norm
            "ln1_g": (64,), "ln1_b": (64,), "ln2_g": (64,), "ln2_b": (64,),
            # FFN
            "ff_w1": (64, 256), "ff_b1": (256,),
            "ff_w2": (256, 64), "ff_b2": (64,),
            # MNIST
            "W1": (784, 256), "b1": (256,),
            "W2": (256, 128), "b2": (128,),
            "W3": (128, 10), "b3": (10,),
            # Regression example (README quick start)
            "w": (10, 5), "b": (5,),
        }
        return shapes.get(name)
    
    def _get_training_config(self) -> dict:
        """Extract training configuration from SIR learn blocks."""
        config = {}
        if not self._compile_result.sir:
            return config
            
        for learn_cfg in self._compile_result.sir.learn_configs:
            # We only support one learn config for now (the first one)
            if learn_cfg.epochs is not None and 'epochs' not in config:
                config['epochs'] = learn_cfg.epochs
            if learn_cfg.batch_size is not None and 'batch_size' not in config:
                config['batch_size'] = learn_cfg.batch_size
            if learn_cfg.optimizer is not None and 'optimizer_spec' not in config:
                config['optimizer_spec'] = self._eval_optimizer_spec(learn_cfg.optimizer)
                
        return config

    def _eval_optimizer_spec(self, expr: Any) -> Optional[Any]:
        """Convert AST optimizer call to OptimizerSpec."""
        from delta.frontend.ast import Call, Identifier, Literal
        if isinstance(expr, Call) and isinstance(expr.func, Identifier):
            name = expr.func.name.lower()
            kwargs = {}
            for k, v in expr.kwargs:
                if isinstance(v, Literal):
                    kwargs[k] = v.value
            
            from delta.stdlib.optim import OptimizerSpec
            lr = kwargs.pop('lr', kwargs.pop('learning_rate', 0.001))
            weight_decay = kwargs.pop('weight_decay', 0.0)
            return OptimizerSpec(type=name, lr=lr, weight_decay=weight_decay, kwargs=kwargs)
        return None
    
    def to(self, device: Union[str, torch.device]) -> "DeltaModel":
        """Move model to device. Use 'auto' for automatic GPU detection."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device) if isinstance(device, str) else device
        self._module = self._module.to(self._device)
        return self
    
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        import inspect
        sig = inspect.signature(self._module.forward)
        bound = sig.bind_partial(*args, **kwargs)
        
        # Fill in missing arguments with empty tensors or None if possible
        # This is useful for 'obs' variables that might not be provided during inference
        for name, param in sig.parameters.items():
            if name not in bound.arguments:
                # If it's a required argument but missing, try to provide a dummy value
                # (This is a bit controversial, but helpful for internal observations)
                # For now, let's just let it fail if we can't find it, but we could
                # look it up in some internal state if we had one.
                pass
                
        inputs = [a.to(self._device) if isinstance(a, torch.Tensor) else a for a in args]
        kw_inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return self._module(*inputs, **kw_inputs)
    
    def parameters(self):
        """Get model parameters."""
        return self._module.parameters()
    
    def train(self):
        """Set to training mode."""
        self._module.train()
        return self
    
    def eval(self):
        """Set to evaluation mode."""
        self._module.eval()
        return self
    
    def fit(
        self,
        data: Union[DataLoader, Any],
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        log_every: int = 10,
        batch_size: Optional[int] = None
    ) -> List[float]:
        """
        Train the model.
        
        Args:
            data: DataLoader or Dataset yielding (x, y) tuples
            epochs: Number of epochs (if None, use learn block or default 100)
            lr: Learning rate (if None, use learn block or default 3e-3)
            log_every: Print loss every N epochs
            batch_size: Batch size (if None, use learn block or default 64)
        
        Returns:
            List of epoch losses
        """
        # Extract default training config from SIR if available
        sir_config = self._get_training_config()
        
        # Merge arguments with SIR config and fallbacks
        epochs = epochs if epochs is not None else (sir_config.get('epochs') or 100)
        batch_size = batch_size if batch_size is not None else (sir_config.get('batch_size') or 64)
        
        # Optimizer setup
        from delta.stdlib.optim import create_optimizer, OptimizerSpec
        optimizer_spec = sir_config.get('optimizer_spec')
        
        if lr is not None:
            # Explicit LR overrides whatever is in the spec
            if optimizer_spec:
                optimizer_spec.lr = lr
            else:
                optimizer_spec = OptimizerSpec(type="adamw", lr=lr)
        elif not optimizer_spec:
            # Fallback default
            optimizer_spec = OptimizerSpec(type="adamw", lr=3e-3)
            
        optimizer = create_optimizer(self._module.parameters(), optimizer_spec)
        
        # Custom loading for dictionary inputs (e.g. {"x": x, "y": y})
        losses = []
        
        print(f"Training on {self._device}...")
        print("─" * 50)
        
        for epoch in range(1, epochs + 1):
            self._module.train()
            epoch_loss = 0.0
            
            for x, y in data:
                # Handle non-tensor labels (e.g., int from MNIST)
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x)
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y)
                x, y = x.to(self._device), y.to(self._device)
                
                optimizer.zero_grad()
                
                # Check if the module expects labels (y) as an input
                # This happens if the user defines 'obs y' and uses it in a constraint
                import inspect
                sig = inspect.signature(self._module.forward)
                forward_params = list(sig.parameters.keys())
                
                if 'y' in forward_params and len(forward_params) >= 2:
                    # Pass both x and y
                    logits = self._module(x, y=y)
                else:
                    # Just pass x
                    logits = self._module(x)
                
                # Check if the output is already a loss (scalar)
                if logits is not None and logits.dim() == 0:
                    loss = logits
                elif logits is not None and logits.dim() == 3:  # Sequence model [B, T, V]
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        y.view(-1)
                    )
                else:  # Classification [B, C]
                    if logits is None:
                        # Fallback if forward returns nothing
                        loss = torch.tensor(0.0, device=self._device, requires_grad=True)
                    else:
                        loss = torch.nn.functional.cross_entropy(logits, y)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(data)
            losses.append(avg_loss)
            
            if epoch % log_every == 0:
                print(f"Epoch {epoch:3d}  │  Loss: {avg_loss:.4f}")
        
        print("─" * 50)
        return losses
    
    def set_vocab(self, chars: str):
        """Set vocabulary for text generation."""
        self._vocab = {c: i for i, c in enumerate(sorted(set(chars)))}
        self._vocab_inv = {i: c for c, i in self._vocab.items()}
        return self
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        ctx_len: int = 32
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Starting text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            ctx_len: Context length for the model
        
        Returns:
            Generated text (excluding prompt)
        """
        if not self._vocab:
            raise ValueError("Call set_vocab(chars) first to set vocabulary")
        
        self._module.eval()
        
        # Encode and pad prompt
        tokens = [self._vocab.get(c, 0) for c in prompt]
        if len(tokens) < ctx_len:
            tokens = [0] * (ctx_len - len(tokens)) + tokens
        
        tokens = torch.tensor([tokens], dtype=torch.long, device=self._device)
        generated = []
        
        for _ in range(max_tokens):
            with torch.no_grad():
                ctx = tokens[:, -ctx_len:]
                logits = self._module(ctx)
                logits = logits[0, -1] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, next_tok.unsqueeze(0)], dim=1)
                generated.append(next_tok.item())
        
        return ''.join(self._vocab_inv.get(t, '?') for t in generated)
    
    def save(self, path: str):
        """Save model weights."""
        torch.save(self._module.state_dict(), path)
    
    def load(self, path: str):
        """Load model weights."""
        self._module.load_state_dict(torch.load(path, map_location=self._device))


def compile(source: Union[str, Path], mode: str = "train") -> DeltaModel:
    """
    Compile a Delta program.
    
    Args:
        source: Path to .delta file or inline Delta code
        mode: Execution mode (train, infer, analyze)
    
    Returns:
        Compiled DeltaModel
    
    Example:
        model = delta.compile("model.delta")
        model.fit(train_loader, epochs=100)
    """
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
        source_file = SourceFile.from_path(Path(source))
    else:
        source_file = SourceFile.from_string(source, "<inline>")
    
    result = Compiler(CompileOptions(mode=mode, optimize=False)).compile(source_file)
    
    if not result.success:
        raise RuntimeError(f"Compilation failed:\n" + "\n".join(str(e) for e in result.errors))
    
    return DeltaModel(result)


def run(
    source: Union[str, Path],
    data: DataLoader,
    epochs: int = 100,
    lr: float = 3e-3,
    device: str = "auto"
) -> DeltaModel:
    """
    Compile and train in one call.
    
    Args:
        source: Path to .delta file
        data: Training DataLoader
        epochs: Number of epochs
        lr: Learning rate
        device: Device ("auto", "cpu", "cuda")
    
    Returns:
        Trained model
    
    Example:
        model = delta.run("model.delta", train_loader, epochs=100)
    """
    model = compile(source)
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    model.fit(data, epochs=epochs, lr=lr)
    return model
