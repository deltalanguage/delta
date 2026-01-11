"""
std.tensor - Tensor operations for Delta.

Provides tensor creation, manipulation, and math functions
that map to PyTorch operations.
"""

from __future__ import annotations
from typing import Optional, Union, Tuple, List, Sequence
import torch
from torch import Tensor


# Type aliases
Shape = Union[int, Tuple[int, ...], List[int]]
DType = Optional[torch.dtype]
Device = Optional[Union[str, torch.device]]


# ============================================================
# Tensor Creation
# ============================================================

def zeros(*shape: int, dtype: DType = None, device: Device = None) -> Tensor:
    """Create a tensor filled with zeros."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return torch.zeros(*shape, dtype=dtype, device=device)


def ones(*shape: int, dtype: DType = None, device: Device = None) -> Tensor:
    """Create a tensor filled with ones."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return torch.ones(*shape, dtype=dtype, device=device)


def full(*shape_and_value, dtype: DType = None, device: Device = None) -> Tensor:
    """Create a tensor filled with a constant value.
    
    Usage: full(d1, d2, ..., fill_value) or full(shape_tuple, fill_value)
    """
    *shape, fill_value = shape_and_value
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return torch.full(shape, fill_value, dtype=dtype, device=device)


def randn(*shape: int, dtype: DType = None, device: Device = None) -> Tensor:
    """Create a tensor with random normal values."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return torch.randn(*shape, dtype=dtype, device=device)


def rand(*shape: int, dtype: DType = None, device: Device = None) -> Tensor:
    """Create a tensor with random uniform values in [0, 1)."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return torch.rand(*shape, dtype=dtype, device=device)


def randint(low: int, high: int, shape: Shape, dtype: DType = None, device: Device = None) -> Tensor:
    """Create a tensor with random integers in [low, high)."""
    if isinstance(shape, int):
        shape = (shape,)
    return torch.randint(low, high, shape, dtype=dtype, device=device)


def arange(start: float, end: Optional[float] = None, step: float = 1, 
           dtype: DType = None, device: Device = None) -> Tensor:
    """Create a 1D tensor with evenly spaced values."""
    if end is None:
        end = start
        start = 0
    return torch.arange(start, end, step, dtype=dtype, device=device)


def linspace(start: float, end: float, steps: int, 
             dtype: DType = None, device: Device = None) -> Tensor:
    """Create a 1D tensor with linearly spaced values."""
    return torch.linspace(start, end, steps, dtype=dtype, device=device)


def eye(n: int, m: Optional[int] = None, dtype: DType = None, device: Device = None) -> Tensor:
    """Create an identity matrix."""
    return torch.eye(n, m or n, dtype=dtype, device=device)


def from_numpy(array) -> Tensor:
    """Create a tensor from a numpy array."""
    return torch.from_numpy(array)


def tensor(data, dtype: DType = None, device: Device = None) -> Tensor:
    """Create a tensor from data."""
    return torch.tensor(data, dtype=dtype, device=device)


# ============================================================
# Initialization Patterns
# ============================================================

def xavier_uniform(shape: Shape, gain: float = 1.0) -> Tensor:
    """Xavier/Glorot uniform initialization."""
    t = zeros(shape)
    torch.nn.init.xavier_uniform_(t, gain=gain)
    return t


def xavier_normal(shape: Shape, gain: float = 1.0) -> Tensor:
    """Xavier/Glorot normal initialization."""
    t = zeros(shape)
    torch.nn.init.xavier_normal_(t, gain=gain)
    return t


def kaiming_uniform(shape: Shape, a: float = 0, mode: str = "fan_in", 
                    nonlinearity: str = "leaky_relu") -> Tensor:
    """Kaiming/He uniform initialization."""
    t = zeros(shape)
    torch.nn.init.kaiming_uniform_(t, a=a, mode=mode, nonlinearity=nonlinearity)
    return t


def kaiming_normal(shape: Shape, a: float = 0, mode: str = "fan_in",
                   nonlinearity: str = "leaky_relu") -> Tensor:
    """Kaiming/He normal initialization."""
    t = zeros(shape)
    torch.nn.init.kaiming_normal_(t, a=a, mode=mode, nonlinearity=nonlinearity)
    return t


# ============================================================
# Shape Operations
# ============================================================

def shape(x: Tensor, dim: Optional[int] = None) -> Union[Tuple[int, ...], int]:
    """Get tensor shape or size along a dimension."""
    if dim is None:
        return tuple(x.shape)
    return x.size(dim)


def size(x: Tensor, dim: Optional[int] = None) -> Union[Tuple[int, ...], int]:
    """Alias for shape."""
    return shape(x, dim)


def numel(x: Tensor) -> int:
    """Number of elements in tensor."""
    return x.numel()


def reshape(x: Tensor, shape: Shape) -> Tensor:
    """Reshape a tensor."""
    return x.reshape(shape)


def flatten(x: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flatten a tensor."""
    return torch.flatten(x, start_dim, end_dim)


def squeeze(x: Tensor, dim: Optional[int] = None) -> Tensor:
    """Remove dimensions of size 1."""
    if dim is None:
        return torch.squeeze(x)
    return torch.squeeze(x, dim)


def unsqueeze(x: Tensor, dim: int) -> Tensor:
    """Add a dimension of size 1."""
    return torch.unsqueeze(x, dim)


def transpose(x: Tensor, dim0: int = 0, dim1: int = 1) -> Tensor:
    """Transpose two dimensions."""
    return torch.transpose(x, dim0, dim1)


def permute(x: Tensor, dims: Sequence[int]) -> Tensor:
    """Permute dimensions."""
    return x.permute(dims)


def expand(x: Tensor, *sizes: int) -> Tensor:
    """Expand tensor to larger size."""
    return x.expand(*sizes)


def repeat(x: Tensor, *sizes: int) -> Tensor:
    """Repeat tensor along dimensions."""
    return x.repeat(*sizes)


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Concatenate tensors."""
    return torch.cat(tensors, dim)


# Alias for concat
concat = cat


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Stack tensors along new dimension."""
    return torch.stack(tensors, dim)


def split(x: Tensor, split_size: int, dim: int = 0) -> Tuple[Tensor, ...]:
    """Split tensor into chunks."""
    return torch.split(x, split_size, dim)


def chunk(x: Tensor, chunks: int, dim: int = 0) -> Tuple[Tensor, ...]:
    """Split tensor into chunks."""
    return torch.chunk(x, chunks, dim)


def slice(x: Tensor, dim: int, start: int, end: int) -> Tensor:
    """Slice tensor along a dimension."""
    return torch.narrow(x, dim, start, end - start)


def narrow(x: Tensor, dim: int, start: int, length: int) -> Tensor:
    """Narrow tensor along a dimension."""
    return torch.narrow(x, dim, start, length)


def index_select(x: Tensor, dim: int, index: Tensor) -> Tensor:
    """Select indices along a dimension."""
    return torch.index_select(x, dim, index)


def gather(x: Tensor, dim: int, index: Tensor) -> Tensor:
    """Gather values along a dimension."""
    return torch.gather(x, dim, index)


# ============================================================
# Math Operations
# ============================================================

def add(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise addition."""
    return torch.add(x, y)


def sub(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise subtraction."""
    return torch.sub(x, y)


def mul(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise multiplication."""
    return torch.mul(x, y)


def div(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise division."""
    return torch.div(x, y)


def pow(x: Tensor, exponent: Union[float, Tensor]) -> Tensor:
    """Element-wise power."""
    return torch.pow(x, exponent)


def sqrt(x: Tensor) -> Tensor:
    """Element-wise square root."""
    return torch.sqrt(x)


def exp(x: Tensor) -> Tensor:
    """Element-wise exponential."""
    return torch.exp(x)


def log(x: Tensor) -> Tensor:
    """Element-wise natural logarithm."""
    return torch.log(x)


def log2(x: Tensor) -> Tensor:
    """Element-wise base-2 logarithm."""
    return torch.log2(x)


def log10(x: Tensor) -> Tensor:
    """Element-wise base-10 logarithm."""
    return torch.log10(x)


def abs(x: Tensor) -> Tensor:
    """Element-wise absolute value."""
    return torch.abs(x)


def neg(x: Tensor) -> Tensor:
    """Element-wise negation."""
    return torch.neg(x)


def sign(x: Tensor) -> Tensor:
    """Element-wise sign."""
    return torch.sign(x)


def floor(x: Tensor) -> Tensor:
    """Element-wise floor."""
    return torch.floor(x)


def ceil(x: Tensor) -> Tensor:
    """Element-wise ceiling."""
    return torch.ceil(x)


def round(x: Tensor) -> Tensor:
    """Element-wise rounding."""
    return torch.round(x)


def clamp(x: Tensor, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tensor:
    """Clamp values to range."""
    return torch.clamp(x, min=min_val, max=max_val)


# ============================================================
# Trigonometric Functions
# ============================================================

def sin(x: Tensor) -> Tensor:
    """Element-wise sine."""
    return torch.sin(x)


def cos(x: Tensor) -> Tensor:
    """Element-wise cosine."""
    return torch.cos(x)


def tan(x: Tensor) -> Tensor:
    """Element-wise tangent."""
    return torch.tan(x)


def asin(x: Tensor) -> Tensor:
    """Element-wise arcsine."""
    return torch.asin(x)


def acos(x: Tensor) -> Tensor:
    """Element-wise arccosine."""
    return torch.acos(x)


def atan(x: Tensor) -> Tensor:
    """Element-wise arctangent."""
    return torch.atan(x)


def sinh(x: Tensor) -> Tensor:
    """Element-wise hyperbolic sine."""
    return torch.sinh(x)


def cosh(x: Tensor) -> Tensor:
    """Element-wise hyperbolic cosine."""
    return torch.cosh(x)


def tanh(x: Tensor) -> Tensor:
    """Element-wise hyperbolic tangent."""
    return torch.tanh(x)


# ============================================================
# Reduction Operations
# ============================================================

def sum(x: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """Sum of elements."""
    if dim is None:
        return torch.sum(x)
    return torch.sum(x, dim=dim, keepdim=keepdim)


def mean(x: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """Mean of elements."""
    if dim is None:
        return torch.mean(x)
    return torch.mean(x, dim=dim, keepdim=keepdim)


def prod(x: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """Product of elements."""
    if dim is None:
        return torch.prod(x)
    return torch.prod(x, dim=dim, keepdim=keepdim)


def max(x: Tensor, dim: Optional[int] = None, keepdim: bool = False):
    """Maximum of elements."""
    if dim is None:
        return torch.max(x)
    return torch.max(x, dim=dim, keepdim=keepdim)


def min(x: Tensor, dim: Optional[int] = None, keepdim: bool = False):
    """Minimum of elements."""
    if dim is None:
        return torch.min(x)
    return torch.min(x, dim=dim, keepdim=keepdim)


def argmax(x: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """Index of maximum element."""
    return torch.argmax(x, dim=dim, keepdim=keepdim)


def argmin(x: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """Index of minimum element."""
    return torch.argmin(x, dim=dim, keepdim=keepdim)


def std(x: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """Standard deviation."""
    if dim is None:
        return torch.std(x)
    return torch.std(x, dim=dim, keepdim=keepdim)


def var(x: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """Variance."""
    if dim is None:
        return torch.var(x)
    return torch.var(x, dim=dim, keepdim=keepdim)


def norm(x: Tensor, p: float = 2, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """Norm of tensor."""
    return torch.norm(x, p=p, dim=dim, keepdim=keepdim)


# ============================================================
# Linear Algebra
# ============================================================

def matmul(x: Tensor, y: Tensor) -> Tensor:
    """Matrix multiplication."""
    return torch.matmul(x, y)


def mm(x: Tensor, y: Tensor) -> Tensor:
    """Matrix-matrix multiplication."""
    return torch.mm(x, y)


def mv(x: Tensor, y: Tensor) -> Tensor:
    """Matrix-vector multiplication."""
    return torch.mv(x, y)


def bmm(x: Tensor, y: Tensor) -> Tensor:
    """Batched matrix multiplication."""
    return torch.bmm(x, y)


def dot(x: Tensor, y: Tensor) -> Tensor:
    """Dot product."""
    return torch.dot(x, y)


def outer(x: Tensor, y: Tensor) -> Tensor:
    """Outer product."""
    return torch.outer(x, y)


def einsum(equation: str, *tensors: Tensor) -> Tensor:
    """Einstein summation."""
    return torch.einsum(equation, *tensors)


def inv(x: Tensor) -> Tensor:
    """Matrix inverse."""
    return torch.linalg.inv(x)


def det(x: Tensor) -> Tensor:
    """Matrix determinant."""
    return torch.linalg.det(x)


def svd(x: Tensor):
    """Singular value decomposition."""
    return torch.linalg.svd(x)


def eig(x: Tensor):
    """Eigenvalue decomposition."""
    return torch.linalg.eig(x)


def solve(A: Tensor, b: Tensor) -> Tensor:
    """Solve linear system Ax = b."""
    return torch.linalg.solve(A, b)


# ============================================================
# Comparison Operations
# ============================================================

def eq(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise equality."""
    return torch.eq(x, y)


def ne(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise inequality."""
    return torch.ne(x, y)


def lt(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise less than."""
    return torch.lt(x, y)


def le(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise less than or equal."""
    return torch.le(x, y)


def gt(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise greater than."""
    return torch.gt(x, y)


def ge(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise greater than or equal."""
    return torch.ge(x, y)


def isnan(x: Tensor) -> Tensor:
    """Check for NaN."""
    return torch.isnan(x)


def isinf(x: Tensor) -> Tensor:
    """Check for infinity."""
    return torch.isinf(x)


def isfinite(x: Tensor) -> Tensor:
    """Check for finite values."""
    return torch.isfinite(x)


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Element-wise conditional."""
    return torch.where(condition, x, y)


# ============================================================
# Matrix Operations
# ============================================================

def tril(x: Tensor, diagonal: int = 0) -> Tensor:
    """Lower triangular part of matrix."""
    return torch.tril(x, diagonal=diagonal)


def triu(x: Tensor, diagonal: int = 0) -> Tensor:
    """Upper triangular part of matrix."""
    return torch.triu(x, diagonal=diagonal)


def diag(x: Tensor, diagonal: int = 0) -> Tensor:
    """Extract or construct diagonal."""
    return torch.diag(x, diagonal=diagonal)


def causal_mask(x: Tensor) -> Tensor:
    """
    Create causal attention mask for transformer.
    
    Returns a mask tensor where future positions are -inf
    and past/current positions are 0.
    
    Args:
        x: Attention scores tensor [..., T, T]
    
    Returns:
        Mask of same shape with -inf for future positions
    """
    T = x.size(-1)
    mask = torch.triu(torch.ones(T, T, device=x.device, dtype=x.dtype), diagonal=1)
    return mask * float('-inf')
