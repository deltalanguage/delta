"""
std.nn - Neural network layers for Delta.

Provides layer definitions that can be used in Delta programs.
All layers are differentiable and work with Delta's constraint system.
"""

from __future__ import annotations
from typing import Optional, Union, Tuple, List, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# Activation Functions
# ============================================================

def relu(x: Tensor) -> Tensor:
    """ReLU activation."""
    return F.relu(x)


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    """Leaky ReLU activation."""
    return F.leaky_relu(x, negative_slope)


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """ELU activation."""
    return F.elu(x, alpha)


def selu(x: Tensor) -> Tensor:
    """SELU activation."""
    return F.selu(x)


def gelu(x: Tensor) -> Tensor:
    """GELU activation."""
    return F.gelu(x)


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation."""
    return torch.sigmoid(x)


def tanh(x: Tensor) -> Tensor:
    """Tanh activation."""
    return torch.tanh(x)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Softmax activation."""
    return F.softmax(x, dim=dim)


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Log softmax activation."""
    return F.log_softmax(x, dim=dim)


def softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """Softplus activation."""
    return F.softplus(x, beta=beta, threshold=threshold)


def softsign(x: Tensor) -> Tensor:
    """Softsign activation."""
    return x / (1 + torch.abs(x))


def mish(x: Tensor) -> Tensor:
    """Mish activation."""
    return x * torch.tanh(F.softplus(x))


def swish(x: Tensor) -> Tensor:
    """Swish/SiLU activation."""
    return x * torch.sigmoid(x)


def hard_sigmoid(x: Tensor) -> Tensor:
    """Hard sigmoid activation."""
    return F.hardsigmoid(x)


def hard_swish(x: Tensor) -> Tensor:
    """Hard swish activation."""
    return F.hardswish(x)


# ============================================================
# Linear Layers
# ============================================================

class Linear(nn.Module):
    """Linear (fully connected) layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias, device, dtype)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Functional linear layer."""
    return F.linear(x, weight, bias)


# ============================================================
# Convolutional Layers
# ============================================================

class Conv1d(nn.Module):
    """1D convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Conv2d(nn.Module):
    """2D convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


def conv1d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
           stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) -> Tensor:
    """Functional 1D convolution."""
    return F.conv1d(x, weight, bias, stride, padding, dilation, groups)


def conv2d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
           stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) -> Tensor:
    """Functional 2D convolution."""
    return F.conv2d(x, weight, bias, stride, padding, dilation, groups)


# ============================================================
# Pooling Layers
# ============================================================

def max_pool1d(x: Tensor, kernel_size: int, stride: Optional[int] = None,
               padding: int = 0) -> Tensor:
    """1D max pooling."""
    return F.max_pool1d(x, kernel_size, stride, padding)


def max_pool2d(x: Tensor, kernel_size: Union[int, Tuple[int, int]],
               stride: Optional[Union[int, Tuple[int, int]]] = None,
               padding: Union[int, Tuple[int, int]] = 0) -> Tensor:
    """2D max pooling."""
    return F.max_pool2d(x, kernel_size, stride, padding)


def avg_pool1d(x: Tensor, kernel_size: int, stride: Optional[int] = None,
               padding: int = 0) -> Tensor:
    """1D average pooling."""
    return F.avg_pool1d(x, kernel_size, stride, padding)


def avg_pool2d(x: Tensor, kernel_size: Union[int, Tuple[int, int]],
               stride: Optional[Union[int, Tuple[int, int]]] = None,
               padding: Union[int, Tuple[int, int]] = 0) -> Tensor:
    """2D average pooling."""
    return F.avg_pool2d(x, kernel_size, stride, padding)


def adaptive_avg_pool1d(x: Tensor, output_size: int) -> Tensor:
    """Adaptive 1D average pooling."""
    return F.adaptive_avg_pool1d(x, output_size)


def adaptive_avg_pool2d(x: Tensor, output_size: Union[int, Tuple[int, int]]) -> Tensor:
    """Adaptive 2D average pooling."""
    return F.adaptive_avg_pool2d(x, output_size)


# ============================================================
# Normalization Layers
# ============================================================

class BatchNorm1d(nn.Module):
    """1D batch normalization."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.bn(x)


class BatchNorm2d(nn.Module):
    """2D batch normalization."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.bn(x)


class LayerNorm(nn.Module):
    """Layer normalization."""
    
    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5,
                 elementwise_affine: bool = True) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.ln(x)


def layer_norm(x: Tensor, normalized_shape: List[int], weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None, eps: float = 1e-5) -> Tensor:
    """Functional layer normalization."""
    return F.layer_norm(x, normalized_shape, weight, bias, eps)


def batch_norm(x: Tensor, running_mean: Optional[Tensor], running_var: Optional[Tensor],
               weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
               training: bool = True, momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    """Functional batch normalization."""
    return F.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)


# ============================================================
# Dropout
# ============================================================

def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Dropout."""
    return F.dropout(x, p, training)


def dropout2d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """2D dropout (spatial dropout)."""
    return F.dropout2d(x, p, training)


class Dropout(nn.Module):
    """Dropout layer."""
    
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x)


# ============================================================
# Recurrent Layers
# ============================================================

class LSTM(nn.Module):
    """LSTM layer."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bias,
            batch_first, dropout, bidirectional
        )
    
    def forward(self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None):
        return self.lstm(x, hidden)


class GRU(nn.Module):
    """GRU layer."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, bias,
            batch_first, dropout, bidirectional
        )
    
    def forward(self, x: Tensor, hidden: Optional[Tensor] = None):
        return self.gru(x, hidden)


# ============================================================
# Attention
# ============================================================

def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None
) -> Tensor:
    """Scaled dot-product attention."""
    d_k = query.size(-1)
    scale = scale or (d_k ** -0.5)
    
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    
    if dropout_p > 0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    return torch.matmul(attn_weights, value)


class MultiheadAttention(nn.Module):
    """Multi-head attention."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True
    ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout, bias, batch_first=batch_first
        )
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return self.mha(query, key, value, key_padding_mask, attn_mask=attn_mask)


# ============================================================
# Embedding
# ============================================================

class Embedding(nn.Module):
    """Embedding layer."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)


def embedding(x: Tensor, weight: Tensor, padding_idx: Optional[int] = None) -> Tensor:
    """Functional embedding lookup."""
    return F.embedding(x, weight, padding_idx)


# ============================================================
# Loss Functions
# ============================================================

def mse_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Mean squared error loss."""
    return F.mse_loss(pred, target, reduction=reduction)


def l1_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """L1 (mean absolute error) loss."""
    return F.l1_loss(pred, target, reduction=reduction)


def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0,
                   reduction: str = "mean") -> Tensor:
    """Smooth L1 (Huber) loss."""
    return F.smooth_l1_loss(pred, target, beta=beta, reduction=reduction)


def cross_entropy(pred: Tensor, target: Tensor, weight: Optional[Tensor] = None,
                  reduction: str = "mean", label_smoothing: float = 0.0) -> Tensor:
    """Cross entropy loss."""
    return F.cross_entropy(pred, target, weight, reduction=reduction,
                          label_smoothing=label_smoothing)


def binary_cross_entropy(pred: Tensor, target: Tensor, weight: Optional[Tensor] = None,
                         reduction: str = "mean") -> Tensor:
    """Binary cross entropy loss."""
    return F.binary_cross_entropy(pred, target, weight, reduction=reduction)


def binary_cross_entropy_with_logits(pred: Tensor, target: Tensor,
                                     weight: Optional[Tensor] = None,
                                     reduction: str = "mean",
                                     pos_weight: Optional[Tensor] = None) -> Tensor:
    """Binary cross entropy with logits loss."""
    return F.binary_cross_entropy_with_logits(pred, target, weight, reduction=reduction,
                                              pos_weight=pos_weight)


def nll_loss(pred: Tensor, target: Tensor, weight: Optional[Tensor] = None,
             reduction: str = "mean") -> Tensor:
    """Negative log likelihood loss."""
    return F.nll_loss(pred, target, weight, reduction=reduction)


def kl_div(pred: Tensor, target: Tensor, reduction: str = "mean",
           log_target: bool = False) -> Tensor:
    """KL divergence loss."""
    return F.kl_div(pred, target, reduction=reduction, log_target=log_target)


def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    """Cosine similarity."""
    return F.cosine_similarity(x1, x2, dim, eps)


def triplet_margin_loss(anchor: Tensor, positive: Tensor, negative: Tensor,
                        margin: float = 1.0, p: float = 2.0,
                        reduction: str = "mean") -> Tensor:
    """Triplet margin loss."""
    return F.triplet_margin_loss(anchor, positive, negative, margin, p, reduction=reduction)


# ============================================================
# Container Module
# ============================================================

# Re-export torch.nn modules for convenience
Sequential = nn.Sequential
Linear = nn.Linear
ReLU = nn.ReLU
Sigmoid = nn.Sigmoid
Tanh = nn.Tanh
Dropout = nn.Dropout
BatchNorm1d = nn.BatchNorm1d
BatchNorm2d = nn.BatchNorm2d
LayerNorm = nn.LayerNorm
Embedding = nn.Embedding
Conv2d = nn.Conv2d
