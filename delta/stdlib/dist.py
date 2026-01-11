"""
std.dist - Probability distributions for Delta.

Provides distribution definitions for probabilistic programming
with reparameterization for gradient flow.
"""

from __future__ import annotations
from typing import Optional, Union, Tuple
from dataclasses import dataclass
import torch
from torch import Tensor
import torch.distributions as D


# ============================================================
# Distribution Wrappers
# ============================================================

class Distribution:
    """Base class for Delta distributions."""
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        """Draw a sample."""
        raise NotImplementedError
    
    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        """Draw a reparameterized sample (for gradients)."""
        raise NotImplementedError
    
    def log_prob(self, value: Tensor) -> Tensor:
        """Log probability of a value."""
        raise NotImplementedError
    
    def entropy(self) -> Tensor:
        """Entropy of the distribution."""
        raise NotImplementedError
    
    @property
    def mean(self) -> Tensor:
        """Mean of the distribution."""
        raise NotImplementedError
    
    @property
    def variance(self) -> Tensor:
        """Variance of the distribution."""
        raise NotImplementedError


# ============================================================
# Continuous Distributions
# ============================================================

class Normal(Distribution):
    """Normal (Gaussian) distribution."""
    
    def __init__(self, loc: Tensor, scale: Tensor) -> None:
        self.loc = loc
        self.scale = scale
        self._dist = D.Normal(loc, scale)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.sample(sample_shape)
    
    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.rsample(sample_shape)
    
    def log_prob(self, value: Tensor) -> Tensor:
        return self._dist.log_prob(value)
    
    def entropy(self) -> Tensor:
        return self._dist.entropy()
    
    @property
    def mean(self) -> Tensor:
        return self.loc
    
    @property
    def variance(self) -> Tensor:
        return self.scale ** 2


class LogNormal(Distribution):
    """Log-normal distribution."""
    
    def __init__(self, loc: Tensor, scale: Tensor) -> None:
        self.loc = loc
        self.scale = scale
        self._dist = D.LogNormal(loc, scale)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.sample(sample_shape)
    
    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.rsample(sample_shape)
    
    def log_prob(self, value: Tensor) -> Tensor:
        return self._dist.log_prob(value)
    
    def entropy(self) -> Tensor:
        return self._dist.entropy()
    
    @property
    def mean(self) -> Tensor:
        return self._dist.mean
    
    @property
    def variance(self) -> Tensor:
        return self._dist.variance


class Uniform(Distribution):
    """Uniform distribution."""
    
    def __init__(self, low: Tensor, high: Tensor) -> None:
        self.low = low
        self.high = high
        self._dist = D.Uniform(low, high)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.sample(sample_shape)
    
    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.rsample(sample_shape)
    
    def log_prob(self, value: Tensor) -> Tensor:
        return self._dist.log_prob(value)
    
    def entropy(self) -> Tensor:
        return self._dist.entropy()
    
    @property
    def mean(self) -> Tensor:
        return (self.low + self.high) / 2
    
    @property
    def variance(self) -> Tensor:
        return (self.high - self.low) ** 2 / 12


class Beta(Distribution):
    """Beta distribution."""
    
    def __init__(self, alpha: Tensor, beta: Tensor) -> None:
        self.alpha = alpha
        self.beta = beta
        self._dist = D.Beta(alpha, beta)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.sample(sample_shape)
    
    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.rsample(sample_shape)
    
    def log_prob(self, value: Tensor) -> Tensor:
        return self._dist.log_prob(value)
    
    def entropy(self) -> Tensor:
        return self._dist.entropy()
    
    @property
    def mean(self) -> Tensor:
        return self._dist.mean
    
    @property
    def variance(self) -> Tensor:
        return self._dist.variance


class Gamma(Distribution):
    """Gamma distribution."""
    
    def __init__(self, concentration: Tensor, rate: Tensor) -> None:
        self.concentration = concentration
        self.rate = rate
        self._dist = D.Gamma(concentration, rate)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.sample(sample_shape)
    
    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.rsample(sample_shape)
    
    def log_prob(self, value: Tensor) -> Tensor:
        return self._dist.log_prob(value)
    
    def entropy(self) -> Tensor:
        return self._dist.entropy()
    
    @property
    def mean(self) -> Tensor:
        return self._dist.mean
    
    @property
    def variance(self) -> Tensor:
        return self._dist.variance


class Exponential(Distribution):
    """Exponential distribution."""
    
    def __init__(self, rate: Tensor) -> None:
        self.rate = rate
        self._dist = D.Exponential(rate)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.sample(sample_shape)
    
    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.rsample(sample_shape)
    
    def log_prob(self, value: Tensor) -> Tensor:
        return self._dist.log_prob(value)
    
    def entropy(self) -> Tensor:
        return self._dist.entropy()
    
    @property
    def mean(self) -> Tensor:
        return 1.0 / self.rate
    
    @property
    def variance(self) -> Tensor:
        return 1.0 / (self.rate ** 2)


class Laplace(Distribution):
    """Laplace distribution."""
    
    def __init__(self, loc: Tensor, scale: Tensor) -> None:
        self.loc = loc
        self.scale = scale
        self._dist = D.Laplace(loc, scale)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.sample(sample_shape)
    
    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.rsample(sample_shape)
    
    def log_prob(self, value: Tensor) -> Tensor:
        return self._dist.log_prob(value)
    
    def entropy(self) -> Tensor:
        return self._dist.entropy()
    
    @property
    def mean(self) -> Tensor:
        return self.loc
    
    @property
    def variance(self) -> Tensor:
        return 2 * self.scale ** 2


# ============================================================
# Discrete Distributions
# ============================================================

class Bernoulli(Distribution):
    """Bernoulli distribution."""
    
    def __init__(self, probs: Optional[Tensor] = None, logits: Optional[Tensor] = None) -> None:
        self.probs = probs
        self.logits = logits
        self._dist = D.Bernoulli(probs=probs, logits=logits)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.sample(sample_shape)
    
    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        # Use Gumbel-Softmax for reparameterization
        if self.logits is not None:
            logits = self.logits
        else:
            logits = torch.log(self.probs / (1 - self.probs + 1e-8))
        
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
        return torch.sigmoid((logits + gumbel) / 0.1)
    
    def log_prob(self, value: Tensor) -> Tensor:
        return self._dist.log_prob(value)
    
    def entropy(self) -> Tensor:
        return self._dist.entropy()
    
    @property
    def mean(self) -> Tensor:
        return self._dist.mean
    
    @property
    def variance(self) -> Tensor:
        return self._dist.variance


class Categorical(Distribution):
    """Categorical distribution."""
    
    def __init__(self, probs: Optional[Tensor] = None, logits: Optional[Tensor] = None) -> None:
        self.probs = probs
        self.logits = logits
        self._dist = D.Categorical(probs=probs, logits=logits)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.sample(sample_shape)
    
    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        # Gumbel-Softmax for reparameterization
        if self.logits is not None:
            logits = self.logits
        else:
            logits = torch.log(self.probs + 1e-8)
        
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
        return torch.softmax((logits + gumbel) / 0.1, dim=-1)
    
    def log_prob(self, value: Tensor) -> Tensor:
        return self._dist.log_prob(value)
    
    def entropy(self) -> Tensor:
        return self._dist.entropy()
    
    @property
    def mean(self) -> Tensor:
        return self._dist.probs
    
    @property
    def variance(self) -> Tensor:
        p = self._dist.probs
        return p * (1 - p)


class Poisson(Distribution):
    """Poisson distribution."""
    
    def __init__(self, rate: Tensor) -> None:
        self.rate = rate
        self._dist = D.Poisson(rate)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.sample(sample_shape)
    
    def log_prob(self, value: Tensor) -> Tensor:
        return self._dist.log_prob(value)
    
    @property
    def mean(self) -> Tensor:
        return self.rate
    
    @property
    def variance(self) -> Tensor:
        return self.rate


# ============================================================
# Multivariate Distributions
# ============================================================

class MultivariateNormal(Distribution):
    """Multivariate normal distribution."""
    
    def __init__(
        self,
        loc: Tensor,
        covariance_matrix: Optional[Tensor] = None,
        scale_tril: Optional[Tensor] = None
    ) -> None:
        self.loc = loc
        self._dist = D.MultivariateNormal(
            loc, covariance_matrix=covariance_matrix, scale_tril=scale_tril
        )
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.sample(sample_shape)
    
    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.rsample(sample_shape)
    
    def log_prob(self, value: Tensor) -> Tensor:
        return self._dist.log_prob(value)
    
    def entropy(self) -> Tensor:
        return self._dist.entropy()
    
    @property
    def mean(self) -> Tensor:
        return self.loc
    
    @property
    def variance(self) -> Tensor:
        return self._dist.variance


class Dirichlet(Distribution):
    """Dirichlet distribution."""
    
    def __init__(self, concentration: Tensor) -> None:
        self.concentration = concentration
        self._dist = D.Dirichlet(concentration)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.sample(sample_shape)
    
    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> Tensor:
        return self._dist.rsample(sample_shape)
    
    def log_prob(self, value: Tensor) -> Tensor:
        return self._dist.log_prob(value)
    
    def entropy(self) -> Tensor:
        return self._dist.entropy()
    
    @property
    def mean(self) -> Tensor:
        return self._dist.mean
    
    @property
    def variance(self) -> Tensor:
        return self._dist.variance


# ============================================================
# Distribution Utilities
# ============================================================

def kl_divergence(p: Distribution, q: Distribution) -> Tensor:
    """KL divergence between two distributions."""
    return D.kl_divergence(p._dist, q._dist)


def sample(dist: Distribution, n: int = 1) -> Tensor:
    """Sample from a distribution."""
    if n == 1:
        return dist.sample()
    return dist.sample((n,))


def observe(value: Tensor, dist: Distribution) -> Tensor:
    """Compute log probability of observed value."""
    return dist.log_prob(value)
