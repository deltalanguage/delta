"""
std.data - Data loading and batching for Delta.

Provides data pipeline utilities with shape tracking.
"""

from __future__ import annotations
from typing import (
    Any, Callable, Iterator, List, Optional, Tuple, TypeVar, Generic,
    Sequence, Union
)
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
import numpy as np


T = TypeVar('T')


# ============================================================
# Dataset Interface
# ============================================================

class Dataset(Generic[T]):
    """Base class for Delta datasets."""
    
    def __len__(self) -> int:
        """Return the number of samples."""
        raise NotImplementedError
    
    def __getitem__(self, index: int) -> T:
        """Return sample at index."""
        raise NotImplementedError
    
    def map(self, fn: Callable[[T], T]) -> 'MappedDataset[T]':
        """Apply a function to each sample."""
        return MappedDataset(self, fn)
    
    def filter(self, predicate: Callable[[T], bool]) -> 'FilteredDataset[T]':
        """Filter samples by predicate."""
        return FilteredDataset(self, predicate)
    
    def shuffle(self, seed: Optional[int] = None) -> 'ShuffledDataset[T]':
        """Shuffle the dataset."""
        return ShuffledDataset(self, seed)
    
    def take(self, n: int) -> 'SlicedDataset[T]':
        """Take first n samples."""
        return SlicedDataset(self, 0, n)
    
    def skip(self, n: int) -> 'SlicedDataset[T]':
        """Skip first n samples."""
        return SlicedDataset(self, n, len(self))
    
    def split(self, ratio: float) -> Tuple['SlicedDataset[T]', 'SlicedDataset[T]']:
        """Split dataset into two parts."""
        n = int(len(self) * ratio)
        return self.take(n), self.skip(n)
    
    def batch(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False
    ) -> 'DataLoader[T]':
        """Create a batched data loader."""
        return DataLoader(self, batch_size, shuffle, drop_last)


class MappedDataset(Dataset[T]):
    """Dataset with mapped transform."""
    
    def __init__(self, base: Dataset[T], fn: Callable[[T], T]) -> None:
        self._base = base
        self._fn = fn
    
    def __len__(self) -> int:
        return len(self._base)
    
    def __getitem__(self, index: int) -> T:
        return self._fn(self._base[index])


class FilteredDataset(Dataset[T]):
    """Filtered dataset view."""
    
    def __init__(self, base: Dataset[T], predicate: Callable[[T], bool]) -> None:
        self._base = base
        # Build index lazily on first access
        self._indices: Optional[List[int]] = None
        self._predicate = predicate
    
    def _build_indices(self) -> None:
        if self._indices is None:
            self._indices = [
                i for i in range(len(self._base))
                if self._predicate(self._base[i])
            ]
    
    def __len__(self) -> int:
        self._build_indices()
        return len(self._indices)
    
    def __getitem__(self, index: int) -> T:
        self._build_indices()
        return self._base[self._indices[index]]


class ShuffledDataset(Dataset[T]):
    """Shuffled dataset view."""
    
    def __init__(self, base: Dataset[T], seed: Optional[int] = None) -> None:
        self._base = base
        rng = np.random.RandomState(seed)
        self._indices = rng.permutation(len(base)).tolist()
    
    def __len__(self) -> int:
        return len(self._base)
    
    def __getitem__(self, index: int) -> T:
        return self._base[self._indices[index]]


class SlicedDataset(Dataset[T]):
    """Sliced dataset view."""
    
    def __init__(self, base: Dataset[T], start: int, end: int) -> None:
        self._base = base
        self._start = max(0, start)
        self._end = min(len(base), end)
    
    def __len__(self) -> int:
        return max(0, self._end - self._start)
    
    def __getitem__(self, index: int) -> T:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range")
        return self._base[self._start + index]


# ============================================================
# Concrete Dataset Types
# ============================================================

class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    """Dataset wrapping tensors."""
    
    def __init__(self, *tensors: Tensor) -> None:
        assert len(tensors) > 0, "At least one tensor required"
        size = tensors[0].size(0)
        assert all(t.size(0) == size for t in tensors), "Size mismatch"
        self._tensors = tensors
    
    def __len__(self) -> int:
        return self._tensors[0].size(0)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        return tuple(t[index] for t in self._tensors)


class ListDataset(Dataset[T]):
    """Dataset from a list."""
    
    def __init__(self, data: List[T]) -> None:
        self._data = data
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index: int) -> T:
        return self._data[index]


class RangeDataset(Dataset[int]):
    """Dataset over a range of integers."""
    
    def __init__(self, start: int, end: int, step: int = 1) -> None:
        self._range = range(start, end, step)
    
    def __len__(self) -> int:
        return len(self._range)
    
    def __getitem__(self, index: int) -> int:
        return self._range[index]


class ConcatDataset(Dataset[T]):
    """Concatenation of multiple datasets."""
    
    def __init__(self, datasets: List[Dataset[T]]) -> None:
        self._datasets = datasets
        self._lengths = [len(d) for d in datasets]
        self._cumsum = np.cumsum([0] + self._lengths).tolist()
    
    def __len__(self) -> int:
        return sum(self._lengths)
    
    def __getitem__(self, index: int) -> T:
        for i, (start, end) in enumerate(zip(self._cumsum[:-1], self._cumsum[1:])):
            if start <= index < end:
                return self._datasets[i][index - start]
        raise IndexError(f"Index {index} out of range")


class ZipDataset(Dataset[Tuple[Any, ...]]):
    """Zip multiple datasets together."""
    
    def __init__(self, datasets: List[Dataset]) -> None:
        self._datasets = datasets
        self._len = min(len(d) for d in datasets)
    
    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        return tuple(d[index] for d in self._datasets)


# ============================================================
# DataLoader
# ============================================================

class DataLoader(Generic[T]):
    """Batched data loader for Delta."""
    
    def __init__(
        self,
        dataset: Dataset[T],
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0
    ) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._num_workers = num_workers
    
    def __len__(self) -> int:
        n = len(self._dataset)
        if self._drop_last:
            return n // self._batch_size
        return (n + self._batch_size - 1) // self._batch_size
    
    def __iter__(self) -> Iterator[List[T]]:
        indices = list(range(len(self._dataset)))
        
        if self._shuffle:
            np.random.shuffle(indices)
        
        batch = []
        for idx in indices:
            batch.append(self._dataset[idx])
            if len(batch) == self._batch_size:
                yield self._collate(batch)
                batch = []
        
        if batch and not self._drop_last:
            yield self._collate(batch)
    
    def _collate(self, batch: List[T]) -> T:
        """Collate a batch of samples."""
        if not batch:
            return batch
        
        first = batch[0]
        
        if isinstance(first, Tensor):
            return torch.stack(batch)
        
        if isinstance(first, tuple):
            return tuple(
                self._collate([sample[i] for sample in batch])
                for i in range(len(first))
            )
        
        if isinstance(first, dict):
            return {
                key: self._collate([sample[key] for sample in batch])
                for key in first.keys()
            }
        
        if isinstance(first, np.ndarray):
            return torch.from_numpy(np.stack(batch))
        
        if isinstance(first, (int, float)):
            return torch.tensor(batch)
        
        return batch


# ============================================================
# PyTorch Integration
# ============================================================

class TorchDatasetWrapper(TorchDataset):
    """Wrap a Delta dataset for PyTorch DataLoader."""
    
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
    
    def __len__(self) -> int:
        return len(self._dataset)
    
    def __getitem__(self, index: int) -> Any:
        return self._dataset[index]


def to_torch_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False
) -> TorchDataLoader:
    """Create a PyTorch DataLoader from a Delta dataset."""
    wrapped = TorchDatasetWrapper(dataset)
    return TorchDataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


# ============================================================
# Convenience Functions
# ============================================================

def from_tensors(*tensors: Tensor) -> TensorDataset:
    """Create a dataset from tensors."""
    return TensorDataset(*tensors)


def from_numpy(*arrays: np.ndarray) -> TensorDataset:
    """Create a dataset from NumPy arrays."""
    tensors = tuple(torch.from_numpy(a) for a in arrays)
    return TensorDataset(*tensors)


def from_list(data: List[T]) -> ListDataset[T]:
    """Create a dataset from a list."""
    return ListDataset(data)


def arange(start: int, end: int, step: int = 1) -> RangeDataset:
    """Create a range dataset."""
    return RangeDataset(start, end, step)


def concat(*datasets: Dataset[T]) -> ConcatDataset[T]:
    """Concatenate datasets."""
    return ConcatDataset(list(datasets))


def zip_datasets(*datasets: Dataset) -> ZipDataset:
    """Zip datasets together."""
    return ZipDataset(list(datasets))


# ============================================================
# Synthetic Data Generators
# ============================================================

def synthetic_regression(
    n_samples: int,
    n_features: int,
    noise: float = 0.1,
    seed: Optional[int] = None
) -> TensorDataset:
    """Generate synthetic regression data."""
    if seed is not None:
        torch.manual_seed(seed)
    
    X = torch.randn(n_samples, n_features)
    w = torch.randn(n_features)
    y = X @ w + noise * torch.randn(n_samples)
    
    return TensorDataset(X, y.unsqueeze(1))


def synthetic_classification(
    n_samples: int,
    n_features: int,
    n_classes: int = 2,
    seed: Optional[int] = None
) -> TensorDataset:
    """Generate synthetic classification data."""
    if seed is not None:
        torch.manual_seed(seed)
    
    X = torch.randn(n_samples, n_features)
    
    if n_classes == 2:
        w = torch.randn(n_features)
        y = (X @ w > 0).long()
    else:
        W = torch.randn(n_features, n_classes)
        y = (X @ W).argmax(dim=1)
    
    return TensorDataset(X, y)


def synthetic_time_series(
    n_samples: int,
    seq_len: int,
    n_features: int,
    seed: Optional[int] = None
) -> TensorDataset:
    """Generate synthetic time series data."""
    if seed is not None:
        torch.manual_seed(seed)
    
    X = torch.randn(n_samples, seq_len, n_features)
    y = X.sum(dim=(1, 2))
    
    return TensorDataset(X, y.unsqueeze(1))
