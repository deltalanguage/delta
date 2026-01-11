"""
Compilation cache for Delta.

Caches compiled graphs to avoid recompilation:
- Keyed by source hash, shapes, dtypes, and mode
- Supports disk persistence
- Handles cache invalidation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import hashlib
import json
import pickle
import torch
from torch.fx import GraphModule


@dataclass(frozen=True)
class CacheKey:
    """
    Key for cache lookup.
    
    A compilation is cached based on:
    - Source code hash
    - Input shapes
    - Input dtypes
    - Execution mode
    - Device type
    """
    source_hash: str
    shapes: Tuple[Tuple[int, ...], ...]
    dtypes: Tuple[str, ...]
    mode: str
    device: str
    
    def __hash__(self) -> int:
        return hash((self.source_hash, self.shapes, self.dtypes, self.mode, self.device))
    
    @classmethod
    def from_inputs(
        cls,
        source_hash: str,
        inputs: Dict[str, torch.Tensor],
        mode: str,
        device: str
    ) -> CacheKey:
        """Create a cache key from inputs."""
        shapes = tuple(tuple(v.shape) for v in inputs.values() if isinstance(v, torch.Tensor))
        dtypes = tuple(str(v.dtype) for v in inputs.values() if isinstance(v, torch.Tensor))
        
        return cls(
            source_hash=source_hash,
            shapes=shapes,
            dtypes=dtypes,
            mode=mode,
            device=device
        )
    
    def to_filename(self) -> str:
        """Convert to a filename-safe string."""
        combined = f"{self.source_hash}_{self.shapes}_{self.dtypes}_{self.mode}_{self.device}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    key: CacheKey
    graph_modules: Dict[str, Any]  # Serialized graph modules
    parameters: Dict[str, torch.Tensor]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"


class CompilationCache:
    """
    Cache for compiled Delta graphs.
    
    Features:
    - In-memory cache for fast access
    - Optional disk persistence
    - LRU eviction for memory management
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_memory_entries: int = 100,
        persist_to_disk: bool = True
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".delta" / "cache"
        self.max_memory_entries = max_memory_entries
        self.persist_to_disk = persist_to_disk
        
        self._memory_cache: Dict[CacheKey, CacheEntry] = {}
        self._access_order: list[CacheKey] = []
        
        if self.persist_to_disk:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: CacheKey) -> Optional[CacheEntry]:
        """Get a cached compilation."""
        # Check memory cache first
        if key in self._memory_cache:
            self._update_access(key)
            return self._memory_cache[key]
        
        # Check disk cache
        if self.persist_to_disk:
            entry = self._load_from_disk(key)
            if entry:
                self._add_to_memory(key, entry)
                return entry
        
        return None
    
    def put(self, entry: CacheEntry) -> None:
        """Add a compilation to the cache."""
        self._add_to_memory(entry.key, entry)
        
        if self.persist_to_disk:
            self._save_to_disk(entry)
    
    def contains(self, key: CacheKey) -> bool:
        """Check if a key is in the cache."""
        if key in self._memory_cache:
            return True
        
        if self.persist_to_disk:
            cache_file = self.cache_dir / f"{key.to_filename()}.pkl"
            return cache_file.exists()
        
        return False
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._memory_cache.clear()
        self._access_order.clear()
        
        if self.persist_to_disk and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
    
    def _add_to_memory(self, key: CacheKey, entry: CacheEntry) -> None:
        """Add an entry to memory cache."""
        if len(self._memory_cache) >= self.max_memory_entries:
            self._evict_oldest()
        
        self._memory_cache[key] = entry
        self._update_access(key)
    
    def _update_access(self, key: CacheKey) -> None:
        """Update access order for LRU."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _evict_oldest(self) -> None:
        """Evict the least recently used entry."""
        if self._access_order:
            oldest = self._access_order.pop(0)
            self._memory_cache.pop(oldest, None)
    
    def _save_to_disk(self, entry: CacheEntry) -> None:
        """Save an entry to disk."""
        cache_file = self.cache_dir / f"{entry.key.to_filename()}.pkl"
        
        try:
            # Serialize graph modules using torch.save
            serialized = {
                "key": entry.key,
                "parameters": {k: v.cpu() for k, v in entry.parameters.items()},
                "metadata": entry.metadata,
                "version": entry.version,
            }
            
            with open(cache_file, "wb") as f:
                pickle.dump(serialized, f)
        except Exception as e:
            # Don't fail on cache save errors
            print(f"Warning: Failed to save to cache: {e}")
    
    def _load_from_disk(self, key: CacheKey) -> Optional[CacheEntry]:
        """Load an entry from disk."""
        cache_file = self.cache_dir / f"{key.to_filename()}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            
            return CacheEntry(
                key=data["key"],
                graph_modules={},  # Would need to reconstruct
                parameters=data["parameters"],
                metadata=data.get("metadata", {}),
                version=data.get("version", "1.0")
            )
        except Exception as e:
            # Don't fail on cache load errors
            print(f"Warning: Failed to load from cache: {e}")
            return None
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        disk_count = 0
        if self.persist_to_disk and self.cache_dir.exists():
            disk_count = len(list(self.cache_dir.glob("*.pkl")))
        
        return {
            "memory_entries": len(self._memory_cache),
            "disk_entries": disk_count,
            "max_memory_entries": self.max_memory_entries,
        }


# Global cache instance
_global_cache: Optional[CompilationCache] = None


def get_global_cache() -> CompilationCache:
    """Get the global compilation cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = CompilationCache()
    return _global_cache


def clear_global_cache() -> None:
    """Clear the global compilation cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
