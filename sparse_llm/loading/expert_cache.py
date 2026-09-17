"""Three-tier LRU expert cache for dynamic expert placement."""

from __future__ import annotations
from collections import OrderedDict
from typing import Any

import torch


class ExpertCache:
    """Three-tier LRU cache: GPU (L1) → CPU (L2) → Storage (L3)."""

    def __init__(
        self,
        gpu_slots: int,
        cpu_slots: int,
        expert_bytes: int,
        storage_loader: Any | None = None
    ):
        """Initialize three-tier cache.

        Args:
            gpu_slots: Number of experts that fit on GPU
            cpu_slots: Number of experts that fit on CPU
            expert_bytes: Bytes per expert (for capacity tracking)
            storage_loader: Optional callable(layer_id, expert_id) -> dict for loading from disk
        """
        self.gpu_slots = gpu_slots
        self.cpu_slots = cpu_slots
        self.expert_bytes = expert_bytes
        self.storage_loader = storage_loader

        # LRU caches (OrderedDict maintains insertion order, move_to_end for LRU)
        self._gpu_cache: OrderedDict[tuple[int, int], dict[str, torch.Tensor]] = OrderedDict()
        self._cpu_cache: OrderedDict[tuple[int, int], dict[str, torch.Tensor]] = OrderedDict()

        # Statistics (use _stats to avoid conflict with stats() method)
        self._stats = {
            "total_accesses": 0,
            "gpu_hits": 0,
            "cpu_hits": 0,
            "storage_hits": 0,
            "evictions": 0,
            "promotions": 0
        }

    def preload_gpu(self, layer_id: int, expert_id: int, weights: dict[str, torch.Tensor]) -> None:
        """Preload expert weights directly to GPU cache.

        Args:
            layer_id: Layer index
            expert_id: Expert index within layer
            weights: Expert weight tensors
        """
        key = (layer_id, expert_id)

        # Evict if cache full
        if len(self._gpu_cache) >= self.gpu_slots:
            self._evict_from_gpu()

        # Move tensors to GPU
        gpu_weights = {name: tensor.cuda() if tensor.device.type == "cpu" else tensor
                       for name, tensor in weights.items()}

        self._gpu_cache[key] = gpu_weights

    def preload_cpu(self, layer_id: int, expert_id: int, weights: dict[str, torch.Tensor]) -> None:
        """Preload expert weights directly to CPU cache.

        Args:
            layer_id: Layer index
            expert_id: Expert index within layer
            weights: Expert weight tensors
        """
        key = (layer_id, expert_id)

        # Evict if cache full
        if len(self._cpu_cache) >= self.cpu_slots:
            self._evict_from_cpu()

        # Move tensors to CPU
        cpu_weights = {name: tensor.cpu() if tensor.device.type != "cpu" else tensor
                       for name, tensor in weights.items()}

        self._cpu_cache[key] = cpu_weights

    def get(self, layer_id: int, expert_id: int) -> tuple[dict[str, torch.Tensor], str]:
        """Get expert weights with LRU promotion.

        Args:
            layer_id: Layer index
            expert_id: Expert index within layer

        Returns:
            (weights, tier) where tier is "gpu", "cpu", or "storage"
        """
        key = (layer_id, expert_id)
        self._stats["total_accesses"] += 1

        # Check GPU cache
        if key in self._gpu_cache:
            self._stats["gpu_hits"] += 1
            self._gpu_cache.move_to_end(key)  # Mark as recently used
            return self._gpu_cache[key], "gpu"

        # Check CPU cache
        if key in self._cpu_cache:
            self._stats["cpu_hits"] += 1
            weights = self._cpu_cache[key]
            self._cpu_cache.move_to_end(key)  # Mark as recently used

            # Promote to GPU
            self._promote_to_gpu(key, weights)
            self._stats["promotions"] += 1

            return self._gpu_cache[key], "cpu"

        # Load from storage
        self._stats["storage_hits"] += 1

        if self.storage_loader is None:
            raise ValueError(f"Expert ({layer_id}, {expert_id}) not in cache and no storage loader configured")

        weights = self.storage_loader(layer_id, expert_id)

        # Load to CPU first
        self.preload_cpu(layer_id, expert_id, weights)

        # Then promote to GPU
        cpu_weights = self._cpu_cache[key]
        self._promote_to_gpu(key, cpu_weights)

        return self._gpu_cache[key], "storage"

    def get_or_load(
        self,
        key_or_expert: tuple[int, int] | int,
        loader: Any,
        layer_id: int | None = None,
        size_bytes: int | None = None,
        pin: bool = False
    ) -> Any:
        """Get expert from cache or load if missing (API compatibility with old cache).

        This method simply calls the loader, which handles the three-tier cache lookup internally.

        Args:
            key_or_expert: Either (layer_id, expert_id) tuple or expert_id int
            loader: Callable that loads and returns an ExpertFFN module
            layer_id: Layer ID (used if key_or_expert is int)
            size_bytes: Size in bytes (ignored - computed from tensors)
            pin: Whether to pin expert (ignored - handled by LRU)

        Returns:
            ExpertFFN module ready for forward pass
        """
        # The loader function is responsible for:
        # 1. Checking the three-tier cache (GPU → CPU → Storage)
        # 2. Loading from storage if needed
        # 3. Creating the ExpertFFN module
        # 4. Loading the weights into the module
        return loader()

    def unpin(self, expert_id: int, layer_id: int | None = None) -> None:
        """Unpin an expert (no-op for compatibility with old cache).

        The old cache had pin/unpin to prevent eviction during execution.
        The three-tier cache uses LRU, so this is a no-op.
        """
        pass

    def contains(self, expert_id: int, layer_id: int | None = None) -> bool:
        """Check if expert is in cache (any tier).

        Args:
            expert_id: Expert identifier
            layer_id: Layer identifier

        Returns:
            True if expert is in GPU or CPU cache
        """
        if layer_id is None:
            layer_id = 0

        key = (layer_id, expert_id)
        return key in self._gpu_cache or key in self._cpu_cache

    def _evict_from_gpu(self) -> None:
        """Evict least recently used expert from GPU to CPU."""
        if not self._gpu_cache:
            return

        # Pop oldest (LRU)
        key, weights = self._gpu_cache.popitem(last=False)
        self._stats["evictions"] += 1

        # Move to CPU cache
        self.preload_cpu(key[0], key[1], weights)

    def _evict_from_cpu(self) -> None:
        """Evict least recently used expert from CPU (discard)."""
        if not self._cpu_cache:
            return

        # Pop oldest (LRU) and discard
        self._cpu_cache.popitem(last=False)
        self._stats["evictions"] += 1

    def _promote_to_gpu(self, key: tuple[int, int], weights: dict[str, torch.Tensor]) -> None:
        """Promote expert from CPU to GPU."""
        # Evict if GPU cache full
        if len(self._gpu_cache) >= self.gpu_slots:
            self._evict_from_gpu()

        # Move to GPU
        gpu_weights = {name: tensor.cuda() for name, tensor in weights.items()}
        self._gpu_cache[key] = gpu_weights

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["total_accesses"]
        if total == 0:
            return {**self._stats, "gpu_hit_rate": 0.0, "cpu_hit_rate": 0.0, "storage_miss_rate": 0.0}

        return {
            **self._stats,
            "gpu_hit_rate": self._stats["gpu_hits"] / total,
            "cpu_hit_rate": self._stats["cpu_hits"] / total,
            "storage_miss_rate": self._stats["storage_hits"] / total
        }

    def stats(self) -> dict[str, Any]:
        """Alias for get_stats() for compatibility with ExpertCacheManager."""
        return self.get_stats()
