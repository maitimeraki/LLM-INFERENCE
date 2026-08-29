import logging
import threading
import time
from collections import OrderedDict
from typing import Dict, Optional

logger = logging.getLogger(__name__)

ExpertKey = tuple[int, int]
_CacheKey = ExpertKey | int


class ExpertCache:
    """Bounded LRU cache for layer-aware expert weights."""

    def __init__(self, max_experts: int = 22):
        if not isinstance(max_experts, int) or isinstance(max_experts, bool) or max_experts < 1:
            raise ValueError("max_experts must be positive")
        self.max_experts = max_experts
        self.cache: OrderedDict[_CacheKey, object] = OrderedDict()
        self.lock = threading.Lock()
        self.access_count: Dict[_CacheKey, int] = {}
        self.last_access_time: Dict[_CacheKey, float] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _validate_id(value: object, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        return value

    @classmethod
    def _normalize_key(
        cls, key_or_expert: ExpertKey | int, layer_id: Optional[int] = None
    ) -> _CacheKey:
        if isinstance(key_or_expert, tuple):
            if len(key_or_expert) != 2:
                raise ValueError("ExpertKey must be a (layer_id, expert_id) tuple")
            tuple_layer, expert_id = key_or_expert
            tuple_layer = cls._validate_id(tuple_layer, "layer_id")
            expert_id = cls._validate_id(expert_id, "expert_id")
            if layer_id is not None:
                layer_id = cls._validate_id(layer_id, "layer_id")
                if layer_id != tuple_layer:
                    raise ValueError("layer_id conflicts with the ExpertKey")
            return (tuple_layer, expert_id)

        expert_id = cls._validate_id(key_or_expert, "expert_id")
        if layer_id is None:
            return expert_id
        return (cls._validate_id(layer_id, "layer_id"), expert_id)

    def _resolve_lookup(self, key_or_expert: ExpertKey | int, layer_id: Optional[int]) -> _CacheKey:
        if isinstance(key_or_expert, tuple) or layer_id is not None:
            return self._normalize_key(key_or_expert, layer_id)

        expert_id = self._normalize_key(key_or_expert)
        if expert_id in self.cache:
            return expert_id

        matches = [
            key
            for key in self.cache
            if isinstance(key, tuple) and key[1] == expert_id
        ]
        if len(matches) > 1:
            raise ValueError(f"expert {expert_id} has an ambiguous layer-aware cache entry")
        if matches:
            return matches[0]
        return expert_id

    def get(self, key_or_expert: ExpertKey | int, layer_id: Optional[int] = None) -> object | None:
        """Get an expert, updating hit/miss counters and LRU metadata."""
        with self.lock:
            key = self._resolve_lookup(key_or_expert, layer_id)
            if key not in self.cache:
                self.misses += 1
                return None
            self.hits += 1
            self.access_count[key] += 1
            self.last_access_time[key] = time.monotonic()
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(
        self,
        key_or_expert: ExpertKey | int,
        weights: object,
        layer_id: Optional[int] = None,
    ) -> None:
        """Add an expert, evicting the least recently used entry if full."""
        key = self._normalize_key(key_or_expert, layer_id)
        with self.lock:
            if key in self.cache:
                self.cache[key] = weights
                self.cache.move_to_end(key)
                self.access_count[key] += 1
                self.last_access_time[key] = time.monotonic()
                return
            if len(self.cache) >= self.max_experts:
                victim, _ = self.cache.popitem(last=False)
                self.access_count.pop(victim, None)
                self.last_access_time.pop(victim, None)
            self.cache[key] = weights
            self.access_count[key] = 1
            self.last_access_time[key] = time.monotonic()

    def contains(self, key_or_expert: ExpertKey | int, layer_id: Optional[int] = None) -> bool:
        """Return whether an expert is cached without changing cache statistics."""
        with self.lock:
            return self._resolve_lookup(key_or_expert, layer_id) in self.cache

    def clear(self) -> None:
        """Clear cached weights and counters."""
        with self.lock:
            self.cache.clear()
            self.access_count.clear()
            self.last_access_time.clear()
            self.hits = self.misses = 0

    def stats(self) -> dict:
        """Return cache occupancy and measured hit/miss statistics."""
        with self.lock:
            accesses = self.hits + self.misses
            return {
                "cached_experts": len(self.cache),
                "max_experts": self.max_experts,
                "total_accesses": accesses,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / accesses if accesses else 0.0,
                "avg_access_count": sum(self.access_count.values()) / len(self.cache)
                if self.cache
                else 0.0,
            }


__all__ = ["ExpertCache", "ExpertKey"]
