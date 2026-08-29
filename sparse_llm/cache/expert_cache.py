import logging
from typing import Dict, Hashable, Optional, Tuple
from collections import OrderedDict
import threading
import time

logger = logging.getLogger(__name__)

ExpertKey = Tuple[int, int]


class ExpertCache:
    """Bounded LRU cache for layer-aware expert weights."""

    def __init__(self, max_experts: int = 22):
        if max_experts < 1:
            raise ValueError("max_experts must be positive")
        self.max_experts = max_experts
        self.cache: OrderedDict[Hashable, object] = OrderedDict()
        self.lock = threading.Lock()
        self.access_count: Dict[Hashable, int] = {}
        self.last_access_time: Dict[Hashable, float] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(expert_id: int, layer_id: Optional[int] = None) -> Hashable:
        return expert_id if layer_id is None else (layer_id, expert_id)

    def get(self, expert_id: int, layer_id: Optional[int] = None) -> Optional[object]:
        """Get an expert and update LRU metadata."""
        key = self._key(expert_id, layer_id)
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            self.hits += 1
            self.access_count[key] += 1
            self.last_access_time[key] = time.monotonic()
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, expert_id: int, weights, layer_id: Optional[int] = None) -> None:
        """Add an expert, evicting the least recently used entry if full."""
        key = self._key(expert_id, layer_id)
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

    def contains(self, expert_id: int, layer_id: Optional[int] = None) -> bool:
        """Return whether an expert is cached without affecting hit statistics."""
        key = self._key(expert_id, layer_id)
        with self.lock:
            return key in self.cache

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
                if self.cache else 0.0,
            }
