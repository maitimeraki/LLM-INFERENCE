"""Thread-safe bounded LRU storage for layer-aware expert weights."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable, Dict, Optional

ExpertKey = tuple[int, int]
_CacheKey = ExpertKey | int


class ExpertCache:
    """Bounded LRU cache with optional byte capacity and pin protection.

    ``max_experts`` remains for compatibility. When ``max_bytes`` is supplied,
    byte capacity is enforced in addition to the entry-count limit.
    """

    def __init__(self, max_experts: int = 22, max_bytes: int | None = None):
        if not isinstance(max_experts, int) or isinstance(max_experts, bool) or max_experts < 1:
            raise ValueError("max_experts must be positive")
        if max_bytes is not None and (
            not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1
        ):
            raise ValueError("max_bytes must be a positive integer")
        self.max_experts = max_experts
        self.max_bytes = max_bytes
        self.cache: OrderedDict[_CacheKey, object] = OrderedDict()
        self.lock = threading.RLock()
        self.access_count: Dict[_CacheKey, int] = {}
        self.last_access_time: Dict[_CacheKey, float] = {}
        self.entry_sizes: Dict[_CacheKey, int] = {}
        self.pin_count: Dict[_CacheKey, int] = {}
        self._inflight: dict[_CacheKey, tuple[threading.Event, list[BaseException]]] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.bytes_used = 0

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
            key for key in self.cache if isinstance(key, tuple) and key[1] == expert_id
        ]
        if len(matches) > 1:
            raise ValueError(f"expert {expert_id} has an ambiguous layer-aware cache entry")
        if matches:
            return matches[0]
        return expert_id

    @staticmethod
    def _infer_size(value: object) -> int:
        size = getattr(value, "nbytes", None)
        if isinstance(size, int) and size >= 0:
            return size
        numel = getattr(value, "numel", None)
        element_size = getattr(value, "element_size", None)
        if callable(numel) and callable(element_size):
            inferred = numel() * element_size()
            if isinstance(inferred, int) and inferred >= 0:
                return inferred
        return 0

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
        *,
        size_bytes: int | None = None,
        _pin: bool = False,
    ) -> None:
        """Add an expert, evicting the oldest unpinned entries if necessary."""
        key = self._normalize_key(key_or_expert, layer_id)
        if size_bytes is None:
            size_bytes = self._infer_size(weights)
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        with self.lock:
            old_size = self.entry_sizes.get(key, 0)
            replacing = key in self.cache
            required = max(0, size_bytes - old_size)
            if self.max_bytes is not None and size_bytes > self.max_bytes:
                available = max(0, self.max_bytes - self.bytes_used + old_size)
                raise MemoryError(
                    f"expert {key!r} requires {size_bytes} bytes; available "
                    f"{available} bytes in the {self.max_bytes}-byte cache budget"
                )
            while (
                (not replacing and len(self.cache) >= self.max_experts)
                or (self.max_bytes is not None and self.bytes_used + required > self.max_bytes)
            ):
                victim = self._oldest_unpinned(exclude=key)
                if victim is None:
                    raise MemoryError("cache capacity is exhausted by pinned expert entries")
                self._remove(victim)
                self.evictions += 1
                replacing = key in self.cache
            if replacing:
                self.bytes_used -= old_size
                self.cache[key] = weights
                self.cache.move_to_end(key)
                self.access_count[key] += 1
            else:
                self.cache[key] = weights
                self.access_count[key] = 1
            self.entry_sizes[key] = size_bytes
            self.bytes_used += size_bytes
            self.pin_count.setdefault(key, 0)
            if _pin:
                self.pin_count[key] += 1
            self.last_access_time[key] = time.monotonic()

    def _oldest_unpinned(self, *, exclude: _CacheKey | None = None) -> _CacheKey | None:
        for key in self.cache:
            if key != exclude and self.pin_count.get(key, 0) == 0:
                return key
        return None

    def _remove(self, key: _CacheKey) -> None:
        self.cache.pop(key, None)
        self.bytes_used -= self.entry_sizes.pop(key, 0)
        self.access_count.pop(key, None)
        self.last_access_time.pop(key, None)
        self.pin_count.pop(key, None)

    def pin(self, key_or_expert: ExpertKey | int, layer_id: Optional[int] = None) -> None:
        """Prevent an existing entry from eviction until it is unpinned."""
        with self.lock:
            key = self._resolve_lookup(key_or_expert, layer_id)
            if key not in self.cache:
                raise KeyError(f"cannot pin uncached expert {key!r}")
            self.pin_count[key] = self.pin_count.get(key, 0) + 1

    def unpin(self, key_or_expert: ExpertKey | int, layer_id: Optional[int] = None) -> None:
        """Release one pin held by a forward pass."""
        with self.lock:
            key = self._resolve_lookup(key_or_expert, layer_id)
            if key not in self.cache:
                return
            count = self.pin_count.get(key, 0)
            if count == 0:
                raise ValueError(f"expert {key!r} is not pinned")
            self.pin_count[key] = count - 1

    def get_or_load(
        self,
        key_or_expert: ExpertKey | int,
        loader: Callable[[], object],
        layer_id: Optional[int] = None,
        *,
        size_bytes: int | None = None,
        pin: bool = False,
    ) -> object:
        """Load one key once when concurrent callers request the same expert.

        When ``pin`` is true, admission and the first lease are one atomic cache
        operation; this prevents a newly loaded entry from being evicted before
        its caller can begin execution.
        """
        key = self._normalize_key(key_or_expert, layer_id)
        with self.lock:
            resolved = self._resolve_lookup(key, None)
            if resolved in self.cache:
                self.hits += 1
                self.access_count[resolved] += 1
                self.last_access_time[resolved] = time.monotonic()
                self.cache.move_to_end(resolved)
                if pin:
                    self.pin_count[resolved] = self.pin_count.get(resolved, 0) + 1
                return self.cache[resolved]
            self.misses += 1
            event_and_error = self._inflight.get(resolved)
            if event_and_error is None:
                event_and_error = (threading.Event(), [])
                self._inflight[resolved] = event_and_error
                owner = True
            else:
                owner = False
            event, errors = event_and_error
        if not owner:
            event.wait()
            with self.lock:
                if errors:
                    raise errors[0]
                if resolved not in self.cache:  # pragma: no cover - defensive race guard
                    raise RuntimeError(
                        f"in-flight load for expert {resolved!r} produced no cache entry"
                    )
                self.hits += 1
                self.access_count[resolved] += 1
                self.last_access_time[resolved] = time.monotonic()
                self.cache.move_to_end(resolved)
                if pin:
                    self.pin_count[resolved] = self.pin_count.get(resolved, 0) + 1
                return self.cache[resolved]
        try:
            value = loader()
            self.put(resolved, value, size_bytes=size_bytes, _pin=pin)
            return value
        except BaseException as error:
            errors.append(error)
            raise
        finally:
            with self.lock:
                event.set()
                self._inflight.pop(resolved, None)

    def contains(self, key_or_expert: ExpertKey | int, layer_id: Optional[int] = None) -> bool:
        """Return whether an expert is cached without changing statistics."""
        with self.lock:
            return self._resolve_lookup(key_or_expert, layer_id) in self.cache

    def clear(self) -> None:
        """Clear cached weights and counters."""
        with self.lock:
            self.cache.clear()
            self.access_count.clear()
            self.last_access_time.clear()
            self.entry_sizes.clear()
            self.pin_count.clear()
            self.bytes_used = 0
            self.hits = self.misses = self.evictions = 0

    def stats(self) -> dict:
        """Return cache occupancy and measured hit/miss statistics."""
        with self.lock:
            accesses = self.hits + self.misses
            return {
                "cached_experts": len(self.cache),
                "max_experts": self.max_experts,
                "max_bytes": self.max_bytes,
                "bytes_used": self.bytes_used,
                "pinned_experts": sum(1 for count in self.pin_count.values() if count),
                "total_accesses": accesses,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": self.hits / accesses if accesses else 0.0,
                "avg_access_count": sum(self.access_count.values()) / len(self.cache)
                if self.cache
                else 0.0,
            }


__all__ = ["ExpertCache", "ExpertKey"]
