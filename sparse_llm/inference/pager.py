"""Synchronous, lease-safe expert paging over an indexed local checkpoint."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch

from sparse_llm.cache import ExpertCache, ExpertKey
from sparse_llm.storage.checkpoint_index import CheckpointIndex


@dataclass(frozen=True)
class LoadedExpert:
    """Materialized tensors for one expert and its measured source tier."""

    key: ExpertKey
    tensors: dict[str, torch.Tensor]
    source_tier: str
    byte_size: int
    load_duration_ms: float
    transfer_duration_ms: float


class ExpertPager:
    """Load indexed expert tensors on demand and lease them during execution.

    Implements three-tier memory hierarchy:
    - GPU tier: bounded active expert cache (self.cache on GPU device)
    - CPU tier: optional resident inactive experts (self._cpu_cache)
    - SSD tier: checkpoint-backed cold tensors (self.index)
    """

    def __init__(
        self,
        index: CheckpointIndex,
        cache: ExpertCache,
        *,
        device: str | torch.device = "cpu",
        enable_cpu_cache: bool = True,
    ) -> None:
        self.index = index
        self.cache = cache
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for expert paging but is unavailable")

        # CPU tier for staging when GPU is the target device
        self._cpu_cache: dict[ExpertKey, dict[str, torch.Tensor]] = {}
        self._enable_cpu_cache = enable_cpu_cache and self.device.type == "cuda"

        self._routed_keys: list[ExpertKey] = []
        self._load_durations_ms = 0.0
        self._transfer_durations_ms = 0.0
        self._cpu_hits = 0
        self._ssd_loads = 0

    @contextmanager
    def lease(self, key: ExpertKey) -> Iterator[LoadedExpert]:
        """Acquire one pinned expert and release its lease on every exit path."""
        key = self._validate_key(key)
        record = self.index.expert(key)
        started = time.perf_counter()
        loaded = self.cache.get_or_load(
            key,
            lambda: self._load(key),
            size_bytes=record.byte_size,
            pin=True,
        )
        try:
            yield loaded
        finally:
            self.cache.unpin(key)
            elapsed = (time.perf_counter() - started) * 1000
            self._load_durations_ms += elapsed

    def _load(self, key: ExpertKey) -> LoadedExpert:
        """Load expert through three-tier hierarchy: check CPU cache, then SSD."""
        source_tier = "ssd"
        load_ms = 0.0

        # Tier 2: Check CPU cache first (if enabled and GPU is target)
        if self._enable_cpu_cache and key in self._cpu_cache:
            load_started = time.perf_counter()
            tensors = self._cpu_cache[key]
            load_ms = (time.perf_counter() - load_started) * 1000
            source_tier = "cpu"
            self._cpu_hits += 1
        else:
            # Tier 3: Load from SSD checkpoint
            load_started = time.perf_counter()
            tensors = self.index.load_expert(key)
            load_ms = (time.perf_counter() - load_started) * 1000
            source_tier = "ssd"
            self._ssd_loads += 1

            # Cache in CPU tier if enabled (for future reuse)
            if self._enable_cpu_cache:
                self._cpu_cache[key] = {name: t.clone() for name, t in tensors.items()}

        # Transfer to target device (Tier 1: GPU or keep on CPU)
        transfer_started = time.perf_counter()
        if self.device.type != "cpu":
            tensors = {name: tensor.to(self.device) for name, tensor in tensors.items()}
        transfer_ms = (time.perf_counter() - transfer_started) * 1000

        record = self.index.expert(key)
        loaded = LoadedExpert(
            key=key,
            tensors=tensors,
            source_tier=source_tier,
            byte_size=record.byte_size,
            load_duration_ms=load_ms,
            transfer_duration_ms=transfer_ms,
        )
        self._transfer_durations_ms += transfer_ms
        return loaded

    def record_routed(self, keys) -> None:
        """Record unique routed keys in deterministic request order."""
        for key in keys:
            normalized = self._validate_key(key)
            if normalized not in self._routed_keys:
                self._routed_keys.append(normalized)

    def request_stats(self) -> dict[str, object]:
        cache_stats = self.cache.stats()
        return {
            "routed_keys": list(self._routed_keys),
            "cache_hits": cache_stats["hits"],
            "cache_misses": cache_stats["misses"],
            "evictions": cache_stats["evictions"],
            "bytes_used": cache_stats["bytes_used"],
            "expert_load_time_ms": self._load_durations_ms,
            "transfer_time_ms": self._transfer_durations_ms,
            "cpu_tier_hits": self._cpu_hits,
            "ssd_tier_loads": self._ssd_loads,
        }

    def clear_request_stats(self) -> None:
        self._routed_keys.clear()
        self._load_durations_ms = 0.0
        self._transfer_durations_ms = 0.0
        self._cpu_hits = 0
        self._ssd_loads = 0

    @staticmethod
    def _validate_key(key: object) -> ExpertKey:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("ExpertKey must be a (layer_index, expert_index) tuple")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in key):
            raise ValueError("ExpertKey indices must be non-negative integers")
        return key


__all__ = ["ExpertPager", "LoadedExpert"]
