"""Synchronous paging runner for MoE layer expert execution with per-layer metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class LayerPageStats:
    """Per-layer paging metrics: cache hits, misses, expert load times."""

    layer_idx: int
    num_experts_requested: int
    cache_hits: int = 0
    cache_misses: int = 0
    expert_load_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "layer_idx": self.layer_idx,
            "num_experts_requested": self.num_experts_requested,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "expert_load_time_ms": self.expert_load_time_ms,
        }


class PagedGenerationRunner:
    """Synchronous paging runner for MoE inference.

    Tracks per-layer cache metrics and ensures experts are paged before execution.
    This is the core sync path for Stage 3; actual model execution is deferred to Task 4.
    """

    def __init__(self, num_layers: int, num_experts: int, cache: Any) -> None:
        """Initialize runner with layer/expert topology and cache.

        Args:
            num_layers: Number of MoE layers in the model
            num_experts: Number of experts per layer
            cache: ExpertCache instance for paging (supports get/put/stats)
        """
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        if not isinstance(num_experts, int) or num_experts < 1:
            raise ValueError("num_experts must be a positive integer")
        if cache is None:
            raise ValueError("cache must not be None")

        self.num_layers = num_layers
        self.num_experts = num_experts
        self.cache = cache
        self.per_layer_stats: dict[int, LayerPageStats] = {}

    def forward_paged_layer(
        self, layer_idx: int, expert_ids: list[int], hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """Execute a layer with paged expert access.

        Before executing the layer, checks cache for each requested expert.
        Records cache hits/misses and returns mock output matching input shape.

        Args:
            layer_idx: Index of the MoE layer
            expert_ids: List of expert IDs to page in
            hidden_state: Input hidden states [batch, seq_len, hidden_size]

        Returns:
            Mock output tensor with same shape as input (deferred to Task 4)
        """
        if not isinstance(layer_idx, int) or layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"layer_idx must be in range 0..{self.num_layers - 1}")
        if not isinstance(expert_ids, list):
            raise ValueError("expert_ids must be a list")
        if not all(isinstance(eid, int) and 0 <= eid < self.num_experts for eid in expert_ids):
            raise ValueError(f"all expert_ids must be in range 0..{self.num_experts - 1}")
        if not isinstance(hidden_state, torch.Tensor):
            raise ValueError("hidden_state must be a torch.Tensor")

        # Initialize per-layer stats if not present
        if layer_idx not in self.per_layer_stats:
            self.per_layer_stats[layer_idx] = LayerPageStats(
                layer_idx=layer_idx, num_experts_requested=len(expert_ids)
            )

        stats = self.per_layer_stats[layer_idx]

        # Check cache for each expert, track hits/misses
        for expert_id in expert_ids:
            expert_key = (layer_idx, expert_id)
            cached_weights = self.cache.get(expert_key)
            if cached_weights is not None:
                stats.cache_hits += 1
            else:
                stats.cache_misses += 1

        # Return mock output matching input shape (real execution deferred)
        return torch.zeros_like(hidden_state)

    def stats_dict(self) -> dict[str, Any]:
        """Aggregate per-layer and total paging metrics.

        Returns:
            Dictionary with per-layer stats and aggregated totals
        """
        per_layer = {}
        total_hits = 0
        total_misses = 0
        total_experts_requested = 0

        for layer_idx in sorted(self.per_layer_stats.keys()):
            stats = self.per_layer_stats[layer_idx]
            per_layer[f"layer_{layer_idx}"] = {
                "num_experts_requested": stats.num_experts_requested,
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "expert_load_time_ms": stats.expert_load_time_ms,
            }
            total_hits += stats.cache_hits
            total_misses += stats.cache_misses
            total_experts_requested += stats.num_experts_requested

        total_accesses = total_hits + total_misses
        hit_rate = total_hits / total_accesses if total_accesses > 0 else 0.0

        return {
            "per_layer": per_layer,
            "aggregated": {
                "total_experts_requested": total_experts_requested,
                "total_cache_hits": total_hits,
                "total_cache_misses": total_misses,
                "total_cache_accesses": total_accesses,
                "hit_rate": hit_rate,
            },
        }


__all__ = ["PagedGenerationRunner", "LayerPageStats"]
