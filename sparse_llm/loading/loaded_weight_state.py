"""Loaded weight state container for Phase 3 output."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch

from sparse_llm.loading.expert_cache import ExpertCache
from sparse_llm.loading.model_introspector import ModelInfo
from sparse_llm.loading.placement_plan import PlacementPlan


@dataclass
class LoadedWeightState:
    """Container for loaded model weights and runtime cache."""

    shared_weights: dict[str, torch.Tensor]
    expert_cache: ExpertCache
    model_info: ModelInfo
    placement_plan: PlacementPlan

    def get_shared_weight(self, name: str) -> torch.Tensor:
        """Get a shared weight tensor by name.

        Args:
            name: Weight tensor name

        Returns:
            Weight tensor

        Raises:
            KeyError: If weight not found
        """
        return self.shared_weights[name]

    def get_expert_weights(self, layer_id: int, expert_id: int) -> tuple[dict[str, torch.Tensor], str]:
        """Get expert weights through cache.

        Args:
            layer_id: Layer index
            expert_id: Expert index

        Returns:
            (weights, tier) tuple
        """
        return self.expert_cache.get(layer_id, expert_id)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get expert cache statistics."""
        return self.expert_cache.get_stats()
