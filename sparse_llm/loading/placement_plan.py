"""Placement plan dataclass for weight placement strategy."""

from __future__ import annotations
from dataclasses import dataclass

from sparse_llm.loading.model_introspector import ModelInfo


@dataclass(frozen=True)
class PlacementPlan:
    """Strategy for placing weights across GPU/CPU/storage tiers."""

    model_info: ModelInfo
    shared_device: str  # e.g., "cuda:0" or "cpu"
    hot_expert_slots: int  # GPU cache size (number of expert slots)
    warm_expert_slots: int  # CPU cache size (number of expert slots)
    cold_expert_count: int  # Experts that will be on storage
    gpu_utilization_pct: float  # Expected GPU memory usage percentage
    cpu_utilization_pct: float  # Expected CPU memory usage percentage
    estimated_load_time_sec: float  # Estimated time to load weights

    @property
    def total_expert_count(self) -> int:
        """Total number of experts in model (across all layers)."""
        if not self.model_info.is_moe:
            return 0
        return self.model_info.num_layers * self.model_info.num_experts

    @property
    def is_gpu_primary(self) -> bool:
        """True if shared weights will be placed on GPU."""
        return self.shared_device.startswith("cuda")

    @property
    def is_cpu_only(self) -> bool:
        """True if no GPU available (CPU-only mode)."""
        return self.shared_device == "cpu"
