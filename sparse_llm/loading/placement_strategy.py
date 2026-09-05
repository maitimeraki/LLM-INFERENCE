"""Placement strategy calculation for optimal weight distribution."""

from __future__ import annotations

from sparse_llm.loading.resource_budget import ResourceBudget
from sparse_llm.loading.model_introspector import ModelInfo
from sparse_llm.loading.placement_plan import PlacementPlan


class PlacementStrategyCalculator:
    """Calculate optimal placement strategy for model weights."""

    def calculate(self, budget: ResourceBudget, model_info: ModelInfo) -> PlacementPlan:
        """Calculate placement plan based on available resources and model architecture.

        Args:
            budget: Available hardware resources
            model_info: Model architecture and size information

        Returns:
            PlacementPlan with optimal weight distribution

        Raises:
            ValueError: If model cannot fit in available resources
        """
        # Handle dense models (no expert paging)
        if not model_info.is_moe:
            return self._plan_dense_model(budget, model_info)

        # Handle MoE models with three-tier placement
        return self._plan_moe_model(budget, model_info)

    def _plan_dense_model(self, budget: ResourceBudget, model_info: ModelInfo) -> PlacementPlan:
        """Plan placement for dense (non-MoE) models."""
        # Check if model fits on GPU
        if len(budget.gpus) > 0 and model_info.total_bytes <= budget.total_gpu_bytes:
            # Fits on GPU
            shared_device = "cuda:0"
            gpu_utilization = (model_info.total_bytes / budget.total_gpu_bytes) * 100
            cpu_utilization = 0.0
            estimated_load_time = 20.0  # Fast GPU load
        elif model_info.total_bytes <= budget.total_cpu_bytes:
            # Doesn't fit on GPU, use CPU
            shared_device = "cpu"
            gpu_utilization = 0.0
            cpu_utilization = (model_info.total_bytes / budget.total_cpu_bytes) * 100
            estimated_load_time = 60.0  # Slower CPU load
        else:
            raise ValueError(
                f"Model size ({model_info.total_bytes / 1024**3:.1f}GB) exceeds available resources. "
                f"Available: GPU {budget.total_gpu_bytes / 1024**3:.1f}GB, CPU {budget.total_cpu_bytes / 1024**3:.1f}GB. "
                f"Free up memory or use a smaller model."
            )

        return PlacementPlan(
            model_info=model_info,
            shared_device=shared_device,
            hot_expert_slots=0,
            warm_expert_slots=0,
            cold_expert_count=0,
            gpu_utilization_pct=gpu_utilization,
            cpu_utilization_pct=cpu_utilization,
            estimated_load_time_sec=estimated_load_time
        )

    def _plan_moe_model(self, budget: ResourceBudget, model_info: ModelInfo) -> PlacementPlan:
        """Plan three-tier placement for MoE models."""
        total_experts = model_info.num_layers * model_info.num_experts
        expert_bytes = model_info.expert_weight_bytes

        # Step 1: Place shared weights on GPU (required)
        if len(budget.gpus) == 0:
            raise ValueError(
                "MoE models require at least one GPU. CPU-only mode not supported for MoE. "
                "Use a dense model or enable GPU."
            )

        if model_info.shared_weight_bytes > budget.total_gpu_bytes:
            raise ValueError(
                f"Shared weights ({model_info.shared_weight_bytes / 1024**3:.1f}GB) don't fit on GPU "
                f"({budget.total_gpu_bytes / 1024**3:.1f}GB). Model too large for this hardware."
            )

        shared_device = "cuda:0"

        # Step 2: Allocate hot expert cache on GPU (remaining space)
        gpu_remaining = budget.total_gpu_bytes - model_info.shared_weight_bytes
        hot_expert_slots = min(total_experts, max(0, int(gpu_remaining / expert_bytes)))

        # Step 3: Allocate warm expert cache on CPU (only for experts not in hot cache)
        remaining_experts = total_experts - hot_expert_slots
        warm_expert_slots = min(remaining_experts, max(0, int(budget.total_cpu_bytes / expert_bytes)))

        # Step 4: Remaining experts go to storage (cold)
        cold_expert_count = total_experts - hot_expert_slots - warm_expert_slots

        # Calculate utilization
        gpu_used = model_info.shared_weight_bytes + (hot_expert_slots * expert_bytes)
        gpu_utilization = (gpu_used / budget.total_gpu_bytes) * 100 if budget.total_gpu_bytes > 0 else 0.0

        cpu_used = warm_expert_slots * expert_bytes
        cpu_utilization = (cpu_used / budget.total_cpu_bytes) * 100 if budget.total_cpu_bytes > 0 else 0.0

        # Estimate load time
        # Shared weights: ~10-20s
        # Hot experts: ~1s per 10 experts on GPU
        # Warm experts: lazy loaded (not counted in cold start)
        estimated_load_time = 15.0 + (hot_expert_slots / 10.0)

        return PlacementPlan(
            model_info=model_info,
            shared_device=shared_device,
            hot_expert_slots=hot_expert_slots,
            warm_expert_slots=warm_expert_slots,
            cold_expert_count=cold_expert_count,
            gpu_utilization_pct=gpu_utilization,
            cpu_utilization_pct=cpu_utilization,
            estimated_load_time_sec=estimated_load_time
        )
