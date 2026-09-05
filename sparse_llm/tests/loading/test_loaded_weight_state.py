"""Tests for LoadedWeightState container."""

import torch
import pytest


def test_loaded_weight_state_creation():
    from sparse_llm.loading.loaded_weight_state import LoadedWeightState
    from sparse_llm.loading.expert_cache import ExpertCache
    from sparse_llm.loading.model_introspector import ModelInfo
    from sparse_llm.loading.placement_plan import PlacementPlan

    model_info = ModelInfo(
        model_id="test", is_moe=True, num_layers=2, num_experts=4,
        num_experts_per_tok=2, shared_weight_bytes=1024, expert_weight_bytes=512, total_bytes=2048
    )

    plan = PlacementPlan(
        model_info=model_info, shared_device="cuda:0", hot_expert_slots=4,
        warm_expert_slots=2, cold_expert_count=2, gpu_utilization_pct=80.0,
        cpu_utilization_pct=20.0, estimated_load_time_sec=30.0
    )

    shared_weights = {"embed": torch.randn(1000, 512)}
    expert_cache = ExpertCache(gpu_slots=4, cpu_slots=2, expert_bytes=512)

    state = LoadedWeightState(
        shared_weights=shared_weights,
        expert_cache=expert_cache,
        model_info=model_info,
        placement_plan=plan
    )

    assert "embed" in state.shared_weights
    assert state.expert_cache.gpu_slots == 4
    assert state.model_info.is_moe == True
