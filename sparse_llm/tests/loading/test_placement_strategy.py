"""Tests for placement strategy calculator."""

import pytest


def test_placement_strategy_moe_model():
    from sparse_llm.loading.placement_strategy import PlacementStrategyCalculator
    from sparse_llm.loading.resource_budget import ResourceBudget, GPUInfo, CPUInfo, StorageInfo
    from sparse_llm.loading.model_introspector import ModelInfo

    # Create budget with 24GB GPU, 32GB CPU (constrained to force cold experts)
    gpu = GPUInfo(device_id=0, total_bytes=24*1024**3, available_bytes=20*1024**3,
                  compute_capability=(8, 0), name="A100")
    cpu = CPUInfo(total_bytes=64*1024**3, available_bytes=32*1024**3, usable_bytes=24*1024**3)
    storage = StorageInfo(path="/cache", available_bytes=500*1024**3, is_ssd=True, estimated_bandwidth_mbps=1200)
    budget = ResourceBudget(gpus=[gpu], cpu=cpu, storage=storage)

    # Create model info (Mixtral-like: 32 layers, 8 experts per layer, 180MB per expert)
    model_info = ModelInfo(
        model_id="mixtral-test",
        is_moe=True,
        num_layers=32,
        num_experts=8,
        num_experts_per_tok=2,
        shared_weight_bytes=2*1024**3,  # 2GB shared
        expert_weight_bytes=180*1024**2,  # 180MB per expert
        total_bytes=50*1024**3
    )

    calculator = PlacementStrategyCalculator()
    plan = calculator.calculate(budget, model_info)

    # Shared weights should fit on GPU
    assert plan.shared_device == "cuda:0"

    # Hot experts should use remaining GPU space
    assert plan.hot_expert_slots > 0

    # Warm experts should use CPU
    assert plan.warm_expert_slots > 0

    # Some experts should be cold (on storage)
    total_experts = 32 * 8  # 256 experts
    assert plan.hot_expert_slots + plan.warm_expert_slots < total_experts


def test_placement_strategy_dense_model():
    from sparse_llm.loading.placement_strategy import PlacementStrategyCalculator
    from sparse_llm.loading.resource_budget import ResourceBudget, GPUInfo, CPUInfo, StorageInfo
    from sparse_llm.loading.model_introspector import ModelInfo

    gpu = GPUInfo(device_id=0, total_bytes=24*1024**3, available_bytes=20*1024**3,
                  compute_capability=(8, 0), name="A100")
    cpu = CPUInfo(total_bytes=64*1024**3, available_bytes=48*1024**3, usable_bytes=38*1024**3)
    storage = StorageInfo(path="/cache", available_bytes=500*1024**3, is_ssd=True, estimated_bandwidth_mbps=1200)
    budget = ResourceBudget(gpus=[gpu], cpu=cpu, storage=storage)

    # Dense model (Llama-like)
    model_info = ModelInfo(
        model_id="llama-test",
        is_moe=False,
        num_layers=32,
        num_experts=0,
        num_experts_per_tok=0,
        shared_weight_bytes=14*1024**3,  # All weights are "shared" for dense
        expert_weight_bytes=0,
        total_bytes=14*1024**3
    )

    calculator = PlacementStrategyCalculator()
    plan = calculator.calculate(budget, model_info)

    # Dense model: no expert paging
    assert plan.hot_expert_slots == 0
    assert plan.warm_expert_slots == 0
    assert plan.cold_expert_count == 0
