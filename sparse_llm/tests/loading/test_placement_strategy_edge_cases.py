"""Additional edge case tests for placement strategy."""

import pytest
from sparse_llm.loading.placement_strategy import PlacementStrategyCalculator
from sparse_llm.loading.resource_budget import ResourceBudget, GPUInfo, CPUInfo, StorageInfo
from sparse_llm.loading.model_introspector import ModelInfo


def test_moe_no_gpu_raises():
    """MoE models require GPU."""
    cpu = CPUInfo(total_bytes=64*1024**3, available_bytes=48*1024**3, usable_bytes=38*1024**3)
    storage = StorageInfo(path="/cache", available_bytes=500*1024**3, is_ssd=True, estimated_bandwidth_mbps=1200)
    budget = ResourceBudget(gpus=[], cpu=cpu, storage=storage)

    model_info = ModelInfo(
        model_id="mixtral-test",
        is_moe=True,
        num_layers=32,
        num_experts=8,
        num_experts_per_tok=2,
        shared_weight_bytes=2*1024**3,
        expert_weight_bytes=180*1024**2,
        total_bytes=50*1024**3
    )

    calculator = PlacementStrategyCalculator()
    with pytest.raises(ValueError, match="MoE models require at least one GPU"):
        calculator.calculate(budget, model_info)


def test_moe_shared_weights_too_large():
    """Shared weights must fit on GPU."""
    gpu = GPUInfo(device_id=0, total_bytes=8*1024**3, available_bytes=6*1024**3,
                  compute_capability=(8, 0), name="RTX 3060")
    cpu = CPUInfo(total_bytes=32*1024**3, available_bytes=24*1024**3, usable_bytes=18*1024**3)
    storage = StorageInfo(path="/cache", available_bytes=500*1024**3, is_ssd=True, estimated_bandwidth_mbps=1200)
    budget = ResourceBudget(gpus=[gpu], cpu=cpu, storage=storage)

    model_info = ModelInfo(
        model_id="mixtral-test",
        is_moe=True,
        num_layers=32,
        num_experts=8,
        num_experts_per_tok=2,
        shared_weight_bytes=10*1024**3,  # Too large
        expert_weight_bytes=180*1024**2,
        total_bytes=50*1024**3
    )

    calculator = PlacementStrategyCalculator()
    with pytest.raises(ValueError, match="Shared weights.*don't fit on GPU"):
        calculator.calculate(budget, model_info)


def test_dense_model_too_large():
    """Dense model that exceeds all available memory."""
    gpu = GPUInfo(device_id=0, total_bytes=8*1024**3, available_bytes=6*1024**3,
                  compute_capability=(8, 0), name="RTX 3060")
    cpu = CPUInfo(total_bytes=16*1024**3, available_bytes=12*1024**3, usable_bytes=8*1024**3)
    storage = StorageInfo(path="/cache", available_bytes=500*1024**3, is_ssd=True, estimated_bandwidth_mbps=1200)
    budget = ResourceBudget(gpus=[gpu], cpu=cpu, storage=storage)

    model_info = ModelInfo(
        model_id="llama-huge",
        is_moe=False,
        num_layers=80,
        num_experts=0,
        num_experts_per_tok=0,
        shared_weight_bytes=70*1024**3,  # 70GB - way too large
        expert_weight_bytes=0,
        total_bytes=70*1024**3
    )

    calculator = PlacementStrategyCalculator()
    with pytest.raises(ValueError, match="Model size.*exceeds available resources"):
        calculator.calculate(budget, model_info)


def test_moe_all_experts_hot():
    """Small MoE where all experts fit on GPU."""
    gpu = GPUInfo(device_id=0, total_bytes=40*1024**3, available_bytes=38*1024**3,
                  compute_capability=(8, 0), name="A100")
    cpu = CPUInfo(total_bytes=64*1024**3, available_bytes=48*1024**3, usable_bytes=38*1024**3)
    storage = StorageInfo(path="/cache", available_bytes=500*1024**3, is_ssd=True, estimated_bandwidth_mbps=1200)
    budget = ResourceBudget(gpus=[gpu], cpu=cpu, storage=storage)

    # Small model: 8 layers, 4 experts, 100MB each
    model_info = ModelInfo(
        model_id="small-moe",
        is_moe=True,
        num_layers=8,
        num_experts=4,
        num_experts_per_tok=2,
        shared_weight_bytes=2*1024**3,  # 2GB shared
        expert_weight_bytes=100*1024**2,  # 100MB per expert
        total_bytes=5*1024**3
    )

    calculator = PlacementStrategyCalculator()
    plan = calculator.calculate(budget, model_info)

    total_experts = 8 * 4  # 32 experts
    assert plan.hot_expert_slots == total_experts  # All fit on GPU
    assert plan.warm_expert_slots == 0
    assert plan.cold_expert_count == 0


def test_dense_model_cpu_fallback():
    """Dense model that doesn't fit on GPU but fits on CPU."""
    gpu = GPUInfo(device_id=0, total_bytes=8*1024**3, available_bytes=6*1024**3,
                  compute_capability=(8, 0), name="RTX 3060")
    cpu = CPUInfo(total_bytes=64*1024**3, available_bytes=48*1024**3, usable_bytes=38*1024**3)
    storage = StorageInfo(path="/cache", available_bytes=500*1024**3, is_ssd=True, estimated_bandwidth_mbps=1200)
    budget = ResourceBudget(gpus=[gpu], cpu=cpu, storage=storage)

    model_info = ModelInfo(
        model_id="llama-13b",
        is_moe=False,
        num_layers=40,
        num_experts=0,
        num_experts_per_tok=0,
        shared_weight_bytes=26*1024**3,  # 26GB - won't fit on 6GB GPU
        expert_weight_bytes=0,
        total_bytes=26*1024**3
    )

    calculator = PlacementStrategyCalculator()
    plan = calculator.calculate(budget, model_info)

    assert plan.shared_device == "cpu"
    assert plan.gpu_utilization_pct == 0.0
    assert plan.cpu_utilization_pct > 0.0
