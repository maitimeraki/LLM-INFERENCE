"""End-to-end integration tests for resource-aware loading system."""

import pytest
import torch


def test_end_to_end_initialization_with_small_model():
    """Test full initialization with a small real model (gpt2)."""
    from sparse_llm.loading import FourPhaseOrchestrator

    # Only run if GPU available and in CI/integration test mode
    if not torch.cuda.is_available():
        pytest.skip("GPU required for integration test")

    # Use gpt2 (small, widely available)
    model_id = "gpt2"

    orchestrator = FourPhaseOrchestrator()

    messages = []
    def progress_callback(msg: str):
        messages.append(msg)
        print(msg)

    # Run full initialization
    state = orchestrator.initialize(model_id, progress_callback=progress_callback)

    # Verify state
    assert state.shared_weights is not None
    assert len(state.shared_weights) > 0
    assert state.model_info.model_id == model_id

    # gpt2 is dense, so no experts
    assert state.model_info.is_moe == False
    assert state.placement_plan.hot_expert_slots == 0

    # Verify all phases reported
    assert any("Phase 1" in msg for msg in messages)
    assert any("Phase 2" in msg for msg in messages)
    assert any("Phase 3" in msg for msg in messages)
    assert any("Phase 4" in msg for msg in messages)


def test_cache_statistics_tracking():
    """Test that cache statistics are tracked correctly."""
    from sparse_llm.loading import ExpertCache
    import torch

    cache = ExpertCache(gpu_slots=2, cpu_slots=2, expert_bytes=1024)

    # Preload expert
    weights = {"w1": torch.randn(10, 10)}
    cache.preload_gpu(0, 0, weights)

    # Access it (GPU hit)
    _, tier = cache.get(0, 0)
    assert tier == "gpu"

    # Check stats
    stats = cache.get_stats()
    assert stats["total_accesses"] == 1
    assert stats["gpu_hits"] == 1
    assert stats["gpu_hit_rate"] == 1.0


def test_orchestrator_handles_errors_gracefully():
    """Test that orchestrator handles invalid model IDs gracefully."""
    from sparse_llm.loading import FourPhaseOrchestrator

    orchestrator = FourPhaseOrchestrator()

    # Try to load non-existent model
    with pytest.raises(Exception):  # Should raise some exception
        orchestrator.initialize("non-existent-model-xyz-123")


def test_placement_plan_validates_gpu_requirement_for_moe():
    """Test that MoE models require GPU."""
    from sparse_llm.loading.placement_strategy import PlacementStrategyCalculator
    from sparse_llm.loading.resource_budget import ResourceBudget, CPUInfo, StorageInfo
    from sparse_llm.loading.model_introspector import ModelInfo

    # Create budget with no GPU
    budget = ResourceBudget(
        gpus=[],
        cpu=CPUInfo(total_bytes=64*1024**3, available_bytes=48*1024**3, usable_bytes=38*1024**3),
        storage=StorageInfo(path="/cache", available_bytes=500*1024**3, is_ssd=True, estimated_bandwidth_mbps=1200)
    )

    # Create MoE model info
    model_info = ModelInfo(
        model_id="test-moe",
        is_moe=True,
        num_layers=2,
        num_experts=4,
        num_experts_per_tok=2,
        shared_weight_bytes=1*1024**3,
        expert_weight_bytes=100*1024**2,
        total_bytes=2*1024**3
    )

    calculator = PlacementStrategyCalculator()

    # Should raise error because MoE requires GPU
    with pytest.raises(ValueError, match="MoE models require at least one GPU"):
        calculator.calculate(budget, model_info)


def test_resource_budget_warnings():
    """Test that resource budget generates appropriate warnings."""
    from sparse_llm.loading.resource_budget import ResourceBudget, GPUInfo, CPUInfo, StorageInfo

    # Create budget with low CPU RAM
    gpu = GPUInfo(device_id=0, total_bytes=24*1024**3, available_bytes=20*1024**3,
                  compute_capability=(8, 0), name="A100")
    cpu = CPUInfo(total_bytes=6*1024**3, available_bytes=5*1024**3, usable_bytes=4*1024**3)  # Low RAM
    storage = StorageInfo(path="/cache", available_bytes=500*1024**3, is_ssd=True, estimated_bandwidth_mbps=1200)

    budget = ResourceBudget(gpus=[gpu], cpu=cpu, storage=storage)

    # Should have warning about low CPU RAM
    assert any("CPU RAM below recommended minimum" in w for w in budget.warnings)


def test_expert_cache_lru_eviction():
    """Test that LRU eviction works correctly."""
    import torch
    from sparse_llm.loading import ExpertCache

    # Create cache with only 2 GPU slots
    cache = ExpertCache(gpu_slots=2, cpu_slots=5, expert_bytes=100*1024**2)

    expert_a = {"w1": torch.randn(10, 10)}
    expert_b = {"w1": torch.randn(10, 10)}
    expert_c = {"w1": torch.randn(10, 10)}

    # Fill GPU cache
    cache.preload_gpu(0, 0, expert_a)
    cache.preload_gpu(0, 1, expert_b)

    # Access expert_a (makes it most recently used)
    cache.get(0, 0)

    # Add third expert - should evict expert_b (LRU)
    cache.preload_gpu(0, 2, expert_c)

    # expert_a should still be on GPU
    _, tier = cache.get(0, 0)
    assert tier == "gpu"

    # expert_b should be evicted to CPU
    _, tier = cache.get(0, 1)
    assert tier == "cpu"  # Was evicted from GPU, now on CPU

    # expert_c should be on GPU
    _, tier = cache.get(0, 2)
    assert tier == "gpu"


def test_model_introspector_detects_dense_model():
    """Test that ModelIntrospector correctly identifies dense models."""
    from sparse_llm.loading.model_introspector import ModelIntrospector
    from unittest.mock import Mock

    # Mock dense model config (Llama-like)
    config = Mock()
    config.num_hidden_layers = 32
    config.hidden_size = 4096
    config.intermediate_size = 11008
    # No num_local_experts attribute

    introspector = ModelIntrospector()
    model_info = introspector.introspect_from_config(config, model_id="llama-test")

    assert model_info.is_moe == False
    assert model_info.num_experts == 0
    assert model_info.expert_weight_bytes == 0
    assert model_info.total_bytes > 0


def test_loaded_weight_state_accessor_methods():
    """Test LoadedWeightState convenience methods."""
    import torch
    from sparse_llm.loading import LoadedWeightState, ExpertCache, ModelInfo, PlacementPlan

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

    # Preload an expert
    expert_weights = {"w1": torch.randn(10, 10)}
    expert_cache.preload_gpu(0, 0, expert_weights)

    state = LoadedWeightState(
        shared_weights=shared_weights,
        expert_cache=expert_cache,
        model_info=model_info,
        placement_plan=plan
    )

    # Test get_shared_weight
    embed = state.get_shared_weight("embed")
    assert embed.shape == (1000, 512)

    # Test get_expert_weights
    weights, tier = state.get_expert_weights(0, 0)
    assert "w1" in weights
    assert tier == "gpu"

    # Test get_cache_stats
    stats = state.get_cache_stats()
    assert "total_accesses" in stats
    assert "gpu_hits" in stats
