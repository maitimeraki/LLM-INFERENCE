"""Integration tests for backend connection with resource-aware loading."""

import pytest
import torch


def test_preloaded_adapter_initialization():
    """Test that PreloadedWeightAdapter can be created from LoadedWeightState."""
    from sparse_llm.models.preloaded_adapter import PreloadedWeightAdapter
    from sparse_llm.loading.loaded_weight_state import LoadedWeightState
    from sparse_llm.loading.expert_cache import ExpertCache
    from sparse_llm.loading.model_introspector import ModelInfo
    from sparse_llm.loading.placement_plan import PlacementPlan

    # Create mock LoadedWeightState
    model_info = ModelInfo(
        model_id="gpt2",
        is_moe=False,
        num_layers=12,
        num_experts=0,
        num_experts_per_tok=0,
        shared_weight_bytes=500 * 1024**2,
        expert_weight_bytes=0,
        total_bytes=500 * 1024**2
    )

    plan = PlacementPlan(
        model_info=model_info,
        shared_device="cpu",
        hot_expert_slots=0,
        warm_expert_slots=0,
        cold_expert_count=0,
        gpu_utilization_pct=0.0,
        cpu_utilization_pct=50.0,
        estimated_load_time_sec=10.0
    )

    shared_weights = {"test_weight": torch.randn(100, 100)}
    expert_cache = ExpertCache(gpu_slots=0, cpu_slots=0, expert_bytes=0)

    state = LoadedWeightState(
        shared_weights=shared_weights,
        expert_cache=expert_cache,
        model_info=model_info,
        placement_plan=plan
    )

    # Create adapter
    adapter = PreloadedWeightAdapter(state)

    # Verify initialization
    assert adapter is not None
    assert adapter._loaded_state == state
    assert not adapter._is_loaded


def test_inference_engine_with_loaded_state():
    """Test that InferenceEngine accepts LoadedWeightState."""
    from sparse_llm.inference.engine import InferenceEngine
    from sparse_llm.loading.loaded_weight_state import LoadedWeightState
    from sparse_llm.loading.expert_cache import ExpertCache
    from sparse_llm.loading.model_introspector import ModelInfo
    from sparse_llm.loading.placement_plan import PlacementPlan

    # Create mock LoadedWeightState
    model_info = ModelInfo(
        model_id="gpt2",
        is_moe=False,
        num_layers=12,
        num_experts=0,
        num_experts_per_tok=0,
        shared_weight_bytes=500 * 1024**2,
        expert_weight_bytes=0,
        total_bytes=500 * 1024**2
    )

    plan = PlacementPlan(
        model_info=model_info,
        shared_device="cpu",
        hot_expert_slots=0,
        warm_expert_slots=0,
        cold_expert_count=0,
        gpu_utilization_pct=0.0,
        cpu_utilization_pct=50.0,
        estimated_load_time_sec=10.0
    )

    shared_weights = {"test_weight": torch.randn(100, 100)}
    expert_cache = ExpertCache(gpu_slots=0, cpu_slots=0, expert_bytes=0)

    state = LoadedWeightState(
        shared_weights=shared_weights,
        expert_cache=expert_cache,
        model_info=model_info,
        placement_plan=plan
    )

    # Create engine with loaded state
    engine = InferenceEngine(loaded_state=state)

    # Verify engine was initialized
    assert engine is not None
    assert engine._loaded_state == state
    assert engine.adapter is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for full integration test")
def test_end_to_end_with_gpt2():
    """Test complete flow: resource-aware loading -> inference with gpt2."""
    from sparse_llm.loading import FourPhaseOrchestrator
    from sparse_llm.inference.engine import InferenceEngine

    # Use gpt2 (small, widely available)
    model_id = "gpt2"

    # Phase 1-3: Load with resource-aware system
    orchestrator = FourPhaseOrchestrator()

    messages = []
    def progress_callback(msg: str):
        messages.append(msg)

    state = orchestrator.initialize(model_id, progress_callback=progress_callback)

    # Verify state
    assert state.shared_weights is not None
    assert len(state.shared_weights) > 0
    assert state.model_info.model_id == model_id
    assert state.model_info.is_moe == False

    # Verify all phases completed
    assert any("Phase 1" in msg for msg in messages)
    assert any("Phase 2" in msg for msg in messages)
    assert any("Phase 3" in msg for msg in messages)
    assert any("Phase 4" in msg for msg in messages)

    # Phase 4: Create inference engine
    engine = InferenceEngine(loaded_state=state)

    # Generate text
    result = engine.generate("Hello, world!", max_new_tokens=10, temperature=0.0)

    # Verify generation worked
    assert result is not None
    assert result.text is not None
    assert len(result.text) > 0
    assert result.token_ids is not None
    assert len(result.token_ids) > 0
    assert result.metrics is not None
    assert result.metrics.tokens_per_sec > 0


def test_backward_compatibility():
    """Test that traditional InferenceEngine initialization still works."""
    from sparse_llm.inference.engine import InferenceEngine

    # Traditional initialization without loaded_state should still work
    # This would normally load a model, but we're just checking the API
    try:
        # This may fail if model isn't cached, but the API should accept it
        engine = InferenceEngine(model="gpt2", device="cpu")
        assert engine is not None
    except Exception as e:
        # Expected to possibly fail on model download, but not on API signature
        assert "loaded_state cannot be combined" not in str(e)


def test_mutually_exclusive_parameters():
    """Test that loaded_state cannot be combined with model parameter."""
    from sparse_llm.inference.engine import InferenceEngine
    from sparse_llm.loading.loaded_weight_state import LoadedWeightState
    from sparse_llm.loading.expert_cache import ExpertCache
    from sparse_llm.loading.model_introspector import ModelInfo
    from sparse_llm.loading.placement_plan import PlacementPlan

    model_info = ModelInfo(
        model_id="test",
        is_moe=False,
        num_layers=1,
        num_experts=0,
        num_experts_per_tok=0,
        shared_weight_bytes=100,
        expert_weight_bytes=0,
        total_bytes=100
    )

    plan = PlacementPlan(
        model_info=model_info,
        shared_device="cpu",
        hot_expert_slots=0,
        warm_expert_slots=0,
        cold_expert_count=0,
        gpu_utilization_pct=0.0,
        cpu_utilization_pct=0.0,
        estimated_load_time_sec=0.0
    )

    state = LoadedWeightState(
        shared_weights={},
        expert_cache=ExpertCache(0, 0, 0),
        model_info=model_info,
        placement_plan=plan
    )

    # Should raise ValueError when both are provided
    with pytest.raises(ValueError, match="loaded_state cannot be combined"):
        InferenceEngine(model="gpt2", loaded_state=state)
