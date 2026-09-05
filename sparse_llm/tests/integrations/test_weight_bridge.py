"""Tests for SparseMoEWeightBridge."""

import pytest
import torch
from sparse_llm.integrations.weight_bridge import SparseMoEWeightBridge
from sparse_llm.loading import LoadedWeightState, ModelInfo, PlacementPlan, ExpertCache


@pytest.fixture
def mock_loaded_state():
    """Create mock LoadedWeightState for testing."""
    model_info = ModelInfo(
        model_id="test-model",
        is_moe=True,
        num_layers=2,
        num_experts=4,
        num_experts_per_tok=2,
        shared_weight_bytes=100_000_000,
        expert_weight_bytes=10_000_000,
        total_bytes=180_000_000,
    )

    placement_plan = PlacementPlan(
        model_info=model_info,
        shared_device="cuda",
        hot_expert_slots=2,
        warm_expert_slots=4,
        cold_expert_count=2,
        gpu_utilization_pct=40.0,
        cpu_utilization_pct=30.0,
        estimated_load_time_sec=5.0,
    )

    # Create mock shared weights
    shared_weights = {
        "model.embed_tokens.weight": torch.randn(32000, 4096),
        "model.layers.0.input_layernorm.weight": torch.randn(4096),
        "model.layers.1.input_layernorm.weight": torch.randn(4096),
    }

    # Create expert cache with mock experts
    expert_cache = ExpertCache(
        gpu_slots=4,  # 2 experts per layer, 2 layers
        cpu_slots=8,  # Double the GPU slots
        expert_bytes=10_000_000,  # 10MB per expert
    )

    # Preload some experts
    for layer_id in range(2):
        for expert_id in range(2):  # First 2 experts per layer
            expert_weights = {
                "w1.weight": torch.randn(14336, 4096),
                "w2.weight": torch.randn(4096, 14336),
                "w3.weight": torch.randn(14336, 4096),
            }
            expert_cache.preload_gpu(layer_id, expert_id, expert_weights)

    return LoadedWeightState(
        model_info=model_info,
        placement_plan=placement_plan,
        shared_weights=shared_weights,
        expert_cache=expert_cache,
    )


def test_bridge_initialization(mock_loaded_state):
    """Test weight bridge initializes with LoadedWeightState."""
    bridge = SparseMoEWeightBridge(mock_loaded_state)

    assert bridge.loaded_state is mock_loaded_state
    assert bridge.expert_cache is mock_loaded_state.expert_cache
    assert bridge.shared_weights is mock_loaded_state.shared_weights
    assert bridge.model_info.is_moe is True


def test_get_shared_weight(mock_loaded_state):
    """Test retrieving shared (non-expert) weights."""
    bridge = SparseMoEWeightBridge(mock_loaded_state)

    # Get shared weight
    weight = bridge.get_weight("model.embed_tokens.weight")

    assert isinstance(weight, torch.Tensor)
    assert weight.shape == (32000, 4096)


def test_get_expert_weight(mock_loaded_state):
    """Test retrieving expert weights from cache."""
    bridge = SparseMoEWeightBridge(mock_loaded_state)

    # Get expert weight (layer 0, expert 1, w1)
    weight_name = "model.layers.0.block_sparse_moe.experts.1.w1.weight"
    weight = bridge.get_weight(weight_name)

    assert isinstance(weight, torch.Tensor)
    assert weight.shape == (14336, 4096)


def test_expert_name_parsing(mock_loaded_state):
    """Test parsing expert weight names."""
    bridge = SparseMoEWeightBridge(mock_loaded_state)

    # Test standard Mixtral-style name
    name = "model.layers.5.block_sparse_moe.experts.3.w2.weight"
    layer_id, expert_id, weight_key = bridge._parse_expert_name(name)

    assert layer_id == 5
    assert expert_id == 3
    assert weight_key == "w2.weight"


def test_weight_not_found_error(mock_loaded_state):
    """Test error handling for missing weights."""
    bridge = SparseMoEWeightBridge(mock_loaded_state)

    # Try to get non-existent shared weight
    with pytest.raises(KeyError, match="not found"):
        bridge.get_weight("model.nonexistent.weight")

    # Try to get expert weight with invalid expert ID
    with pytest.raises((KeyError, ValueError)):
        bridge.get_weight("model.layers.0.experts.999.w1.weight")
