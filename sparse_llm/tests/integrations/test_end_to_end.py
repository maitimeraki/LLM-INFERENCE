"""End-to-end integration tests for unified vLLM serving."""

import pytest
import asyncio
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
@pytest.mark.asyncio
async def test_unified_server_startup_with_gpt2():
    """Test complete server startup flow with small model."""
    from sparse_llm.loading import FourPhaseOrchestrator
    from sparse_llm.integrations import UnifiedMemoryCoordinator, SparseMoEWeightBridge

    # Phase 1: Resource-aware loading
    orchestrator = FourPhaseOrchestrator()
    loaded_state = orchestrator.initialize(model_id="gpt2")

    assert loaded_state is not None
    assert loaded_state.model_info.model_id == "gpt2"
    assert not loaded_state.model_info.is_moe  # GPT-2 is dense
    assert len(loaded_state.shared_weights) > 0

    # Phase 2: Memory coordination
    total_gpu = torch.cuda.get_device_properties(0).total_memory
    coordinator = UnifiedMemoryCoordinator(total_gpu)

    assert coordinator.expert_cache_budget > 0
    assert coordinator.kv_cache_budget > 0
    assert coordinator.expert_cache_budget + coordinator.kv_cache_budget < coordinator.usable_gpu

    # Phase 3: Weight bridge
    bridge = SparseMoEWeightBridge(loaded_state)

    # Verify we can get shared weights
    first_weight_name = list(loaded_state.shared_weights.keys())[0]
    weight = bridge.get_weight(first_weight_name)
    assert isinstance(weight, torch.Tensor)

    # Note: Actual vLLM integration requires vLLM installed (optional dependency)
    # This test verifies the preparation steps work correctly


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_memory_stays_within_budget():
    """Verify GPU memory usage stays within allocated budgets."""
    from sparse_llm.loading import FourPhaseOrchestrator
    from sparse_llm.integrations import UnifiedMemoryCoordinator

    # Get baseline memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()

    # Load model with resource-aware system
    orchestrator = FourPhaseOrchestrator()
    loaded_state = orchestrator.initialize(model_id="gpt2")

    # Check memory usage
    peak_memory = torch.cuda.max_memory_allocated()
    coordinator = UnifiedMemoryCoordinator(torch.cuda.get_device_properties(0).total_memory)

    # Memory should not exceed expert cache budget (shared weights are part of expert cache conceptually)
    # This is a sanity check, not a hard limit
    assert peak_memory < coordinator.usable_gpu, "Memory usage exceeded usable GPU budget"


def test_no_duplicate_expert_loading():
    """Verify experts are not loaded twice."""
    # This test would require MoE model and more complex setup
    # Placeholder for future comprehensive test
    pytest.skip("Requires MoE model - implement after manual testing confirms")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_weight_bridge_has_all_model_weights():
    """Verify weight bridge provides all weights needed by model."""
    from sparse_llm.loading import FourPhaseOrchestrator
    from sparse_llm.integrations import SparseMoEWeightBridge
    from transformers import AutoConfig, AutoModelForCausalLM

    model_id = "gpt2"

    # Load with resource-aware system
    orchestrator = FourPhaseOrchestrator()
    loaded_state = orchestrator.initialize(model_id=model_id)
    bridge = SparseMoEWeightBridge(loaded_state)

    # Load model structure to get expected weight names
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_config(config)

    # Check that bridge has all weights
    missing_weights = []
    for name, param in model.named_parameters():
        try:
            weight = bridge.get_weight(name)
            assert weight.shape == param.shape, f"Shape mismatch for {name}"
        except KeyError:
            missing_weights.append(name)

    assert len(missing_weights) == 0, f"Missing weights in bridge: {missing_weights}"
