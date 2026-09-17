"""Tests for StreamingExpertLoader."""

import pytest
import torch
from sparse_llm.loading.streaming_expert_loader import StreamingExpertLoader

# Test constants matching ExpertFFN layout
HIDDEN_DIM = 4096
EXPERT_DIM = 14336


def test_streaming_expert_loader_initialization():
    """Test that StreamingExpertLoader can be initialized."""
    def dummy_loader(layer_id, expert_id):
        return {
            "w1.weight": torch.randn(EXPERT_DIM, HIDDEN_DIM),  # (expert_dim, hidden_dim)
            "w2.weight": torch.randn(HIDDEN_DIM, EXPERT_DIM),  # (hidden_dim, expert_dim)
        }

    loader = StreamingExpertLoader(
        storage_loader=dummy_loader,
        device="cpu",
        max_parallel=2
    )

    assert loader.device.type == "cpu"
    assert loader.max_parallel == 2


def test_load_expert_returns_weights():
    """Test that load_expert returns weights dict."""
    def dummy_loader(layer_id, expert_id):
        return {
            "w1.weight": torch.randn(EXPERT_DIM, HIDDEN_DIM),
            "w2.weight": torch.randn(HIDDEN_DIM, EXPERT_DIM),
        }

    loader = StreamingExpertLoader(
        storage_loader=dummy_loader,
        device="cpu",
        max_parallel=2
    )

    weights = loader.load_expert(0, 0)
    assert "w1.weight" in weights
    assert "w2.weight" in weights


def test_load_experts_parallel():
    """Test parallel expert loading."""
    def dummy_loader(layer_id, expert_id):
        return {
            "w1.weight": torch.randn(EXPERT_DIM, HIDDEN_DIM),
            "w2.weight": torch.randn(HIDDEN_DIM, EXPERT_DIM),
        }

    loader = StreamingExpertLoader(
        storage_loader=dummy_loader,
        device="cpu",
        max_parallel=4
    )

    results = loader.load_experts_parallel(layer_id=0, expert_ids=[0, 1, 2])

    assert len(results) == 3
    assert 0 in results
    assert 1 in results
    assert 2 in results
    assert "w1.weight" in results[0]
    assert "w2.weight" in results[0]


def test_memory_guard_integration():
    """Test that memory guard can be set."""
    def dummy_loader(layer_id, expert_id):
        return {
            "w1.weight": torch.randn(EXPERT_DIM, HIDDEN_DIM),
            "w2.weight": torch.randn(HIDDEN_DIM, EXPERT_DIM),
        }

    loader = StreamingExpertLoader(
        storage_loader=dummy_loader,
        device="cpu",
        max_parallel=2
    )

    # Create a mock memory guard
    class MockMemoryGuard:
        def __init__(self):
            self.can_allocate_called = False
            self.eviction_called = False

        def can_allocate(self, size):
            self.can_allocate_called = True
            return True

        def _trigger_eviction(self):
            self.eviction_called = True

    guard = MockMemoryGuard()
    loader.set_memory_guard(guard)

    assert loader.memory_guard is guard


def test_execute_and_free():
    """Test that expert is executed and memory is freed."""
    def dummy_loader(layer_id, expert_id):
        return {
            "w1.weight": torch.randn(EXPERT_DIM, HIDDEN_DIM),
            "w2.weight": torch.randn(HIDDEN_DIM, EXPERT_DIM),
        }

    loader = StreamingExpertLoader(
        storage_loader=dummy_loader,
        device="cpu",
        max_parallel=2
    )

    # Create test input [batch, hidden_dim]
    hidden = torch.randn(1, HIDDEN_DIM)

    # Execute expert
    output = loader.execute_expert(layer_id=0, expert_id=0, hidden_state=hidden)

    assert output.shape == hidden.shape
    assert isinstance(output, torch.Tensor)


def test_safe_load_to_device():
    """Test safe loading to device with memory guard."""
    def dummy_loader(layer_id, expert_id):
        return {
            "w1.weight": torch.randn(EXPERT_DIM, HIDDEN_DIM),
            "w2.weight": torch.randn(HIDDEN_DIM, EXPERT_DIM),
        }

    loader = StreamingExpertLoader(
        storage_loader=dummy_loader,
        device="cpu",
        max_parallel=2
    )

    weights = loader.load_expert(0, 0)
    device_weights = loader.safe_load_to_device(weights)

    assert all(v.device == loader.device for v in device_weights.values())


def test_no_cache_imports():
    """Verify that streaming_expert_loader has no cache dependencies."""
    import inspect
    from sparse_llm.loading import streaming_expert_loader

    source = inspect.getsource(streaming_expert_loader)

    # These should NOT appear in the source
    forbidden_imports = [
        "HierarchicalExpertLoader",
        "ExpertCache",
        "ExpertCacheManager",
        "_gpu_cache",
        "_cpu_cache",
    ]

    for forbidden in forbidden_imports:
        assert forbidden not in source, f"StreamingExpertLoader should not import {forbidden}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
