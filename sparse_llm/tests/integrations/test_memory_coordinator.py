"""Tests for UnifiedMemoryCoordinator."""

import pytest
from sparse_llm.integrations import UnifiedMemoryCoordinator


def test_memory_allocation_ratios():
    """Verify GPU memory split into correct ratios."""
    # Simulate 24GB GPU
    total_gpu = 24 * 1024 ** 3

    coordinator = UnifiedMemoryCoordinator(total_gpu_bytes=total_gpu)

    # Should use 85% of total
    assert coordinator.usable_gpu == int(total_gpu * 0.85)

    # Check allocation ratios
    expected_expert = int(coordinator.usable_gpu * 0.40)
    expected_kv = int(coordinator.usable_gpu * 0.50)
    expected_overhead = int(coordinator.usable_gpu * 0.10)

    assert coordinator.expert_cache_budget == expected_expert
    assert coordinator.kv_cache_budget == expected_kv
    assert coordinator.overhead_budget == expected_overhead

    # Verify configurations
    vllm_config = coordinator.get_vllm_config()
    assert "gpu_memory_utilization" in vllm_config
    assert "max_num_seqs" in vllm_config
    assert "enforce_eager" in vllm_config

    expert_config = coordinator.get_expert_cache_config()
    assert expert_config["gpu_cache_bytes"] == expected_expert
    assert expert_config["cpu_cache_bytes"] > 0


def test_vllm_memory_utilization_calculation():
    """Verify vLLM gets correct GPU utilization value."""
    total_gpu = 24 * 1024 ** 3
    coordinator = UnifiedMemoryCoordinator(total_gpu_bytes=total_gpu)

    vllm_config = coordinator.get_vllm_config()

    # vLLM should get 50% of 85% total = 58.8% of total
    expected_utilization = 0.50 / 0.85
    actual_utilization = vllm_config["gpu_memory_utilization"]

    assert abs(actual_utilization - expected_utilization) < 0.01


def test_batch_size_estimation():
    """Verify batch size scales with available KV cache."""
    # Small GPU (8GB)
    small_gpu = 8 * 1024 ** 3
    small_coordinator = UnifiedMemoryCoordinator(total_gpu_bytes=small_gpu)
    small_batch = small_coordinator.get_vllm_config()["max_num_seqs"]

    # Large GPU (80GB)
    large_gpu = 80 * 1024 ** 3
    large_coordinator = UnifiedMemoryCoordinator(total_gpu_bytes=large_gpu)
    large_batch = large_coordinator.get_vllm_config()["max_num_seqs"]

    # Larger GPU should support larger batches
    assert large_batch > small_batch

    # But capped at reasonable maximum
    assert small_batch >= 8
    assert large_batch <= 32
