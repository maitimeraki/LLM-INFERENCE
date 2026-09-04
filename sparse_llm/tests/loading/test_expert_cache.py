"""Tests for ExpertCache three-tier LRU cache."""

import torch
import pytest


def test_expert_cache_gpu_hit():
    from sparse_llm.loading.expert_cache import ExpertCache

    # Create cache with 3 GPU slots, 2 CPU slots
    cache = ExpertCache(gpu_slots=3, cpu_slots=2, expert_bytes=100*1024**2)

    # Preload expert (0, 0) on GPU
    expert_weights = {
        "w1": torch.randn(1024, 4096),
        "w2": torch.randn(4096, 1024),
        "w3": torch.randn(1024, 4096)
    }
    cache.preload_gpu(layer_id=0, expert_id=0, weights=expert_weights)

    # Get expert - should be GPU hit
    retrieved, tier = cache.get(layer_id=0, expert_id=0)

    assert tier == "gpu"
    assert "w1" in retrieved
    assert cache.stats["gpu_hits"] == 1
    assert cache.stats["cpu_hits"] == 0
    assert cache.stats["storage_hits"] == 0


def test_expert_cache_lru_eviction():
    import torch
    from sparse_llm.loading.expert_cache import ExpertCache

    # Create cache with only 2 GPU slots
    cache = ExpertCache(gpu_slots=2, cpu_slots=5, expert_bytes=100*1024**2)

    expert_a = {"w1": torch.randn(1024, 4096)}
    expert_b = {"w1": torch.randn(1024, 4096)}
    expert_c = {"w1": torch.randn(1024, 4096)}

    # Fill GPU cache
    cache.preload_gpu(0, 0, expert_a)
    cache.preload_gpu(0, 1, expert_b)

    # Access expert_a (makes it most recently used)
    cache.get(0, 0)

    # Add third expert - should evict expert_b (LRU)
    cache.preload_gpu(0, 2, expert_c)

    # expert_b should be evicted to CPU
    retrieved, tier = cache.get(0, 1)
    assert tier == "cpu"  # Was evicted from GPU, now on CPU
