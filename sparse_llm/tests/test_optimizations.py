"""
Test suite for performance optimizations in SparseLLM.

Tests validate:
1. All layers are processed (not just layer 0)
2. KV cache uses pre-allocated buffers (no copies)
3. Three-tier expert cache works correctly (GPU → CPU → Storage)
4. Performance improvements are measurable
"""

import time
import torch
import pytest
from pathlib import Path

# Test configuration
TEST_MODEL = "mistralai/Mixtral-8x7B-v0.1"  # Or your preferred test model
USE_SMALL_MODEL = True  # Set False to test with full model


class TestLayerProcessing:
    """Test that all transformer layers are processed correctly."""

    def test_all_layers_processed_in_decode(self):
        """Verify decode phase processes all num_layers, not just layer 0."""
        from sparse_llm.inference.moe_inference_engine import CustomMoEInferenceEngine, InferenceConfig

        config = InferenceConfig(
            model_path=TEST_MODEL if not USE_SMALL_MODEL else "gpt2",  # Use GPT-2 for quick test
            max_tokens=5,
            expert_cache_size=4
        )

        engine = CustomMoEInferenceEngine(TEST_MODEL if not USE_SMALL_MODEL else "gpt2", config)

        # Track which layers are accessed
        layers_accessed = set()
        original_process_single = engine.expert_processor.process_single

        def tracked_process_single(*args, **kwargs):
            layer_id = kwargs.get('layer_id', 0)
            layers_accessed.add(layer_id)
            return original_process_single(*args, **kwargs)

        engine.expert_processor.process_single = tracked_process_single

        # Generate tokens
        prompt = "Hello, world!"
        output = engine.generate(prompt, max_tokens=5)

        # Verify all layers were accessed
        expected_layers = set(range(engine.model_info.num_layers))
        assert layers_accessed == expected_layers, (
            f"Expected all {engine.model_info.num_layers} layers to be processed, "
            f"but only {len(layers_accessed)} were accessed: {sorted(layers_accessed)}"
        )

        print(f"✓ All {engine.model_info.num_layers} layers processed correctly")

    def test_all_layers_processed_in_prefill(self):
        """Verify prefill phase processes all num_layers, not just layer 0."""
        from sparse_llm.inference.moe_inference_engine import CustomMoEInferenceEngine, InferenceConfig

        config = InferenceConfig(
            model_path="gpt2",
            max_tokens=1
        )

        engine = CustomMoEInferenceEngine("gpt2", config)

        # Track which layers are accessed during prefill
        layers_accessed = set()
        original_process_batch = engine.expert_processor.process_batch

        def tracked_process_batch(*args, **kwargs):
            layer_id = kwargs.get('layer_id', 0)
            layers_accessed.add(layer_id)
            return original_process_batch(*args, **kwargs)

        engine.expert_processor.process_batch = tracked_process_batch

        # Run prefill
        prompt = "Test prompt"
        engine.generate(prompt, max_tokens=1)

        # Verify all layers were accessed
        expected_layers = set(range(engine.model_info.num_layers))
        assert layers_accessed == expected_layers, (
            f"Prefill should process all {engine.model_info.num_layers} layers, "
            f"but only processed {len(layers_accessed)}: {sorted(layers_accessed)}"
        )

        print(f"✓ Prefill processes all {engine.model_info.num_layers} layers correctly")


class TestKVCacheOptimization:
    """Test that KV cache uses pre-allocated buffers efficiently."""

    def test_kv_cache_no_reallocation(self):
        """Verify KV cache doesn't reallocate tensors on every decode step."""
        from sparse_llm.inference.attention_engine import AttentionEngine

        engine = AttentionEngine(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            max_seq_len=128,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Prefill
        batch_size = 1
        seq_len = 10
        hidden_dim = engine.hidden_dim
        input_embeddings = torch.randn(batch_size, seq_len, hidden_dim)

        _, kv_cache = engine.prefill(input_embeddings)

        # Store initial buffer IDs
        initial_keys_id = id(kv_cache['keys'])
        initial_values_id = id(kv_cache['values'])

        # Run decode steps
        for step in range(5):
            token_embedding = torch.randn(batch_size, 1, hidden_dim)
            _, kv_cache = engine.decode(token_embedding, kv_cache)

            # Verify buffer IDs haven't changed (no reallocation!)
            assert id(kv_cache['keys']) == initial_keys_id, (
                f"KV cache keys buffer was reallocated at step {step}. "
                "This defeats the pre-allocation optimization!"
            )
            assert id(kv_cache['values']) == initial_values_id, (
                f"KV cache values buffer was reallocated at step {step}. "
                "This defeats the pre-allocation optimization!"
            )

        print("✓ KV cache uses pre-allocated buffers (no reallocation)")

    def test_kv_cache_memory_efficiency(self):
        """Verify KV cache memory doesn't grow linearly with sequence length."""
        from sparse_llm.inference.attention_engine import AttentionEngine

        engine = AttentionEngine(
            num_layers=4,
            num_heads=8,
            head_dim=64,
            max_seq_len=1024,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        batch_size = 1
        seq_len = 10
        hidden_dim = engine.hidden_dim
        input_embeddings = torch.randn(batch_size, seq_len, hidden_dim)

        _, kv_cache = engine.prefill(input_embeddings)

        # Memory footprint after prefill
        keys_bytes = kv_cache['keys'].element_size() * kv_cache['keys'].nelement()
        values_bytes = kv_cache['values'].element_size() * kv_cache['values'].nelement()
        initial_memory = keys_bytes + values_bytes

        # Run 100 decode steps
        for step in range(100):
            token_embedding = torch.randn(batch_size, 1, hidden_dim)
            _, kv_cache = engine.decode(token_embedding, kv_cache)

        # Memory footprint after 100 decodes
        keys_bytes = kv_cache['keys'].element_size() * kv_cache['keys'].nelement()
        values_bytes = kv_cache['values'].element_size() * kv_cache['values'].nelement()
        final_memory = keys_bytes + values_bytes

        # Memory should be IDENTICAL (pre-allocated buffer)
        assert initial_memory == final_memory, (
            f"KV cache memory grew from {initial_memory} to {final_memory} bytes. "
            "Pre-allocation optimization is not working!"
        )

        print(f"✓ KV cache memory stable at {initial_memory / 1024**2:.2f} MB")


class TestThreeTierCache:
    """Test three-tier expert cache (GPU → CPU → Storage)."""

    def test_gpu_cache_hit_fast(self):
        """Verify GPU cache hits are faster than storage loads."""
        from sparse_llm.loading.expert_cache import ExpertCache

        cache = ExpertCache(
            gpu_slots=2,
            cpu_slots=2,
            expert_bytes=1024 * 1024,  # 1MB per expert
            storage_loader=lambda layer_id, expert_id: {
                'w1.weight': torch.randn(512, 1024),
                'w2.weight': torch.randn(1024, 512),
                'w3.weight': torch.randn(512, 1024)
            }
        )

        # First access: storage load (slow)
        start = time.monotonic()
        weights1, tier1 = cache.get(0, 0)
        storage_time = time.monotonic() - start

        assert tier1 == "storage", f"First access should be storage, got {tier1}"

        # Second access: GPU cache hit (fast)
        start = time.monotonic()
        weights2, tier2 = cache.get(0, 0)
        gpu_time = time.monotonic() - start

        assert tier2 == "gpu", f"Second access should be GPU cache hit, got {tier2}"

        # GPU cache hit should be at least 10x faster
        speedup = storage_time / (gpu_time + 1e-6)
        assert speedup > 10, (
            f"GPU cache hit should be much faster than storage. "
            f"Got speedup: {speedup:.1f}x (expected >10x)"
        )

        print(f"✓ GPU cache hit is {speedup:.1f}x faster than storage load")

    def test_three_tier_promotion(self):
        """Verify experts promote through tiers: Storage → CPU → GPU."""
        from sparse_llm.loading.expert_cache import ExpertCache

        cache = ExpertCache(
            gpu_slots=2,
            cpu_slots=2,
            expert_bytes=1024 * 1024,
            storage_loader=lambda layer_id, expert_id: {
                'w1.weight': torch.randn(512, 1024),
                'w2.weight': torch.randn(1024, 512),
                'w3.weight': torch.randn(512, 1024)
            }
        )

        # Load 3 experts
        _, tier1 = cache.get(0, 0)  # Expert 0 → GPU
        _, tier2 = cache.get(0, 1)  # Expert 1 → GPU
        _, tier3 = cache.get(0, 2)  # Expert 2 → GPU (evicts 0 to CPU)

        assert tier1 == "storage", "First load should be from storage"
        assert tier2 == "storage", "Second load should be from storage"
        assert tier3 == "storage", "Third load should be from storage"

        # Now access expert 0 again - should be in CPU, promote to GPU
        _, tier4 = cache.get(0, 0)
        assert tier4 == "cpu", "Expert 0 should be promoted from CPU to GPU"

        # Verify stats
        stats = cache.get_stats()
        assert stats['gpu_hits'] > 0, "Should have GPU cache hits"
        assert stats['cpu_hits'] > 0, "Should have CPU cache hits"
        assert stats['promotions'] > 0, "Should have promotions from CPU to GPU"

        print(f"✓ Three-tier cache working: {stats}")


class TestPerformanceBenchmark:
    """Benchmark performance improvements."""

    def test_token_generation_speed(self):
        """Measure tokens/sec and verify it's reasonable."""
        if not torch.cuda.is_available():
            pytest.skip("GPU required for performance test")

        from sparse_llm.inference.moe_inference_engine import CustomMoEInferenceEngine, InferenceConfig

        config = InferenceConfig(
            model_path="gpt2",
            max_tokens=50,
            expert_cache_size=8
        )

        engine = CustomMoEInferenceEngine("gpt2", config)

        prompt = "Once upon a time"

        # Warmup
        _ = engine.generate(prompt, max_tokens=5)

        # Benchmark
        start = time.monotonic()
        output = engine.generate(prompt, max_tokens=50)
        elapsed = time.monotonic() - start

        stats = engine.get_statistics()
        tokens_per_sec = stats['average_tokens_per_second']

        print(f"\n{'='*60}")
        print(f"Performance Benchmark Results:")
        print(f"  Generated: {stats['total_tokens_generated']} tokens")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Speed: {tokens_per_sec:.2f} tokens/sec")
        print(f"  Cache hit rate: {stats.get('cache_hit_rate', 0):.1%}")
        print(f"  Memory: {stats['memory_usage_gb']:.2f} GB")
        print(f"{'='*60}\n")

        # Assert minimum performance (adjust based on hardware)
        # For GPT-2 on modern GPU, we should get at least 5 tok/s
        # For MoE models with optimizations, target is 10+ tok/s
        min_speed = 5.0 if config.model_path == "gpt2" else 3.0

        assert tokens_per_sec >= min_speed, (
            f"Token generation speed {tokens_per_sec:.2f} tok/s is below "
            f"minimum {min_speed} tok/s. Check for performance regressions."
        )

        print(f"✓ Token generation speed: {tokens_per_sec:.2f} tok/s (target: 10+ tok/s)")


def run_validation_suite():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("SparseLLM Optimization Validation Suite")
    print("="*60 + "\n")

    test_classes = [
        TestLayerProcessing,
        TestKVCacheOptimization,
        TestThreeTierCache,
        TestPerformanceBenchmark
    ]

    results = []
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 40)

        test_instance = test_class()
        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                try:
                    method = getattr(test_instance, method_name)
                    print(f"\n  Running {method_name}...")
                    method()
                    results.append((test_class.__name__, method_name, "PASS"))
                    print(f"  ✓ {method_name} PASSED")
                except Exception as e:
                    results.append((test_class.__name__, method_name, f"FAIL: {e}"))
                    print(f"  ✗ {method_name} FAILED: {e}")

    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)
    passed = sum(1 for _, _, status in results if status == "PASS")
    failed = len(results) - passed

    for test_class, method, status in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {test_class}.{method}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)} tests")

    if failed > 0:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return False
    else:
        print("\n✅ All optimizations validated successfully!")
        return True


if __name__ == "__main__":
    success = run_validation_suite()
    exit(0 if success else 1)
