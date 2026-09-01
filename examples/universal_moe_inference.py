#!/usr/bin/env python3
"""Example: Universal MoE inference for any open-source MoE model.

This example demonstrates how the universal adapter automatically handles
different MoE architectures without model-specific code.
"""

from sparse_llm import InferenceEngine, DevicePolicy, UniversalMoEAdapter

# Example 1: Mixtral model with automatic paging
def example_mixtral():
    print("=" * 60)
    print("Example 1: Mixtral-8x7B with Expert Paging")
    print("=" * 60)

    engine = InferenceEngine(
        model="mistralai/Mixtral-8x7B-v0.1",
        device="cuda",
        dtype="bfloat16",
        expert_cache_bytes=2 * 1024 * 1024 * 1024,  # 2GB cache
    )

    result = engine.generate(
        prompt="Explain how mixture of experts works in transformers:",
        max_new_tokens=100,
        temperature=0.7,
    )

    print(f"Generated: {result.text}")
    print(f"\nMetrics:")
    print(f"  Prompt tokens: {result.metrics.prompt_tokens}")
    print(f"  Generated tokens: {result.metrics.generated_tokens}")
    print(f"  Prefill latency: {result.metrics.prefill_latency_ms:.2f}ms")
    print(f"  Decode latency: {result.metrics.decode_latency_ms:.2f}ms")
    print(f"  Throughput: {result.metrics.throughput_tokens_per_sec:.2f} tok/s")
    print(f"  Cache hits: {result.metrics.cache_hits}")
    print(f"  Cache misses: {result.metrics.cache_misses}")
    print(f"  Expert load time: {result.metrics.expert_load_time_ms:.2f}ms")
    print()


# Example 2: Qwen2-MoE model
def example_qwen2_moe():
    print("=" * 60)
    print("Example 2: Qwen2-57B-A14B-Instruct with Expert Paging")
    print("=" * 60)

    engine = InferenceEngine(
        model="Qwen/Qwen2-57B-A14B-Instruct",
        device="cuda",
        dtype="bfloat16",
        expert_cache_bytes=3 * 1024 * 1024 * 1024,  # 3GB cache
    )

    result = engine.generate(
        prompt="What are the benefits of sparse mixture of experts?",
        max_new_tokens=80,
        temperature=0.0,  # Greedy
    )

    print(f"Generated: {result.text}")
    print(f"Cache efficiency: {result.metrics.cache_hits}/{result.metrics.cache_hits + result.metrics.cache_misses}")
    print()


# Example 3: DeepSeek-V2
def example_deepseek_v2():
    print("=" * 60)
    print("Example 3: DeepSeek-V2 with Expert Paging")
    print("=" * 60)

    engine = InferenceEngine(
        model="deepseek-ai/DeepSeek-V2",
        device="cuda",
        dtype="bfloat16",
        expert_cache_bytes=4 * 1024 * 1024 * 1024,  # 4GB cache
    )

    result = engine.generate(
        prompt="Explain the architecture of large language models:",
        max_new_tokens=120,
        temperature=0.5,
    )

    print(f"Generated: {result.text}")
    print(f"Peak VRAM: {result.metrics.peak_vram_bytes / (1024**3):.2f} GB")
    print()


# Example 4: Using UniversalMoEAdapter directly
def example_direct_adapter():
    print("=" * 60)
    print("Example 4: Direct UniversalMoEAdapter Usage")
    print("=" * 60)

    policy = DevicePolicy(
        device="cuda",
        dtype="bfloat16",
        expert_cache_bytes=2 * 1024 * 1024 * 1024,
    )

    adapter = UniversalMoEAdapter(
        model_id="mistralai/Mixtral-8x7B-v0.1",
        policy=policy,
        checkpoint_path="./models/mixtral-8x7b",  # Local checkpoint
        cache_size_bytes=2 * 1024 * 1024 * 1024,
    )

    # Check paging capabilities
    validation = adapter.validate_paging()
    print(f"Paging eligible: {validation.eligible}")
    if validation.failure_reasons:
        print(f"Reasons: {validation.failure_reasons}")

    # Load and generate
    adapter.load()
    result = adapter.generate(
        "Write a Python function to calculate fibonacci:",
        max_new_tokens=100,
    )

    print(f"Generated: {result.text}")

    # Check capabilities
    caps = adapter.capabilities
    print(f"\nModel Capabilities:")
    print(f"  Model type: {caps.model_type}")
    print(f"  Architecture: {caps.architecture}")
    print(f"  Is MoE: {caps.is_moe}")
    print(f"  Layers: {caps.num_hidden_layers}")
    print(f"  Experts: {caps.num_experts}")
    print(f"  Top-K: {caps.top_k_experts}")
    print(f"  Expert paging: {caps.expert_paging}")

    if adapter.paging_capabilities:
        paging = adapter.paging_capabilities
        print(f"\nPaging Capabilities:")
        print(f"  Architecture ID: {paging.architecture_id}")
        print(f"  Adapter version: {paging.adapter_version}")
        print(f"  Validation passed: {paging.validation_passed}")
        print(f"  Supports prefill: {paging.supports_prefill}")
        print(f"  Supports decode: {paging.supports_decode}")
    print()


# Example 5: Custom model with generic mapper
def example_custom_moe():
    print("=" * 60)
    print("Example 5: Custom/Unknown MoE Architecture")
    print("=" * 60)

    # The universal adapter will try to detect expert patterns automatically
    engine = InferenceEngine(
        model="path/to/custom-moe-model",
        device="cuda",
        dtype="float16",
        expert_cache_bytes=2 * 1024 * 1024 * 1024,
    )

    # Check if paging was detected
    caps = engine.capabilities
    print(f"Model type: {caps.model_type}")
    print(f"Is MoE: {caps.is_moe}")
    print(f"Expert paging enabled: {caps.expert_paging}")

    # Will use generic inference if paging fails
    result = engine.generate(
        "Hello, how are you?",
        max_new_tokens=50,
    )
    print(f"Generated: {result.text}")
    print()


# Example 6: Comparing paged vs non-paged inference
def example_comparison():
    print("=" * 60)
    print("Example 6: Paged vs Non-Paged Inference Comparison")
    print("=" * 60)

    model_id = "mistralai/Mixtral-8x7B-v0.1"
    prompt = "Explain quantum computing in simple terms:"

    # Non-paged inference (standard)
    print("Running WITHOUT expert paging...")
    engine_standard = InferenceEngine(
        model=model_id,
        device="cuda",
        dtype="bfloat16",
    )
    result_standard = engine_standard.generate(prompt, max_new_tokens=100)

    print(f"Standard inference:")
    print(f"  Decode latency: {result_standard.metrics.decode_latency_ms:.2f}ms")
    print(f"  Peak VRAM: {result_standard.metrics.peak_vram_bytes / (1024**3):.2f} GB")

    # Paged inference
    print("\nRunning WITH expert paging...")
    engine_paged = InferenceEngine(
        model=model_id,
        device="cuda",
        dtype="bfloat16",
        expert_cache_bytes=2 * 1024 * 1024 * 1024,
    )
    result_paged = engine_paged.generate(prompt, max_new_tokens=100)

    print(f"Paged inference:")
    print(f"  Decode latency: {result_paged.metrics.decode_latency_ms:.2f}ms")
    print(f"  Peak VRAM: {result_paged.metrics.peak_vram_bytes / (1024**3):.2f} GB")
    print(f"  Cache hits: {result_paged.metrics.cache_hits}")
    print(f"  Cache misses: {result_paged.metrics.cache_misses}")
    print(f"  Expert load time: {result_paged.metrics.expert_load_time_ms:.2f}ms")

    # Calculate savings
    vram_savings = result_standard.metrics.peak_vram_bytes - result_paged.metrics.peak_vram_bytes
    print(f"\nVRAM savings: {vram_savings / (1024**3):.2f} GB")
    print()


if __name__ == "__main__":
    # Run examples
    # NOTE: These require the respective models to be available locally or for download

    print("\n" + "=" * 60)
    print("Universal MoE Inference Examples")
    print("=" * 60 + "\n")

    # Uncomment to run specific examples:
    # example_mixtral()
    # example_qwen2_moe()
    # example_deepseek_v2()
    # example_direct_adapter()
    # example_custom_moe()
    # example_comparison()

    print("Examples complete. Uncomment specific examples to run them.")
