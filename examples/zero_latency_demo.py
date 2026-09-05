#!/usr/bin/env python3
"""Zero-latency inference demo.

This example demonstrates the DirectVLLMEngine for sub-millisecond
overhead inference.

Run in WSL for zero latency:
    wsl
    cd /mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE
    python3 examples/zero_latency_demo.py
"""

import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sparse_llm.integrations.vllm_direct import DirectVLLMEngine


def demo_single_generation():
    """Demo: Single text generation."""
    print("\n" + "="*60)
    print("Demo 1: Single Generation")
    print("="*60)

    # Initialize engine
    engine = DirectVLLMEngine(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        gpu_cache_gb=2.0,
        cpu_cache_gb=14.0,
    )

    # Generate
    prompt = "Explain the concept of sparse neural networks in 3 sentences."

    start = time.perf_counter()
    response = engine.generate(prompt, max_tokens=100, temperature=0.7)
    latency_ms = (time.perf_counter() - start) * 1000

    print(f"\n📝 Prompt: {prompt}")
    print(f"\n🤖 Response: {response}")
    print(f"\n⚡ Latency: {latency_ms:.1f}ms")

    return engine


def demo_batch_generation(engine: DirectVLLMEngine):
    """Demo: Batch generation (parallel)."""
    print("\n" + "="*60)
    print("Demo 2: Batch Generation")
    print("="*60)

    prompts = [
        "What is machine learning?",
        "What is deep learning?",
        "What is reinforcement learning?",
    ]

    start = time.perf_counter()
    responses = engine.generate(prompts, max_tokens=50, temperature=0.7)
    latency_ms = (time.perf_counter() - start) * 1000

    print(f"\n📊 Generated {len(responses)} responses in {latency_ms:.1f}ms")
    print(f"   Average per response: {latency_ms / len(responses):.1f}ms")

    for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
        print(f"\n{i}. {prompt}")
        print(f"   → {response[:100]}...")


def demo_chat_completion(engine: DirectVLLMEngine):
    """Demo: Chat completion."""
    print("\n" + "="*60)
    print("Demo 3: Chat Completion")
    print("="*60)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What's the difference between MoE and dense models?"},
    ]

    start = time.perf_counter()
    response = engine.chat(messages, max_tokens=150, temperature=0.7)
    latency_ms = (time.perf_counter() - start) * 1000

    print(f"\n💬 Chat:")
    for msg in messages:
        print(f"   {msg['role'].title()}: {msg['content']}")

    print(f"\n🤖 Assistant: {response}")
    print(f"\n⚡ Latency: {latency_ms:.1f}ms")


def demo_statistics(engine: DirectVLLMEngine):
    """Demo: Engine statistics."""
    print("\n" + "="*60)
    print("Demo 4: Statistics")
    print("="*60)

    stats = engine.get_stats()

    print(f"\n📊 Engine Stats:")
    print(f"   Model: {stats['model']}")
    print(f"   Mode: {stats['mode']}")

    print(f"\n🧠 Paging Stats:")
    paging = stats['paging']
    print(f"   GPU cached experts: {paging['gpu_cached_experts']}")
    print(f"   CPU cached experts: {paging['cpu_cached_experts']}")
    print(f"   Total loads: {paging['total_loads']}")
    print(f"   Cache hit rate: {paging['cache_hit_rate']:.1%}")
    print(f"   Avg load time: {paging['avg_load_time_ms']:.2f}ms")


def main():
    print("="*60)
    print("SparseLLM Zero-Latency Inference Demo")
    print("="*60)
    print("\n⚡ Running in direct mode (zero overhead)")
    print("📍 Execution: WSL (native Linux)")
    print("💾 Files: C: drive (Windows)")
    print("🎯 Target: Sub-millisecond inference calls")

    try:
        # Run demos
        engine = demo_single_generation()
        demo_batch_generation(engine)
        demo_chat_completion(engine)
        demo_statistics(engine)

        print("\n" + "="*60)
        print("✅ All demos completed successfully!")
        print("="*60)

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\n💡 Make sure VLLM is installed:")
        print("   wsl")
        print("   pip3 install vllm torch transformers")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
