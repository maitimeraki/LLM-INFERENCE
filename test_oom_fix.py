#!/usr/bin/env python3
"""Test script to verify OOM fix works correctly."""

import sys
import torch

def test_memory_allocation():
    """Test that memory allocation calculation works correctly."""
    print("="*80)
    print("Testing Memory Allocation Fix")
    print("="*80)

    # Test 1: Import modules
    print("\n[1/5] Testing imports...")
    try:
        from sparse_llm.loading.model_introspector import ModelIntrospector, ModelInfo
        from sparse_llm.loading.memory_budget_calculator import DynamicMemoryBudgetCalculator, UserRequest
        from sparse_llm.loading.resource_profiler import ResourceProfiler
        print("✅ All imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test 2: Check ModelInfo has new fields
    print("\n[2/5] Testing ModelInfo fields...")
    try:
        # Check if ModelInfo has the new fields
        fields = ModelInfo.__dataclass_fields__.keys()
        required = {'hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads'}
        if required.issubset(fields):
            print(f"✅ ModelInfo has all required fields: {required}")
        else:
            missing = required - set(fields)
            print(f"❌ ModelInfo missing fields: {missing}")
            return False
    except Exception as e:
        print(f"❌ Field check failed: {e}")
        return False

    # Test 3: Profile resources
    print("\n[3/5] Testing resource profiling...")
    try:
        profiler = ResourceProfiler()
        budget = profiler.profile()
        print(f"✅ Resource profiler works")
        print(f"   GPU: {budget.total_gpu_bytes / 1024**3:.2f}GB")
        print(f"   CPU: {budget.total_cpu_bytes / 1024**3:.2f}GB")
    except Exception as e:
        print(f"❌ Resource profiling failed: {e}")
        return False

    # Test 4: Introspect a model
    print("\n[4/5] Testing model introspection...")
    try:
        introspector = ModelIntrospector()
        # Use a small model for testing
        model_info = introspector.introspect("Qwen/Qwen1.5-MoE-A2.7B")
        print(f"✅ Model introspection works")
        print(f"   Model: {model_info.model_id}")
        print(f"   Is MoE: {model_info.is_moe}")
        print(f"   Layers: {model_info.num_layers}")
        print(f"   Hidden size: {model_info.hidden_size}")
        print(f"   Intermediate size: {model_info.intermediate_size}")
        print(f"   Attention heads: {model_info.num_attention_heads}")
        print(f"   KV heads: {model_info.num_key_value_heads}")
    except Exception as e:
        print(f"❌ Model introspection failed: {e}")
        return False

    # Test 5: Calculate memory allocation
    print("\n[5/5] Testing memory allocation calculation...")
    try:
        calculator = DynamicMemoryBudgetCalculator()
        user_request = UserRequest(
            max_model_len=2048,
            dtype="float16",
            quantization=None,
            tensor_parallel_size=1,
            allow_cpu_offload=True,
            allow_ssd_offload=True
        )

        allocation = calculator.calculate(budget, model_info, user_request)

        if allocation.can_fulfill:
            print(f"✅ Memory allocation calculation works")
            print(f"   GPU shared weights: {allocation.gpu_shared_weights / 1024**3:.2f}GB")
            print(f"   GPU KV cache: {allocation.gpu_kv_cache / 1024**3:.2f}GB")
            print(f"   GPU activation buffer: {allocation.gpu_activation_buffer / 1024**3:.2f}GB")
            print(f"   GPU hot experts: {allocation.gpu_hot_expert_count}")
            print(f"   vLLM gpu_memory_utilization: {allocation.vllm_gpu_memory_utilization:.4f}")
            print(f"   Total GPU required: {allocation.total_gpu_required / 1024**3:.2f}GB")
            print(f"   Total GPU available: {allocation.total_gpu_available / 1024**3:.2f}GB")
        else:
            print(f"❌ Cannot fulfill allocation: {allocation.rejection_reason}")
            return False

    except Exception as e:
        print(f"❌ Memory allocation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80)
    return True

if __name__ == "__main__":
    success = test_memory_allocation()
    sys.exit(0 if success else 1)
