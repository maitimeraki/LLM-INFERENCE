#!/usr/bin/env python3
"""Integration test for SparseLLM native and vLLM modes."""

import json
import subprocess
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_native_inference():
    """Test native SparseLLM inference via main.py."""
    print("\n[TEST] Testing Native Inference")
    print("=" * 60)

    cmd = [
        "python", "main.py",
        "--model", "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "--prompt", "Say 'Hello from SparseLLM' and nothing else.",
        "--max-new-tokens", "20",
        "--cache-bytes", "1000000000",
        "--json"
    ]

    print(f"Command: {' '.join(cmd)}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=True
        )

        elapsed = time.time() - start
        output = json.loads(result.stdout)

        print(f"\n[PASS] SUCCESS (took {elapsed:.1f}s)")
        print(f"Generated: {output['text'][:100]}...")
        print(f"Speed: {output['metrics']['tokens_per_second']:.1f} tok/s")

        if output.get('paging_diagnostics', {}).get('enabled'):
            diag = output['paging_diagnostics']
            print(f"Cache hits: {diag['cache_hits']}")
            print(f"Cache misses: {diag['cache_misses']}")
            hit_rate = diag['cache_hits'] / (diag['cache_hits'] + diag['cache_misses']) * 100
            print(f"Hit rate: {hit_rate:.1f}%")

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] FAILED: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"\n[FAIL] TIMEOUT after 5 minutes")
        return False
    except json.JSONDecodeError as e:
        print(f"\n[FAIL] JSON Parse Error: {e}")
        print(f"Raw output: {result.stdout}")
        return False


def test_vllm_availability():
    """Check if vLLM integration is available."""
    print("\n[TEST] Testing vLLM Availability")
    print("=" * 60)

    try:
        from sparse_llm.integrations import VLLM_AVAILABLE

        if VLLM_AVAILABLE:
            print("[PASS] vLLM integration available")

            try:
                import vllm
                print(f"[INFO] vLLM version: {vllm.__version__}")
                return True
            except Exception as e:
                print(f"[WARN] vLLM installed but not importable: {e}")
                return False
        else:
            print("[WARN] vLLM not available (Linux/Docker required)")
            print("       Native inference will work fine on Windows")
            return False

    except Exception as e:
        print(f"[FAIL] Integration check failed: {e}")
        return False


def test_model_registry():
    """Test model registry and adapter loading."""
    print("\n[TEST] Testing Model Registry")
    print("=" * 60)

    try:
        from sparse_llm.models.registry import get_model_adapter

        # Test Mixtral adapter
        adapter_cls = get_model_adapter("mistralai/Mixtral-8x7B-Instruct-v0.1")
        print(f"[PASS] Mixtral adapter: {adapter_cls.__name__}")

        # Test Qwen adapter
        adapter_cls = get_model_adapter("Qwen/Qwen1.5-MoE-A2.7B")
        print(f"[PASS] Qwen adapter: {adapter_cls.__name__}")

        # Test generic adapter
        adapter_cls = get_model_adapter("gpt2")
        print(f"[PASS] Generic adapter: {adapter_cls.__name__}")

        return True

    except Exception as e:
        print(f"[FAIL] Registry test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("SparseLLM Integration Test Suite")
    print("=" * 60)

    results = {
        "Model Registry": test_model_registry(),
        "vLLM Availability": test_vllm_availability(),
        "Native Inference": test_native_inference(),
    }

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status:10} {test_name}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)

    if all_passed:
        print("[SUCCESS] All tests passed!")
        return 0
    else:
        print("[WARNING] Some tests failed (see above)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
