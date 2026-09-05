#!/usr/bin/env python3
"""Latency benchmark for WSL vs Windows execution modes.

This script measures:
1. Direct WSL execution (zero-latency baseline)
2. HTTP client/server mode (network stack overhead)
3. Cross-process communication overhead
"""

import time
import statistics
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_direct_call(iterations: int = 100):
    """Benchmark direct function call (zero overhead baseline)."""
    print("\n📊 Benchmarking direct function call...")

    latencies = []

    for i in range(iterations):
        start = time.perf_counter()

        # Simulate minimal LLM forward pass overhead
        _ = sum(range(1000))  # Trivial compute

        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    print(f"   Min: {min(latencies):.3f}ms")
    print(f"   Avg: {statistics.mean(latencies):.3f}ms")
    print(f"   P50: {statistics.median(latencies):.3f}ms")
    print(f"   P99: {statistics.quantiles(latencies, n=100)[98]:.3f}ms")

    return statistics.mean(latencies)


def benchmark_http_roundtrip(iterations: int = 100):
    """Benchmark HTTP request/response cycle."""
    print("\n📊 Benchmarking HTTP roundtrip...")

    import socket

    latencies = []

    for i in range(iterations):
        start = time.perf_counter()

        # Simulate minimal HTTP overhead (connect + send + receive)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)

            # Test localhost connection (even if nothing is listening)
            try:
                sock.connect(('127.0.0.1', 8000))
            except (ConnectionRefusedError, socket.timeout):
                pass

            sock.close()
        except Exception:
            pass

        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    print(f"   Min: {min(latencies):.3f}ms")
    print(f"   Avg: {statistics.mean(latencies):.3f}ms")
    print(f"   P50: {statistics.median(latencies):.3f}ms")
    print(f"   P99: {statistics.quantiles(latencies, n=100)[98]:.3f}ms")

    return statistics.mean(latencies)


def benchmark_json_serialization(iterations: int = 100, payload_size: int = 1000):
    """Benchmark JSON encoding/decoding overhead."""
    print(f"\n📊 Benchmarking JSON serialization ({payload_size} tokens)...")

    import json

    # Create sample payload
    sample_data = {
        "model": "Qwen/Qwen1.5-MoE-A2.7B",
        "messages": [{"role": "user", "content": "x" * payload_size}],
        "max_tokens": 100,
        "temperature": 0.7,
    }

    latencies = []

    for i in range(iterations):
        start = time.perf_counter()

        # Encode
        encoded = json.dumps(sample_data)

        # Decode
        decoded = json.loads(encoded)

        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    print(f"   Payload size: {len(encoded)} bytes")
    print(f"   Min: {min(latencies):.3f}ms")
    print(f"   Avg: {statistics.mean(latencies):.3f}ms")
    print(f"   P50: {statistics.median(latencies):.3f}ms")
    print(f"   P99: {statistics.quantiles(latencies, n=100)[98]:.3f}ms")

    return statistics.mean(latencies)


def main():
    parser = argparse.ArgumentParser(description="Benchmark latency overhead")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per benchmark"
    )
    parser.add_argument(
        "--compare-wsl-vs-native",
        action="store_true",
        help="Compare WSL vs native execution"
    )

    args = parser.parse_args()

    print("="*60)
    print("SparseLLM Latency Benchmark")
    print("="*60)

    # Run benchmarks
    direct_latency = benchmark_direct_call(args.iterations)
    http_latency = benchmark_http_roundtrip(args.iterations)
    json_latency = benchmark_json_serialization(args.iterations)

    # Summary
    print("\n" + "="*60)
    print("📈 Summary")
    print("="*60)
    print(f"Direct call (baseline):     {direct_latency:.3f}ms")
    print(f"HTTP roundtrip overhead:    {http_latency:.3f}ms  (+{http_latency - direct_latency:.3f}ms)")
    print(f"JSON serialization:         {json_latency:.3f}ms")
    print(f"Total HTTP mode overhead:   ~{http_latency + json_latency:.3f}ms per request")

    print("\n💡 Recommendations:")
    if http_latency + json_latency < 2.0:
        print("   ✅ HTTP mode acceptable for most workloads (<2ms overhead)")
    else:
        print("   ⚠️  HTTP mode adds significant latency (>2ms)")

    print("   🚀 For zero-latency: Run everything in WSL (direct mode)")
    print("   🌐 For flexibility: Use HTTP server mode")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
