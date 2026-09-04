#!/bin/bash
# Unified WSL launcher for SparseLLM with zero-latency VLLM integration
#
# Usage:
#   ./run_wsl.sh inference --model <model> --prompt "Hello"
#   ./run_wsl.sh server --model <model> --port 8000
#
# This script ensures:
# - All execution in WSL (zero cross-boundary latency)
# - Direct VLLM integration (no HTTP overhead)
# - Proper Python environment setup

set -e

PROJECT_ROOT="/mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE"
cd "$PROJECT_ROOT"

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found in WSL"
    exit 1
fi

# Check VLLM installation
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "⚠️  VLLM not installed in WSL. Installing..."
    pip3 install vllm torch transformers --quiet
fi

# Ensure SparseLLM dependencies
if ! python3 -c "import sparse_llm" 2>/dev/null; then
    echo "📦 Installing SparseLLM dependencies..."
    pip3 install -e . --quiet
fi

MODE=$1
shift

case "$MODE" in
    inference)
        echo "🚀 Running inference (zero-latency mode)"
        python3 main.py "$@"
        ;;

    server)
        echo "🌐 Starting VLLM server"
        python3 vllm_server.py "$@"
        ;;

    benchmark)
        echo "⚡ Running latency benchmark"
        python3 scripts/benchmark_latency.py "$@"
        ;;

    *)
        echo "Usage: $0 {inference|server|benchmark} [options]"
        echo ""
        echo "Examples:"
        echo "  $0 inference --model Qwen/Qwen1.5-MoE-A2.7B --prompt 'Hello'"
        echo "  $0 server --model Qwen/Qwen1.5-MoE-A2.7B --port 8000"
        echo "  $0 benchmark --compare-wsl-vs-native"
        exit 1
        ;;
esac
