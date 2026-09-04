#!/bin/bash
# Start SparseLLM vLLM Server with Expert Paging
# Usage: ./scripts/start_server.sh [model_name]

set -e

MODEL="${1:-mistralai/Mixtral-8x7B-Instruct-v0.1}"
GPU_CACHE_GB="${GPU_CACHE_GB:-2}"
CPU_CACHE_GB="${CPU_CACHE_GB:-14}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "🚀 Starting SparseLLM vLLM Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Model:           $MODEL"
echo "GPU Cache:       ${GPU_CACHE_GB}GB"
echo "CPU Cache:       ${CPU_CACHE_GB}GB"
echo "Port:            $PORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python vllm_server.py \
  --model "$MODEL" \
  --gpu-cache-gb "$GPU_CACHE_GB" \
  --cpu-cache-gb "$CPU_CACHE_GB" \
  --port "$PORT" \
  --host "$HOST"
