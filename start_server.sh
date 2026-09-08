#!/bin/bash
# SparseLLM Production Server Launcher
# Starts FastAPI server with vLLM integration for OpenAI-compatible endpoints
# Usage: ./start_server.sh [model] [port]

set -e

echo "===================================================================="
echo "SparseLLM Production Server"
echo "===================================================================="
echo

# Force the script to activate your local virtual environment
if [ -f "/home/anupam/env/.venv/bin/activate" ]; then
    source /home/anupam/env/.venv/bin/activate
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "[ERROR] .env file not found!"
    echo "Please create .env with your HF_TOKEN:"
    echo
    echo "HF_TOKEN=your_huggingface_token_here"
    echo
    exit 1
fi

# Load environment variables
export $(cat .env | xargs)

# Check dependencies
if ! python -c "import vllm" 2>/dev/null; then
    echo "[ERROR] vLLM not installed!"
    echo "Install with: pip install vllm"
    echo
    exit 1
fi

if ! python -c "import fastapi" 2>/dev/null; then
    echo "[ERROR] FastAPI not installed!"
    echo "Install with: pip install fastapi uvicorn"
    echo
    exit 1
fi

# Set default configuration
MODEL="${1:-Qwen/Qwen1.5-MoE-A2.7B}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8000}"
TENSOR_PARALLEL="${4:-1}"
GPU_MEMORY_UTIL="${5:-0.9}"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Host: $HOST:$PORT"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo
echo "Starting server..."
echo "===================================================================="
echo
echo "Server will be available at:"
echo "  - http://localhost:$PORT/v1/chat/completions"
echo "  - http://localhost:$PORT/v1/completions"
echo "  - http://localhost:$PORT/health"
echo "  - http://localhost:$PORT/metrics"
echo
echo "Press Ctrl+C to stop the server"
echo "===================================================================="
echo

# Run serve.py
python serve.py \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"
