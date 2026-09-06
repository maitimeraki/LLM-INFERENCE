#!/bin/bash
# Startup script for SparseLLM server in WSL with proper environment variables

# HuggingFace cache paths
export HF_HOME=/mnt/d/users/anupam/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/d/users/anupam/.cache/huggingface

# WSL compatibility
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

# Version check bypass
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Force legacy V0 engine (V1 has OOM issues with weight injection)
export VLLM_USE_V0_ENGINE=1

# Activate virtual environment
source /home/anupam/env/.venv/bin/activate

# Run server
python serve.py "$@"
