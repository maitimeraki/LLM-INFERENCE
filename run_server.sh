#!/bin/bash

# Export environment variables for HuggingFace and vLLM
export HF_HOME=/mnt/d/users/anupam/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/d/users/anupam/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

# Activate virtual environment if it exists
if [ -f "/home/anupam/env/.venv/bin/activate" ]; then
    source /home/anupam/env/.venv/bin/activate
fi

# Run the server
cd /mnt/c/users/anupam/desktop/llm/local-inference
python serve.py \
    --model Qwen/Qwen1.5-MoE-A2.7B \
    --tensor-parallel-size 1 \
    --max-model-len 2048 \
    --port 8000
