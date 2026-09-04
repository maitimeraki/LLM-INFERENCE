"""vLLM Server with SparseLLM Expert Paging

This script launches vLLM's OpenAI-compatible API server with SparseLLM's
expert paging system integrated.

Usage:
    python vllm_server.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --gpu-cache-gb 2

The server exposes:
    - POST /v1/chat/completions (OpenAI-compatible)
    - POST /v1/completions (OpenAI-compatible)
    - GET /health
    - GET /metrics
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.openai.api_server import run_server as vllm_run_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("ERROR: vLLM not installed. Install with: pip install vllm")
    sys.exit(1)

from sparse_llm.integrations.vllm_plugin import (
    VLLMSparseExpertPager,
    replace_moe_layers_with_paged,
)
from sparse_llm.models.shared_weight_loader import SharedExpertWeightLoader


class SparseLLMServer:
    """vLLM server with SparseLLM expert paging."""

    def __init__(
        self,
        model: str,
        gpu_cache_gb: float = 2.0,
        cpu_cache_gb: float = 14.0,
        host: str = "0.0.0.0",
        port: int = 8000,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model = model
        self.gpu_cache_gb = gpu_cache_gb
        self.cpu_cache_gb = cpu_cache_gb
        self.host = host
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

        self.expert_pager = None
        self.engine = None

    def initialize_expert_paging(self, model_config) -> VLLMSparseExpertPager:
        """Initialize expert paging system."""

        num_layers = getattr(model_config, 'num_hidden_layers', 32)
        num_experts = getattr(model_config, 'num_local_experts', 8)

        # Create weight loader for experts
        weight_loader = SharedExpertWeightLoader(
            model_name=self.model,
            num_layers=num_layers,
            num_experts=num_experts,
        )

        # Create expert pager
        expert_pager = VLLMSparseExpertPager(
            gpu_cache_bytes=int(self.gpu_cache_gb * 1024**3),
            cpu_cache_bytes=int(self.cpu_cache_gb * 1024**3),
            num_layers=num_layers,
            num_experts=num_experts,
            expert_weight_loader=weight_loader,
        )

        print(f"✅ Expert paging initialized:")
        print(f"   GPU cache: {self.gpu_cache_gb:.1f}GB")
        print(f"   CPU cache: {self.cpu_cache_gb:.1f}GB")
        print(f"   Layers: {num_layers}, Experts: {num_experts}")

        return expert_pager

    def create_vllm_engine_args(self) -> AsyncEngineArgs:
        """Create vLLM engine arguments."""

        args = AsyncEngineArgs(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=None,  # Auto-detect
            enforce_eager=False,  # Use CUDA graphs when possible
        )

        return args

    def start(self):
        """Start vLLM server with expert paging."""

        print("="*60)
        print("SparseLLM + vLLM Server")
        print("="*60)
        print(f"Model: {self.model}")
        print(f"Host: {self.host}:{self.port}")
        print(f"GPU cache: {self.gpu_cache_gb}GB (for active experts)")
        print(f"CPU cache: {self.cpu_cache_gb}GB (for warm experts)")
        print("="*60)

        # Create engine args
        engine_args = self.create_vllm_engine_args()

        # Create async engine
        print("\n🔄 Initializing vLLM engine...")
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Get model config
        model_config = engine.engine.model_config

        # Initialize expert paging
        print("\n🔄 Initializing expert paging...")
        self.expert_pager = self.initialize_expert_paging(model_config)

        # Replace MoE layers with paged versions
        print("\n🔄 Replacing MoE layers with paged versions...")
        model = engine.engine.model_executor.driver_worker.model_runner.model
        model = replace_moe_layers_with_paged(model, self.expert_pager)

        print("\n✅ Server ready!")
        print(f"\nEndpoints:")
        print(f"  - POST http://{self.host}:{self.port}/v1/chat/completions")
        print(f"  - POST http://{self.host}:{self.port}/v1/completions")
        print(f"  - GET  http://{self.host}:{self.port}/health")
        print(f"  - GET  http://{self.host}:{self.port}/metrics")
        print("\nTest with curl:")
        print(f"""
curl http://{self.host}:{self.port}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{self.model}",
    "messages": [{{"role": "user", "content": "Hello!"}}]
  }}'
""")

        # Start vLLM's OpenAI API server
        import uvicorn
        from vllm.entrypoints.openai.api_server import app

        # Attach expert pager to app state for metrics endpoint
        app.state.expert_pager = self.expert_pager

        uvicorn.run(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
        )


def main():
    parser = argparse.ArgumentParser(
        description="vLLM server with SparseLLM expert paging"
    )

    # Model args
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., mistralai/Mixtral-8x7B-Instruct-v0.1)"
    )

    # Expert paging args
    parser.add_argument(
        "--gpu-cache-gb",
        type=float,
        default=2.0,
        help="GPU cache size for active experts (GB)"
    )
    parser.add_argument(
        "--cpu-cache-gb",
        type=float,
        default=14.0,
        help="CPU cache size for warm experts (GB)"
    )

    # Server args
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )

    # vLLM args
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0)"
    )

    args = parser.parse_args()

    # Create and start server
    server = SparseLLMServer(
        model=args.model,
        gpu_cache_gb=args.gpu_cache_gb,
        cpu_cache_gb=args.cpu_cache_gb,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    try:
        server.start()
    except KeyboardInterrupt:
        print("\n\n✋ Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
