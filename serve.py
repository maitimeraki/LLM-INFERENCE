#!/usr/bin/env python3
"""Unified Production Server: Conflict-Free vLLM Integration

This server integrates SparseLLM's resource-aware 4-phase loading with vLLM's
high-performance serving infrastructure without conflicts.

KEY INTEGRATION POINTS:

1. UnifiedMemoryCoordinator: Splits GPU memory (40% expert cache, 50% KV cache)
   to prevent OOM conflicts between systems.

2. SparseMoEWeightBridge: Makes vLLM use pre-loaded weights from LoadedWeightState
   instead of loading from scratch (eliminates duplicate loading).

3. Weight Injection: Replaces vLLM's model parameters with pre-loaded weights
   during engine initialization (single source of truth for weights).

ARCHITECTURE:

Phase 1: Resource-Aware Loading
  FourPhaseOrchestrator loads weights across GPU/CPU/Storage tiers
  Creates LoadedWeightState with shared_weights and expert_cache

Phase 2: vLLM Integration
  UnifiedMemoryCoordinator establishes memory budgets
  SparseMoEWeightBridge connects vLLM to LoadedWeightState
  AsyncLLMEngine initialized with coordinated config
  Weights injected from LoadedWeightState (no duplicate loading)

Phase 3: Serving
  OpenAI-compatible API endpoints
  Request batching with vLLM scheduler
  Persistent weights across requests

USAGE:

    python serve.py --model mistralai/Mixtral-8x7B-Instruct-v0.1
    python serve.py --model meta-llama/Llama-3.1-8B --port 8080

See docs/VLLM_INTEGRATION.md for complete documentation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any

import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# Check vLLM availability
try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("ERROR: vLLM not installed. Install with: pip install vllm")
    sys.exit(1)

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("ERROR: FastAPI not installed. Install with: pip install fastapi uvicorn")
    sys.exit(1)

# SparseLLM imports
from sparse_llm.loading import FourPhaseOrchestrator, LoadedWeightState
from sparse_llm.integrations.vllm_bridge import initialize_vllm_with_coordination


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UnifiedServer:
    """Unified server combining resource-aware loading with vLLM serving."""

    def __init__(
        self,
        model: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        storage_path: str | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
    ):
        """Initialize unified server.

        Args:
            model: HuggingFace model ID or local path
            host: Server host address
            port: Server port
            storage_path: Optional custom storage path for cold experts
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization factor
            max_model_len: Maximum model sequence length
        """
        self.model = model
        self.host = host
        self.port = port
        self.storage_path = storage_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

        # Runtime state
        self.loaded_state: LoadedWeightState | None = None
        self.vllm_engine: AsyncLLMEngine | None = None
        self.weight_bridge = None
        self.allocation = None  # NEW: Memory allocation info
        self.startup_time: float = 0.0
        self.request_count: int = 0

    async def initialize_vllm_with_coordination_method(self):
        """Initialize vLLM using OOM-safe coordination (NEW METHOD).

        This replaces the old two-phase approach with a single coordinated initialization
        that prevents OOM by pre-calculating memory allocation.

        Returns:
            Tuple of (vllm_engine, weight_bridge, memory_allocation, loaded_state)
        """
        logger.info("="*70)
        logger.info("🚀 OOM-SAFE INITIALIZATION (Memory-Coordinated)")
        logger.info("="*70)

        phase_start = time.time()

        # Use the new OOM-safe initialization function
        vllm_engine, weight_bridge, memory_allocation = initialize_vllm_with_coordination(
            model_id=self.model,
            user_vllm_params={
                "max_model_len": self.max_model_len,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,  # Will be overridden by calculator
            },
            storage_path=self.storage_path
        )

        phase_time = time.time() - phase_start

        # Extract loaded state from weight bridge
        loaded_state = weight_bridge.loaded_state

        logger.info("")
        logger.info("="*70)
        logger.info(f"✅ OOM-safe initialization complete in {phase_time:.1f}s")
        logger.info("="*70)
        logger.info(f"   Model: {loaded_state.model_info.model_id}")
        logger.info(f"   Type: {'MoE' if loaded_state.model_info.is_moe else 'Dense'}")

        if loaded_state.model_info.is_moe:
            logger.info(f"   Layers: {loaded_state.model_info.num_layers}")
            logger.info(f"   Experts: {loaded_state.model_info.num_experts}")
            logger.info(f"   Hot experts (GPU): {memory_allocation.gpu_hot_experts}")
            logger.info(f"   Warm experts (CPU): {memory_allocation.cpu_warm_experts}")
            logger.info(f"   Cold experts (Storage): {memory_allocation.ssd_cold_experts}")

        logger.info(f"   vLLM gpu_memory_utilization: {memory_allocation.vllm_gpu_memory_utilization:.3f}")
        logger.info(f"   GPU utilization: {memory_allocation.gpu_total_used / 1024**3:.2f}GB")

        return vllm_engine, weight_bridge, memory_allocation, loaded_state

    async def startup(self):
        """Server startup: OOM-safe initialization with memory coordination."""
        startup_start = time.time()

        logger.info("")
        logger.info("🚀 Starting Unified SparseLLM Production Server (OOM-Safe Mode)")
        logger.info(f"   Model: {self.model}")
        logger.info(f"   Host: {self.host}:{self.port}")
        logger.info(f"   Max model length: {self.max_model_len or 'auto'}")
        logger.info("")

        # NEW: Single-phase OOM-safe initialization
        self.vllm_engine, self.weight_bridge, self.allocation, self.loaded_state = \
            await self.initialize_vllm_with_coordination_method()

        self.startup_time = time.time() - startup_start

        logger.info("")
        logger.info("="*70)
        logger.info("✅ SERVER READY (OOM-Safe)")
        logger.info("="*70)
        logger.info(f"   Total startup time: {self.startup_time:.1f}s")
        logger.info(f"   Memory coordination: ACTIVE")
        logger.info(f"   Endpoints:")
        logger.info(f"      POST http://{self.host}:{self.port}/v1/chat/completions")
        logger.info(f"      POST http://{self.host}:{self.port}/v1/completions")
        logger.info(f"      GET  http://{self.host}:{self.port}/health")
        logger.info(f"      GET  http://{self.host}:{self.port}/metrics")
        logger.info("")
        logger.info("Test with curl:")
        logger.info(f"""
curl http://{self.host}:{self.port}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{self.model}",
    "messages": [{{"role": "user", "content": "Hello!"}}],
    "max_tokens": 100
  }}'
""")

    async def shutdown(self):
        """Server shutdown: cleanup resources."""
        logger.info("")
        logger.info("🛑 Shutting down server...")

        if self.vllm_engine:
            logger.info("   Stopping vLLM engine...")
            # vLLM cleanup is handled automatically

        logger.info(f"   Processed {self.request_count} requests during this session")
        logger.info("✅ Shutdown complete")

    async def generate_completion(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stream: bool = False,
    ) -> Dict[str, Any] | AsyncIterator[str]:
        """Generate completion using vLLM engine.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response

        Returns:
            Completion response or async iterator for streaming
        """
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized")

        self.request_count += 1
        request_id = random_uuid()

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        # Generate
        start_time = time.time()

        if stream:
            # Streaming response
            async def stream_results():
                async for request_output in self.vllm_engine.generate(
                    prompt, sampling_params, request_id
                ):
                    for output in request_output.outputs:
                        yield output.text

            return stream_results()
        else:
            # Non-streaming response
            final_output = None
            async for request_output in self.vllm_engine.generate(
                prompt, sampling_params, request_id
            ):
                final_output = request_output

            if not final_output:
                raise RuntimeError("No output generated")

            output = final_output.outputs[0]
            elapsed_time = time.time() - start_time

            return {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.model,
                "choices": [{
                    "index": 0,
                    "text": output.text,
                    "logprobs": None,
                    "finish_reason": output.finish_reason,
                }],
                "usage": {
                    "prompt_tokens": len(final_output.prompt_token_ids),
                    "completion_tokens": len(output.token_ids),
                    "total_tokens": len(final_output.prompt_token_ids) + len(output.token_ids),
                },
                "metrics": {
                    "elapsed_time_ms": elapsed_time * 1000,
                    "tokens_per_sec": len(output.token_ids) / elapsed_time if elapsed_time > 0 else 0,
                }
            }

    async def generate_chat_completion(
        self,
        messages: list[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stream: bool = False,
    ) -> Dict[str, Any] | AsyncIterator[str]:
        """Generate chat completion using vLLM engine.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response

        Returns:
            Chat completion response or async iterator for streaming
        """
        # Convert messages to prompt
        # Note: This is a simple implementation. For production, use proper chat templates
        prompt = self._format_chat_prompt(messages)

        # Generate using completion endpoint
        result = await self.generate_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )

        if stream:
            return result

        # Convert completion to chat completion format
        return {
            "id": result["id"],
            "object": "chat.completion",
            "created": result["created"],
            "model": result["model"],
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["choices"][0]["text"],
                },
                "finish_reason": result["choices"][0]["finish_reason"],
            }],
            "usage": result["usage"],
            "metrics": result.get("metrics", {}),
        }

    def _format_chat_prompt(self, messages: list[Dict[str, str]]) -> str:
        """Format chat messages into a prompt string.

        Args:
            messages: List of chat messages with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        # Simple chat format - for production, use model-specific templates
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics including resource-aware stats.

        Returns:
            Metrics dictionary
        """
        metrics = {
            "server": {
                "model": self.model,
                "startup_time_sec": self.startup_time,
                "request_count": self.request_count,
                "uptime_sec": time.time() - (time.time() - self.startup_time) if self.startup_time > 0 else 0,
            },
            "model_info": {},
            "resource_aware": {},
        }

        if self.loaded_state:
            metrics["model_info"] = {
                "model_id": self.loaded_state.model_info.model_id,
                "is_moe": self.loaded_state.model_info.is_moe,
                "num_layers": self.loaded_state.model_info.num_layers,
                "num_experts": self.loaded_state.model_info.num_experts if self.loaded_state.model_info.is_moe else 0,
            }

            if self.loaded_state.model_info.is_moe:
                cache_stats = self.loaded_state.get_cache_stats()
                metrics["resource_aware"] = {
                    "expert_cache": cache_stats,
                    "placement": {
                        "gpu_experts": self.loaded_state.placement_plan.hot_expert_slots,
                        "cpu_experts": self.loaded_state.placement_plan.warm_expert_slots,
                        "storage_experts": self.loaded_state.placement_plan.cold_expert_count,
                        "gpu_utilization_pct": self.loaded_state.placement_plan.gpu_utilization_pct,
                        "cpu_utilization_pct": self.loaded_state.placement_plan.cpu_utilization_pct,
                    }
                }

            # Add memory allocation info if available
            if self.allocation:
                metrics["memory_allocation"] = {
                    "vllm_gpu_memory_utilization": self.allocation.vllm_gpu_memory_utilization,
                    "gpu_total_used_gb": self.allocation.gpu_total_used / 1024**3,
                    "gpu_hot_experts": self.allocation.gpu_hot_experts,
                    "cpu_warm_experts": self.allocation.cpu_warm_experts,
                    "ssd_cold_experts": self.allocation.ssd_cold_experts,
                    "can_fulfill": self.allocation.can_fulfill,
                }

        return metrics


# Global server instance
server: UnifiedServer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown."""
    global server

    # Startup
    if server:
        await server.startup()

    yield

    # Shutdown
    if server:
        await server.shutdown()


# Create FastAPI app
app = FastAPI(
    title="SparseLLM Unified Server",
    description="Resource-aware weight loading + vLLM high-performance serving",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global server

    if not server or not server.vllm_engine:
        raise HTTPException(status_code=503, detail="Server not ready")

    return {
        "status": "healthy",
        "model": server.model,
        "uptime_sec": time.time() - (time.time() - server.startup_time) if server.startup_time > 0 else 0,
    }


@app.get("/metrics")
async def get_metrics():
    """Get server metrics."""
    global server

    if not server:
        raise HTTPException(status_code=503, detail="Server not ready")

    return server.get_metrics()


@app.post("/v1/completions")
async def create_completion(request: Request):
    """OpenAI-compatible completions endpoint."""
    global server

    if not server or not server.vllm_engine:
        raise HTTPException(status_code=503, detail="Server not ready")

    body = await request.json()

    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 100)
    temperature = body.get("temperature", 0.0)
    top_p = body.get("top_p", 1.0)
    stream = body.get("stream", False)

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        result = await server.generate_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )

        if stream:
            async def stream_response():
                async for chunk in result:
                    yield f"data: {json.dumps({'choices': [{'text': chunk}]})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_response(), media_type="text/event-stream")

        return result

    except Exception as e:
        logger.error(f"Error generating completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    """OpenAI-compatible chat completions endpoint."""
    global server

    if not server or not server.vllm_engine:
        raise HTTPException(status_code=503, detail="Server not ready")

    body = await request.json()

    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 100)
    temperature = body.get("temperature", 0.0)
    top_p = body.get("top_p", 1.0)
    stream = body.get("stream", False)

    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    try:
        result = await server.generate_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )

        if stream:
            async def stream_response():
                async for chunk in result:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_response(), media_type="text/event-stream")

        return result

    except Exception as e:
        logger.error(f"Error generating chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified SparseLLM Production Server: Resource-Aware Loading + vLLM"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path (e.g., mistralai/Mixtral-8x7B-Instruct-v0.1)"
    )

    # Server arguments
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )

    # Resource-aware loading arguments
    parser.add_argument(
        "--storage-path",
        type=str,
        default=None,
        help="Custom storage path for cold experts"
    )

    # vLLM arguments
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization factor (default: 0.9)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model sequence length (optional)"
    )

    args = parser.parse_args()

    # Create global server instance
    global server
    server = UnifiedServer(
        model=args.model,
        host=args.host,
        port=args.port,
        storage_path=args.storage_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
