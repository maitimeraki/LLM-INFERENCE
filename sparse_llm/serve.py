"""
FastAPI server for SparseLLM MoE inference.

Exposes OpenAI-compatible API endpoint for Claude Code integration.
"""

import asyncio
import logging
import sys
import time
from typing import List, Literal, Optional
from uuid import uuid4

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel

from sparse_llm.inference.moe_inference_engine import (
    CustomMoEInferenceEngine,
    InferenceConfig,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
console = Console()

# FastAPI app
app = FastAPI(
    title="SparseLLM MoE Server",
    description="OpenAI-compatible inference server for MoE models",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: Optional[CustomMoEInferenceEngine] = None
engine_config: Optional[InferenceConfig] = None


# Request/Response Models
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "sparse-moe"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


@app.on_event("startup")
async def load_model():
    """Load MoE model on server startup."""
    global engine

    console.print("\n[yellow]Loading MoE model...[/yellow]")
    console.print(f"[dim]Model: {engine_config.model_path}[/dim]")
    console.print(f"[dim]Device: {engine_config.device}[/dim]")

    try:
        start_time = time.monotonic()

        # Initialize engine
        engine = CustomMoEInferenceEngine(
            model_path=engine_config.model_path,
            config=engine_config
        )

        elapsed = time.monotonic() - start_time
        console.print(f"[green]OK Model loaded successfully in {elapsed:.1f}s![/green]")

        # Log memory usage
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            console.print(f"[dim]GPU memory: {mem_gb:.2f}GB[/dim]")

    except Exception as e:
        console.print(f"[red]X Failed to load model: {e}[/red]")
        logger.exception("Model loading failed")
        sys.exit(1)


@app.on_event("shutdown")
async def shutdown_model():
    """Cleanup on server shutdown."""
    global engine
    if engine is not None:
        console.print("\n[yellow]Shutting down model...[/yellow]")
        del engine
        engine = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        console.print("[green]OK Shutdown complete[/green]")


@app.get("/")
async def root():
    """Health check endpoint."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "ready",
        "model": "sparse-moe",
        "engine": "custom-pytorch",
        "version": "0.1.0"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    stats = engine.get_statistics()

    return {
        "status": "healthy",
        "model_loaded": True,
        "statistics": {
            "total_requests": stats["total_requests"],
            "total_tokens": stats["total_tokens_generated"],
            "avg_tokens_per_sec": stats["average_tokens_per_second"],
            "memory_usage_gb": stats["memory_usage_gb"]
        }
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": "sparse-moe",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "sparse-llm"
            }
        ]
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    This is the main endpoint that Claude Code will use.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert messages to prompt
        prompt = convert_messages_to_prompt(request.messages)

        # Log request
        logger.info(f"Received request: {len(request.messages)} messages, "
                   f"max_tokens={request.max_tokens}, temp={request.temperature}")

        # Generate response
        start_time = time.monotonic()

        generated_text = engine.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stream=False  # Streaming handled separately
        )

        elapsed = time.monotonic() - start_time

        # Count tokens (approximate)
        prompt_tokens = len(prompt.split())  # Rough approximation
        completion_tokens = len(generated_text.split())

        logger.info(f"Generated {completion_tokens} tokens in {elapsed:.2f}s "
                   f"({completion_tokens / elapsed:.1f} tok/s)")

        # Build response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{generate_id()}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=generated_text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

        return response

    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get detailed engine statistics."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return engine.get_statistics()


def convert_messages_to_prompt(messages: List[Message]) -> str:
    """
    Convert chat messages to model prompt format.

    Supports common formats:
    - ChatML: <|system|>, <|user|>, <|assistant|>
    - Llama: [INST], [/INST]
    - DeepSeek: Similar to ChatML
    """
    prompt_parts = []

    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"<|system|>\n{msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"<|user|>\n{msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"<|assistant|>\n{msg.content}")

    # Add final assistant tag for generation
    prompt_parts.append("<|assistant|>")

    return "\n\n".join(prompt_parts)


def generate_id() -> str:
    """Generate unique ID for responses."""
    return str(uuid4())[:8]


def start_server(
    model_path: str = "deepseek-ai/DeepSeek-V3",
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = "cuda",
    expert_cache_size: int = 4,
    max_tokens: int = 2048,
    dtype: str = "float16"
):
    """
    Start the FastAPI server with MoE model.

    Args:
        model_path: HuggingFace model ID or local path
        host: Server host
        port: Server port
        device: Device to use (cuda/cpu)
        expert_cache_size: Number of experts to cache
        max_tokens: Default max tokens to generate
        dtype: Model dtype (float16/bfloat16/float32)
    """
    global engine_config

    # Create inference config
    engine_config = InferenceConfig(
        model_path=model_path,
        device=device,
        dtype=dtype,
        expert_cache_size=expert_cache_size,
        max_tokens=max_tokens,
        enable_predictive_prefetch=True,
        use_flash_attention=True,
    )

    # Print banner
    base_url = f"http://localhost:{port}"

    console.print("\n" + "=" * 70)
    console.print(Panel.fit(
        f"[bold green]SparseLLM MoE Server[/bold green]\n\n"
        f"Base URL: [cyan]{base_url}[/cyan]\n\n"
        f"[bold]Configuration:[/bold]\n"
        f"  Model: {model_path}\n"
        f"  Device: {device}\n"
        f"  Expert cache: {expert_cache_size} experts\n"
        f"  Max tokens: {max_tokens}\n\n"
        f"[bold yellow]Add this to Claude Code settings.json:[/bold yellow]\n"
        f'{{"baseURL": "{base_url}/v1"}}',
        border_style="green",
        padding=(1, 2)
    ))
    console.print("=" * 70 + "\n")

    console.print(f"[dim]Server starting on {host}:{port}...[/dim]\n")

    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    # For direct execution: python serve.py
    start_server()
