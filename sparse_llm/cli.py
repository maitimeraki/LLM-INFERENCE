"""
Command-line interface for SparseLLM.

Provides commands for:
- serve: Start inference server
- generate: Single generation (CLI mode)
- stats: View model statistics
"""

import click
import logging
from typing import Optional


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool):
    """SparseLLM - Efficient MoE Inference Engine"""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


@cli.command()
@click.option('--model', '-m',
              default='deepseek-ai/DeepSeek-V3',
              help='Model path or HuggingFace model ID')
@click.option('--host', default='0.0.0.0', help='Server host')
@click.option('--port', '-p', default=8000, help='Server port')
@click.option('--device', '-d',
              default='cuda',
              type=click.Choice(['cuda', 'cpu', 'auto']),
              help='Device to use')
@click.option('--cache-size', '-c',
              default=4,
              help='[DEPRECATED] Use --gpu-cache-gb instead')
@click.option('--gpu-cache-gb',
              default=None,
              type=float,
              help='GPU cache size in GB (auto-detected if not specified)')
@click.option('--cpu-cache-gb',
              default=None,
              type=float,
              help='CPU cache size in GB (auto-detected if not specified)')
@click.option('--allow-cpu-offload/--no-cpu-offload',
              default=True,
              help='Allow warm experts on CPU (default: enabled)')
@click.option('--allow-ssd-offload/--no-ssd-offload',
              default=True,
              help='Allow cold experts on storage (default: enabled)')
@click.option('--max-tokens',
              default=2048,
              help='Default maximum tokens to generate')
@click.option('--max-seq-len',
              default=1024,
              type=int,
              help='Maximum sequence length for KV cache (default: 1024, reduces memory)')
@click.option('--dtype',
              default='float16',
              type=click.Choice(['float16', 'bfloat16', 'float32', 'int8', 'int4', 'fp8']),
              help='Runtime computation and KV cache dtype')
@click.option('--quantization', '-q',
              default=None,
              type=click.Choice(['int4', 'awq', 'gptq', 'fp8', 'int8', 'float16', 'bfloat16', 'float32']),
              help='Weight quantization format — weight-only storage, does not affect KV cache')
def serve(
    model: str,
    host: str,
    port: int,
    device: str,
    cache_size: int,
    gpu_cache_gb: float,
    cpu_cache_gb: float,
    allow_cpu_offload: bool,
    allow_ssd_offload: bool,
    max_tokens: int,
    max_seq_len: int,
    dtype: str,
    quantization: str | None
):
    """
    Start MoE inference server with three-tier expert caching.

    The server exposes an OpenAI-compatible API that can be used with
    Claude Code by configuring the baseURL in settings.json.

    Three-tier caching:
    - GPU (Hot): Fastest, smallest - frequently used experts
    - CPU (Warm): Fast, medium - occasionally used experts
    - Storage (Cold): Slow, largest - rarely used experts loaded on-demand

    Example:
        sparse-llm serve --model Qwen/Qwen1.5-MoE-A2.7B --port 8000

    With manual tier configuration:
        sparse-llm serve --model Qwen/Qwen1.5-MoE-A2.7B \\
            --gpu-cache-gb 8 --cpu-cache-gb 16 --port 8000

    Then add to Claude Code settings.json:
        {"baseURL": "http://localhost:8000/v1"}
    """
    from sparse_llm.serve import start_server

    # Auto-detect device if needed
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Show deprecation warning for --cache-size
    if cache_size != 4:
        click.echo("[DEPRECATED] --cache-size is deprecated, use --gpu-cache-gb instead")

    click.echo(f"Starting server with model: {model}")
    click.echo(f"Device: {device}")
    click.echo(f"Three-tier caching:")
    click.echo(f"  GPU tier: {'auto' if gpu_cache_gb is None else f'{gpu_cache_gb}GB'}")
    click.echo(f"  CPU tier: {'auto' if cpu_cache_gb is None else f'{cpu_cache_gb}GB'} "
               f"({'enabled' if allow_cpu_offload else 'disabled'})")
    click.echo(f"  Storage tier: {'enabled' if allow_ssd_offload else 'disabled'}")

    start_server(
        model_path=model,
        host=host,
        port=port,
        device=device,
        expert_cache_size=cache_size,
        max_tokens=max_tokens,
        max_seq_len=max_seq_len,
        dtype=dtype,
        quantization=quantization,
        gpu_cache_gb=gpu_cache_gb if gpu_cache_gb is not None else 0.0,
        cpu_cache_gb=cpu_cache_gb if cpu_cache_gb is not None else 0.0,
        allow_cpu_offload=allow_cpu_offload,
        allow_ssd_offload=allow_ssd_offload,
    )


@cli.command()
@click.option('--model', '-m',
              default='deepseek-ai/DeepSeek-V3',
              help='Model path or HuggingFace model ID')
@click.option('--prompt', '-p',
              required=True,
              help='Text prompt to generate from')
@click.option('--max-tokens',
              default=100,
              help='Maximum tokens to generate')
@click.option('--temperature', '-t',
              default=0.7,
              help='Sampling temperature')
@click.option('--device', '-d',
              default='cuda',
              type=click.Choice(['cuda', 'cpu', 'auto']),
              help='Device to use')
@click.option('--dtype',
              default='float16',
              type=click.Choice(['float16', 'bfloat16', 'float32', 'int8', 'int4', 'fp8']),
              help='Runtime computation dtype')
@click.option('--quantization', '-q',
              default=None,
              type=click.Choice(['int4', 'awq', 'gptq', 'fp8', 'int8', 'float16', 'bfloat16', 'float32']),
              help='Weight quantization format')
@click.option('--streaming-mode',
              is_flag=True,
              help='Enable streaming mode for low-resource devices (no caching, load-execute-free)')
@click.option('--max-parallel-expert-loads',
              default=4,
              type=int,
              help='Number of parallel expert loads in streaming mode (default: 4)')
def generate(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    device: str,
    dtype: str,
    quantization: str | None,
    streaming_mode: bool,
    max_parallel_expert_loads: int,
):
    """
    Generate text from a prompt (CLI mode).

    Example:
        sparse-llm generate -p "Write a Python function to sort a list"
    """
    from sparse_llm.inference.moe_inference_engine import (
        CustomMoEInferenceEngine,
        InferenceConfig
    )
    import torch

    # Auto-detect device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    click.echo(f"Loading model: {model}")
    click.echo(f"Device: {device}")
    click.echo(f"Dtype: {dtype}" + (f" + {quantization}" if quantization else "") + "\n")

    # Create config
    config = InferenceConfig(
        model_path=model,
        device=device,
        max_tokens=max_tokens,
        temperature=temperature,
        dtype=dtype,
        quantization=quantization,
        streaming_mode=streaming_mode,
        max_parallel_expert_loads=max_parallel_expert_loads,
    )

    if streaming_mode:
        click.echo("Streaming mode enabled: load-execute-free pattern (no caching)")

    # Initialize engine
    click.echo("Initializing engine...")
    engine = CustomMoEInferenceEngine(model_path=model, config=config)

    # Generate
    click.echo("\nGenerating...\n")
    click.echo("-" * 70)

    result = engine.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )

    click.echo(result)
    click.echo("-" * 70)

    # Show stats
    stats = engine.get_statistics()
    click.echo(f"\nTokens/sec: {stats['average_tokens_per_second']:.1f}")
    click.echo(f"Memory: {stats['memory_usage_gb']:.2f}GB")


@cli.command()
@click.option('--url',
              default='http://localhost:8000',
              help='Server URL')
def stats(url: str):
    """
    View server statistics.

    Example:
        sparse-llm stats --url http://localhost:8000
    """
    import requests
    from rich.console import Console
    from rich.table import Table

    console = Console()

    try:
        response = requests.get(f"{url}/stats", timeout=5)
        response.raise_for_status()
        data = response.json()

        # Create table
        table = Table(title="SparseLLM Server Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Requests", str(data.get('total_requests', 0)))
        table.add_row("Total Tokens", str(data.get('total_tokens_generated', 0)))
        table.add_row("Avg Tokens/Sec", f"{data.get('average_tokens_per_second', 0):.1f}")
        table.add_row("Memory Usage", f"{data.get('memory_usage_gb', 0):.2f}GB")

        # Cache stats
        cache_stats = data.get('cache_stats', {})
        if cache_stats:
            table.add_row("Cache Hit Rate", f"{cache_stats.get('hit_rate', 0):.1%}")
            table.add_row("Cache Size", str(cache_stats.get('size', 0)))

        console.print(table)

    except requests.exceptions.ConnectionError:
        click.echo(f"Error: Could not connect to server at {url}")
        click.echo("Make sure the server is running with: sparse-llm serve")
    except Exception as e:
        click.echo(f"Error: {e}")


@cli.command()
def version():
    """Show version information."""
    click.echo("SparseLLM v0.1.0")
    click.echo("Custom PyTorch MoE Inference Engine")


if __name__ == '__main__':
    cli()
