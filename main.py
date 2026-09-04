#!/usr/bin/env python3
"""Command-line entry point for measured local causal-LM generation."""

from __future__ import annotations

import argparse
import json
import os
from dotenv import load_dotenv
load_dotenv()
from sparse_llm import DevicePolicy, InferenceEngine
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate text with any Transformers-compatible causal language model."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local checkpoint directory")
    parser.add_argument("--prompt", required=True, help="Prompt to tokenize and continue")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum number of generated tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature; zero uses greedy decoding")
    parser.add_argument("--device", default="auto", help="Execution device, such as auto, cpu, or cuda")
    parser.add_argument(
        "--dtype",
        default=None,
        choices=("float32", "float16", "bfloat16", "fp32", "fp16", "bf16"),
        help="Optional model dtype",
    )
    parser.add_argument("--revision", default=None, help="Optional immutable model revision or commit")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not contact the model hub; require local cached/model files",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code (disabled by default)",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="Optional Transformers device map, for example auto",
    )
    parser.add_argument(
        "--offload-folder",
        default=None,
        help="Optional folder used by a Transformers device-map offload policy",
    )
    parser.add_argument(
        "--cache-bytes",
        type=int,
        default=None,
        help="Reserved expert-paging budget in bytes; generic adapters do not page experts",
    )
    parser.add_argument(
        "--use-resource-aware",
        action="store_true",
        help="Use resource-aware four-phase initialization (experimental)"
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON result")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.cache_bytes is not None and args.cache_bytes < 1:
        raise ValueError("--cache-bytes must be positive")

    # Resource-aware loading path (experimental)
    if args.use_resource_aware:
        from sparse_llm.loading import FourPhaseOrchestrator

        print("Using resource-aware initialization...")
        orchestrator = FourPhaseOrchestrator()

        def progress_callback(message: str):
            print(message)

        state = orchestrator.initialize(
            model_id=args.model,
            progress_callback=progress_callback
        )

        # TODO: Integrate state with InferenceEngine
        print("\nResource-aware loading complete. Backend integration pending.")
        print(f"Loaded model: {state.model_info.model_id}")
        print(f"Is MoE: {state.model_info.is_moe}")
        print(f"Shared weights: {len(state.shared_weights)} tensors")

        if state.model_info.is_moe:
            cache_stats = state.get_cache_stats()
            print(f"Expert cache: GPU slots={state.expert_cache.gpu_slots}, CPU slots={state.expert_cache.cpu_slots}")
            print(f"Cache stats: {cache_stats}")

        return 0

    # Traditional loading path (existing code)
    policy = DevicePolicy(
        device=args.device,
        dtype=args.dtype,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        offload_folder=args.offload_folder,
        expert_cache_bytes=args.cache_bytes,
    )
    engine = InferenceEngine(
        model=args.model,
        device=policy.device,
        dtype=policy.dtype,
        revision=policy.revision,
        local_files_only=policy.local_files_only,
        trust_remote_code=policy.trust_remote_code,
        device_map=policy.device_map,
        offload_folder=policy.offload_folder,
        expert_cache_bytes=policy.expert_cache_bytes,
    )
    result = engine.generate(args.prompt, args.max_new_tokens, args.temperature)
    payload = {
        "text": result.text,
        "token_ids": result.token_ids,
        "metrics": result.metrics.to_dict(),
        "capabilities": engine.capabilities.to_dict(),
    }

    # Add paging diagnostics when expert paging is enabled or available.
    if engine.capabilities.expert_paging:
        payload["paging_diagnostics"] = {
            "enabled": True,
            "cache_hits": result.metrics.cache_hits,
            "cache_misses": result.metrics.cache_misses,
            "expert_load_time_ms": result.metrics.expert_load_time_ms,
        }

    if args.cache_bytes is not None:
        payload["configured_expert_cache_bytes"] = args.cache_bytes
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(result.text)
        print(json.dumps({"metrics": payload["metrics"], "capabilities": payload["capabilities"]}, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
