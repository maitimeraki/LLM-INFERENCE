# Universal Open-Source Model Inference Implementation Plan

> **Authority:** `docs/GOAL.md` plus the audit recorded below. This plan replaces the former Mixtral-only plan.

**Goal:** Deliver a truthful local inference vertical slice for any Hugging Face-compatible causal language model, while enabling expert paging only for architectures whose module topology and routing semantics are explicitly validated.

**Architecture:** One public runtime and one CLI. A lazy model registry discovers a model from its local/Hugging Face configuration, selects a generic Transformers adapter for compatible dense or unknown models, and selects a validated architecture adapter only when its capabilities can be proven. The generic path never fabricates routing or treats dense layers as experts. The MoE path owns layer-aware expert residency and uses the model's actual router.

**Tech stack:** Python 3.10+, PyTorch, Hugging Face Transformers, optional safetensors, pytest. CPU tests use tiny local fixtures or fakes and never download a checkpoint.

## Audit Findings Driving This Revision

The prior implementation does not reach the project goal:

- `sparse_llm/inference/engine.py` creates random tensors and fabricated text.
- `sparse_llm/core/inference.py` is a second incompatible engine that averages expert tensors instead of running a model.
- `sparse_llm/models/moe.py` is a toy MoE unrelated to checkpoint internals.
- The cache and local storage are useful primitives but are not connected to real model modules.
- The plan names Mixtral as the only adapter and has no capability contract or safe fallback for other architectures.
- CLI, README, and configuration contain stale unsupported model and benchmark claims.

The implementation must remove these false paths rather than layering another facade over them.

## Universal Compatibility Contract

“Any open-source model” means any locally available or Hugging Face-hosted model that Transformers can load as a causal language model through the supported API. Compatibility is discovered, not inferred from a model-name allow-list.

The runtime must:

1. Accept a model identifier or local checkpoint directory.
2. Load the model's tokenizer and `AutoModelForCausalLM` lazily.
3. Default to `trust_remote_code=False`; make opting in explicit.
4. Support offline/local-files-only operation and an optional fixed revision.
5. Select device and dtype without importing or downloading at package import time.
6. Use the model's own `generate`/forward implementation, tokenizer, vocabulary, cache behavior, and stopping semantics.
7. Fail with an actionable error when the artifact is not a causal language model or cannot fit the requested device policy.
8. Expose model capabilities as data: model type, architecture, dense/MoE classification, layer/expert counts when known, top-k when known, KV-cache support, and whether safe expert paging is available.

Generic compatibility is not a promise that every model fits in a given amount of VRAM. Memory limits must produce diagnostics or use an established Transformers device/offload policy; they must not silently change model semantics.

## Capability-Driven MoE Specialization

Expert paging is opt-in. An adapter may claim `expert_paging=True` only after validating:

- stable module paths for shared layers and experts;
- exact layer/expert identity as `(layer_index, expert_index)`;
- the model's router output, top-k behavior, and expert input/output contract;
- safe replacement or wrapping of expert modules without changing state-dict keys or numerical semantics;
- a supported dtype/device transfer path and a recoverable cold-storage representation.

Unknown or unsupported MoE architectures use the generic Transformers path and keep their normal loading behavior. They are never routed through the toy `SparseMoELayer`, guessed by model name, or advertised as paged. Mixtral is the first validated specialization, not the application limit. Additional adapters are added independently after conformance tests.

The pager must use a bounded byte budget, layer-aware keys, in-flight load de-duplication, and pinning while a forward pass uses an expert. It must never evict a pinned expert. If a request cannot be admitted within the configured budget, it fails clearly.

## Memory and Performance Rules

- Keep shared/non-expert weights resident where the selected device policy allows.
- Keep inactive experts on CPU or checkpoint-backed local storage; do not duplicate the complete expert set in VRAM.
- Account for model weights, active experts, KV cache, activations, allocator overhead, and safety margin.
- Separate prefill and decode measurements and reuse the model's KV cache.
- Use synchronous LRU first. Pinned CPU buffers, asynchronous transfers, streams, prefetch, batching, quantization, prediction, network storage, and multi-GPU are later optimizations and cannot be correctness dependencies.
- Every benchmark value is measured at runtime; no static throughput or cache-hit promises are permitted.

## Implementation Tasks

### Task 1 — Replace the plan and define stable public contracts

Update this document and add focused contracts for:

- `ModelCapabilities`, `ModelAdapter`, and `ModelRegistry`;
- `GenerationMetrics` and `GenerationResult`;
- explicit `DevicePolicy`/loading options;
- `ExpertKey = (layer_index, expert_index)` and capability-gated pager hooks.

Keep imports safe when Transformers is not installed. Preserve existing cache/storage compatibility exports while removing claims that they execute a model.

### Task 2 — Implement the universal lazy Transformers adapter

Create a model-independent adapter that:

- loads `AutoTokenizer` and `AutoModelForCausalLM` only on construction/load;
- accepts local paths and model IDs, `device`, dtype, `revision`, `local_files_only`, `trust_remote_code`, and optional established device-map/offload settings;
- discovers capabilities from the loaded config/module tree;
- performs real deterministic generation with `use_cache=True` when supported;
- measures prompt tokenization, prefill, decode, throughput, peak CUDA memory, and returns generated token IDs/text;
- exposes zero/unsupported expert metrics instead of pretending generic generation swapped experts.

Use dependency injection/fakes so CPU tests can verify behavior without a checkpoint.

### Task 3 — Consolidate runtime entrypoints and add validated specialization boundary

Replace both fabricated engines with one `InferenceEngine` that delegates to an adapter. Keep `sparse_llm.core.inference` as a compatibility re-export, and make the toy `SparseMoELayer` clearly non-production or remove it from the runtime path.

Add a registry/adapter boundary for validated MoE implementations. Implement capability reporting and pager interfaces without claiming unsupported architectures are paged. The first generic implementation may report paging unavailable; a concrete Mixtral pager is only accepted if it runs against a tiny structural fixture and reference-output conformance test.

### Task 4 — Make cache/storage production-safe primitives

Retain layer-aware compatibility behavior, then extend the cache for:

- byte-based capacity as the authoritative limit;
- optional entry size accounting;
- pin/unpin protection;
- deterministic eviction of unpinned entries;
- observable hit/miss/eviction/occupancy metrics;
- concurrency-safe in-flight load de-duplication where it is needed by the pager.

Keep atomic local persistence, CPU detachment, safe tensor loading, and actionable errors. Do not introduce network or speculative storage in this milestone.

### Task 5 — Replace CLI, configuration, docs, and claims

Replace demo execution with a CLI accepting model, prompt, max-new-tokens, device, dtype, cache budget, local-files-only, revision, and trust-remote-code controls. Print generated text and measured metrics as JSON or readable output. Update README and package metadata to describe universal compatible-model support, honest limitations, and the staged MoE optimization path. Remove stale 100B+, fixed benchmark, and phase-demo claims.

### Task 6 — Verification and operational checks

Add CPU-only tests for:

- registry selection and lazy imports;
- capability discovery and unsupported paging fallback;
- deterministic generation with a fake tokenizer/model;
- prefill/decode metric accounting;
- device/dtype/load-option validation;
- cache byte limits, pinning, and concurrent load de-duplication;
- storage compatibility and atomic failure cleanup.

Run focused tests, the full suite, `python -m compileall`, and CLI help without model downloads. If a real checkpoint is available locally, run a short smoke test and record the exact model, device, and measured results; otherwise report that environmental limitation explicitly.

## Global Constraints

- The public flow is model-independent and must not branch on a hardcoded model-name list.
- Only a validated adapter may enable expert paging; unknown architectures use truthful generic fallback.
- No random input tensors, fabricated generated text, fake router decisions, or static performance values.
- CPU tests must not download checkpoints or require CUDA.
- `trust_remote_code=False` and local/offline operation are safe defaults.
- Do not add prediction, speculative prefetch, batching, network storage, custom INT4 kernels, or multi-GPU execution to the correctness path.
- Preserve data safety: atomic writes and conservative tensor deserialization.
- Keep the public package importable without eagerly importing Transformers.

## Verification Definition of Done

The work is complete only when:

- `prompt → tokenizer → real compatible causal model → real generation → text and measured metrics` is executable;
- the same deterministic request produces the same output under fixed settings;
- a dense model and an unknown/unsupported MoE model follow the generic path without unsafe paging claims;
- layer-aware cache/storage primitives remain compatible and tested;
- the CLI help path does not load a model;
- the repository contains no unsupported static benchmark or universal-fit claim;
- all CPU-only tests pass, with real-checkpoint checks separately identified by environment.

## Deferred Roadmap

After this baseline is measured: validated Mixtral expert-module paging; established quantization backends; pinned-memory/asynchronous transfer overlap; immediate observed-next-use prefetch; continuous batching and paged KV storage; then optional routing locality/prediction and multi-worker serving. Each stage requires a correctness comparison and measured improvement against the preceding baseline.
