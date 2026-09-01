# SparseLLM Architecture

## Purpose

SparseLLM provides a model-independent local inference flow for open-source
causal language models that are loadable through Hugging Face Transformers:

```text
model ID or local checkpoint
  → lazy tokenizer/model loading
  → real prefill
  → autoregressive decoding with the model's KV-cache semantics
  → generated text, token IDs, and measured metrics
```

This repository is a correctness-first baseline. It does not claim that every
model fits a fixed VRAM size, and it does not report unmeasured latency,
throughput, cache-hit, or memory values.

## Runtime boundaries

### `ModelAdapter`

`ModelAdapter` is the stable runtime contract. An adapter owns model loading,
capability discovery, and generation. `TransformersCausalLMAdapter` is the
generic implementation and accepts both Hub identifiers and local directories.
It imports Transformers only when loading starts, so package imports and CLI
help do not download or instantiate checkpoints.

`DevicePolicy` carries device, dtype, revision, offline, remote-code, device-map,
offload-folder, optional expert-cache settings, and the user's loading and
quantization mode. Safe defaults are CPU/GPU auto-selection and
`trust_remote_code=False`; callers must opt into custom remote code explicitly.
Quantization is not hard-coded: `quantization="auto"` performs no runtime
conversion and loads the checkpoint representation published by the user’s
configured model source, including its declared quantization when the selected
runtime supports it. An explicit user mode may request an available
quantization backend or format during loading. The adapter must reject an
unavailable or incompatible request with an actionable error rather than
silently changing precision or falling back to another quantization policy.

### Framework roles

Hugging Face Transformers remains the universal model-loading and reference
execution path. Hugging Face Accelerate is the low-memory loading and placement
integration for meta-device initialization, CPU offload, and explicit device
maps where supported. vLLM is the planned serving layer for continuous
batching, paged attention, request scheduling, and efficient multi-user
execution after single-request paging is correct. vLLM does not replace the
validated SparseLLM expert pager: dynamic layer-aware CPU/SSD-to-GPU expert
residency remains an adapter capability and must be integrated and benchmarked
explicitly.

### `ModelRegistry`

The registry checks explicitly registered architecture adapters first and then
uses the generic Transformers fallback. Selection is capability-driven and not
based on a model-name allow-list. A specialization may be registered only when
it validates the model's module topology, routing behavior, tensor shapes, and
reference-output conformance.

### Capabilities and safe fallback

`ModelCapabilities` reports model type, architecture, dense/MoE classification,
layer/expert counts when discoverable, KV-cache support, and whether expert
paging is actually enabled.

A dense model uses the normal Transformers path. An unknown or unsupported MoE
model also uses the normal Transformers path. The generic adapter reports
`expert_paging=False`; it never guesses module paths, fabricates router output,
or swaps tensors merely because a configuration contains an MoE-like field.

## Memory hierarchy

The generic baseline delegates placement to Transformers and PyTorch. Where a
validated MoE adapter exists in a later milestone, the intended hierarchy is:

- GPU: shared model weights, active experts, bounded expert cache, KV cache, and
  activations;
- CPU: inactive expert tensors and optional pinned staging buffers;
- local storage: checkpoint-backed cold tensors or atomic local artifacts.

Expert identity is always `(layer_index, expert_index)`. The cache supports an
authoritative byte budget, deterministic LRU eviction, pinning during forward
execution, and in-flight load de-duplication. Pinned entries cannot be evicted;
if capacity cannot be admitted, the operation fails with a diagnostic.

The current generic adapter does not connect this cache to arbitrary model
internals. Cache metrics therefore remain zero for generic generation rather
than pretending that expert swapping occurred.

## Measurement

Each generation returns:

- prompt and generated token counts;
- prefill latency;
- decode latency;
- generated-token throughput;
- peak CUDA memory allocation when running on CUDA;
- expert cache hits/misses and expert load time, when a paging adapter owns
  those measurements.

Prefill is measured separately from decode. Decode reuses `past_key_values` when
the model provides them and falls back to full-context forward passes when it
does not. Fixed generation settings can therefore be compared against a
reference Transformers run without conflating transfer time and model compute.

## Operational sequence

1. Construct `InferenceEngine` with a model ID/path or injected adapter.
2. Constructing the engine does not load checkpoint artifacts.
3. Call `generate`; the adapter loads tokenizer/model resources, selects the
   requested device policy, and runs real tokenization and decoding.
4. Inspect `GenerationResult` and `ModelCapabilities`.
5. Record the exact model, revision, device, dtype, prompt, token limit, and
   measured metrics for reproducible comparisons.

The CLI in `main.py` follows the same sequence and exposes offline, revision,
remote-code, device-map, and offload controls.

## Extension and scaling path

The order is deliberate:

1. generic Transformers correctness and deterministic CPU fixtures;
2. a validated adapter for one concrete MoE topology, starting with Mixtral as
   a validation target rather than a global model restriction;
3. module-integrated layer-aware expert paging with reference conformance;
4. established quantization backends with numerical and memory measurements;
5. pinned-memory and asynchronous transfer overlap;
6. observed-next-use prefetch;
7. shared cache, continuous batching, and paged KV storage;
8. optional multi-worker serving and multi-GPU placement.

Prediction, speculative prefetch, network storage, custom INT4 kernels,
batching, and multi-GPU execution are not correctness dependencies for the
baseline.

## Verification

CPU tests use injected fake tokenizers/models or temporary local tensors and do
not download checkpoints. The verification suite checks:

- lazy imports and registry fallback;
- capability discovery and safe unsupported-MoE behavior;
- deterministic real forward/decode behavior;
- cache byte accounting, pinning, LRU eviction, and concurrent load sharing;
- atomic local storage and conservative tensor loading;
- CLI help and Python compilation.

A real-checkpoint smoke test is separate and environment-dependent. It must not
be represented by a synthetic benchmark or a static README table.
