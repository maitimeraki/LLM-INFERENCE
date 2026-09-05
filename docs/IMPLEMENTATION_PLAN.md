# Universal Open-Source Model Inference Implementation Plan

> **Authority:** `docs/GOAL.md` plus the audit recorded below. This plan replaces the former Mixtral-only plan.

**Goal:** Deliver a truthful local inference vertical slice for any Hugging Face-compatible causal language model, while enabling expert paging only for architectures whose module topology and routing semantics are explicitly validated.

**Architecture:** One public runtime and one CLI. A lazy model registry discovers a model from its local/Hugging Face configuration, selects a generic Transformers adapter for compatible dense or unknown models, and selects a validated architecture adapter only when its capabilities can be proven. The generic path never fabricates routing or treats dense layers as experts. The MoE path owns layer-aware expert residency and uses the model's actual router.

**Tech stack:** Python 3.10+, PyTorch, Hugging Face Transformers, Hugging Face Accelerate for low-memory initialization and placement, optional safetensors, pytest, and vLLM for the later multi-user serving path. Quantization backends are optional runtime integrations selected by the user; bitsandbytes is the first candidate, with GPTQ/AWQ evaluated when their model and hardware support is verified. CPU tests use tiny local fixtures or fakes and never download a checkpoint.

**Loading and quantization policy:** The runtime does not impose a quantization format. With the default `quantization="auto"` mode, a model is loaded in the representation supplied by the configured Hugging Face or local checkpoint, including checkpoint-declared quantization when the selected runtime supports it; no runtime quantization conversion is performed. A user may explicitly request a supported quantization backend or format during loading. The request, selected backend, effective dtype, and fallback/error behavior must be visible in capabilities and diagnostics. Unsupported or incompatible user selections fail clearly rather than silently changing the model’s precision. vLLM is introduced only after the single-request reference and paging paths are correct; it provides serving, scheduling, continuous batching, and paged attention, while the validated SparseLLM pager remains responsible for architecture-specific layer-aware expert residency.

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

## Next Implementation Phase — Real Low-VRAM MoE Execution

The universal baseline is complete, but it does not yet satisfy the central low-VRAM objective. The next phase must connect the router, checkpoint tensors, expert cache, and real model modules so that a validated MoE model can execute with only the required expert working set in VRAM.

This phase is architecture-specialized by design. The public application remains model-independent: dense models and unsupported MoE models continue to use the generic Transformers adapter. Only a validated architecture adapter may activate the paging path. Mixtral is the first validation target because its topology and routing contract are well documented; it is not the universal model limit.

### Non-negotiable feasibility statement

A model having 9–10 billion active parameters per token does not mean 9–10 billion parameters can fit in 6 GB VRAM. A 40B or 50B checkpoint still requires approximately 20–25 GB merely for 4-bit weight storage, before shared weights, metadata, KV cache, activations, temporary buffers, and allocator overhead. FP16/BF16 storage is substantially larger.

Therefore, the 6 GB target requires all of the following where the selected model needs them:

- cold expert weights on local SSD or CPU memory;
- only the current layer's required experts admitted to GPU memory;
- shared weights placed by an explicit device policy rather than assumed resident;
- an established quantization backend for weights that cannot fit in FP16/BF16;
- a memory planner that reserves space for KV cache, activations, transfers, and safety margin;
- an actionable rejection when the non-expert resident footprint alone exceeds the available budget.

The implementation must never claim that arbitrary 40B/50B models run in 6 GB until the exact checkpoint, dtype, context length, device, and measured peak memory have been recorded.

## Target Architecture for the Paging Runtime

### 1. Public runtime and adapter boundary

Keep `InferenceEngine`, `ModelRegistry`, `ModelAdapter`, `ModelCapabilities`, and the generic Transformers adapter as the public entrypoint.

Add an optional expert-paging capability contract owned by a specialized adapter. The contract must describe:

- validated architecture identity and adapter version;
- model layer count and expert count;
- top-k routing semantics and router output shape;
- exact layer-aware expert identity;
- shared versus expert tensor groups;
- expert tensor names, shapes, dtypes, and byte sizes;
- whether prefill and decode are supported;
- whether quantized expert execution is supported;
- whether the adapter can preserve reference output within the declared tolerance.

The generic adapter must remain unchanged for unsupported architectures and must report expert paging as unavailable. Capability detection may discover that a model appears to be MoE, but appearance alone cannot enable paging.

### 2. Artifact and checkpoint index layer

Create a checkpoint-aware local artifact layer between the adapter and storage backend.

Responsibilities:

- read the model configuration and checkpoint index without materializing the full model;
- prefer `safetensors` and reject unsafe formats by default;
- map every expert tensor to an exact `(layer_index, expert_index)` identity;
- map shared tensors separately so they are never mistaken for experts;
- record tensor offsets, shapes, dtype, storage file, checksum where available, and byte size;
- support local checkpoint directories and fixed revisions;
- validate that the index matches the loaded configuration before execution;
- detect missing, duplicated, overlapping, or shape-incompatible expert tensors;
- expose a cold-load operation that reads only the requested expert tensors.

The index must be architecture-specific and validated. Generic string matching over arbitrary model state-dict names is not sufficient.

Expected implementation boundary:

- `sparse_llm/storage/checkpoint_index.py` for artifact inspection and tensor-location metadata;
- extensions to `sparse_llm/storage/backend.py` only where the existing storage contract is insufficient;
- no network storage in this phase;
- no full-checkpoint copy into the existing `.pt` expert cache.

### 3. Empty-weight model construction and shared-weight placement

A specialized adapter must construct the model without first materializing every expert on the GPU.

Startup flow:

1. Load and validate configuration and tokenizer.
2. Build the model with empty/meta parameters or an equivalent safe low-memory initialization mechanism.
3. Load shared embeddings, attention, normalization, router, and output tensors according to the memory policy.
4. Replace only validated expert modules with lazy expert handles while preserving module paths and state-dict compatibility.
5. Attach the checkpoint index and pager to the adapter.
6. Run a structural validation before accepting requests.

The adapter must prove that the replacement preserves:

- module registration and parameter naming;
- router inputs and outputs;
- expert input/output shapes;
- routing weights and top-k behavior;
- serialization and state-dict expectations;
- device and dtype semantics.

If shared weights cannot fit in the requested device/CPU policy, startup must fail before generation with a memory breakdown. Expert paging must not be used to hide an impossible resident shared-weight footprint.

### 4. Three-tier expert memory hierarchy

The pager must manage three explicit tiers:

- GPU tier: bounded active expert cache and temporary execution buffers;
- CPU tier: optional resident inactive expert tensors and reusable pinned staging buffers;
- local SSD tier: checkpoint-backed cold tensors accessed through the artifact index.

Every cache entry is identified by `(layer_index, expert_index)` and carries:

- source location and current tier;
- tensor shape, dtype, and byte size;
- load state;
- pin/lease count;
- last-use information;
- checksum or validation state where available.

The existing `ExpertCache` remains the eviction primitive, but the pager becomes responsible for integrating it with actual model modules. Cache admission is byte-based and must include a configurable safety margin. Expert-count limits are diagnostic compatibility controls only, not the memory authority.

Required lifecycle:

1. Resolve a requested expert key.
2. Return the GPU entry if present and valid.
3. Join an existing in-flight load if another request is loading the same key.
4. Otherwise load from CPU or indexed SSD storage.
5. Validate shape, dtype, and byte size.
6. Transfer to GPU under the admission budget.
7. Pin the entry for the duration of expert execution.
8. Release the pin after execution and update measured statistics.
9. Evict only unpinned entries when capacity is needed.
10. Preserve the CPU/SSD source so eviction is recoverable.

If all evictable capacity is exhausted, fail with the requested key, current occupancy, pinned entries, required bytes, and available bytes. Never evict an expert that is in use.

### 5. Actual router-to-expert execution

The specialized adapter must execute the model's own forward path. It must not precompute expert IDs from a second toy router and must not replace expert computation with averaged weights.

For each supported MoE layer:

1. Run the original shared transformation and router.
2. Capture the actual top-k expert indices and routing probabilities.
3. Validate the routing tensor shape and index range.
4. Deduplicate expert keys within the current layer and request.
5. Acquire the required layer-aware expert handles.
6. Group token positions by expert to avoid one tiny operation per token.
7. Execute the original expert network with the original parameters and dtype.
8. Multiply expert outputs by the model-provided routing probabilities.
9. Combine outputs using the architecture's original dispatch semantics.
10. Release all expert leases before leaving the layer.

The first implementation scope is batch size one with one request at a time. It must support both prompt prefill and autoregressive decode. Decode must reuse the model's KV cache and must not reload all prompt tokens on every step.

The implementation must record actual routed keys, cache hits, cache misses, load source, load duration, transfer duration, and peak memory. These values must come from the integrated pager, not from synthetic counters.

### 6. Memory planner and admission policy

Before loading a checkpoint, calculate a conservative budget containing:

- resident shared model weights;
- currently admitted expert bytes;
- temporary expert inputs and outputs;
- KV cache for the configured context and generated-token limit;
- tokenizer/model runtime overhead where measurable;
- CUDA allocator reserve and safety margin.

The planner must expose:

- total device capacity;
- reserved non-expert bytes;
- available expert budget;
- configured maximum context and generation length;
- estimated bytes for one expert and the largest routed group;
- whether the request can be admitted.

The planner must reject configurations where one required expert or one required routed group cannot fit. It must not silently reduce context length, generation length, or model precision. Any automatic adjustment must be explicit in the request configuration and visible in diagnostics.

### 7. Established quantization integration

Quantization is required for many 40B/50B low-VRAM deployments, but it must not be implemented with a custom speculative kernel in this phase.

After the FP16/BF16 paging path is correct on a small structural fixture:

- integrate an established, version-compatible quantization backend;
- support quantized shared and/or expert tensors only where the backend supports the architecture;
- account for scales, metadata, dequantization buffers, and temporary workspace;
- compare logits and generated tokens against the unquantized reference;
- measure peak memory, load time, decode latency, and numerical error;
- fail clearly when the requested backend or dtype is unavailable.

No fixed compression ratio, perplexity loss, throughput, or VRAM number may be written into documentation without a measurement for the exact model and hardware.

## End-to-End Request Flow

### Startup flow

1. The user supplies a model ID or local checkpoint, device policy, dtype, cache byte budget, context limit, and generation limit.
2. The registry selects a specialized adapter only if the architecture validator passes; otherwise it selects the generic adapter.
3. The specialized adapter reads configuration, tokenizer metadata, and the checkpoint index.
4. The artifact layer validates all shared and expert tensor locations.
5. The memory planner calculates the resident and dynamic budget.
6. The model is initialized with empty expert parameters and shared weights placed according to policy.
7. Lazy expert handles are installed and structurally validated.
8. The runtime reports capabilities, memory plan, and paging availability before accepting inference.

### Prefill flow

1. Tokenize the prompt with the model's tokenizer.
2. Allocate only the prompt KV-cache and activation budget admitted by the planner.
3. Run the real model forward pass layer by layer.
4. At every MoE layer, use the model's router output to determine actual expert keys.
5. Load, validate, pin, execute, and release only the experts required by that layer.
6. Store the resulting KV cache for decode.
7. Record prefill latency, routed experts, cache behavior, loads, and peak memory.

### Decode flow

1. Feed only the next token and the existing KV cache into the model.
2. Run the same real router and expert dispatch path as prefill.
3. Reuse resident experts when possible.
4. Load missing experts through the pager and evict only unpinned entries.
5. Produce the next-token logits using the original output head and stopping semantics.
6. Continue until EOS or the explicit generation limit.
7. Record per-token and aggregate decode metrics.

### Cache-miss flow

1. The router requests a layer-aware key.
2. The pager checks the GPU cache.
3. If absent, it checks CPU resident storage.
4. If absent, it reads only the indexed tensor range from local SSD.
5. It validates the tensor and stages it using the configured dtype/device policy.
6. It admits the tensor only if the memory planner allows it.
7. It pins the entry, executes the expert, and releases the pin.
8. It records the source tier, load time, transfer time, and eviction result.

### Failure flow

The runtime must fail before or during generation with an actionable structured diagnostic for:

- unsupported architecture or unverifiable expert topology;
- missing checkpoint index or tensor location;
- tensor shape, dtype, checksum, or routing mismatch;
- insufficient shared-weight or expert-cache budget;
- pinned-cache deadlock or exhausted admission capacity;
- unavailable CUDA, quantization backend, or required dtype;
- corrupted or unsafe checkpoint files;
- KV-cache/context budget exhaustion.

A failed request must release all expert leases and temporary resources before returning the error.

## Implementation Work Packages for Coding Agents

### Work Package A — Paging contracts and model capability validation

Update the model capability and adapter boundary with an explicit, optional paging contract. Add validation results and failure reasons. Keep generic fallback behavior unchanged. Add tests proving that dense models and unsupported MoE structures cannot enable paging.

### Work Package B — Checkpoint artifact index

Implement indexed local checkpoint inspection with safe tensor formats, exact expert tensor mapping, shape/dtype/size metadata, and validation diagnostics. Add temporary local fixtures for shared tensors and multiple layers with repeated expert IDs. Do not download a real model in unit tests.

### Work Package C — Lazy expert module integration

Implement the first concrete architecture adapter using a tiny structural fixture that mirrors the selected MoE topology. Construct empty expert modules, attach lazy handles, preserve state-dict paths, and verify that the original router and expert outputs are still used.

### Work Package D — Pager and memory planner integration

Connect the layer-aware cache and storage index to actual expert handles. Implement byte admission, pin leases, in-flight load sharing, CPU/SSD fallback, memory accounting, and failure cleanup. Add tests for eviction safety, concurrent requests for one expert, oversized experts, and pinned capacity exhaustion.

### Work Package E — Real prefill and decode conformance

Run the specialized adapter through real prompt tokenization, prefill, KV-cache reuse, and autoregressive decode on the tiny fixture. Compare logits, selected expert keys, routing probabilities, generated token IDs, and final text with a reference non-paged Transformers execution.

### Work Package F — Mixtral validation adapter

Only after the fixture passes, implement the first checkpoint-specific adapter for Mixtral. Validate configuration, module paths, router semantics, tensor names, expert shapes, and state-dict compatibility against a documented local checkpoint. Do not register it globally until the conformance suite passes.

### Work Package G — Quantized low-VRAM path

Integrate an established quantization backend after correctness of the unquantized pager. The backend and format are selected by the user rather than by a hard-coded project policy. Measure exact memory use and numerical behavior, and report the effective representation. If the requested backend cannot load the selected model, fail clearly unless the user explicitly selected an allowed fallback mode; never silently fall back to unquantized weights.

### Work Package H — Operational benchmark and admission report

Add a reproducible benchmark command for one request at a time. Report model revision, hardware, device, dtype, context length, generation length, shared bytes, expert-cache bytes, KV-cache bytes, prefill latency, decode latency, throughput, cache hits/misses, expert load times, and peak VRAM. Do not publish default performance targets.

## Testing and Conformance Gates

### Gate 1 — Structural safety

The adapter must reject an architecture when any required module path, tensor mapping, router shape, expert shape, or state-dict relationship is unverifiable. Generic fallback must remain available.

### Gate 2 — Pager correctness

For a tiny deterministic fixture, paged execution must match reference execution for:

- tokenized input IDs;
- router-selected expert keys per layer;
- routing probabilities;
- layer logits and final logits within a documented tolerance;
- generated token IDs and stopping behavior;
- KV-cache shape and reuse;
- final decoded text.

### Gate 3 — Residency correctness

Tests must prove that:

- inactive experts are not placed on GPU;
- only requested layer-aware keys are admitted;
- repeated expert IDs across layers do not collide;
- pinned entries remain resident during execution;
- concurrent requests share one in-flight load;
- eviction never removes a leased expert;
- all leases are released after success and failure.

### Gate 4 — Budget correctness

Tests must prove that the planner accounts for shared weights, experts, KV cache, activations, overhead, and safety margin. It must reject an oversized expert, impossible shared footprint, and insufficient routed-group budget with actionable diagnostics.

### Gate 5 — Real checkpoint smoke test

When a documented local checkpoint is available, run a short smoke test with a fixed revision or local path. Record exact hardware and measurements. If the checkpoint is unavailable, mark the test environment-limited; do not substitute synthetic benchmark values.

### Gate 6 — Regression and compatibility

Run the complete CPU suite, compile checks, CLI help, generic dense-model fixture, unsupported-MoE fallback fixture, storage safety tests, and specialized fixture conformance tests. The generic path must remain independent of the specialized pager.

## Performance and Scalability Sequence

The implementation order must prioritize correctness and bandwidth efficiency:

1. synchronous single-request paging on a tiny fixture;
2. indexed local SSD reads and CPU residency;
3. byte-based cache admission and lease-safe eviction;
4. real Mixtral structural and checkpoint conformance;
5. established quantization for the exact target hardware;
6. pinned CPU staging and one asynchronous transfer path;
7. CUDA streams and readiness events with measured overlap;
8. immediate next-layer prefetch based on observed routing;
9. shared cache and continuous batching;
10. paged KV storage and request scheduling;
11. multi-worker or multi-GPU serving only after single-worker measurements justify it.

Do not add router prediction, speculative prefetch, network storage, batching, or multi-GPU execution before the synchronous single-request pager is correct and measurable.

## Next-Phase Global Constraints

- The generic model flow remains universal and must not regress.
- Expert paging is enabled only by a validated architecture adapter.
- No model-name allow-list may determine compatibility or paging eligibility.
- The real model router, routing probabilities, expert modules, and stopping semantics are authoritative.
- No averaged expert weights, synthetic router decisions, fabricated text, or random model inputs are permitted.
- All expert identities are layer-aware.
- Local checkpoint reads must be indexed and safe; do not deserialize arbitrary executable objects.
- GPU memory is governed by bytes plus reserve, not expert count alone.
- Every in-flight load, lease, transfer, and temporary allocation must have failure cleanup.
- Any 6 GB result must be measured on the exact model, revision, dtype, context, generation length, and hardware.
- Established quantization may be used only after unquantized paging correctness is proven; custom INT4 kernels remain out of scope.
- CPU tests remain offline and checkpoint-free.

## Next-Phase Definition of Done

This phase is complete only when:

- a validated MoE adapter executes the original model router and expert computation;
- a real prompt reaches tokenizer, prefill, paged expert execution, KV-cache decode, and generated text;
- only the required layer-aware expert working set is admitted to GPU memory;
- the cache, storage index, module handles, and memory planner are connected in the real forward path;
- paged and reference execution agree within documented numerical tolerances;
- failures release leases and report actionable memory or artifact diagnostics;
- a documented Mixtral checkpoint passes the short smoke test, or the environmental limitation is explicitly recorded;
- no unsupported architecture is advertised as paged;
- measured results show whether the selected 40B/50B target is feasible under the requested 6 GB policy.

## Deferred After the Paging Phase

Only after the definition of done above is satisfied should the project add continuous batching, paged KV storage, router locality analysis, observed-next-use prefetch, multi-worker serving, network-backed cold storage, or multi-GPU execution. Each addition requires a reference-output comparison, failure-isolation tests, and measurements of latency, bandwidth, throughput, memory, and reliability.
