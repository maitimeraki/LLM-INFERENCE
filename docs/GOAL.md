# SparseLLM: Updated Project Goal

**Status:** Active implementation goal  
**Target runtime:** PyTorch + Hugging Face Transformers  
**First supported model:** Mixtral 8x7B-Instruct-v0.1 (or a compatible local Mixtral checkpoint)  
**Hardware target:** Single consumer GPU, starting at 6–8 GB VRAM

## Mission

Build a measurable local inference prototype that runs one real Mixture-of-Experts checkpoint on a small GPU by keeping non-expert model components resident where possible and moving expert weights between CPU/SSD storage and a bounded GPU cache.

The first goal is **correctness and an honest baseline**, not a claim of arbitrary 100B+ model support or sub-2 ms token latency.

## First Deliverable: Real Vertical Slice

The initial implementation must provide this path:

```text
prompt
  → tokenizer
  → supported Mixtral checkpoint
  → real prefill and autoregressive decoding
  → generated text and metrics
```

The baseline must:

1. Load a real Hugging Face-compatible Mixtral checkpoint, locally or from the configured model source.
2. Use the model's actual tokenizer, router, tensor shapes, expert count, and top-k routing behavior.
3. Keep expert cache identity layer-aware: `(layer_index, expert_index)`.
4. Store inactive expert tensors on CPU or in a local memory-mapped/checkpoint-backed store.
5. Use a bounded GPU cache with simple LRU eviction before adding predictive scoring.
6. Return generated tokens/text rather than fabricated demo output.
7. Record prefill latency, decode latency, throughput, peak VRAM, cache hits/misses, and expert load time.
8. Compare output behavior against a reference Transformers path on a small fixture or supported reduced model.

## Implementation Order

### Milestone 1 — Correctness and baseline

- Consolidate the runtime around one inference entrypoint.
- Add a model adapter for one supported Mixtral checkpoint format.
- Implement real tokenizer and generation flow.
- Replace random tensors and placeholder output.
- Add focused tests that run without a GPU or downloaded checkpoint.

### Milestone 2 — Expert residency

- Add layer-aware expert handles and storage metadata.
- Move only requested experts into a bounded GPU cache.
- Deduplicate in-flight loads and protect experts currently used by a forward pass.
- Measure synchronous cache-miss cost and memory usage.

### Milestone 3 — Proven memory reduction

- Integrate an established quantization backend where available.
- Validate layerwise numerical error, perplexity/generation quality, peak memory, and load time.
- Account for resident attention/shared weights, KV cache, activations, allocator overhead, and safety margin.

### Milestone 4 — Measured overlap

- Add pinned CPU staging buffers, one asynchronous transfer worker, CUDA streams, and readiness events.
- Start with immediate prefetch based on observed next-use information.
- Compare against LRU-only and reactive baselines before adding speculation.

### Milestone 5 — Serving improvements

- Add request scheduling, shared cache use, and paged KV storage only after single-request decode is correct.
- Evaluate co-activation and router prediction per layer as optional optimizations.

## Explicit Non-Goals for the First Milestone

The first milestone will not promise or require:

- arbitrary 100B+ model compatibility;
- a nonexistent or unverified model name;
- 90% cache hit rate, 10–50 tokens/sec, or sub-2 ms decode latency;
- custom CUDA INT4 kernels;
- Markov/domain prediction as a correctness dependency;
- three-level speculative prefetch;
- network expert storage;
- multi-GPU execution;
- batch sizes 1–16 before single-request correctness is measured.

## Success Criteria

The vertical slice is successful when:

- a documented Mixtral checkpoint can generate text through the real tokenizer/model path;
- the same request is deterministic under fixed generation settings;
- cache hits and misses are observable and reproducible;
- GPU memory stays below the configured budget on the test hardware, or the system fails with a clear diagnostic;
- benchmark output contains measured values rather than static performance claims;
- all CPU-only regression tests pass without requiring model downloads.

Every optimization added afterward must show an improvement against the baseline without changing model correctness beyond a documented tolerance.
