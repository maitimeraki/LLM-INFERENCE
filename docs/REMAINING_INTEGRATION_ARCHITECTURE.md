# Remaining Integration Architecture

## Purpose

This document defines the implementation path for components that are partially integrated or not yet integrated into the local inference runtime. It intentionally excludes components already connected to the core execution path.

The target is a verified single-device inference path for one real mixture-of-experts checkpoint before adding distributed or throughput-oriented features.

## Current Boundary

The runtime already contains adapter, routing, expert paging, cache, metrics, memory-accounting, scheduling, storage, and prefetch primitives. The remaining work is to make those primitives one coherent inference pipeline and prove it against a real checkpoint.

The required dependency order is:

1. Test and runtime environment correctness
2. Global architecture adapter registration
3. Memory-safe model loading and shared-weight placement
4. Quantized expert materialization
5. Generation-loop integration for paging and prefetch
6. Paged KV cache and continuous batching
7. Network, multi-worker, and multi-GPU execution
8. Reproducible performance tooling

Later stages must not begin until the prior stage has an end-to-end test and measurable acceptance criteria.

## Stage 0: Establish a Green, Reproducible Baseline

### Goal

Make test and runtime commands execute from a documented environment, then resolve the existing adapter-contract failure before extending the system.

### Integration steps

1. Define one supported installation path for development and tests.
2. Ensure the package is importable when the test suite starts.
3. Run the complete unit suite from that environment.
4. Resolve the universal-adapter tensor-mapper contract failure.
5. Record the exact test command in project documentation and future benchmark tooling.

### Exit criteria

- The complete unit suite passes from a clean checkout.
- A developer can run tests without manually changing import paths.
- Adapter discovery tests distinguish supported architectures from generic fallback behavior.

## Stage 1: Globally Register a Real Architecture Adapter

### Goal

Turn the registry from an extension point into a usable default model-selection path.

### Target flow

Checkpoint configuration -> architecture detection -> registered architecture adapter -> model topology extraction -> router extraction -> expert tensor mapping -> pager metadata -> inference engine.

### Integration steps

1. Select one local, supported MoE checkpoint family as the first reference architecture.
2. Implement or finalize its model-specific adapter using the checkpoint’s real configuration and tensor names.
3. Register that adapter in the process-wide default model registry.
4. Keep the generic fallback only for architectures that are deliberately unsupported.
5. Make registry precedence explicit when two adapters could recognize the same configuration.
6. Add a fixture based on a real checkpoint configuration, without requiring a model download during ordinary unit tests.
7. Add an optional integration test that uses a locally available checkpoint.

### Exit criteria

- Default registry lookup selects the model-specific adapter for the reference architecture.
- Generic fallback is not selected for that architecture.
- The adapter extracts real router, layer, expert, and shared-tensor metadata.
- A locally available checkpoint can be detected without manual adapter registration.

## Stage 2: Memory-Safe Loading and Shared-Weight Placement

### Goal

Load the model without first materializing the full checkpoint in accelerator memory, while preserving shared and tied weights exactly once.

### Target placement policy

- Shared model weights remain resident according to an explicit device and offload policy.
- Router weights remain resident with the layers that require routing decisions.
- Expert weights are represented as pageable materialization units.
- Tied weights share a single physical placement and are never copied solely because they have multiple names.
- The pager owns only expert tensors. It must not page shared tensors, router tensors, or tied aliases independently.

### Integration steps

1. Define one placement-policy object used by loading, adapter extraction, pager construction, and generation.
2. Separate checkpoint tensor names into shared, router, expert, and tied-alias groups before placement begins.
3. Add an empty or meta-parameter initialization path, or verify an equivalent Transformers-supported loading path that guarantees the same memory bound.
4. Materialize shared and router weights directly to their policy-selected location.
5. Build expert pager records without materializing all experts on the accelerator.
6. Resolve tied aliases to their canonical shared tensor before any transfer or paging decision.
7. Expose an inspectable placement report containing tensor category, canonical identity, initial location, and paging ownership.
8. Reject configurations that classify one tensor as both shared and pageable expert data.

### Exit criteria

- Peak loading memory is bounded by shared weights, active expert capacity, and transfer workspace rather than total expert size.
- Tied weights have one canonical placement.
- Shared and router tensors are never submitted to expert eviction.
- A placement report can explain every checkpoint tensor’s runtime ownership.

## Stage 3: Integrate Quantization at Expert Materialization

### Goal

Make quantization a real loading and execution policy rather than a standalone utility.

### Target flow

Checkpoint expert tensor -> quantization policy -> cold representation -> pager transfer -> execution representation -> expert invocation.

### Integration steps

1. Define the supported first quantization format and its scope.
2. Keep shared weights and router weights at an explicitly chosen precision unless the selected architecture proves another policy safe.
3. Quantize expert tensors as they enter cold storage or a checkpoint index.
4. Store all scale, zero-point, group-size, and layout metadata with the pageable expert record.
5. Define whether dequantization occurs on host memory, accelerator memory, or inside an execution kernel.
6. Make pager capacity accounting use physical transferred bytes and execution workspace, not only original tensor size.
7. Select an execution backend that can consume the chosen representation, or temporarily dequantize into a bounded active-expert workspace.
8. Add numerical comparison tests against the non-quantized reference model for fixed prompts and routing choices.
9. Add memory and latency measurements for quantized versus non-quantized expert residency.

### Exit criteria

- Expert records carry the metadata required to execute quantized weights.
- The normal generation path actually uses quantized expert data when enabled.
- Numerical drift stays within a documented tolerance.
- Memory savings and any latency cost are measured per supported architecture.

## Stage 4: Connect Paging, Router Prediction, and Prefetch to Generation

### Goal

Make expert selection drive concrete page residency decisions during prefill and decode.

### Target flow

Token batch -> layer router -> selected experts -> pager residency check -> required transfers -> expert execution -> next-layer prefetch candidate -> next token.

### Integration steps

1. Define the generation-loop ownership boundary for router decisions, pager requests, and expert execution.
2. For each MoE layer, obtain the actual selected expert identifiers from the architecture adapter’s router path.
3. Request selected experts from the pager before expert execution.
4. Block only when a selected expert is not ready; never execute an expert from stale or partially transferred data.
5. Use layer-aware cache keys consistently across router output, pager records, metrics, and prefetch candidates.
6. Add immediate next-layer prefetch only after synchronous correctness is proven.
7. Attach router prediction to prefetch as an optimization hint, not as a correctness dependency.
8. Fall back to on-demand loading when prediction is unavailable or wrong.
9. Collect per-layer page hits, misses, transfer time, waiting time, prediction accuracy, and unused-prefetch waste.

### Exit criteria

- A real routed inference request uses pageable experts, not all-resident expert modules.
- Each execution uses exactly the router-selected expert set.
- Wrong prefetch predictions cannot alter model output or crash generation.
- Metrics identify whether paging helps or harms each supported workload.

## Stage 5: Add Asynchronous Transfers and CUDA Readiness

### Goal

Overlap safe expert movement with compute after the synchronous generation path is correct.

### Target flow

Router decision -> transfer scheduled on transfer stream -> readiness event -> compute stream waits when required -> expert execution.

### Integration steps

1. Preserve the synchronous paging path as the correctness reference.
2. Define a transfer ownership model for buffers, streams, and readiness events.
3. Schedule eligible page-ins on a dedicated transfer stream.
4. Record a readiness event only after the transferred expert is safe to execute.
5. Make the compute stream wait for readiness only for selected experts.
6. Prevent eviction or reuse of a transfer buffer until all dependent execution has completed.
7. Add backpressure limits for outstanding transfers and prefetched-but-unused expert pages.
8. Compare asynchronous and synchronous execution using the same prompt set and cache state.

### Exit criteria

- No expert is consumed before its readiness event completes.
- Buffer ownership remains correct under concurrent page-in and eviction.
- Output matches the synchronous reference.
- Measured decode latency improves for at least one defined workload before the feature is enabled by default.

## Stage 6: Integrate Paged KV Cache and Continuous Batching

### Goal

Support multiple active requests without treating all KV state and expert activity as one monolithic batch.

### Target architecture

Request admission -> batch scheduler -> sequence state -> paged KV allocation -> per-layer routed execution -> token emission -> request completion and reclamation.

### Integration steps

1. Define the request lifecycle and the owner of sequence state.
2. Connect the scheduler to the public inference entry point.
3. Represent each sequence’s KV state as pages with explicit allocation and reclamation.
4. Ensure KV page ownership is separate from expert page ownership.
5. Group compatible decode work without requiring all requests to have equal prompt length or generation length.
6. Keep each request’s sampling state, stopping criteria, and error state isolated.
7. Set hard capacity limits for active sequences, KV pages, expert pages, and queued transfers.
8. Define overload behavior, including queueing, rejection, or bounded waiting.
9. Measure throughput, token latency, tail latency, and cache pressure under mixed request lengths.

### Exit criteria

- Multiple active requests make progress without cross-request KV corruption.
- Ending one request returns only its own KV pages.
- Expert paging and KV paging are independently observable and capacity-bounded.
- Continuous batching improves throughput without violating per-request output correctness.

## Stage 7: Network Cold Storage, Multi-Worker, and Multi-GPU Execution

### Goal

Scale only after the single-device runtime has stable semantics and performance measurements.

### Network cold storage

1. Define immutable expert-page identities using checkpoint version, layer, expert, tensor layout, and quantization format.
2. Add checksum validation before a fetched page becomes executable.
3. Define local-disk cache behavior, remote fetch behavior, retry boundaries, and error propagation.
4. Ensure corrupted or mismatched pages are rejected rather than silently executed.
5. Treat network storage as a cold-source replacement behind the existing expert-page abstraction.

### Multi-worker serving

1. Define worker ownership of models, accelerator devices, caches, and request queues.
2. Keep request routing outside model execution so worker restart and admission control are explicit.
3. Decide whether each worker has independent expert cache state or a coordinated host-level cache.
4. Define health reporting, worker draining, overload behavior, and request cancellation.
5. Add process-level integration tests before exposing a network service interface.

### Multi-GPU execution

1. Choose one initial parallelism model: replicated workers, expert parallelism, tensor parallelism, or pipeline parallelism.
2. Do not combine parallelism models in the first implementation.
3. Define canonical ownership of each shared tensor, router tensor, expert page, and KV page across devices.
4. Define cross-device transfer and synchronization points before adding communication libraries.
5. Measure one-device baseline first, then require a documented scaling benefit for each additional device.

### Exit criteria

- Network page identity and integrity are verified before execution.
- Worker and device ownership are explicit and testable.
- Multi-device behavior preserves output correctness and shows measurable benefit versus a single-device baseline.

## Stage 8: Benchmark and Regression Tooling

### Goal

Make architecture and performance claims reproducible.

### Benchmark command requirements

The benchmark interface must accept:

- Model checkpoint and architecture adapter
- Device and placement policy
- Quantization policy
- Expert cache capacity
- Prefill and decode prompt sets
- Batch and concurrency settings
- Warm-up and measured iteration counts
- Optional locally available checkpoint validation mode

### Required report sections

- Environment and software versions
- Checkpoint and architecture adapter identity
- Placement policy and quantization policy
- Peak host and accelerator memory
- Prefill latency
- Decode latency per token
- End-to-end request latency
- Throughput
- Expert page hits, misses, transfer bytes, and wait time
- Prefetch prediction accuracy and wasted work
- KV cache pressure
- Batch queue time and tail latency when batching is enabled
- Numerical comparison result when quantization is enabled

### Integration steps

1. Convert existing metrics and memory data into one stable report schema.
2. Add a repeatable benchmark command that writes machine-readable results and a human-readable summary.
3. Include the exact input prompts and runtime policy in every result artifact.
4. Establish a single-device non-quantized reference baseline for the first supported architecture.
5. Add comparison runs for paging, quantization, and asynchronous prefetch.
6. Define regression thresholds separately for correctness, memory, prefill latency, decode latency, and throughput.
7. Run the smallest stable subset in automated tests and reserve hardware-specific benchmarks for dedicated environments.

### Exit criteria

- A result can be reproduced from the recorded command, model identity, and policy settings.
- Every optimization has before-and-after measurements against the same reference workload.
- Performance regressions are detected separately from unit-test failures.

## Final Integration Gate

The system is ready to advance beyond the current phase only when one real supported checkpoint has all of the following:

1. Automatic architecture selection through the default registry.
2. Memory-safe loading with explicit shared-weight ownership.
3. Pageable expert execution driven by the model’s real router.
4. Correct generation on fixed test prompts.
5. Quantized expert execution when the policy is enabled.
6. Measured memory, latency, and paging behavior.
7. A reproducible benchmark result.

Multi-worker, network-backed, and multi-GPU implementations remain deferred until this gate passes. They are scale features, not substitutes for a proven single-device inference path.
