# Stages 3, 5–8: Complete Inference Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) to implement this plan task-by-task with review checkpoints.

**Goal:** Build a production-grade local inference system with quantization, asynchronous expert paging, continuous batching, and reproducible benchmarking on a single consumer GPU (6–8 GB VRAM).

**Architecture:** 
- Stage 3: Quantize expert tensors at cold-storage boundary; dequantize on-demand during execution
- Stage 5: Overlap expert transfers with compute via pinned buffers, CUDA streams, and readiness events
- Stage 6: Schedule multiple concurrent requests without blocking; page KV cache separately from experts
- Stage 7: Enable multi-GPU expert parallelism and network-backed cold storage (deferred if single-GPU baseline unstable)
- Stage 8: Measure and validate all optimizations against reproducible baselines

**Tech Stack:** PyTorch, CUDA (optional), Hugging Face Transformers, existing pager/cache/storage

**Spec:** `docs/REMAINING_INTEGRATION_ARCHITECTURE.md` (Stages 3, 5–8)

## Global Constraints

- Minimal: one quantization backend (INT8 or FP8); defer mixed-precision per-layer tuning
- Backward-compatible: generic adapters unaffected; Mixtral baseline remains functional
- All 89 prior tests must pass (no regressions)
- Single-device first: prove each stage on one GPU before adding multi-GPU complexity
- Synchronous reference baseline always available: async optimizations validated against sync path

---

## File Structure

| File | Responsibility |
|------|---|
| `sparse_llm/models/quantization.py` | Quantization policy, format detection, scale/zero-point extraction |
| `sparse_llm/inference/quantized_loader.py` | Load quantized experts from checkpoint; handle format conversion |
| `sparse_llm/inference/async_pager.py` | CUDA stream management, readiness events, pinned buffers, overlap logic |
| `sparse_llm/inference/scheduler.py` | Request admission, sequence state, KV page allocation, decode batching |
| `sparse_llm/inference/continuous_batch.py` | Batch scheduler, multi-request generation, per-request state isolation |
| `sparse_llm/models/expert_parallelism.py` | Multi-GPU expert distribution (Stage 7) |
| `sparse_llm/storage/network_cache.py` | Network-backed expert storage, checksums (Stage 7) |
| `sparse_llm/benchmark/runner.py` | Reproducible benchmark command, result schema, regression detection |
| Tests: `sparse_llm/tests/test_quantization.py`, `test_async_pager.py`, `test_scheduler.py`, `test_benchmark.py` | Per-stage validation |

---

## Task Breakdown

### Stage 3: Quantization Integration (2 tasks)

**Task 1: Quantization Policy and Format Detection**
- Detect checkpoint quantization (bfloat16, int8, fp8, etc.)
- Define QuantizationPolicy (format, group_size, scale_type)
- No actual loading yet; policy only

**Task 2: Quantized Expert Loading**
- Load experts with quantization enabled
- Dequantize on-demand in execution kernel
- Numerical validation against non-quantized baseline (fixed prompts)
- Measure memory and latency

### Stage 5: Async Transfers (3 tasks)

**Task 3: CUDA Streams and Readiness Events**
- Pinned CPU buffers for transfers
- Dedicated transfer stream
- Readiness events after transfer complete
- Compute stream waits only for selected experts

**Task 4: Asynchronous Expert Prefetch**
- Predict next experts from router output
- Schedule prefetch on transfer stream
- Compare async vs sync latency on fixed workload

**Task 5: Backpressure and Transfer Scheduling**
- Limit outstanding transfers (prevent OOM)
- Handle prefetch misses gracefully
- Fall back to sync if async queue full

### Stage 6: Continuous Batching (3 tasks)

**Task 6: Request Scheduler and Sequence State**
- Define request lifecycle (admission → generation → completion)
- Track per-request KV cache, sampling state, stopping criteria
- Isolate request state to prevent corruption

**Task 7: Paged KV Cache**
- Allocate KV pages per request
- Reclaim pages on request completion
- Track KV page occupancy separately from expert cache

**Task 8: Decode Batching and Token Scheduling**
- Group compatible decode work (variable lengths)
- Per-request sampling and stopping logic
- Measure throughput, tail latency under mixed request workloads

### Stage 7: Multi-Worker and Multi-GPU (2 tasks)

**Task 9: Expert Parallelism (Single Machine)**
- Distribute experts across GPUs (replicated or sharded)
- Define cross-device transfer points
- Measure speedup on multi-GPU baseline

**Task 10: Network-Backed Storage**
- Immutable expert-page identities (checkpoint version, layer, expert, format)
- Checksum validation before execution
- Local-disk → remote-fetch fallback path

### Stage 8: Benchmarking (2 tasks)

**Task 11: Benchmark Command and Result Schema**
- Reproducible benchmark CLI (model, device, policy, cache, batch settings)
- Machine-readable results + human summary
- Exact prompts and runtime state in artifact

**Task 12: Regression Detection and Reporting**
- Baseline establishment (no-quant, no-async, single-request)
- Per-stage comparison (quant vs baseline, async vs sync, etc.)
- Regression threshold definitions (correctness, memory, latency, throughput)

---

## Execution Path & Dependencies

```
Task 1 (Quantization policy) → Task 2 (Quantized loading)
     ↓
Task 3 (CUDA streams) → Task 4 (Async prefetch) → Task 5 (Backpressure)
     ↓
Task 6 (Scheduler) → Task 7 (Paged KV) → Task 8 (Decode batching)
     ↓
Task 9 (Expert parallelism) ↔ Task 10 (Network storage)
     ↓
Task 11 (Benchmark CLI) → Task 12 (Regression detection)
```

**Linear dependency:** Each task builds on previous. Can parallelize Stage 3 and Stage 5 once Task 2 and Task 3 complete (independent code paths).

---

## Scope Note

This plan covers Stages 3, 5–8 in full detail. **Stage 7 (multi-worker/multi-GPU) is execution-gated:** if single-device async baseline shows no improvement or regressions, Stage 7 is deferred. This prevents premature optimization.

**Estimated scope:** 12 tasks, ~200–300 lines per task (implementation + tests), 6–10 hours subagent execution time.

---

