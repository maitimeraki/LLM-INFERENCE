# SparseLLM Inference Engine — Verified Engineering Assessment

> Re-verified against source on 2026-09-11. Previous version contained several
> stale claims that mischaracterized the current codebase. This version marks each
> claim as VERIFIED / PARTIAL / DISPUTED / FALSE based on a full read of the
> relevant files.

---

## Verdict

The pipeline runs end-to-end and emits tokens. The core architecture — norms, residuals,
Q/K/V projections, full-layer loops, three-tier cache, safetensors index — is **correctly
implemented** in the current source. Performance is the primary concern: the hot path
still burns disk I/O on every expert load because `get_or_load` bypasses the three-tier
cache, and `safe_open` is called per-key instead of once per file.

---

## Claims vs. Reality (Updated)

| Earlier claim | Status | Evidence |
|---|---|---|
| "Decode loops all layers" | **VERIFIED** | `_forward_one_token` (moe:637–644) iterates all layers; `_decode_phase` and `_decode_phase_streaming` both call it. No `layer_id = 0` hardcode. |
| "KV cache is zero-copy" | **VERIFIED** | `decode` writes in-place to pre-allocated buffer (attn:265–266); no copies. |
| "Three-tier cache checks GPU/CPU before storage" | **VERIFIED** | `get()` checks GPU → CPU → storage in order (cache:100–133); promotion logic is present. |
| "Three-tier cache is used on hot path" | **DISPUTED** — `get_or_load` is `return loader()` (cache:162), so callers that pass a loader bypass the tier-aware `get()`. HOWEVER, `_create_expert_loader` explicitly calls `get()` before falling back to storage (moe:1123–1132), so the hot path IS tier-aware. See note below. |
| "GPU cache hit <5ms, 100x faster" | **PARTIAL** — The tier logic exists but is only reached via `_create_expert_loader`'s explicit `get()` call. Callers that pass a loader directly to `ExpertProcessor._get_expert` go through `get_or_load` → `return loader()`, bypassing the cache. See note below. |
| "Cache statistics now accurate" | **VERIFIED** — `cache_hits`/`cache_misses` counters in `_get_expert` are conditional (expert_processor:134,140); unconditional increment was removed. |
| "5–8 tok/s achieved" | **DISPUTED** — No benchmark in-repo reproduces this. Claim stands as aspirational. |
| "Backward compatible, no breaking changes" | **PARTIAL** — `CustomMoEAdapter` constructs `ExpertCache(max_experts=..., max_bytes=...)` (old signature) but the current `ExpertCache.__init__` uses `gpu_slots, cpu_slots, expert_bytes, storage_loader`. This is a genuine signature mismatch. |

### Note on `get_or_load` bypass

`ExpertCache.get_or_load` is literally `return loader()` (cache:162). Any code path that
passes a loader directly to `get_or_load` bypasses the three-tier cache entirely.
`_create_expert_loader` (moe:1103–1158) works around this by calling `get()` explicitly
before the fallback to `_load_expert_from_storage`. However, `ExpertProcessor._get_expert`
passes the loader to `get_or_load`, going straight to storage. The three-tier cache is
real but its tier-aware `get()` is only reached via `_create_expert_loader`, not via the
processor's hot path.

Fix: change `get_or_load` to actually use `get()` first, or remove the pass-through
`get_or_load` layer entirely and have all callers go through `get()`.

---

## Layer-by-Layer Trace (Updated)

| # | Stage | Required | Actual | Status |
|---|---|---|---|---|
| 0 | Embedding | `embed_tokens` lookup | Real weights, real lookup (`_embed_tokens`, moe:1042–1073) | ✅ correct |
| 1 | Pre-attention norm | RMSNorm | Loaded in `_load_attention_and_norm_weights`, applied in `_apply_attention_layer` via `_get_norm` → `_as_norm` (moe:476–558, 560–584) | ✅ correct |
| 2 | Q/K/V projection | `x·Wq`, `x·Wk`, `x·Wv` | `load_layer_weights` binds projections per layer; `_project_qkv` applies them (attn:110–134, 299–320). Identity fallback with warning when keys absent. | ✅ correct (with fallback) |
| 3 | Attention scores + softmax | present | SDPA or `_manual_attention` (attn:205–213, 276–284, 358–400) | ✅ correct |
| 4 | Output projection `Wo` | `attn·Wo` | `_apply_out_proj` (attn:322–327), called in prefill/decode | ✅ correct |
| 5 | Residual after attention | `h = h + attn(h)` | `_apply_attention_layer` returns `h + attn_out` (moe:584) | ✅ correct |
| 6 | Post-attention norm | RMSNorm | Loaded alongside input norm, applied in `_apply_moe_layer` (moe:485–491, 588–589) | ✅ correct |
| 7 | Router / gate | `softmax(x·Wg)` | Real gate weights loaded and per-layer swapped (moe:438–463, 730–742, 815–823, 592–602) | ✅ correct |
| 8 | Expert FFN | top-k experts | `ExpertFFN` with real weights loaded via `_create_expert_loader` (expert_processor:16–50) | ✅ weight content correct; loading bypasses cache on processor path |
| 9 | Expert combine | `Σ wᵢ·Eᵢ(x)` | Weighted sum in `process_single` (expert_processor:331–339) | ✅ correct |
| 10 | Residual after MoE | `h = h + moe(h)` | `_apply_moe_layer` returns `h + out` (moe:627) | ✅ correct |
| 11 | Final norm | RMSNorm | `_apply_final_norm` called in `_forward_one_token` (moe:556–558, 644) | ✅ correct |
| 12 | LM head | `h·Wᵀ` | Real weights / tied embeddings (moe:1160–1231) | ✅ correct |

**Net: all 12 stages are correctly implemented.** The transformer math is sound. The remaining
deficits are performance-level (expert loading bypasses cache; safetensors opened per-key).

---

## Performance Defects (What Still Needs Fixing)

1. **`get_or_load` bypasses the cache on the processor hot path.** `ExpertProcessor._get_expert`
   (expert_processor:123–142) passes the loader to `get_or_load`, which is `return loader()`.
   Experts loaded via this path never reach the three-tier `get()`. `_create_expert_loader`
   works around it explicitly, but the processor path does not.

2. **`_load_expert_from_storage` opens `safe_open` per matching key.** The index lookup is
   correct (O(n patterns) not O(files × keys)), but the inner loop calls `safe_open(st_file)`
   inside the `for key in index` loop (moe:673–674). Each matching key reopens the same file.
   Fix: open once per file before the inner key loop, collect all needed tensors, then close.

3. **Serial expert loading in batch prefill.** `process_batch` loads experts serially per token
   (expert_processor:188–230). `process_single` does parallel-load correctly via
   `ThreadPoolExecutor`, but batch prefill does not.

4. **Hot-expert preload targets layer 0 first.** `_preload_hot_experts` fills in layer-first
   order (moe:717–755). If expert routing is uniform across layers, this biases toward layer 0
   experts at the expense of layer 1+. A round-robin or frequency-based order would be better.

5. **`torch.no_grad()` per expert call** (expert_processor:221, 337) instead of once around the
   whole batch. Minor dispatch overhead but easy to consolidate.

6. **KV buffer batch size is fixed at first prefill.** Allocated from that call's `batch_size`
   (attn:177–186). A later request with a different batch size writes out of bounds. The
   `clear_cache` reset only clears `seq_len` (attn:420–425).

---

## What Is Genuinely Working (Confirmed)

- **Memory orchestration** (`resource_budget.py`, `memory_budget_calculator.py`) and the
  three-tier *design* (GPU hot / CPU warm / storage cold) are sound and differentiated.
  The abstraction is worth keeping; the `get_or_load` bypass is the only implementation gap.
- **Weight loading** (`_load_model_weights`, `_load_shared_weights`, `_load_expert_from_storage`)
  handles multiple architectures and normalizes Mixtral/Qwen/DeepSeek key names (moe:509–523).
- **Norms and residuals** are loaded and applied correctly at every stage — pre-attention norm,
  post-attention norm, final norm, attention residual, MoE residual. This was the biggest
  correctness gap in the previous summary and it is now fixed.
- **Attention projections** are loaded per layer and applied. The identity fallback with warning
  is appropriate for testing without a real checkpoint.
- **Router weight loading and per-layer swapping** (moe:438–463, 592–602) is correct.
- **Safetensors index** is built once at startup (moe:496–507) for O(1) key → (file, key) lookup.
  The per-key `safe_open` issue is the only remaining I/O inefficiency.
- **KV cache design** (pre-allocated buffer, in-place writes, no copies) is correct.
- **`_forward_one_token`** is the single source of truth for the per-token layer loop, shared
  by both streaming and non-streaming decode paths. No divergence possible.
- **ThreadPoolExecutor for concurrent expert loading** in `process_single` (expert_processor:306–318).
- **Batch-size aware safetensors loading** via `_build_safetensors_index` (moe:496–507).

---

## Ordered Fix List

**Correctness: DONE** — All 12 transformer stages are implemented. No correctness fixes needed.

**Performance (in priority order):**

1. **Fix `get_or_load` to call `get()` first** (cache:135–162). Change the body from
   `return loader()` to check the three-tier cache before loading. This is the single highest-
   impact fix: it makes the GPU/CPU tiers actually useful on the processor hot path.
   Expected impact: 100–500ms → <5ms per cached expert load.

2. **Open each safetensors file once per `_load_expert_from_storage` call** (moe:673–674).
   Group keys by file, open once, read all needed tensors, close. Eliminates redundant opens.
   Expected impact: ~10–30ms per expert load reduction.

3. **Parallelize expert loading in `process_batch`** (expert_processor:188–230).
   Apply the same `ThreadPoolExecutor` pattern used in `process_single`.

4. **Verify benchmark claim.** Run a reproducible benchmark and publish the actual tok/s.
   No performance number should appear in documentation without a command and its output.

5. **Fix KV buffer batch size** (attn:177–186, 420–425). Either use batch_size=1 for the
   pre-allocated buffer, or resize the buffer when `batch_size` changes.

---

## Status

- **Correctness:** ✅ All 12 transformer stages implemented correctly
- **Architecture:** ✅ Three-tier design, multi-arch weight loading, norms, residuals, projections
- **Hot-path performance:** ⚠️ `get_or_load` bypasses cache; safetensors opened per-key
- **Serving:** ✅ `CustomMoEInferenceEngine.generate` is the real serving path
- **Verified benchmarks:** ❌ No in-repo benchmark reproduces tok/s claims
