# Universal Sparse LLM Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a truthful universal local inference path for any locally available or Hugging Face-hosted causal language model that `AutoModelForCausalLM` can load, while keeping unsupported MoE architectures on the generic path.

**Architecture:** `TransformersCausalLMAdapter` is the safe universal adapter and loads tokenizer/model resources lazily, discovers capabilities from loaded configuration/module metadata, performs real prefill and cached decode, and returns measured results. `InferenceEngine` is the single runtime entrypoint. `ModelRegistry` selects only validated specializations; unknown dense and MoE models use the generic adapter with `expert_paging=False`. Existing cache, storage, pager, and memory modules remain reusable primitives but are not connected to generic generation until a structural adapter proves reference-output equivalence.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, safetensors, pytest, standard library.

**Spec:** `docs/IMPLEMENTATION_PLAN.md`

## Global Constraints

- The public flow is model-independent and must not branch on a hardcoded model-name list.
- Any compatible model means a local or Hugging Face artifact loadable as a causal LM through the supported Transformers API.
- `trust_remote_code=False` and local/offline operation are safe defaults.
- Transformers and model files must not be imported or loaded at package import or CLI-help time.
- Generic models and unsupported MoE models must never fabricate routing or claim expert paging.
- No random input tensors, fabricated generated text, averaged expert weights, or static performance values.
- CPU tests must not download checkpoints or require CUDA.
- GPU memory accounting is byte-based; expert count is diagnostic only.
- Preserve atomic writes and conservative tensor deserialization.
- Do not add prediction, speculative prefetch, batching, network storage, custom INT4 kernels, or multi-GPU execution to the correctness path.
- All implementation edits for this change stay under `sparse_llm`; no git worktree is used for implementation.

---

### Task 1: Stabilize universal contracts and exports

**Files:**
- Modify: `sparse_llm/models/adapters.py`
- Modify: `sparse_llm/models/paging.py`
- Modify: `sparse_llm/inference/metrics.py`
- Modify: `sparse_llm/models/__init__.py`
- Modify: `sparse_llm/inference/__init__.py`
- Modify: `sparse_llm/__init__.py`
- Test: `sparse_llm/tests/test_models.py`
- Test: `sparse_llm/tests/test_paging.py`

**Interfaces:**
- `DevicePolicy.resolve_device() -> torch.device` and `DevicePolicy.torch_dtype() -> torch.dtype | None` remain the single loading-option conversion boundary.
- `ModelCapabilities.to_dict() -> dict[str, Any]` reports discovered model type, architecture, dense/MoE classification, layer/expert/top-k metadata, KV-cache support, and `expert_paging`.
- `ModelAdapter.capabilities`, `paging_capabilities`, `validate_paging()`, `load()`, and `generate()` remain the adapter contract.
- `GenerationMetrics.to_dict() -> dict[str, int | float]` and `GenerationResult(text, token_ids, metrics)` remain stable result objects.
- `ExpertKey` means `(layer_index, expert_index)` everywhere paging is represented.

- [ ] **Step 1: Add failing contract tests**

```python
def test_capability_serialization_is_json_safe():
    capabilities = ModelCapabilities(
        model_id="local-model",
        model_type="custom",
        architecture="CustomForCausalLM",
        architecture_classification="moe",
        is_moe=True,
        num_hidden_layers=4,
        num_experts=8,
        top_k_experts=2,
    )
    payload = capabilities.to_dict()
    assert payload["architecture_classification"] == "moe"
    assert payload["num_experts"] == 8
    assert payload["expert_paging"] is False


def test_generic_adapter_reports_paging_unavailable():
    adapter = TransformersCausalLMAdapter("local-model", policy=DevicePolicy(device="cpu"))
    result = adapter.validate_paging()
    assert not result.available
    assert "architecture" in result.reason
```

- [ ] **Step 2: Run focused contract tests and confirm the missing/incorrect contract fails**

Run: `pytest sparse_llm/tests/test_models.py sparse_llm/tests/test_paging.py -q`

Expected: the new contract assertion identifies any serialization or generic-paging mismatch without requiring a model download.

- [ ] **Step 3: Implement only the smallest contract corrections**

Keep value objects serializable and explicit. Ensure a generic adapter has no paging evidence by default. Ensure paging validation includes a structured availability flag and reason, and never infers availability merely from `is_moe`.

- [ ] **Step 4: Run contract tests**

Run: `pytest sparse_llm/tests/test_models.py sparse_llm/tests/test_paging.py -q`

Expected: PASS.

---

### Task 2: Make the universal adapter and registry truthful

**Files:**
- Modify: `sparse_llm/models/adapters.py`
- Modify: `sparse_llm/models/registry.py`
- Modify: `sparse_llm/models/__init__.py`
- Modify: `sparse_llm/models/universal_adapter.py`
- Test: `sparse_llm/tests/test_models.py`
- Test: `sparse_llm/tests/test_universal_adapter.py`

**Interfaces:**
- `TransformersCausalLMAdapter(model_id, policy=None, tokenizer=None, model=None, config=None, tokenizer_loader=None, model_loader=None)` remains dependency-injectable.
- `load() -> TransformersCausalLMAdapter` lazily resolves Transformers loaders, forwards revision/offline/custom-code/dtype/device-map/offload options, and evaluates the model.
- `generate(prompt, max_new_tokens=32, temperature=0.0) -> GenerationResult` uses actual tokenizer/model behavior and cached decode when returned by the backend.
- `ModelRegistry.create(model_id, policy=None, **kwargs) -> ModelAdapter` must not select a paging specialization from an identifier heuristic.
- The generic adapter is selected for dense and unsupported/unknown MoE models.

- [ ] **Step 1: Add failing tests for universal fallback and lazy loading**

```python
def test_registry_does_not_claim_paging_from_model_name():
    adapter = create_model_adapter("mixtral-looking-name", policy=DevicePolicy(device="cpu"))
    assert isinstance(adapter, TransformersCausalLMAdapter)
    assert adapter.capabilities.expert_paging is False


def test_unknown_moe_uses_generic_capabilities(fake_moe_backend):
    adapter = TransformersCausalLMAdapter(
        "local-moe",
        policy=DevicePolicy(device="cpu", local_files_only=True),
        tokenizer=fake_moe_backend.tokenizer,
        model=fake_moe_backend.model,
        config=fake_moe_backend.config,
    )
    assert adapter.capabilities.is_moe is True
    assert adapter.capabilities.expert_paging is False
    assert not adapter.validate_paging().available
```

- [ ] **Step 2: Run the targeted tests and verify they fail or expose unsafe selection**

Run: `pytest sparse_llm/tests/test_models.py sparse_llm/tests/test_universal_adapter.py -q`

Expected: any universal adapter or registry path that attempts guessed topology/mapping is identified.

- [ ] **Step 3: Remove guessed generic paging selection**

Keep architecture-specific mapper code out of the default registry unless its validator can prove module paths, router semantics, expert identities, tensor contracts, and state-dict preservation. Unknown or unsupported MoE configurations must delegate to ordinary `TransformersCausalLMAdapter` behavior and retain `expert_paging=False`. Do not route generic models through `SparseMoELayer` or reconstruct expert computation from guessed gate/up/down weights.

- [ ] **Step 4: Harden capability discovery from config and module metadata**

Use configuration fields and loaded module metadata only to report capabilities. Preserve unknown values as `None`, classify MoE only when explicit expert/routing metadata exists, and never turn classification into a paging claim. Keep all loader imports inside `_loaders()`.

- [ ] **Step 5: Run universal adapter tests**

Run: `pytest sparse_llm/tests/test_models.py sparse_llm/tests/test_universal_adapter.py -q`

Expected: PASS, including dense and unsupported-MoE fallback cases.

---

### Task 3: Verify one runtime and real generation behavior

**Files:**
- Modify: `sparse_llm/inference/engine.py`
- Modify: `sparse_llm/core/inference.py`
- Modify: `sparse_llm/inference/metrics.py`
- Modify: `sparse_llm/inference/__init__.py`
- Modify: `sparse_llm/__init__.py`
- Test: `sparse_llm/tests/test_models.py`
- Test: `sparse_llm/tests/test_system.py`

**Interfaces:**
- `InferenceEngine(model, device="auto", dtype=None, revision=None, local_files_only=False, trust_remote_code=False, device_map=None, offload_folder=None, expert_cache_bytes=None, adapter=None)` owns one adapter and exposes `capabilities`, `load()`, `generate()`, `stats()`, and `clear_cache()`.
- `sparse_llm.core.inference.InferenceEngine` is a compatibility re-export of the same class.
- Generation metrics report measured prefill/decode/throughput/peak-VRAM values and zero expert activity on the generic path.

- [ ] **Step 1: Add tests proving delegation and one-engine identity**

```python
def test_core_and_public_engine_are_the_same_class():
    from sparse_llm.core.inference import InferenceEngine as CoreEngine
    from sparse_llm.inference.engine import InferenceEngine
    assert CoreEngine is InferenceEngine


def test_engine_does_not_fabricate_expert_metrics(fake_adapter):
    engine = InferenceEngine(adapter=fake_adapter)
    result = engine.generate("hello", max_new_tokens=1)
    assert result.metrics.cache_hits == 0
    assert result.metrics.cache_misses == 0
    assert result.metrics.expert_load_time_ms == 0.0
```

- [ ] **Step 2: Run the runtime tests**

Run: `pytest sparse_llm/tests/test_models.py sparse_llm/tests/test_system.py -q`

Expected: FAIL only where duplicate engines, stale constructor arguments, or fabricated metrics remain.

- [ ] **Step 3: Consolidate engine delegation**

Keep one adapter-backed implementation. Construct cache/storage only as explicit primitives if the current API requires them, but do not count their idle state as expert execution. Preserve compatibility re-export and remove any old model-averaging or random-input path from runtime reachability.

- [ ] **Step 4: Verify deterministic fake generation and metric edge cases**

Run: `pytest sparse_llm/tests/test_models.py sparse_llm/tests/test_system.py -q`

Expected: PASS for deterministic greedy output, zero-token generation, cache/no-cache decode, and measured metrics.

---

### Task 4: Harden cache, pager, memory, and storage boundaries without wiring speculative execution

**Files:**
- Modify: `sparse_llm/cache/expert_cache.py`
- Modify: `sparse_llm/inference/pager.py`
- Modify: `sparse_llm/inference/memory.py`
- Modify: `sparse_llm/storage/checkpoint_index.py`
- Modify: `sparse_llm/storage/backend.py`
- Test: `sparse_llm/tests/test_cache.py`
- Test: `sparse_llm/tests/test_paging.py`
- Test: `sparse_llm/tests/test_storage.py`

**Interfaces:**
- Cache capacity is authoritative in bytes, with layer-aware keys, pin/unpin leases, deterministic eviction, observable hit/miss/eviction/occupancy metrics, and in-flight load de-duplication.
- `ExpertPager` admits only validated requested experts, uses `ExpertKey`, and releases leases on success and failure.
- `MemoryPlanner` rejects impossible shared/routed/KV/activation budgets with actionable diagnostics.
- `CheckpointIndex` validates safe local artifact metadata and exact layer/expert tensor mappings.
- `LocalSSDStorage` retains atomic replacement, CPU detachment, and conservative tensor loading.

- [ ] **Step 1: Add failing tests for oversized entries and failed in-flight loads**

```python
def test_oversized_cache_entry_reports_required_and_available_bytes():
    cache = ExpertCache(max_experts=4, max_bytes=4)
    with pytest.raises(MemoryError, match="required.*available"):
        cache.put((0, 0), torch.zeros(8, dtype=torch.float32))


def test_failed_inflight_load_is_retryable():
    pager = make_pager()
    calls = 0
    def loader():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("read failed")
        return torch.ones(1)
    with pytest.raises(OSError):
        pager.cache.get_or_load((0, 0), loader)
    assert pager.cache.get_or_load((0, 0), loader) is not None
    assert calls == 2
```

- [ ] **Step 2: Run cache/pager/storage tests and confirm failures**

Run: `pytest sparse_llm/tests/test_cache.py sparse_llm/tests/test_paging.py sparse_llm/tests/test_storage.py -q`

Expected: failures identify only missing diagnostics or cleanup; no model download is involved.

- [ ] **Step 3: Implement bounded byte admission and cleanup**

Reject entries larger than available capacity with required/available byte values and current occupancy. Ensure failed loader futures/events are removed so a later request can retry. Never evict pinned entries. Keep `contains()` observational and count hits/misses only for `get()`/load operations.

- [ ] **Step 4: Run primitive tests**

Run: `pytest sparse_llm/tests/test_cache.py sparse_llm/tests/test_paging.py sparse_llm/tests/test_storage.py -q`

Expected: PASS.

---

### Task 5: Full sparse_llm verification and deferred-boundary audit

**Files:**
- Modify only failing files under `sparse_llm`.
- Test: all existing `sparse_llm/tests/*.py`.

- [ ] **Step 1: Search sparse_llm for forbidden runtime behavior**

Run: `rg -n "torch\\.randn|Generated output|average|averaged|expert_paging *= *True|predicate.*model_id|fallback_expert_forward" sparse_llm`

Expected: no forbidden behavior is reachable from the public generic runtime; architecture-specific code may remain only behind explicit validation and tests.

- [ ] **Step 2: Run the complete CPU suite**

Run: `pytest -q`

Expected: all tests pass without downloading a checkpoint or requiring CUDA.

- [ ] **Step 3: Run compile checks**

Run: `python -m compileall -q sparse_llm`

Expected: exit code 0.

- [ ] **Step 4: Verify package import is lightweight**

Run:

```bash
python -c "import sys; import sparse_llm; assert 'transformers' not in sys.modules; print('ok')"
```

Expected: `ok` without model loading.

- [ ] **Step 5: Record the universal baseline boundary**

The implementation is complete when every compatible causal model can use the generic tokenizer/model/generation path, dense and unsupported MoE models report truthful fallback capabilities, cache/storage primitives remain safe and tested, and architecture-specific paging remains blocked until Work Packages B-E pass structural and reference-output gates. No real-checkpoint result is substituted when the environment lacks model files or hardware.
