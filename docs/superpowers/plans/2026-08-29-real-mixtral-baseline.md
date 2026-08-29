# Real Mixtral Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the random multi-user demo with one truthful, testable Hugging Face Mixtral generation path plus layer-aware expert cache and safe local storage primitives.

**Architecture:** `main.py` becomes a thin CLI over one `InferenceEngine`. The engine owns a lazy `MixtralAdapter`, an `ExpertCache`, and `LocalSSDStorage`; the adapter uses the reference Transformers model/tokenizer path for real prefill and autoregressive decoding, while cache/storage remain explicit, independently measurable primitives until Milestone 2 integrates expert-module swapping. No prediction, speculative prefetch, batching, quantization kernel, or network storage is on the correctness path.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, pytest, standard-library `argparse`/`time`/`pathlib`.

**Spec:** `docs/GOAL.md` and `docs/IMPLEMENTATION_PLAN.md`

## Global Constraints

- First model target is Mixtral 8x7B-Instruct-v0.1 or a compatible local Mixtral checkpoint.
- CPU-only regression tests must run without model downloads.
- Do not claim production readiness or benchmark values without measurements.
- Prediction, speculative prefetch, batching, network storage, and custom INT4 are deferred.
- Transformers and model files are imported/loaded lazily; `python main.py --help` and package import do not download a model.
- The reference adapter must not pretend that its cache metrics represent expert swapping; cache/storage metrics are zero or measured only when those primitives are explicitly exercised.

---

### Task 1: Public runtime API and layer-aware LRU cache

**Files:**
- Modify: `sparse_llm/cache/expert_cache.py`
- Modify: `sparse_llm/cache/__init__.py`
- Modify: `sparse_llm/__init__.py`
- Create: `sparse_llm/tests/test_cache.py`
- Replace: `sparse_llm/tests/test_all_phases.py` with focused CPU-only API tests
- Replace: `sparse_llm/tests/test_system.py` with focused CPU-only API tests

**Interfaces:**
- Consumes: existing `ExpertCache(max_experts: int)` call sites.
- Produces: `ExpertKey = tuple[int, int]`; `ExpertCache.get(key_or_expert: ExpertKey | int, layer_id: int | None = None) -> object | None`; matching `put`, `contains`, `clear`, and `stats` methods. Tuple keys are the preferred API and always mean `(layer_id, expert_id)`.
- Integer compatibility remains for legacy callers only when unambiguous: a direct legacy integer entry is accepted, one matching layer-aware entry may be resolved, and multiple matching layer-aware entries raise `ValueError` instead of returning the wrong layer.
- `stats()` returns `cached_experts`, `max_experts`, `total_accesses`, `hits`, `misses`, `hit_rate`, and `avg_access_count`.

- [ ] **Step 1: Write the failing cache tests**

```python
from sparse_llm.cache import ExpertCache
import pytest


def test_layer_aware_keys_do_not_collide():
    cache = ExpertCache(max_experts=2)
    cache.put((0, 3), "layer-zero")
    cache.put((1, 3), "layer-one")

    assert cache.get((0, 3)) == "layer-zero"
    assert cache.get((1, 3)) == "layer-one"
    assert cache.stats()["hits"] == 2


def test_integer_lookup_rejects_ambiguous_layer_match():
    cache = ExpertCache(max_experts=3)
    cache.put((0, 2), "a")
    cache.put((1, 2), "b")

    with pytest.raises(ValueError, match="ambiguous"):
        cache.get(2)


def test_lru_eviction_and_miss_counters():
    cache = ExpertCache(max_experts=2)
    cache.put((0, 0), "a")
    cache.put((0, 1), "b")
    assert cache.get((0, 0)) == "a"
    cache.put((0, 2), "c")

    assert not cache.contains((0, 1))
    assert cache.get((0, 1)) is None
    assert cache.stats()["misses"] == 1
    assert cache.stats()["hit_rate"] == pytest.approx(1 / 2)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `pytest sparse_llm/tests/test_cache.py -q`

Expected: FAIL because tuple keys and ambiguity handling are not implemented by the current integer-oriented API.

- [ ] **Step 3: Implement normalized keys and bounded LRU behavior**

Use one internal `OrderedDict[ExpertKey | int, object]` and one lock. Normalize tuple arguments directly; reject malformed tuples and conflicting `layer_id`. On an integer lookup, first check a direct integer entry, then collect layer-aware matches for that expert ID and raise `ValueError` if more than one exists. Count only `get` calls as hits or misses; `contains` must remain observational. Evict the oldest key before inserting a new one and remove its access metadata.

- [ ] **Step 4: Export the cache and run the tests**

Export `ExpertCache` and `ExpertKey` from `sparse_llm/cache/__init__.py`; run `pytest sparse_llm/tests/test_cache.py -q` and expect all tests to pass.

- [ ] **Step 5: Make package-level exports import-safe**

Export only lightweight symbols from `sparse_llm/__init__.py` initially (`ExpertCache`, `ExpertKey`); do not import Transformers there. Replace the two executable legacy test scripts with pytest tests that import the public API without constructing a model.

- [ ] **Step 6: Commit the cache boundary**

```bash
git add sparse_llm/cache sparse_llm/__init__.py sparse_llm/tests
git commit -m "feat: add layer-aware bounded expert cache"
```

---

### Task 2: Safe local expert storage

**Files:**
- Modify: `sparse_llm/storage/backend.py`
- Modify: `sparse_llm/storage/base.py` to re-export the canonical backend types for compatibility
- Modify: `sparse_llm/storage/__init__.py`
- Modify: `sparse_llm/storage/local_ssd.py` to re-export `LocalSSDStorage` instead of maintaining a second implementation
- Create: `sparse_llm/tests/test_storage.py`

**Interfaces:**
- Consumes: `ExpertKey` from `sparse_llm.cache.expert_cache` and CPU tensors.
- Produces: `StorageBackend` abstract interface and `LocalSSDStorage(base_path: str | Path = "./cache/experts")` with `load_expert`, `save_expert`, and `exists` accepting tuple keys or unambiguous integer compatibility arguments.
- File names are deterministic and layer-aware: `layer_0000_expert_0003.pt`; legacy integer-only keys use `expert_000003.pt`.
- `save_expert` stores a detached CPU tensor via a temporary file followed by `os.replace`; `load_expert` maps to CPU and requests `weights_only=True`, with a compatibility fallback only for Transformers/PyTorch versions that reject that keyword.

- [ ] **Step 1: Write storage tests using `tmp_path`**

```python
import pytest
import torch

from sparse_llm.storage import LocalSSDStorage


def test_save_load_exists_and_layer_isolation(tmp_path):
    storage = LocalSSDStorage(tmp_path)
    first = torch.tensor([1.0, 2.0])
    second = torch.tensor([3.0])

    storage.save_expert((0, 4), first)
    storage.save_expert((1, 4), second)

    assert storage.exists((0, 4))
    assert storage.exists((1, 4))
    assert not storage.exists((2, 4))
    assert torch.equal(storage.load_expert((0, 4)), first)
    assert torch.equal(storage.load_expert((1, 4)), second)


def test_missing_expert_has_actionable_error(tmp_path):
    storage = LocalSSDStorage(tmp_path)

    with pytest.raises(FileNotFoundError, match="layer 2.*expert 9"):
        storage.load_expert((2, 9))
```

- [ ] **Step 2: Run the storage tests and verify they fail**

Run: `pytest sparse_llm/tests/test_storage.py -q`

Expected: FAIL because the current backend does not expose the canonical storage interface or tuple-key validation.

- [ ] **Step 3: Define the canonical backend contract**

Put `ExpertKey`, `StorageBackend`, and `LocalSSDStorage` in `storage/backend.py`. Keep abstract methods limited to `load_expert`, `save_expert`, and `exists`; do not carry the obsolete metadata registry into the baseline. Add a private key/path helper that validates non-negative integer layer and expert IDs and creates the exact deterministic filename.

- [ ] **Step 4: Implement atomic tensor writes and safe loads**

Create the base directory in the constructor. Write a detached CPU tensor to a same-directory temporary file, flush and `fsync` it, then atomically replace the destination. Always clean up the temporary path in `finally`. On load, raise `FileNotFoundError` with the resolved layer/expert and path; call `torch.load(path, map_location="cpu", weights_only=True)` and retry without `weights_only` only after a `TypeError`.

- [ ] **Step 5: Remove the duplicate implementation and run tests**

Turn `storage/base.py` and `storage/local_ssd.py` into compatibility re-exports, export `StorageBackend` and `LocalSSDStorage` from `storage/__init__.py`, and run `pytest sparse_llm/tests/test_storage.py -q` expecting PASS.

- [ ] **Step 6: Commit storage**

```bash
git add sparse_llm/storage sparse_llm/tests/test_storage.py
git commit -m "feat: consolidate atomic local expert storage"
```

---

### Task 3: Lazy Mixtral adapter, real generation, and measured metrics

**Files:**
- Create: `sparse_llm/inference/metrics.py`
- Create: `sparse_llm/models/mixtral.py`
- Replace: `sparse_llm/inference/engine.py`
- Replace: `sparse_llm/core/inference.py` with a compatibility re-export of the one engine
- Modify: `sparse_llm/inference/__init__.py`
- Modify: `sparse_llm/models/__init__.py`
- Modify: `sparse_llm/config/__init__.py` to contain the minimal first-milestone configuration
- Create: `sparse_llm/tests/test_adapter.py`

**Interfaces:**
- `GenerationMetrics` is a dataclass with `prompt_tokens`, `generated_tokens`, `prefill_latency_ms`, `decode_latency_ms`, `throughput_tokens_per_sec`, `peak_vram_bytes`, `cache_hits`, `cache_misses`, and `expert_load_time_ms`, plus `to_dict() -> dict[str, int | float]`.
- `GenerationResult` is a dataclass with `text: str`, `token_ids: list[int]`, and `metrics: GenerationMetrics`.
- `MixtralAdapter(model_name: str, device: str = "auto", local_files_only: bool = False, seed: int = 0)` stores configuration without importing Transformers or loading files. `load() -> None` lazily creates the actual `AutoTokenizer` and `AutoModelForCausalLM`, selects CPU/CUDA, validates an available explicit CUDA device, and calls `eval()`.
- `MixtralAdapter.generate(prompt: str, max_new_tokens: int = 32, temperature: float = 0.0) -> GenerationResult` performs a real tokenizer → model prefill → cached autoregressive decode path. `temperature <= 0` is greedy; positive temperature samples from logits using the configured seed.
- `InferenceEngine(model_name, device="auto", max_cached_experts=22, storage_path="./cache/experts", local_files_only=False)` owns the adapter, `ExpertCache`, and `LocalSSDStorage`; `generate(...)` delegates to the adapter and returns `GenerationResult`.
- The engine must not report fabricated expert activity: until expert-module integration is implemented, cache hit/miss and expert-load metrics are zero and documented as reference-path metrics.

- [ ] **Step 1: Write import-safety and deterministic adapter tests with monkeypatched dependencies**

```python
import sys
import types

import pytest

def test_package_import_does_not_import_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    import sparse_llm
    assert "transformers" not in sys.modules or sys.modules["transformers"] is None


def test_generation_result_is_deterministic_with_fake_backend(monkeypatch):
    # Install a tiny fake transformers module whose model emits a fixed next token.
    class FakeTokenizer:
        eos_token_id = 0
        def __call__(self, prompt, return_tensors="pt"):
            import torch
            return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
        def decode(self, ids, skip_special_tokens=True):
            return "fixed output"

    class FakeModel:
        config = types.SimpleNamespace(model_type="mixtral")
        def eval(self): return self
        def to(self, device): return self
        def __call__(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
            import torch
            logits = torch.zeros((1, input_ids.shape[1], 4))
            logits[:, -1, 3] = 1
            return types.SimpleNamespace(logits=logits, past_key_values=())

    fake = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer()),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)

    from sparse_llm.models.mixtral import MixtralAdapter
    adapter = MixtralAdapter("local-mixtral", device="cpu", local_files_only=True)
    first = adapter.generate("hello", max_new_tokens=2)
    second = adapter.generate("hello", max_new_tokens=2)
    assert first.text == second.text
    assert first.metrics.generated_tokens == 2
```

- [ ] **Step 2: Run the adapter tests and verify they fail**

Run: `pytest sparse_llm/tests/test_adapter.py -q`

Expected: FAIL because the lazy adapter, result types, and measured decode loop do not exist.

- [ ] **Step 3: Implement metrics dataclasses**

Use `time.perf_counter()` for CPU and CUDA-synchronized boundaries. `to_dict()` must preserve metric names from `docs/GOAL.md`; use `0` for peak VRAM on CPU and reset/read CUDA peak allocation only when the selected device is CUDA. Calculate throughput from generated token count divided by decode seconds, returning `0.0` when no token was generated.

- [ ] **Step 4: Implement lazy model loading**

Keep `transformers` imports inside `MixtralAdapter.load`. Resolve `auto` to CUDA when available, otherwise CPU; reject `cuda` when unavailable with a clear `RuntimeError`. Choose float16 on CUDA and float32 on CPU unless an explicit dtype configuration is added. Load `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained` with `local_files_only` forwarded, move the model to the selected device, set evaluation mode, and reject an explicitly incompatible non-Mixtral `config.model_type` with a diagnostic naming the expected checkpoint family.

- [ ] **Step 5: Implement actual prefill and autoregressive decoding**

Tokenize the prompt with the loaded tokenizer and move tensor inputs to the selected device. Run one `model(...)` call over the full prompt to obtain logits and `past_key_values`, select/sample the first token, then loop one token at a time with the growing attention mask and returned cache. Stop on `eos_token_id` or `max_new_tokens`; decode only generated IDs with the actual tokenizer. Synchronize CUDA around measurements. Use a local `torch.Generator` seeded with `seed` for positive-temperature sampling so repeated calls with the same settings are reproducible.

- [ ] **Step 6: Implement the single engine and compatibility exports**

Construct `ExpertCache` and `LocalSSDStorage` once in `InferenceEngine`; pass their current statistics into the result only as zero/uninstrumented reference metrics. Do not instantiate prediction, prefetch, quantization, request queues, worker threads, or random toy tensors. Make `sparse_llm/core/inference.py` re-export this engine so there is one implementation, and export adapter/result/metrics symbols from the relevant package `__init__.py` files without eager Transformers imports.

- [ ] **Step 7: Run CPU-only adapter tests**

Run: `pytest sparse_llm/tests/test_adapter.py -q`

Expected: PASS using only monkeypatched fake Transformers classes; no model download occurs.

- [ ] **Step 8: Commit the real generation boundary**

```bash
git add sparse_llm/models sparse_llm/inference sparse_llm/core/inference.py sparse_llm/config sparse_llm/tests/test_adapter.py
git commit -m "feat: add lazy Mixtral reference generation"
```

---

### Task 4: CLI, dependency metadata, and truthful documentation

**Files:**
- Replace: `main.py`
- Modify: `pyproject.toml`
- Replace: `README.md`
- Modify: `docs/ARCHITECTURE.md` to remove unsupported first-milestone claims and point at the measured baseline
- Create: `sparse_llm/tests/test_cli.py`

**Interfaces:**
- CLI flags: `--model`, `--prompt`, `--max-new-tokens`, `--temperature`, `--device {auto,cpu,cuda}`, `--cache-size`, `--storage-path`, and `--local-files-only`.
- `main(argv: list[str] | None = None) -> int` returns a nonzero code for generation failures and does not load a model while parsing `--help`.
- Successful CLI output contains the generated text followed by measured metric lines or JSON fields for prefill latency, decode latency, throughput, peak VRAM, cache hits/misses, and expert load time.

- [ ] **Step 1: Write CLI help and failure-path tests**

```python
from main import main


def test_help_does_not_load_a_model(capsys):
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "--model" in output
    assert "--max-new-tokens" in output


def test_missing_local_model_returns_error(monkeypatch, capsys):
    class FailingEngine:
        def __init__(self, **kwargs): pass
        def generate(self, **kwargs): raise FileNotFoundError("checkpoint missing")
    monkeypatch.setattr("main.InferenceEngine", FailingEngine)

    assert main(["--model", "missing", "--prompt", "hello", "--local-files-only"]) == 1
    assert "checkpoint missing" in capsys.readouterr().err
```

- [ ] **Step 2: Run CLI tests and verify the generation test fails**

Run: `pytest sparse_llm/tests/test_cli.py -q`

Expected: FAIL because `main.py` still runs the random multi-user demos and has no argument parser.

- [ ] **Step 3: Replace `main.py` with a thin argparse entrypoint**

Parse all documented flags before constructing `InferenceEngine`; pass model/device/cache/storage/local-files-only through unchanged; call `generate` once; print text and `metrics.to_dict()` as stable JSON. Catch expected model/load/runtime errors, write a concise message to stderr, and return `1`; keep `if __name__ == "__main__": raise SystemExit(main())`.

- [ ] **Step 4: Align package metadata with the supported scope**

Update the project description to measured local Mixtral inference, retain only dependencies required by the reference path and tests, and add a console-script entry point only if it does not duplicate the existing `python main.py` path. Do not add speculative prediction, serving, or custom-kernel dependencies.

- [ ] **Step 5: Rewrite README and architecture claims**

Document installation, the Mixtral checkpoint identifier, local checkpoint/offline usage, CPU-only help/test commands, one CLI example, the measured metric schema, and the explicit boundary that expert cache/storage are baseline primitives pending Milestone 2 integration. Remove invented throughput, latency, hit-rate, VRAM, “production-grade,” arbitrary 100B+, 4-bit, prediction, prefetch, and batch guarantees. Link readers to `docs/GOAL.md` and state that real benchmark values must come from a run.

- [ ] **Step 6: Run the CLI tests and help command**

Run: `pytest sparse_llm/tests/test_cli.py -q`

Expected: PASS.

Run: `python main.py --help`

Expected: help text exits successfully without importing/loading a checkpoint.

- [ ] **Step 7: Commit the CLI and documentation**

```bash
git add main.py pyproject.toml README.md docs/ARCHITECTURE.md sparse_llm/tests/test_cli.py
git commit -m "feat: replace random demos with Mixtral CLI"
```

---

### Task 5: Full verification and cleanup of obsolete runtime paths

**Files:**
- Modify or delete: stale phase-demo modules and tests that are no longer reachable from the public runtime (`sparse_llm/prefetch/`, `sparse_llm/quantization/`, obsolete router/scheduling demo code)
- Modify: `sparse_llm/tests/` only as needed to remove imports of `SparseInference` and fabricated production claims

**Interfaces:**
- Consumes: the public `InferenceEngine`, `MixtralAdapter`, `ExpertCache`, and `LocalSSDStorage` interfaces from Tasks 1–4.
- Produces: a package that has one supported runtime path, passes CPU-only tests without network/model access, and gives clear diagnostics when a real checkpoint cannot be loaded.

- [ ] **Step 1: Search for forbidden fabricated behavior and duplicate entrypoints**

Run: `rg -n "random|Generated output|production-grade|100B|tok/s|cache hit|phase|SparseInference|dummy_input|torch\.randn" README.md main.py sparse_llm docs`

Expected: no production-facing random output, static benchmark values, unsupported model names, or old public API references. Legacy implementation files may remain only if they are unreachable and explicitly documented as deferred; otherwise remove them rather than maintaining a second runtime.

- [ ] **Step 2: Run the focused CPU suite**

Run: `pytest sparse_llm/tests/test_cache.py sparse_llm/tests/test_storage.py sparse_llm/tests/test_adapter.py sparse_llm/tests/test_cli.py -q`

Expected: all tests pass without downloading a checkpoint.

- [ ] **Step 3: Run the complete suite and static checks**

Run: `pytest -q`

Expected: all tests pass; any environment failure must identify the missing optional package or unavailable hardware rather than being hidden.

Run: `python -m compileall -q main.py sparse_llm`

Expected: exit code 0.

- [ ] **Step 4: Exercise a real checkpoint only when available**

Run with a documented local or configured Hugging Face checkpoint:

```bash
python main.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --prompt "Explain sparse mixture-of-experts routing." --max-new-tokens 16 --temperature 0 --device auto
```

Expected: actual tokenizer-generated text and measured metrics. If the environment lacks the checkpoint, CUDA memory, credentials, or network access, report that check as skipped with the exact diagnostic; do not substitute fabricated output.

- [ ] **Step 5: Inspect the final diff and status**

Run: `git diff main~5..HEAD --stat && git status --short`

Expected: only the planned runtime, tests, and documentation changes remain, with no generated cache/checkpoint artifacts committed.

- [ ] **Step 6: Commit cleanup if needed**

```bash
git add sparse_llm README.md docs main.py pyproject.toml
git commit -m "chore: remove obsolete baseline demo paths"
```

## Self-review checklist

- [x] Goal requirements map to tasks: real tokenizer/model/generation (Task 3), layer-aware cache (Task 1), local storage and atomic writes (Task 2), bounded cache and metrics (Tasks 1–3), CLI and truthful docs (Task 4), CPU-only tests and verification (Task 5).
- [x] No task claims that reference Transformers generation has already integrated expert swapping; that work remains the explicitly deferred Milestone 2 boundary.
- [x] Interfaces use the same names and signatures throughout the plan.
- [x] Tests avoid model downloads and use a fake backend for adapter behavior.
- [x] Static performance claims and unsupported model names are removed from production-facing documentation.
