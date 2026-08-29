# Real MoE Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the random demo with a truthful, testable baseline for one real Mixtral-compatible checkpoint.

**Architecture:** Keep one lightweight CLI/runtime entrypoint. Separate model loading/generation from the existing cache primitives, and make cache keys layer-aware without implementing speculative prediction or custom kernels.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, pytest; CPU-only tests must not download a model.

**Spec:** docs/GOAL.md

## Global Constraints

- First model target is Mixtral 8x7B-Instruct-v0.1 or a compatible local Mixtral checkpoint.
- CPU-only regression tests must run without model downloads.
- Do not claim production readiness or benchmark values without measurements.
- Prediction, speculative prefetch, batching, network storage, and custom INT4 are deferred.

### Task 1: Runtime package and layer-aware cache

- Add `sparse_llm/__init__.py` exports.
- Update `ExpertCache` to accept `(layer_id, expert_id)` keys while preserving simple integer compatibility only if unambiguous.
- Add hit/miss counters and tests for eviction and layer isolation.

### Task 2: Safe local expert storage

- Consolidate the usable local storage class under `storage/backend.py`.
- Create directories, use atomic writes, and use `weights_only=True` where supported.
- Add temporary-directory tests for save/load/exists and missing experts.

### Task 3: Real model adapter and generation

- Add a Hugging Face adapter that loads tokenizer/model lazily, selects CPU/CUDA, and performs deterministic generation.
- Expose `generate(prompt, max_new_tokens, temperature)` and benchmark metrics.
- Keep imports lazy so package import and CPU tests work without Transformers/model files.

### Task 4: CLI and documentation

- Replace `main.py` random multi-user demos with arguments for model, prompt, token count, device, and cache size.
- Print generated text and measured metrics.
- Remove unsupported static performance claims from README and point to `docs/GOAL.md`.

### Task 5: Verification

- Run focused tests, then the complete suite.
- Run CLI help without downloading a model.
- Inspect the diff for unsupported claims and report any environment-limited checks.
