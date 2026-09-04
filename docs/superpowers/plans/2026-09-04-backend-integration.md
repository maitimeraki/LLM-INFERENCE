# Backend Integration for Resource-Aware Loading

> **For agentic workers:** Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Connect LoadedWeightState from resource-aware loading to InferenceEngine for actual text generation.

**Architecture:** Bridge resource-aware loaded weights into existing inference pipeline by modifying InferenceEngine to optionally consume LoadedWeightState and creating adapters that use pre-loaded weights.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, transformers

**Spec:** Enable end-to-end inference using resource-aware loaded weights, maintaining backward compatibility with existing loading paths.

## Global Constraints

- Backward compatibility: existing InferenceEngine usage must continue working
- No changes to external APIs except adding optional parameters
- LoadedWeightState integration is opt-in via explicit parameter
- All existing tests must pass
- Type hints required for all new code

---

## Tasks

### Task 1: Modify InferenceEngine to Accept LoadedWeightState

**Files:**
- Modify: `sparse_llm/inference/engine.py`
- Test: Manual verification with small model

**Goal:** Add optional `loaded_state` parameter to InferenceEngine that bypasses traditional loading.

**Steps:**
1. Read current `InferenceEngine.__init__` signature
2. Add `loaded_state: LoadedWeightState | None = None` parameter
3. Add conditional initialization path that uses loaded_state if provided
4. Preserve all existing initialization for backward compatibility
5. Test both paths work (with and without loaded_state)

### Task 2: Create Pre-loaded Weight Adapter

**Files:**
- Create: `sparse_llm/models/preloaded_adapter.py`
- Test: Unit test

**Goal:** Create adapter that wraps pre-loaded weights from LoadedWeightState.

**Steps:**
1. Create `PreloadedWeightAdapter` class implementing ModelAdapter interface
2. Accept LoadedWeightState in constructor
3. Implement forward pass using shared_weights and expert_cache
4. Handle both MoE and dense models
5. Add tests verifying weight access patterns

### Task 3: Wire Expert Cache into Forward Pass

**Files:**
- Modify: `sparse_llm/models/preloaded_adapter.py`

**Goal:** Integrate expert cache calls during forward pass for dynamic expert loading.

**Steps:**
1. Detect expert access patterns in forward pass
2. Call expert_cache.get() for expert weights
3. Handle cache misses gracefully
4. Track cache statistics during generation

### Task 4: Update main.py to Use Full Integration

**Files:**
- Modify: `main.py`

**Goal:** Remove early exit and complete the inference path with resource-aware loading.

**Steps:**
1. Remove the early exit (lines 88-99)
2. Pass loaded_state to InferenceEngine
3. Continue with generation as normal
4. Print cache statistics after generation

### Task 5: End-to-End Integration Test

**Files:**
- Create: `sparse_llm/tests/test_backend_integration.py`

**Goal:** Verify complete flow from resource-aware loading to text generation.

**Steps:**
1. Load small model (gpt2) with resource-aware system
2. Create InferenceEngine with loaded_state
3. Generate text with simple prompt
4. Verify output is valid
5. Check cache statistics
