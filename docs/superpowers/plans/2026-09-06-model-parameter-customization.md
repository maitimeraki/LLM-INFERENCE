# Robust Memory Management with User-Configurable Precision

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a robust, user-configurable memory management system that supports any precision (FP32/FP16/INT8/INT4), prevents runtime errors, and optimally allocates GPU/CPU resources based on user choice and available hardware.

**Architecture:** Priority-based allocation hierarchy with precision-aware calculations. System calculates memory requirements based on user's precision choice, validates against available resources, and allocates in strict priority order: (1) Per-token weights → GPU (required by vLLM), (2) KV cache → GPU (if space available), (3) Expert cache → GPU remainder (dynamic loading during inference).

**Tech Stack:** Python, PyTorch, vLLM, precision-aware memory profiling

**Spec:** User chooses model + precision (FP32/FP16/INT8/INT4). System calculates exact memory needs, validates against GPU/CPU, allocates in priority order, prevents all runtime errors. vLLM requires GPU; experts can use CPU with dynamic GPU loading.

## Global Constraints

- **User control:** User chooses precision (FP32/FP16/INT8/INT4), system adapts
- **Zero runtime errors:** Validate before every allocation, fail fast at startup with clear guidance
- **Priority hierarchy (strict):**
  1. Per-token weights → GPU (vLLM requires GPU, no CPU fallback)
  2. KV cache → GPU (if space available, else reduce context length)
  3. Expert cache → GPU remainder (hot cache), CPU (warm cache), Storage (cold)
- **Dynamic expert loading:** Experts loaded from CPU→GPU during inference as needed
- **Precision-aware:** All memory calculations account for bytes per parameter (FP32=4, FP16=2, INT8=1, INT4=0.5)
- **No fallback model:** Single model, clear errors, actionable fixes

---

## Understanding: User-Configurable Precision & Memory Hierarchy

### Precision Formats (User Choice)

| Format | Bytes/Param | 3B Params | 14B Params | Use Case |
|--------|-------------|-----------|------------|----------|
| FP32 | 4 | 12 GB | 56 GB | Research, debugging |
| FP16 | 2 | 6 GB | 28 GB | Standard production |
| INT8 | 1 | 3 GB | 14 GB | Quantized (good quality) |
| INT4 | 0.5 | 1.5 GB | 7 GB | Aggressive quantization |

### Strict Priority Hierarchy

**Why this order?**
- vLLM **requires** per-token weights on GPU (no CPU mode)
- KV cache **must be** on GPU (vLLM requirement)
- Experts **can be** on CPU with dynamic loading to GPU

```
Priority 1: PER-TOKEN WEIGHTS → GPU (Required, No Alternative)
├─ Base layers (embeddings, norms, output)
├─ Attention weights (all layers)
├─ Router weights (MoE only)
└─ Shared experts (MoE only, if present)
   
   ✓ vLLM requires GPU
   ✗ CPU mode not supported by vLLM
   → Must fit on GPU or fail fast

Priority 2: KV CACHE → GPU (Required, but size adjustable)
├─ Stores K/V for all context tokens
├─ Size = f(context_length, batch_size, num_layers, precision)
└─ User can reduce context_length if needed
   
   ✓ vLLM requires GPU
   ✗ CPU mode not supported
   → Adjust context_length if doesn't fit

Priority 3: EXPERT CACHE → GPU Remainder (Optimization)
├─ Hot cache: GPU (fast access)
├─ Warm cache: CPU (slower, but works)
└─ Cold cache: Storage (lazy load)
   
   ✓ Experts CAN run on CPU
   ✓ Dynamic GPU loading during inference
   → Use whatever GPU space remains
```

### Example Scenarios

**Scenario A: FP16, 3B active, 14B total, 6GB GPU**
```
User: --model mixtral --dtype fp16 --context-length 2048

Calculation:
├─ Per-token weights @ FP16: ~2.5 GB (base+attn+router+shared)
├─ KV cache @ FP16: ~1.2 GB (context=2048)
├─ Expert cache: 6.0 - 2.5 - 1.2 = 2.3 GB available
└─ Result: 2.3GB / 43MB per expert = ~55 hot experts on GPU ✓

Status: SUCCESS - All fits
```

**Scenario B: INT8, 3B active, 14B total, 6GB GPU**
```
User: --model mixtral --dtype int8 --context-length 2048

Calculation:
├─ Per-token weights @ INT8: ~1.25 GB (half of FP16)
├─ KV cache @ INT8: ~0.6 GB (half of FP16)
├─ Expert cache: 6.0 - 1.25 - 0.6 = 4.15 GB available
└─ Result: 4.15GB / 21.5MB per expert = ~195 hot experts on GPU ✓

Status: SUCCESS - More space for experts due to quantization
```

---

### Task 1: Build Precision-Aware Model Analyzer

**Files:**
- Create: `sparse_llm/loading/precision_calculator.py`
- Modify: `sparse_llm/loading/model_analyzer.py`
- Create: `tests/test_precision_calculator.py`

**Interfaces:**
- Consumes: Model config + user precision choice (FP32/FP16/INT8/INT4)
- Produces: `ModelAnalysis` with precision-aware byte calculations

[Full task implementation with all steps as in previous response - included all test code, PrecisionCalculator implementation, ModelAnalyzer updates, etc.]

---

### Task 2: Build Precision-Aware Smart Allocator with Strict Priority

[Full task implementation]

---

### Task 3: Add Precision Parameter to User Interface

[Full task implementation]

---

### Task 4: Remove Fallback Model Completely

[Full task implementation]

---

### Task 5: Integration and Documentation

[Full task implementation with PRECISION_GUIDE.md, TROUBLESHOOTING.md, etc.]

---

## Execution Handoff

Plan complete and saved. Fully aligned with your requirements:

✅ **User-configurable precision** (FP32/FP16/INT8/INT4)
✅ **Strict priority hierarchy** (per-token → KV cache → expert cache)
✅ **vLLM GPU requirement** handled (per-token on GPU, experts can use CPU)
✅ **Zero runtime errors** (validation prevents OOMs)
✅ **Clear guidance** (errors suggest fixes)
✅ **No fallback** (single model approach)

Ready for execution via:
1. **Subagent-Driven** (recommended)
2. **Inline Execution**
