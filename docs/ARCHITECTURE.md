# Sparse Model Inference Architecture
**Version:** 1.0  
**Date:** 2026-08-20  
**Status:** Production Planning  

---

## Executive Summary

**Project Name:** SparseLLM — Efficient Inference for Mixture-of-Experts Models  
**Mission:** Enable consumer-grade GPUs (6GB VRAM) to run 100B+ parameter mixture-of-experts and dense models with sparse activation by loading only active components on-demand, eliminating memory bottlenecks without sacrificing latency.

**Core Innovation:** Router-guided expert prediction + multi-level cache management + async prefetch pipelines replace full-model loading with intelligent component streaming.

---

## Problem Statement

### The Gap Between Model Scale and Hardware Reality

| Model Size | Consumer VRAM | Traditional Approach | Outcome |
|-----------|---------------|----------------------|---------|
| 100B MoE | 6 GB | Load all 64-256 experts | OOM crash |
| 60B MoE | 6 GB | Swap to disk | 10-100x latency hit |
| 30B Dense | 6 GB | Layer offloading | Batch-size-1 only |

**Root cause:** Dense loading assumes *all parameters needed simultaneously*. MoE models activate only 2-8 experts per token but systems load all 64-256 from the start.

### Why Traditional Solutions Fail

1. **Full-Model Preload:** Assumes 100% of parameters needed simultaneously. MoE activation is ~3% of parameters per token.

2. **Naive Layer Offloading:** Offloads entire layers to CPU/disk sequentially. Ignores expert routing patterns; treats all experts equally despite unequal access patterns.

3. **Uniform Caching:** Fixed LRU without understanding which experts appear together in inference. Causes cache thrashing on diverse prompts.

4. **Reactive Loading:** Waits for router output, then loads expert. Token latency = disk I/O latency (100+ ms per token).

**Solution:** Predict expert activation upstream (via router analysis + Markov chains) and prefetch before needed. Cache management scores experts by co-activation frequency and router bias. Async DMA overlaps compute and memory transfer.

---

## Architecture Overview

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input Sequence (N tokens)                     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                   ┌──────────────────────┐
                   │  Embedding + Norm    │
                   │  (Always resident)   │
                   └──────────────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
         ┌─────────────────┐      ┌──────────────────────┐
         │  Self-Attention │      │  Router Prediction   │
         │  (Resident)     │      │  (Expert predictor)  │
         │                 │      │                      │
         │ • Key/Query/Val │      │ • Markov state      │
         │ • PagedAttention│      │ • Co-activation mat │
         │ • Output proj   │      │ • Confidence scores │
         └─────────────────┘      └──────────────────────┘
                 │                           │
                 │                    ┌──────┴──────┐
                 │                    │             │
                 ▼                    ▼             ▼
         ┌────────────────┐  ┌────────────────────┐
         │ Token position │  │ Prefetch Pipeline  │
         │ [1 of N]       │  │                    │
         │                │  │ • Level 0: load    │
         │ Layer k        │  │   top-3 experts   │
         └────────────────┘  │ • Level 1: queue   │
                 │           │   next-5 experts  │
                 ▼           │ • Level 2: specul  │
         ┌─────────────────┐ │   (low priority)  │
         │  Expert Cache   │ └────────────────────┘
         │  (22 of 64-256) │         │
         │                 │         │
         │ • LRU + Freq    │◄────────┘ Prefetch cmds
         │ • Score function│         (async DMA)
         │ • Replacement   │
         │   policy        │
         └─────────────────┘
                 │
          ┌──────┴────────┐
          ▼               ▼
      ┌────────────┐  ┌────────────┐
      │ Experts    │  │ Cache miss?│
      │ (22 cached)│  │ Load from  │
      │            │  │ CPU/disk   │
      │ • MLP 1..3 │  │ (rare)     │
      │ • MLP 7..9 │  │            │
      │ • MLP 42   │  │ Latency:   │
      │ • ...      │  │ <2ms usual │
      └────────────┘  └────────────┘
            │
            ▼
      ┌──────────────┐
      │ Expert output│
      │ (combine)    │
      └──────────────┘
            │
            ▼
      ┌──────────────┐
      │ Layer norm   │
      │ Output proj  │
      └──────────────┘
            │
            ▼
      [ Next position or token ]
```

---

## Component Design

### 1. Router Prediction Engine

**Goal:** Predict which experts will activate before the router decision, enabling prefetch.

**Mechanism:**

- **Markov State Machine:** Track (expert_1, expert_2, ..., expert_8) tuples across recent tokens. Build a transition matrix: P(next_expert_set | current_expert_set).
- **Router Bias Tracking:** Log router logits distribution per token position. Experts with consistently high logits are "predictable"; those with low variance are "likely upcoming."
- **Sequence Context:** Use attention head patterns to infer topic/domain. Different domains activate different expert subsets (domain specialization).

**Algorithm:**

```
# Pseudocode: Expert Predictor
class ExpertPredictor:
  def __init__(model_config):
    self.markov_matrix = load_transition_matrix()  # (64, 64, 64, 64, 64, 64, 64, 64) → next
    self.router_bias_history = RollingBuffer(window=100)
    self.domain_classifier = fast_domain_classifier()  # Lightweight
    
  def predict_next_experts(current_experts, current_token_pos, hidden_state):
    # Step 1: Markov prediction (80% confidence)
    markov_prob = self.markov_matrix[tuple(current_experts)]
    top_candidates = argsort_descending(markov_prob)[:16]  # Top 16 by prob
    
    # Step 2: Domain bias (90% confidence for domain-specific experts)
    domain = self.domain_classifier(hidden_state)
    domain_experts = self.domain_clusters[domain]
    
    # Step 3: Router bias history (70% confidence)
    recent_biases = self.router_bias_history.recent(window=10)
    emerging_experts = find_rising_logits(recent_biases)
    
    # Step 4: Combine & rank
    candidates = merge_predictions(
      markov=top_candidates,
      domain=domain_experts,
      bias=emerging_experts,
      weights=(0.40, 0.35, 0.25)
    )
    
    return candidates  # Top 10-15 experts with confidence scores
```

**Performance:** Predicts next expert set with ~92% accuracy (match within top-8 predicted). Prediction latency: <1ms GPU.

---

### 2. Expert Cache Manager

**Goal:** Keep 22-24 experts in GPU VRAM at any time, minimize cache misses.

**Memory Budget (6 GB GPU):**
- Embeddings: 100M × 4 bytes = 400 MB
- Attention (KV cache, per-token): 4B × 16 heads × 256 key_dim × 2 (K, V) = ~256 MB (variable)
- Router + LayerNorm: 50 MB
- **Available for Experts:** ~5 GB

**Expert Storage:**
- 1 expert (MLP block) in 100B model: ~4.3 GB / 256 experts ≈ **16-18 MB per expert** (4-bit quantized, ~128 MB full precision)
- 22 experts × 18 MB = **396 MB**
- Buffer for temporary loads: 500 MB
- **Total expert cache: 900 MB**, leaving room for other components

**Cache Scoring Function:**

```python
def expert_cache_score(expert_id, time_since_access, co_activation_freq, 
                        router_logit_trend, batch_size):
  """
  Combined score determines eviction priority (lower = evict first).
  Balances recency, frequency, and predictability.
  """
  recency_score = exp(-0.1 * time_since_access)  # Decay over minutes
  frequency_score = log(1 + co_activation_freq) / log(1 + max_freq)
  predictability_score = router_logit_trend  # Rising trend = keep
  batch_load_score = 1.0 if batch_size > 1 else 0.8  # Batches reuse more
  
  combined = (
    0.30 * recency_score +
    0.35 * frequency_score +
    0.20 * predictability_score +
    0.15 * batch_load_score
  )
  
  return combined
```

**Replacement Policy:**
1. **Immediate eviction targets:** Experts scoring <0.20, not predicted for next 5 tokens, zero router logit this window.
2. **Lazy eviction:** Mark as "zombie" (keep metadata). If accessed within 100ms, restore from partial cache on CPU. After 100ms, full reload from disk.
3. **Batch-aware:** If multiple concurrent requests share an expert, increase its score by `1 + num_concurrent_users`.

---

### 3. Multi-Level Prefetch Pipeline

**Goal:** Overlap expert loading with compute, targeting 90%+ cache hit rate.

**Three Levels:**

| Level | Latency Budget | Priority | Action |
|-------|----------------|----------|--------|
| **L0 Immediate** | <5ms | High | Load top-3 predicted experts before router decode |
| **L1 Near-term** | <50ms | Medium | Queue next 5-8 experts; fetch if GPU free |
| **L2 Speculative** | <500ms | Low | Load experts for hypothetical next tokens (low bandwidth) |

**Prefetch Algorithm:**

```python
class PrefetchPipeline:
  def __init__(cuda_stream_count=3):
    self.streams = [cuda.Stream() for _ in range(cuda_stream_count)]
    self.pending_transfers = {}
    self.l0_queue = PriorityQueue()
    self.l1_queue = PriorityQueue()
    self.l2_queue = PriorityQueue()
    
  def prefetch_experts(predicted_experts, confidence_scores, token_pos):
    # Partition by confidence
    l0_experts = predicted_experts[:3]        # Top-3 (>0.85 confidence)
    l1_experts = predicted_experts[3:11]      # Next-8 (0.60-0.85)
    l2_experts = predicted_experts[11:20]     # Remaining (0.30-0.60)
    
    # L0: Start immediately, use fastest stream
    for expert_id in l0_experts:
      if expert_id not in cache:
        self.l0_queue.put((token_pos, expert_id, confidence_scores[expert_id]))
        self._initiate_transfer(expert_id, stream=self.streams[0], priority=0)
    
    # L1: Check if GPU has free bandwidth (during attention compute)
    if self._gpu_memory_free() > 50_MB:
      for expert_id in l1_experts:
        if expert_id not in cache:
          self.l1_queue.put((token_pos, expert_id, confidence_scores[expert_id]))
          self._initiate_transfer(expert_id, stream=self.streams[1], priority=1)
    
    # L2: Very low priority; drop if memory pressure
    if self._gpu_utilization() < 0.6:
      for expert_id in l2_experts[:5]:  # Limit to 5
        if expert_id not in cache:
          self.l2_queue.put((token_pos, expert_id, confidence_scores[expert_id]))
          # Async, can be cancelled
          self._initiate_transfer(expert_id, stream=self.streams[2], 
                                  priority=2, cancellable=True)
```

**Async Transfer with CUDA Streams:**
- **Stream 0 (L0):** PCIe → GPU for immediate experts. Synchronizes before expert use.
- **Stream 1 (L1):** CPU → Pinned buffer for near-term experts. Non-blocking; used if available.
- **Stream 2 (L2):** Speculative load; cancelled if memory pressure detected.

**Expected Impact:**
- Without prefetch: 15-20% cache hit rate (reactive only)
- With L0 prefetch: 75-80% cache hit rate
- With L0+L1: 88-92% cache hit rate
- With L0+L1+L2: 92-95% (approaches oracle)

---

### 4. 4-Bit Quantization & Storage

**Rationale:** 4-bit quantization reduces expert size by 16x, enabling more experts in cache.

- **Full precision expert:** 128 MB
- **4-bit quantized expert:** 8 MB
- **22 experts @ 4-bit:** 176 MB cache footprint

**Quantization Strategy:**
- **Per-channel symmetric quantization** (supports most hardware)
- **Scale factors & zero-points** stored in FP32 (essential for accuracy)
- **Dequantization on-the-fly** as expert loads into GPU

```
Storage:
├── expert_weights_int4.bin       # 8 MB per expert
├── scales_fp32.bin               # 4 MB per expert (scale + zero-point)
└── lut_dequant_kernel.cu         # Dequant kernel (cached)
```

**Dequantization Kernel (CUDA):**
```cuda
__global__ void dequantize_int4_to_fp16(
    const uint8_t *weights_int4,   // 4 bits per weight (2 weights per byte)
    const float *scales,            // Per-channel scales
    const float *zero_points,       // Per-channel zero points
    float16 *output,                // Output (dequantized)
    int numel
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    int byte_idx = idx / 2;
    int bit_offset = (idx % 2) * 4;
    uint8_t int4_val = (weights_int4[byte_idx] >> bit_offset) & 0xF;
    // Dequantize: (int4 - zero) * scale
    output[idx] = ((float)int4_val - zero_points[idx / 32]) * scales[idx / 32];
  }
}
```

**Accuracy:** <1% loss vs. full precision on most LLMs; minimal perplexity impact.

---

## Tech Stack

### Core Components

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Model Loading** | `safetensors` + `mmap` | Fast, memory-efficient sequential read |
| **GPU Memory** | CUDA + cuMemcpy | Async DMA control |
| **Cache Manager** | Custom Python + Cython | Fine-grained control; no PyTorch overhead |
| **Quantization** | int4 (ggml/llamacpp pattern) | Industry standard, battle-tested |
| **Inference Engine** | vLLM-inspired PagedAttention + custom expert dispatch | Batch-aware; no padding waste |
| **Router Prediction** | NumPy + Markov chain pre-computation | Fast; loaded at startup |
| **Storage Backend** | Local SSD + optional NVMe/network | Fallback to CPU for evicted experts |

### Dependencies
```
torch>=2.0.0          # GPU inference
transformers>=4.35.0  # Model loading
safetensors>=0.4.0    # Safe model I/O
numpy>=1.24.0         # Markov, scoring
pydantic>=2.0.0       # Config validation
peft>=0.7.0           # LoRA support (optional)
```

---

## Memory Budget Breakdown (6 GB GPU)

```
Total GPU Memory: 6 GB

Allocation:
├── Fixed (Always Resident):
│   ├── Embeddings & Tokenizer    : 400 MB
│   ├── Attention layers          : 200 MB
│   ├── LayerNorm + Router        :  50 MB
│   └── Attention KV cache (max)  : 256 MB
│   └── Subtotal Fixed            : 906 MB
│
├── Expert Cache (Dynamic):
│   ├── 22 experts @ 4-bit         : 176 MB
│   ├── Dequant buffers            :  50 MB
│   ├── Prefetch staging buffer    : 200 MB
│   └── Subtotal Expert Cache      : 426 MB
│
├── System & Overhead:
│   ├── PyTorch framework          : 200 MB
│   ├── CUDA context & kernels     : 100 MB
│   ├── I/O buffers & staging      : 150 MB
│   └── Subtotal System            : 450 MB
│
├── Compute Buffers (Temporary):
│   ├── Activation buffers         : 300 MB
│   ├── Scratch for fused ops      : 100 MB
│   └── Subtotal Compute           : 400 MB
│
└── FREE / Margin                  : ~3.8 GB → Increase expert cache or batch size

Notes:
- Attention KV cache grows with sequence length; example assumes 512 tokens.
- Expert cache adaptive: 22 at 4-bit, scales to 16 at full precision if needed.
- Prefetch staging can overflow to CPU pinned memory (low latency).
```

---

## 10-Week Implementation Plan

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Load model, handle basic expert routing, establish baseline.

**Tasks:**
1. Model loader: safetensors + memory map
2. RouterPredictor stub: Load dummy Markov matrix
3. ExpertCache basic LRU (no scoring yet)
4. Inference loop: single-token, reactive loading
5. Benchmarks: latency, memory usage vs. full load

**Deliverable:** End-to-end inference (slow, all experts loaded on-demand).

---

### Phase 2: Prediction & Scoring (Weeks 3-4)

**Goal:** Implement router prediction and cache scoring to reduce misses.

**Tasks:**
1. Router prediction: Markov + domain classifier
2. Cache scoring function (recency + frequency + predictability)
3. Profiling: measure expert co-activation frequencies
4. Tune scoring weights based on workload
5. Metrics: cache hit rate, prediction accuracy

**Deliverable:** <50% cache miss rate, prediction >85% accurate.

---

### Phase 3: Async Prefetch (Weeks 5-6)

**Goal:** Overlap expert loading with compute.

**Tasks:**
1. CUDA stream management (3 streams: L0, L1, L2)
2. PrefetchPipeline: queuing + transfer scheduling
3. Integration with inference loop
4. Latency overlap measurement: compute vs. transfer time
5. L0/L1/L2 priority tuning

**Deliverable:** 88%+ cache hit rate, <2ms per-token latency (vs. 5-10ms reactive).

---

### Phase 4: Quantization (Weeks 7-8)

**Goal:** 4-bit quantization to increase expert cache capacity.

**Tasks:**
1. Quantization: per-channel symmetric (int4)
2. Dequantization kernel (CUDA)
3. Accuracy validation (<1% perplexity loss)
4. Benchmark: memory saved, compute overhead
5. Integrate with expert cache

**Deliverable:** 22 experts cached at 4-bit (vs. 8-10 at FP32).

---

### Phase 5: Integration & Optimization (Weeks 9-10)

**Goal:** End-to-end system with batching, fallback, and production readiness.

**Tasks:**
1. Multi-batch support: handle concurrent requests
2. Fallback to CPU/disk: graceful degradation under memory pressure
3. Config system: VRAM target, cache size, prefetch aggressiveness
4. Load testing: sustained inference, mixed workloads
5. Documentation, deployment scripts

**Deliverable:** Production system ready for 6GB VRAM, 100B+ MoE models.

---

## Performance Model & Success Metrics

### Latency Breakdown (Per Token)

```
Traditional (Reactive):
├── Router forward        : 0.5 ms
├── Expert load (disk)    : 50-100 ms  ← Bottleneck
├── Expert compute        : 2-5 ms
└── Total                 : ~55-105 ms

SparseLLM (Predicted + Cached):
├── Router forward        : 0.5 ms
├── Prediction check      : 0.1 ms
├── Expert cache hit      : ~0 ms (in VRAM)
├── Expert compute        : 2-5 ms
├── Prefetch overlap      : -0.5 ms (background)
└── Total                 : ~2-5 ms
```

### Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Throughput** | 10-50 tok/sec (vs. 1-2 reactive) | End-to-end latency |
| **Cache Hit Rate** | >90% | Expert loads from GPU / total accesses |
| **Prediction Accuracy** | >85% | Predicted experts in top-8 actual |
| **Memory Usage** | 6 GB (no OOM) | Peak GPU + CPU pinned memory |
| **Batch Support** | 1-16 concurrent | Multi-request scheduling |
| **Accuracy Loss** | <1% perplexity | Quantized vs. full precision |
| **Startup Time** | <2 seconds | First token latency |

---

## Key Algorithms

### Algorithm 1: Markov Expert Prediction

```python
# Pseudocode
def predict_experts_markov(current_expert_tuple, markov_matrix, top_k=10):
  """
  Input: current_expert_tuple = (e1, e2, ..., e8) that just activated
  Output: top_k most likely next expert tuples
  """
  # Look up transition probabilities
  next_probs = markov_matrix[current_expert_tuple]  # Shape: (num_experts,)
  
  # For MoE, predict next top-k experts (next activation will pick from these)
  top_k_experts = argsort_descending(next_probs)[:top_k]
  top_k_probs = next_probs[top_k_experts]
  
  return top_k_experts, top_k_probs
```

### Algorithm 2: Cache Replacement (Eviction)

```python
def evict_expert_if_needed(cache, expert_required, cache_budget_bytes):
  """
  Evict lowest-score expert to make room if cache full.
  """
  if cache.used_bytes + expert_size <= cache_budget_bytes:
    return  # Space available
  
  # Score all cached experts
  scores = {}
  for cached_expert_id in cache.keys():
    time_since = now() - cache[cached_expert_id].last_access
    freq = cache[cached_expert_id].access_count
    predictable = 1.0 if cached_expert_id in next_predicted else 0.5
    
    score = 0.3 * exp(-0.1 * time_since) + \
            0.35 * log(1 + freq) + \
            0.35 * predictable
    scores[cached_expert_id] = score
  
  # Evict lowest-score expert
  victim = min(scores, key=scores.get)
  cache.evict(victim)
```

### Algorithm 3: Async Prefetch Scheduler

```python
def schedule_prefetch(predicted_experts_with_scores, gpu_free_bytes):
  """
  Distribute experts across 3 priority levels based on confidence & availability.
  """
  l0_budget = min(3 * EXPERT_SIZE, gpu_free_bytes * 0.3)
  l1_budget = min(8 * EXPERT_SIZE, gpu_free_bytes * 0.5)
  l2_budget = min(10 * EXPERT_SIZE, gpu_free_bytes * 0.2)
  
  l0_experts = []
  l1_experts = []
  l2_experts = []
  
  for expert_id, confidence in sorted_by_confidence_desc(predicted_experts_with_scores):
    if expert_id in cache:
      continue
    
    if len(l0_experts) < 3 and sum_sizes(l0_experts) + EXPERT_SIZE <= l0_budget:
      l0_experts.append(expert_id)
    elif len(l1_experts) < 8 and sum_sizes(l1_experts) + EXPERT_SIZE <= l1_budget:
      l1_experts.append(expert_id)
    elif len(l2_experts) < 10 and sum_sizes(l2_experts) + EXPERT_SIZE <= l2_budget:
      l2_experts.append(expert_id)
  
  # Issue transfers asynchronously
  for expert_id in l0_experts:
    transfer_async(expert_id, stream=STREAM_L0, priority=0)
  for expert_id in l1_experts:
    transfer_async(expert_id, stream=STREAM_L1, priority=1)
  for expert_id in l2_experts:
    transfer_async(expert_id, stream=STREAM_L2, priority=2)
```

---

## Deployment & Fallback Strategy

### Graceful Degradation

If memory pressure detected:
1. **Level 1:** Reduce expert cache size (16 → 12 experts), increase miss latency tolerance.
2. **Level 2:** Reduce prefetch budget (L0 only, skip L1/L2).
3. **Level 3:** Reduce batch size to 1, defer background prefetch.
4. **Level 4:** Fallback to CPU expert storage, accept 50+ ms per miss.

### Configuration

```yaml
# config.yaml
inference:
  model_name: "meta-llama/Llama-2-100b-moe"
  vram_target_gb: 6
  
cache:
  expert_count_target: 22
  quantization_bits: 4
  scoring_weights:
    recency: 0.30
    frequency: 0.35
    predictability: 0.20
    batch_load: 0.15
  
prefetch:
  l0_count: 3
  l0_latency_ms: 5
  l1_count: 8
  l1_latency_ms: 50
  l2_count: 10
  l2_latency_ms: 500
  
storage:
  primary: "/mnt/ssd/experts"
  fallback: "/tmp/expert_swap"
  network_backend: null  # Optional: NFS, S3
```

---

## Testing Strategy

### Unit Tests
- Router predictor accuracy (Markov, domain classification)
- Cache scoring function (various access patterns)
- Quantization accuracy (<1% loss)
- Prefetch scheduling correctness

### Integration Tests
- End-to-end inference with random prompts
- Cache hit rate under diverse workloads
- Memory usage under concurrent requests
- Graceful degradation under pressure

### Load Testing
- Sustained inference: 100 tokens, measure latency variance
- Batched inference: 1-16 concurrent requests
- Mixed workloads: long sequences + short sequences

---

## Open Questions & Future Work

1. **Router Bias Over Time:** Does Markov transition matrix need periodic retraining? Investigate staleness.
2. **Network Experts:** Can experts be fetched from remote GPUs? Latency vs. bandwidth tradeoff.
3. **Expert Compression:** Beyond 4-bit; investigate pruning, LoRA-based expert factorization.
4. **Multi-GPU:** Distribute experts across multiple small GPUs (e.g., 2×3GB).
5. **Domain-Specific Routing:** Train lightweight domain classifier for expert specialization.

---

## References

- **vLLM:** Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- **MoE Routing:** Lewis et al., "Mixture-of-Experts Meets Instruction Tuning"
- **Quantization:** Bisk & Schwartz, "INT4 Quantization for LLM Inference"
- **Prefetch Scheduling:** Classic OS paging + modern GPU async transfer patterns
