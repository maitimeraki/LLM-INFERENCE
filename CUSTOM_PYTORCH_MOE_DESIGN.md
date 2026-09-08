# Custom PyTorch MoE Inference System - Design Specification

## Executive Summary

**Goal:** Achieve 20-30 tokens/sec decode speed for MoE models in low-resource environments through custom PyTorch implementation with intelligent prefill/decode separation and expert caching.

**Core Innovation:** Batch-compute all router decisions during prefill phase, then maintain hot experts in GPU during decode phase to eliminate per-token loading overhead.

---

## System Architecture Overview

### Three-Phase Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: PREFILL                          │
│  Process entire prompt (1k-100k tokens) in one pass         │
│  - Batch attention across all tokens                         │
│  - Compute ALL router decisions upfront                      │
│  - Preload required experts into GPU                         │
│  - Process tokens grouped by expert assignment               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 2: DECODE                           │
│  Generate new tokens one at a time                          │
│  - Route only the new token                                 │
│  - Reuse KV cache from prefill                              │
│  - Keep hot experts in GPU cache                            │
│  - Swap experts only when necessary                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                PHASE 3: CLEANUP                              │
│  Release resources after generation complete                │
│  - Clear KV cache                                           │
│  - Evict experts from GPU                                   │
│  - Reset state for next request                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Design

### Component 1: Router Calculator

**Purpose:** Determine which experts should process which tokens

**Responsibilities:**
- Accept hidden states from attention layer
- Compute router logits for all tokens
- Return top-k expert indices and weights
- Support both batch (prefill) and single-token (decode) modes

**Input:**
- Hidden states tensor: shape [batch_size, sequence_length, hidden_dim]
- Number of experts to select (k=2 typically)

**Output:**
- Expert indices: shape [batch_size, sequence_length, k]
- Expert weights: shape [batch_size, sequence_length, k]

**Performance Requirements:**
- Prefill: Must handle 1k-100k tokens in <500ms
- Decode: Must handle 1 token in <5ms
- Should run entirely on GPU (no CPU transfers)

---

### Component 2: Expert Cache Manager

**Purpose:** Maintain hot experts in GPU memory with LRU eviction

**Responsibilities:**
- Track which experts are currently in GPU memory
- Load experts from CPU/disk when needed
- Evict least-recently-used experts when cache full
- Provide fast lookup for expert availability

**Cache Structure:**
- Maximum capacity: 4-6 experts (configurable based on VRAM)
- Each expert: ~3-4GB for 7B parameter MoE
- Total cache size: 12-24GB VRAM

**Cache Operations:**

1. **Check Operation (constant time):**
   - Input: Expert ID
   - Output: Boolean (is expert in cache?)
   - Latency: <1ms

2. **Load Operation (blocking):**
   - Input: Expert ID
   - Action: Load weights from CPU RAM to GPU
   - Latency: 100-200ms per expert
   - Trigger eviction if cache full

3. **Evict Operation:**
   - Strategy: LRU (Least Recently Used)
   - Action: Move expert weights from GPU to CPU
   - Update tracking metadata

**Optimization:** Maintain access timestamp for each cached expert to enable efficient LRU decisions

---

### Component 3: Attention Engine

**Purpose:** Compute attention for both prefill and decode phases with KV cache

**Two Operating Modes:**

**Mode A: Prefill Attention**
- Input: All prompt tokens [batch, seq_len, hidden_dim]
- Compute: Full self-attention across all tokens
- Output: Hidden states + KV cache
- Cache: Store keys/values for all prompt tokens
- Latency: ~1-3 seconds for 1k tokens

**Mode B: Decode Attention**
- Input: Single new token [batch, 1, hidden_dim]
- Compute: Attention over new token + cached KV
- Output: Hidden states for new token
- Cache: Append new token's KV to existing cache
- Latency: ~20-30ms per token (target)

**KV Cache Structure:**
- Keys cache: [batch, num_layers, seq_len, num_heads, head_dim]
- Values cache: [batch, num_layers, seq_len, num_heads, head_dim]
- Size: ~2-4GB for 4k context length
- Management: Append-only during decode, clear between requests

---

### Component 4: Expert Processor

**Purpose:** Execute expert feed-forward networks on assigned tokens

**Two Processing Strategies:**

**Strategy A: Batched Processing (Prefill)**
1. Group tokens by expert assignment
2. For each expert:
   - Load expert if not cached
   - Process all tokens assigned to this expert in batch
   - Store outputs
3. Reorder outputs to match original token sequence

**Strategy B: Single-Token Processing (Decode)**
1. Get expert assignments for new token (usually 2 experts)
2. Check if experts are cached
3. If not cached, load expert (triggers LRU eviction)
4. Process token through both experts
5. Combine expert outputs using router weights

**Expert Loading Priority:**
- During prefill: Load experts in order of frequency (most-used first)
- During decode: Keep last-used experts cached
- Predictive: Optionally prefetch likely next experts

---

### Component 5: Worker Pool for Async Operations

**Purpose:** Overlap computation with data transfers

**Worker Types:**

**Worker Type 1: Expert Loader**
- Runs in background thread
- Loads next-likely expert from disk/RAM to GPU
- Operates while current token is being processed
- Reduces effective swap latency from 150ms to near-zero

**Worker Type 2: Router Prefetcher**
- Computes router decisions ahead of time when possible
- Useful for speculative execution scenarios
- Lower priority than main decode path

**Coordination Strategy:**
- Main thread: Runs attention + current expert processing
- Background thread: Loads next expert in parallel
- Synchronization: Only wait if expert not ready when needed

---

## Data Flow Specification

### Flow 1: Prefill Phase (Prompt Processing)

**Step 1: Input Preparation**
- Receive tokenized prompt: [batch=1, seq_len=N tokens]
- Validate sequence length is within limits
- Initialize empty KV cache for all layers

**Step 2: Attention Pass**
- Process all tokens through each transformer layer
- For each layer:
  - Compute Q, K, V matrices for all tokens
  - Apply self-attention (all tokens attend to all previous)
  - Store K, V in cache for decode phase
- Output: Hidden states [1, N, hidden_dim]

**Step 3: Batch Router Computation**
- Pass all hidden states through router network
- Get router logits: [1, N, num_experts]
- Compute top-k for each token: [1, N, k=2]
- Result: Expert assignments for all N tokens

**Step 4: Expert Usage Analysis**
- Count frequency of each expert across all tokens
- Identify which experts are needed
- Preload top-4 most frequent experts into GPU cache

**Step 5: Grouped Expert Processing**
- Create token groups by expert assignment
- For each required expert:
  - Ensure expert is loaded in GPU
  - Process all tokens assigned to this expert
  - Store expert outputs
- Combine expert outputs using router weights
- Reorder to original token sequence

**Step 6: Output Preparation**
- Pass combined outputs through layer norm
- Generate logits for next token prediction
- Store final hidden state for decode initialization
- Keep KV cache in memory

**Prefill Complete:** System is now ready for decode phase

---

### Flow 2: Decode Phase (Token Generation)

**Step 1: Single-Token Attention**
- Input: Previous token's hidden state [1, 1, hidden_dim]
- For each layer:
  - Compute Q for new token
  - Retrieve cached K, V for all previous tokens
  - Compute attention: new token attends to all cached tokens
  - Append new K, V to cache
- Output: New hidden state [1, 1, hidden_dim]

**Step 2: Single-Token Routing**
- Pass new hidden state through router
- Get router logits: [1, 1, num_experts]
- Compute top-k=2 expert indices and weights
- Result: 2 experts for this token

**Step 3: Expert Cache Check**
- For each of the 2 selected experts:
  - Check if expert is in GPU cache
  - If YES: Mark as "ready"
  - If NO: Trigger load operation (100-200ms)

**Step 4: Expert Processing**
- Process token through first expert → output1
- Process token through second expert → output2
- Combine using router weights: output = w1*output1 + w2*output2

**Step 5: Token Prediction**
- Pass combined output through LM head
- Get logits for vocabulary: [1, 1, vocab_size]
- Apply sampling strategy (greedy/top-k/nucleus)
- Select next token ID

**Step 6: Loop Control**
- Append predicted token to sequence
- Update KV cache with new token's keys/values
- Check stopping condition:
  - EOS token generated?
  - Maximum length reached?
  - User-specified stop sequence?
- If not done: Repeat from Step 1 with new token

**Decode Complete:** Return generated sequence

---

### Flow 3: Expert Swapping (When Cache Miss)

**Trigger:** Decode needs expert not in GPU cache

**Step 1: LRU Victim Selection**
- Check current GPU cache size
- If at capacity (4-6 experts):
  - Review access timestamps for all cached experts
  - Select least recently used expert for eviction
  - Mark for removal

**Step 2: Eviction (if needed)**
- Transfer victim expert weights from GPU to CPU RAM
- This is async - doesn't block if not needed immediately
- Free GPU memory
- Update cache tracking metadata

**Step 3: Expert Loading**
- Identify source location (CPU RAM or disk)
- Transfer expert weights to GPU VRAM
- Register in cache with current timestamp
- Mark as "ready"

**Step 4: Resume Decode**
- Continue with expert processing
- Update LRU tracking

**Optimization:** Predictive prefetching can eliminate this wait

---

## Optimization Strategies

### Strategy 1: Prefill Optimization

**Objective:** Process 1k-100k tokens as fast as possible (throughput)

**Techniques:**

1. **Batch Attention Operations**
   - Use GPU tensor cores for matrix multiplication
   - Process all tokens in parallel
   - Target: 200-500 tokens/second throughput

2. **Memory-Efficient Expert Loading**
   - Load experts in frequency order (most-used first)
   - Keep top-4 in GPU for entire prefill
   - Reduces loading overhead by 60-80%

3. **Computation Fusion**
   - Fuse layer norm + router computation
   - Reduce kernel launches
   - Save 10-20% computation time

4. **Mixed Precision**
   - Use FP16 for attention (faster, lower memory)
   - Keep experts in FP32 or INT8 based on accuracy needs
   - 2x speedup for attention

---

### Strategy 2: Decode Optimization

**Objective:** Minimize latency per token (20-30ms target)

**Techniques:**

1. **Expert Cache Hit Rate Maximization**
   - Keep 4-6 experts hot in GPU
   - 80%+ cache hit rate → most tokens generate in 30-50ms
   - 20% cache miss → some tokens take 150ms (acceptable)

2. **KV Cache Efficiency**
   - Store in contiguous memory
   - Use in-place append operations
   - Minimize host-device transfers

3. **Attention Kernel Optimization**
   - Implement or use FlashAttention
   - Reduces decode attention from 20ms to 5-10ms
   - Critical for hitting 30+ tok/s

4. **Router Computation Offload**
   - Router network is small (few MB)
   - Can run on GPU without blocking expert execution
   - Overlap router compute with expert loading

---

### Strategy 3: Expert Swapping Optimization

**Objective:** Hide 100-200ms loading latency

**Techniques:**

1. **Predictive Prefetching**
   - Analyze router patterns during prefill
   - Predict likely experts for decode
   - Prefetch in background before needed

2. **Async Pipeline**
   - Load expert N+1 while processing token N
   - Requires: Router must predict 1 token ahead
   - Eliminates swap latency entirely when prediction correct

3. **Partial Loading**
   - Load only most-used expert layers first
   - Reduces loading time by 30-50%
   - Trade accuracy for speed (optional)

4. **Expert Quantization**
   - Store experts as INT8 or INT4 on disk
   - Convert to FP16 when loading to GPU
   - Reduces loading time by 2-4x
   - Reduces disk footprint by 2-4x

---

### Strategy 4: Memory Management

**GPU Memory Budget (24GB example):**

```
Model Static Components:        6 GB
  - Attention layers
  - Router networks  
  - Embeddings/LM head

Expert Cache (4 experts):      16 GB
  - Expert weights @ 4GB each

KV Cache (4k context):          3 GB
  - Keys and values across layers

Working Memory:                  2 GB
  - Intermediate activations
  - Computation buffers

----------------------------------
Total:                         27 GB (need memory optimization)
```

**Optimization:**
- Use INT8 experts: 16GB → 8GB
- Use efficient attention: 3GB → 1.5GB
- Revised total: ~18GB (fits comfortably)

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)

**Milestone 1.1: Router Calculator**
- Implement forward pass for router network
- Support batch mode (prefill) and single mode (decode)
- Test with dummy hidden states
- Validate output shapes and ranges

**Milestone 1.2: Basic Attention Engine**
- Implement standard self-attention
- Add KV cache creation and management
- Test prefill mode with 1k tokens
- Verify attention scores are correct

**Milestone 1.3: Expert Processor Scaffold**
- Create expert loading mechanism from disk/RAM
- Implement single expert forward pass
- Test with one expert, one token
- Validate output against reference implementation

**Deliverable:** Can process prompt and generate 1 token (very slow, no optimization)

---

### Phase 2: Prefill Pipeline (Week 2)

**Milestone 2.1: Batch Router Computation**
- Modify router to process all tokens at once
- Implement top-k selection across batch
- Profile computation time (target: <500ms for 1k tokens)

**Milestone 2.2: Expert Frequency Analysis**
- Count expert usage across all prefill tokens
- Identify top-4 most frequent experts
- Implement expert preloading logic

**Milestone 2.3: Grouped Expert Processing**
- Group tokens by expert assignment
- Process each group through its expert in batch
- Reorder outputs to match original sequence
- Validate output correctness

**Milestone 2.4: Integration Test**
- Run full prefill on 1k token prompt
- Measure end-to-end time
- Verify KV cache is populated correctly
- Test with multiple prompts

**Deliverable:** Prefill processes 1k tokens in 2-5 seconds

---

### Phase 3: Expert Cache System (Week 3)

**Milestone 3.1: Cache Data Structure**
- Implement LRU cache with fixed capacity
- Add check/load/evict operations
- Test cache behavior with dummy experts
- Verify LRU eviction policy works correctly

**Milestone 3.2: Expert Loading Pipeline**
- Implement CPU RAM → GPU transfer
- Measure loading latency (target: <200ms)
- Add memory tracking and validation
- Handle out-of-memory conditions gracefully

**Milestone 3.3: Cache Integration with Decode**
- Modify decode to check cache before expert use
- Trigger loading on cache miss
- Update LRU timestamps after each use
- Profile cache hit rate

**Milestone 3.4: Cache Optimization**
- Implement async expert loading in background thread
- Test overlap of load with compute
- Measure effective latency reduction

**Deliverable:** Decode with 80%+ cache hit rate at 15-20 tok/s

---

### Phase 4: Decode Optimization (Week 4)

**Milestone 4.1: Attention Optimization**
- Integrate FlashAttention or equivalent
- Optimize KV cache access patterns
- Profile decode attention latency (target: <10ms)

**Milestone 4.2: Expert Processing Optimization**
- Fuse operations where possible
- Optimize memory layout for expert weights
- Profile expert compute time (target: <15ms)

**Milestone 4.3: End-to-End Decode Pipeline**
- Integrate all optimized components
- Run full generation (100+ tokens)
- Measure average tok/s
- Profile to identify remaining bottlenecks

**Milestone 4.4: Multi-Request Testing**
- Test with various prompt lengths
- Test with different sampling strategies
- Measure latency distribution (p50, p95, p99)
- Validate generation quality

**Deliverable:** Consistent 20-30 tok/s decode speed with cached experts

---

### Phase 5: Advanced Optimizations (Week 5+)

**Milestone 5.1: Predictive Expert Prefetching**
- Analyze router patterns during prefill
- Implement prefetch policy
- Test effectiveness (measure cache hit rate improvement)

**Milestone 5.2: Expert Quantization**
- Implement INT8/INT4 expert storage
- Add quantization/dequantization on load
- Measure accuracy impact
- Measure speed improvement

**Milestone 5.3: Memory Optimization**
- Implement gradient checkpointing if needed
- Optimize buffer allocation
- Reduce peak memory usage
- Enable larger models or deeper caches

**Milestone 5.4: Monitoring and Profiling**
- Add detailed latency metrics
- Implement cache statistics
- Add memory usage tracking
- Create performance dashboard

**Deliverable:** Production-ready system with monitoring and advanced optimizations

---

## Testing Strategy

### Unit Tests

**Test 1: Router Calculator**
- Input: Known hidden states
- Expected: Correct top-k expert indices
- Verify: Weights sum to 1.0

**Test 2: Expert Cache**
- Scenario: Fill cache beyond capacity
- Expected: LRU expert evicted
- Verify: Cache size remains at limit

**Test 3: Attention with KV Cache**
- Input: Sequence of tokens
- Expected: Same output as without cache
- Verify: Cache grows correctly

**Test 4: Expert Processing**
- Input: Single token, two experts
- Expected: Weighted combination output
- Verify: Matches reference implementation

---

### Integration Tests

**Test 5: Full Prefill**
- Input: 1k token prompt
- Measure: Total time (<5 seconds)
- Verify: KV cache populated, experts loaded

**Test 6: Full Decode**
- Input: Start from prefill state
- Generate: 100 tokens
- Measure: Average tok/s (>20)
- Verify: Coherent output text

**Test 7: Cache Miss Scenario**
- Setup: Force cache to fill with non-optimal experts
- Decode: Generate tokens that need different experts
- Measure: Swapping latency
- Verify: System recovers and continues

---

### Performance Tests

**Test 8: Latency Distribution**
- Generate 1000 tokens
- Measure per-token latency
- Compute p50, p95, p99
- Target: p95 < 50ms

**Test 9: Cache Hit Rate**
- Run on diverse prompts
- Measure cache hits vs misses
- Target: >80% hit rate

**Test 10: Memory Stability**
- Run continuous generation for 1 hour
- Monitor GPU memory usage
- Verify: No memory leaks

**Test 11: Throughput Benchmark**
- Run parallel decode requests (if batching supported)
- Measure total tokens/sec
- Compare against baseline

---

## Integration Points for Coding Agents

### Entry Points

**Entry Point 1: serve.py**
- Add new inference endpoint: `/v1/generate`
- Route requests to CustomMoEInference class
- Handle streaming responses

**Entry Point 2: sparse_llm/integrations/**
- Create new module: `pytorch_moe_inference.py`
- Implement main inference class
- Import and use existing memory budget calculator

**Entry Point 3: Configuration**
- Add config options to `serve.py`:
  - `expert_cache_size`: Number of experts to keep in GPU
  - `prefetch_enabled`: Enable predictive prefetching
  - `quantization_bits`: 4, 8, or 16 bit experts
- Load from environment variables or config file

---

### Existing Systems to Leverage

**Leverage 1: Memory Budget Calculator**
- Location: `sparse_llm/loading/memory_budget_calculator.py`
- Use: Calculate optimal cache size based on available VRAM
- Integration: Call during initialization to set expert_cache_size

**Leverage 2: Model Introspector**
- Location: `sparse_llm/loading/model_introspector.py`
- Use: Extract model architecture details
- Integration: Get num_experts, hidden_dim, num_layers

**Leverage 3: vLLM Memory Coordinator**
- Location: `sparse_llm/integrations/vllm_memory_coordinator.py`
- Insight: Learn from memory management patterns
- Don't directly use: This is vLLM-specific, but patterns are applicable

---

### New Modules to Create

**Module 1: sparse_llm/inference/router_calculator.py**
```
Purpose: Compute router decisions
Classes:
  - RouterCalculator
    - __init__(router_weights, num_experts, top_k)
    - forward(hidden_states, batch_mode=True)
    - get_expert_assignments(router_logits)
```

**Module 2: sparse_llm/inference/expert_cache.py**
```
Purpose: Manage GPU expert cache
Classes:
  - ExpertCache
    - __init__(capacity, expert_size_gb)
    - check(expert_id) → bool
    - load(expert_id, weights)
    - evict() → expert_id
    - get(expert_id) → weights
```

**Module 3: sparse_llm/inference/attention_engine.py**
```
Purpose: Handle attention with KV caching
Classes:
  - AttentionEngine
    - __init__(num_layers, num_heads, head_dim)
    - prefill(input_ids) → hidden, kv_cache
    - decode(token_id, kv_cache) → hidden, updated_cache
```

**Module 4: sparse_llm/inference/expert_processor.py**
```
Purpose: Execute expert forward passes
Classes:
  - ExpertProcessor
    - __init__(expert_cache)
    - process_batch(hidden, expert_ids) → output
    - process_single(hidden, expert_ids) → output
```

**Module 5: sparse_llm/inference/moe_inference_engine.py**
```
Purpose: Main orchestrator
Classes:
  - CustomMoEInferenceEngine
    - __init__(model_path, config)
    - prefill(prompt_tokens)
    - generate(max_tokens, temperature, top_p)
    - cleanup()
```

---

### API Design for Integration

**API Method 1: Initialize Engine**
```
Input:
  - model_path: Path to model weights
  - expert_cache_size: Number of experts in GPU (default: 4)
  - device: "cuda" or "cpu"
  
Output:
  - engine instance
  
Example Usage:
  engine = CustomMoEInferenceEngine(
    model_path="/models/mixtral-8x7b",
    expert_cache_size=4,
    device="cuda"
  )
```

**API Method 2: Generate Text**
```
Input:
  - prompt: String to start generation
  - max_tokens: Maximum tokens to generate
  - temperature: Sampling temperature
  - top_p: Nucleus sampling threshold
  - stream: Boolean for streaming output
  
Output:
  - If stream=False: Complete generated text
  - If stream=True: Generator yielding tokens
  
Example Usage:
  output = engine.generate(
    prompt="Explain quantum computing",
    max_tokens=100,
    temperature=0.7,
    stream=False
  )
```

**API Method 3: Get Statistics**
```
Input: None

Output:
  - Dictionary with performance metrics:
    - average_tokens_per_second
    - cache_hit_rate
    - total_tokens_generated
    - memory_usage_gb
    
Example Usage:
  stats = engine.get_statistics()
  print(f"Speed: {stats['average_tokens_per_second']} tok/s")
```

---

## Validation Criteria

### Performance Targets

**Target 1: Decode Speed**
- Metric: Tokens per second during decode
- Threshold: 20-30 tok/s average
- Measurement: Generate 100 tokens, compute average
- Status: PASS if average >= 20 tok/s

**Target 2: Cache Hit Rate**
- Metric: Percentage of decode steps with experts cached
- Threshold: >= 80%
- Measurement: Count hits/misses over 100 tokens
- Status: PASS if hit_rate >= 0.80

**Target 3: Prefill Throughput**
- Metric: Tokens processed per second in prefill
- Threshold: >= 200 tok/s
- Measurement: Process 1k tokens, measure time
- Status: PASS if throughput >= 200 tok/s

**Target 4: Memory Usage**
- Metric: Peak GPU memory during generation
- Threshold: <= 24GB for RTX 3090/4090
- Measurement: Monitor nvidia-smi during generation
- Status: PASS if peak <= 24GB

---

### Quality Targets

**Target 5: Output Correctness**
- Metric: Compare output to reference implementation
- Threshold: Exact match or <1% difference
- Measurement: Generate from same seed, compare tokens
- Status: PASS if outputs match

**Target 6: Numerical Stability**
- Metric: Check for NaN or Inf in outputs
- Threshold: Zero occurrences
- Measurement: Run 1000 token generation
- Status: PASS if no NaN/Inf detected

**Target 7: Memory Stability**
- Metric: GPU memory growth over time
- Threshold: <100MB growth per hour
- Measurement: Monitor for 2 hours continuous generation
- Status: PASS if growth < 100MB/hour

---

## Risk Mitigation

### Risk 1: Expert Loading Latency Too High

**Symptom:** Cache misses take >200ms, hurting average tok/s

**Mitigation:**
- Implement expert quantization (INT8) → 2x faster loading
- Use NVMe SSD for expert storage → 2x faster disk reads  
- Implement predictive prefetching → eliminate wait time
- Increase cache size (use more VRAM) → fewer misses

---

### Risk 2: Cache Hit Rate Too Low

**Symptom:** <80% cache hits, frequent swapping

**Mitigation:**
- Analyze router patterns during prefill → identify stable experts
- Increase cache capacity from 4 to 6 experts
- Implement expert popularity analysis → keep hot experts cached
- Use larger GPU (40GB vs 24GB) → cache more experts

---

### Risk 3: Attention Becomes Bottleneck

**Symptom:** Decode attention takes >30ms per token

**Mitigation:**
- Integrate FlashAttention library → 3-4x speedup
- Implement custom CUDA kernel for KV cache access
- Use FP16 precision for attention → 2x speedup
- Optimize memory layout for cache-friendly access

---

### Risk 4: Memory Overflow

**Symptom:** OOM errors during prefill or decode

**Mitigation:**
- Reduce expert cache size dynamically based on context length
- Implement gradient checkpointing (if training)
- Use mixed precision (FP16 for activations, FP32 for experts)
- Chunk long prompts into smaller segments

---

### Risk 5: Accuracy Degradation

**Symptom:** Generated text is incoherent or incorrect

**Mitigation:**
- Validate against reference implementation outputs
- Check router weight normalization (should sum to 1.0)
- Verify expert combination math is correct
- Test with quantization disabled first, then enable

---

## Success Metrics Summary

**Primary Metric:** Decode speed of 20-30 tokens/sec
- Measured over 100 token generation
- With 1k token prompt prefill
- On RTX 3090 24GB or equivalent

**Secondary Metrics:**
- Cache hit rate >= 80%
- Prefill throughput >= 200 tok/s  
- Memory usage <= 24GB
- Output matches reference implementation
- No memory leaks over extended runs

**System Ready for Production When:**
- All primary and secondary metrics achieved
- Integration tests passing
- Documentation complete
- Monitoring/observability implemented

---

## Document Control

**Version:** 1.0  
**Date:** 2026-09-08  
**Purpose:** Design specification for coding agents  
**Next Steps:** Begin Phase 1 implementation

**Coding Agents Should:**
1. Read this document completely before coding
2. Implement phases sequentially (don't skip ahead)
3. Write tests for each milestone
4. Profile after each phase to validate performance
5. Ask questions if design is ambiguous

**Success looks like:** 20-30 tok/s decode with custom PyTorch, no frameworks, running on consumer hardware.
