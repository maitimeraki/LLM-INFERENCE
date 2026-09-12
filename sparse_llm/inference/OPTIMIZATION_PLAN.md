# SparseLLM Optimization Plan
## Goal: Achieve 10 tokens/sec on consumer GPUs

## Current Issues (Severity: Critical → High → Medium)

### CRITICAL (Blocking functionality)
1. ❌ **Only Layer 0 Processed** - Model outputs nonsense
   - Fix: Loop through all `num_layers` in decode phase
   - File: `moe_inference_engine.py:795-857`
   
2. ❌ **Attention Uses Random Projections** - No learned Q/K/V
   - Fix: Load real attention weights (q_proj, k_proj, v_proj, o_proj)
   - File: `attention_engine.py:58-137`

3. ❌ **Missing Layer Norms & Residuals**
   - Fix: Add pre_attention_norm, post_attention_norm, residual connections
   - File: `moe_inference_engine.py:699-764`

### HIGH (Performance killers)
4. ⚠️ **KV Cache Recreated Every Token** (50-200ms overhead)
   - Fix: Pre-allocate max_seq_len cache, use slice indexing
   - File: `attention_engine.py:279-324`

5. ⚠️ **Expert Loading from Disk Every Token** (100-500ms per expert)
   - Fix: Keep hot experts in GPU cache, implement true LRU
   - File: `loading/expert_cache.py:87-133`

6. ⚠️ **Sequential Expert Loading** (2x latency for top_k=2)
   - Fix: Parallel expert loading with async I/O
   - File: `expert_processor.py:247-341`

### MEDIUM (Optimizations)
7. 🔧 **No Torch Compile** - Missing 20-30% speedup
8. 🔧 **No FlashAttention-2** - Missing 2-3x attention speedup
9. 🔧 **No Quantization** - FP16 → INT8 saves 50% memory
10. 🔧 **CPU/GPU Transfer Overhead** - Pinned memory not used

---

## Implementation Priority

### Week 1: Core Functionality (Issues #1-3)
**Goal:** Get model working correctly (even if slow)

**Step 1.1: Fix Layer Processing**
```python
# moe_inference_engine.py:_decode_phase()
for layer_idx in range(self.num_layers):
    # Attention
    current_hidden = self._apply_attention_layer(current_hidden, layer_idx, kv_cache)
    
    # MoE routing
    expert_indices, expert_weights = self.router_calculator.forward(
        current_hidden, layer_idx=layer_idx  # Use layer-specific router
    )
    
    # Expert processing
    current_hidden = self.expert_processor.process_single(
        hidden_state=current_hidden,
        expert_indices=expert_indices,
        expert_weights=expert_weights,
        layer_id=layer_idx  # Now iterates through all layers
    )
```

**Step 1.2: Load Real Attention Weights**
```python
# attention_engine.py: Add weight loading
class AttentionEngine:
    def __init__(self, ...):
        self.attention_weights = {}  # {layer_idx: {q_proj, k_proj, v_proj, o_proj}}
    
    def load_weights_from_checkpoint(self, shared_weights: dict):
        for layer_idx in range(self.num_layers):
            self.attention_weights[layer_idx] = {
                'q_proj': shared_weights[f'model.layers.{layer_idx}.self_attn.q_proj.weight'],
                'k_proj': shared_weights[f'model.layers.{layer_idx}.self_attn.k_proj.weight'],
                'v_proj': shared_weights[f'model.layers.{layer_idx}.self_attn.v_proj.weight'],
                'o_proj': shared_weights[f'model.layers.{layer_idx}.self_attn.o_proj.weight'],
            }
    
    def decode(self, token_id, kv_cache, layer_idx):
        # Project with REAL weights
        q = F.linear(hidden_states, self.attention_weights[layer_idx]['q_proj'])
        k = F.linear(hidden_states, self.attention_weights[layer_idx]['k_proj'])
        v = F.linear(hidden_states, self.attention_weights[layer_idx]['v_proj'])
```

**Step 1.3: Add Layer Norms**
```python
# Load RMSNorm/LayerNorm weights
self.layer_norms = {
    layer_idx: {
        'input_layernorm': shared_weights[f'model.layers.{layer_idx}.input_layernorm.weight'],
        'post_attention_layernorm': shared_weights[f'model.layers.{layer_idx}.post_attention_layernorm.weight'],
    }
}

# Apply in forward pass
def _apply_attention_layer(self, hidden, layer_idx, kv_cache):
    residual = hidden
    hidden = self._rms_norm(hidden, self.layer_norms[layer_idx]['input_layernorm'])
    hidden, kv_cache = self.attention_engine.decode(hidden, kv_cache, layer_idx)
    hidden = hidden + residual  # Residual connection
    return hidden, kv_cache
```

**Expected Result:** Model generates coherent text (accuracy > 90%), but slow (1-3 tok/s).

---

### Week 2: Performance Optimization (Issues #4-6)

**Step 2.1: Pre-allocated KV Cache**
```python
# attention_engine.py:__init__()
def __init__(self, ...):
    # Pre-allocate ONCE for max sequence length
    self.kv_cache_buffer = {
        'keys': torch.zeros(
            1, self.num_layers, self.max_seq_len, self.num_heads, self.head_dim,
            device=self.device, dtype=self.dtype
        ),
        'values': torch.zeros(
            1, self.num_layers, self.max_seq_len, self.num_heads, self.head_dim,
            device=self.device, dtype=self.dtype
        ),
        'current_len': 0  # Track used length
    }

def decode(self, token_id, kv_cache, layer_idx):
    current_pos = kv_cache['current_len']
    
    # Write NEW token K/V directly to buffer (no copy!)
    kv_cache['keys'][:, layer_idx, current_pos:current_pos+1] = new_k
    kv_cache['values'][:, layer_idx, current_pos:current_pos+1] = new_v
    
    # Slice for attention (no copy, just view)
    k = kv_cache['keys'][:, layer_idx, :current_pos+1]
    v = kv_cache['values'][:, layer_idx, :current_pos+1]
```

**Expected Speedup:** 2-3x decode speed (3-9 tok/s).

**Step 2.2: Proper GPU Expert Cache**
```python
# loading/expert_cache.py: Fix get() to check GPU first
def get(self, layer_id: int, expert_id: int):
    key = (layer_id, expert_id)
    
    # GPU cache hit → IMMEDIATE return (no I/O!)
    if key in self._gpu_cache:
        self._gpu_cache.move_to_end(key)
        return self._gpu_cache[key], "gpu"
    
    # Cache miss → load asynchronously
    if key in self._cpu_cache:
        return self._promote_async(key)
    
    # Cold load (unavoidable penalty)
    return self._load_from_storage_async(key)
```

**Expected Speedup:** 10-20x for cached experts (now 9-18 tok/s if 80%+ cache hits).

**Step 2.3: Parallel Expert Loading**
```python
# expert_processor.py:process_single()
def process_single(self, hidden_state, expert_indices, expert_weights, layer_id):
    # Load all top-k experts in parallel (ThreadPoolExecutor)
    with ThreadPoolExecutor(max_workers=self.top_k) as executor:
        expert_futures = {
            executor.submit(self._get_expert, expert_id, layer_id): (expert_id, weight)
            for expert_id, weight in zip(expert_indices[0], expert_weights[0])
        }
        
        # Compute as experts become available
        output = torch.zeros_like(hidden_state)
        for future in as_completed(expert_futures):
            expert, (expert_id, weight) = future.result(), expert_futures[future]
            output += weight * expert(hidden_state)
    
    return output
```

**Expected Speedup:** 1.5-2x for multi-expert (now 13-36 tok/s).

---

### Week 3: Advanced Optimizations (Issues #7-10)

**Step 3.1: Torch Compile**
```python
# Compile hot paths
self.attention_engine.decode = torch.compile(
    self.attention_engine.decode,
    mode="reduce-overhead"  # Optimize for latency
)

self.expert_ffn_forward = torch.compile(
    ExpertFFN.forward,
    mode="reduce-overhead"
)
```

**Expected Speedup:** 1.2-1.3x (now 16-47 tok/s).

**Step 3.2: FlashAttention-2**
```python
# Use optimized CUDA kernels
from flash_attn import flash_attn_with_kvcache

attn_output = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=kv_cache['current_len'],
    causal=True
)
```

**Expected Speedup:** 2-3x attention (now 20-60 tok/s).

---

## Target Performance Breakdown

| Component | Current (ms/tok) | Optimized (ms/tok) | Speedup |
|-----------|------------------|-------------------|---------|
| KV Cache Copy | 100 | 1 | 100x |
| Attention (32 layers) | 200 | 30 | 6.7x |
| Expert Load (disk) | 300 | 5 | 60x |
| Expert Compute | 50 | 20 | 2.5x |
| Router + Sampling | 20 | 10 | 2x |
| **Total** | **670ms** | **66ms** | **10.1x** |

**Result:** 15 tokens/sec (exceeds 10 tok/s goal) ✅

---

## Quick Win: Immediate 3x Speedup

If you only fix **Issue #4 (KV Cache)** and **Issue #5 (Expert Cache)**:
- Current: ~1.5 tok/s → Target: ~5 tok/s
- Time: 2-3 hours of coding
- Files: `attention_engine.py`, `loading/expert_cache.py`

This gets you halfway to the goal with minimal changes!
