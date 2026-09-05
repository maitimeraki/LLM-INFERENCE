# Resource-Aware Weight Loading & Persistence System Design

**Status:** Architecture Design (Phase 1-4 Complete, Ready for Implementation Planning)

**Scope:** Universal weight loading system for ANY downloadable open-source model (MoE and non-MoE architectures)

**Goals:**
- Zero-friction weight loading that adapts to available hardware resources
- Unified approach across research (main.py) and production (serve.py) backends
- Automatic fallback for both MoE and non-MoE models
- Sub-5-minute startup for users with "ready to serve" inference endpoint

---

## **Executive Summary**

This document describes a four-phase initialization system that loads model weights intelligently onto GPU/CPU/storage based on detected system resources. The system is **architecture-agnostic**: it automatically detects whether a model is MoE or non-MoE, applies the appropriate loading strategy, and adapts placement to available hardware without requiring users to understand memory budgets or architecture specifics.

**Key Innovation:** Three-tier weight placement (GPU → CPU → Storage) with runtime expert cache and LRU eviction, enabling single-machine inference for models that normally require multiple GPUs or high-end hardware.

---

## **Part 1: System Overview & Core Concepts**

### **Why This System Exists**

**Current Problem:**
- When users run `main.py` with a large MoE model, weights load once and disappear after generation (no persistence)
- When users run `vllm_server.py`, vLLM loads the entire model from scratch into memory, wasting VRAM on cold experts
- No system detects available resources before loading, leading to OOM crashes mid-inference
- Two separate code paths (Transformers vs vLLM) with incompatible loading strategies
- Users must manually estimate VRAM budgets or risk crashes

**What This System Solves:**
1. **Automatic resource detection** before loading ANY weights
2. **Intelligent placement** based on detected resources (not hardcoded)
3. **Unified loading** across all backends (Transformers, vLLM, future backends)
4. **Graceful degradation** for any model type (MoE or dense)
5. **Persistent weights** until explicit shutdown (no reload between requests)
6. **Observable metrics** so users understand what's happening

---

### **Universal Architecture Support**

This system is designed to work with ANY model that can be downloaded from Hugging Face or loaded from local files. The system automatically adapts to different architectures:

**MoE Models Detected & Supported:**
- Mixtral (mistralai/Mixtral-8x7B-*, any variant)
- Qwen2-MoE (Qwen/Qwen2-57B-A14B-Instruct, any variant)
- DeepSeek-V2 (deepseek-ai/DeepSeek-V2, any MoE variant)
- DBRX (databricks/dbrx-instruct, any variant)
- Any other MoE architecture with detectable expert patterns

**Non-MoE Dense Models (Automatic Fallback):**
- Llama2, Llama3 (meta-llama/Llama-2-70B-*, etc.)
- Mistral (mistralai/Mistral-7B, etc.)
- Phi, Qwen (non-MoE variants)
- Any other dense transformer model

**Behavior by Model Type:**
- **MoE models:** Apply three-tier placement (GPU hot experts → CPU warm experts → Storage cold experts)
- **Dense models:** Standard loading with device-map optimization (no expert paging, but still resource-aware)
- **Hybrid/Unknown:** Automatic fallback to safe defaults

---

### **Four-Phase Initialization Architecture**

The system always executes four sequential phases. Each phase completes before the next starts, providing clear progress feedback to users.

**Phase 1: Resource Profiling (2-5 seconds)**
- Detect all available hardware resources
- Measure GPU VRAM per device, CPU RAM, storage space
- Benchmark storage performance (SSD vs HDD)
- Return ResourceBudget (what we have available)

**Phase 2: Placement Strategy Calculation (1-2 seconds)**
- Analyze model weights: size, distribution, shared vs expert tensors
- Determine optimal placement for this model on this hardware
- Return PlacementPlan (where each weight should go)

**Phase 3: Weight Loading (45-90 seconds)**
- Execute PlacementPlan by loading weights
- Direct load to target device (GPU/CPU/storage)
- Create runtime caching structure for dynamic expert movement
- Return LoadedWeightState (actual weights ready to use)

**Phase 4: Backend Integration (10-20 seconds)**
- Inject weights into chosen backend (Transformers or vLLM)
- Start serving or ready for generation
- Weights persist until shutdown

**Total Cold Start:** 60-120 seconds (one-time cost, then persistent)

---

## **Part 2: Phase 1 - Resource Profiling**

### **Purpose**
Discover and measure all compute resources on the system before loading any weights. This is the foundation for intelligent placement decisions.

### **Detection Strategy**

**GPU Detection (Multi-GPU Aware):**
- Query `CUDA` libraries for number of available GPUs
- For each GPU:
  - Read total VRAM from device properties
  - Measure free VRAM available RIGHT NOW (not theoretical)
  - Apply safety margin (reserve 10% or 500MB minimum to prevent fragmentation crashes)
  - Record compute capability (optimization hints for later)
  - Separate GPU 0 (primary) from GPU 1+ (overflow targets)

- Output: List of GPUs with (device_id, total_memory, available_memory, compute_capability)

**CPU Detection:**
- Query system memory using platform-specific tools
- Get available RAM (not total, but what's free NOW)
- Apply conservative margin: Use 80% of available (leave 20% for OS, other processes, swap safety)
- Minimum requirement: System must have at least 8GB CPU RAM (warn if less)

- Output: (total_memory, usable_memory)

**Storage Detection:**
- Default location: `~/.cache/sparse_llm/experts` (user's home cache directory)
- Allow override via `--storage-path` flag
- Measure available disk space at that path
- Detect storage type (SSD vs HDD):
  - Use OS-specific methods (Linux: check `/sys/block/*/queue/rotational`)
  - Fallback: Small benchmark (write 100MB, measure throughput)
  - If detection fails: Assume SSD (optimistic for performance)
- Minimum requirement: 10GB free (cold experts need this)

- Output: (path, available_bytes, is_ssd, estimated_bandwidth_mbps)

**Resource Validation:**
- If no GPU: Accept CPU-only mode, warn about slowness (10-50x slower)
- If GPU detected but broken: Fall back to CPU
- If storage < 10GB: Reject model loading (recommend freeing space)
- If total resources < model size: Reject with specific error and suggestions

### **Safety Margins & Conservative Estimates**

Why we don't use 100% of available resources:
- PyTorch allocates VRAM in chunks, causing fragmentation
- Inference creates temporary tensors during forward pass
- OS needs memory for swapping, other processes
- Better to underestimate and succeed than overestimate and crash

**Applied margins:**
- GPU: Use 85% of available (reserve 15% fragmentation buffer)
- CPU: Use 80% of available (reserve 20% for OS and other processes)
- Storage: Use 90% of available (reserve 10% for filesystem overhead)

### **Output: ResourceBudget Structure**

The profiler returns a structured ResourceBudget containing:

**GPU Information:**
- Number of GPUs detected
- Per-GPU: device_id, total_bytes, available_bytes, compute_capability
- Total GPU budget after safety margins
- Primary GPU designation (GPU 0)

**CPU Information:**
- Total RAM bytes
- Available bytes
- Usable bytes after safety margin (80%)
- Minimum threshold (must be > 8GB)

**Storage Information:**
- Path (filesystem location)
- Available bytes at that path
- SSD vs HDD detection result
- Estimated I/O bandwidth
- Minimum requirement check (must be > 10GB)

**Decision Support:**
- Recommended GPU cache size (conservative fraction of available)
- Recommended CPU cache size (conservative fraction of available)
- Total system capacity estimate
- Warnings (if any resource constraint detected)

### **Error Messages & User Feedback**

**Success Case:**
```
🔍 Profiling system resources...
   ✓ GPU 0: NVIDIA A100 (24GB total, 22.8GB available)
   ✓ GPU 1: NVIDIA A100 (24GB total, 22.1GB available)
   ✓ CPU: 64GB total, 48GB available (38.4GB usable after margin)
   ✓ Storage: /mnt/ssd (500GB available, SSD detected at 1200MB/s)

📊 Resource Budget:
   GPU total capacity: 45GB (85% of available across GPUs)
   CPU total capacity: 38.4GB (80% of available)
   Storage: 450GB (90% of available for experts)
```

**Error Cases:**
- Insufficient GPU VRAM: Clear message with model size requirement
- Insufficient CPU RAM: Warning but allow (may fail later)
- Insufficient storage: Reject with "Free up space or specify --storage-path"
- No CUDA available: Accept CPU-only with performance warning

---

## **Part 3: Phase 2 - Placement Strategy Calculation**

### **Purpose**
Given ResourceBudget and model configuration, calculate exactly where each weight should be placed to optimize performance.

### **Model Introspection**

**Universal Architecture Detection:**

The system automatically detects model type without hardcoding specific architectures:

1. **Load Model Config (not weights)** — Fast, no I/O
   - Read config.json from model directory
   - Extract: num_hidden_layers, num_local_experts (or num_experts), num_experts_per_tok
   - Note architecture type from config (mixtral, qwen, deepseek, etc.)

2. **Weight Classification** — Classify every tensor as shared, expert, or other
   - Shared weights: Embeddings, attention, norms, router, LM head (always needed, small-ish)
   - Expert weights: Feed-forward experts in MoE layers (large, many, accessed selectively)
   - Other weights: Unused or deprecated parameters

3. **Size Calculation:**
   - Measure total model size (from safetensors index or config)
   - Calculate shared weight total
   - Calculate per-expert size (large MoE experts: 140-200MB each)
   - Determine expert count and distribution

### **Placement Decision Algorithm**

**Goal:** Maximize GPU cache hit rate while respecting available resources.

**Three-Tier Hierarchy:**
```
Tier 1: GPU (fastest, smallest) → Shared weights + hot expert cache
Tier 2: CPU (medium speed, medium size) → Warm expert cache
Tier 3: Storage (slowest, largest) → Cold experts (on-demand)
```

**Placement Steps:**

**Step 1: Place Shared Weights on GPU**
- These are used in EVERY forward pass (attention, norms, router, embeddings)
- Always place on fastest GPU (GPU 0)
- If shared weights don't fit on GPU: REJECT model (fundamental requirement)
- Typical shared weight size: 1-3GB

**Step 2: Allocate Hot Expert Cache on GPU**
- Calculate: `gpu_remaining = gpu_budget - shared_weights_size`
- Determine: `max_hot_experts = floor(gpu_remaining / expert_size)`
- Strategy: Allocate entire remaining GPU space as LRU cache for experts
  - Allows runtime eviction of cold experts when new ones needed
  - System learns which experts are hot during generation
- Example: 17.9GB remaining GPU / 180MB per expert = ~99 expert slots

**Step 3: Allocate Warm Expert Cache on CPU**
- Calculate: `max_warm_experts = floor(cpu_budget / expert_size)`
- These are experts not on GPU but shouldn't need storage I/O
- Example: 25GB CPU / 180MB per expert = ~138 expert slots

**Step 4: Remaining Experts → Storage (Lazy Loading)**
- All experts beyond hot+warm go to storage as lazy references
- Don't load these yet (storage I/O would block startup)
- On cache miss during inference: Load from storage → CPU cache → GPU cache (dynamic promotion)

**Step 5: Multi-GPU Distribution** (if multiple GPUs)
- Primary GPU (GPU 0): Shared weights + hot expert cache
- Secondary GPUs (GPU 1+): Additional hot expert cache (overflow)
- Distribute experts evenly across GPUs to minimize inter-GPU traffic
- Each layer keeps its experts on same GPU when possible

### **Dynamic vs Static Placement**

**Static Placement:** Pre-determine "expert 0-7 are hot, 8-47 are warm"
- Pros: Predictable, simple
- Cons: Wrong if request patterns don't match predictions

**Dynamic Placement (Recommended):** Use LRU cache
- Allocate entire GPU/CPU budget as cache
- Let runtime determine which experts are actually hot based on request patterns
- System learns: "This user always activates experts 1,3,5 → keep them hot"
- Adapts to different workloads automatically

### **Architecture-Aware Adjustments**

The algorithm adapts based on detected model type:

**For MoE Models:**
- Apply full three-tier placement
- Calculate based on expert count and sizes
- Use LRU caching for dynamic hot/cold

**For Dense Models (non-MoE):**
- No expert paging (all weights stay where loaded)
- Use device_map="auto" for automatic placement
- Offload to CPU/storage if model doesn't fit on GPU
- No caching layer needed

**For Hybrid Models (some MoE layers, some dense):**
- Identify which layers have experts
- Apply tiered placement only to expert layers
- Standard loading for dense layers

### **Output: PlacementPlan Structure**

The strategy calculator returns a detailed plan containing:

**Model Metadata:**
- Model ID
- Architecture type (detected automatically)
- Total size
- Layer count, expert count, expert sizes

**Shared Weight Placement:**
- Target device (which GPU)
- List of tensor names
- Total size in bytes

**Expert Placement:**
- Hot experts: device, count, cache policy, total size
- Warm experts: device, count, cache policy, total size
- Cold experts: storage path, count, lazy load references

**Cache Configuration:**
- GPU cache size (LRU limit)
- CPU cache size (LRU limit)
- Eviction policy (LRU recommended)

**Metadata & Predictions:**
- Estimated load time (for progress reporting)
- GPU utilization percentage (how full will GPU be)
- CPU utilization percentage
- Expected cache hit rate (85-95% for typical workloads)

---

## **Part 4: Phase 3 - Weight Loading**

### **Purpose**
Execute the PlacementPlan by actually loading weights from disk and placing them on correct devices.

### **Streaming Load Strategy**

**Key Principle:** Load directly to target device, don't copy afterward
- Don't load to CPU then move to GPU (wastes time + temporary memory)
- Use safetensors selective loading: load only needed tensors
- Target each tensor to final device during load: `load(tensor, map_location="cuda:0")`

### **Loading Execution Steps**

**Step 1: Initialize Weight Index**
- Read safetensors index from model directory (or build from config)
- Create map: tensor_name → (file, byte_offset, byte_length)
- Enables loading individual tensors without loading entire model
- Fast operation (just JSON parsing)

**Step 2: Load Shared Weights to GPU**
- Extract shared weight names from PlacementPlan
- For each shared tensor:
  - Locate in safetensors index
  - Load with `map_location` pointing to target GPU
  - Verify shape and dtype
  - Store reference in shared_weights dict
- Progress indicator: "Loading shared weights... [████████] 2.1GB"

**Step 3: Load Hot Experts to GPU**
- Extract hot expert keys (layer_id, expert_id) from plan
- For each hot expert:
  - Determine tensor names (universal pattern matching)
  - Load expert tensors (w1, w2, w3 or gate_proj, up_proj, down_proj)
  - Wrap in ExpertWeights object
  - Insert into GPU cache tier
- Progress: "Loading hot experts... [████████] 8/99 experts"

**Step 4: Load Warm Experts (with Lazy Option)**
- Strategy A (Eager): Load all warm experts to CPU immediately
  - Pros: Faster runtime inference
  - Cons: Slower startup
- Strategy B (Lazy): Just register paths, load on first access
  - Pros: Faster startup (aligns with zero-friction goal)
  - Cons: First access to cold expert is slower
- **Default: Lazy** (recommended for cold-start performance)
- Store: CPU cache references or storage paths

**Step 5: Register Cold Experts**
- Don't load at all
- Store file paths and byte offsets
- Will load on-demand during inference when cache misses

**Step 6: Initialize Runtime Expert Cache**
- Create ExpertCache structure with GPU + CPU tiers
- Pre-populate GPU cache with loaded hot experts
- Pre-populate CPU cache with loaded warm experts (if eager) or empty (if lazy)
- Configure LRU eviction policies
- Initialize metrics tracking (hits, misses, evictions)

### **Loading Optimizations**

**Parallelization (Multi-GPU):**
- Load GPU 0 and GPU 1 experts in separate threads
- Parallel loading reduces total time by 30-40%
- Synchronize at end to ensure all weights ready

**Sequential Chunks (HDD vs SSD):**
- If storage is HDD: Load experts in large sequential chunks (better for spinning disk)
- If storage is SSD: Load individual experts (random access is fast)
- Adaptively choose based on detected storage type

**Background Prefetch (CPU):**
- While loading GPU weights, prefetch warm experts to CPU in background
- Overlap I/O operations
- By time GPU loading finishes, warm experts already on CPU

### **Error Handling During Load**

**Out of Memory:**
- If GPU OOM during loading: Fall back to CPU for overflow experts
- If CPU OOM: Further fall back to storage-only mode (all cold)
- Never crash silently; provide clear error with memory requirements

**Corrupted Weights:**
- Verify safetensors checksums where available
- If load fails: Attempt retry once
- If retry fails: Clear error message with recovery instructions
- Suggest: "Re-download model using: huggingface-cli download <model>"

**Slow Storage:**
- If loading takes >5 minutes: Warn about HDD performance
- Suggest: "Move model to SSD for 10x faster loading"
- Continue anyway (don't block, just alert)

**Partial Failure:**
- If loading one expert fails: Skip it, mark as unavailable
- Continue loading other experts
- During runtime: If unavailable expert needed, attempt reload or skip
- Don't fail entire model for one corrupted tensor

### **Progress & User Feedback**

Users wait 45-90 seconds during this phase. Show granular progress:

```
⏳ Loading model weights (this may take 1-2 minutes)...

[Phase 3 of 4] Loading model weights

[1/4] Loading shared weights to GPU 0...
   ├─ model.embed_tokens.weight [████████████] 512MB
   ├─ model.layers.*.self_attn.* [████████████] 1.2GB
   └─ model.norm.weight [████████████] 8MB
   ✓ Shared weights loaded (2.1GB in 8.3s)

[2/4] Loading hot experts to GPU 0...
   ├─ Layer 0, experts 0-7 [████████████] 1.4GB
   ├─ Layer 1, experts 0-5 [████████████] 900MB
   └─ ... (more layers)
   ✓ Hot experts loaded (1.4GB in 12.1s) [8/99 expert slots]

[3/4] Registering warm experts (lazy loading)...
   ✓ 48 expert references registered
   ✓ Ready to load on-demand

[4/4] Registering cold experts (on-demand)...
   ✓ 200 expert references registered
   ✓ Will load from storage only if needed

✅ Model weights loaded successfully (total: 58.7s)
   GPU: 3.5GB loaded (hot experts + shared)
   CPU: 0GB loaded (lazy warm experts)
   Storage: 43.5GB available on-demand (cold experts)
   
Ready for inference!
```

---

## **Part 5: Phase 4 - Backend Integration**

### **Purpose**
Connect LoadedWeightState to the chosen execution backend (Transformers for main.py or vLLM for serve.py).

### **Key Design Principle**

**Backends don't reload weights. They use pre-loaded weights.**
- Backend receives LoadedWeightState with all weights already placed
- Backend integrates ExpertCache for runtime expert movement
- No weight re-download or re-placement (weights already there)

### **Integration Path A: Transformers Backend (main.py)**

**Strategy:** Inject pre-loaded weights into standard Transformers model

**Steps:**
1. Load model structure only (no weights): Use AutoConfig, then create model from config
2. Model has parameter structure but no data (fast, no I/O)
3. Inject shared weights using standard state_dict mechanism
4. Replace MoE forward passes with PagedMoELayer (uses our expert cache)
5. Set model to eval mode
6. Ready for generation

**Key Points:**
- Weights already loaded in LoadedWeightState
- Just need to wire them into model parameters
- Standard Transformers inference loop
- No special executor needed

### **Integration Path B: vLLM Backend (serve.py)**

**Strategy:** Custom executor that uses pre-loaded weights during serving

**Steps:**
1. Create custom executor class inheriting from vLLM's ExecutorBase
2. Executor receives LoadedWeightState in constructor
3. Override execute_model() method (called for each forward pass)
4. Inside execute_model():
   - Standard forward through shared layers
   - For MoE layers: Call PagedMoELayer
   - PagedMoELayer queries ExpertCache for experts
   - Cache handles GPU/CPU/storage tier movement
   - Return logits to vLLM for token generation
5. vLLM handles batching, scheduling, OpenAI API

**Key Points:**
- ExpertCache is persistent across requests
- Runtime LRU eviction happens during inference
- Weights loaded in Phase 3 stay loaded
- No reload between requests

### **Universal Backend Contract**

Both backends follow same initialization pattern:

1. Receive ResourceBudget from Phase 1
2. Receive PlacementPlan from Phase 2
3. Receive LoadedWeightState from Phase 3
4. Initialize with pre-loaded weights (no re-loading)
5. Use ExpertCache for runtime expert management

This contract enables:
- Easy testing (mock weights, mock cache)
- Future backend addition (TensorRT, ONNX, etc.)
- Consistent initialization across all entry points

### **Backend-Specific Differences**

**Transformers (main.py):**
- Load → Generate → Exit
- Weights exist for duration of generate() call
- Simple, stateless
- Good for testing, research, one-off inference

**vLLM (serve.py):**
- Load → Serve indefinitely
- Weights persist across requests
- Stateful (cache evolves with request patterns)
- Good for production, teams, continuous serving
- Handles batching, parallel requests

---

## **Part 6: Runtime - Expert Cache with LRU**

### **Purpose**
During inference, dynamically manage experts across GPU/CPU/storage tiers based on access patterns.

### **Cache Architecture**

**Three-Tier Hierarchy:**
```
GPU Cache (Tier 1: L1) - Fastest, smallest
   └─→ CPU Cache (Tier 2: L2) - Fast, medium
       └─→ Storage (Tier 3: L3) - Slow, largest
```

**Cache Operations:**
- **Lookup:** Check GPU → CPU → Storage (in order)
- **Promote:** Move expert from slower tier to faster (storage→CPU→GPU)
- **Evict:** Remove from cache tier when full, move to slower tier if needed

### **LRU (Least Recently Used) Eviction**

**Why LRU:**
- Simple, proven, well-understood
- Works well for expert access patterns (some consistently hot, others rare)
- Low overhead (just track access timestamps)

**How LRU Works:**
- Each expert tracks: last_accessed (timestamp), access_count (frequency)
- When cache full and new expert needed:
  1. Find expert with oldest timestamp
  2. Evict to slower tier
  3. Load new expert to freed space
  4. Update timestamps

### **Cache Miss Flow** (What happens when expert not in cache)

**Example:** Need expert (layer=5, expert=3)

**Check GPU Cache:**
- Expert in GPU cache? Yes → Use immediately (sub-millisecond)
- No → Check CPU cache

**Check CPU Cache:**
- Expert in CPU cache? Yes → Promote to GPU
  - If GPU cache full: Evict LRU expert to CPU first (cascading eviction)
  - Move expert CPU→GPU (via PCIe, ~2-5ms)
  - Update access timestamps
  - Return expert
- No → Load from storage

**Load from Storage:**
- Read expert from disk using safetensors offset
- Load to CPU memory first (disk I/O, ~10-200ms depending on SSD/HDD)
- Insert into CPU cache
- If CPU cache full: Evict LRU expert
- Immediately promote CPU→GPU (as above)
- Return expert

### **Cache Promotion Strategy**

**Static vs Dynamic Promotion:**

**Static:** Pre-determine "expert 0-7 are hot, never evict"
- Pro: Predictable
- Con: Wrong if access patterns differ

**Dynamic (Recommended):** Track access frequency during runtime
- Threshold: Promote CPU→GPU after 2+ accesses in short window
- Reason: If expert accessed >1 time, likely hot; keep on GPU
- If expert accessed only once: Keep on CPU (save GPU space for others)
- If expert never re-accessed: Don't promote from storage to GPU

### **Performance Characteristics by Tier**

**GPU Cache Hit:**
- Latency: ~0.01ms (memory bandwidth)
- Throughput: No impact (expert already there)

**CPU Cache Hit (promoted to GPU):**
- Latency: ~2-5ms per expert (PCIe transfer)
- Throughput: Depends on PCIe speed (Gen 3: 16GB/s, Gen 4: 32GB/s)

**Storage Hit (loaded to CPU, then GPU):**
- SSD: ~10-30ms (disk I/O + PCIe)
- HDD: ~50-200ms (seek time + rotational latency + I/O)

**Impact on Generation:**
- 95% GPU hit rate → ~1200 tokens/sec (excellent)
- 80% GPU + 15% CPU → ~800 tokens/sec (good)
- 50% GPU + 30% CPU + 20% storage → ~200 tokens/sec (acceptable)
- <50% GPU hit rate → Poor performance, user should increase GPU cache

### **Cache Statistics & Observability**

**Metrics tracked:**
- total_accesses: Total expert lookups
- gpu_hits: Found in GPU cache
- cpu_hits: Found in CPU cache (promoted)
- storage_hits: Loaded from storage
- evictions: Experts removed from cache
- promotions: CPU→GPU promotions
- average_load_time_ms: Average storage load latency

**Cache Hit Rate Formula:**
```
gpu_hit_rate = gpu_hits / total_accesses
cpu_hit_rate = cpu_hits / total_accesses
storage_miss_rate = storage_hits / total_accesses

Healthy:
  >90% GPU, <5% CPU, <5% storage = Excellent performance
  >70% GPU, <20% CPU, <10% storage = Good performance
  >50% GPU, >30% storage = Poor, user should profile
```

---

## **Part 7: Architecture-Agnostic Model Handling**

### **Universal Architecture Detection**

The system works with ANY model by detecting its architecture automatically:

**Detection Algorithm:**

1. **Load config.json** (no weights, just metadata)
2. **Check for expert count** using universal patterns:
   - Look for attributes: num_local_experts, num_experts, n_routed_experts, moe_num_experts
   - Check any of these (different architectures use different names)
3. **Check for expert routing:**
   - Look for attributes: num_experts_per_tok, num_experts_per_token, top_k, moe_top_k
   - These indicate MoE models
4. **Decision tree:**
   - If expert_count > 1 AND routing_param > 0 → MoE model (apply tiered placement)
   - If expert_count missing or = 1 → Dense model (standard loading)
   - If unknown: Safe fallback to dense model loading

**MoE Models Handled:**
- Mixtral (block_sparse_moe.experts with top-k routing)
- Qwen2-MoE (mlp.experts with gating)
- DeepSeek-V2 (moe.experts with expert selection)
- DBRX (expert layers with routing)
- Any MoE that follows standard transformer patterns

**Dense Models Handled:**
- Llama2, Llama3 (standard dense FFN)
- Mistral (non-MoE variant, dense FFN)
- Phi, Qwen (dense variants)
- Any transformer without expert routing

### **Behavior by Model Type**

**MoE Model Detected:**
- Apply full three-tier placement
- Calculate placement based on expert count and sizes
- Use LRU caching for dynamic expert management
- Generate metrics about cache hit rates, expert access patterns

**Dense Model Detected:**
- Skip expert paging logic
- Use standard device_map="auto" loading
- Load model onto GPU, or split across GPU/CPU if large
- No expert cache (not needed)
- Simpler initialization, faster for small dense models

**Fallback (Unknown Architecture):**
- Assume dense model (safe default)
- Load with automatic device placement
- If fails: Provide clear error message with model architecture needed
- Suggest: "Report model architecture to enable optimization"

### **Weight Classification (Universal)**

For ANY model (MoE or dense), system classifies weights into:

**Shared Weights (always needed, load first):**
- Embeddings: model.embed_tokens, tokenizer embeddings
- Attention: self_attn.q_proj, k_proj, v_proj, o_proj (all layers)
- Layer norms: input_layernorm, post_attention_layernorm (all layers)
- Router/Gate: block_sparse_moe.gate or mlp.gate (MoE only)
- LM Head: lm_head weight
- Final norm: model.norm

**Expert Weights (for MoE only, pageable):**
- Feed-forward experts: w1/w2/w3 or gate_proj/up_proj/down_proj
- Per layer per expert

**Other Weights:**
- Unused parameters, deprecated tensors, backward-compat aliases

**Universal Classification:**
The system doesn't hardcode model names. Instead:
- Look for pattern: tensor_name contains "layer", "expert", "moe"
- Match naming patterns across known architectures
- Classify based on patterns, not on model_type

This works because:
- All transformers follow similar layer structure
- Expert weights are in predictable namespaces
- Classification rules generalize across architectures

---

## **Part 8: Fallback & Error Handling**

### **Graceful Degradation for Any Model**

**GPU Unavailable → CPU Fallback:**
- Detect: CUDA not available or device error
- Action: Load entire model to CPU
- Performance: 10-50x slower but functional
- Message: "GPU not available, using CPU. Inference will be slow."

**GPU Out of Memory → CPU Offload:**
- Detect: CUDA OOM during loading
- Action: Move some weights to CPU automatically
- For MoE: Reduce hot expert count, more to CPU
- For dense: Split layers across GPU/CPU
- Transparent to user (just slower)

**CPU Out of Memory → Storage Fallback:**
- Detect: System memory > 90% used
- Action: Evict warm experts to storage earlier
- Rely more on storage (very slow but won't crash)
- Warning: "Memory pressure detected, performance degraded"

**Storage Unavailable → In-Memory Only:**
- Detect: Storage path inaccessible or no space
- Action: Keep all experts in GPU/CPU only
- If experts don't fit: Fail with clear error
- Message: "Storage unavailable and experts don't fit in GPU/CPU"

### **Architecture-Specific Error Handling**

**MoE Model But Expert Loading Fails:**
- Can't identify expert structure?
- Fallback: Load as dense model (all weights stay in place)
- Performance: Slower (all experts always in memory)
- Not ideal but functional

**Dense Model But Incorrectly Detected as MoE:**
- If expert count detection is wrong
- Fallback: Ignore expert count, load as dense
- Check: If no forward pass works, try dense loading

**Model Config Missing:**
- Can't read config.json?
- Safe default: Load as dense model
- Ask user to verify model format
- Clear error: "Could not auto-detect architecture, trying dense loading"

### **Path to Production Reliability**

**Critical Not Critical:**
- Phase 1 failure (can't detect resources) → Reject immediately (user system issue)
- Phase 2 failure (can't calculate placement) → Reject with clear reason (model compatibility)
- Phase 3 failure during shared weights → Reject (model won't work)
- Phase 3 failure during expert loading → Warn but continue (some experts available)
- Phase 4 failure → Reject (backend integration issue)

**User Recovery Paths:**
- "Model too large" → Provide size estimate, suggest smaller variant
- "GPU OOM" → Suggest increasing CPU cache, reducing batch size
- "Storage slow" → Suggest moving to SSD, or increasing CPU cache
- "Architecture unknown" → Provide model config snippet for diagnosis

---

## **Part 9: Unified Entry Points (All Architectures)**

### **Entry Point 1: Quick Testing (main.py)**

```
Usage: python main.py --model <model-id> --prompt "..." [--gpu-cache-gb <n>]

Behavior:
1. Phase 1: Profile resources
2. Phase 2: Calculate placement
3. Phase 3: Load weights
4. Phase 4: Initialize Transformers backend
5. Generate from prompt
6. Exit (weights freed)
```

**Auto-Detects:** MoE or dense model → applies appropriate strategy
**User Experience:** "Works for any model, no configuration needed"

### **Entry Point 2: Zero-Config Server (quickstart.py)**

```
Usage: python quickstart.py --model <model-id> [--port 8000]

Behavior:
1. Phase 1: Profile resources
2. Phase 2: Calculate placement (with auto defaults)
3. Phase 3: Load weights
4. Phase 4: Initialize vLLM backend
5. Start OpenAI-compatible server
6. Weights persist until server stops
```

**Auto-Detects:** MoE or dense, auto-optimizes for serving
**User Experience:** "One command, works with any model"

### **Entry Point 3: Production Server (serve.py)**

```
Usage: python serve.py \
  --model <model-id> \
  [--backend vllm|transformers] \
  [--gpu-cache-gb <n>] \
  [--cpu-cache-gb <n>] \
  [--port 8000]

Behavior:
1. Phase 1: Profile resources (with user hints)
2. Phase 2: Calculate placement (with user overrides)
3. Phase 3: Load weights
4. Phase 4: Initialize backend
5. Start server with full observability
6. Weights persist
```

**Auto-Detects:** MoE or dense, respects user hints
**User Experience:** "Full control with sensible defaults"

---

## **Part 10: Observable Metrics & Health**

### **Metrics Endpoint (serve.py only)**

**GET /metrics** returns JSON with:

**Model Info:**
- Model ID, architecture (auto-detected)
- Total size, layers, experts (if MoE)

**Placement Info:**
- Shared weights: size, device
- Hot experts: count, device, cache size
- Warm experts: count, device
- Cold experts: count (in storage)

**Cache Stats:**
- Total accesses, GPU hits, CPU hits, storage hits
- Hit rates per tier
- Evictions, promotions
- Average load time per tier

**Performance:**
- Requests/sec, tokens/sec
- Average latency per request
- Peak memory used

**Resources:**
- GPU utilization %
- CPU utilization %
- Storage space used

### **Health Check**

**GET /health:**
- Model loaded? Yes/No
- Cache healthy? Yes/No
- GPU available? Yes/No
- Last request time

Status codes: 200 (healthy), 503 (loading/unhealthy), 500 (error)

### **Logs & Debug Output**

**INFO level (default):**
- Phase completion timestamps
- Resource summaries
- Server ready message

**DEBUG level:**
- Per-expert load times
- Cache hit/miss per request
- Eviction decisions

**TRACE level:**
- Every expert access
- Every cache operation
- Tensor movement details

---

## **Part 11: Configuration for Any Model**

### **Command-Line Configuration**

**Automatic (recommended for most users):**
```
python serve.py --model mistralai/Mixtral-8x7B-Instruct-v0.1
```
System auto-detects resources and model architecture, applies optimal defaults.

**With Hints (for users who know their hardware):**
```
python serve.py \
  --model Qwen/Qwen2-57B-A14B \
  --gpu-cache-gb 16 \
  --cpu-cache-gb 24
```
System respects user guidance, but adapts if values seem off.

**Full Manual (power users):**
```
python serve.py \
  --model deepseek-ai/DeepSeek-V2 \
  --backend vllm \
  --gpu-cache-gb 20 \
  --cpu-cache-gb 32 \
  --storage-path /mnt/ssd/models \
  --metrics-port 9090 \
  --log-level debug
```

### **Environment Variables** (for containers)

```
SPARSE_LLM_MODEL=mistralai/Mixtral-8x7B
SPARSE_LLM_GPU_CACHE_GB=16
SPARSE_LLM_CPU_CACHE_GB=24
SPARSE_LLM_BACKEND=vllm
SPARSE_LLM_PORT=8000
SPARSE_LLM_STORAGE_PATH=/models/experts
```

### **Configuration File** (optional YAML)

```yaml
model:
  id: mistralai/Mixtral-8x7B-Instruct-v0.1
  revision: main

resources:
  gpu_cache_gb: auto
  cpu_cache_gb: auto
  storage_path: ~/.cache/sparse_llm

server:
  backend: vllm
  port: 8000
  metrics_port: 9090

cache:
  eviction_policy: lru
  promotion_threshold: 2

logging:
  level: info
```

---

## **Part 12: Implementation Phases & Success Criteria**

### **Success Criteria for This Design**

The system is production-ready when:

1. **Universal Architecture Support:**
   - ✓ Mixtral (any variant) loads and serves
   - ✓ Qwen2-MoE (any variant) loads and serves
   - ✓ DeepSeek-V2 loads and serves
   - ✓ Llama3-70B (dense) loads and serves
   - ✓ Unknown architecture gracefully falls back

2. **Resource Efficiency:**
   - ✓ Cold-start overhead: 60-120 seconds
   - ✓ Runtime latency: <20ms for expert swap (GPU→CPU→Storage)
   - ✓ Memory efficiency: Uses 70-85% less VRAM than naive loading

3. **Zero Friction:**
   - ✓ No user configuration needed: `python serve.py --model mixtral`
   - ✓ Clear progress messages during 60-120s startup
   - ✓ Weights persist (no reload between requests)

4. **Observability:**
   - ✓ Metrics endpoint shows cache hit rates, expert placement
   - ✓ Health check endpoint reflects system status
   - ✓ Logs explain what's happening at each phase

5. **Reliability:**
   - ✓ OOM crash → graceful fallback (slower but works)
   - ✓ Missing resource → clear error with recovery path
   - ✓ Corrupted weight → skip and log, don't crash

---

## **Part 13: Appendix - Technical References**

### **Universal Model Architecture Patterns**

**MoE Detection Signals:**
- `num_local_experts` or `num_experts` in config > 1
- `num_experts_per_tok` or `top_k` routing parameter
- Expert tensor names containing "experts" or "moe"

**Shared Weight Patterns:**
- `embed_tokens` or `wte` (embeddings)
- `self_attn` (attention layers)
- `layernorm` or `ln` (layer norms)
- `gate` or `router` (MoE only)
- `lm_head` (language model head)

**Expert Weight Patterns:**
- `w1`, `w2`, `w3` (Mixtral-style)
- `gate_proj`, `up_proj`, `down_proj` (Qwen-style)
- `fc1`, `fc2` (alternative naming)
- Under `experts` or `moe` namespace per layer

---

## **Conclusion**

This four-phase system enables **any downloadable open-source model** to be served efficiently on single machines by intelligently using available GPU/CPU/storage resources. The system is:

- **Universal:** Works with MoE and dense models, auto-detects architecture
- **Automatic:** Users run one command, no configuration needed
- **Persistent:** Weights load once, stay loaded until shutdown
- **Observable:** Clear metrics and progress feedback
- **Reliable:** Graceful degradation, no silent crashes

The design prioritizes **zero user friction** while maintaining **production reliability**. Implementation can proceed as soon as this design is approved.

