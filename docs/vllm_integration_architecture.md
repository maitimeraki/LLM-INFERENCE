# vLLM Integration Architecture: Conflict-Free Design

## Executive Summary

This document defines the architecture for integrating SparseLLM's resource-aware expert loading with vLLM's high-performance serving infrastructure. The design eliminates conflicts by establishing clear ownership boundaries and creating a unified memory management layer.

**Key Principle**: Resource-aware loading owns weight management, vLLM owns inference execution.

---

## 1. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          UNIFIED SERVER (serve.py)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴──────────────┐
                    │                              │
                    ▼                              ▼
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│   RESOURCE-AWARE LOADING LAYER   │  │      vLLM EXECUTION LAYER        │
│  (Weight Management & Memory)    │  │   (Inference & Scheduling)       │
└──────────────────────────────────┘  └──────────────────────────────────┘
            │                                      │
            │ LoadedWeightState                   │
            ▼                                      ▼
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│  FourPhaseOrchestrator           │  │  AsyncLLMEngine                  │
│  ├─ ResourceProfiler             │  │  ├─ Model Executor               │
│  ├─ ModelIntrospector            │  │  ├─ Scheduler (batch-aware)     │
│  ├─ PlacementStrategy            │  │  └─ KV Cache Manager            │
│  └─ WeightLoader                 │  └──────────────────────────────────┘
└──────────────────────────────────┘               │
            │                                      │
            │                                      │ uses weights from
            ▼                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED MEMORY COORDINATOR                             │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  GPU Memory Allocation Strategy:                                   │  │
│  │  Total GPU = Expert Cache (40%) + KV Cache (50%) + Overhead (10%) │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
            │                                      │
            ▼                                      ▼
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│    EXPERT WEIGHT PROVIDER        │  │     vLLM PAGED ATTENTION         │
│  (Three-Tier Cache)              │  │     (KV Cache Only)              │
│                                  │  │                                  │
│  ┌────────────────────────────┐  │  │  ┌────────────────────────────┐ │
│  │ GPU Cache (L1)             │  │  │  │ Page Pool                  │ │
│  │ • LRU eviction             │  │  │  │ • Request batching         │ │
│  │ • Pin protection           │  │  │  │ • Memory blocks            │ │
│  │ • 40% of GPU memory        │  │  │  │ • 50% of GPU memory        │ │
│  └────────────────────────────┘  │  │  └────────────────────────────┘ │
│  ┌────────────────────────────┐  │  │                                  │
│  │ CPU Cache (L2)             │  │  │                                  │
│  │ • Warm experts ready       │  │  │                                  │
│  │ • Fast promotion to GPU    │  │  │                                  │
│  └────────────────────────────┘  │  │                                  │
│  ┌────────────────────────────┐  │  │                                  │
│  │ Storage Cache (L3)         │  │  │                                  │
│  │ • Safetensors on disk      │  │  │                                  │
│  │ • Lazy loading             │  │  │                                  │
│  └────────────────────────────┘  │  │                                  │
└──────────────────────────────────┘  └──────────────────────────────────┘
```

---

## 2. Memory Management Strategy

### 2.1 GPU Memory Allocation Formula

```python
total_gpu_bytes = torch.cuda.get_device_properties(0).total_memory
usable_gpu = total_gpu_bytes * 0.85  # Leave 15% buffer

expert_cache_bytes = usable_gpu * 0.40  # 40% for experts
kv_cache_bytes = usable_gpu * 0.50      # 50% for KV cache
overhead_bytes = usable_gpu * 0.10      # 10% for activations/overhead
```

**Rationale**:
- **40% expert cache**: MoE models need ~2-4 experts active per layer. With 8 experts/layer and 32 layers, 40% allows 10-12 experts cached on GPU.
- **50% KV cache**: vLLM's primary memory consumer for batched inference. More KV cache = more parallel requests.
- **10% overhead**: Activations, intermediate tensors, CUDA kernels.

### 2.2 Memory Coordination Protocol

**Problem**: Two systems trying to allocate GPU memory independently causes OOM errors.

**Solution**: Single allocation at startup, static boundaries.

```python
class UnifiedMemoryCoordinator:
    """Coordinates GPU memory between expert cache and vLLM KV cache."""
    
    def __init__(self, total_gpu_bytes: int):
        self.total_gpu = total_gpu_bytes
        self.usable_gpu = int(total_gpu_bytes * 0.85)
        
        # Static allocation boundaries
        self.expert_cache_budget = int(self.usable_gpu * 0.40)
        self.kv_cache_budget = int(self.usable_gpu * 0.50)
        self.overhead_budget = int(self.usable_gpu * 0.10)
        
    def get_vllm_config(self) -> dict:
        """Return vLLM engine configuration."""
        return {
            "gpu_memory_utilization": 0.50,  # 50% for KV cache
            "max_num_seqs": self._calculate_max_batch_size(),
            "enforce_eager": True,  # Disable CUDA graphs initially
        }
    
    def get_expert_cache_config(self) -> dict:
        """Return expert cache configuration."""
        return {
            "gpu_cache_bytes": self.expert_cache_budget,
            "cpu_cache_bytes": self._calculate_cpu_budget(),
        }
    
    def _calculate_max_batch_size(self) -> int:
        """Calculate safe batch size for available KV cache."""
        # Conservative: 2GB KV cache supports ~16-32 parallel requests
        # depending on sequence length
        return min(32, max(8, self.kv_cache_budget // (128 * 1024 * 1024)))
```

### 2.3 Preventing Memory Conflicts

**Rules**:
1. **No dynamic allocation after startup**: All memory pools pre-allocated
2. **Expert cache never exceeds budget**: Hard limit with LRU eviction
3. **vLLM never exceeds budget**: `gpu_memory_utilization` enforces limit
4. **Monitor actual usage**: Log warnings if approaching limits

---

## 3. Expert Loading Flow

### 3.1 Integration Point: Custom Model Loader

**Problem**: vLLM loads weights from HuggingFace by default, ignoring LoadedWeightState.

**Solution**: Inject pre-loaded weights during vLLM model initialization.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Expert Loading Flow                          │
└─────────────────────────────────────────────────────────────────┘
                                  
1. STARTUP (Before vLLM)
   ┌────────────────────────────────────────────────┐
   │ FourPhaseOrchestrator.initialize()             │
   │ → LoadedWeightState with:                      │
   │   • shared_weights (attention, mlp_shared)     │
   │   • expert_cache (3-tier, pre-populated)       │
   │   • model_info, placement_plan                 │
   └────────────────────────────────────────────────┘
                    │
                    ▼
2. vLLM INITIALIZATION
   ┌────────────────────────────────────────────────┐
   │ AsyncLLMEngine.from_engine_args()              │
   │ → Creates ModelExecutor                        │
   │ → Initializes model weights                    │
   └────────────────────────────────────────────────┘
                    │
                    ▼
3. WEIGHT INJECTION
   ┌────────────────────────────────────────────────┐
   │ SparseMoEWeightBridge                          │
   │ • Replaces HuggingFace loader                  │
   │ • Returns weights from LoadedWeightState       │
   │ • Shared weights: Direct tensor reference      │
   │ • Expert weights: Cache lookup                 │
   └────────────────────────────────────────────────┘
                    │
                    ▼
4. INFERENCE (Runtime)
   ┌────────────────────────────────────────────────┐
   │ vLLM Forward Pass                              │
   │ → Calls MoE layer                              │
   │ → MoE layer requests experts via bridge        │
   │ → Bridge: expert_cache.get(layer_id, expert_id)│
   │ → Returns GPU tensor (with auto-promotion)     │
   └────────────────────────────────────────────────┘
```

### 3.2 Implementation: SparseMoEWeightBridge

This is the critical integration layer that makes vLLM use pre-loaded weights.

```python
class SparseMoEWeightBridge:
    """Bridge between vLLM's model executor and LoadedWeightState.
    
    This class intercepts vLLM's weight loading and provides weights
    from the pre-loaded LoadedWeightState instead.
    """
    
    def __init__(self, loaded_state: LoadedWeightState):
        self.loaded_state = loaded_state
        self.expert_cache = loaded_state.expert_cache
        self.shared_weights = loaded_state.shared_weights
        
    def get_weight(self, name: str) -> torch.Tensor:
        """Get weight tensor by name.
        
        Routes to appropriate source:
        - Shared weights: Direct lookup
        - Expert weights: Cache with auto-promotion
        """
        # Parse weight name: "model.layers.5.block_sparse_moe.experts.3.w1.weight"
        if ".experts." in name:
            layer_id, expert_id = self._parse_expert_name(name)
            expert_weights, tier = self.expert_cache.get(layer_id, expert_id)
            
            # Extract specific weight from expert dict
            weight_key = self._get_weight_key_from_name(name)
            return expert_weights[weight_key]
        else:
            # Shared weight
            return self.shared_weights[name]
    
    def _parse_expert_name(self, name: str) -> tuple[int, int]:
        """Extract layer_id and expert_id from weight name."""
        # Example: "model.layers.5.block_sparse_moe.experts.3.w1.weight"
        parts = name.split(".")
        layer_idx = parts.index("layers") + 1
        expert_idx = parts.index("experts") + 1
        return int(parts[layer_idx]), int(parts[expert_idx])
    
    def _get_weight_key_from_name(self, name: str) -> str:
        """Convert vLLM weight name to expert cache key."""
        # Example: "model.layers.5.block_sparse_moe.experts.3.w1.weight" → "w1"
        parts = name.split(".")
        expert_idx = parts.index("experts") + 2
        return ".".join(parts[expert_idx:])
```

### 3.3 vLLM Integration Method

**Approach**: Custom Worker with weight loading hooks

```python
class SparseExpertWorker(Worker):
    """Custom vLLM worker that uses SparseMoEWeightBridge for weights."""
    
    def __init__(self, loaded_state: LoadedWeightState, *args, **kwargs):
        self.weight_bridge = SparseMoEWeightBridge(loaded_state)
        super().__init__(*args, **kwargs)
    
    def load_model(self):
        """Override model loading to use pre-loaded weights."""
        # Create model structure (without loading weights)
        self.model = self._create_empty_model()
        
        # Populate model with pre-loaded weights
        self._inject_weights_from_bridge()
        
        # Move to GPU
        self.model = self.model.cuda()
        
    def _inject_weights_from_bridge(self):
        """Inject weights from LoadedWeightState into model."""
        for name, param in self.model.named_parameters():
            weight_tensor = self.weight_bridge.get_weight(name)
            param.data = weight_tensor
```

---

## 4. Batching Strategy: Cache-Aware Scheduling

### 4.1 Problem: Cache Thrashing

Standard vLLM batching groups requests by arrival time. This causes:
- Request A needs experts [0, 3, 7]
- Request B needs experts [1, 4, 5]
- Request C needs experts [2, 6, 3]

Result: 9 different experts loaded, constant GPU↔CPU transfers, poor cache hit rate.

### 4.2 Solution: Expert-Affinity Scheduling

Group requests that need similar experts together in batches.

```python
class ExpertAffinityScheduler:
    """Schedule requests to minimize expert cache thrashing."""
    
    def __init__(self, expert_cache: ExpertCache, max_batch_size: int = 32):
        self.expert_cache = expert_cache
        self.max_batch_size = max_batch_size
        self.pending_requests: list[Request] = []
        
    async def schedule_batch(self) -> list[Request]:
        """Select next batch using expert affinity."""
        if not self.pending_requests:
            return []
        
        # Strategy 1: Cache-hot first (fast path)
        batch = self._select_cache_hot_batch()
        if len(batch) >= self.max_batch_size // 2:
            return batch
        
        # Strategy 2: Expert clustering (group similar requests)
        return self._select_clustered_batch()
    
    def _select_cache_hot_batch(self) -> list[Request]:
        """Select requests whose experts are already on GPU."""
        batch = []
        
        for req in self.pending_requests[:]:
            if len(batch) >= self.max_batch_size:
                break
            
            # Predict required experts (use routing scores from model config)
            required_experts = self._predict_experts(req)
            
            # Check if most experts are cached
            cached_ratio = self._get_cache_hit_ratio(required_experts)
            if cached_ratio > 0.7:  # 70% already on GPU
                batch.append(req)
                self.pending_requests.remove(req)
        
        return batch
    
    def _select_clustered_batch(self) -> list[Request]:
        """Group requests by expert overlap."""
        if not self.pending_requests:
            return []
        
        # Start with first request
        seed_request = self.pending_requests[0]
        seed_experts = self._predict_experts(seed_request)
        batch = [seed_request]
        self.pending_requests.remove(seed_request)
        
        # Add requests with high expert overlap
        for req in self.pending_requests[:]:
            if len(batch) >= self.max_batch_size:
                break
            
            req_experts = self._predict_experts(req)
            overlap = len(seed_experts & req_experts) / len(seed_experts | req_experts)
            
            if overlap > 0.5:  # 50% overlap
                batch.append(req)
                self.pending_requests.remove(req)
        
        return batch
    
    def _predict_experts(self, request: Request) -> set[tuple[int, int]]:
        """Predict which experts a request will activate.
        
        For MoE models, this uses:
        1. Top-k routing (typically k=2)
        2. Static routing patterns from model config
        3. Token-level expert prediction (advanced)
        """
        # Simplified: Assume top-2 experts per layer
        # In production, use router weights to predict
        experts = set()
        for layer_id in range(self.model_info.num_layers):
            # Predict top-2 experts for this layer
            # (In reality, run a lightweight router forward pass)
            top_experts = self._lightweight_route(request, layer_id, k=2)
            for expert_id in top_experts:
                experts.add((layer_id, expert_id))
        return experts
    
    def _get_cache_hit_ratio(self, experts: set[tuple[int, int]]) -> float:
        """Check what fraction of experts are on GPU."""
        if not experts:
            return 0.0
        cached = sum(1 for e in experts if self.expert_cache.contains(e[0], e[1]))
        return cached / len(experts)
```

### 4.3 Integration with vLLM Scheduler

```python
def create_vllm_engine_with_expert_scheduling(
    loaded_state: LoadedWeightState,
    memory_coordinator: UnifiedMemoryCoordinator,
) -> AsyncLLMEngine:
    """Create vLLM engine with expert-aware scheduling."""
    
    # Create base engine
    engine_args = AsyncEngineArgs(
        model=loaded_state.model_info.model_id,
        **memory_coordinator.get_vllm_config(),
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Wrap scheduler with expert-affinity layer
    original_scheduler = engine.engine.scheduler
    expert_scheduler = ExpertAffinityScheduler(
        expert_cache=loaded_state.expert_cache,
        max_batch_size=memory_coordinator.get_vllm_config()["max_num_seqs"],
    )
    
    # Monkey-patch schedule method
    def expert_aware_schedule():
        # Get batch from expert scheduler
        batch = asyncio.run(expert_scheduler.schedule_batch())
        # Let vLLM scheduler finalize (handle preemption, etc.)
        return original_scheduler._finalize_batch(batch)
    
    engine.engine.scheduler.schedule = expert_aware_schedule
    
    return engine
```

---

## 5. CUDA Graph Handling

### 5.1 Problem

vLLM uses CUDA graphs for ~20% speedup by pre-compiling GPU kernel execution. But CUDA graphs require:
- **Fixed memory addresses**: Expert weights at static locations
- **Fixed control flow**: Same operations every time

Dynamic expert loading breaks both requirements.

### 5.2 Solution: Hybrid Approach

**Phase 1 (MVP)**: Disable CUDA graphs entirely
```python
engine_args = AsyncEngineArgs(
    model=model_id,
    enforce_eager=True,  # Disable CUDA graphs
)
```

**Trade-off**: Lose ~20% throughput, gain dynamic expert loading.

**Phase 2 (Optimized)**: Conditional CUDA graphs
```python
class ConditionalCUDAGraphManager:
    """Enable CUDA graphs when expert cache is stable."""
    
    def __init__(self, expert_cache: ExpertCache):
        self.expert_cache = expert_cache
        self.graph_enabled = False
        self.compiled_graphs = {}
        self.last_expert_set = None
        
    def should_use_graph(self) -> bool:
        """Enable graph if expert set hasn't changed in N requests."""
        current_expert_set = self._get_gpu_expert_set()
        
        if current_expert_set == self.last_expert_set:
            self.stable_count += 1
        else:
            self.stable_count = 0
            self.last_expert_set = current_expert_set
        
        # Enable after 100 requests with same experts
        return self.stable_count > 100
    
    def get_or_compile_graph(self, expert_set: frozenset):
        """Get cached graph or compile new one."""
        if expert_set not in self.compiled_graphs:
            # Compile CUDA graph for this expert configuration
            graph = self._compile_graph_for_experts(expert_set)
            self.compiled_graphs[expert_set] = graph
        return self.compiled_graphs[expert_set]
```

**Phase 3 (Advanced)**: Expert pinning
```python
# Pin most common experts to prevent eviction
hot_experts = expert_cache.get_most_accessed(top_k=8)
for layer_id, expert_id in hot_experts:
    expert_cache.pin(layer_id, expert_id)

# Now CUDA graphs can use pinned experts safely
engine_args = AsyncEngineArgs(
    model=model_id,
    enforce_eager=False,  # Enable CUDA graphs
)
```

---

## 6. Implementation Plan

### Phase 1: Minimal Integration (Week 1-2)

**Goal**: Make both systems coexist without conflicts.

1. **Unified Memory Coordinator**
   - Implement GPU memory splitting (40/50/10)
   - Configure vLLM with `gpu_memory_utilization=0.5`
   - Configure expert cache with 40% budget
   - File: `sparse_llm/integrations/memory_coordinator.py`

2. **Weight Bridge**
   - Implement `SparseMoEWeightBridge`
   - Hook into vLLM's weight loading
   - Test with single request
   - File: `sparse_llm/integrations/weight_bridge.py`

3. **Disable CUDA Graphs**
   - Set `enforce_eager=True`
   - Document performance trade-off
   - File: Update `serve.py`

4. **Fix serve.py**
   - Remove duplicate `SharedExpertWeightLoader` creation
   - Use `LoadedWeightState` directly
   - Inject weights via bridge

**Success Criteria**: 
- Single request completes end-to-end
- No OOM errors
- Expert cache hit rate > 0%

### Phase 2: Scheduling Optimization (Week 3-4)

**Goal**: Reduce cache thrashing with smart batching.

1. **Expert Affinity Scheduler**
   - Implement cache-hot batch selection
   - Implement expert clustering
   - File: `sparse_llm/integrations/expert_scheduler.py`

2. **Lightweight Router**
   - Predict expert usage for pending requests
   - Use model config routing patterns
   - File: `sparse_llm/routing/predictor.py`

3. **Integration**
   - Wrap vLLM scheduler
   - Monitor cache hit rate improvements
   - File: Update `serve.py`

**Success Criteria**:
- Batch size 16-32 works without thrashing
- Cache hit rate > 70%
- Throughput within 50% of standard vLLM

### Phase 3: CUDA Graph Recovery (Week 5-6)

**Goal**: Recover CUDA graph performance for stable workloads.

1. **Conditional Graph Manager**
   - Detect stable expert sets
   - Compile graphs per configuration
   - File: `sparse_llm/integrations/graph_manager.py`

2. **Expert Pinning**
   - Pin hot experts to prevent eviction
   - Reserve GPU space for pinned experts
   - File: Update `ExpertCache`

3. **Monitoring**
   - Track graph compilation overhead
   - A/B test graph vs eager mode
   - File: Add metrics to `/metrics` endpoint

**Success Criteria**:
- CUDA graphs enabled for 80%+ of requests
- Throughput within 80% of standard vLLM
- No regression in cache hit rate

---

## 7. Code Sketches

### 7.1 Updated serve.py (Key Changes)

```python
class UnifiedServer:
    """Fixed server with proper integration."""
    
    async def startup(self):
        # Phase 1: Load weights with resource-aware system
        orchestrator = FourPhaseOrchestrator()
        self.loaded_state = orchestrator.initialize(
            model_id=self.model,
            storage_path=self.storage_path,
            progress_callback=logger.info
        )
        
        # Phase 2: Create memory coordinator
        from sparse_llm.integrations import UnifiedMemoryCoordinator
        self.memory_coordinator = UnifiedMemoryCoordinator(
            total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory
        )
        
        # Phase 3: Create weight bridge
        from sparse_llm.integrations import SparseMoEWeightBridge
        self.weight_bridge = SparseMoEWeightBridge(self.loaded_state)
        
        # Phase 4: Initialize vLLM with custom worker
        engine_args = AsyncEngineArgs(
            model=self.model,
            **self.memory_coordinator.get_vllm_config(),
            # CRITICAL: Disable CUDA graphs for dynamic experts
            enforce_eager=True,
        )
        
        # Create engine with weight bridge injection
        self.vllm_engine = await self._create_engine_with_bridge(
            engine_args, self.weight_bridge
        )
        
        logger.info("✅ Integration complete")
    
    async def _create_engine_with_bridge(
        self, 
        engine_args: AsyncEngineArgs, 
        weight_bridge: SparseMoEWeightBridge
    ) -> AsyncLLMEngine:
        """Create vLLM engine with injected weight bridge."""
        
        # Create engine
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Inject weights into model
        model = engine.engine.model_executor.driver_worker.model_runner.model
        self._inject_weights(model, weight_bridge)
        
        return engine
    
    def _inject_weights(self, model, weight_bridge):
        """Replace model weights with pre-loaded ones."""
        for name, param in model.named_parameters():
            try:
                weight_tensor = weight_bridge.get_weight(name)
                param.data = weight_tensor
            except KeyError:
                logger.warning(f"Weight not found in bridge: {name}")
```

### 7.2 UnifiedMemoryCoordinator

```python
# sparse_llm/integrations/memory_coordinator.py

class UnifiedMemoryCoordinator:
    """Coordinates GPU memory between expert cache and vLLM KV cache."""
    
    EXPERT_CACHE_RATIO = 0.40
    KV_CACHE_RATIO = 0.50
    OVERHEAD_RATIO = 0.10
    SAFETY_MARGIN = 0.85  # Use 85% of total GPU
    
    def __init__(self, total_gpu_bytes: int):
        self.total_gpu = total_gpu_bytes
        self.usable_gpu = int(total_gpu_bytes * self.SAFETY_MARGIN)
        
        # Pre-allocate budgets
        self.expert_cache_budget = int(self.usable_gpu * self.EXPERT_CACHE_RATIO)
        self.kv_cache_budget = int(self.usable_gpu * self.KV_CACHE_RATIO)
        self.overhead_budget = int(self.usable_gpu * self.OVERHEAD_RATIO)
        
        logger.info(f"GPU Memory Allocation:")
        logger.info(f"  Total: {self.total_gpu / 1e9:.2f} GB")
        logger.info(f"  Usable: {self.usable_gpu / 1e9:.2f} GB")
        logger.info(f"  Expert Cache: {self.expert_cache_budget / 1e9:.2f} GB")
        logger.info(f"  KV Cache: {self.kv_cache_budget / 1e9:.2f} GB")
        logger.info(f"  Overhead: {self.overhead_budget / 1e9:.2f} GB")
    
    def get_vllm_config(self) -> dict:
        """Return vLLM configuration respecting memory budget."""
        return {
            "gpu_memory_utilization": self.KV_CACHE_RATIO / self.SAFETY_MARGIN,
            "max_num_seqs": self._estimate_max_batch_size(),
            "enforce_eager": True,  # Phase 1: disable CUDA graphs
            "trust_remote_code": True,
        }
    
    def get_expert_cache_config(self) -> dict:
        """Return expert cache configuration."""
        # Estimate CPU budget (10x GPU budget, typical for RAM)
        cpu_budget = self.expert_cache_budget * 10
        
        return {
            "gpu_cache_bytes": self.expert_cache_budget,
            "cpu_cache_bytes": cpu_budget,
        }
    
    def _estimate_max_batch_size(self) -> int:
        """Estimate safe batch size for KV cache budget."""
        # Rule of thumb: 128MB KV cache per request at 2048 context
        bytes_per_request = 128 * 1024 * 1024
        max_batch = self.kv_cache_budget // bytes_per_request
        return max(8, min(32, max_batch))
```

### 7.3 SparseMoEWeightBridge

```python
# sparse_llm/integrations/weight_bridge.py

class SparseMoEWeightBridge:
    """Bridge between vLLM model and LoadedWeightState.
    
    This is the critical integration point that makes vLLM use
    pre-loaded weights instead of loading from HuggingFace.
    """
    
    def __init__(self, loaded_state: LoadedWeightState):
        self.loaded_state = loaded_state
        self.expert_cache = loaded_state.expert_cache
        self.shared_weights = loaded_state.shared_weights
        self.model_info = loaded_state.model_info
        
        # Cache for parsed weight names
        self._name_cache: dict[str, tuple[str, int, int]] = {}
        
    def get_weight(self, name: str) -> torch.Tensor:
        """Get weight tensor by name.
        
        Args:
            name: Full parameter name (e.g., "model.layers.5.block_sparse_moe.experts.3.w1.weight")
        
        Returns:
            Weight tensor on GPU
        """
        # Check if expert weight
        if ".experts." in name:
            return self._get_expert_weight(name)
        else:
            return self._get_shared_weight(name)
    
    def _get_expert_weight(self, name: str) -> torch.Tensor:
        """Get expert weight from cache."""
        # Parse name
        if name not in self._name_cache:
            layer_id, expert_id, weight_key = self._parse_expert_name(name)
            self._name_cache[name] = (weight_key, layer_id, expert_id)
        else:
            weight_key, layer_id, expert_id = self._name_cache[name]
        
        # Get expert weights from cache (auto-promotes to GPU if needed)
        expert_weights, tier = self.expert_cache.get(layer_id, expert_id)
        
        # Extract specific weight
        if weight_key not in expert_weights:
            available = list(expert_weights.keys())
            raise KeyError(
                f"Weight '{weight_key}' not found in expert ({layer_id}, {expert_id}). "
                f"Available: {available}"
            )
        
        weight_tensor = expert_weights[weight_key]
        
        # Ensure on GPU
        if weight_tensor.device.type != "cuda":
            weight_tensor = weight_tensor.cuda()
        
        return weight_tensor
    
    def _get_shared_weight(self, name: str) -> torch.Tensor:
        """Get shared weight (non-expert)."""
        if name not in self.shared_weights:
            raise KeyError(f"Shared weight '{name}' not found in LoadedWeightState")
        
        weight_tensor = self.shared_weights[name]
        
        # Ensure on GPU
        if weight_tensor.device.type != "cuda":
            weight_tensor = weight_tensor.cuda()
        
        return weight_tensor
    
    def _parse_expert_name(self, name: str) -> tuple[int, int, str]:
        """Parse expert weight name into (layer_id, expert_id, weight_key).
        
        Example:
            "model.layers.5.block_sparse_moe.experts.3.w1.weight"
            → (5, 3, "w1.weight")
        """
        parts = name.split(".")
        
        # Find layer index
        try:
            layer_idx_pos = parts.index("layers") + 1
            layer_id = int(parts[layer_idx_pos])
        except (ValueError, IndexError):
            raise ValueError(f"Cannot parse layer_id from weight name: {name}")
        
        # Find expert index
        try:
            expert_idx_pos = parts.index("experts") + 1
            expert_id = int(parts[expert_idx_pos])
        except (ValueError, IndexError):
            raise ValueError(f"Cannot parse expert_id from weight name: {name}")
        
        # Weight key is everything after "experts.{id}."
        weight_key = ".".join(parts[expert_idx_pos + 1:])
        
        return layer_id, expert_id, weight_key
```

---

## 8. Trade-offs and Limitations

### 8.1 Performance Trade-offs

| Feature | Standard vLLM | Integrated System | Trade-off |
|---------|---------------|-------------------|-----------|
| **Throughput** | 100% | 50-80% (Phase 1-3) | CUDA graphs disabled initially |
| **Memory Efficiency** | Low (loads all experts) | High (dynamic loading) | Worth it for large models |
| **Batch Size** | 64-128 | 16-32 | Smaller batches to reduce thrashing |
| **Latency (single)** | Low | Low | Similar for single requests |
| **Latency (batched)** | Low | Medium | Expert loading overhead |

### 8.2 When to Use This Integration

**Use integrated system when**:
- Model too large for GPU memory (e.g., Mixtral 8x7B on 24GB GPU)
- Running on consumer hardware (4090, 3090)
- Memory more valuable than throughput
- Serving diverse workloads (cache benefits from locality)

**Use standard vLLM when**:
- Model fits in GPU memory
- Maximum throughput required
- Uniform workload (all experts needed)
- Production with high QPS

### 8.3 Known Limitations

1. **CUDA Graph Incompatibility (Phase 1)**
   - ~20% throughput loss vs standard vLLM
   - Mitigated in Phase 3 with conditional graphs

2. **Batch Size Constraints**
   - Smaller batches (16-32 vs 64-128)
   - Reduces GPU utilization
   - Necessary to prevent cache thrashing

3. **Expert Loading Latency**
   - First access: 50-100ms (from storage)
   - Second access: 5-10ms (from CPU)
   - Third+ access: <1ms (from GPU)
   - Predictive loading in scheduler mitigates this

4. **Memory Overhead**
   - 3-tier cache infrastructure
   - Weight bridge bookkeeping
   - ~5-10% memory overhead

5. **Complexity**
   - Two systems to monitor
   - More failure modes
   - Harder to debug

---

## 9. Validation and Testing

### 9.1 Integration Tests

```python
def test_memory_coordination():
    """Verify memory budgets are respected."""
    coordinator = UnifiedMemoryCoordinator(total_gpu_bytes=24 * 1024**3)
    
    # Check allocation ratios
    assert coordinator.expert_cache_budget / coordinator.usable_gpu ≈ 0.40
    assert coordinator.kv_cache_budget / coordinator.usable_gpu ≈ 0.50
    
    # Verify vLLM config
    config = coordinator.get_vllm_config()
    assert config["gpu_memory_utilization"] ≈ 0.588  # 50% of 85%

def test_weight_bridge():
    """Verify weight bridge returns correct tensors."""
    loaded_state = FourPhaseOrchestrator().initialize(model_id="test-model")
    bridge = SparseMoEWeightBridge(loaded_state)
    
    # Test shared weight
    shared_weight = bridge.get_weight("model.embed_tokens.weight")
    assert shared_weight.device.type == "cuda"
    
    # Test expert weight
    expert_weight = bridge.get_weight("model.layers.5.experts.3.w1.weight")
    assert expert_weight.device.type == "cuda"

def test_end_to_end_inference():
    """Run full inference pipeline."""
    server = UnifiedServer(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
    await server.startup()
    
    # Single request
    result = await server.generate_completion(
        prompt="Hello, world!",
        max_tokens=100
    )
    assert len(result["choices"]) > 0
    
    # Batched requests
    tasks = [
        server.generate_completion(prompt=f"Request {i}", max_tokens=50)
        for i in range(16)
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == 16
    
    # Check cache stats
    cache_stats = server.loaded_state.get_cache_stats()
    assert cache_stats["gpu_hit_rate"] > 0.5  # Should have some hits

def test_no_oom():
    """Verify no out-of-memory errors under load."""
    server = UnifiedServer(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
    await server.startup()
    
    # Sustained load: 100 requests
    for i in range(100):
        result = await server.generate_completion(
            prompt=f"Load test {i}",
            max_tokens=100
        )
        
        # Monitor GPU memory
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        assert reserved < coordinator.usable_gpu  # Never exceed budget
```

### 9.2 Performance Benchmarks

```bash
# Baseline: Standard vLLM (all experts loaded)
python benchmark.py --mode standard --batch-size 64

# Phase 1: Integrated (CUDA graphs disabled)
python benchmark.py --mode integrated --batch-size 16

# Phase 3: Integrated (CUDA graphs enabled, expert pinning)
python benchmark.py --mode integrated-optimized --batch-size 32

# Metrics to track:
# - Tokens/sec (throughput)
# - Latency (p50, p95, p99)
# - GPU memory usage
# - Cache hit rate
# - Expert loading overhead
```

---

## 10. Migration Path

### From Current Broken serve.py

1. **Add UnifiedMemoryCoordinator**
   - Import and instantiate before vLLM engine
   - Use `get_vllm_config()` for engine args
   - Use `get_expert_cache_config()` for expert cache

2. **Add SparseMoEWeightBridge**
   - Pass `LoadedWeightState` to bridge
   - Inject bridge into vLLM worker

3. **Remove Duplicate Loaders**
   - Delete `SharedExpertWeightLoader` creation in `initialize_vllm_engine()`
   - Delete `VLLMSparseExpertPager` creation
   - Use `LoadedWeightState.expert_cache` directly

4. **Set enforce_eager=True**
   - Add to `AsyncEngineArgs`
   - Document as Phase 1 limitation

### From Separate Systems (vllm_server.py)

1. **Merge Initialization**
   - Move `FourPhaseOrchestrator.initialize()` before vLLM
   - Pass `LoadedWeightState` to vLLM

2. **Unified Memory**
   - Replace separate `gpu_cache_gb` and `cpu_cache_gb` args
   - Use `UnifiedMemoryCoordinator` instead

3. **Single Expert Cache**
   - Remove duplicate expert cache in `VLLMSparseExpertPager`
   - Use `LoadedWeightState.expert_cache` for all expert access

---

## 11. Monitoring and Observability

### 11.1 Key Metrics

```python
@app.get("/metrics")
async def get_detailed_metrics():
    """Enhanced metrics endpoint."""
    return {
        "memory": {
            "gpu_total_bytes": coordinator.total_gpu,
            "gpu_allocated_bytes": torch.cuda.memory_allocated(),
            "gpu_reserved_bytes": torch.cuda.memory_reserved(),
            "expert_cache_bytes": server.loaded_state.expert_cache.bytes_used,
            "expert_cache_utilization": (
                server.loaded_state.expert_cache.bytes_used / 
                coordinator.expert_cache_budget
            ),
        },
        "expert_cache": {
            "gpu_cached_experts": len(server.loaded_state.expert_cache._gpu_cache),
            "cpu_cached_experts": len(server.loaded_state.expert_cache._cpu_cache),
            "hit_rate": server.loaded_state.expert_cache.stats["gpu_hit_rate"],
            "promotions": server.loaded_state.expert_cache.stats["promotions"],
            "evictions": server.loaded_state.expert_cache.stats["evictions"],
        },
        "vllm": {
            "running_requests": len(engine.engine.scheduler.running),
            "waiting_requests": len(engine.engine.scheduler.waiting),
            "max_batch_size": coordinator.get_vllm_config()["max_num_seqs"],
        },
        "performance": {
            "total_requests": server.request_count,
            "avg_tokens_per_sec": calculate_avg_throughput(),
            "uptime_seconds": time.time() - server.startup_time,
        },
    }
```

### 11.2 Warning Thresholds

```python
def check_health_and_warn():
    """Monitor system health and log warnings."""
    
    # Memory pressure
    if torch.cuda.memory_allocated() > coordinator.usable_gpu * 0.95:
        logger.warning("GPU memory at 95% capacity")
    
    # Cache thrashing
    cache_stats = server.loaded_state.expert_cache.stats
    if cache_stats["evictions"] > cache_stats["total_accesses"] * 0.5:
        logger.warning("High cache eviction rate - consider larger cache or smaller batches")
    
    # Low hit rate
    if cache_stats["gpu_hit_rate"] < 0.3:
        logger.warning("Low GPU cache hit rate - expert affinity scheduling recommended")
```

---

## 12. Future Optimizations

### 12.1 Expert Prefetching

Predict next experts needed and load them before requested.

```python
class ExpertPrefetcher:
    """Prefetch experts before they're requested."""
    
    def predict_next_experts(
        self, 
        current_token_ids: list[int],
        current_layer: int
    ) -> list[tuple[int, int]]:
        """Predict which experts will be needed in next layers."""
        # Use lightweight router to predict expert usage
        predictions = []
        for layer_id in range(current_layer + 1, current_layer + 3):
            top_experts = self.router.predict(current_token_ids, layer_id)
            predictions.extend([(layer_id, e) for e in top_experts])
        return predictions
    
    async def prefetch_async(self, expert_keys: list[tuple[int, int]]):
        """Asynchronously load experts in background."""
        for layer_id, expert_id in expert_keys:
            if not self.expert_cache.contains(layer_id, expert_id):
                # Load from storage to CPU in background
                asyncio.create_task(self._load_to_cpu(layer_id, expert_id))
```

### 12.2 Expert Compression

Store experts in compressed format (quantized) in CPU/storage tiers.

```python
class CompressedExpertCache:
    """Three-tier cache with compression."""
    
    def preload_cpu(self, layer_id: int, expert_id: int, weights: dict):
        # Quantize to int8 before storing on CPU
        compressed_weights = {
            name: quantize_tensor(tensor, bits=8)
            for name, tensor in weights.items()
        }
        super().preload_cpu(layer_id, expert_id, compressed_weights)
    
    def _promote_to_gpu(self, key, weights):
        # Dequantize when promoting to GPU
        decompressed = {
            name: dequantize_tensor(tensor)
            for name, tensor in weights.items()
        }
        super()._promote_to_gpu(key, decompressed)
```

### 12.3 Multi-GPU Support

Distribute expert cache across multiple GPUs.

```python
class MultiGPUExpertCache:
    """Shard expert cache across GPUs."""
    
    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus
        self.caches = [
            ExpertCache(..., device=f"cuda:{i}")
            for i in range(num_gpus)
        ]
    
    def get(self, layer_id: int, expert_id: int):
        # Hash expert to GPU
        gpu_id = (layer_id * 8 + expert_id) % self.num_gpus
        return self.caches[gpu_id].get(layer_id, expert_id)
```

---

## 13. Summary

This architecture resolves conflicts between resource-aware loading and vLLM by:

1. **Clear ownership**: Resource-aware system manages weights, vLLM manages inference
2. **Unified memory**: Static GPU allocation (40% experts, 50% KV cache, 10% overhead)
3. **Weight bridge**: Injects pre-loaded weights into vLLM models
4. **Smart batching**: Groups requests by expert affinity to reduce thrashing
5. **Phased approach**: Start simple (no CUDA graphs), optimize later (conditional graphs)

**Implementation Priority**:
- **Phase 1 (MVP)**: Memory coordinator + weight bridge + disable CUDA graphs
- **Phase 2 (Optimize)**: Expert affinity scheduling
- **Phase 3 (Recover)**: Conditional CUDA graphs + expert pinning

**Expected Results**:
- Phase 1: 50% throughput vs standard vLLM, 3x memory efficiency
- Phase 2: 65% throughput, 70%+ cache hit rate
- Phase 3: 80% throughput, CUDA graphs for stable workloads

This design makes both systems complementary rather than conflicting, delivering memory efficiency without sacrificing vLLM's core benefits.
