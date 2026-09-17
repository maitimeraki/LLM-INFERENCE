# Unified Memory Synchronization: vLLM Integration + Dynamic Multi-Tier Architecture
**Date**: 2026-09-06  
**Status**: Planning  
**Priority**: Critical

## Executive Summary

**Goal**: Solve the vLLM OOM problem while enabling large models on limited resources (4-6GB GPU) with dynamic, user-configurable context windows up to 1M tokens, **guaranteeing zero OOM errors**.

**Unified Solution**: **Adaptive Multi-Tier Memory Orchestration with vLLM Integration**

This plan merges two complementary approaches:
1. **vLLM Synchronization**: Pre-calculate memory budgets BEFORE loading to prevent OOM during vLLM initialization
2. **Dynamic Multi-Tier**: Runtime adaptive allocation across GPU/CPU/SSD for flexible context lengths

### Core Innovations

- **Pre-reservation**: Calculate ALL memory needs (model weights + vLLM KV cache + activations) BEFORE any loading
- **Three-tier weight management**: GPU ↔ CPU ↔ SSD for expert weights (ALREADY IMPLEMENTED)
- **Three-tier KV cache management**: GPU ↔ CPU ↔ Disk for KV cache using vLLM's native `--cpu-offload-gb` and LMCache
- **Unified multi-tier coordination**: Coordinate BOTH expert weights AND KV cache across all three tiers simultaneously
- **vLLM native integration**: Use vLLM's built-in KV cache offloading (no fork needed)
- **Real-time monitoring**: Memory pressure tracking and predictive OOM prevention
- **User control preserved**: All existing vLLM parameters remain user-configurable (dtype, quantization, max-model-len, temperature, etc.)
- **Admission control**: Reject requests before loading if resources insufficient
- **Zero OOM guarantee**: System automatically cascades GPU → CPU → SSD for both weights and KV cache

### Architectural Foundation: Reusing Proven Integrations

**CRITICAL PRINCIPLE**: This plan EXTENDS existing, proven components rather than rebuilding from scratch.

**CRITICAL CLARIFICATION**: vLLM is the ONLY serving backend. We do NOT replace vLLM's KV cache management. vLLM's PagedAttention handles ALL KV cache operations (GPU-based). This plan fixes the coordination between expert weight loading and vLLM initialization.

From **Resource-Aware Weight Loading** (`RESOURCE_AWARE_WEIGHT_LOADING_DESIGN.md`):
- ✅ **FourPhaseOrchestrator**: Already implements Phase 1-4 initialization pattern
- ✅ **ResourceProfiler**: Already detects GPU/CPU/SSD with validated safety margins (85%/80%/90%)
- ✅ **PlacementStrategyCalculator**: Already calculates three-tier weight placement
- ✅ **Three-tier weight cache**: GPU (hot) → CPU (warm) → SSD (cold) already working

From **Universal Model Support** (`IMPLEMENTATION_PLAN.md`):
- ✅ **Architecture-agnostic loading**: Works with ANY Hugging Face model
- ✅ **Capability-driven specialization**: MoE vs dense detection and fallback
- ✅ **Layer-aware expert cache**: `LayerAwareCache` with LRU, pin/unpin, concurrency-safe

From **Prior vLLM Integration** (`VLLM_INTEGRATION.md`):
- ⚠️ **vLLM bridge exists**: But causes OOM during initialization (PROBLEM TO FIX)
- ✅ **Weight loader integration**: Structure for injecting weights into vLLM
- ✅ **User parameter passing**: vLLM command-line arguments already configurable

**What This Plan Adds**:
1. **Unified three-tier coordination** - Coordinate BOTH expert weights AND KV cache across GPU/CPU/SSD
2. **Dynamic vLLM multi-tier parameters** - Calculate `--gpu-memory-utilization` AND `--cpu-offload-gb` based on expert allocation
3. **Pre-calculation phase** - Adds "Phase 0" before `FourPhaseOrchestrator` for unified tier planning
4. **Admission control** - Rejects requests BEFORE loading if total resources (weights + KV cache) exceed available across all tiers
5. **Per-request validation** - Checks if user's `max_model_len` + expert needs fits across GPU/CPU/SSD
6. **LMCache integration** - Enable disk-tier KV cache for 100K-200K+ context windows on small GPUs
7. **Cascading fallback** - Automatically uses GPU → CPU → SSD tiers to prevent OOM at any stage

**Implementation Strategy**: EXTEND existing components. Use vLLM's native multi-tier KV cache features (`--cpu-offload-gb`, `--enable-lmcache`) and coordinate with existing expert weight three-tier system (`LayerAwareCache`).

---

## Problem Statement

### Current Issues

1. **vLLM OOM Error**: Sequential loading (shared weights → experts → vLLM initialization) causes OOM when vLLM tries to allocate its internal KV cache
2. **No Pre-Calculation**: System doesn't calculate total memory budget (weights + vLLM KV cache + activations) BEFORE loading
3. **No Coordination Between Tiers**: Expert weights and KV cache compete for GPU/CPU/SSD space without coordination
4. **Sequential Loading Race**: Expert weights load to GPU first, then vLLM tries to allocate remaining space, but calculation is wrong
5. **No Admission Control**: Requests accepted without checking if available memory can handle them
6. **Limited Context Windows**: Without proper tier coordination, context limited to ~8K-16K tokens on 4-6GB GPUs

**CRITICAL CLARIFICATION**: vLLM DOES support multi-tier KV cache offloading natively:
- GPU tier: PagedAttention (fastest)
- CPU tier: `--cpu-offload-gb` parameter (medium speed)
- Disk tier: LMCache integration (slower but unlimited)

**The Problem**: We load expert weights WITHOUT coordinating with vLLM's KV cache tiers, causing conflicts and OOM.

**What vLLM Already Does Well**:
- ✅ Efficient three-tier KV cache (GPU → CPU → Disk)
- ✅ Multi-user request batching and scheduling
- ✅ Continuous batching for optimal throughput
- ✅ User-configurable `--cpu-offload-gb`, `--max-model-len`, `--dtype`, `--quantization`
- ✅ Per-request temperature, top_p, max_tokens control
- ✅ Automatic tier promotion/demotion based on access patterns

**What We Need to Fix**:
- ❌ Pre-calculate memory budget BEFORE loading any weights
- ❌ Coordinate expert weight placement (GPU/CPU/SSD) with vLLM's KV cache tiers
- ❌ Set vLLM's `--gpu-memory-utilization` AND `--cpu-offload-gb` dynamically based on expert cache size
- ❌ Unified tier management: both expert weights and KV cache use coordinated GPU/CPU/SSD allocation
- ❌ Reject requests before loading if total memory (weights + KV cache) exceeds available resources across all tiers

### User Requirements

1. **vLLM Integration**: Use vLLM's optimized inference without OOM conflicts
2. **Flexible Context**: Support 2K to 1M token context (user-specified per request)
3. **Limited Hardware**: Must work on 4GB or 6GB GPUs
4. **Zero OOM Guarantee**: System must never crash with OOM
5. **Predictable Performance**: Users should know latency/throughput tradeoffs
6. **Runtime Configurability**: Users control ALL parameters at server startup and per-request
7. **Multi-User Support**: vLLM handles request batching/scheduling for multiple concurrent users

### Current System Architecture (IMPORTANT CONTEXT)

**The system already uses vLLM as the primary serving backend:**

- **vLLM handles**: Request batching, scheduling, KV cache management, continuous batching, paged attention
- **vLLM optimizes**: Multi-user serving automatically (single or multiple users, doesn't matter)
- **User controls at server startup**:
  - Model precision: `--dtype` (float16, bfloat16, float32, auto)
  - Quantization: `--quantization` (awq, gptq, squeezellm, fp8, etc.)
  - KV cache size: `--gpu-memory-utilization` (fraction of GPU for KV cache)
  - Max model length: `--max-model-len` (context window limit)
  - Tensor parallelism: `--tensor-parallel-size`
  - All standard vLLM parameters via command-line

- **User controls per-request** (OpenAI-compatible API):
  - `temperature`, `top_p`, `top_k`, `max_tokens`
  - `frequency_penalty`, `presence_penalty`
  - `stop` sequences
  - Any generation parameter

- **Custom configurations in `main.py`**:
  - Additional model-specific settings
  - Resource-aware loading preferences
  - Cache strategies

**KEY INSIGHT**: vLLM ALREADY manages KV cache efficiently with PagedAttention. The OOM problem is NOT vLLM's KV management—it's the **sequential loading** that doesn't pre-calculate total memory needs before initializing vLLM.

---

## Solution Architecture

### Core Principle: **Adaptive Resource Orchestration**

Instead of fixed ratios, dynamically calculate allocations based on:
1. **User Request**: Context length, batch size, generation length
2. **Model Architecture**: Shared weights, expert size, attention mechanism
3. **Available Resources**: Current GPU/CPU/SSD usage
4. **Runtime Pressure**: Real-time memory monitoring

### Three-Tier Architecture

**UNIFIED MULTI-TIER SYSTEM**: Both expert weights AND KV cache use coordinated three-tier placement.

```
┌─────────────────────────────────────────────────────────────────────┐
│              ADAPTIVE MEMORY ORCHESTRATOR (Phase 0)                  │
│  Pre-calculates unified allocation across ALL tiers for BOTH:        │
│    1. Expert weights (SparseLLM manages)                             │
│    2. KV cache (vLLM manages with --cpu-offload-gb + LMCache)        │
└─────────────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │  GPU TIER     │  │  CPU TIER     │  │  SSD/DISK     │
    │  (Hot/Fast)   │  │  (Warm/Med)   │  │  (Cold/Slow)  │
    └───────────────┘  └───────────────┘  └───────────────┘
            │                  │                  │
    ┌───────┴────────┐ ┌──────┴───────┐ ┌────────┴────────┐
    │ EXPERT WEIGHTS │ │ EXPERT WEIGHTS│ │ EXPERT WEIGHTS  │
    │ ─────────────  │ │ ────────────  │ │ ──────────────  │
    │ Shared: 2.7GB  │ │ Warm experts  │ │ Cold experts    │
    │ Hot experts:   │ │ (10-30 exp)   │ │ (100+ experts)  │
    │ (2-5 experts)  │ │               │ │                 │
    │ ─────────────  │ │ ────────────  │ │ ──────────────  │
    │                │ │               │ │                 │
    │ KV CACHE       │ │ KV CACHE      │ │ KV CACHE        │
    │ (vLLM)         │ │ (vLLM)        │ │ (LMCache)       │
    │ ─────────────  │ │ ────────────  │ │ ──────────────  │
    │ Recent tokens  │ │ Older tokens  │ │ Archive tokens  │
    │ (0-8K context) │ │ (8K-64K ctx)  │ │ (64K-200K ctx)  │
    │ PagedAttention │ │ --cpu-offload │ │ LMCache disk    │
    └────────────────┘ └───────────────┘ └─────────────────┘
```

### Tier Coordination Details

**GPU Tier (Hot - Fastest)**:
- Expert weights: Shared weights (always) + 2-5 hot experts (most frequent)
- KV cache: Recent tokens (0-8K context) via vLLM PagedAttention
- Total budget: Dynamically split based on workload
- Example: 6GB GPU → 3GB weights + 2.5GB KV cache + 0.5GB activations

**CPU Tier (Warm - Medium Speed)**:
- Expert weights: 10-30 warm experts via SparseLLM LayerAwareCache
- KV cache: Older tokens (8K-64K context) via vLLM `--cpu-offload-gb`
- Coordination: Reserve CPU space for BOTH, not just one
- Example: 32GB CPU → 10GB expert weights + 20GB KV cache + 2GB system

**SSD/Disk Tier (Cold - Slow but Unlimited)**:
- Expert weights: 100+ cold experts via SparseLLM safetensors storage
- KV cache: Archive tokens (64K-200K+ context) via LMCache
- No coordination needed: Both use separate storage locations
- Example: 500GB SSD → 50GB expert weights + 100GB KV cache archives

**Key Innovation**: Pre-calculate allocation for ALL resources across ALL tiers BEFORE loading anything.

---

## Integration with Existing Systems

### Leveraging Prior Integrations

This plan builds upon **proven architectural patterns** from prior integrations:

1. **Four-Phase Orchestration** (from `RESOURCE_AWARE_WEIGHT_LOADING_DESIGN.md`)
   - Already implemented: `FourPhaseOrchestrator` class
   - Proven pattern: Resource profiling → Placement strategy → Weight loading → Backend integration
   - **Reuse**: Extend existing orchestrator rather than rebuild from scratch

2. **Resource Profiling** (from `ResourceProfiler` class)
   - Already implemented: GPU/CPU/SSD detection with safety margins
   - Validated margins: 85% GPU, 80% CPU, 90% storage
   - **Reuse**: Add dynamic allocation on top of existing profiler

3. **Three-Tier Placement** (from `PlacementStrategyCalculator`)
   - Already implemented: Hot (GPU) → Warm (CPU) → Cold (SSD) tiers
   - Expert cache with LRU eviction already working
   - **Reuse**: Extend for KV cache in addition to expert weights

4. **Universal Model Support** (from `IMPLEMENTATION_PLAN.md`)
   - Architecture-agnostic loading already validated
   - MoE detection and fallback strategies proven
   - **Reuse**: Apply same capability-driven approach

### What's New in This Plan

1. **Dynamic `--gpu-memory-utilization` calculation** - Compute based on expert weight allocation, not fixed value
2. **Pre-calculation admission control** - Validate BEFORE loading that total memory fits
3. **vLLM synchronization** - Set vLLM parameters dynamically based on expert cache size
4. **Per-request memory validation** - Check if user's requested context length + batch size fits
5. **Preserved user control** - ALL existing vLLM CLI parameters remain configurable

**REMOVED from original plan** (not needed given vLLM architecture):
- ❌ Three-tier KV cache - vLLM's PagedAttention already handles KV cache optimally on GPU
- ❌ CPU/SSD KV cache offloading - vLLM manages this internally, no need to duplicate
- ❌ Custom KV cache tiers - Would conflict with vLLM's paged attention system

### Component Mapping: Existing → New

This table shows how new components build upon existing ones:

| Existing Component | Location | Status | New Component | Purpose |
|-------------------|----------|--------|---------------|---------|
| `FourPhaseOrchestrator` | `sparse_llm/loading/orchestrator.py` | ✅ Working | `AdaptiveMemoryOrchestrator` | Wraps orchestrator with Phase 0 pre-calculation |
| `ResourceProfiler` | `sparse_llm/loading/profiler.py` | ✅ Working | `AdaptiveResourceProfiler` | Extends with dynamic vLLM parameter calculation |
| `PlacementStrategyCalculator` | `sparse_llm/loading/strategy.py` | ✅ Working | *(reuse as-is)* | Now receives dynamic budgets for expert weights |
| `LayerAwareCache` | `sparse_llm/core/cache.py` | ✅ Working | *(reuse as-is)* | Already handles expert weight caching |
| `WeightLoader` | `sparse_llm/loading/loader.py` | ✅ Working | *(reuse as-is)* | Already loads to GPU/CPU/SSD tiers |
| *(none - new)* | N/A | ❌ Missing | `DynamicMemoryBudgetCalculator` | NEW: Calculate vLLM memory budget based on expert allocation |
| *(none - new)* | N/A | ❌ Missing | `MemoryPressureMonitor` | NEW: Real-time pressure tracking |
| *(none - new)* | N/A | ❌ Missing | `AdaptiveRequestHandler` | NEW: Admission control logic |
| `vllm_bridge.py` | `sparse_llm/integrations/vllm_bridge.py` | ⚠️ OOM issue | *(fix existing)* | Add memory coordination to prevent OOM |

**Key Insight**: ~70% of required components already exist. This plan primarily adds pre-calculation and vLLM coordination logic.

---

### Architectural Flow: Before vs After

**BEFORE (Current - causes OOM):**
```
User starts server:
  python serve.py --model Mixtral-8x7B --max-model-len 32768
    ↓
FourPhaseOrchestrator.initialize()
    ↓ Phase 1: Profile resources (GPU/CPU/SSD detection)
    ↓ Phase 2: Calculate placement (expert weights to GPU/CPU/SSD)
    ↓ Phase 3: Load weights (shared weights + hot experts → GPU)
    ↓           (Expert weights now occupy ~3-4GB of 6GB GPU)
    ↓ Phase 4: Initialize vLLM (tries to allocate GPU KV cache for 32K context)
    ↓
vLLM tries to allocate 2.5GB KV cache on GPU → But only 2GB free!
    ↓
OOM! 💥
```

**AFTER (This Plan - zero OOM with unified three-tier coordination):**
```
User starts server with long context:
  python serve.py --model Mixtral-8x7B --max-model-len 131072 --dtype bfloat16
    ↓
Phase 0: UNIFIED PRE-CALCULATION (NEW)
    ├─ Detect available resources:
    │    - GPU: 6GB (5.1GB usable after 85% margin)
    │    - CPU: 32GB (25.6GB usable after 80% margin)
    │    - SSD: 500GB (450GB usable after 90% margin)
    │
    ├─ Calculate model requirements:
    │    - Shared weights: 2.7GB (must be GPU)
    │    - Total experts: 192 experts × 180MB = 34.5GB
    │    - KV cache for 131K context: ~25GB (2×24×2048×131072×2 bytes)
    │    - Activations: 0.5GB
    │
    ├─ UNIFIED TIER ALLOCATION across GPU/CPU/SSD:
    │
    │  GPU Tier (5.1GB total):
    │    ├─ Shared weights: 2.7GB
    │    ├─ Hot experts: 3 experts × 180MB = 0.54GB
    │    ├─ KV cache (recent 8K tokens): 1.5GB
    │    ├─ Activations: 0.5GB
    │    └─ Total: 5.24GB → Fits! (with small margin)
    │
    │  CPU Tier (25.6GB total):
    │    ├─ Warm experts: 20 experts × 180MB = 3.6GB
    │    ├─ KV cache (8K-64K tokens): 11GB
    │    └─ Total: 14.6GB → Fits!
    │
    │  SSD Tier (450GB total):
    │    ├─ Cold experts: 169 experts × 180MB = 30.4GB
    │    ├─ KV cache (64K-131K tokens): 13GB
    │    └─ Total: 43.4GB → Fits!
    │
    ├─ Calculate vLLM parameters:
    │    - gpu_memory_utilization = 1.5GB / 6GB = 0.25
    │    - cpu_offload_gb = 11 (reserve 11GB CPU for KV overflow)
    │    - enable_lmcache = true (for disk-tier KV cache)
    │
    └─ Validation: ✅ Total allocation fits across all tiers!
    ↓
AdaptiveMemoryOrchestrator (wraps FourPhaseOrchestrator)
    ↓ Phase 1: Profile resources (REUSE existing ResourceProfiler)
    ↓ Phase 2: Calculate placement with unified tier budgets
    ↓ Phase 3: Load expert weights to GPU/CPU/SSD tiers
    ↓ Phase 4: Initialize vLLM with coordinated parameters:
    ↓           --gpu-memory-utilization 0.25
    ↓           --cpu-offload-gb 11
    ↓           --enable-lmcache
    ↓           --max-model-len 131072
    ↓
vLLM initializes successfully:
    ├─ GPU KV cache: 1.5GB (no conflict with weights!)
    ├─ CPU KV cache: 11GB (automatically offloads when GPU full)
    └─ Disk KV cache: 13GB (LMCache archives oldest tokens)
    ↓
Success! ✅ Server ready for 131K context inference
    ↓
User sends request with 100K context + generation:
    ↓
vLLM handles inference with automatic tier management:
    ├─ Tokens 0-8K: GPU (fast, ~50 tokens/sec)
    ├─ Tokens 8K-64K: CPU offload (medium, ~20 tokens/sec)
    └─ Tokens 64K-100K: Disk offload (slower, ~5 tokens/sec)
    ↓
Response generated successfully without OOM!
```

**Critical Differences**:
1. **Unified pre-calculation** allocates BOTH weights AND KV cache across ALL tiers
2. **Coordinated tier budgets** ensure expert weights and KV cache don't compete
3. **vLLM receives THREE parameters**: `--gpu-memory-utilization`, `--cpu-offload-gb`, `--enable-lmcache`
4. **Cascading fallback**: GPU full → CPU, CPU full → SSD (for both weights and KV cache)
5. **Zero OOM guarantee**: System validates total fit before loading anything

---

## Component 1: Dynamic Memory Budget Calculator

### Purpose
Calculate memory allocations **on-demand** based on user request parameters, replacing fixed ratios with adaptive allocation.

### Integration Point
**Extends**: `sparse_llm.loading.ResourceProfiler`  
**Used by**: `AdaptiveMemoryOrchestrator` (new component wrapping `FourPhaseOrchestrator`)

### Preserves Existing Safety Margins

From `RESOURCE_AWARE_WEIGHT_LOADING_DESIGN.md`, the following validated safety margins MUST be preserved:

| Resource | Safety Margin | Reason |
|----------|---------------|---------|
| GPU VRAM | 85% utilization (15% reserved) | PyTorch memory fragmentation, temporary tensors during forward pass |
| CPU RAM | 80% utilization (20% reserved) | OS needs, other processes, swap safety |
| SSD/Storage | 90% utilization (10% reserved) | Filesystem overhead, temporary files |

**CRITICAL**: These margins are validated from prior integration. Do not modify without explicit user approval.

### Design

```python
@dataclass
class UserRequest:
    """User's inference configuration (from vLLM server startup parameters)."""
    max_model_len: int = 8192          # From --max-model-len (vLLM parameter)
    dtype: str = "auto"                # From --dtype (vLLM parameter)
    quantization: Optional[str] = None # From --quantization (vLLM parameter)
    tensor_parallel_size: int = 1      # From --tensor-parallel-size
    
    # User can still specify preferences for expert placement
    allow_cpu_offload: bool = True
    allow_ssd_offload: bool = True

@dataclass
class ModelInfo:
    """Model architecture metadata."""
    model_id: str
    is_moe: bool
    num_experts: int
    experts_per_layer: int
    num_layers: int
    hidden_size: int
    total_params: int
    shared_weight_bytes: int  # Non-expert weights (embeddings, attention, etc.)
    expert_weight_bytes: int  # Single expert size
    
@dataclass
class MemoryAllocation:
    """Calculated memory distribution."""
    # GPU allocation
    gpu_shared_weights: int      # Shared weights (always GPU)
    gpu_hot_experts: int         # Number of hot experts on GPU
    gpu_expert_bytes: int        # Bytes for hot experts
    gpu_activation_buffer: int   # Forward pass activations
    gpu_vllm_kv_cache: int       # Reserved for vLLM's KV cache
    gpu_total_used: int          # Total GPU usage
    
    # CPU allocation
    cpu_warm_experts: int        # Number of experts on CPU
    cpu_expert_bytes: int        # Bytes for warm experts
    
    # SSD allocation
    ssd_cold_experts: int        # Number of experts on SSD
    ssd_expert_bytes: int        # Bytes for cold experts
    
    # vLLM coordination
    vllm_gpu_memory_utilization: float  # Calculated value for vLLM's --gpu-memory-utilization
    
    # Validation
    can_fulfill: bool
    rejection_reason: Optional[str] = None

class DynamicMemoryBudgetCalculator:
    """
    Calculates memory allocation dynamically based on:
    1. User's vLLM parameters (max_model_len, dtype, quantization)
    2. Model architecture (shared weights, expert count/size)
    3. Available hardware resources (GPU/CPU/SSD)
    
    Key responsibility: Calculate vLLM's --gpu-memory-utilization parameter
    to prevent conflict with expert weights.
    """
    
    def __init__(
        self,
        resource_budget: ResourceBudget,  # From existing ResourceProfiler
        model_info: ModelInfo
    ):
        self.resource_budget = resource_budget
        self.model_info = model_info
    
    def calculate_allocation(
        self, 
        user_request: UserRequest
    ) -> MemoryAllocation:
        """
        Calculate memory allocation that prevents vLLM OOM.
        
        Algorithm:
        1. Calculate vLLM's KV cache needs (based on --max-model-len)
        2. Calculate shared weight size (always on GPU)
        3. Calculate remaining GPU space for hot experts
        4. Offload remaining experts to CPU/SSD
        5. Return vllm_gpu_memory_utilization parameter
        """
        
        gpu_available = self.resource_budget.gpu_usable_bytes  # Already has 85% safety margin
        cpu_available = self.resource_budget.cpu_usable_bytes  # Already has 80% safety margin
        ssd_available = self.resource_budget.ssd_usable_bytes  # Already has 90% safety margin
        
        # Step 1: Calculate vLLM KV cache size
        # vLLM needs: (max_model_len * batch_size * num_layers * hidden_size * 2 (K+V) * dtype_bytes)
        dtype_bytes = self._get_dtype_bytes(user_request.dtype)
        kv_cache_bytes = self._calculate_vllm_kv_cache(
            max_model_len=user_request.max_model_len,
            num_layers=self.model_info.num_layers,
            hidden_size=self.model_info.hidden_size,
            dtype_bytes=dtype_bytes
        )
        
        # Step 2: Shared weights must be on GPU (critical for performance)
        shared_weights_bytes = self.model_info.shared_weight_bytes
        
        # Step 3: Activation buffer (temporary tensors during forward pass)
        activation_buffer = self._estimate_activation_buffer(
            hidden_size=self.model_info.hidden_size,
            dtype_bytes=dtype_bytes
        )
        
        # Step 4: Calculate remaining GPU space for hot experts
        gpu_reserved = shared_weights_bytes + kv_cache_bytes + activation_buffer
        gpu_remaining_for_experts = gpu_available - gpu_reserved
        
        if gpu_remaining_for_experts < 0:
            # Cannot fit even without any experts on GPU
            return MemoryAllocation(
                gpu_shared_weights=shared_weights_bytes,
                gpu_hot_experts=0,
                gpu_expert_bytes=0,
                gpu_activation_buffer=activation_buffer,
                gpu_vllm_kv_cache=kv_cache_bytes,
                gpu_total_used=gpu_reserved,
                cpu_warm_experts=0,
                cpu_expert_bytes=0,
                ssd_cold_experts=0,
                ssd_expert_bytes=0,
                vllm_gpu_memory_utilization=0.0,
                can_fulfill=False,
                rejection_reason=f"Insufficient GPU memory: need {gpu_reserved / 1e9:.2f}GB, have {gpu_available / 1e9:.2f}GB. Try: reduce --max-model-len, enable --quantization, or use smaller model."
            )
        
        # Step 5: Allocate hot experts to GPU
        expert_size = self.model_info.expert_weight_bytes
        num_hot_experts = min(
            int(gpu_remaining_for_experts / expert_size),
            self.model_info.num_experts  # Can't exceed total experts
        )
        gpu_expert_bytes = num_hot_experts * expert_size
        
        # Step 6: Offload remaining experts to CPU/SSD
        remaining_experts = self.model_info.num_experts - num_hot_experts
        
        if user_request.allow_cpu_offload:
            num_warm_experts = min(
                int(cpu_available / expert_size),
                remaining_experts
            )
            cpu_expert_bytes = num_warm_experts * expert_size
            remaining_experts -= num_warm_experts
        else:
            num_warm_experts = 0
            cpu_expert_bytes = 0
        
        if user_request.allow_ssd_offload and remaining_experts > 0:
            num_cold_experts = remaining_experts
            ssd_expert_bytes = num_cold_experts * expert_size
            
            if ssd_expert_bytes > ssd_available:
                return MemoryAllocation(
                    gpu_shared_weights=shared_weights_bytes,
                    gpu_hot_experts=num_hot_experts,
                    gpu_expert_bytes=gpu_expert_bytes,
                    gpu_activation_buffer=activation_buffer,
                    gpu_vllm_kv_cache=kv_cache_bytes,
                    gpu_total_used=gpu_reserved + gpu_expert_bytes,
                    cpu_warm_experts=num_warm_experts,
                    cpu_expert_bytes=cpu_expert_bytes,
                    ssd_cold_experts=0,
                    ssd_expert_bytes=0,
                    vllm_gpu_memory_utilization=0.0,
                    can_fulfill=False,
                    rejection_reason=f"Insufficient storage: need {ssd_expert_bytes / 1e9:.2f}GB, have {ssd_available / 1e9:.2f}GB. Free up disk space."
                )
        else:
            num_cold_experts = 0
            ssd_expert_bytes = 0
        
        # Step 7: Calculate vLLM's gpu_memory_utilization parameter
        # This tells vLLM how much of the GPU it can use for KV cache
        # Formula: (KV cache size) / (total GPU memory)
        total_gpu_memory = self.resource_budget.gpu_total_bytes
        vllm_gpu_memory_utilization = kv_cache_bytes / total_gpu_memory
        
        # Step 8: Validate total allocation
        gpu_total_used = shared_weights_bytes + gpu_expert_bytes + kv_cache_bytes + activation_buffer
        
        return MemoryAllocation(
            gpu_shared_weights=shared_weights_bytes,
            gpu_hot_experts=num_hot_experts,
            gpu_expert_bytes=gpu_expert_bytes,
            gpu_activation_buffer=activation_buffer,
            gpu_vllm_kv_cache=kv_cache_bytes,
            gpu_total_used=gpu_total_used,
            cpu_warm_experts=num_warm_experts,
            cpu_expert_bytes=cpu_expert_bytes,
            ssd_cold_experts=num_cold_experts,
            ssd_expert_bytes=ssd_expert_bytes,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            can_fulfill=True,
            rejection_reason=None
        )
    max_generation_length: int = 512 # Tokens to generate
    allow_cpu_offload: bool = True  # Enable CPU tier for KV cache
    allow_ssd_offload: bool = True  # Enable SSD tier for KV cache
    min_gpu_kv_tokens: int = 512    # Keep at least this many tokens on GPU


@dataclass
class MemoryAllocation:
    """Calculated memory allocation across all tiers."""
    
    # Model weights
    gpu_shared_weights: int         # Shared weights (always GPU)
    gpu_hot_experts: int            # Hot expert count
    cpu_warm_experts: int           # Warm expert count
    ssd_cold_experts: int           # Cold expert count (safetensors)
    
    # KV cache per tier
    gpu_kv_cache_bytes: int         # Recent tokens
    gpu_kv_cache_tokens: int        # How many tokens fit
    cpu_kv_cache_bytes: int         # Older tokens
    cpu_kv_cache_tokens: int        # How many tokens fit
    ssd_kv_cache_bytes: int         # Archive tokens
    ssd_kv_cache_tokens: int        # How many tokens fit
    
    # Operational
    activation_buffer: int          # Forward pass activations
    safety_margin: int              # Unused reserve
    
    # Predictions
    expected_gpu_usage: int         # Total GPU bytes used
    expected_cpu_usage: int         # Total CPU bytes used
    expected_ssd_usage: int         # Total SSD bytes used
    can_fulfill: bool               # Whether request is feasible
    rejection_reason: str | None    # Why it can't be fulfilled


class DynamicMemoryBudgetCalculator:
    """Calculate memory allocations based on user request."""
    
    def __init__(
        self,
        gpu_total_bytes: int,
        cpu_total_bytes: int,
        ssd_total_bytes: int,
        model_info: ModelInfo
    ):
        self.gpu_total = gpu_total_bytes
        self.cpu_total = cpu_total_bytes
        self.ssd_total = ssd_total_bytes
        self.model_info = model_info
    
    def calculate_allocation(
        self, 
        request: UserRequest
    ) -> MemoryAllocation:
        """
        Calculate memory allocation for user request.
        
        Algorithm:
        1. Calculate exact model weight requirements (immutable)
        2. Calculate KV cache requirements based on context length
        3. Allocate KV cache across tiers (GPU → CPU → SSD)
        4. Calculate activation buffer needs
        5. Validate total doesn't exceed available resources
        6. Return allocation or rejection
        """
        
        # Step 1: Model weights (immutable requirements)
        shared_weights = self.model_info.shared_weight_bytes
        single_expert_bytes = self.model_info.expert_weight_bytes
        
        # Reserve GPU for shared weights (non-negotiable)
        gpu_remaining = self.gpu_total * 0.85 - shared_weights
        
        if gpu_remaining < 0:
            return MemoryAllocation(
                can_fulfill=False,
                rejection_reason=f"Shared weights ({shared_weights/1e9:.2f}GB) "
                                f"exceed GPU capacity ({self.gpu_total/1e9:.2f}GB)"
            )
        
        # Step 2: Calculate total KV cache needed
        kv_cache_per_token = self._calculate_kv_cache_per_token()
        total_kv_needed = (
            kv_cache_per_token 
            * request.max_context_length 
            * request.batch_size
        )
        
        # Step 3: Calculate activation buffer
        activation_bytes = self._estimate_activation_buffer(request)
        
        # Step 4: Allocate expert cache on GPU (what's left after KV)
        # We need to balance: more experts = better throughput, more KV = longer context
        # Strategy: Reserve space for minimum KV cache first
        min_gpu_kv = kv_cache_per_token * request.min_gpu_kv_tokens * request.batch_size
        
        expert_budget = gpu_remaining - min_gpu_kv - activation_bytes - (self.gpu_total * 0.1)
        hot_experts = max(0, int(expert_budget / single_expert_bytes))
        
        # Step 5: Allocate remaining KV cache across tiers
        gpu_kv_budget = gpu_remaining - (hot_experts * single_expert_bytes) - activation_bytes
        
        kv_allocation = self._allocate_kv_across_tiers(
            total_kv_needed=total_kv_needed,
            gpu_kv_budget=gpu_kv_budget,
            cpu_available=self.cpu_total * 0.8,
            ssd_available=self.ssd_total * 0.9,
            kv_per_token=kv_cache_per_token,
            batch_size=request.batch_size,
            allow_cpu=request.allow_cpu_offload,
            allow_ssd=request.allow_ssd_offload
        )
        
        if not kv_allocation["can_fit"]:
            return MemoryAllocation(
                can_fulfill=False,
                rejection_reason=kv_allocation["reason"]
            )
        
        # Step 6: Allocate remaining experts to CPU/SSD
        total_experts = self.model_info.num_layers * self.model_info.num_experts
        remaining_experts = total_experts - hot_experts
        
        cpu_expert_budget = self.cpu_total * 0.8 - kv_allocation["cpu_kv_bytes"]
        warm_experts = min(remaining_experts, int(cpu_expert_budget / single_expert_bytes))
        cold_experts = remaining_experts - warm_experts
        
        # Step 7: Build allocation
        return MemoryAllocation(
            # Weights
            gpu_shared_weights=shared_weights,
            gpu_hot_experts=hot_experts,
            cpu_warm_experts=warm_experts,
            ssd_cold_experts=cold_experts,
            
            # KV cache
            gpu_kv_cache_bytes=kv_allocation["gpu_kv_bytes"],
            gpu_kv_cache_tokens=kv_allocation["gpu_kv_tokens"],
            cpu_kv_cache_bytes=kv_allocation["cpu_kv_bytes"],
            cpu_kv_cache_tokens=kv_allocation["cpu_kv_tokens"],
            ssd_kv_cache_bytes=kv_allocation["ssd_kv_bytes"],
            ssd_kv_cache_tokens=kv_allocation["ssd_kv_tokens"],
            
            # Operational
            activation_buffer=activation_bytes,
            safety_margin=int(self.gpu_total * 0.1),
            
            # Predictions
            expected_gpu_usage=shared_weights + (hot_experts * single_expert_bytes) 
                              + kv_allocation["gpu_kv_bytes"] + activation_bytes,
            expected_cpu_usage=(warm_experts * single_expert_bytes) 
                              + kv_allocation["cpu_kv_bytes"],
            expected_ssd_usage=(cold_experts * single_expert_bytes) 
                              + kv_allocation["ssd_kv_bytes"],
            can_fulfill=True,
            rejection_reason=None
        )
    
    def _calculate_kv_cache_per_token(self) -> int:
        """Calculate bytes needed per token for KV cache."""
        # K and V for each layer
        # Formula: 2 (K+V) * num_layers * hidden_size * 2 (bf16)
        return (
            2  # K and V
            * self.model_info.num_layers
            * self.model_info.hidden_size
            * 2  # bf16
        )
    
    def _allocate_kv_across_tiers(
        self,
        total_kv_needed: int,
        gpu_kv_budget: int,
        cpu_available: int,
        ssd_available: int,
        kv_per_token: int,
        batch_size: int,
        allow_cpu: bool,
        allow_ssd: bool
    ) -> dict:
        """
        Allocate KV cache across GPU/CPU/SSD tiers.
        
        Strategy:
        - GPU: Most recent tokens (highest priority, fastest access)
        - CPU: Older tokens (medium priority, medium speed)
        - SSD: Archive tokens (low priority, slow but unlimited)
        """
        
        # Calculate how many tokens fit in each tier
        gpu_kv_tokens = int(gpu_kv_budget / (kv_per_token * batch_size))
        
        remaining_needed = total_kv_needed - (gpu_kv_tokens * kv_per_token * batch_size)
        
        cpu_kv_tokens = 0
        cpu_kv_bytes = 0
        if remaining_needed > 0 and allow_cpu:
            cpu_kv_tokens = int(min(
                cpu_available / (kv_per_token * batch_size),
                remaining_needed / (kv_per_token * batch_size)
            ))
            cpu_kv_bytes = cpu_kv_tokens * kv_per_token * batch_size
            remaining_needed -= cpu_kv_bytes
        
        ssd_kv_tokens = 0
        ssd_kv_bytes = 0
        if remaining_needed > 0 and allow_ssd:
            ssd_kv_tokens = int(min(
                ssd_available / (kv_per_token * batch_size),
                remaining_needed / (kv_per_token * batch_size)
            ))
            ssd_kv_bytes = ssd_kv_tokens * kv_per_token * batch_size
            remaining_needed -= ssd_kv_bytes
        
        # Check if we could fit everything
        total_allocated_tokens = gpu_kv_tokens + cpu_kv_tokens + ssd_kv_tokens
        total_requested_tokens = int(total_kv_needed / (kv_per_token * batch_size))
        
        can_fit = total_allocated_tokens >= total_requested_tokens
        
        reason = None
        if not can_fit:
            reason = (
                f"Cannot fit {total_requested_tokens} tokens. "
                f"Can only fit {total_allocated_tokens} tokens "
                f"(GPU: {gpu_kv_tokens}, CPU: {cpu_kv_tokens}, SSD: {ssd_kv_tokens}). "
                f"Reduce context length or enable offloading tiers."
            )
        
        return {
            "gpu_kv_bytes": gpu_kv_tokens * kv_per_token * batch_size,
            "gpu_kv_tokens": gpu_kv_tokens,
            "cpu_kv_bytes": cpu_kv_bytes,
            "cpu_kv_tokens": cpu_kv_tokens,
            "ssd_kv_bytes": ssd_kv_bytes,
            "ssd_kv_tokens": ssd_kv_tokens,
            "can_fit": can_fit,
            "reason": reason
        }
    
    def _estimate_activation_buffer(self, request: UserRequest) -> int:
        """Estimate activation memory for forward pass."""
        # Rough estimate: hidden_size * intermediate_size * batch_size * 4
        return (
            self.model_info.hidden_size 
            * self.model_info.intermediate_size 
            * request.batch_size 
            * 4  # safety factor
        )
```

### Example Calculations

#### Scenario 1: 6GB GPU, 2K Context (Conservative)
```python
request = UserRequest(
    max_context_length=2048,
    batch_size=1,
    max_generation_length=512,
    allow_cpu_offload=True,
    allow_ssd_offload=True,
    min_gpu_kv_tokens=512
)

allocation = calculator.calculate_allocation(request)

Result:
├─ GPU (5.1GB usable):
│  ├─ Shared weights: 2.7GB
│  ├─ Hot experts (4): 0.72GB
│  ├─ KV cache (2048 tokens): 0.8GB
│  ├─ Activation buffer: 0.5GB
│  └─ Safety margin: 0.38GB
├─ CPU:
│  ├─ Warm experts (20): 3.6GB
│  └─ KV cache: 0GB (not needed, fits in GPU)
└─ SSD:
   ├─ Cold experts (168): stored as safetensors
   └─ KV cache: 0GB (not needed)

✅ Can fulfill: Yes
```

#### Scenario 2: 6GB GPU, 64K Context (Ambitious)
```python
request = UserRequest(
    max_context_length=65536,  # 64K tokens!
    batch_size=1,
    allow_cpu_offload=True,
    allow_ssd_offload=True,
    min_gpu_kv_tokens=4096  # Keep recent 4K on GPU
)

allocation = calculator.calculate_allocation(request)

Result:
├─ GPU (5.1GB usable):
│  ├─ Shared weights: 2.7GB
│  ├─ Hot experts (2): 0.36GB  # Fewer experts to make room for KV
│  ├─ KV cache (4096 tokens): 0.8GB
│  ├─ Activation buffer: 0.5GB
│  └─ Safety margin: 0.74GB
├─ CPU (32GB):
│  ├─ Warm experts (30): 5.4GB
│  └─ KV cache (32768 tokens): 13GB  # Middle 32K tokens
└─ SSD (500GB):
   ├─ Cold experts (160): 28.8GB (safetensors)
   └─ KV cache (28672 tokens): 11.5GB  # Oldest 28K tokens

✅ Can fulfill: Yes
⚠️  Performance impact: CPU/SSD access will slow down attention
```

#### Scenario 3: 4GB GPU, 1M Context (Extreme)
```python
request = UserRequest(
    max_context_length=1048576,  # 1M tokens!
    batch_size=1,
    allow_cpu_offload=True,
    allow_ssd_offload=True,
    min_gpu_kv_tokens=2048
)

allocation = calculator.calculate_allocation(request)

Result:
├─ GPU (3.4GB usable):
│  ├─ Shared weights: 2.7GB
│  ├─ Hot experts (1): 0.18GB  # Minimal experts
│  ├─ KV cache (2048 tokens): 0.15GB
│  ├─ Activation buffer: 0.2GB
│  └─ Safety margin: 0.17GB
├─ CPU (32GB):
│  ├─ Warm experts (10): 1.8GB
│  └─ KV cache (65536 tokens): 26GB  # ~64K tokens
└─ SSD (1TB):
   ├─ Cold experts (181): 32.6GB
   └─ KV cache (980992 tokens): 392GB  # Remaining ~980K tokens

✅ Can fulfill: Yes
⚠️  Performance: ~1-2 tokens/sec due to SSD access
⚠️  Latency: First token ~30s due to attention over 1M tokens
```

---

## Component 2: Unified Three-Tier Memory Coordinator

### Purpose
Coordinate memory allocation across ALL three tiers (GPU/CPU/SSD) for BOTH expert weights AND KV cache to prevent OOM and enable long context windows on limited hardware.

### Integration Point
**Coordinates**: Both SparseLLM expert cache AND vLLM KV cache tiers simultaneously  
**Extends**: `sparse_llm.integrations.vllm_bridge.py` (existing vLLM integration)  
**Uses**: `DynamicMemoryBudgetCalculator` output to set coordinated parameters  
**Pattern**: Unified orchestrator that calculates allocation for both systems before initializing either

### Design Principles

**CRITICAL INSIGHT**: Both expert weights and KV cache need multi-tier placement on limited hardware:
- Expert weights: Managed by SparseLLM (`LayerAwareCache` for GPU/CPU, safetensors for SSD)
- KV cache: Managed by vLLM (PagedAttention for GPU, `--cpu-offload-gb` for CPU, LMCache for disk)
- **Coordination needed**: Both compete for the SAME GPU/CPU/SSD resources

**The Solution**: Pre-calculate unified allocation across all tiers for both systems, then configure each with non-conflicting parameters.

### vLLM's Native Multi-Tier KV Cache Support

vLLM already provides three-tier KV cache management:

1. **GPU Tier** (via PagedAttention):
   - Automatic block-based memory management
   - Fastest access (0.1-0.5ms latency)
   - User controls via `--gpu-memory-utilization`

2. **CPU Tier** (via `--cpu-offload-gb`):
   - Automatically offloads least-recently-used KV blocks from GPU to CPU when GPU cache is full
   - Medium speed access (1-5ms latency)
   - User controls via `--cpu-offload-gb N` (reserve N GB of CPU for KV overflow)

3. **Disk Tier** (via LMCache):
   - Archives oldest KV cache blocks to disk when CPU cache is full
   - Slow access (10-100ms latency) but unlimited capacity
   - User controls via `--enable-lmcache`

**Our Integration**: Calculate the memory budgets for each tier that account for BOTH expert weights AND KV cache needs.

### Unified Allocation Algorithm

```
Input:
  - model_info (shared weights, expert count/size, num_layers, hidden_size)
  - user_request (max_model_len, dtype, quantization)
  - resource_budget (GPU, CPU, SSD usable bytes with safety margins)

Algorithm:
  1. Calculate KV cache size needed for max_model_len:
     kv_bytes_total = 2 (K+V) × num_layers × hidden_size × max_model_len × dtype_bytes
  
  2. Calculate expert weight total size:
     expert_bytes_total = num_experts × expert_size
  
  3. GPU Tier Allocation:
     a. Reserve for shared weights (must be GPU): shared_weight_bytes
     b. Reserve for activations: activation_buffer_bytes
     c. Remaining GPU budget: gpu_remaining = gpu_usable - shared_weights - activations
     d. Split remaining between hot experts and GPU KV cache:
        - Strategy 1 (balanced): 50/50 split
        - Strategy 2 (weight-heavy): 70% experts, 30% KV (for MoE models)
        - Strategy 3 (KV-heavy): 30% experts, 70% KV (for long context)
        - Use user preference or auto-detect based on max_model_len
     e. gpu_expert_budget = gpu_remaining × expert_ratio
     f. gpu_kv_budget = gpu_remaining × kv_ratio
     g. num_hot_experts = floor(gpu_expert_budget / expert_size)
     h. gpu_kv_tokens = floor(gpu_kv_budget / kv_bytes_per_token)
  
  4. CPU Tier Allocation:
     a. Remaining experts after GPU: remaining_experts = total_experts - num_hot_experts
     b. Remaining KV tokens after GPU: remaining_kv_tokens = max_model_len - gpu_kv_tokens
     c. Split CPU between warm experts and CPU KV cache:
        cpu_expert_budget = min(remaining_experts × expert_size, cpu_usable × 0.4)
        cpu_kv_budget = cpu_usable - cpu_expert_budget
     d. num_warm_experts = floor(cpu_expert_budget / expert_size)
     e. cpu_kv_tokens = floor(cpu_kv_budget / kv_bytes_per_token)
  
  5. SSD Tier Allocation:
     a. Remaining experts after CPU: num_cold_experts = remaining_experts - num_warm_experts
     b. Remaining KV tokens after CPU: ssd_kv_tokens = max_model_len - gpu_kv_tokens - cpu_kv_tokens
     c. Validate SSD capacity:
        ssd_needed = (num_cold_experts × expert_size) + (ssd_kv_tokens × kv_bytes_per_token)
        if ssd_needed > ssd_usable: REJECT (insufficient storage)
  
  6. Calculate vLLM parameters:
     a. gpu_memory_utilization = gpu_kv_budget / gpu_total_memory
     b. cpu_offload_gb = cpu_kv_budget / (1024^3)
     c. enable_lmcache = (ssd_kv_tokens > 0)
  
  7. Return unified allocation with parameters for both systems

Output:
  - Expert weight allocation: num_hot_experts, num_warm_experts, num_cold_experts
  - KV cache allocation: gpu_kv_tokens, cpu_kv_tokens, ssd_kv_tokens
  - vLLM parameters: gpu_memory_utilization, cpu_offload_gb, enable_lmcache
  - Validation: can_fulfill (boolean), rejection_reason (if false)
```

### Tier Coordination Example (6GB GPU, 128K context)

```
Resources Available:
  GPU: 6GB (5.1GB usable)
  CPU: 32GB (25.6GB usable)
  SSD: 500GB (450GB usable)

Model: Mixtral-8x7B (192 experts, 180MB each)
Request: max_model_len=131072 (128K tokens)

Step 1: Calculate totals
  - KV cache total: 2×24×2048×131072×2 = 25.6GB
  - Expert weights total: 192×180MB = 34.5GB
  - Shared weights: 2.7GB

Step 2: GPU Allocation (5.1GB)
  - Shared weights: 2.7GB (fixed)
  - Activations: 0.5GB (fixed)
  - Remaining: 1.9GB
  - Split 40/60 (experts/KV for long context):
    - GPU experts: 0.76GB → 4 experts
    - GPU KV cache: 1.14GB → ~6K tokens

Step 3: CPU Allocation (25.6GB)
  - Split 20/80 (experts/KV):
    - CPU experts: 5.1GB → 28 experts
    - CPU KV cache: 20.5GB → ~110K tokens

Step 4: SSD Allocation (450GB)
  - Cold experts: 160 experts × 180MB = 28.8GB
  - SSD KV cache: ~15K tokens × 196KB = 2.9GB
  - Total: 31.7GB → Fits!

Step 5: vLLM parameters
  - gpu_memory_utilization = 1.14GB / 6GB = 0.19
  - cpu_offload_gb = 20
  - enable_lmcache = true

Result: ✅ Can support 128K context on 6GB GPU with unified three-tier coordination!
```

### Key Coordination Points

1. **No Double Allocation**: Each memory region allocated to exactly one purpose (expert weights OR KV cache, never both)
2. **Tier-Aware Split**: Each tier splits its budget between experts and KV based on workload characteristics
3. **Cascading Fallback**: Both systems use GPU → CPU → SSD cascade when their tier fills up
4. **Independent Management**: After allocation, SparseLLM and vLLM manage their respective tiers independently
5. **Observable Metrics**: Both systems report per-tier usage for unified monitoring

---

```python
class VLLMMemoryCoordinator:
    """
    Coordinates memory between expert weights and vLLM KV cache.
    
    Prevents OOM by:
    1. Pre-calculating expert weight placement
    2. Calculating vLLM's --gpu-memory-utilization dynamically
    3. Passing coordinated parameters to vLLM initialization
    """
    
    def __init__(
        self,
        model_id: str,
        user_vllm_params: Dict[str, Any]  # User's original vLLM parameters
    ):
        self.model_id = model_id
        self.user_vllm_params = user_vllm_params
        
    def initialize_with_coordination(self) -> Tuple[LLM, MemoryAllocation]:
        """
        Initialize vLLM with coordinated memory allocation.
        
        Returns:
            (vllm_engine, memory_allocation)
        """
        
        # Step 1: Profile resources (reuse existing ResourceProfiler)
        profiler = ResourceProfiler()
        resource_budget = profiler.profile()
        
        # Step 2: Introspect model (reuse existing ModelIntrospector)
        introspector = ModelIntrospector()
        model_info = introspector.introspect(self.model_id)
        
        # Step 3: Create user request from vLLM params
        user_request = UserRequest(
            max_model_len=self.user_vllm_params.get("max_model_len", 8192),
            dtype=self.user_vllm_params.get("dtype", "auto"),
            quantization=self.user_vllm_params.get("quantization", None),
            tensor_parallel_size=self.user_vllm_params.get("tensor_parallel_size", 1),
            allow_cpu_offload=True,  # Default to enabled
            allow_ssd_offload=True
        )
        
        # Step 4: Calculate allocation
        calculator = DynamicMemoryBudgetCalculator(resource_budget, model_info)
        allocation = calculator.calculate_allocation(user_request)
        
        # Step 5: Check if can fulfill
        if not allocation.can_fulfill:
            raise RuntimeError(
                f"Cannot initialize model: {allocation.rejection_reason}"
            )
        
        # Step 6: Load expert weights using existing FourPhaseOrchestrator
        orchestrator = FourPhaseOrchestrator()
        weight_state = orchestrator.initialize(
            model_id=self.model_id,
            placement_plan=self._create_placement_plan(allocation)
        )
        
        # Step 7: Initialize vLLM with coordinated parameters
        vllm_params = {
            **self.user_vllm_params,  # Preserve all user parameters
            "gpu_memory_utilization": allocation.vllm_gpu_memory_utilization,  # Override this one
            "model": self.model_id,
            "dtype": user_request.dtype,
            "quantization": user_request.quantization,
            "max_model_len": user_request.max_model_len,
            "tensor_parallel_size": user_request.tensor_parallel_size,
        }
        
        logger.info(f"Initializing vLLM with gpu_memory_utilization={allocation.vllm_gpu_memory_utilization:.3f}")
        logger.info(f"Expert allocation: {allocation.gpu_hot_experts} GPU, {allocation.cpu_warm_experts} CPU, {allocation.ssd_cold_experts} SSD")
        
        vllm_engine = LLM(**vllm_params)
        
        return vllm_engine, allocation
```

### Key Changes from Current Implementation

**Before (causes OOM)**:
```python
# Current vllm_bridge.py
def initialize_vllm(model_id, gpu_memory_utilization=0.9):
    # Load expert weights first (occupies ~50% GPU)
    load_expert_weights(model_id)
    
    # Try to initialize vLLM with 90% GPU usage
    # But 50% already used! → OOM
    llm = LLM(model=model_id, gpu_memory_utilization=0.9)
```

**After (prevents OOM)**:
```python
# New coordinated approach
def initialize_vllm(model_id, user_params):
    # Calculate FIRST (don't load anything yet)
    allocation = calculate_allocation(model_id, user_params)
    
    # Load expert weights with calculated budget
    load_expert_weights(model_id, allocation)
    
    # Initialize vLLM with calculated parameter (e.g., 0.2)
    # Only uses remaining GPU space → Success!
    llm = LLM(model=model_id, gpu_memory_utilization=allocation.vllm_gpu_memory_utilization)
```

---

### Design

```python
class HierarchicalKVCache:
    """Three-tier KV cache with automatic tier management."""
    
    def __init__(
        self,
        allocation: MemoryAllocation,
        model_info: ModelInfo
    ):
        self.allocation = allocation
        self.model_info = model_info
        
        # Initialize tier managers
        self.gpu_cache = GPUKVCache(
            capacity_tokens=allocation.gpu_kv_cache_tokens,
            num_layers=model_info.num_layers,
            hidden_size=model_info.hidden_size
        )
        
        self.cpu_cache = CPUKVCache(
            capacity_tokens=allocation.cpu_kv_cache_tokens,
            num_layers=model_info.num_layers,
            hidden_size=model_info.hidden_size
        )
        
        self.ssd_cache = SSDKVCache(
            capacity_tokens=allocation.ssd_kv_cache_tokens,
            num_layers=model_info.num_layers,
            hidden_size=model_info.hidden_size,
            storage_path=Path.home() / ".cache" / "sparse_llm" / "kv_cache"
        )
        
        # Access tracking for promotion decisions
        self.access_tracker = AccessTracker()
        
        # Statistics
        self.stats = {
            "gpu_hits": 0,
            "cpu_hits": 0,
            "ssd_hits": 0,
            "promotions": 0,
            "demotions": 0,
            "total_accesses": 0
        }
    
    def get_kv_for_positions(
        self, 
        seq_id: int,
        positions: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get K and V tensors for specified positions.
        
        Returns:
            (K, V) tensors on GPU (promoted if needed)
        """
        self.stats["total_accesses"] += 1
        
        # Check each tier in order: GPU → CPU → SSD
        result_k_parts = []
        result_v_parts = []
        positions_to_fetch = set(positions)
        
        # Try GPU first (fastest)
        if self.gpu_cache.has_positions(seq_id, positions):
            k, v = self.gpu_cache.get(seq_id, positions)
            self.stats["gpu_hits"] += 1
            return k, v
        
        # Try CPU second
        cpu_positions = self.cpu_cache.get_available_positions(seq_id, positions)
        if cpu_positions:
            k_cpu, v_cpu = self.cpu_cache.get(seq_id, cpu_positions)
            result_k_parts.append(k_cpu)
            result_v_parts.append(v_cpu)
            positions_to_fetch -= set(cpu_positions)
            self.stats["cpu_hits"] += 1
            
            # Promote hot positions to GPU
            if self._should_promote(cpu_positions):
                self._promote_to_gpu(seq_id, cpu_positions, k_cpu, v_cpu)
        
        # Try SSD last (slowest)
        if positions_to_fetch:
            ssd_positions = list(positions_to_fetch)
            k_ssd, v_ssd = self.ssd_cache.get(seq_id, ssd_positions)
            result_k_parts.append(k_ssd)
            result_v_parts.append(v_ssd)
            self.stats["ssd_hits"] += 1
            
            # Promote to CPU for future access
            self._promote_to_cpu(seq_id, ssd_positions, k_ssd, v_ssd)
        
        # Concatenate results and ensure on GPU
        k = torch.cat(result_k_parts, dim=0).cuda()
        v = torch.cat(result_v_parts, dim=0).cuda()
        
        return k, v
    
    def append_kv(
        self,
        seq_id: int,
        position: int,
        k: torch.Tensor,
        v: torch.Tensor
    ):
        """
        Append new K/V pair for a position.
        
        Strategy:
        - Always write to GPU first (for immediate use)
        - Evict oldest from GPU to CPU when full
        - Evict oldest from CPU to SSD when full
        """
        # Write to GPU
        if self.gpu_cache.is_full():
            # Evict oldest to CPU
            evicted_pos, evicted_k, evicted_v = self.gpu_cache.evict_oldest(seq_id)
            self.cpu_cache.append(seq_id, evicted_pos, evicted_k, evicted_v)
            self.stats["demotions"] += 1
        
        self.gpu_cache.append(seq_id, position, k, v)
        
        # If CPU is full, evict to SSD
        if self.cpu_cache.is_full():
            evicted_pos, evicted_k, evicted_v = self.cpu_cache.evict_oldest(seq_id)
            self.ssd_cache.append(seq_id, evicted_pos, evicted_k, evicted_v)
            self.stats["demotions"] += 1
    
    def _should_promote(self, positions: list[int]) -> bool:
        """Decide if positions should be promoted to GPU."""
        # Promote if accessed multiple times recently
        access_count = self.access_tracker.get_access_count(positions)
        return access_count >= 2
    
    def _promote_to_gpu(
        self, 
        seq_id: int, 
        positions: list[int],
        k: torch.Tensor,
        v: torch.Tensor
    ):
        """Promote positions from CPU to GPU."""
        for i, pos in enumerate(positions):
            if not self.gpu_cache.is_full():
                self.gpu_cache.append(seq_id, pos, k[i], v[i])
                self.cpu_cache.remove(seq_id, pos)
                self.stats["promotions"] += 1
    
    def _promote_to_cpu(
        self,
        seq_id: int,
        positions: list[int],
        k: torch.Tensor,
        v: torch.Tensor
    ):
        """Promote positions from SSD to CPU."""
        for i, pos in enumerate(positions):
            if not self.cpu_cache.is_full():
                self.cpu_cache.append(seq_id, pos, k[i], v[i])
                self.stats["promotions"] += 1


class GPUKVCache:
    """GPU-resident KV cache with FIFO eviction."""
    
    def __init__(self, capacity_tokens: int, num_layers: int, hidden_size: int):
        self.capacity_tokens = capacity_tokens
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Pre-allocate tensors (avoids fragmentation)
        self.k_cache = torch.zeros(
            (capacity_tokens, num_layers, hidden_size),
            dtype=torch.bfloat16,
            device="cuda"
        )
        self.v_cache = torch.zeros(
            (capacity_tokens, num_layers, hidden_size),
            dtype=torch.bfloat16,
            device="cuda"
        )
        
        # Metadata: position → cache_slot mapping
        self.position_to_slot: dict[tuple[int, int], int] = {}  # (seq_id, pos) → slot
        self.slot_to_position: dict[int, tuple[int, int]] = {}  # slot → (seq_id, pos)
        self.next_slot = 0
        self.num_slots_used = 0
    
    def append(self, seq_id: int, position: int, k: torch.Tensor, v: torch.Tensor):
        """Append KV to cache."""
        slot = self.next_slot
        self.k_cache[slot] = k
        self.v_cache[slot] = v
        self.position_to_slot[(seq_id, position)] = slot
        self.slot_to_position[slot] = (seq_id, position)
        
        self.next_slot = (self.next_slot + 1) % self.capacity_tokens
        self.num_slots_used = min(self.num_slots_used + 1, self.capacity_tokens)
    
    def get(self, seq_id: int, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Get KV for positions."""
        slots = [self.position_to_slot[(seq_id, pos)] for pos in positions]
        return self.k_cache[slots], self.v_cache[slots]
    
    def has_positions(self, seq_id: int, positions: list[int]) -> bool:
        """Check if all positions are in cache."""
        return all((seq_id, pos) in self.position_to_slot for pos in positions)
    
    def is_full(self) -> bool:
        return self.num_slots_used >= self.capacity_tokens
    
    def evict_oldest(self, seq_id: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Evict oldest entry (FIFO)."""
        # Find oldest slot (first in circular buffer)
        oldest_slot = (self.next_slot - self.num_slots_used) % self.capacity_tokens
        seq_id, position = self.slot_to_position[oldest_slot]
        
        k = self.k_cache[oldest_slot].clone()
        v = self.v_cache[oldest_slot].clone()
        
        del self.position_to_slot[(seq_id, position)]
        del self.slot_to_position[oldest_slot]
        self.num_slots_used -= 1
        
        return position, k, v


class CPUKVCache:
    """CPU-resident KV cache (same interface as GPU)."""
    
    def __init__(self, capacity_tokens: int, num_layers: int, hidden_size: int):
        self.capacity_tokens = capacity_tokens
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Store on CPU
        self.cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.insertion_order: list[tuple[int, int]] = []
    
    def append(self, seq_id: int, position: int, k: torch.Tensor, v: torch.Tensor):
        key = (seq_id, position)
        self.cache[key] = (k.cpu(), v.cpu())
        self.insertion_order.append(key)
    
    def get(self, seq_id: int, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        keys = [(seq_id, pos) for pos in positions]
        k_list = [self.cache[key][0] for key in keys]
        v_list = [self.cache[key][1] for key in keys]
        return torch.stack(k_list), torch.stack(v_list)
    
    def get_available_positions(self, seq_id: int, positions: list[int]) -> list[int]:
        return [pos for pos in positions if (seq_id, pos) in self.cache]
    
    def is_full(self) -> bool:
        return len(self.cache) >= self.capacity_tokens
    
    def evict_oldest(self, seq_id: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        oldest_key = self.insertion_order.pop(0)
        k, v = self.cache.pop(oldest_key)
        return oldest_key[1], k, v
    
    def remove(self, seq_id: int, position: int):
        key = (seq_id, position)
        if key in self.cache:
            del self.cache[key]
            self.insertion_order.remove(key)


class SSDKVCache:
    """SSD-backed KV cache using memory-mapped files."""
    
    def __init__(
        self, 
        capacity_tokens: int, 
        num_layers: int, 
        hidden_size: int,
        storage_path: Path
    ):
        self.capacity_tokens = capacity_tokens
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.storage_path = storage_path
        
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Use safetensors for efficient serialization
        self.cache_files: dict[int, Path] = {}  # seq_id → cache file
        self.metadata: dict[tuple[int, int], dict] = {}  # (seq_id, pos) → offset info
    
    def append(self, seq_id: int, position: int, k: torch.Tensor, v: torch.Tensor):
        """Append to SSD (batched writes for efficiency)."""
        cache_file = self._get_cache_file(seq_id)
        
        # Serialize KV pair
        tensors = {
            f"k_{position}": k.cpu(),
            f"v_{position}": v.cpu()
        }
        
        from safetensors.torch import save_file
        save_file(tensors, cache_file, metadata={"position": str(position)})
        
        self.metadata[(seq_id, position)] = {
            "file": cache_file,
            "position": position
        }
    
    def get(self, seq_id: int, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Load from SSD."""
        cache_file = self._get_cache_file(seq_id)
        
        from safetensors.torch import load_file
        tensors = load_file(cache_file)
        
        k_list = [tensors[f"k_{pos}"] for pos in positions]
        v_list = [tensors[f"v_{pos}"] for pos in positions]
        
        return torch.stack(k_list), torch.stack(v_list)
    
    def _get_cache_file(self, seq_id: int) -> Path:
        if seq_id not in self.cache_files:
            self.cache_files[seq_id] = self.storage_path / f"seq_{seq_id}_kv.safetensors"
        return self.cache_files[seq_id]
```

---

## Component 3: Memory Pressure Monitor

### Purpose
Real-time monitoring and predictive OOM prevention.

### Design

```python
class MemoryPressureMonitor:
    """Real-time memory monitoring with predictive OOM prevention."""
    
    def __init__(
        self,
        gpu_capacity: int,
        cpu_capacity: int,
        ssd_capacity: int,
        warning_threshold: float = 0.85,
        critical_threshold: float = 0.95
    ):
        self.gpu_capacity = gpu_capacity
        self.cpu_capacity = cpu_capacity
        self.ssd_capacity = ssd_capacity
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        # Historical data
        self.gpu_history: list[float] = []
        self.cpu_history: list[float] = []
        self.trend_window = 10
        
        # Alerts
        self.alerts: list[str] = []
    
    def check_pressure(self) -> dict[str, Any]:
        """Check current memory pressure across all tiers."""
        gpu_used = torch.cuda.memory_allocated()
        gpu_utilization = gpu_used / self.gpu_capacity
        
        cpu_used = self._get_process_memory()
        cpu_utilization = cpu_used / self.cpu_capacity
        
        # Track history
        self.gpu_history.append(gpu_utilization)
        self.cpu_history.append(cpu_utilization)
        
        # Keep only recent history
        if len(self.gpu_history) > self.trend_window:
            self.gpu_history.pop(0)
            self.cpu_history.pop(0)
        
        # Detect trends
        gpu_trend = self._calculate_trend(self.gpu_history)
        cpu_trend = self._calculate_trend(self.cpu_history)
        
        # Generate alerts
        self.alerts = []
        
        if gpu_utilization > self.critical_threshold:
            self.alerts.append(
                f"CRITICAL: GPU memory at {gpu_utilization:.1%}. "
                f"Automatic offloading will trigger."
            )
        elif gpu_utilization > self.warning_threshold:
            self.alerts.append(
                f"WARNING: GPU memory at {gpu_utilization:.1%}. "
                f"Consider reducing batch size or context length."
            )
        
        if cpu_utilization > self.critical_threshold:
            self.alerts.append(
                f"CRITICAL: CPU memory at {cpu_utilization:.1%}. "
                f"May experience swapping."
            )
        
        # Predictive: if trend suggests OOM in next N steps
        if gpu_trend > 0.05:  # Growing fast
            steps_to_oom = (1.0 - gpu_utilization) / gpu_trend
            if steps_to_oom < 5:
                self.alerts.append(
                    f"PREDICTIVE: GPU OOM predicted in {steps_to_oom:.0f} steps. "
                    f"Triggering preventive offloading."
                )
        
        return {
            "gpu_utilization": gpu_utilization,
            "cpu_utilization": cpu_utilization,
            "gpu_trend": gpu_trend,
            "cpu_trend": cpu_trend,
            "pressure_level": self._get_pressure_level(gpu_utilization),
            "alerts": self.alerts,
            "should_offload": gpu_utilization > self.warning_threshold,
            "should_reject": gpu_utilization > self.critical_threshold
        }
    
    def _calculate_trend(self, history: list[float]) -> float:
        """Calculate linear trend (slope) of utilization."""
        if len(history) < 2:
            return 0.0
        
        # Simple linear regression
        n = len(history)
        x_mean = (n - 1) / 2
        y_mean = sum(history) / n
        
        numerator = sum((i - x_mean) * (history[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _get_pressure_level(self, utilization: float) -> str:
        if utilization < self.warning_threshold:
            return "normal"
        elif utilization < self.critical_threshold:
            return "warning"
        else:
            return "critical"
    
    def _get_process_memory(self) -> int:
        """Get current process memory usage."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss
    
    def predict_request_feasibility(
        self,
        request_memory: int,
        tier: str = "gpu"
    ) -> bool:
        """
        Predict if a new request can be admitted without OOM.
        
        Args:
            request_memory: Bytes needed for request
            tier: Which tier to check ("gpu", "cpu", "ssd")
        
        Returns:
            True if request can be safely admitted
        """
        if tier == "gpu":
            current_used = torch.cuda.memory_allocated()
            would_be_used = current_used + request_memory
            utilization = would_be_used / self.gpu_capacity
            return utilization < self.critical_threshold
        elif tier == "cpu":
            current_used = self._get_process_memory()
            would_be_used = current_used + request_memory
            utilization = would_be_used / self.cpu_capacity
            return utilization < self.critical_threshold
        else:  # SSD
            return True  # Assume SSD always has space (validated at calculation time)
```

---

## Component 4: Adaptive Request Handler

### Purpose
Per-request admission control and dynamic resource allocation.

### Design

```python
class AdaptiveRequestHandler:
    """
    Handle incoming requests with dynamic memory allocation.
    Guarantees no OOM by rejecting requests that can't fit.
    """
    
    def __init__(
        self,
        calculator: DynamicMemoryBudgetCalculator,
        monitor: MemoryPressureMonitor,
        kv_cache_manager: HierarchicalKVCache
    ):
        self.calculator = calculator
        self.monitor = monitor
        self.kv_cache_manager = kv_cache_manager
        
        # Active requests
        self.active_requests: dict[str, ActiveRequest] = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "accepted_requests": 0,
            "rejected_requests": 0,
            "rejection_reasons": {}
        }
    
    async def handle_request(
        self,
        request_id: str,
        user_request: UserRequest
    ) -> RequestResult:
        """
        Handle inference request with admission control.
        
        Returns:
            RequestResult with status and allocation
        """
        self.stats["total_requests"] += 1
        
        # Step 1: Check memory pressure
        pressure = self.monitor.check_pressure()
        
        if pressure["should_reject"]:
            self.stats["rejected_requests"] += 1
            reason = "System under critical memory pressure"
            self._record_rejection(reason)
            return RequestResult(
                request_id=request_id,
                status="rejected",
                reason=reason,
                suggestions=[
                    "Wait for current requests to complete",
                    "Reduce context length",
                    "Reduce batch size"
                ]
            )
        
        # Step 2: Calculate allocation for this request
        allocation = self.calculator.calculate_allocation(user_request)
        
        if not allocation.can_fulfill:
            self.stats["rejected_requests"] += 1
            self._record_rejection(allocation.rejection_reason)
            return RequestResult(
                request_id=request_id,
                status="rejected",
                reason=allocation.rejection_reason,
                suggestions=self._generate_suggestions(user_request, allocation)
            )
        
        # Step 3: Predictive check - will this cause OOM?
        if not self.monitor.predict_request_feasibility(
            request_memory=allocation.expected_gpu_usage,
            tier="gpu"
        ):
            self.stats["rejected_requests"] += 1
            reason = "Request would cause GPU OOM"
            self._record_rejection(reason)
            return RequestResult(
                request_id=request_id,
                status="rejected",
                reason=reason,
                suggestions=[
                    f"Reduce context from {user_request.max_context_length} to "
                    f"{user_request.max_context_length // 2}",
                    "Enable CPU offloading" if not user_request.allow_cpu_offload else None,
                    "Wait for memory to free up"
                ]
            )
        
        # Step 4: Accept request and allocate resources
        self.active_requests[request_id] = ActiveRequest(
            request_id=request_id,
            user_request=user_request,
            allocation=allocation,
            start_time=time.time()
        )
        
        self.stats["accepted_requests"] += 1
        
        return RequestResult(
            request_id=request_id,
            status="accepted",
            allocation=allocation,
            estimated_performance=self._estimate_performance(allocation)
        )
    
    def _generate_suggestions(
        self,
        user_request: UserRequest,
        allocation: MemoryAllocation
    ) -> list[str]:
        """Generate helpful suggestions for rejected requests."""
        suggestions = []
        
        # If context length is the issue
        if "context" in (allocation.rejection_reason or "").lower():
            # Binary search for max feasible context
            max_feasible = self._find_max_feasible_context(user_request)
            suggestions.append(
                f"Reduce context length from {user_request.max_context_length} "
                f"to {max_feasible} or less"
            )
        
        # If offloading is disabled
        if not user_request.allow_cpu_offload:
            suggestions.append("Enable CPU offloading with allow_cpu_offload=True")
        
        if not user_request.allow_ssd_offload and user_request.max_context_length > 32768:
            suggestions.append(
                "Enable SSD offloading for long contexts (allow_ssd_offload=True)"
            )
        
        # If batch size is large
        if user_request.batch_size > 1:
            suggestions.append(f"Reduce batch size from {user_request.batch_size} to 1")
        
        return [s for s in suggestions if s]
    
    def _find_max_feasible_context(self, user_request: UserRequest) -> int:
        """Binary search for maximum feasible context length."""
        low = 512
        high = user_request.max_context_length
        max_feasible = low
        
        while low <= high:
            mid = (low + high) // 2
            test_request = UserRequest(
                max_context_length=mid,
                batch_size=user_request.batch_size,
                allow_cpu_offload=user_request.allow_cpu_offload,
                allow_ssd_offload=user_request.allow_ssd_offload
            )
            
            allocation = self.calculator.calculate_allocation(test_request)
            
            if allocation.can_fulfill:
                max_feasible = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return max_feasible
    
    def _estimate_performance(self, allocation: MemoryAllocation) -> dict:
        """Estimate performance metrics for allocation."""
        # Calculate expected latency based on tier distribution
        gpu_fraction = allocation.gpu_kv_cache_tokens / (
            allocation.gpu_kv_cache_tokens 
            + allocation.cpu_kv_cache_tokens 
            + allocation.ssd_kv_cache_tokens
        )
        
        cpu_fraction = allocation.cpu_kv_cache_tokens / (
            allocation.gpu_kv_cache_tokens 
            + allocation.cpu_kv_cache_tokens 
            + allocation.ssd_kv_cache_tokens
        )
        
        ssd_fraction = 1 - gpu_fraction - cpu_fraction
        
        # Estimate tokens/sec
        # GPU: 100 tok/s, CPU: 20 tok/s, SSD: 2 tok/s (rough estimates)
        weighted_throughput = (
            gpu_fraction * 100 
            + cpu_fraction * 20 
            + ssd_fraction * 2
        )
        
        # Estimate first token latency (prefill)
        total_tokens = (
            allocation.gpu_kv_cache_tokens 
            + allocation.cpu_kv_cache_tokens 
            + allocation.ssd_kv_cache_tokens
        )
        
        # Prefill scales with context length
        prefill_latency = (
            gpu_fraction * (total_tokens / 1000) * 0.1  # 0.1s per 1K tokens on GPU
            + cpu_fraction * (total_tokens / 1000) * 0.5  # 0.5s per 1K tokens on CPU
            + ssd_fraction * (total_tokens / 1000) * 2.0  # 2s per 1K tokens on SSD
        )
        
        return {
            "estimated_throughput_tokens_per_sec": weighted_throughput,
            "estimated_first_token_latency_sec": prefill_latency,
            "tier_distribution": {
                "gpu_percent": gpu_fraction * 100,
                "cpu_percent": cpu_fraction * 100,
                "ssd_percent": ssd_fraction * 100
            }
        }
    
    def _record_rejection(self, reason: str):
        """Track rejection reasons for diagnostics."""
        if reason not in self.stats["rejection_reasons"]:
            self.stats["rejection_reasons"][reason] = 0
        self.stats["rejection_reasons"][reason] += 1


@dataclass
class ActiveRequest:
    """Track active request state."""
    request_id: str
    user_request: UserRequest
    allocation: MemoryAllocation
    start_time: float
    tokens_generated: int = 0


@dataclass
class RequestResult:
    """Result of request handling."""
    request_id: str
    status: str  # "accepted" or "rejected"
    reason: str | None = None
    allocation: MemoryAllocation | None = None
    estimated_performance: dict | None = None
    suggestions: list[str] = None
```

---

## Integration with Existing System

### Modified Initialization Flow

```python
class UnifiedServer:
    """Updated server with adaptive memory synchronization."""
    
    async def startup(self):
        logger.info("="*70)
        logger.info("ADAPTIVE MEMORY SYNCHRONIZATION INITIALIZATION")
        logger.info("="*70)
        
        # Phase 0: Profile hardware
        profiler = ResourceProfiler()
        budget = profiler.profile()
        
        logger.info(f"Hardware detected:")
        logger.info(f"  GPU: {budget.total_gpu_bytes / 1e9:.2f}GB")
        logger.info(f"  CPU: {budget.total_cpu_bytes / 1e9:.2f}GB")
        logger.info(f"  SSD: {budget.storage.available_bytes / 1e9:.2f}GB")
        
        # Phase 1: Create adaptive calculator (NO LOADING YET)
        introspector = ModelIntrospector()
        model_info = introspector.introspect(self.model)
        
        self.calculator = DynamicMemoryBudgetCalculator(
            gpu_total_bytes=budget.total_gpu_bytes,
            cpu_total_bytes=budget.total_cpu_bytes,
            ssd_total_bytes=budget.storage.available_bytes,
            model_info=model_info
        )
        
        logger.info(f"Model: {model_info.model_id}")
        logger.info(f"  Type: {'MoE' if model_info.is_moe else 'Dense'}")
        logger.info(f"  Layers: {model_info.num_layers}")
        if model_info.is_moe:
            logger.info(f"  Experts: {model_info.num_experts} per layer")
        
        # Phase 2: Calculate allocation for DEFAULT request
        default_request = UserRequest(
            max_context_length=self.default_context_length or 2048,
            batch_size=1,
            allow_cpu_offload=True,
            allow_ssd_offload=True
        )
        
        default_allocation = self.calculator.calculate_allocation(default_request)
        
        if not default_allocation.can_fulfill:
            raise ValueError(
                f"Cannot fulfill default configuration: {default_allocation.rejection_reason}"
            )
        
        logger.info(f"Default allocation calculated:")
        logger.info(f"  Context length: {default_request.max_context_length} tokens")
        logger.info(f"  GPU KV cache: {default_allocation.gpu_kv_cache_tokens} tokens")
        logger.info(f"  CPU KV cache: {default_allocation.cpu_kv_cache_tokens} tokens")
        logger.info(f"  SSD KV cache: {default_allocation.ssd_kv_cache_tokens} tokens")
        logger.info(f"  Hot experts: {default_allocation.gpu_hot_experts}")
        
        # Phase 3: Load model weights according to allocation
        orchestrator = FourPhaseOrchestrator()
        self.loaded_state = orchestrator.initialize(
            model_id=self.model,
            storage_path=self.storage_path,
            max_hot_experts=default_allocation.gpu_hot_experts,  # Constrained by allocation
            progress_callback=logger.info
        )
        
        # Phase 4: Initialize KV cache manager
        self.kv_cache_manager = HierarchicalKVCache(
            allocation=default_allocation,
            model_info=model_info
        )
        
        # Phase 5: Initialize memory monitor
        self.monitor = MemoryPressureMonitor(
            gpu_capacity=budget.total_gpu_bytes,
            cpu_capacity=budget.total_cpu_bytes,
            ssd_capacity=budget.storage.available_bytes
        )
        
        # Phase 6: Initialize request handler
        self.request_handler = AdaptiveRequestHandler(
            calculator=self.calculator,
            monitor=self.monitor,
            kv_cache_manager=self.kv_cache_manager
        )
        
        # Phase 7: Initialize vLLM engine (if using)
        if self.use_vllm:
            await self._initialize_vllm_with_allocation(default_allocation)
        
        logger.info("✅ Adaptive memory synchronization ready")
    
    async def generate_completion(
        self,
        prompt: str,
        max_context_length: int | None = None,
        max_tokens: int = 512,
        batch_size: int = 1
    ) -> dict:
        """Generate completion with dynamic resource allocation."""
        
        # Build user request
        user_request = UserRequest(
            max_context_length=max_context_length or self.default_context_length,
            batch_size=batch_size,
            max_generation_length=max_tokens,
            allow_cpu_offload=True,
            allow_ssd_offload=True
        )
        
        # Handle request with admission control
        result = await self.request_handler.handle_request(
            request_id=str(uuid.uuid4()),
            user_request=user_request
        )
        
        if result.status == "rejected":
            raise HTTPException(
                status_code=429,  # Too Many Requests
                detail={
                    "error": "Request rejected due to insufficient resources",
                    "reason": result.reason,
                    "suggestions": result.suggestions
                }
            )
        
        # Proceed with generation
        # ... (existing generation logic)
        
        return {
            "completion": generated_text,
            "allocation": result.allocation,
            "performance": result.estimated_performance
        }
```

### API Endpoints

```python
@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible completions with dynamic allocation."""
    
    result = await server.generate_completion(
        prompt=request.prompt,
        max_context_length=request.max_context_length,
        max_tokens=request.max_tokens
    )
    
    return {
        "id": result["request_id"],
        "object": "text_completion",
        "choices": [{
            "text": result["completion"],
            "index": 0,
            "finish_reason": "length"
        }],
        "usage": {
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": result["tokens_generated"],
            "total_tokens": len(request.prompt.split()) + result["tokens_generated"]
        },
        "allocation_info": {
            "gpu_kv_tokens": result["allocation"].gpu_kv_cache_tokens,
            "cpu_kv_tokens": result["allocation"].cpu_kv_cache_tokens,
            "ssd_kv_tokens": result["allocation"].ssd_kv_cache_tokens,
            "hot_experts": result["allocation"].gpu_hot_experts
        },
        "performance_estimate": result["performance"]
    }


@app.get("/memory/status")
async def memory_status():
    """Real-time memory status across all tiers."""
    
    pressure = server.monitor.check_pressure()
    
    return {
        "gpu": {
            "utilization": f"{pressure['gpu_utilization']:.1%}",
            "allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB",
            "total": f"{server.monitor.gpu_capacity / 1e9:.2f}GB",
            "trend": pressure["gpu_trend"]
        },
        "cpu": {
            "utilization": f"{pressure['cpu_utilization']:.1%}",
            "trend": pressure["cpu_trend"]
        },
        "pressure_level": pressure["pressure_level"],
        "alerts": pressure["alerts"],
        "active_requests": len(server.request_handler.active_requests),
        "kv_cache_stats": server.kv_cache_manager.stats
    }


@app.post("/memory/calculate")
async def calculate_allocation(request: AllocationRequest):
    """
    Calculate memory allocation for hypothetical request.
    Helps users understand what configuration will work.
    """
    
    user_request = UserRequest(
        max_context_length=request.max_context_length,
        batch_size=request.batch_size,
        allow_cpu_offload=request.allow_cpu_offload,
        allow_ssd_offload=request.allow_ssd_offload
    )
    
    allocation = server.calculator.calculate_allocation(user_request)
    
    if allocation.can_fulfill:
        performance = server.request_handler._estimate_performance(allocation)
        
        return {
            "can_fulfill": True,
            "allocation": {
                "gpu_kv_tokens": allocation.gpu_kv_cache_tokens,
                "cpu_kv_tokens": allocation.cpu_kv_cache_tokens,
                "ssd_kv_tokens": allocation.ssd_kv_cache_tokens,
                "hot_experts": allocation.gpu_hot_experts,
                "warm_experts": allocation.cpu_warm_experts
            },
            "performance_estimate": performance
        }
    else:
        # Find max feasible and return suggestions
        max_feasible = server.request_handler._find_max_feasible_context(user_request)
        suggestions = server.request_handler._generate_suggestions(user_request, allocation)
        
        return {
            "can_fulfill": False,
            "reason": allocation.rejection_reason,
            "max_feasible_context": max_feasible,
            "suggestions": suggestions
        }
```

---

## Performance Characteristics

### Latency Profile by Context Length

| Context Length | GPU KV | CPU KV | SSD KV | First Token | Throughput | Hardware |
|---------------|--------|--------|--------|-------------|------------|----------|
| 2K tokens     | 2K     | 0      | 0      | ~0.5s       | 80 tok/s   | 6GB GPU  |
| 8K tokens     | 4K     | 4K     | 0      | ~1.5s       | 50 tok/s   | 6GB GPU  |
| 32K tokens    | 4K     | 28K    | 0      | ~5s         | 30 tok/s   | 6GB GPU  |
| 128K tokens   | 4K     | 64K    | 60K    | ~30s        | 15 tok/s   | 6GB GPU  |
| 1M tokens     | 2K     | 64K    | 934K   | ~300s       | 2 tok/s    | 4GB GPU  |

### Memory Efficiency Comparison

| Approach | 2K Context | 32K Context | 128K Context | 1M Context |
|----------|------------|-------------|--------------|------------|
| **Standard vLLM** | ✅ 6GB | ❌ 24GB | ❌ 96GB | ❌ 768GB |
| **Ours (GPU only)** | ✅ 6GB | ❌ 12GB | ❌ 48GB | ❌ 384GB |
| **Ours (GPU+CPU)** | ✅ 6GB | ✅ 8GB | ✅ 20GB | ❌ 150GB |
| **Ours (GPU+CPU+SSD)** | ✅ 6GB | ✅ 8GB | ✅ 12GB | ✅ 50GB |

### Tier Access Patterns

**Short Context (2-8K)**:
- 100% GPU cache hits
- Zero CPU/SSD access
- Full throughput

**Medium Context (8-64K)**:
- 80% GPU hits (recent tokens)
- 20% CPU hits (older tokens)
- ~60% throughput

**Long Context (64K-512K)**:
- 60% GPU hits (recent)
- 30% CPU hits (middle)
- 10% SSD hits (oldest)
- ~30% throughput

**Extreme Context (512K-1M)**:
- 40% GPU hits
- 30% CPU hits
- 30% SSD hits
- ~15% throughput

---

## User Experience Examples

### Example 1: Conservative User (Default)

```python
# User doesn't specify anything - gets sensible defaults
response = requests.post("http://localhost:8000/v1/completions", json={
    "prompt": "Once upon a time",
    "max_tokens": 100
})

# System uses:
# - 2048 token context (default)
# - All GPU KV cache
# - 4 hot experts on GPU
# Result: Fast, stable, 80 tok/s
```

### Example 2: Power User (Custom)

```python
# User wants long context
response = requests.post("http://localhost:8000/v1/completions", json={
    "prompt": very_long_document,
    "max_context_length": 65536,  # 64K tokens
    "max_tokens": 500
})

# System calculates:
# - 4K GPU KV + 60K CPU KV
# - 2 hot experts (made room for KV)
# - Estimated 30 tok/s
# User gets clear performance prediction before running
```

### Example 3: Rejected Request with Suggestions

```python
# User asks for impossible config
response = requests.post("http://localhost:8000/v1/completions", json={
    "prompt": "Hello",
    "max_context_length": 1000000,  # 1M tokens
    "batch_size": 4,  # 4 concurrent
    "allow_cpu_offload": False,  # GPU only
    "allow_ssd_offload": False
})

# Response: 429 Too Many Requests
{
    "error": "Request rejected",
    "reason": "Cannot fit 1M token context in GPU-only mode",
    "suggestions": [
        "Enable CPU offloading (allow_cpu_offload=True)",
        "Enable SSD offloading (allow_ssd_offload=True)",
        "Reduce batch size from 4 to 1",
        "Reduce context to 4096 tokens for GPU-only mode"
    ],
    "max_feasible_gpu_only": 4096
}
```

### Example 4: Check Before Running

```python
# User wants to know if config will work BEFORE running
response = requests.post("http://localhost:8000/memory/calculate", json={
    "max_context_length": 131072,  # 128K
    "batch_size": 1,
    "allow_cpu_offload": True,
    "allow_ssd_offload": True
})

# Response:
{
    "can_fulfill": true,
    "allocation": {
        "gpu_kv_tokens": 4096,
        "cpu_kv_tokens": 65536,
        "ssd_kv_tokens": 61440,
        "hot_experts": 2
    },
    "performance_estimate": {
        "estimated_throughput_tokens_per_sec": 15.2,
        "estimated_first_token_latency_sec": 25.8,
        "tier_distribution": {
            "gpu_percent": 45.2,
            "cpu_percent": 35.1,
            "ssd_percent": 19.7
        }
    }
}

# Now user knows: Yes it will work, but expect ~15 tok/s with ~26s prefill
```

---

## Implementation Roadmap

### Prerequisites: Verify Existing Integration Status

**CRITICAL**: Before starting new work, confirm these components from prior integrations are operational:

- [ ] `FourPhaseOrchestrator` (from `RESOURCE_AWARE_WEIGHT_LOADING_DESIGN.md`)
  - Located: `sparse_llm/loading/orchestrator.py`
  - Validates: Resource profiling → Placement → Loading → Backend integration works
  
- [ ] `ResourceProfiler` (from `sparse_llm/loading/profiler.py`)
  - Validates: GPU/CPU/SSD detection with safety margins (85%/80%/90%)
  
- [ ] `PlacementStrategyCalculator` (from `sparse_llm/loading/strategy.py`)
  - Validates: Three-tier weight placement calculation works
  
- [ ] `LayerAwareCache` (from `sparse_llm/core/cache.py`)
  - Validates: Expert weight caching with LRU, pin/unpin, GPU↔CPU movement
  - Confirms: In-flight load de-duplication works
  
- [ ] `WeightLoader` (from `sparse_llm/loading/loader.py`)
  - Validates: Loads weights to correct tiers (GPU/CPU/SSD)
  
- [ ] vLLM integration exists (from `VLLM_INTEGRATION.md`)
  - Located: `sparse_llm/integrations/vllm_bridge.py`
  - Note: May currently OOM, but structure should exist

**If any component is missing or broken:** Fix existing integration FIRST before proceeding with this plan.

---

### Week 1: Core Components (Building on Existing Foundation)

**Extend existing profiler with unified three-tier allocation:**

- [ ] **Extend `ResourceProfiler`** → `UnifiedResourceProfiler`
  - Add: Unified allocation calculation for BOTH expert weights AND KV cache across GPU/CPU/SSD
  - Keep: Existing GPU/CPU/SSD detection and safety margins (85%/80%/90%)
  - New method: `calculate_unified_allocation(model_info, user_request)` 
  - Returns: Coordinated budgets for expert weights AND KV cache per tier

- [ ] Implement `UnifiedMemoryBudgetCalculator`
  - Consumes: `UnifiedResourceProfiler` output
  - Produces: `UnifiedMemoryAllocation` with:
    - Expert allocation: num_hot_experts, num_warm_experts, num_cold_experts
    - KV allocation: gpu_kv_tokens, cpu_kv_tokens, ssd_kv_tokens
    - vLLM parameters: gpu_memory_utilization, cpu_offload_gb, enable_lmcache
  - Algorithm: Split each tier's budget between experts and KV cache
  - Validation: Total allocation fits across all tiers

- [ ] Implement tier split strategies
  - Balanced strategy: 50/50 split per tier
  - Weight-heavy strategy: 70% experts, 30% KV (for MoE models with short context)
  - KV-heavy strategy: 30% experts, 70% KV (for long context workloads)
  - Auto-detect: Choose strategy based on max_model_len and model architecture

- [ ] Implement `MemoryPressureMonitor`
  - Real-time monitoring of GPU/CPU/SSD pressure for BOTH expert weights AND KV cache
  - Track per-tier utilization: expert_bytes_used + kv_bytes_used vs tier_capacity
  - Trend detection (increasing/stable/decreasing)
  - Alert generation for predictive OOM

- [ ] Unit tests for unified calculator
  - Test: 2K, 8K, 32K, 128K, 200K context lengths (via --max-model-len)
  - Test: Different tier split strategies
  - Test: Different dtypes (bfloat16, float16, float32)
  - Test: Different quantizations (awq, gptq, none)
  - Test: 4GB, 6GB, 24GB GPU configurations with various CPU/SSD sizes
  - Test: Edge cases (GPU-only, CPU required, SSD required)

---

### Week 2: Unified Three-Tier Coordinator (Critical Integration)

**Coordinate expert weights AND KV cache across all tiers:**

- [ ] Implement `UnifiedThreeTierCoordinator`
  - Pre-calculates unified allocation BEFORE loading anything
  - Coordinates BOTH SparseLLM expert cache AND vLLM KV cache
  - Sets vLLM's three-tier parameters: `--gpu-memory-utilization`, `--cpu-offload-gb`, `--enable-lmcache`
  - Configures SparseLLM's expert placement across GPU/CPU/SSD tiers
  
- [ ] **Modify `FourPhaseOrchestrator`** → Add "Phase 0: Unified Pre-Calculation"
  - Phase 0: Calculate unified tier allocation (NEW)
  - Phase 1: Resource profiling (EXISTING)
  - Phase 2: Expert placement strategy with allocated budgets (EXISTING, now coordinated)
  - Phase 3: Load expert weights to allocated tiers (EXISTING)
  - Phase 4: Initialize vLLM with coordinated KV cache parameters (EXISTING, now multi-tier)

- [ ] Update `vllm_bridge.py` with unified coordination
  - Replace single `--gpu-memory-utilization` with three-tier parameters
  - Add `--cpu-offload-gb` calculation based on CPU tier KV allocation
  - Add `--enable-lmcache` when SSD tier KV allocation > 0
  - Preserve all other user-configurable vLLM parameters

- [ ] Integration tests for unified three-tier system
  - Test: 4GB GPU + 16GB CPU + 100GB SSD with 64K context
    - Verify: Expert weights use GPU/CPU/SSD as calculated
    - Verify: vLLM KV cache uses coordinated GPU/CPU/disk tiers
    - Verify: No OOM during initialization or runtime
  - Test: 6GB GPU + 32GB CPU + 500GB SSD with 128K context
  - Test: Different split strategies produce valid allocations
  - Test: vLLM initialization succeeds with calculated parameters

---

### Week 3: Admission Control & Request Validation

**Validate total resources across all tiers before loading:**

- [ ] Implement `UnifiedAdmissionController`
  - Pre-validates server startup parameters
  - Checks: expert_weights_total + kv_cache_total <= (GPU + CPU + SSD available)
  - Validates per-tier: Each tier can hold its allocated expert + KV budget
  - Rejects BEFORE loading if insufficient resources across ANY tier

- [ ] Implement unified admission control logic
  - Server startup validation with detailed breakdown
  - Display allocation plan to user:
    ```
    GPU Tier: 3.2GB experts + 1.5GB KV = 4.7GB / 5.1GB available ✓
    CPU Tier: 5.4GB experts + 15GB KV = 20.4GB / 25.6GB available ✓
    SSD Tier: 28GB experts + 10GB KV = 38GB / 450GB available ✓
    Total: Can support 128K context on this hardware
    ```
  - Clear rejection reasons with tier-specific breakdown
  - Example: "GPU tier exhausted: need 6.2GB, have 5.1GB. Suggestions: reduce hot experts, reduce GPU KV allocation, enable CPU offload"

- [ ] Implement predictive OOM prevention
  - Monitor pressure during runtime for BOTH systems
  - Track: expert_cache pressure + vLLM KV pressure per tier
  - Alert if approaching limits (85% utilization warning, 95% critical)

- [ ] Add rejection reason tracking with actionable suggestions
  - Reason categories: 
    - GPU tier exhausted (weights + KV exceed GPU capacity)
    - CPU tier exhausted (CPU offload needed but insufficient RAM)
    - SSD tier exhausted (disk space insufficient for cold storage)
    - Invalid max_model_len (context too large for available resources)
  - Suggestions:
    - Reduce --max-model-len to lower KV cache requirements
    - Enable --quantization to reduce weight sizes
    - Use smaller model (fewer experts)
    - Increase CPU RAM or free up disk space

---

### Week 4: Integration & User Control

**Preserve all existing vLLM parameters, add unified tier management:**

- [ ] Add `UnifiedServer` wrapper
  - Wraps `UnifiedThreeTierCoordinator`
  - Accepts all standard vLLM CLI parameters
  - Adds unified tier management transparently
  - Displays allocation plan before starting server

- [ ] Update server startup script (`serve.py`)
  - Accept all vLLM parameters: `--max-model-len`, `--dtype`, `--quantization`, etc.
  - Accept unified tier preferences:
    - `--tier-split-strategy` (balanced/weight-heavy/kv-heavy/auto)
    - `--allow-cpu-offload` (default: true)
    - `--allow-ssd-offload` (default: true)
  - Remove user-facing `--gpu-memory-utilization` and `--cpu-offload-gb` (calculated automatically)
  - Display calculated unified allocation before starting:
    ```
    Unified Memory Allocation:
    ─────────────────────────────────────
    GPU Tier (5.1GB):
      Expert Weights: 3.2GB (16 hot experts)
      KV Cache: 1.5GB (~8K tokens)
    
    CPU Tier (25.6GB):
      Expert Weights: 5.4GB (28 warm experts)
      KV Cache: 15GB (~80K tokens)
    
    SSD Tier (450GB):
      Expert Weights: 28GB (148 cold experts)
      KV Cache: 10GB (~50K tokens)
    
    Total Context Supported: 138K tokens
    vLLM Parameters: --gpu-memory-utilization 0.25 --cpu-offload-gb 15 --enable-lmcache
    ─────────────────────────────────────
    Press Enter to start server or Ctrl+C to adjust...
    ```

- [ ] Add monitoring API endpoints:
  - `GET /memory/status` - Real-time usage for expert weights AND KV cache per tier
  - `GET /memory/allocation` - Show unified allocation plan
  - `GET /memory/pressure` - Per-tier pressure metrics
  - `POST /v1/completions` - Standard OpenAI-compatible (already exists via vLLM)
  - `POST /v1/chat/completions` - Standard OpenAI-compatible (already exists via vLLM)

- [ ] Preserve per-request parameters (already handled by vLLM):
  - `temperature`, `top_p`, `top_k`, `max_tokens`
  - `frequency_penalty`, `presence_penalty`
  - `stop` sequences
  - All standard OpenAI parameters

- [ ] End-to-end testing with unified three-tier system
  - Test: Start server with 32K context, verify GPU/CPU tiers used
  - Test: Start server with 128K context, verify GPU/CPU/SSD tiers used
  - Test: Different dtype and quantization combinations
  - Test: Multi-user concurrent requests (vLLM handles batching)
  - Test: Rejection when parameters exceed resources
  - Test: Runtime tier transitions (GPU→CPU→SSD for both weights and KV)

---

### Week 5: Optimization & Performance Tuning

- [ ] Profile unified tier allocation accuracy
  - Measure: Predicted vs actual usage per tier for BOTH systems
  - Tune: Split ratios if predictions consistently off
  - Optimize: Safety margins if too conservative

- [ ] Optimize tier transition performance
  - Expert weights: Batch expert loads where possible
  - KV cache: Verify vLLM's automatic offloading is efficient
  - Use pinned memory for faster CPU↔GPU transfers

- [ ] Add tier split strategy auto-tuning
  - Detect: Is workload more expert-intensive or context-intensive?
  - Adjust: Split ratios dynamically based on actual usage patterns
  - Learn: Remember optimal splits for specific model + context combinations

- [ ] Benchmark unified system vs baseline
  - Baseline: Current system (OOMs at ~8-16K context on 6GB GPU)
  - Target: New unified system (supports 100K-200K context on same hardware)
  - Measure: Maximum supported context length, throughput, latency, startup time
  - Measure: Per-tier hit rates (GPU/CPU/SSD access frequencies)

---

### Week 6: Documentation & Production Readiness

- [ ] Update `RESOURCE_AWARE_LOADING_USAGE.md`
  - Add: Unified three-tier architecture explanation
  - Add: Server startup examples with various configurations
  - Add: How expert weights and KV cache share tier resources
  - Add: Tier split strategy selection guide

- [ ] Write comprehensive user guide: "Unified Three-Tier Memory Management"
  - Explain: How SparseLLM and vLLM coordinate tier usage
  - Show: All vLLM parameters users can customize
  - Show: Tier split strategies and when to use each
  - Show: Expected context lengths by GPU/CPU/SSD configuration
  - Show: Troubleshooting unified allocation issues

- [ ] Add unified monitoring dashboard (optional)
  - Real-time: Per-tier usage for expert weights + KV cache
  - Visual: Stacked bar charts showing tier utilization
  - Historical: Request handling with tier breakdowns
  - Alerts: Per-tier approaching capacity warnings

- [ ] Performance tuning documentation by hardware configuration
  - Document: Expected context lengths for common configs:
    - 4GB GPU + 16GB CPU + 100GB SSD: ~32K-64K context
    - 6GB GPU + 32GB CPU + 500GB SSD: ~128K-200K context
    - 24GB GPU + 64GB CPU + 1TB SSD: ~500K-1M context
  - Document: Impact of dtype/quantization on capacity
  - Presets: Configuration templates for different use cases:
    - "fast-short-context": Maximize GPU allocation for speed
    - "balanced": Even distribution across tiers
    - "maximum-context": Maximize SSD usage for longest context

- [ ] Production deployment guide
  - Hardware recommendations for target context lengths
  - Configuration templates with unified tier parameters
  - Monitoring setup for both SparseLLM and vLLM metrics
  - Troubleshooting checklist for unified tier issues
  - Best practices for tier split strategy selection

---

## Risk Mitigation

### Risk 1: SSD Latency Too High
**Impact**: 1M context becomes unusable  
**Mitigation**:
- Use memory-mapped files for faster access
- Prefetch likely-accessed tokens
- Compress old KV cache (quantize to int8)
- Batch SSD reads (load 1K tokens at once, not 1)

### Risk 2: CPU-GPU Transfer Bottleneck
**Impact**: CPU tier doesn't help throughput  
**Mitigation**:
- Use pinned memory for transfers
- Overlap transfers with compute
- Batch transfers (move multiple tokens together)
- Use CUDA streams for async transfers

### Risk 3: Calculation Inaccurate
**Impact**: OOM despite predictions  
**Mitigation**:
- Add 15% safety buffer to all calculations
- Runtime validation before allocation
- Monitor actual vs predicted usage
- Automatic adjustment based on history

### Risk 4: User Confusion
**Impact**: Users don't know what config to use  
**Mitigation**:
- Provide `/memory/calculate` endpoint
- Show performance estimates upfront
- Clear error messages with suggestions
- Preset configurations (fast/balanced/maximum-context)

---

## Summary: Integration Strategy

### Components to Reuse (No Changes)
- ✅ `LayerAwareCache` - Expert weight caching works as-is
- ✅ `WeightLoader` - Weight loading to tiers works as-is
- ✅ `ModelIntrospector` - Model architecture detection works as-is
- ✅ Safety margins (85%/80%/90%) - Validated, keep unchanged

### Components to Extend (Minimal Changes)
- 🔧 `ResourceProfiler` → Add `calculate_dynamic_allocation(user_request)`
- 🔧 `PlacementStrategyCalculator` → Accept dynamic budgets instead of fixed ratios
- 🔧 `FourPhaseOrchestrator` → Wrap with `AdaptiveMemoryOrchestrator` (adds Phase 0)
- 🔧 `vllm_bridge.py` → Add memory coordination logic

### Components to Build (New)
- 🆕 `DynamicMemoryBudgetCalculator` - Per-request allocation calculation
- 🆕 `ThreeTierKVCache` - KV cache tiering (mirrors LayerAwareCache pattern)
- 🆕 `MemoryPressureMonitor` - Real-time pressure tracking
- 🆕 `AdaptiveRequestHandler` - Admission control logic

### Migration Path for Users

**Current usage (will continue to work):**
```python
from sparse_llm.loading import FourPhaseOrchestrator

orchestrator = FourPhaseOrchestrator()
state = orchestrator.initialize("mistralai/Mixtral-8x7B-Instruct-v0.1")
```

**New usage (recommended):**
```python
from sparse_llm.adaptive import AdaptiveMemoryOrchestrator

orchestrator = AdaptiveMemoryOrchestrator()
state = orchestrator.initialize(
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_context_length=32768,  # User-specified
    batch_size=1,
    allow_cpu_offload=True,
    allow_ssd_offload=True
)
```

**Backwards compatibility:** Old API wraps new API with default context_length=2048.

---

## Success Criteria

✅ **Zero OOM Guarantee**: No crashes due to memory during vLLM initialization or runtime  
✅ **Flexible Context**: Support user-specified context lengths (2K to 128K+) via `--max-model-len`  
✅ **User Control Preserved**: ALL vLLM parameters remain configurable (dtype, quantization, temperature, max_tokens, etc.)  
✅ **Predictable**: Memory allocation calculated and displayed before server starts  
✅ **Rejection Clarity**: Clear reasons + actionable suggestions when configuration exceeds resources  
✅ **Monitoring**: Real-time memory status for expert weights and vLLM KV cache  
✅ **Performance**: vLLM's native performance maintained (no slowdown from coordination)  
✅ **Multi-User**: vLLM's request batching and scheduling work unchanged  
✅ **Backwards Compatible**: Existing vLLM workflows continue working with calculated `--gpu-memory-utilization`  

---

## Appendix: Mathematical Formulas

### KV Cache Size Per Token
```
bytes_per_token = 2 (K+V) × num_layers × hidden_size × 2 (bf16)

Example (Qwen1.5-MoE-A2.7B):
= 2 × 24 layers × 2048 hidden × 2 bytes
= 196,608 bytes per token
≈ 192 KB per token

For 1M context:
= 192 KB × 1M tokens
= 192 GB total KV cache
```

### Activation Buffer Size
```
activation_bytes = batch_size × hidden_size × intermediate_size × 4 (safety)

Example:
= 1 × 2048 × 5632 × 4
= 46,137,344 bytes
≈ 44 MB
```

### Expert Weight Size
```
expert_bytes = 3 × hidden_size × intermediate_size × 2 (bf16)

Example:
= 3 × 2048 × 5632 × 2
= 69,206,016 bytes
≈ 66 MB per expert
```

### Total Experts Storage
```
total = num_layers × num_experts × expert_bytes

Example (Qwen):
= 24 × 8 × 66 MB
= 12.67 GB for all experts
```
