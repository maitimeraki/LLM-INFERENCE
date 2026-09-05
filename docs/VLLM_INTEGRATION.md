# vLLM Integration Guide

## Overview

This document explains how SparseLLM's resource-aware loading integrates with vLLM to provide both memory efficiency AND high throughput.

## Conflict Resolution Architecture

### Problem Statement

Running vLLM and resource-aware expert loading together previously caused five conflicts:

1. **Memory Conflicts**: Both systems allocated GPU memory independently → OOM crashes
2. **Duplicate Loading**: Weights loaded twice (once by each system) → wasted memory
3. **Expert Control**: Two systems competing to manage experts → cache thrashing
4. **CUDA Graphs**: Dynamic loading broke vLLM's compiled graphs → crashes
5. **Batch Thrashing**: vLLM batching caused excessive expert swapping → poor performance

### Solution Architecture

Three-layer integration:

**Layer 1: Unified Memory Coordination**
- `UnifiedMemoryCoordinator` splits GPU memory: 40% expert cache, 50% KV cache, 10% overhead
- Both systems respect static budgets established at startup
- No runtime memory conflicts

**Layer 2: Weight Bridge**
- `SparseMoEWeightBridge` makes vLLM use pre-loaded weights from `LoadedWeightState`
- Zero duplicate loading - single source of truth
- Expert cache managed by resource-aware system only

**Layer 3: Serving Layer** (future optimization)
- Expert-affinity scheduling groups similar requests
- Reduces cache thrashing from batching
- Conditional CUDA graph re-enablement

## Usage

### Basic Server Start

```bash
python serve.py --model mistralai/Mixtral-8x7B-Instruct-v0.1
```

### How It Works

1. **Phase 1: Resource-Aware Loading** (60-90s)
   - Detects GPU/CPU/Storage resources
   - Loads model weights across 3 tiers
   - Creates expert cache with LRU eviction

2. **Phase 2: vLLM Integration** (10-20s)
   - Memory coordinator allocates GPU budgets
   - Weight bridge connects systems
   - vLLM engine starts with pre-loaded weights

3. **Phase 3: Serving** (persistent)
   - OpenAI-compatible API endpoints
   - Weights stay loaded until shutdown
   - Expert cache handles dynamic loading

### API Endpoints

- `POST /v1/completions` - Text completion
- `POST /v1/chat/completions` - Chat completion
- `GET /health` - Health check
- `GET /metrics` - Detailed metrics (memory, cache stats, throughput)

### Configuration

**Memory Allocation** (automatic, can override):
```bash
# Default: 40% expert cache, 50% KV cache
# Override with environment variables:
export SPARSE_EXPERT_CACHE_RATIO=0.30  # 30% for experts
export SPARSE_KV_CACHE_RATIO=0.60       # 60% for KV cache
```

**Batch Size** (automatic based on available memory):
```bash
# Small GPU (8GB): batch size ~8-12
# Medium GPU (24GB): batch size ~16-24
# Large GPU (80GB): batch size ~24-32
```

## Performance Expectations

### Memory Efficiency

- **Without integration**: Need entire model in GPU memory
  - Mixtral 8x7B: Requires 72GB+ GPU memory
  - Not runnable on consumer hardware

- **With integration**: Run on 24GB GPU
  - Hot experts (2-4): GPU cache (~8-10GB)
  - Warm experts (8-12): CPU cache (~20-30GB)
  - Cold experts (remaining): Storage

### Throughput

| Configuration | Throughput | Notes |
|--------------|------------|-------|
| Standard vLLM | 100% | All experts in GPU memory |
| Phase 1 (MVP) | 50-65% | CUDA graphs disabled, basic integration |
| Phase 2 (Optimized) | 65-75% | Expert-affinity scheduling |
| Phase 3 (Advanced) | 75-85% | Conditional CUDA graphs |

### Latency

- Single request: Similar to standard vLLM (<100ms overhead)
- Batched requests: +10-50ms overhead for expert loading
- Cold experts: +100-200ms first access (from storage)
- Warm experts: +5-10ms (from CPU)
- Hot experts: <1ms (from GPU)

## Monitoring

### Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

Returns:
```json
{
  "memory": {
    "gpu_allocated_bytes": ...,
    "expert_cache_utilization": 0.85,
    ...
  },
  "expert_cache": {
    "gpu_cached_experts": 12,
    "cpu_cached_experts": 24,
    "hit_rate": 0.72,
    ...
  },
  "vllm": {
    "running_requests": 8,
    "max_batch_size": 16,
    ...
  }
}
```

### Key Metrics to Watch

- **GPU Cache Hit Rate**: Should be >70% after warmup
- **Expert Cache Utilization**: Should stay <95% (indicates healthy eviction)
- **Memory Allocated**: Should never exceed coordinator budgets
- **Evictions vs Promotions**: High eviction rate suggests batch thrashing

## Troubleshooting

### OOM Errors

**Symptom**: `torch.cuda.OutOfMemoryError` during startup or inference

**Cause**: Memory coordination not working

**Fix**:
1. Check memory coordinator is initialized: Look for "Unified Memory Coordination Initialized" in logs
2. Verify GPU memory: `torch.cuda.mem_get_info()`
3. Reduce expert cache ratio if needed: `export SPARSE_EXPERT_CACHE_RATIO=0.30`

### Low Cache Hit Rate (<50%)

**Symptom**: Metrics show gpu_hit_rate < 0.5

**Cause**: Cache thrashing from diverse request batches

**Fix**:
1. Reduce batch size: Will group more similar requests
2. Wait for Phase 2: Expert-affinity scheduling will fix this
3. Increase GPU cache budget: `export SPARSE_EXPERT_CACHE_RATIO=0.50`

### Weights Not Loading

**Symptom**: "Weight not found in bridge" errors

**Cause**: Weight name mismatch between vLLM and LoadedWeightState

**Fix**:
1. Check model architecture: Some models use different naming
2. Enable debug logging: `export SPARSE_LOG_LEVEL=DEBUG`
3. Report issue with model name and error details

## Comparison: main.py vs serve.py

### main.py (Research/CLI)

**Purpose**: Single-shot inference, experimentation, testing

**Capabilities**:
- Resource-aware loading with `--use-resource-aware`
- Runs one inference and exits
- Detailed statistics output
- No server overhead

**When to use**:
- Testing new models
- Benchmarking expert loading
- Debugging weight placement
- Quick experiments

**Example**:
```bash
python main.py --model gpt2 --prompt "Hello" --use-resource-aware
```

### serve.py (Production)

**Purpose**: Persistent serving, high throughput, production deployment

**Capabilities**:
- Resource-aware loading integrated with vLLM
- OpenAI-compatible API
- Request batching
- Persistent weights across requests
- Metrics endpoint

**When to use**:
- Production deployments
- Serving to coding agents (Claude Code, etc.)
- Multi-request workloads
- Need high throughput

**Example**:
```bash
python serve.py --model mistralai/Mixtral-8x7B-Instruct-v0.1
```

## Backward Compatibility

### Existing main.py Usage (Unchanged)

All existing main.py usage continues to work:

```bash
# Traditional loading (still works)
python main.py --model gpt2 --prompt "Hello"

# Resource-aware loading (still works)
python main.py --model gpt2 --prompt "Hello" --use-resource-aware

# All flags preserved
python main.py --model MODEL --prompt PROMPT \
  --device cuda --dtype float16 --cache-bytes 1000000000
```

The vLLM integration code is isolated in `sparse_llm/integrations/` and never imported by main.py.

### Old vllm_server.py (Deprecated)

The original vllm_server.py is deprecated but still functional. It uses the old integration approach (separate loading).

**Migration path**:
- Replace `python vllm_server.py --model MODEL` with `python serve.py --model MODEL`
- serve.py provides same API endpoints with better integration

## Architecture Details

For implementers and contributors, see:
- `docs/vllm_integration_architecture.md` - Complete technical design
- `sparse_llm/integrations/` - Integration code
- `sparse_llm/tests/integrations/` - Integration tests
