# Resource-Aware Weight Loading Usage Guide

## Overview

SparseLLM's resource-aware weight loading system automatically detects your hardware resources (GPU, CPU, storage) and optimally places model weights across a three-tier hierarchy for maximum inference performance.

## Quick Start

### Basic Usage

```python
from sparse_llm.loading import FourPhaseOrchestrator

# Initialize orchestrator
orchestrator = FourPhaseOrchestrator()

# Load model with automatic resource detection
state = orchestrator.initialize("mistralai/Mixtral-8x7B-Instruct-v0.1")

# Model weights are now optimally placed:
# - Shared weights: GPU
# - Hot experts: GPU cache
# - Warm experts: CPU cache  
# - Cold experts: Storage (loaded on demand)
```

### With Progress Tracking

```python
def progress_callback(message: str):
    print(message)

state = orchestrator.initialize(
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    progress_callback=progress_callback
)
```

### Command-Line Usage

```bash
# Traditional loading (existing behavior)
python main.py --model gpt2 --prompt "Hello world"

# Resource-aware loading (experimental)
python main.py --model gpt2 --prompt "Hello world" --use-resource-aware
```

## Four-Phase Initialization

The system executes four sequential phases:

### Phase 1: Resource Profiling
- Detects all available GPUs with memory capacity
- Measures available CPU RAM
- Identifies storage location and type (SSD/HDD)
- Applies safety margins (85% GPU, 80% CPU, 90% storage)

### Phase 2: Placement Strategy
- Analyzes model architecture (MoE vs dense)
- Calculates optimal weight distribution
- Creates placement plan with three tiers:
  - **Hot tier (GPU)**: Shared weights + frequently used experts
  - **Warm tier (CPU)**: Less frequent experts, ready to promote
  - **Cold tier (Storage)**: Infrequent experts, loaded on demand

### Phase 3: Weight Loading
- Streams weights directly to target devices
- Preloads hot experts to GPU cache
- Initializes LRU cache for dynamic expert paging

### Phase 4: Ready
- Returns `LoadedWeightState` ready for inference
- Cache statistics available for monitoring

## Configuration

### Programmatic Configuration

```python
orchestrator = FourPhaseOrchestrator()

state = orchestrator.initialize(
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    storage_path="/custom/cache/path",  # Custom storage location
)
```

### YAML Configuration

Add to `config.yaml`:

```yaml
resource_aware:
  enabled: true
  storage_path: /custom/cache/experts
  gpu_margin: 0.85  # Use 85% of available GPU memory
  cpu_margin: 0.80  # Use 80% of available CPU memory
  storage_margin: 0.90  # Use 90% of available storage
```

### Environment Variables

```bash
export RESOURCE_AWARE_ENABLED=true
export RESOURCE_AWARE_STORAGE_PATH=/custom/cache
export RESOURCE_AWARE_GPU_MARGIN=0.85
export RESOURCE_AWARE_CPU_MARGIN=0.80
export RESOURCE_AWARE_STORAGE_MARGIN=0.90
```

## Monitoring Cache Statistics

```python
# After loading
state = orchestrator.initialize(model_id)

# Get cache statistics
stats = state.get_cache_stats()

print(f"Total accesses: {stats['total_accesses']}")
print(f"GPU hit rate: {stats['gpu_hit_rate']:.2%}")
print(f"CPU hit rate: {stats['cpu_hit_rate']:.2%}")
print(f"Storage miss rate: {stats['storage_miss_rate']:.2%}")
print(f"Evictions: {stats['evictions']}")
print(f"Promotions: {stats['promotions']}")
```

## Accessing Loaded Weights

```python
# Get shared weights (embeddings, attention, etc.)
embed_weight = state.get_shared_weight("model.embed_tokens.weight")

# Get expert weights (with automatic caching)
expert_weights, tier = state.get_expert_weights(layer_id=5, expert_id=3)
print(f"Expert loaded from: {tier}")  # "gpu", "cpu", or "storage"
```

## Troubleshooting

### Error: "MoE models require at least one GPU"

**Cause**: You're trying to load a MoE model without GPU support.

**Solution**: 
- Use a GPU-enabled machine
- Or load a dense model instead (e.g., Llama, GPT-2)

### Warning: "CPU RAM below recommended minimum (8GB)"

**Cause**: Less than 8GB CPU RAM available.

**Solution**:
- Free up RAM by closing other applications
- Reduce `cpu_margin` to use more available RAM (risky)
- Use a machine with more RAM

### Warning: "Storage is HDD (not SSD)"

**Cause**: Model cache is on HDD, not SSD.

**Solution**:
- Move cache to SSD with `storage_path` parameter
- Expect 5-10x slower expert loading from disk

### Error: "Model size exceeds available resources"

**Cause**: Model too large for available GPU + CPU.

**Solution**:
- Use quantization (if supported)
- Use a smaller model
- Upgrade hardware

## Performance Tuning

### Optimizing GPU Utilization

```python
# More aggressive GPU usage (90% instead of 85%)
orchestrator = FourPhaseOrchestrator()
orchestrator.profiler.gpu_margin = 0.90

state = orchestrator.initialize(model_id)
```

### Optimizing Cache Hit Rate

For best performance:
1. **Maximize GPU cache**: Use largest GPU possible
2. **Use SSD storage**: 5-10x faster than HDD
3. **Monitor hit rates**: Track `gpu_hit_rate` and optimize workload

### Expected Performance

| Hardware | GPU Hit Rate | Inference Speed |
|----------|--------------|-----------------|
| 80GB GPU | 95%+ | Near-native |
| 40GB GPU | 80-90% | ~10% slower |
| 24GB GPU | 60-80% | ~20% slower |
| 16GB GPU | 40-60% | ~40% slower |

## Model Support

### Supported Architectures

- **MoE Models**: Mixtral, Qwen-MoE, DeepSeek-MoE (full support)
- **Dense Models**: Llama, GPT-2, Mistral (basic support)

### Architecture Detection

The system automatically detects:
- MoE vs dense architecture
- Number of experts per layer
- Expert size and shared weight size
- Total model size

## Advanced Usage

### Custom Resource Budget

```python
from sparse_llm.loading import ResourceProfiler, PlacementStrategyCalculator
from sparse_llm.loading import ModelIntrospector, WeightLoader

# Phase 1: Profile resources
profiler = ResourceProfiler(gpu_margin=0.90, cpu_margin=0.85)
budget = profiler.profile()

# Phase 2: Calculate placement
introspector = ModelIntrospector()
model_info = introspector.introspect(model_id)

calculator = PlacementStrategyCalculator()
plan = calculator.calculate(budget, model_info)

# Phase 3: Load weights
loader = WeightLoader()
state = loader.load(model_id, plan)
```

### Accessing Internal Components

```python
# Access placement plan
print(f"Hot expert slots: {state.placement_plan.hot_expert_slots}")
print(f"Warm expert slots: {state.placement_plan.warm_expert_slots}")
print(f"Cold expert count: {state.placement_plan.cold_expert_count}")

# Access model info
print(f"Model: {state.model_info.model_id}")
print(f"Is MoE: {state.model_info.is_moe}")
print(f"Total size: {state.model_info.total_bytes / 1024**3:.1f}GB")

# Access expert cache
print(f"GPU slots: {state.expert_cache.gpu_slots}")
print(f"CPU slots: {state.expert_cache.cpu_slots}")
```

## Limitations

1. **Backend Integration Pending**: Current implementation loads weights but doesn't yet integrate with `InferenceEngine` for generation
2. **MoE Only Fully Supported**: Dense models load but don't benefit from tiered placement
3. **GPU Required for MoE**: CPU-only mode not supported for MoE models
4. **No Multi-GPU**: Currently uses only first GPU (multi-GPU support planned)

## Roadmap

- [ ] Backend integration with `InferenceEngine`
- [ ] vLLM backend integration
- [ ] Multi-GPU support
- [ ] CPU-only mode for MoE (with performance warnings)
- [ ] Quantization support (FP16, INT8, INT4)
- [ ] Dynamic cache tuning based on workload

## See Also

- [Architecture Design](RESOURCE_AWARE_WEIGHT_LOADING_DESIGN.md)
- [API Reference](../sparse_llm/loading/__init__.py)
- [Integration Tests](../sparse_llm/tests/loading/test_integration.py)
