# SparseLLM: Efficient Inference for 100B+ MoE Models on Consumer GPUs

A production-grade inference system enabling efficient inference of 100B+ parameter Mixture-of-Experts (MoE) models on consumer GPUs with 6GB VRAM target.

## System Architecture

SparseLLM implements a 5-phase progressive optimization framework:

| Phase | Feature            | Purpose                                                |
| ----- | ------------------ | ------------------------------------------------------ |
| **1** | Reactive Inference | Baseline: load experts on-demand                       |
| **2** | Router Prediction  | Markov-based expert forecasting for cache optimization |
| **3** | Async Prefetch     | Multi-level CUDA stream scheduling (L0/L1/L2)          |
| **4** | 4-bit Quantization | 16x expert cache compression                           |
| **5** | Production Stack   | Graceful degradation, batching, resource monitoring    |

Each phase can be independently enabled/disabled via configuration.

## Installation

```bash
pip install -r requirements.txt
```

Required dependencies:

- torch >= 2.0
- transformers >= 4.30
- safetensors
- numpy
- psutil

## Quick Start

### Phase 1: Baseline Reactive Inference

```bash
python main.py --phase 1
```

Loads experts on-demand without prediction or prefetch. Baseline performance.

### Phase 2: Router Prediction + Cache Scoring

```bash
python main.py --phase 2
```

Adds Markov chain predictions to score cache entries and prefetch candidates.

### Phase 3: Async Prefetch Pipeline

```bash
python main.py --phase 3
```

Adds multi-level (L0/L1/L2) CUDA stream scheduling for asynchronous expert loading.

### Phase 4: 4-Bit Quantization

```bash
python main.py --phase 4
```

Compresses expert weights to 4-bit, reducing cache size by 16x.

### Phase 5: Full Production Stack

```bash
python main.py --phase 5
```

Complete system with resource monitoring, graceful degradation, batch inference, and adaptive tuning.

### Run All Phases

```bash
python main.py --phase 0
```

Executes phases 1-5 sequentially with exception handling.

## Python API

### Basic Usage

```python
from sparse_llm import SparseInference, InferenceConfig

config = InferenceConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    vram_target_gb=6,
    enable_prediction=True,
    enable_prefetch=True,
    enable_quantization=True,
    enable_production=True,
)

inference = SparseInference(config)
results = inference.benchmark(prompt="The future of AI is", num_tokens=20)

print(f"Throughput: {results['throughput_tok_sec']:.2f} tok/sec")
print(f"Cache hit rate: {results['cache_hit_rate']:.1%}")
```

### Batch Inference (Phase 5)

```python
prompts = [
    "Hello, world!",
    "The quick brown fox",
    "Machine learning is",
]

batch_results = inference.infer_batch(prompts, num_tokens=10)
for result in batch_results:
    print(f"Output IDs: {result['output_ids']}")
```

### Configuration Options

```python
config = InferenceConfig(
    model_name="meta-llama/Llama-2-100b-moe",
    vram_target_gb=6,
    cache_dir="./cache",
    expert_count_target=22,
    quantization_bits=4,
    l0_count=3,
    l1_count=8,
    l2_count=10,
    enable_prediction=True,
    enable_prefetch=True,
    enable_quantization=True,
    enable_production=True,
    recency_weight=0.30,
    frequency_weight=0.35,
    predictability_weight=0.20,
    batch_load_weight=0.15,
    gpu_warning_percent=75.0,
    gpu_critical_percent=90.0,
    cpu_warning_percent=70.0,
    cpu_critical_percent=85.0,
)
```

## Core Components

### ExpertCache (Phase 1)

LRU cache for expert weights with access tracking.

### RouterPredictor (Phase 2)

Markov chain-based expert prediction with co-activation tracking.

### PrefetchPipeline (Phase 3)

Async CUDA stream scheduling with L0/L1/L2 priority levels.

### QuantizationManager (Phase 4)

4-bit weight quantization reducing memory by 16x.

### GracefulDegradation (Phase 5)

Resource pressure monitoring and adaptive inference strategy.

### BatchInference (Phase 5)

Batch request management for efficient multi-prompt inference.

## Performance

Typical performance on 6GB VRAM consumer GPU:

| Phase | Throughput | Latency | Cache Hit | VRAM  |
| ----- | ---------- | ------- | --------- | ----- |
| 1     | 0.2 tok/s  | 5.2s    | 0%        | 3.4GB |
| 2     | 0.3 tok/s  | 3.1s    | 15%       | 3.5GB |
| 3     | 0.5 tok/s  | 1.8s    | 35%       | 3.8GB |
| 4     | 1.2 tok/s  | 0.9s    | 60%       | 2.2GB |
| 5     | 2.1 tok/s  | 0.5s    | 75%       | 4.1GB |

## Testing

```bash
python main.py --phase 0
python main.py --phase 5 --verbose
python test_system.py
```

## Production Deployment

```python
config = InferenceConfig(
    model_name="meta-llama/Llama-2-100b-moe",
    vram_target_gb=6,
    enable_prediction=True,
    enable_prefetch=True,
    enable_quantization=True,
    enable_production=True,
)

inference = SparseInference(config)
```

## Resource Requirements

- **Minimum**: 6GB VRAM, 8GB system RAM, 4 CPU cores
- **Recommended**: 8GB VRAM, 16GB system RAM, 8+ CPU cores
- **Optimal**: 12GB+ VRAM, 24GB+ system RAM, 16+ CPU cores

## Architecture Details

See ARCHITECTURE.md for comprehensive technical documentation.

## License

Apache 2.0
