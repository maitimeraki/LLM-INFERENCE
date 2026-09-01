# SparseLLM - Universal MoE Inference

SparseLLM is a universal, architecture-agnostic local inference runtime for **any** open-source Mixture-of-Experts (MoE) language model.

**Key Features:**
- 🌐 **Universal**: Works with ANY MoE model (Mixtral, Qwen2-MoE, DeepSeek-V2, DBRX, and more)
- 🧠 **Architecture-Agnostic**: Automatically detects and adapts to different architectures
- 💾 **Memory Efficient**: Dynamic expert paging reduces VRAM usage significantly
- ⚡ **Fast**: On-demand expert loading with intelligent caching
- 🔧 **Simple API**: Unified interface for all models

```text
model identifier or local checkpoint
  → automatic architecture detection
  → lazy tokenizer/model loading with expert paging
  → real prefill and autoregressive decoding with on-demand expert swapping
  → generated text, token IDs, and measured metrics (cache hits, expert load times)
```

## Scope

The **universal adapter** automatically handles:
- **Mixtral** (Mistral MoE variants)
- **Qwen2-MoE** (Qwen2 MoE series)
- **DeepSeek-V2** (DeepSeek MoE models)
- **DBRX** (Databricks MoE)
- **Generic MoE** (automatic pattern detection for unknown architectures)

The runtime automatically:
1. Detects model architecture from config
2. Maps expert tensor names based on architecture patterns
3. Builds checkpoint index from safetensors files
4. Pages experts on-demand during inference
5. Falls back gracefully to standard inference if paging cannot be enabled

**No model-specific code required!** The system detects the architecture and sets up paging automatically.

### Memory behavior

- Dense and unknown architectures use the model's normal Transformers loading
  behavior, including an explicit device map/offload policy when configured.
- Expert paging is disabled unless an architecture adapter validates module
  topology, router semantics, layer-aware expert identity, and numerical
  conformance.
- The bounded `ExpertCache` supports layer-aware keys, byte accounting, pinning,
  and safe LRU eviction for that future specialized path.
- A model that does not fit the requested device policy fails with an actionable
  error; the project does not promise that every model fits in a fixed VRAM size.

## Installation

```bash
pip install -e ".[dev]"
```

Runtime dependencies are PyTorch and Transformers. `safetensors` is included for
safe model artifacts. Tests do not download checkpoints.

## Command-line usage

The CLI does not load a model when showing help:

```bash
python main.py --help
```

Generate from a Hub model ID or local checkpoint directory:

```bash
python main.py \
  --model <hugging-face-id-or-local-path> \
  --prompt "Explain mixture-of-experts routing" \
  --max-new-tokens 32 \
  --device auto \
  --dtype float16 \
  --json
```

Useful operational controls include `--revision`, `--local-files-only`,
`--device-map`, `--offload-folder`, and `--trust-remote-code`. Remote code is
disabled by default. Generation output contains runtime-measured prefill and
decode latency, throughput, peak CUDA allocation when available, and honest
expert-cache metrics (zero for the generic adapter because it does not swap
experts).

## Python API

### Basic Usage (Works with ANY MoE model)

```python
from sparse_llm import InferenceEngine

# Works with Mixtral
engine = InferenceEngine(
    model="mistralai/Mixtral-8x7B-v0.1",
    device="cuda",
    dtype="bfloat16",
    expert_cache_bytes=2 * 1024**3,  # 2GB expert cache
)

result = engine.generate(
    "Explain how mixture of experts works:",
    max_new_tokens=100,
    temperature=0.7,
)

print(result.text)
print(f"Cache hits: {result.metrics.cache_hits}")
print(f"Cache misses: {result.metrics.cache_misses}")
print(f"Expert load time: {result.metrics.expert_load_time_ms:.2f}ms")
```

### Universal - Works with Multiple Architectures

```python
# Qwen2-MoE
engine = InferenceEngine(
    model="Qwen/Qwen2-57B-A14B-Instruct",
    device="cuda",
    expert_cache_bytes=3 * 1024**3,
)

# DeepSeek-V2
engine = InferenceEngine(
    model="deepseek-ai/DeepSeek-V2",
    device="cuda",
    expert_cache_bytes=4 * 1024**3,
)

# DBRX
engine = InferenceEngine(
    model="databricks/dbrx-instruct",
    device="cuda",
    expert_cache_bytes=2 * 1024**3,
)

# Any other MoE model - automatic detection!
engine = InferenceEngine(
    model="organization/new-moe-model",
    device="cuda",
    expert_cache_bytes=2 * 1024**3,
)
```

### Direct Universal Adapter Usage

```python
from sparse_llm import UniversalMoEAdapter, DevicePolicy

policy = DevicePolicy(
    device="cuda",
    dtype="bfloat16",
    expert_cache_bytes=2 * 1024**3,
)

adapter = UniversalMoEAdapter(
    model_id="mistralai/Mixtral-8x7B-v0.1",
    policy=policy,
)

# Check if paging is available
validation = adapter.validate_paging()
print(f"Paging eligible: {validation.eligible}")

# Load and generate
adapter.load()
result = adapter.generate("Hello, world!", max_new_tokens=50)

# Inspect capabilities
caps = adapter.capabilities
print(f"Model: {caps.model_type}")
print(f"Experts: {caps.num_experts}")
print(f"Top-K: {caps.top_k_experts}")
print(f"Paging enabled: {caps.expert_paging}")
```

### Advanced Configuration

```python
from sparse_llm import DevicePolicy, InferenceEngine

engine = InferenceEngine(
    model="org/model-or-local-directory",
    device="auto",  # auto, cuda, cpu
    dtype="float16",  # float32, float16, bfloat16
    local_files_only=False,
    trust_remote_code=False,
    device_map="auto",  # For multi-GPU
    offload_folder="./offload",  # CPU offloading
    expert_cache_bytes=2 * 1024**3,  # Expert cache size
)

result = engine.generate(
    "The future of local inference is",
    max_new_tokens=100,
    temperature=0.7,  # 0.0 for greedy
)

print(result.text)
print(result.metrics.to_dict())
print(engine.capabilities.to_dict())
```

## Current status and roadmap

The current baseline prioritizes correctness and measurement:

1. universal lazy Transformers adapter and real generation;
2. one canonical adapter-backed inference engine;
3. layer-aware cache/storage primitives with safe eviction and atomic local I/O;
4. capability reporting and safe fallback for unsupported MoE layouts.

After the baseline is measured, the roadmap is to add a validated Mixtral
expert-module pager, established quantization backends, asynchronous transfer
overlap, observed-next-use prefetch, batching/paged KV storage, and optional
multi-worker serving. Each optimization must preserve reference output within a
documented tolerance and show measured improvement against the baseline.

The project does not claim arbitrary 100B+ support, fixed throughput, fixed
latency, fixed cache-hit rates, custom INT4 kernels, speculative routing, or
production readiness without hardware-specific measurements.

## Testing

```bash
python -m pytest -q
python -m compileall -q sparse_llm main.py
```

CPU tests use fakes/tiny fixtures and do not require CUDA or network access. A
real-checkpoint smoke test is environment-dependent and must record the exact
model, revision, device, dtype, and measured results.

See [docs/GOAL.md](docs/GOAL.md), [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md),
and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for requirements and design.
