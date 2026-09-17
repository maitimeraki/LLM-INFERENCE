# Streaming Mode for Low-Resource Devices

## Overview

Streaming mode disables caching and uses a load-execute-free pattern for expert weights. This provides predictable memory usage at the cost of lower throughput.

## When to Use

| Scenario | Recommendation |
|----------|----------------|
| GPU >= 16GB | Use cache mode (default) |
| GPU 8-16GB | Try cache mode first, fallback to streaming |
| GPU < 8GB | Use streaming mode |
| CPU-only | Use streaming mode |

## Memory Comparison

### Cache Mode (Default)
```
Memory = Shared weights + KV cache + Expert cache (GPU) + Expert cache (CPU)
```

### Streaming Mode
```
Memory = Shared weights + KV cache + ONE expert (peak)
```

For an 8GB GPU running Mixtral-8x7B:
- Shared weights: ~1 GB
- KV cache (1K context): ~1 GB
- One expert peak: ~0.5 GB
- **Total: ~3 GB** (vs ~6-8 GB with cache)

## Usage

```bash
# Enable streaming mode
sparse-llm generate --model mistralai/Mixtral-8x7B-v0.1 \
    --prompt "Hello" \
    --streaming-mode

# Tune parallelism (default: 4)
sparse-llm generate --model mistralai/Mixtral-8x7B-v0.1 \
    --prompt "Hello" \
    --streaming-mode \
    --max-parallel-expert-loads 8
```

## Performance

| Metric | Cache Mode | Streaming Mode |
|--------|-----------|----------------|
| Throughput | Higher | Lower |
| Memory | Variable | Predictable |
| Cold start | Faster (hot experts) | Slower (no hot experts) |
| GPU requirements | Higher | Lower |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ StreamingExpertLoader                                       │
├─────────────────────────────────────────────────────────────┤
│ 1. Router selects expert_ids[]                             │
│ 2. ThreadPoolExecutor loads experts in parallel              │
│ 3. MemoryGuard checks GPU availability                      │
│ 4. Execute each expert                                      │
│ 5. FREE: del expert, empty_cache()                          │
│ 6. Repeat for next token                                   │
└─────────────────────────────────────────────────────────────┘
```

## NO-CACHE Guarantee

Streaming mode has **ZERO dependencies** on cache code:
- No `HierarchicalExpertLoader`
- No `ExpertCache` or `ExpertCacheManager`
- No LRU eviction logic
- No promotion/demotion between tiers

## Future Improvements

- [ ] KV cache CPU offloading for streaming mode
- [ ] Hybrid mode: hot experts cached, cold experts streamed
- [ ] Predictive streaming: prefetch experts before needed
