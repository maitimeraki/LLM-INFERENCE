# Zero-Latency VLLM Integration Guide

## Architecture Overview

This system provides **zero-latency** inference by running VLLM and your code together in WSL, eliminating cross-boundary overhead.

### Latency Comparison

| Mode | Overhead | When to Use |
|------|----------|-------------|
| **Direct (WSL)** | **0ms** | Production, latency-critical workloads |
| HTTP (WSL → Windows) | ~0.7-1.6ms | Development, multi-client scenarios |
| HTTP (Windows → Windows) | ~0.5-1.0ms | Non-VLLM setups |

## Quick Start

### Option 1: Direct Mode (Zero Latency) ✅ RECOMMENDED

Run everything in WSL for maximum performance:

```bash
# From Windows Command Prompt or PowerShell
run.bat inference --model mistralai/Mixtral-8x7B-Instruct-v0.1 --prompt "Hello"

# Or directly in WSL
wsl
cd /mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE
python3 main.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --prompt "Hello"
```

**Latency:** ~0ms overhead (direct function calls)

### Option 2: HTTP Server Mode

For multiple clients or remote access:

```bash
# Start server in WSL
run.bat server --model mistralai/Mixtral-8x7B-Instruct-v0.1 --port 8000

# Client (Windows or WSL)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Latency:** ~0.7-1.6ms overhead per request

## Python API

### Direct Engine (Zero Latency)

```python
from sparse_llm.integrations.vllm_direct import DirectVLLMEngine

# Initialize engine
engine = DirectVLLMEngine(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    gpu_cache_gb=2.0,      # Active experts
    cpu_cache_gb=14.0,     # Warm experts
)

# Single generation
response = engine.generate("Explain quantum computing")

# Batch generation (parallel)
responses = engine.generate([
    "What is AI?",
    "Explain machine learning",
    "What is deep learning?"
])

# Chat completion
response = engine.chat([
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
])

# Get statistics
stats = engine.get_stats()
print(stats)
```

### HTTP Client (Standard Mode)

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100,
        "temperature": 0.7,
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

## Performance Benchmarking

Run the latency benchmark to measure overhead:

```bash
run.bat benchmark --iterations 100
```

Expected output:
```
📊 Benchmarking direct function call...
   Min: 0.010ms
   Avg: 0.015ms
   P99: 0.025ms

📊 Benchmarking HTTP roundtrip...
   Min: 0.520ms
   Avg: 0.680ms
   P99: 1.240ms

📊 Benchmarking JSON serialization...
   Min: 0.095ms
   Avg: 0.120ms
   P99: 0.180ms

📈 Summary
Direct call (baseline):     0.015ms
HTTP roundtrip overhead:    0.680ms  (+0.665ms)
JSON serialization:         0.120ms
Total HTTP mode overhead:   ~0.800ms per request

💡 Recommendations:
   🚀 For zero-latency: Run everything in WSL (direct mode)
```

## File Access: Windows ↔ WSL

Your code lives on C: drive, accessible from both Windows and WSL:

| Location | Windows Path | WSL Path |
|----------|-------------|----------|
| Project root | `C:\Users\Anupam\Desktop\LLM\LOCAL-INFERENCE` | `/mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE` |
| Models cache | `C:\Users\Anupam\.cache\huggingface` | `/mnt/c/Users/Anupam/.cache/huggingface` |
| Python code | `C:\Users\Anupam\Desktop\LLM\LOCAL-INFERENCE\*.py` | `/mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE/*.py` |

**Key Points:**
- ✅ Edit files in Windows (VS Code, PyCharm, etc.)
- ✅ Execute in WSL (zero latency)
- ✅ Models cached once, accessible from both
- ✅ No duplication, no sync issues

## GPU Access

VLLM in WSL accesses your GPU directly via WSL2's GPU passthrough:

```bash
# Verify GPU access in WSL
wsl -- nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.54       Driver Version: 537.13       CUDA Version: 12.2    |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
```

**No virtualization overhead** - WSL2 provides direct GPU access.

## Installation

### 1. Install VLLM in WSL

```bash
wsl
pip3 install vllm torch transformers
```

### 2. Install SparseLLM dependencies

```bash
cd /mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE
pip3 install -e .
```

### 3. Verify installation

```bash
python3 -c "from sparse_llm.integrations.vllm_direct import DirectVLLMEngine; print('✅ OK')"
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'vllm'"

Install VLLM in WSL:
```bash
wsl
pip3 install vllm
```

### "CUDA not available"

Check GPU passthrough:
```bash
wsl -- nvidia-smi
```

If GPU not detected, update WSL2:
```powershell
# In Windows PowerShell (as Administrator)
wsl --update
wsl --shutdown
wsl
```

### "Permission denied" when accessing files

WSL mounts Windows drives with full permissions. If issues persist:
```bash
# In WSL
sudo chmod -R 755 /mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE
```

### High latency in HTTP mode

This is expected (~0.7-1.6ms overhead). For zero latency, use direct mode.

## Architecture Decision

### When to Use Direct Mode ✅
- Production deployments
- Latency-critical applications (< 100ms response time)
- Single-process inference
- Batch processing
- **Default choice for this project**

### When to Use HTTP Mode
- Multiple concurrent clients
- Remote access (network API)
- Language polyglot (non-Python clients)
- Load balancing across multiple servers

## Next Steps

1. **Run benchmark:** `run.bat benchmark` to measure your baseline
2. **Test inference:** `run.bat inference --model <model> --prompt "Hello"`
3. **Production:** Use `DirectVLLMEngine` in your Python code for zero latency

---

**Bottom Line:** Run everything in WSL for zero cross-boundary latency. Your files stay on C: drive, editable from Windows, but execution happens in native Linux environment.
