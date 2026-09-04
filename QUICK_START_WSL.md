# Quick Start: Zero-Latency WSL Setup

## ⚠️ Common Issue: Network Error?

If you get `[Errno 101] Network is unreachable` when running from WSL:

**→ See [WSL_NETWORK_FIX.md](WSL_NETWORK_FIX.md) for step-by-step fix**

**Quick fix:** Use your downloaded models locally instead of downloading:

```bash
# Find your models on D: drive
find /mnt/d -name "*Mixtral*" -type d 2>/dev/null

# Or run helper:
bash scripts/find_models.sh

# Use full path with --local-files-only flag
python3 main.py --model /mnt/d/path/to/model/snapshots/<hash> --prompt "Hello" --local-files-only
```

---

## TL;DR (If Network Works)

```bash
# 1. Install VLLM in WSL
wsl
pip3 install vllm torch transformers

# 2. Run inference (zero latency)
run.bat inference --model mistralai/Mixtral-8x7B-Instruct-v0.1 --prompt "Hello"

# 3. Benchmark latency
run.bat benchmark
```

Done! Your code runs in WSL with zero cross-boundary overhead.

---

## What This Setup Does

✅ **Code:** Stays on C: drive (edit in Windows)  
✅ **Execution:** Runs in WSL (native Linux performance)  
✅ **VLLM:** Direct integration (zero serialization overhead)  
✅ **GPU:** Native access (no virtualization penalty)  
✅ **Models:** Cached once, shared between both

---

## Usage Modes

### Mode 1: Direct Inference ⚡ (Recommended)

```bash
# From Windows
run.bat inference --model <model-id> --prompt "Your prompt"

# From WSL
wsl
cd /mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE
python3 main.py --model <model-id> --prompt "Your prompt"
```

**Latency:** ~0ms overhead

### Mode 2: Python API 🐍

```python
# Create file: test_inference.py
from sparse_llm.integrations.vllm_direct import DirectVLLMEngine

engine = DirectVLLMEngine(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
response = engine.generate("What is AI?")
print(response)
```

```bash
# Run in WSL
wsl python3 test_inference.py
```

**Latency:** ~0ms overhead

### Mode 3: HTTP Server 🌐

```bash
# Start server (in WSL)
run.bat server --model mistralai/Mixtral-8x7B-Instruct-v0.1 --port 8000

# Use from anywhere
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistralai/Mixtral-8x7B-Instruct-v0.1", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Latency:** ~0.7-1.6ms overhead per request

---

## File Locations

| What | Windows Path | WSL Path |
|------|-------------|----------|
| Your code | `C:\Users\Anupam\Desktop\LLM\LOCAL-INFERENCE` | `/mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE` |
| Models | `C:\Users\Anupam\.cache\huggingface` | `/mnt/c/Users/Anupam/.cache/huggingface` |

**No duplication needed** - WSL automatically mounts C: drive at `/mnt/c/`

---

## Verify Setup

### 1. Check WSL

```bash
wsl -- uname -a
# Expected: Linux ... x86_64 GNU/Linux
```

### 2. Check Python

```bash
wsl -- python3 --version
# Expected: Python 3.8+ 
```

### 3. Check VLLM

```bash
wsl -- python3 -c "import vllm; print(vllm.__version__)"
# Expected: 0.x.x (any version)
```

### 4. Check GPU

```bash
wsl -- nvidia-smi
# Expected: GPU list with CUDA version
```

### 5. Check Project

```bash
wsl -- ls /mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE
# Expected: main.py, sparse_llm/, etc.
```

All checks pass? You're ready!

---

## Troubleshooting

### ⚠️ "Network is unreachable" (Most Common Issue)

**Problem:** `[Errno 101] Network is unreachable` when running from WSL

**Root Cause:** WSL networking issue - cannot reach huggingface.co

**Solution (Choose ONE):**

#### Option A: Use Local Models (Recommended for Your Setup) ✅

Since you already downloaded models to D: drive, just point to them:

```bash
# From WSL (you're already here)
cd /mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE

# Use local model path instead of downloading
python3 main.py \
  --model /mnt/d/path/to/your/Mixtral-8x7B-Instruct-v0.1 \
  --prompt "Hello" \
  --local-files-only
```

**Find your model location:**
```bash
# Check Windows HF_HOME location
wsl -- cmd.exe /c "echo %HF_HOME%"

# If HF_HOME is D:\models (example), use:
# /mnt/d/models/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/<hash>
```

**Quick finder script:**
```bash
# Find Mixtral model
find /mnt/d -name "*Mixtral*" -type d 2>/dev/null | head -5
```

#### Option B: Fix WSL Networking (If You Need Downloads)

**Step 1: Test connectivity**
```bash
wsl -- curl -I https://huggingface.co
# Should return HTTP 200, not timeout
```

**Step 2: Fix networking**
```powershell
# Run in Windows PowerShell (as Administrator)
wsl --shutdown
netsh winsock reset
netsh int ip reset all
netsh winhttp reset proxy
ipconfig /flushdns

# Restart WSL
wsl
```

**Step 3: If still broken, update DNS in WSL**
```bash
# In WSL
sudo rm /etc/resolv.conf
sudo bash -c 'echo "nameserver 8.8.8.8" > /etc/resolv.conf'
sudo bash -c 'echo "nameserver 8.8.4.4" >> /etc/resolv.conf'
sudo chattr +i /etc/resolv.conf  # Prevent auto-reset
```

**Step 4: Test again**
```bash
curl -I https://huggingface.co
# Should work now
```

#### Option C: Set HF_HOME to Use Cached Models

**If models are on D: drive, tell Python to look there:**

```bash
# In WSL, before running
export HF_HOME=/mnt/d/your/huggingface/cache
export TRANSFORMERS_CACHE=/mnt/d/your/huggingface/cache

# Then run
python3 main.py \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --prompt "Hello" \
  --local-files-only
```

**Or create .env file:**
```bash
# From Windows, create .env in project root
echo HF_HOME=/mnt/d/your/huggingface/cache > .env
echo TRANSFORMERS_CACHE=/mnt/d/your/huggingface/cache >> .env
```

---

### "vllm not found"
```bash
wsl
pip3 install vllm torch transformers
```

### "CUDA not available"
```bash
# Update WSL
wsl --update
wsl --shutdown

# Verify GPU
wsl -- nvidia-smi
```

### "File not found"
```bash
# Check mount
wsl ls /mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE

# If empty, check WSL config
wsl cat /etc/wsl.conf
# Should have: [automount] enabled = true
```

### Still stuck?
See full guides:
- [ZERO_LATENCY_GUIDE.md](ZERO_LATENCY_GUIDE.md) - Complete usage guide
- [WSL_VLLM_ARCHITECTURE.md](WSL_VLLM_ARCHITECTURE.md) - Architecture details

---

## Why This Works

1. **WSL mounts C: drive** - Your files are accessible as `/mnt/c/...`
2. **No network overhead** - Direct function calls (same process)
3. **No serialization** - No JSON encoding/decoding
4. **Native GPU** - WSL2 GPU passthrough (zero virtualization)
5. **Single Python environment** - One set of dependencies in WSL

**Result:** Zero latency between your code and VLLM inference.

---

## Next Steps

1. ✅ **Verify:** `run.bat benchmark`
2. ✅ **Test:** `run.bat inference --model <your-model> --prompt "test"`
3. ✅ **Integrate:** Use `DirectVLLMEngine` in your Python code
4. ✅ **Deploy:** Run production workloads in WSL for best performance

---

**Need help?** See [ZERO_LATENCY_GUIDE.md](ZERO_LATENCY_GUIDE.md) for detailed instructions.
