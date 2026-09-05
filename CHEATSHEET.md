# SparseLLM + WSL Cheat Sheet

## Your Setup
- **Code:** `C:\Users\Anupam\Desktop\LLM\LOCAL-INFERENCE`
- **Models:** `D:\` (downloaded via HF_HOME)
- **Virtual Env:** WSL `env` (vllm installed)
- **Execution:** All in WSL for best performance

---

## Quick Commands

### Find Your Downloaded Models
```bash
# From Windows:
find_models.bat

# From WSL:
bash scripts/find_models.sh

# Manual search:
find /mnt/d -name "*Mixtral*" -type d 2>/dev/null
```

### Run Inference (Local Models)
```bash
# In WSL (with your venv activated):
cd /mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE
source env/bin/activate  # If not already activated

python3 main.py \
  --model /mnt/d/path/to/model/snapshots/<hash> \
  --prompt "Your prompt here" \
  --local-files-only \
  --max-new-tokens 100
```

### Run Inference (Network Working)
```bash
python3 main.py \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --prompt "Your prompt here" \
  --max-new-tokens 100
```

### Set HF Cache Path (Permanent)
```bash
# In WSL:
echo 'export HF_HOME=/mnt/d/your/cache/path' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/mnt/d/your/cache/path' >> ~/.bashrc
source ~/.bashrc
```

---

## Common Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `--model` | Model ID or path | `mistralai/Mixtral-8x7B-Instruct-v0.1` |
| `--prompt` | Your input text | `"Explain quantum computing"` |
| `--max-new-tokens` | Output length | `100` |
| `--temperature` | Sampling temp (0=greedy) | `0.7` |
| `--local-files-only` | **Never download** | (no value, just flag) |
| `--device` | CPU/CUDA/auto | `cuda` |
| `--dtype` | Precision | `float16` or `bfloat16` |
| `--json` | JSON output | (no value, just flag) |

---

## Troubleshooting Flowchart

```
┌─────────────────────────────────┐
│ Error: Network unreachable?     │
└────────────┬────────────────────┘
             │
             ↓
┌─────────────────────────────────┐
│ Do you need to download NEW     │
│ models?                          │
└────────────┬────────────────────┘
             │
      ┌──────┴──────┐
      │             │
     NO            YES
      │             │
      ↓             ↓
   Use local    Fix WSL
   models       networking
   (Option A)   (Option B)
      │             │
      ↓             ↓
   find_models   See WSL_NETWORK_FIX.md
   + use path    Section: Fix Networking
   + add flag:
   --local-files-only
```

---

## Path Translation (Windows ↔ WSL)

| Windows Path | WSL Path |
|--------------|----------|
| `C:\Users\Anupam\...` | `/mnt/c/Users/Anupam/...` |
| `D:\models\...` | `/mnt/d/models/...` |
| `%USERPROFILE%` | `/mnt/c/Users/Anupam` |

**Rule:** `X:\path\to\file` → `/mnt/x/path/to/file` (lowercase drive letter)

---

## File Locations Reference

```
C:\Users\Anupam\Desktop\LLM\LOCAL-INFERENCE\
├── 📄 WSL_NETWORK_FIX.md        ← Read this first for network errors
├── 📄 QUICK_START_WSL.md        ← Quick reference guide
├── 📄 CHEATSHEET.md             ← This file
├── 🚀 find_models.bat           ← Windows: Find downloaded models
├── 🚀 run.bat                   ← Windows: Run inference
├── 🐍 main.py                   ← Entry point
├── scripts/
│   ├── 🔍 find_models.sh        ← WSL: Find models
│   └── benchmark_latency.py     ← Performance testing
└── docs/
    ├── ZERO_LATENCY_GUIDE.md    ← Complete guide
    ├── WSL_SETUP_SUMMARY.md     ← Setup overview
    └── WSL_VLLM_ARCHITECTURE.md ← Technical details
```

---

## Typical Workflow

### First Time Setup
```bash
# 1. Activate venv (in WSL)
cd /mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE
source env/bin/activate

# 2. Find models
bash scripts/find_models.sh

# 3. Set HF cache (if on D:)
export HF_HOME=/mnt/d/your/path
export TRANSFORMERS_CACHE=/mnt/d/your/path

# 4. Test
python3 main.py \
  --model /mnt/d/.../snapshots/<hash> \
  --prompt "Hello" \
  --local-files-only
```

### Daily Usage
```bash
# Activate venv
source env/bin/activate

# Run inference
python3 main.py --model <model> --prompt "..." --local-files-only
```

---

## Performance Tips

1. **Always use `--local-files-only`** when models are cached
   - Avoids network lookup
   - Faster startup

2. **Use full snapshot paths** for maximum speed
   - Direct file access
   - No model resolution overhead

3. **Set environment variables** for cleaner commands
   ```bash
   export HF_HOME=/mnt/d/models
   export TRANSFORMERS_CACHE=/mnt/d/models
   ```

4. **Keep code on C:, models on D:** (or separate drives)
   - Better I/O parallelism
   - Avoid drive congestion

---

## Need Help?

| Issue | Document |
|-------|----------|
| Network errors | [WSL_NETWORK_FIX.md](WSL_NETWORK_FIX.md) |
| Setup guide | [QUICK_START_WSL.md](QUICK_START_WSL.md) |
| Performance | [ZERO_LATENCY_GUIDE.md](ZERO_LATENCY_GUIDE.md) |
| Architecture | [WSL_VLLM_ARCHITECTURE.md](WSL_VLLM_ARCHITECTURE.md) |

---

## Quick Diagnostics

```bash
# Check WSL working
wsl -- uname -a

# Check Python
wsl -- python3 --version

# Check venv location
wsl -- which python3  # Should show /path/to/env/bin/python3

# Check packages
wsl -- python3 -c "import torch; print(torch.__version__)"
wsl -- python3 -c "import transformers; print(transformers.__version__)"

# Check GPU
wsl -- nvidia-smi

# Check network
wsl -- curl -I https://huggingface.co

# Check project files
wsl -- ls /mnt/c/Users/Anupam/Desktop/LLM/LOCAL-INFERENCE

# Check models
wsl -- ls /mnt/d  # Adjust to your model drive
```

---

**Last Updated:** 2026-09-04  
**Your Config:** Windows 11 + WSL + D: drive models + C: drive code
