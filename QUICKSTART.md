# Quick Start Guide - SparseLLM Server for Claude Code

## Overview

This setup lets you run Claude Code requests through your local MoE inference server.

## Installation & Setup

### 1. Install the package
```bash
pip install -e .
```

### 2. Start the server
```bash
sparse-llm serve --port 8000
```

You'll see:
```
╭────────────────────────────────────────╮
│ SparseLLM MoE Server                   │
│                                        │
│ Base URL: http://localhost:8000        │
│                                        │
│ Add this to Claude Code settings.json: │
│ {"baseURL": "http://localhost:8000/v1"}│
╰────────────────────────────────────────╯

Loading MoE model...
✓ Model loaded successfully!
```

### 3. Configure Claude Code

**Windows:** `%USERPROFILE%\.claude\settings.json`
**Linux/Mac:** `~/.claude/settings.json`

Add:
```json
{
  "baseURL": "http://localhost:8000/v1"
}
```

### 4. Use Claude Code

Now all Claude Code requests go through your local server:

```bash
claude-code chat "Write a Python function to parse JSON"
```

## What Happens

```
┌─────────────┐
│ Claude Code │ → Request
└─────────────┘
       ↓
┌─────────────────────┐
│ Your Server         │
│ localhost:8000/v1   │
│                     │
│ - Tokenize          │
│ - Prefill phase     │
│ - Expert routing    │
│ - Decode phase      │
│ - Cache experts     │
└─────────────────────┘
       ↓
┌─────────────┐
│ Claude Code │ ← Response
└─────────────┘
```

## Testing

### Test the server directly:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sparse-moe",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### View server statistics:
```bash
sparse-llm stats
```

or

```bash
curl http://localhost:8000/stats
```

## Command Reference

### Start server (various options)
```bash
# Basic
sparse-llm serve

# Custom port
sparse-llm serve --port 9000

# CPU mode
sparse-llm serve --device cpu

# Larger cache
sparse-llm serve --cache-size 8

# Full configuration
sparse-llm serve \
  --model deepseek-ai/DeepSeek-V3 \
  --port 8000 \
  --device cuda \
  --cache-size 4 \
  --max-tokens 2048 \
  --dtype float16
```

### Generate directly (without server)
```bash
sparse-llm generate -p "Write a sorting function" --max-tokens 200
```

### View stats
```bash
sparse-llm stats --url http://localhost:8000
```

### Check version
```bash
sparse-llm version
```

## Troubleshooting

### Server won't start
```bash
# Check GPU
nvidia-smi

# Try CPU mode
sparse-llm serve --device cpu

# Check dependencies
python test_server_setup.py
```

### Claude Code not connecting
```bash
# Verify server is running
curl http://localhost:8000/

# Check settings.json exists and has correct URL
cat ~/.claude/settings.json  # Linux/Mac
type %USERPROFILE%\.claude\settings.json  # Windows
```

### Out of memory
```bash
# Reduce cache size
sparse-llm serve --cache-size 2

# Use float16 (if not already)
sparse-llm serve --dtype float16
```

## Performance Tips

1. **Warm up the cache**: First few requests will be slower
2. **Use GPU**: 10-20x faster than CPU
3. **Increase cache size**: More experts cached = faster inference
4. **Monitor memory**: Use `nvidia-smi` to watch VRAM usage

## Next Steps

- Monitor server logs for performance metrics
- Adjust cache size based on your GPU memory
- Try different models by changing `--model` parameter
- Set up as a system service for always-on availability
