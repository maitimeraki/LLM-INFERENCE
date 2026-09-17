# SparseLLM Server Setup

This document explains how to use the SparseLLM MoE inference server with Claude Code.

## Architecture

```
User → Claude Code → SparseLLM Server (http://localhost:8000/v1)
                            ↓
                    Custom PyTorch MoE Engine
                            ↓
                    Expert Cache + Router
                            ↓
                    Generated Response
```

## Installation

1. **Install the package:**
```bash
pip install -e .
```

This will install:
- Core dependencies (torch, transformers, etc.)
- Server dependencies (fastapi, uvicorn)
- CLI tool (`sparse-llm` command)

## Usage

### Starting the Server

**Basic usage:**
```bash
sparse-llm serve
```

**With options:**
```bash
sparse-llm serve --port 8000 --cache-size 4 --device cuda
```

**Available options:**
- `--model, -m`: Model path or HuggingFace ID (default: deepseek-ai/DeepSeek-V3)
- `--host`: Server host (default: 0.0.0.0)
- `--port, -p`: Server port (default: 8000)
- `--device, -d`: Device (cuda/cpu/auto)
- `--cache-size, -c`: Number of experts to cache (default: 4)
- `--max-tokens`: Default max tokens to generate (default: 2048)
- `--dtype`: Model data type (float16/bfloat16/float32)

### Server Output

When the server starts, you'll see:
```
╭────────────────────────────────────────╮
│ SparseLLM MoE Server                   │
│                                        │
│ Base URL: http://localhost:8000        │
│                                        │
│ Configuration:                         │
│   Model: deepseek-ai/DeepSeek-V3      │
│   Device: cuda                         │
│   Expert cache: 4 experts              │
│   Max tokens: 2048                     │
│                                        │
│ Add this to Claude Code settings.json: │
│ {"baseURL": "http://localhost:8000/v1"}│
╰────────────────────────────────────────╯
```

Copy the base URL for Claude Code configuration.

## Configuring Claude Code

### Option 1: settings.json (Recommended)

Add the baseURL to your Claude Code settings:

**Location:**
- Linux/Mac: `~/.claude/settings.json`
- Windows: `%USERPROFILE%\.claude\settings.json`

**Content:**
```json
{
  "baseURL": "http://localhost:8000/v1"
}
```

### Option 2: Environment Variable

```bash
export ANTHROPIC_API_BASE="http://localhost:8000/v1"
```

## How It Works

1. **Server starts** and loads the MoE model into memory
2. **Expert cache** is initialized with the specified number of hot experts
3. **Claude Code** sends requests to `/v1/chat/completions`
4. **Server processes** the request using the MoE inference engine:
   - Tokenizes input
   - Runs prefill phase (batch processes prompt)
   - Analyzes expert usage and preloads hot experts
   - Runs decode phase (generates tokens one by one)
   - Uses expert cache for fast inference
5. **Response returned** in OpenAI-compatible format

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/
```

Response:
```json
{
  "status": "ready",
  "model": "sparse-moe",
  "engine": "custom-pytorch",
  "version": "0.1.0"
}
```

### List Models
```bash
curl http://localhost:8000/v1/models
```

### Chat Completions (Main Endpoint)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sparse-moe",
    "messages": [
      {"role": "user", "content": "Write a Python function to sort a list"}
    ],
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

### Server Statistics
```bash
curl http://localhost:8000/stats
```

Response:
```json
{
  "total_requests": 42,
  "total_tokens_generated": 15234,
  "average_tokens_per_second": 27.3,
  "memory_usage_gb": 12.5,
  "cache_stats": {
    "hit_rate": 0.82,
    "size": 4
  }
}
```

## CLI Commands

### Serve (Start Server)
```bash
sparse-llm serve --port 8000
```

### Generate (CLI Mode)
Generate text directly without server:
```bash
sparse-llm generate -p "Write a function to parse JSON" --max-tokens 200
```

### Stats (View Server Stats)
```bash
sparse-llm stats --url http://localhost:8000
```

### Version
```bash
sparse-llm version
```

## Testing the Setup

Run the verification script:
```bash
python test_server_setup.py
```

This checks:
- All imports work
- Dependencies are installed
- Configuration can be created
- FastAPI routes are registered
- CLI commands are available

## Example Workflow

1. **Start the server:**
```bash
sparse-llm serve --port 8000 --cache-size 4
```

2. **Configure Claude Code:**
Edit `~/.claude/settings.json`:
```json
{
  "baseURL": "http://localhost:8000/v1"
}
```

3. **Use Claude Code:**
```bash
claude-code chat "Generate a FastAPI endpoint for user authentication"
```

4. **Claude Code will:**
   - Send the request to your local server
   - Your MoE engine processes it
   - Response is streamed back to Claude Code
   - All processing happens locally on your hardware

## Performance Expectations

With a typical MoE model (e.g., DeepSeek-V3):
- **Prefill:** 200-500 tokens/sec
- **Decode:** 20-30 tokens/sec
- **Cache hit rate:** 80%+ after warmup
- **Memory:** ~12-16GB GPU VRAM (depends on cache size)

## Troubleshooting

### Server won't start
- Check GPU is available: `nvidia-smi`
- Try CPU mode: `sparse-llm serve --device cpu`
- Check dependencies: `python test_server_setup.py`

### Claude Code not connecting
- Verify server is running: `curl http://localhost:8000/`
- Check settings.json has correct URL
- Try restarting Claude Code

### Out of memory
- Reduce cache size: `--cache-size 2`
- Use CPU offload: The engine automatically enables this
- Use smaller dtype: `--dtype float16`

### Slow generation
- Check GPU utilization: `nvidia-smi`
- Increase cache size: `--cache-size 8`
- Verify CUDA is being used (check startup logs)

## Architecture Details

### Components

1. **serve.py**: FastAPI server with OpenAI-compatible API
2. **cli.py**: Command-line interface
3. **moe_inference_engine.py**: Core MoE inference logic
4. **expert_cache_manager.py**: Expert caching with prefetching
5. **router_calculator.py**: Expert routing decisions
6. **attention_engine.py**: Attention mechanism with KV cache

### Request Flow

```
1. Claude Code sends request
   ↓
2. FastAPI endpoint receives it
   ↓
3. Convert messages to prompt format
   ↓
4. MoE Engine: Prefill phase
   - Tokenize prompt
   - Run attention (create KV cache)
   - Batch compute router decisions
   - Preload hot experts
   ↓
5. MoE Engine: Decode phase
   - Generate tokens one by one
   - Use expert cache for fast inference
   - Update KV cache
   ↓
6. Convert output to OpenAI format
   ↓
7. Return response to Claude Code
```

## Next Steps

- Add streaming support for real-time token generation
- Implement request batching for concurrent requests
- Add metrics dashboard (Prometheus/Grafana)
- Support multiple model backends
- Add model hot-swapping without restart
