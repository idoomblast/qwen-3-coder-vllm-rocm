# 🚀 Qwen3-Coder-Next on MI300X

Deploy **Qwen3-Coder-Next** (80B MoE, 3B active params) on AMD Instinct MI300X via **llama.cpp** (built from source with ROCm) + Docker Compose, with Cloudflare Tunnel ingress.

Uses **Unsloth UD-Q4_K_XL** GGUF quantization — best quality/performance ratio per [Unsloth benchmarks](https://unsloth.ai/docs/models/qwen3-coder-next).

## 📊 Benchmark Results

> Single MI300X (192GB HBM3) — llama.cpp (latest, ROCm 7.2) — Flash Attention — Unsloth UD-Q4_K_XL GGUF

### Latency (Streaming)

| Prompt Type | TTFT | Total Time | TPS |
|-------------|------|------------|-----|
| **Short** (26 chars) | 59ms | 0.15s | 36.9 tok/s |
| **Medium** (91 chars) | 95ms | 5.85s | 91.5 tok/s |
| **Long** (320 chars) | 101ms | 5.86s | **107.2 tok/s** |
| **Code** (146 chars) | 104ms | 5.87s | 92.6 tok/s |
| **Reasoning** (136 chars) | 104ms | 2.12s | 66.8 tok/s |

### Throughput & Concurrency

| Metric | Value |
|--------|-------|
| **TTFT** | ~50–100ms |
| **Peak TPS (streaming)** | **107.2 tok/s** |
| **Sustained throughput** | **87.7 tok/s** |
| **Concurrent throughput (2 reqs)** | **132.1 tok/s** |
| **Concurrent throughput (4 reqs)** | **132.6 tok/s** |
| **Context window** | 256K tokens (128K per slot × 2) |
| **Tool calling** | 7/7 tests passed ✅ |
| **Model size (GGUF)** | 42 GB |
| **VRAM usage** | ~83 GB / 192 GB |

### Tool Calling

| Scenario | Status | Time |
|----------|--------|------|
| Single Tool — Weather | ✅ | 0.85s |
| Single Tool — Calculator | ✅ | 0.62s |
| Multi-Tool Selection (6 tools) | ✅ | 0.67s |
| Parallel Tool Calls | ✅ | 0.67s |
| Sequential Chain (tool → answer) | ✅ | 3.29s |
| No Tool Needed (direct answer) | ✅ | 0.30s |
| Complex — Code + File | ✅ | 1.48s |

## 🏗️ Architecture

```
Internet → Cloudflare Tunnel → cloudflared → llama-server:8080 (mapped to :8000)
                                                    ↓
                                           MI300X (192GB HBM3)
                                           Qwen3-Coder-Next UD-Q4_K_XL
                                           llama.cpp (ROCm 7.2, gfx942)
```

### Services

| Service | Image | Purpose |
|---------|-------|---------|
| `llama-server` | `llama-cpp-rocm-mi300x:latest` (custom build) | OpenAI-compatible inference server |
| `cloudflared` | `cloudflare/cloudflared:latest` | Cloudflare Tunnel ingress |

## ⚡ Quick Start

### 1. Clone & Configure

```bash
git clone <repo-url>
cd vllm
cp .env.example .env
```

Edit `.env` with your credentials:

```dotenv
HF_TOKEN=hf_your-token-here
VLLM_API_KEY=sk-your-api-key-here
CLOUDFLARE_TUNNEL_TOKEN=your-cloudflared-token
```

### 2. Download Model

```bash
pip install huggingface_hub[hf_xet]

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='unsloth/Qwen3-Coder-Next-GGUF',
    local_dir='./models/Qwen3-Coder-Next-UD-Q4_K_XL',
    allow_patterns=['*UD-Q4_K_XL*'],
    token='YOUR_HF_TOKEN',
)
"
```

### 3. Build & Start

```bash
# Build llama.cpp from source with ROCm (first time ~15-30 min)
docker compose build

# Start services
docker compose up -d
```

> ⏳ First build takes ~15-30 minutes (compiling ROCm HIP kernels for MI300X).
> Subsequent starts take ~30 seconds (model loading from disk).

### 4. Verify

```bash
# Health check
curl http://localhost:8000/health

# Quick test
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-coder-next","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

## 📡 API Usage

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-next",
    "messages": [
      {"role": "user", "content": "Write a Python quicksort function"}
    ],
    "max_tokens": 1024
  }'
```

### Tool Calling

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-next",
    "messages": [
      {"role": "user", "content": "What is the weather in Jakarta?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

### OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-your-api-key",
)

response = client.chat.completions.create(
    model="qwen3-coder-next",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## 🧪 Benchmarks

### Performance Benchmark

```bash
# Quick smoke test
python3 benchmark.py --test smoke

# Full benchmark (latency + throughput + concurrency)
python3 benchmark.py --test all --runs 3
```

### Tool Calling Benchmark

```bash
python3 benchmark_tools.py
```

### SWE-bench

```bash
# Install dependencies
pip install swebench datasets

# Inference (generate patches) — 5 instances test
python3 swe_bench_run.py infer --subset verified --max-instances 5

# Full SWE-bench Lite (300 instances)
python3 swe_bench_run.py infer --subset lite

# Evaluate (Docker-based, needs ~120GB disk)
python3 swe_bench_run.py eval --predictions results/predictions_lite.jsonl --subset lite
```

## 🔧 Configuration

### Key llama.cpp Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--ctx-size` | 262144 | Full 256K context window |
| `--n-gpu-layers` | 99 | Offload all layers to GPU |
| `--flash-attn` | on | Flash Attention for long context |
| `--parallel` | 2 | 2 concurrent request slots |
| `--cont-batching` | — | Continuous batching for throughput |
| `--cache-type-k` | q8_0 | Quantized KV cache (key) |
| `--cache-type-v` | q8_0 | Quantized KV cache (value) |
| `--alias` | qwen3-coder-next | Custom API model name |

### Qwen-Recommended Sampling Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--temp` | 1.0 | Qwen recommended |
| `--top-p` | 0.95 | Qwen recommended |
| `--top-k` | 40 | Qwen recommended |
| `--min-p` | 0.01 | llama.cpp default is 0.05, Qwen recommends 0.01 |
| `--repeat-penalty` | 1.0 | Disabled as recommended |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `HIP_VISIBLE_DEVICES` | `0` — Select GPU device |
| `HSA_OVERRIDE_GFX_VERSION` | `9.4.2` — MI300X GFX target |
| `VLLM_API_KEY` | API key for authentication |
| `CLOUDFLARE_TUNNEL_TOKEN` | Cloudflare Tunnel token |

### Tuning Tips

```bash
# Lower context for less VRAM usage
--ctx-size 131072    # 128K instead of 256K

# More concurrent users
--parallel 4         # 4 slots (context split across slots)

# CPU-only inference (no GPU)
--n-gpu-layers 0

# Increase prompt cache size
--cache-ram 16384    # 16 GB prompt cache
```

## 📁 Project Structure

```
.
├── docker-compose.yaml    # llama-server + Cloudflared services
├── Dockerfile.rocm        # Multi-stage build: llama.cpp latest + ROCm 7.2
├── .env.example           # Environment template
├── .gitignore
├── benchmark.py           # Performance benchmark (latency/throughput/concurrency)
├── benchmark_tools.py     # Tool calling benchmark
├── swe_bench_run.py       # SWE-bench runner (inference + evaluation)
├── BENCHMARK_REPORT.md    # Detailed benchmark report
└── models/                # GGUF model files (~42GB, gitignored)
    └── Qwen3-Coder-Next-UD-Q4_K_XL/
        └── Qwen3-Coder-Next-UD-Q4_K_XL.gguf
```

## 📋 Requirements

- **GPU**: AMD Instinct MI300X (192GB HBM3)
- **Docker**: with ROCm support (`/dev/kfd`, `/dev/dri`)
- **Disk**: ~60GB free for model + Docker image
- **Build time**: ~15-30 min first build (ROCm kernel compilation)

## 📝 Notes

- **Why llama.cpp over vLLM?** — For this MoE model (3B active params), llama.cpp is significantly faster. TTFT dropped from ~43s (vLLM cold) to ~89ms, and TPS increased from ~2.5 to ~107 tok/s.
- **Why UD-Q4_K_XL?** — Unsloth Dynamic quantization outperforms standard Q4_K_M in benchmarks (Aider Polyglot, LiveCodeBench). 42GB fits comfortably in 192GB HBM3 with room for 256K KV cache.
- **Why custom Dockerfile?** — Official `rocm/llama.cpp` Docker images only go up to build b6652 which doesn't support the `qwen3next` architecture. We build from latest source.
- **Context splitting** — With `--parallel 2` and `--ctx-size 262144`, llama.cpp splits to 128K per slot. Still plenty for most coding tasks.
- Model files are downloaded from HuggingFace and stored in `./models/` (gitignored).

## License

MIT
