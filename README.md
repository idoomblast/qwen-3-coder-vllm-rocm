# 🚀 Qwen3-Coder-Next on MI300X

Deploy **Qwen3-Coder-Next** (80B MoE, FP8 Dynamic) on AMD Instinct MI300X via vLLM + Docker Compose, with Cloudflare Tunnel ingress.

## 📊 Benchmark Results

> Single MI300X (192GB HBM3) — vLLM v0.15.1 ROCm — Triton Flash Attention

| Metric | Value |
|--------|-------|
| **TTFT** | ~140ms |
| **Single-user TPS** | ~97 tok/s |
| **Peak throughput (4 concurrent)** | 292 tok/s |
| **Context window** | 256K tokens |
| **Tool calling** | 7/7 tests passed ✅ |
| **Model memory** | 80.96 GiB |
| **KV cache available** | 75.52 GiB (FP8) |
| **Max concurrent (256K)** | 24.82x |

## 🏗️ Architecture

```
Internet → Cloudflare Tunnel → cloudflared → vllm:8000
                                               ↓
                                      MI300X (192GB HBM3)
                                      Qwen3-Coder-Next FP8
```

### Services

| Service | Image | Purpose |
|---------|-------|---------|
| `vllm` | `vllm/vllm-openai-rocm:v0.15.1` | OpenAI-compatible inference server |
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

### 2. Start Services

```bash
docker compose up -d
```

> ⏳ First start takes ~5 minutes (model download ~80GB + torch.compile warmup).  
> Subsequent starts take ~2-3 minutes (model cached in `./models/`).

### 3. Verify

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
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

### Key vLLM Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--max-model-len` | 262144 | Full 256K context window |
| `--kv-cache-dtype` | fp8 | 2x KV cache capacity |
| `--gpu-memory-utilization` | 0.95 | Use 95% of GPU memory |
| `--max-num-seqs` | 4 | Max concurrent requests |
| `--enable-chunked-prefill` | - | Overlap prefill & decode |
| `--tool-call-parser` | qwen3_coder | Native tool calling support |
| `--served-model-name` | qwen3-coder-next | Custom API model name |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `VLLM_ROCM_USE_AITER` | `0` — Disabled (JIT compile fails in container) |
| `VLLM_USE_TRITON_FLASH_ATTN` | `1` — Triton Flash Attention backend |
| `HF_ENDPOINT` | `https://hf-mirror.com` — Mirror for blocked networks |

### Tuning Tips

```yaml
# Lower context for faster response
--max-model-len=131072    # 128K instead of 256K

# More concurrent users, less context
--max-num-seqs=8
--max-model-len=65536     # 64K

# Disable compilation for faster cold start
--enforce-eager
```

## 📁 Project Structure

```
.
├── docker-compose.yaml    # vLLM + Cloudflared services
├── .env.example           # Environment template
├── .gitignore
├── benchmark.py           # Performance benchmark (latency/throughput/concurrency)
├── benchmark_tools.py     # Tool calling benchmark
├── swe_bench_run.py       # SWE-bench runner (inference + evaluation)
└── models/                # Auto-downloaded model cache (~80GB, gitignored)
```

## 📋 Requirements

- **GPU**: AMD Instinct MI300X (192GB HBM3)
- **Docker**: with ROCm support (`/dev/kfd`, `/dev/dri`)
- **Disk**: ~100GB free for model cache
- **Network**: HuggingFace access (or mirror)

## 📝 Notes

- First request after cold start has ~13-46s TTFT due to Triton kernel JIT compilation. Subsequent requests are ~140ms.
- AITER backend is disabled because JIT compilation of paged attention kernels fails inside the Docker container. Triton backend is used instead with near-equivalent performance.
- Model auto-downloads from HuggingFace on first start and caches to `./models/` volume.
- The `hf-mirror.com` endpoint is used to bypass CloudFront IP blocks on some server networks.

## License

MIT
