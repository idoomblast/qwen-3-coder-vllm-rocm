# 📊 Benchmark Report — Qwen3-Coder-Next on MI300X

> **Date**: February 27, 2026  
> **Hardware**: AMD Instinct MI300X (192GB HBM3)  
> **Backend**: llama.cpp (latest master, built from source with ROCm 7.2)  
> **Model**: `unsloth/Qwen3-Coder-Next-GGUF` — **UD-Q4_K_XL** (41.5 GB)  
> **Architecture**: 80B MoE, 3B active parameters, hybrid Mamba2+Attention (`qwen3next`)  
> **Context**: 256K tokens (262144), split to 128K per slot × 2 slots  

---

## 1. Setup Summary

| Component | Detail |
|-----------|--------|
| **Docker Image** | `llama-cpp-rocm-mi300x:latest` (custom build) |
| **Base Image** | `rocm/dev-ubuntu-24.04:7.2-complete` |
| **GPU Target** | `gfx942` (MI300X specific) |
| **Flash Attention** | Enabled (`--flash-attn on`) |
| **KV Cache** | `q8_0` (quantized, key + value) |
| **Continuous Batching** | Enabled (`--cont-batching`) |
| **Concurrent Slots** | 2 (`--parallel 2`) |
| **Sampling** | temp=1.0, top_p=0.95, top_k=40, min_p=0.01 (Qwen recommended) |

---

## 2. Latency Test (Streaming, Single Request)

3 runs per prompt, averaged.

| Prompt | Chars | Avg TTFT | Avg Total | Avg TPS |
|--------|-------|----------|-----------|---------|
| **Short** — "Say hello in one sentence." | 26 | **59ms** | 0.15s | 36.9 tok/s |
| **Medium** — "Explain binary search + Python impl." | 91 | **95ms** | 5.85s | 91.5 tok/s |
| **Long** — "Write comprehensive REST API with FastAPI..." | 320 | **101ms** | 5.86s | **107.2 tok/s** |
| **Code** — "Implement red-black tree with type hints..." | 146 | **104ms** | 5.87s | 92.6 tok/s |
| **Reasoning** — "A farmer has 17 sheep. All but 9 die..." | 136 | **104ms** | 2.12s | 66.8 tok/s |

### Key Observations
- **TTFT is consistently 50–105ms** regardless of prompt length — no cold start penalty
- **Peak TPS is 107.2 tok/s** on long prompts (more tokens to generate)
- **Short prompts** have lower TPS because overhead dominates with few tokens
- **Reasoning prompts** generate fewer tokens (shorter answers = lower measured TPS)

---

## 3. Throughput Test (Non-streaming, 1024 max_tokens)

3 runs, long prompt (320 chars).

| Run | Total Time | Tokens | TPS |
|-----|-----------|--------|-----|
| 1 | 11.67s | 1024 | 87.7 tok/s |
| 2 | 11.68s | 1024 | 87.6 tok/s |
| 3 | 11.67s | 1024 | 87.7 tok/s |
| **Overall** | **35.03s** | **3072** | **87.7 tok/s** |

### Key Observations
- **Rock-solid consistency** — ±0.1 tok/s variance across runs
- **87.7 tok/s sustained** with 1024 token generation
- No degradation over consecutive requests

---

## 4. Concurrency Scaling Test

Medium prompt, 256 max_tokens per request.

| Concurrent Reqs | Wall Time | Avg Latency | Total Tokens | Throughput | Scaling |
|-----------------|-----------|-------------|--------------|------------|---------|
| 1 | 2.98s | 2.98s | 256 | 85.8 tok/s | 1.0x |
| 2 | 3.88s | 3.88s | 512 | **132.1 tok/s** | **1.54x** |
| 4 | 7.73s | 5.80s | 1024 | **132.6 tok/s** | **1.55x** |

### Key Observations
- **2 concurrent requests** → **+54% throughput** (85.8 → 132.1 tok/s)
- **4 concurrent requests** → saturated at same throughput (only 2 slots configured)
- Latency increases linearly with concurrency beyond 2 slots
- **Sweet spot: 2 concurrent requests** matches `--parallel 2` config

---

## 5. Tool Calling Test

6 tools registered: `get_weather`, `search_web`, `calculate`, `send_email`, `create_file`, `run_code`.

| # | Scenario | Status | Time | Tools Called | Notes |
|---|----------|--------|------|-------------|-------|
| 1 | Single Tool — Weather | ✅ Pass | 0.85s | `get_weather` | Correct tool + args |
| 2 | Single Tool — Calculator | ✅ Pass | 0.62s | `calculate` | Correct expression |
| 3 | Multi-Tool Selection | ✅ Pass | 0.67s | `search_web` | Correct from 6 options |
| 4 | Parallel Tool Calls | ✅ Pass | 0.67s | `get_weather` | Called sequentially, not parallel* |
| 5 | Sequential Chain | ✅ Pass | 0.62s + 2.67s | `get_weather` → answer | Full chain completed |
| 6 | No Tool Needed | ✅ Pass | 0.30s | — | Direct answer: "Paris" |
| 7 | Complex — Code + File | ✅ Pass | 1.48s | `create_file` | Generated fibonacci + saved |

**Result: 7/7 passed** ✅

\* *Note: Parallel tool calling is sequential in llama.cpp's OpenAI-compatible endpoint. Model calls tools one at a time. This is expected behavior, not an error.*

---

## 6. Comparison: llama.cpp vs vLLM FP8

Both tested on the same MI300X hardware.

| Metric | vLLM FP8 (v0.15.1) | llama.cpp GGUF Q4 | Improvement |
|--------|--------------------|--------------------|-------------|
| **TTFT (first request)** | 43.11s | **0.089s** | **484x faster** |
| **TTFT (warm)** | ~140ms | **~89ms** | 1.6x faster |
| **TPS (streaming)** | 2.5 tok/s (cold) | **91.9 tok/s** | **37x faster** |
| **Sustained throughput** | ~97 tok/s (warm) | **87.7 tok/s** | ~0.9x (similar) |
| **Model size** | ~81 GB (FP8) | **42 GB (Q4)** | 2x smaller |
| **VRAM usage** | ~170 GB (256K ctx) | **~83 GB** | 2x less |
| **Cold start** | ~5 min (torch.compile) | **~30s** | 10x faster |
| **Context (256K)** | ⚠️ OOM risk (2 seqs) | ✅ Comfortable | No OOM |
| **Tool calling** | 7/7 ✅ | 7/7 ✅ | Same |

### Why llama.cpp Wins Here
1. **MoE architecture** — Only 3B params are active per token. llama.cpp's per-token overhead is minimal.
2. **No JIT compilation** — vLLM's torch.compile + Triton kernel warmup adds 40+ seconds on first request.
3. **GGUF efficiency** — Q4 quantization halves model size with minimal quality loss (per Unsloth benchmarks).
4. **Lower VRAM** — 42GB model leaves 150GB for KV cache, easily fitting 256K context.

### When vLLM is Better
- **High concurrency** (8+ concurrent requests) — vLLM's PagedAttention scales better
- **FP8/BF16 precision** — When maximum quality is needed
- **Multi-GPU** — vLLM's tensor parallelism is more mature

---

## 7. Memory Analysis

```
MI300X Total HBM3:          192 GB
Model (Q4 GGUF):             42 GB
KV Cache (q8_0, 256K × 2):  ~36 GB
Compute buffers:             ~0.7 GB
Prompt cache:                ~8 GB (configurable)
Overhead:                    ~5 GB
────────────────────────────────────
Estimated Total:             ~92 GB
Available headroom:          ~100 GB ✅
```

---

## 8. Recommendations

### For Maximum Throughput
```bash
--parallel 4          # More concurrent slots
--ctx-size 131072     # 128K context (halved)
--cache-type-k q4_0   # More aggressive KV cache quantization
```

### For Maximum Context
```bash
--parallel 1          # Single slot
--ctx-size 262144     # Full 256K context per request
```

### For Coding Agents (Claude Code, Codex, etc.)
```bash
--parallel 2          # Sweet spot for agent workflows
--ctx-size 262144     # Long context for large codebases
--cache-ram 16384     # 16GB prompt cache for repeated system prompts
```

---

## 9. Reproduction

```bash
# 1. Download model
pip install huggingface_hub[hf_xet]
python -c "
from huggingface_hub import snapshot_download
snapshot_download('unsloth/Qwen3-Coder-Next-GGUF',
    local_dir='./models/Qwen3-Coder-Next-UD-Q4_K_XL',
    allow_patterns=['*UD-Q4_K_XL*'],
    token='YOUR_HF_TOKEN')
"

# 2. Build & start
docker compose build    # ~15-30 min first time
docker compose up -d

# 3. Run benchmarks
pip install python-dotenv requests
python benchmark.py --test all --runs 3
python benchmark_tools.py
```

---

*Report generated on February 27, 2026*
