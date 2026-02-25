#!/usr/bin/env python3
"""
vLLM Benchmark Script for Qwen3-Coder-Next-FP8-Dynamic on MI300X
=================================================================
Tests: Latency (TTFT + TPS), Throughput, Concurrency Scaling
"""

import os
import requests
import json
import time
import statistics
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Config ───────────────────────────────────────────────────────
BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("VLLM_API_KEY", "")
MODEL = os.getenv("VLLM_MODEL", "qwen3-coder-next")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# ─── Prompts untuk benchmark ─────────────────────────────────────
PROMPTS = {
    "short": "Say hello in one sentence.",
    "medium": "Explain how a binary search algorithm works. Include a Python implementation with comments.",
    "long": "Write a comprehensive REST API in Python using FastAPI that includes: user authentication with JWT tokens, CRUD operations for a blog post system, input validation with Pydantic models, error handling middleware, database integration with SQLAlchemy, and rate limiting. Include detailed comments explaining each section.",
    "code": "Write a Python function that implements a red-black tree with insert, delete, search, and rebalance operations. Include type hints and docstrings.",
    "reasoning": "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? Think step by step and explain your reasoning carefully.",
}


def health_check():
    """Check if vLLM server is healthy"""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False


def single_request(prompt: str, max_tokens: int = 512, stream: bool = True) -> dict:
    """
    Send a single request and measure:
    - TTFT (Time To First Token)
    - Total time
    - Tokens generated
    - TPS (Tokens Per Second)
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": stream,
    }

    start = time.perf_counter()
    ttft = None
    output_tokens = 0
    full_response = ""

    if stream:
        try:
            r = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                headers=HEADERS,
                json=payload,
                stream=True,
                timeout=300,
            )
            r.raise_for_status()

            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if ttft is None:
                                ttft = time.perf_counter() - start
                            output_tokens += 1  # approx: 1 chunk ≈ 1 token
                            full_response += content
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            return {"error": str(e)}
    else:
        try:
            r = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=300,
            )
            r.raise_for_status()
            result = r.json()
            ttft = time.perf_counter() - start
            usage = result.get("usage", {})
            output_tokens = usage.get("completion_tokens", 0)
            full_response = result["choices"][0]["message"]["content"]
        except Exception as e:
            return {"error": str(e)}

    total_time = time.perf_counter() - start

    # Better token count from non-streaming usage stats
    if not stream and output_tokens > 0:
        tps = output_tokens / total_time if total_time > 0 else 0
    else:
        # Rough estimate for streaming: ~4 chars per token
        estimated_tokens = len(full_response) / 4
        output_tokens = int(estimated_tokens)
        tps = output_tokens / total_time if total_time > 0 else 0

    return {
        "ttft": ttft,
        "total_time": total_time,
        "output_tokens": output_tokens,
        "tps": tps,
        "response_length": len(full_response),
    }


def concurrent_request(prompt: str, max_tokens: int = 256):
    """Wrapper for concurrent testing"""
    return single_request(prompt, max_tokens=max_tokens, stream=False)


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 1: Latency Test (Single Request)
# ═══════════════════════════════════════════════════════════════════
def bench_latency(runs: int = 3):
    print("\n" + "=" * 70)
    print("📊 BENCHMARK 1: LATENCY TEST (Single Request, Streaming)")
    print("=" * 70)

    for label, prompt in PROMPTS.items():
        print(f"\n🔹 Prompt: [{label}] ({len(prompt)} chars)")
        print(f"   \"{prompt[:80]}...\"" if len(prompt) > 80 else f"   \"{prompt}\"")

        results = []
        for i in range(runs):
            sys.stdout.write(f"   Run {i+1}/{runs}... ")
            sys.stdout.flush()
            r = single_request(prompt, max_tokens=512, stream=True)
            if "error" in r:
                print(f"❌ Error: {r['error']}")
                continue
            results.append(r)
            print(
                f"TTFT={r['ttft']:.3f}s | "
                f"Total={r['total_time']:.2f}s | "
                f"~{r['output_tokens']} tokens | "
                f"{r['tps']:.1f} tok/s"
            )

        if results:
            avg_ttft = statistics.mean([r["ttft"] for r in results if r["ttft"]])
            avg_tps = statistics.mean([r["tps"] for r in results])
            avg_total = statistics.mean([r["total_time"] for r in results])
            print(f"   📈 AVG: TTFT={avg_ttft:.3f}s | Total={avg_total:.2f}s | {avg_tps:.1f} tok/s")


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 2: Throughput Test (Non-streaming)
# ══��════════════════════════════════════════════════════════════════
def bench_throughput(runs: int = 3):
    print("\n" + "=" * 70)
    print("📊 BENCHMARK 2: THROUGHPUT TEST (Non-streaming, max_tokens=1024)")
    print("=" * 70)

    prompt = PROMPTS["long"]
    print(f"   Prompt: [{len(prompt)} chars]")

    results = []
    for i in range(runs):
        sys.stdout.write(f"   Run {i+1}/{runs}... ")
        sys.stdout.flush()
        r = single_request(prompt, max_tokens=1024, stream=False)
        if "error" in r:
            print(f"❌ Error: {r['error']}")
            continue
        results.append(r)
        print(
            f"Total={r['total_time']:.2f}s | "
            f"{r['output_tokens']} tokens | "
            f"{r['tps']:.1f} tok/s"
        )

    if results:
        total_tokens = sum(r["output_tokens"] for r in results)
        total_time = sum(r["total_time"] for r in results)
        avg_tps = total_tokens / total_time if total_time > 0 else 0
        print(f"   📈 Overall: {total_tokens} tokens in {total_time:.2f}s = {avg_tps:.1f} tok/s")


# ═���═════════════════════════════════════════════════════════════════
# BENCHMARK 3: Concurrency Test
# ═══════════════════════════════════════════════════════════════════
def bench_concurrency(max_concurrent: int = 4):
    print("\n" + "=" * 70)
    print("📊 BENCHMARK 3: CONCURRENCY SCALING TEST")
    print("=" * 70)

    prompt = PROMPTS["medium"]
    concurrency_levels = [1, 2, 4]
    concurrency_levels = [c for c in concurrency_levels if c <= max_concurrent]

    for n in concurrency_levels:
        print(f"\n🔹 Concurrent requests: {n}")
        start = time.perf_counter()

        results = []
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = [executor.submit(concurrent_request, prompt, 256) for _ in range(n)]
            for future in as_completed(futures):
                r = future.result()
                if "error" not in r:
                    results.append(r)

        wall_time = time.perf_counter() - start

        if results:
            total_tokens = sum(r["output_tokens"] for r in results)
            avg_latency = statistics.mean([r["total_time"] for r in results])
            throughput = total_tokens / wall_time if wall_time > 0 else 0

            print(
                f"   Wall time: {wall_time:.2f}s | "
                f"Avg latency: {avg_latency:.2f}s | "
                f"Total tokens: {total_tokens} | "
                f"Throughput: {throughput:.1f} tok/s"
            )
        else:
            print("   ❌ All requests failed")


# ═══════════════════════════════════════════════════════════════════
# BENCHMARK 4: Quick Smoke Test
# ═══════════════════════════════════════════════════════════════════
def bench_smoke():
    print("\n" + "=" * 70)
    print("🔥 QUICK SMOKE TEST")
    print("=" * 70)

    print("   Sending: 'Hello, who are you?'")
    r = single_request("Hello, who are you?", max_tokens=128, stream=True)
    if "error" in r:
        print(f"   ❌ Error: {r['error']}")
        return False
    print(f"   ✅ TTFT={r['ttft']:.3f}s | Total={r['total_time']:.2f}s | ~{r['output_tokens']} tokens | {r['tps']:.1f} tok/s")
    return True


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="vLLM Benchmark for Qwen3-Coder-Next")
    parser.add_argument(
        "--test",
        choices=["all", "smoke", "latency", "throughput", "concurrency"],
        default="all",
        help="Which benchmark to run (default: all)",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test (default: 3)")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrency level (default: 4)")
    args = parser.parse_args()

    print("╔═══════════════════════════���══════════════════════════════════════════╗")
    print("║  vLLM Benchmark — Qwen3-Coder-Next-FP8-Dynamic on MI300X          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"   Server:  {BASE_URL}")
    print(f"   Model:   {MODEL}")

    # Health check
    if not health_check():
        print("\n❌ Server tidak responding! Pastikan container sudah running.")
        sys.exit(1)
    print("   Health:  ✅ OK\n")

    if args.test in ("all", "smoke"):
        ok = bench_smoke()
        if not ok and args.test == "smoke":
            sys.exit(1)

    if args.test in ("all", "latency"):
        bench_latency(runs=args.runs)

    if args.test in ("all", "throughput"):
        bench_throughput(runs=args.runs)

    if args.test in ("all", "concurrency"):
        bench_concurrency(max_concurrent=args.max_concurrent)

    print("\n" + "=" * 70)
    print("✅ BENCHMARK SELESAI!")
    print("=" * 70)


if __name__ == "__main__":
    main()
