#!/usr/bin/env python3
"""
SWE-bench Runner — Using local vLLM as OpenAI-compatible backend
================================================================
Jalanin SWE-bench Verified/Lite dengan Qwen3-Coder-Next via vLLM

Flow:
  1. Load dataset (SWE-bench Verified / Lite)
  2. Kirim tiap issue ke model via OpenAI API (pointing ke vLLM)
  3. Model generate patch
  4. Save predictions → JSONL
  5. Evaluate pakai swebench harness (Docker-based)

Usage:
  # Step 1: Generate predictions (inference)
  python3 swe_bench_run.py infer --subset verified --max-instances 5

  # Step 2: Evaluate predictions
  python3 swe_bench_run.py eval --predictions results/predictions.jsonl

  # All-in-one (infer + eval)
  python3 swe_bench_run.py run --subset lite --max-instances 10
"""

import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from textwrap import dedent

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # Bypass CloudFront block

try:
    from datasets import load_dataset
except ImportError:
    print("❌ Install datasets: pip install datasets")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("❌ Install requests: pip install requests")
    sys.exit(1)

# ─── Config ───────────────────────────────────────────────────────
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("API_KEY", "")
MODEL = os.getenv("VLLM_MODEL", "qwen3-coder-next")
RESULTS_DIR = Path("./results")

# Dataset mapping
DATASETS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}

# ─── Prompt Template ─────────────────────────────────────────────
SYSTEM_PROMPT = dedent("""\
You are an expert software engineer. You will be given a GitHub issue description and the relevant code from the repository.
Your task is to generate a patch (in unified diff format) that resolves the issue.

Rules:
1. Output ONLY the patch in unified diff format (starting with --- and +++)
2. The patch should be minimal — only change what's necessary to fix the issue
3. Do NOT include any explanation, just the raw diff
4. Make sure the patch applies cleanly to the original code
""").strip()

USER_PROMPT_TEMPLATE = dedent("""\
## Repository: {repo}

## Issue: {title}
{problem_statement}

## Relevant Code (hints):
{hints_text}

## Task
Generate a minimal unified diff patch that fixes this issue. Output ONLY the diff, nothing else.
""").strip()


def health_check():
    """Check vLLM server"""
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        return r.status_code == 200
    except:
        return False


def generate_patch(instance: dict, max_tokens: int = 4096, temperature: float = 0.0) -> str:
    """Send an SWE-bench instance to vLLM and get a patch"""
    repo = instance.get("repo", "unknown")
    title = instance.get("problem_statement", "")[:200]
    problem = instance.get("problem_statement", "")
    hints = instance.get("hints_text", "")

    # Build prompt
    user_msg = USER_PROMPT_TEMPLATE.format(
        repo=repo,
        title=title,
        problem_statement=problem,
        hints_text=hints[:8000] if hints else "(No hints provided)",
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["## ", "```\n\n"],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    try:
        r = requests.post(
            f"{VLLM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=300,
        )
        r.raise_for_status()
        result = r.json()
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        return content, usage
    except Exception as e:
        return f"ERROR: {e}", {}


def extract_diff(text: str) -> str:
    """Extract unified diff from model response"""
    lines = text.split("\n")
    diff_lines = []
    in_diff = False

    for line in lines:
        # Start of diff
        if line.startswith("---") or line.startswith("diff --git"):
            in_diff = True
        if in_diff:
            diff_lines.append(line)
        # Also capture if wrapped in code block
        if line.strip() == "```diff":
            in_diff = True
            continue
        if in_diff and line.strip() == "```":
            in_diff = False

    if diff_lines:
        return "\n".join(diff_lines)
    # Fallback: return raw text (let evaluation harness handle it)
    return text


# ═══════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════
def run_inference(subset: str, max_instances: int, split: str = "test", output_file: str = None):
    """Generate predictions for SWE-bench instances"""
    dataset_name = DATASETS.get(subset, subset)
    RESULTS_DIR.mkdir(exist_ok=True)

    if output_file is None:
        output_file = RESULTS_DIR / f"predictions_{subset}.jsonl"
    else:
        output_file = Path(output_file)

    print(f"\n{'='*70}")
    print(f"📝 SWE-bench Inference")
    print(f"{'='*70}")
    print(f"   Dataset:    {dataset_name}")
    print(f"   Split:      {split}")
    print(f"   Max:        {max_instances if max_instances > 0 else 'ALL'}")
    print(f"   Model:      {MODEL}")
    print(f"   Output:     {output_file}")

    if not health_check():
        print("   ❌ vLLM server not responding!")
        sys.exit(1)
    print("   Server:     ✅ OK\n")

    # Load dataset
    print("   Loading dataset...")
    ds = load_dataset(dataset_name, split=split)
    total = len(ds)
    print(f"   Total instances: {total}")

    if max_instances > 0:
        ds = ds.select(range(min(max_instances, total)))
        print(f"   Running on: {len(ds)} instances\n")

    # Load existing predictions (resume support)
    existing = set()
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                try:
                    pred = json.loads(line)
                    existing.add(pred["instance_id"])
                except:
                    pass
        print(f"   ♻️  Resuming: {len(existing)} already completed\n")

    # Run inference
    success = 0
    errors = 0
    total_tokens = 0
    start_time = time.time()

    for i, instance in enumerate(ds):
        instance_id = instance["instance_id"]

        if instance_id in existing:
            print(f"   [{i+1}/{len(ds)}] {instance_id} — ⏭️  skipped (already done)")
            success += 1
            continue

        print(f"   [{i+1}/{len(ds)}] {instance_id}...", end=" ", flush=True)

        patch_text, usage = generate_patch(instance)
        tokens = usage.get("completion_tokens", 0)
        total_tokens += tokens

        if patch_text.startswith("ERROR:"):
            print(f"❌ {patch_text}")
            errors += 1
            continue

        # Extract diff from response
        model_patch = extract_diff(patch_text)

        # Save prediction in SWE-bench format
        prediction = {
            "instance_id": instance_id,
            "model_patch": model_patch,
            "model_name_or_path": MODEL,
        }

        with open(output_file, "a") as f:
            f.write(json.dumps(prediction) + "\n")

        success += 1
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        print(f"✅ {tokens} tokens | {elapsed:.0f}s elapsed | ~{avg_time:.1f}s/instance")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"📊 Inference Complete")
    print(f"{'='*70}")
    print(f"   ✅ Success:    {success}/{len(ds)}")
    print(f"   ❌ Errors:     {errors}")
    print(f"   🎯 Tokens:     {total_tokens:,}")
    print(f"   ⏱️  Time:       {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"   📄 Output:     {output_file}")
    print(f"{'='*70}\n")

    return str(output_file)


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════
def run_evaluation(predictions_path: str, subset: str = "lite", max_workers: int = 4, run_id: str = None):
    """Evaluate predictions using SWE-bench Docker harness"""
    dataset_name = DATASETS.get(subset, subset)

    if run_id is None:
        run_id = f"qwen3-coder-next-{subset}-{int(time.time())}"

    print(f"\n{'='*70}")
    print(f"🧪 SWE-bench Evaluation (Docker-based)")
    print(f"{'='*70}")
    print(f"   Dataset:       {dataset_name}")
    print(f"   Predictions:   {predictions_path}")
    print(f"   Max workers:   {max_workers}")
    print(f"   Run ID:        {run_id}")
    print()

    # Check predictions file
    if not Path(predictions_path).exists():
        print(f"   ❌ Predictions file not found: {predictions_path}")
        sys.exit(1)

    # Count predictions
    with open(predictions_path) as f:
        num_preds = sum(1 for _ in f)
    print(f"   Predictions:   {num_preds} instances")

    # Run evaluation
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset_name,
        "--predictions_path", predictions_path,
        "--max_workers", str(max_workers),
        "--run_id", run_id,
    ]

    print(f"\n   Running: {' '.join(cmd)}\n")
    print("   " + "─" * 50)

    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)
        print("   " + "─" * 50)

        if result.returncode == 0:
            print(f"\n   ✅ Evaluation complete!")
            print(f"   📁 Results in: ./evaluation_results/")
            print(f"   📁 Logs in:    ./logs/")
        else:
            print(f"\n   ❌ Evaluation failed (exit code {result.returncode})")

    except subprocess.TimeoutExpired:
        print(f"\n   ⏰ Evaluation timed out after 2 hours")
    except FileNotFoundError:
        print(f"\n   ❌ swebench not found. Install: pip install swebench")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Runner with local vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
        Examples:
          # Quick test: 5 instances from Verified
          python3 swe_bench_run.py infer --subset verified --max-instances 5

          # Full Lite (300 instances)
          python3 swe_bench_run.py infer --subset lite

          # Evaluate predictions
          python3 swe_bench_run.py eval --predictions results/predictions_lite.jsonl --subset lite

          # All-in-one
          python3 swe_bench_run.py run --subset lite --max-instances 20
        """),
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Infer
    p_infer = subparsers.add_parser("infer", help="Generate predictions")
    p_infer.add_argument("--subset", choices=["lite", "verified", "full"], default="verified")
    p_infer.add_argument("--max-instances", type=int, default=0, help="Max instances (0=all)")
    p_infer.add_argument("--split", default="test")
    p_infer.add_argument("--output", default=None, help="Output JSONL path")

    # Eval
    p_eval = subparsers.add_parser("eval", help="Evaluate predictions")
    p_eval.add_argument("--predictions", required=True, help="Path to predictions JSONL")
    p_eval.add_argument("--subset", choices=["lite", "verified", "full"], default="lite")
    p_eval.add_argument("--max-workers", type=int, default=4)
    p_eval.add_argument("--run-id", default=None)

    # Run (infer + eval)
    p_run = subparsers.add_parser("run", help="Infer + Evaluate")
    p_run.add_argument("--subset", choices=["lite", "verified", "full"], default="verified")
    p_run.add_argument("--max-instances", type=int, default=0)
    p_run.add_argument("--max-workers", type=int, default=4)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  🏆 SWE-bench — Qwen3-Coder-Next-FP8-Dynamic on MI300X            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    if args.command == "infer":
        run_inference(args.subset, args.max_instances, args.split, args.output)

    elif args.command == "eval":
        run_evaluation(args.predictions, args.subset, args.max_workers, args.run_id)

    elif args.command == "run":
        output = run_inference(args.subset, args.max_instances)
        run_evaluation(output, args.subset, args.max_workers)


if __name__ == "__main__":
    main()
