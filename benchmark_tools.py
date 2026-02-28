#!/usr/bin/env python3
"""
vLLM Tool Calling Benchmark — Qwen3-Coder-Next on MI300X
=========================================================
Tests: Single tool, Multi-tool, Parallel tools, Sequential tool chains
"""

import os
import requests
import json
import time
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from same directory as script
load_dotenv(Path(__file__).parent / ".env")

# ─── Config ───────────────────────────────────────────────────────
BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("VLLM_API_KEY", "")
MODEL = os.getenv("VLLM_MODEL", "qwen3-coder-next")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# ─── Tool Definitions ────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Jakarta' or 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '2 + 2' or 'sqrt(144)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject"
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a file with given content",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path"
                    },
                    "content": {
                        "type": "string",
                        "description": "File content"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute Python code and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    }
]

# ─── Test Scenarios ───────────────────────────────────────────────
SCENARIOS = [
    {
        "name": "Single Tool — Weather",
        "description": "Simple single tool call",
        "messages": [
            {"role": "user", "content": "What's the weather in Jakarta?"}
        ],
        "expect_tools": ["get_weather"],
    },
    {
        "name": "Single Tool — Calculator",
        "description": "Math calculation tool call",
        "messages": [
            {"role": "user", "content": "Calculate 1847 * 293 + 17"}
        ],
        "expect_tools": ["calculate"],
    },
    {
        "name": "Multi-Tool Selection",
        "description": "Model harus pilih tool yang tepat dari 6 tools",
        "messages": [
            {"role": "user", "content": "Search the web for the latest Python 3.13 release notes"}
        ],
        "expect_tools": ["search_web"],
    },
    {
        "name": "Parallel Tool Calls",
        "description": "Model seharusnya call multiple tools sekaligus",
        "messages": [
            {"role": "user", "content": "I need the weather in Jakarta and Tokyo, and also calculate 2^16 for me."}
        ],
        "expect_tools": ["get_weather", "calculate"],
    },
    {
        "name": "Sequential Chain — Tool + Follow-up",
        "description": "Tool call → fake result → model generates final answer",
        "messages": [
            {"role": "user", "content": "What's the weather in Bandung? If it's above 30°C, suggest indoor activities."},
            # Will be extended with tool result in the test
        ],
        "expect_tools": ["get_weather"],
        "follow_up": {
            "tool_result": json.dumps({"location": "Bandung", "temperature": 35, "unit": "celsius", "condition": "sunny"}),
            "tool_name": "get_weather",
        }
    },
    {
        "name": "No Tool Needed",
        "description": "Model should NOT call any tool",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "expect_tools": [],
    },
    {
        "name": "Complex — Code + File",
        "description": "Model diminta nulis kode dan save ke file",
        "messages": [
            {"role": "user", "content": "Write a Python fibonacci function and save it to /tmp/fib.py"}
        ],
        "expect_tools": ["run_code", "create_file"],
    },
]


def tool_call_request(messages, tools=None, max_tokens=1024):
    """Send a tool-calling request and return parsed result"""
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    start = time.perf_counter()
    try:
        r = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        result = r.json()
    except Exception as e:
        return {"error": str(e), "time": time.perf_counter() - start}

    elapsed = time.perf_counter() - start
    choice = result["choices"][0]
    message = choice["message"]
    usage = result.get("usage", {})

    tool_calls = message.get("tool_calls", [])
    parsed_calls = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = fn.get("arguments", "")
        parsed_calls.append({
            "id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": args,
        })

    return {
        "time": elapsed,
        "finish_reason": choice.get("finish_reason", ""),
        "content": message.get("content", ""),
        "tool_calls": parsed_calls,
        "tool_names": [c["name"] for c in parsed_calls],
        "num_tool_calls": len(parsed_calls),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def run_scenario(scenario):
    """Run a single tool calling scenario"""
    name = scenario["name"]
    desc = scenario["description"]
    messages = list(scenario["messages"])  # copy
    expect = scenario["expect_tools"]
    follow_up = scenario.get("follow_up")

    print(f"\n{'─' * 60}")
    print(f"🔧 {name}")
    print(f"   {desc}")
    print(f"   Prompt: \"{messages[-1]['content'][:80]}\"")
    if expect:
        print(f"   Expected tools: {expect}")
    else:
        print(f"   Expected: NO tool call (direct answer)")

    # Step 1: Initial request
    result = tool_call_request(messages, tools=TOOLS)

    if "error" in result:
        print(f"   ❌ Error: {result['error']}")
        return {"status": "error", "name": name}

    print(f"\n   📤 Response ({result['time']:.2f}s):")
    print(f"      Finish reason: {result['finish_reason']}")
    print(f"      Tokens: {result['prompt_tokens']} prompt + {result['completion_tokens']} completion = {result['total_tokens']} total")

    if result["tool_calls"]:
        print(f"      🔧 Tool calls: {result['num_tool_calls']}")
        for i, tc in enumerate(result["tool_calls"]):
            args_str = json.dumps(tc["arguments"], ensure_ascii=False)
            if len(args_str) > 100:
                args_str = args_str[:100] + "..."
            print(f"         [{i+1}] {tc['name']}({args_str})")

        # Validate expected tools
        called = set(result["tool_names"])
        expected = set(expect)
        if expected and expected.issubset(called):
            print(f"      ✅ Correct tools called!")
        elif expected and not expected.issubset(called):
            missing = expected - called
            print(f"      ⚠️  Missing expected: {missing}")
            extra = called - expected
            if extra:
                print(f"      ℹ️  Extra calls: {extra}")
    else:
        if result["content"]:
            content_preview = result["content"][:150].replace("\n", " ")
            print(f"      💬 Content: \"{content_preview}...\"" if len(result["content"]) > 150 else f"      💬 Content: \"{content_preview}\"")

        if not expect:
            print(f"      ✅ Correctly answered without tools!")
        else:
            print(f"      ⚠️  Expected tool call but got text response")

    # Step 2: Follow-up with tool result (if applicable)
    if follow_up and result["tool_calls"]:
        print(f"\n   📥 Simulating tool result for chain test...")
        tc = result["tool_calls"][0]

        # Build conversation with tool result
        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"]),
                }
            }]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": follow_up["tool_result"],
        })

        result2 = tool_call_request(messages, tools=TOOLS)
        if "error" not in result2:
            print(f"      ⏱️  Follow-up: {result2['time']:.2f}s | {result2['completion_tokens']} tokens")
            if result2["content"]:
                content_preview = result2["content"][:200].replace("\n", " ")
                print(f"      💬 Final answer: \"{content_preview}\"")
            if result2["tool_calls"]:
                print(f"      🔧 Additional tool calls: {[tc['name'] for tc in result2['tool_calls']]}")
            print(f"      ✅ Chain completed!")
        else:
            print(f"      ❌ Follow-up error: {result2['error']}")

    status = "pass"
    if expect and not result["tool_calls"]:
        status = "warn"
    elif not expect and result["tool_calls"]:
        status = "warn"

    return {
        "status": status,
        "name": name,
        "time": result["time"],
        "num_tool_calls": result["num_tool_calls"],
        "tools_called": result["tool_names"],
    }


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  🔧 Tool Calling Benchmark — Qwen3-Coder-Next FP8 on MI300X       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"   Server:  {BASE_URL}")
    print(f"   Model:   {MODEL}")
    print(f"   Tools:   {len(TOOLS)} registered ({', '.join(t['function']['name'] for t in TOOLS)})")

    # Health check
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        assert r.status_code == 200
        print("   Health:  ✅ OK")
    except:
        print("   Health:  ❌ Server down!")
        sys.exit(1)

    # Run all scenarios
    results = []
    for scenario in SCENARIOS:
        result = run_scenario(scenario)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("📊 TOOL CALLING BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<35} {'Status':<8} {'Time':>7} {'Tools':>6}")
    print("-" * 60)
    for r in results:
        status_icon = {"pass": "✅", "warn": "⚠️ ", "error": "❌"}.get(r["status"], "?")
        time_str = f"{r.get('time', 0):.2f}s"
        tools_str = str(r.get("num_tool_calls", 0))
        print(f"   {r['name']:<32} {status_icon}  {time_str:>7} {tools_str:>5}")

    passed = sum(1 for r in results if r["status"] == "pass")
    warned = sum(1 for r in results if r["status"] == "warn")
    failed = sum(1 for r in results if r["status"] == "error")
    total_time = sum(r.get("time", 0) for r in results)

    print("-" * 60)
    print(f"   Total: {len(results)} tests | ✅ {passed} pass | ⚠️  {warned} warn | ❌ {failed} error")
    print(f"   Total time: {total_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
