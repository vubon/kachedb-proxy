#!/usr/bin/env python3
"""Interactive terminal chat client connecting to kachedb-proxy.

Supports:
  • Single-turn mode (default for prompt testing - each prompt is tested independently)
  • Multi-turn mode (/mode multi - maintains ongoing chat context)
  • Real-time streaming SSE tokens
  • Precise latency measurement (Server retrieval time & TTFT)
  • Color-coded KacheDB cache badges & speedup metrics
  • Telemetry metrics (/stats)
"""

import json
import time
import urllib.request

PROXY_URL = "http://127.0.0.1:6385/v1/chat/completions"
SAVINGS_URL = "http://127.0.0.1:6385/savings"
MODEL = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# Session-level latency tracking for dynamic speedup calculation
llm_prompt_baselines = {}  # prompt_text (lowercase) -> total_ms
session_llm_times = []  # list of upstream LLM generation times


def fetch_savings() -> dict:
    try:
        with urllib.request.urlopen(SAVINGS_URL, timeout=2) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def print_stats():
    stats = fetch_savings()
    if "error" in stats:
        print(f"{YELLOW}Could not fetch stats: {stats['error']}{RESET}")
        return

    print(
        f"\n{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}"
    )
    print(f"{BOLD}📊 KacheDB Proxy Live Token Economics{RESET}")
    print(f"{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    print(f"  • Total Requests:         {stats.get('total_requests', 0):,}")
    print(f"  • Cache Hits:             {stats.get('cache_hits', 0):,}")
    print(f"  • Exact Hits (Tier 1):    {stats.get('exact_hits', 0):,}")
    print(f"  • Semantic Hits (Tier 2): {stats.get('semantic_hits', 0):,}")
    print(f"  • Hit Ratio:              {stats.get('hit_ratio_percent', 0.0)}%")
    print(f"  • Prompt Tokens Saved:    {stats.get('prompt_tokens_saved', 0):,}")
    print(f"  • Completion Tokens Saved:{stats.get('completion_tokens_saved', 0):,}")
    print(f"  • Total Tokens Saved:     {stats.get('total_tokens_saved', 0):,}")
    print(
        f"  • Avg Cache Latency:      {stats.get('avg_cache_latency_ms', 0.0):.2f} ms"
    )
    print(
        f"  • Avg Upstream Latency:   {stats.get('avg_upstream_latency_ms', 0.0):.2f} ms"
    )
    print(
        f"{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}\n"
    )


def chat_turn(messages: list) -> str:
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 150,
            "stream": True,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        PROXY_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    full_reply = []
    t_start = time.perf_counter()
    t_first_token = None

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            headers = response.headers
            cache_status = headers.get("KacheDB-Cache", "UNKNOWN")
            tier = headers.get("KacheDB-Tier", "")
            score = headers.get("KacheDB-Score", "")
            server_lat = headers.get("KacheDB-Latency-Ms", "")

            for line in response:
                line_str = line.decode("utf-8").strip()
                if not line_str or not line_str.startswith("data: "):
                    continue
                data_part = line_str[6:]
                if data_part == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_part)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                            if cache_status == "HIT":
                                if tier == "1-Exact":
                                    badge = f"{GREEN}{BOLD}[⚡ TIER 1 EXACT HIT]{RESET}"
                                else:
                                    badge = f"{CYAN}{BOLD}[🎯 TIER 2 SEMANTIC HIT (Score: {score})]{RESET}"
                            else:
                                badge = f"{YELLOW}{BOLD}[🌐 UPSTREAM LLM MISS]{RESET}"

                            print(
                                f"\n{badge} {BOLD}Assistant:{RESET} ",
                                end="",
                                flush=True,
                            )

                        print(content, end="", flush=True)
                        full_reply.append(content)
                except json.JSONDecodeError:
                    pass

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0

        prompt_key = ""
        if messages and isinstance(messages[-1], dict):
            prompt_key = messages[-1].get("content", "").strip().lower()

        # Print latency summary footer
        if cache_status == "HIT":
            hit_lat = float(server_lat) if server_lat else total_ms
            exact_baseline = llm_prompt_baselines.get(prompt_key)
            session_baseline = session_llm_times[-1] if session_llm_times else None

            if exact_baseline is not None:
                speedup = max(1.0, exact_baseline / max(0.1, hit_lat))
                print(
                    f"\n{DIM}⏱️  Retrieved from KacheDB in {BOLD}{hit_lat:.1f}ms{RESET}{DIM} (vs {exact_baseline:.1f}ms measured LLM run for this prompt — {BOLD}{speedup:.0f}x faster{RESET}{DIM}){RESET}\n"
                )
            elif session_baseline is not None:
                speedup = max(1.0, session_baseline / max(0.1, hit_lat))
                print(
                    f"\n{DIM}⏱️  Retrieved from KacheDB in {BOLD}{hit_lat:.1f}ms{RESET}{DIM} (vs {session_baseline:.1f}ms session LLM baseline — {BOLD}{speedup:.0f}x faster{RESET}{DIM}){RESET}\n"
                )
            else:
                print(
                    f"\n{DIM}⏱️  Retrieved from KacheDB in {BOLD}{hit_lat:.1f}ms{RESET}{DIM} (sub-millisecond in-memory cache hit){RESET}\n"
                )
        else:
            if prompt_key:
                llm_prompt_baselines[prompt_key] = total_ms
            session_llm_times.append(total_ms)

            print(
                f"\n{DIM}⏱️  Generated by LLM upstream in {BOLD}{total_ms:.1f}ms{RESET}{DIM} (cached in KacheDB for subsequent queries){RESET}\n"
            )

    except Exception as e:  # noqa: BLE001
        print(f"\n{YELLOW}Request failed: {e}{RESET}\n")

    return "".join(full_reply)


def main():
    print(
        f"{BOLD}══════════════════════════════════════════════════════════════════{RESET}"
    )
    print(f"{BOLD}  ⚡ KacheDB AI Interactive Chat Client (Connected to :6385){RESET}")
    print("  • Upstream:   llama-server (qwen2.5-0.5b-instruct)")
    print("  • Memory:     KacheDB Container (:6380)")
    print(
        f"  • Modes:      {GREEN}/mode single{RESET} (isolated queries, default) | {CYAN}/mode multi{RESET} (conversation history)"
    )
    print(
        f"  • Commands:   {CYAN}/stats{RESET} (metrics), {CYAN}/clear{RESET} (reset history), {CYAN}/exit{RESET} (quit)"
    )
    print(
        f"{BOLD}══════════════════════════════════════════════════════════════════{RESET}\n"
    )

    mode = "single"  # 'single' or 'multi'
    history = []

    while True:
        try:
            if mode == "single":
                prompt_label = f"{BOLD}You {DIM}[single-prompt]{RESET}{BOLD}: {RESET}"
            else:
                turns = len(history) // 2
                prompt_label = (
                    f"{BOLD}You {DIM}[multi turn {turns + 1}]{RESET}{BOLD}: {RESET}"
                )

            prompt = input(prompt_label).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q", "/exit", "/quit"):
            print("Goodbye!")
            break
        if prompt.lower() in ("/clear", "/reset"):
            history.clear()
            print(f"{GREEN}✓ Chat history reset.{RESET}\n")
            continue
        if prompt.lower() in ("/stats", "/metrics"):
            print_stats()
            continue
        if prompt.lower() == "/mode single":
            mode = "single"
            history.clear()
            print(
                f"{GREEN}✓ Switched to SINGLE-PROMPT mode. Each prompt is evaluated independently for instant cache hits.{RESET}\n"
            )
            continue
        if prompt.lower() == "/mode multi":
            mode = "multi"
            print(
                f"{CYAN}✓ Switched to MULTI-TURN mode. Maintains conversation history context across turns.{RESET}\n"
            )
            continue

        if mode == "single":
            messages = [{"role": "user", "content": prompt}]
            chat_turn(messages)
        else:
            history.append({"role": "user", "content": prompt})
            reply = chat_turn(history)
            if reply:
                history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
