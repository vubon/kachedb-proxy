"""
Throughput and Latency Stress Benchmark for kachedb-proxy.

Measures:
1. Requests Per Second (RPS / QPS) on Semantic Cache HITs
2. P50, P90, P99 Latency distributions
3. Token and cost reduction metrics
"""

from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import time

import requests  # type: ignore[import-untyped]

BENCHMARK_PROMPTS = [
    "How do I configure memory arena slabs in KacheDB?",
    "What is the time complexity of SIMD cosine distance search?",
    "How does the timing wheel active expiry engine work?",
    "Explain the zero-copy POSIX shared memory buffer architecture.",
    "How do I set a key with TTL expiration in Python SDK?",
]


def warm_up_cache(base_url: str) -> None:
    """Pre-warm the proxy cache with the benchmark queries."""
    print("🔥 Warming up KacheDB proxy cache with benchmark prompts...")
    for prompt in BENCHMARK_PROMPTS:
        payload = {
            "model": "o3-mini",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        try:
            requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=5.0)
        except requests.RequestException:
            pass
    time.sleep(0.5)


def send_request(base_url: str, prompt: str) -> tuple[float, bool]:
    """Send a single chat completion request and return (latency_ms, is_hit)."""
    payload = {
        "model": "o3-mini",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    t0 = time.perf_counter()
    res = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=5.0)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    is_hit = (
        res.headers.get("KacheDB-Cache") == "HIT"
        or res.headers.get("X-KacheDB-Cache") == "HIT"
    )
    return latency_ms, is_hit


def run_throughput_benchmark(
    base_url: str = "http://127.0.0.1:6385",
    total_requests: int = 500,
    concurrency: int = 20,
) -> None:
    print(
        "════════════════════════════════════════════════════════════════════════════════════════"
    )
    print(f"  ⚡ KacheDB Reverse Proxy Throughput Benchmark ({base_url})")
    print(f"  Total Requests: {total_requests} | Concurrency: {concurrency} workers")
    print(
        "════════════════════════════════════════════════════════════════════════════════════════\n"
    )

    warm_up_cache(base_url)

    latencies: list[float] = []
    hits = 0

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                send_request, base_url, BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]
            )
            for i in range(total_requests)
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                lat, is_hit = future.result()
                latencies.append(lat)
                if is_hit:
                    hits += 1
            except requests.RequestException as e:
                print(f"Request error: {e}")

    total_time_s = time.perf_counter() - t_start
    qps = len(latencies) / total_time_s

    if not latencies:
        print("❌ No successful responses received.")
        return

    latencies.sort()
    p50 = statistics.median(latencies)
    p90 = latencies[int(len(latencies) * 0.90)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg_lat = statistics.mean(latencies)
    hit_ratio = (hits / len(latencies)) * 100.0

    print("📊 Benchmark Results:")
    print(f"  • Completed Requests:     {len(latencies)} / {total_requests}")
    print(f"  • Total Duration:         {total_time_s:.2f} s")
    print(f"  • Throughput (QPS):       {qps:.1f} req/sec")
    print(f"  • Cache Hit Ratio:        {hit_ratio:.1f}% ({hits} hits)")
    print(
        "────────────────────────────────────────────────────────────────────────────────────────"
    )
    print("⏱️ Latency Distribution:")
    print(f"  • Mean Latency:           {avg_lat:.2f} ms")
    print(f"  • P50 Median Latency:     {p50:.2f} ms")
    print(f"  • P90 Latency:            {p90:.2f} ms")
    print(f"  • P99 Latency:            {p99:.2f} ms")
    print(
        f"  • Min / Max:              {min(latencies):.2f} ms / {max(latencies):.2f} ms"
    )
    print(
        "════════════════════════════════════════════════════════════════════════════════════════\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KacheDB Proxy Throughput Benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:6385", help="Proxy base URL")
    parser.add_argument(
        "-n", "--requests", type=int, default=500, help="Total requests"
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=20, help="Concurrent workers"
    )
    args = parser.parse_args()

    run_throughput_benchmark(
        base_url=args.url, total_requests=args.requests, concurrency=args.concurrency
    )
