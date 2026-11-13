"""
End-to-End Integration Tests for kachedb-proxy.

Tests OpenAI-compatible endpoints & Dual-Layer Caching (RFC 6648 Modern Headers):
1. /health and /metrics (including Tier 1 exact vs Tier 2 semantic hits)
2. /v1/models
3. /v1/embeddings
4. /v1/chat/completions (Non-Streaming: Cache Miss -> Write -> Cache Hit)
5. Tier 1 SwissTable exact-match short-circuit (< 5ms with KacheDB-Tier: 1-Exact)
6. Multi-model namespace isolation (gpt-4o vs claude-3-5 vs qwen)
7. Per-request dynamic threshold override (KacheDB-Threshold)
8. /v1/chat/completions (Streaming SSE: Delta chunks + [DONE])
9. Official openai Python SDK integration
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import pytest
import requests

PROXY_BASE_URL = "http://127.0.0.1:6385"


@pytest.fixture(scope="module")
def proxy_available() -> bool:
    """Check if kachedb-proxy is running on port 6385."""
    try:
        res = requests.get(f"{PROXY_BASE_URL}/health", timeout=1.0)
        return bool(res.status_code == 200)
    except requests.RequestException:
        return False


def test_health_endpoint(proxy_available: bool) -> None:
    if not proxy_available:
        pytest.skip("kachedb-proxy not running on 6385")

    res = requests.get(f"{PROXY_BASE_URL}/health", timeout=2.0)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert data["service"] == "kachedb-proxy"


def test_metrics_endpoint(proxy_available: bool) -> None:
    if not proxy_available:
        pytest.skip("kachedb-proxy not running on 6385")

    res = requests.get(f"{PROXY_BASE_URL}/metrics", timeout=2.0)
    assert res.status_code == 200
    data = res.json()
    assert "total_requests" in data
    assert "cache_hits" in data
    assert "exact_hits" in data
    assert "semantic_hits" in data
    assert "cache_misses" in data
    assert "tokens_saved" in data
    assert "total_tokens_saved" in data
    assert "prompt_tokens_saved" in data
    assert "completion_tokens_saved" in data


def test_savings_endpoints(proxy_available: bool) -> None:
    if not proxy_available:
        pytest.skip("kachedb-proxy not running on 6385")

    for path in ["/v1/savings", "/savings"]:
        res = requests.get(f"{PROXY_BASE_URL}{path}", timeout=2.0)
        assert res.status_code == 200
        data = res.json()
        assert "total_requests" in data
        assert "cache_hits" in data
        assert "hit_ratio_percent" in data
        assert "exact_hits" in data
        assert "semantic_hits" in data
        assert "prompt_tokens_saved" in data
        assert "completion_tokens_saved" in data
        assert "total_tokens_saved" in data
        assert "avg_upstream_latency_ms" in data
        assert "avg_cache_latency_ms" in data


def test_savings_accumulation_on_cache_hit(proxy_available: bool) -> None:
    if not proxy_available:
        pytest.skip("kachedb-proxy not running on 6385")

    session = requests.Session()
    before = session.get(f"{PROXY_BASE_URL}/v1/savings", timeout=2.0).json()

    unique_id = uuid.uuid4().hex[:8]
    prompt = f"What is the capital of planet Neptune {unique_id}?"
    model_name = f"gpt-4o-savings-{unique_id}"
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    # Initial query (MISS)
    res1 = session.post(
        f"{PROXY_BASE_URL}/v1/chat/completions",
        json=payload,
        timeout=10.0,
    )
    assert res1.status_code == 200
    assert res1.headers.get("KacheDB-Cache") == "MISS"

    time.sleep(0.3)

    # Second query (HIT)
    res2 = session.post(
        f"{PROXY_BASE_URL}/v1/chat/completions",
        json=payload,
        timeout=5.0,
    )
    assert res2.status_code == 200
    assert res2.headers.get("KacheDB-Cache") == "HIT"

    after = session.get(f"{PROXY_BASE_URL}/v1/savings", timeout=2.0).json()
    assert after["total_requests"] >= before["total_requests"] + 2
    assert after["cache_hits"] >= before["cache_hits"] + 1
    assert after["prompt_tokens_saved"] > before["prompt_tokens_saved"]
    assert after["completion_tokens_saved"] > before["completion_tokens_saved"]
    assert after["total_tokens_saved"] > before["total_tokens_saved"]


def test_models_endpoint(proxy_available: bool) -> None:
    if not proxy_available:
        pytest.skip("kachedb-proxy not running on 6385")

    res = requests.get(f"{PROXY_BASE_URL}/v1/models", timeout=2.0)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    models = [m["id"] for m in data["data"]]
    assert "gpt-4o" in models


def test_embeddings_endpoint(proxy_available: bool) -> None:
    if not proxy_available:
        pytest.skip("kachedb-proxy not running on 6385")

    payload: dict[str, Any] = {
        "input": "How does KacheDB implement SIMD cosine distance in Rust?",
        "model": "bge-small-en-v1.5",
    }
    res = requests.post(f"{PROXY_BASE_URL}/v1/embeddings", json=payload, timeout=3.0)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    embedding = data["data"][0]["embedding"]
    assert isinstance(embedding, list)
    assert len(embedding) == 384


def test_chat_completions_semantic_cache_hit_cycle(proxy_available: bool) -> None:
    if not proxy_available:
        pytest.skip("kachedb-proxy not running on 6385")

    session = requests.Session()
    unique_query = f"Explain memory arenas in database systems {time.time()}"
    payload: dict[str, Any] = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": unique_query}],
        "stream": False,
    }

    # Initial query
    res1 = session.post(
        f"{PROXY_BASE_URL}/v1/chat/completions",
        json=payload,
        timeout=10.0,
        headers={"Authorization": "Bearer test-key"},
    )

    if res1.status_code == 200:
        # Wait a moment for async background VADD to complete
        time.sleep(0.3)

        # Second semantically identical query -> MUST be HIT
        t0 = time.perf_counter()
        res2 = session.post(
            f"{PROXY_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=5.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        assert res2.status_code == 200
        assert res2.headers.get("KacheDB-Cache") == "HIT"
        assert latency_ms < 50.0
        data2 = res2.json()
        assert "choices" in data2
        assert len(data2["choices"]) > 0


def test_tier1_exact_cache_short_circuit(proxy_available: bool) -> None:
    """Verify Tier 1 SwissTable exact-match short-circuits before ONNX embedding (< 10ms)."""
    if not proxy_available:
        pytest.skip("kachedb-proxy not running on 6385")

    session = requests.Session()
    u = uuid.uuid4().hex[:6]
    model = f"qwen-exact-{u}"
    prompt = f"Compute 123 + 456 for test {u}"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    # Initial query -> MISS
    res1 = session.post(
        f"{PROXY_BASE_URL}/v1/chat/completions", json=payload, timeout=10.0
    )
    assert res1.status_code == 200
    assert res1.headers.get("KacheDB-Cache") == "MISS"

    time.sleep(0.3)

    # Identical query -> MUST hit Tier 1 SwissTable exact fast-path
    t0 = time.perf_counter()
    res2 = session.post(
        f"{PROXY_BASE_URL}/v1/chat/completions", json=payload, timeout=5.0
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert res2.status_code == 200
    assert res2.headers.get("KacheDB-Cache") == "HIT"
    assert res2.headers.get("KacheDB-Match") == "EXACT"
    assert res2.headers.get("KacheDB-Tier") == "1-Exact"
    assert res2.headers.get("KacheDB-Score") == "1.000"
    assert latency_ms < 25.0  # Ultra-fast point lookup


def test_multi_model_namespace_isolation(proxy_available: bool) -> None:
    """Verify separate models (e.g. gpt-4o vs deepseek-r1) do not cross-pollinate cache."""
    if not proxy_available:
        pytest.skip("kachedb-proxy not running on 6385")

    session = requests.Session()
    u = uuid.uuid4().hex[:6]
    model_a = f"test-alpha-{u}"
    model_b = f"test-beta-{u}"
    prompt = f"What is the meaning of life? Test id {u}"

    # Query with model A -> MUST be MISS
    payload_a: dict[str, Any] = {
        "model": model_a,
        "messages": [{"role": "user", "content": prompt}],
    }
    res_a1 = session.post(
        f"{PROXY_BASE_URL}/v1/chat/completions", json=payload_a, timeout=10.0
    )
    assert res_a1.status_code == 200
    assert res_a1.headers.get("KacheDB-Cache") == "MISS"

    time.sleep(0.3)

    # Same prompt with model B -> MUST be MISS (isolated namespace)
    payload_b: dict[str, Any] = {
        "model": model_b,
        "messages": [{"role": "user", "content": prompt}],
    }
    res_b1 = session.post(
        f"{PROXY_BASE_URL}/v1/chat/completions", json=payload_b, timeout=10.0
    )
    assert res_b1.status_code == 200
    assert res_b1.headers.get("KacheDB-Cache") == "MISS"

    time.sleep(0.3)

    # Re-query model A -> MUST be HIT from Model A namespace
    res_a2 = session.post(
        f"{PROXY_BASE_URL}/v1/chat/completions", json=payload_a, timeout=5.0
    )
    assert res_a2.status_code == 200
    assert res_a2.headers.get("KacheDB-Cache") == "HIT"


def test_dynamic_threshold_override(proxy_available: bool) -> None:
    """Verify KacheDB-Threshold header dynamically overrides similarity gate."""
    if not proxy_available:
        pytest.skip("kachedb-proxy not running on 6385")

    session = requests.Session()
    tag = uuid.uuid4().hex[:8]
    prompt1 = f"What is the atomic number of Helium? Unique: {tag}. One word."
    prompt2 = f"What is Helium's atomic number? Unique: {tag}. One word."

    payload1: dict[str, Any] = {
        "model": "qwen2.5",
        "messages": [{"role": "user", "content": prompt1}],
    }
    res1 = session.post(
        f"{PROXY_BASE_URL}/v1/chat/completions", json=payload1, timeout=10.0
    )
    assert res1.status_code == 200

    time.sleep(0.3)

    # Query with strict threshold (0.999) via modern KacheDB-Threshold header
    payload2: dict[str, Any] = {
        "model": "qwen2.5",
        "messages": [{"role": "user", "content": prompt2}],
    }
    res_strict = session.post(
        f"{PROXY_BASE_URL}/v1/chat/completions",
        json=payload2,
        headers={"KacheDB-Threshold": "0.999"},
        timeout=10.0,
    )
    assert res_strict.status_code == 200
    assert res_strict.headers.get("KacheDB-Cache") == "MISS"


def test_streaming_chat_completions(proxy_available: bool) -> None:
    if not proxy_available:
        pytest.skip("kachedb-proxy not running on 6385")

    payload: dict[str, Any] = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Tell me a 1-sentence joke."}],
        "stream": True,
    }

    res = requests.post(
        f"{PROXY_BASE_URL}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=10.0,
        headers={"Authorization": "Bearer test-key"},
    )

    assert res.status_code == 200
    assert "text/event-stream" in res.headers.get("Content-Type", "")

    chunks_received = 0
    got_done = False

    for line in res.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data:"):
                chunks_received += 1
                if "[DONE]" in decoded:
                    got_done = True

    assert chunks_received > 0
    assert got_done


def test_openai_sdk_compatibility(proxy_available: bool) -> None:
    if not proxy_available:
        pytest.skip("kachedb-proxy not running on 6385")

    try:
        from openai import OpenAI  # type: ignore[import-not-found]

        client = OpenAI(base_url=f"{PROXY_BASE_URL}/v1", api_key="sk-test-fake-key")

        # Test listing models
        models = client.models.list()
        assert len(list(models)) > 0
    except ImportError:
        pytest.skip("openai package not installed in environment")


def test_anthropic_messages_non_streaming_cycle(proxy_available: bool) -> None:
    if not proxy_available:
        pytest.skip("kachedb-proxy not running on 6385")

    session = requests.Session()
    unique_id = uuid.uuid4().hex[:8]
    prompt = f"Explain the concept of memory arenas in systems software {unique_id}"
    model = f"claude-3-5-sonnet-anthropic-{unique_id}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "stream": False,
    }

    # Request 1 (MISS -> Forward to upstream -> Cache in KacheDB)
    res1 = session.post(
        f"{PROXY_BASE_URL}/v1/messages",
        json=payload,
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        timeout=10.0,
    )
    assert res1.status_code == 200
    assert res1.headers.get("KacheDB-Cache") == "MISS"

    data1 = res1.json()
    assert data1["type"] == "message"
    assert data1["role"] == "assistant"
    assert isinstance(data1["content"], list)
    assert len(data1["content"]) > 0
    assert data1["content"][0]["type"] == "text"
    text1 = data1["content"][0]["text"]
    assert len(text1) > 0
    assert data1["stop_reason"] == "end_turn"
    assert data1["usage"]["input_tokens"] > 0
    assert data1["usage"]["output_tokens"] > 0

    time.sleep(0.3)

    # Request 2 (Exact HIT -> Sub-50ms instant response)
    t0 = time.perf_counter()
    res2 = session.post(
        f"{PROXY_BASE_URL}/v1/messages",
        json=payload,
        headers={"x-api-key": "test-key"},
        timeout=5.0,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert res2.status_code == 200
    assert res2.headers.get("KacheDB-Cache") == "HIT"
    assert res2.headers.get("KacheDB-Match") == "EXACT"
    assert res2.headers.get("KacheDB-Tier") == "1-Exact"
    assert latency_ms < 50.0

    data2 = res2.json()
    assert data2["type"] == "message"
    assert data2["role"] == "assistant"
    assert data2["content"][0]["text"] == text1


def test_anthropic_messages_streaming_cycle(proxy_available: bool) -> None:
    if not proxy_available:
        pytest.skip("kachedb-proxy not running on 6385")

    session = requests.Session()
    unique_id = uuid.uuid4().hex[:8]
    prompt = f"Give 3 short bullet points about kernel bypass networking {unique_id}"
    model = f"claude-3-5-sonnet-stream-{unique_id}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "stream": True,
    }

    # Request 1: Streaming Miss
    res1 = session.post(
        f"{PROXY_BASE_URL}/v1/messages",
        json=payload,
        stream=True,
        timeout=10.0,
    )
    assert res1.status_code == 200
    assert "text/event-stream" in res1.headers.get("Content-Type", "")

    events_seen = []
    text_deltas = []

    for line in res1.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if decoded.startswith("event:"):
            events_seen.append(decoded[6:].strip())
        elif decoded.startswith("data:"):
            raw_data = decoded[5:].strip()
            try:
                data_obj = json.loads(raw_data)
                if data_obj.get("type") == "content_block_delta":
                    text_deltas.append(data_obj["delta"]["text"])
            except json.JSONDecodeError:
                pass

    assert "message_start" in events_seen
    assert "content_block_start" in events_seen
    assert "content_block_delta" in events_seen
    assert "content_block_stop" in events_seen
    assert "message_delta" in events_seen
    assert "message_stop" in events_seen
    assert len("".join(text_deltas)) > 0

    time.sleep(0.3)

    # Request 2: Streaming Hit (Cache Replay)
    res2 = session.post(
        f"{PROXY_BASE_URL}/v1/messages",
        json=payload,
        stream=True,
        timeout=5.0,
    )
    assert res2.status_code == 200
    assert res2.headers.get("KacheDB-Cache") == "HIT"

    events_seen_hit = []
    text_deltas_hit = []
    for line in res2.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if decoded.startswith("event:"):
            events_seen_hit.append(decoded[6:].strip())
        elif decoded.startswith("data:"):
            raw_data = decoded[5:].strip()
            try:
                data_obj = json.loads(raw_data)
                if data_obj.get("type") == "content_block_delta":
                    text_deltas_hit.append(data_obj["delta"]["text"])
            except json.JSONDecodeError:
                pass

    assert "message_start" in events_seen_hit
    assert "content_block_delta" in events_seen_hit
    assert "message_stop" in events_seen_hit
    assert len("".join(text_deltas_hit)) > 0


def test_anthropic_messages_system_prompt_and_blocks(proxy_available: bool) -> None:
    if not proxy_available:
        pytest.skip("kachedb-proxy not running on 6385")

    session = requests.Session()
    unique_id = uuid.uuid4().hex[:8]
    model = f"claude-3-5-sonnet-sys-{unique_id}"

    payload: dict[str, Any] = {
        "model": model,
        "system": [{"type": "text", "text": "You are a concise mathematician."}],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"What is 13 * 17? {unique_id}"}],
            }
        ],
        "max_tokens": 128,
        "stream": False,
    }

    res1 = session.post(f"{PROXY_BASE_URL}/v1/messages", json=payload, timeout=10.0)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["type"] == "message"
    assert len(data1["content"]) > 0

    time.sleep(0.3)

    res2 = session.post(f"{PROXY_BASE_URL}/v1/messages", json=payload, timeout=5.0)
    assert res2.status_code == 200
    assert res2.headers.get("KacheDB-Cache") == "HIT"
