# ⚡ KacheDB AI Semantic Reverse Proxy (`kachedb-proxy`)

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Version](https://img.shields.io/badge/version-0.1.0--alpha-green.svg)](Cargo.toml)
[![Coverage](https://img.shields.io/badge/coverage-95.19%25-brightgreen.svg)](target/llvm-cov/html/index.html)
[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg)](https://www.rust-lang.org/)

**`kachedb-proxy`** is an ultra-high-throughput, dual-protocol (OpenAI & Anthropic) semantic reverse proxy written in Rust that provides **sub-millisecond prompt caching** and **80–95% token cost reduction** for AI developer tools, IDEs, and autonomous coding agents.

It implements both standard OpenAI (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`) and Anthropic Messages (`/v1/messages`) REST specifications, serving as an instant, drop-in accelerator for **Cursor**, **Claude Code**, **Cline**, **Aider**, **Continue.dev**, **LangChain**, and official OpenAI/Anthropic SDKs.

---

## 🏗️ Architecture & Dual-Tier Request Lifecycle

```text
┌────────────────────────────────────────────────────────────────────────┐
│         AI DEVELOPER TOOLS, CODING AGENTS & CLIENT SDKs                │
│    (Cursor, Claude Code, Cline, Aider, LangChain, OpenAI / Anthropic)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP Requests (Port 6385)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        KACHEDB-PROXY (PORT 6385)                       │
│                                                                        │
│  [Step 1: Canonicalize Prompt & Scoped Model Isolation]                │
│                                                                        │
│  [Step 2: Tier 1 SwissTable Exact Lookup (< 100 µs)]                   │
│   └── HIT? ──> Return Instant Completion (< 1.5 ms total, $0 billed)   │
│                                                                        │
│  [Step 3: On-Device ONNX Vector Generation (BGE-Small, CPU)]           │
│                                                                        │
│  [Step 4: Tier 2 SIMD Vector Cosine Matching in KacheDB]               │
│   └── HIT? ──> Return Semantic Cached Completion (< 5 ms, $0 billed)   │
│                                                                        │
│  [Step 5: Cache Miss? ──> Forward to Upstream LLM Server]              │
│   - Auto-translate Anthropic <-> OpenAI if upstream is llama.cpp/OAI   │
│   - Upstream response saved to BOTH Tier 1 & Tier 2 in background      │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
    (Testing: Zero API Keys Needed)       (Production: Frontier Providers)
                    ▼                                ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│        LOCAL LLAMA.CPP SERVER        │  │     OPENAI / ANTHROPIC API   │
│   llama-server -m model.gguf -p 8081 │  │     https://api.openai.com   │
└──────────────────────────────────────┘  └──────────────────────────────┘
```

---

## 🌟 Key Features

* **⚡ Sub-Millisecond Exact Hits (Tier 1):** Serves exact query hits from KacheDB SwissTable memory pool in $< 1.5\text{ ms}$ client latency ($< 100\ \mu\text{s}$ server latency).
* **🎯 Sub-5ms Semantic Hits (Tier 2):** Serves semantically equivalent prompts in $< 5\text{ ms}$ via SIMD vector cosine distance matching.
* **🧠 $0 Local On-Device Embeddings:** Generates 384-dimensional vector embeddings on CPU via ONNX Runtime (`fastembed-rs`). **Zero external API keys required.**
* **🔄 Universal Dual-Protocol Gateway:**
  - OpenAI `/v1/chat/completions` (JSON & Server-Sent Events).
  - Anthropic `/v1/messages` (JSON & Anthropic SSE events: `message_start`, `content_block_delta`, `message_delta`, etc.).
  - Protocol bridge: translates Claude Code requests to local `llama.cpp` or OpenAI upstreams seamlessly.
* **⚡ Precision Token Economics:** Tracks exact prompt, completion, and total tokens saved along with dual-tier cache hit ratios and latency speedups via `GET /v1/savings`.
* **🛡️ Model & Tenant Isolation + Dynamic Thresholds:** Scoped caches prevent cross-model completion leakage; isolate multi-tenant workspaces via `KacheDB-Namespace: <tenant>` and override similarity thresholds dynamically per request via `KacheDB-Threshold: 0.88`.

---

## 🚀 Quickstart

### 1. 1-Command Docker Compose Stack
Launch the complete stack (KacheDB + `kachedb-proxy`) in one command:
```bash
# Point upstream to OpenAI (or local llama.cpp)
export OPENAI_API_KEY="sk-..."
docker compose -f docker-compose.ai.yml up -d
```
Or run **100% offline with zero API keys** using local `llama.cpp`:
```bash
docker compose -f docker-compose.ai.yml --profile local-llm up -d
```

### 2. Run from Source
```bash
# Build production release binary
make release

# Run proxy (points to local llama.cpp on 8081 or OpenAI API)
make run
```

### 3. Test via cURL
```bash
# First request (Cache MISS -> Upstream forward -> Cached in KacheDB)
curl -i http://localhost:6385/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
    "model": "o3-mini",
    "messages": [{"role": "user", "content": "How do I reverse a linked list in Rust?"}],
    "stream": false
  }'

# Second semantically identical request (Cache HIT -> Sub-5ms Instant Return)
curl -i http://localhost:6385/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "o3-mini",
    "messages": [{"role": "user", "content": "Explain how to reverse a linked list in Rust."}],
    "stream": false
  }'
```

Response Headers:
```http
HTTP/1.1 200 OK
KacheDB-Cache: HIT
KacheDB-Match: SEMANTIC
KacheDB-Tier: 2-Semantic
KacheDB-Score: 0.942
Content-Type: application/json
```

---

## 🔌 IDE & Framework Integrations

### 1. Cursor IDE
1. Open **Cursor Settings** $\rightarrow$ **Models**.
2. Enable **OpenAI API Key** and toggle **Override OpenAI Base URL**.
3. Set Base URL:
   ```text
   http://localhost:6385/v1
   ```

### 2. Aider (Terminal AI Pair Programmer)
```bash
export OPENAI_BASE_URL="http://localhost:6385/v1"
export OPENAI_API_KEY="your-api-key"
aider
```

### 3. Continue.dev (VS Code Extension)
In `~/.continue/config.json`:
```json
{
  "models": [
    {
      "title": "o3-mini (Cached via KacheDB)",
      "provider": "openai",
      "model": "o3-mini",
      "apiBase": "http://localhost:6385/v1",
      "apiKey": "your-api-key"
    }
  ]
}
```

### 4. Python OpenAI SDK
```python
from openai import OpenAI

# Simply redirect base_url to kachedb-proxy
client = OpenAI(base_url="http://localhost:6385/v1", api_key="your-api-key")

response = client.chat.completions.create(
    model="o3-mini",
    messages=[{"role": "user", "content": "Explain memory slabs in Rust."}],
    stream=False,
)
print(response.choices[0].message.content)
```

### 5. Claude Code & Anthropic SDK
`kachedb-proxy` natively implements the Anthropic Messages API (`/v1/messages`):
```bash
# Point Claude Code directly to kachedb-proxy
export ANTHROPIC_BASE_URL="http://localhost:6385"
export ANTHROPIC_API_KEY="your-api-key"
claude
```
Or via official Python `anthropic` SDK:
```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:6385",
    api_key="your-api-key",
)
message = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "How do memory arenas work?"}],
)
print(message.content[0].text)
```

### 6. Cline (Autonomous AI Coding Agent)
In **Cline Settings**:
1. Select **OpenAI Compatible** (or **Anthropic Compatible**).
2. Set **Base URL**: `http://localhost:6385/v1` (or `http://localhost:6385` for Anthropic).
3. Set **API Key**: enter any placeholder key (or real provider key).
4. Set **Model ID**: `o3-mini`, `gpt-4.5`, `claude-sonnet-5-5`, or `deepseek-r1`.

### 7. LangChain & LlamaIndex
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:6385/v1",
    api_key="your-api-key",
    model="o3-mini",
)
response = llm.invoke("What is a SwissTable in Rust?")
print(response.content)
```

### 8. LiteLLM Proxy / Router
```python
import litellm

litellm.api_base = "http://localhost:6385/v1"
response = litellm.completion(
    model="openai/o3-mini", messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## 🛠️ Developer Tooling & Makefile

`kachedb-proxy` includes an automated developer Makefile for tests, linting, code coverage, and Docker:

| Command | Action |
| :--- | :--- |
| `make build` | Compile debug binary |
| `make release` | Compile optimized production release binary |
| `make test` / `make test-unit` | Run all 61 Rust unit tests |
| `make test-e2e` | Run Python E2E integration test suite against local `llama.cpp` |
| `make coverage` | Terminal code coverage breakdown via `cargo llvm-cov` |
| `make coverage-html` | Generate interactive HTML coverage report at `target/llvm-cov/html/index.html` |
| `make lint` | Run `clippy --all-targets -- -D warnings`, `cargo fmt --check`, and `ruff` |
| `make fmt` | Auto-format Rust and Python codebases |
| `make compose-up` | Launch full KacheDB + Proxy stack in Docker |
| `make compose-local-llm` | Launch full stack + local `llama.cpp` for 100% offline development |

---

## ⚙️ Configuration Reference

| Flag | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `-p, --port` | `KACHEDB_PROXY_PORT` | `6385` | Local HTTP listening port |
| `--host` | `KACHEDB_PROXY_HOST` | `0.0.0.0` | Host interface to bind to |
| `-u, --upstream-url` (`--upstream`) | `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | Upstream LLM provider URL |
| `-k, --upstream-api-key` (`--api-key`) | `UPSTREAM_API_KEY` | None | Default API key for upstream authentication |
| `--kachedb-host` | `KACHEDB_HOST` | `127.0.0.1` | KacheDB server host |
| `--kachedb-port` | `KACHEDB_PORT` | `6379` | KacheDB server port |
| `-t, --similarity-threshold` (`--threshold`) | `SIMILARITY_THRESHOLD`| `0.85` | Cosine similarity threshold for semantic hits (0.0–1.0) |
| `--ttl-seconds` (`--ttl`) | `CACHE_TTL_SECONDS` | `86400` | Expiration time for cached responses in seconds (24h) |
| `--embedding-model` | `EMBEDDING_MODEL` | `bge-small-en-v1.5` | Local ONNX embedding model |
| `--mock-embedder` | `MOCK_EMBEDDER` | `false` | Deterministic hashing embedder for testing |

### Per-Request HTTP Headers

| Header | Example | Description |
| :--- | :--- | :--- |
| `KacheDB-Threshold` | `0.88` | Override the default cosine similarity threshold (`0.0`–`1.0`) for this request |
| `KacheDB-Namespace` | `team-alpha` | Scope exact and semantic cache lookups/writes to an isolated tenant or workspace namespace |

---

## 📊 Telemetry & Token Economics Endpoints

* **`GET /health`**: Returns proxy status, backend targets, and active embedding model.
* **`GET /metrics`**: Real-time metrics breakdown for Prometheus/scrapers.
* **`GET /v1/savings`** (or `GET /savings`): Real-time token economics and cache efficiency metrics:
  ```json
  {
    "total_requests": 14250,
    "cache_hits": 9975,
    "hit_ratio_percent": 70.0,
    "exact_hits": 5700,
    "semantic_hits": 4275,
    "prompt_tokens_saved": 18200000,
    "completion_tokens_saved": 4150000,
    "total_tokens_saved": 22350000,
    "avg_upstream_latency_ms": 385.2,
    "avg_cache_latency_ms": 4.8
  }
  ```

### ⚡ Live Terminal Token Economics Report
Send `SIGUSR1` to the proxy process (or shut down via `CTRL+C`) to print a styled terminal token economics summary:
```bash
kill -USR1 $(pgrep kachedb-proxy)
```
```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ KacheDB Proxy Token Savings & Efficiency Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Requests Handled:           14,250
Exact Cache Hits (Tier 1):         5,700 (40.0%)
Semantic Cache Hits (Tier 2):      4,275 (30.0%)
Total Cache Hit Ratio:             70.0%
Prompt Tokens Saved:          18,200,000
Completion Tokens Saved:       4,150,000
Total Tokens Saved:           22,350,000
Avg Upstream Latency:             385.2 ms
Avg Cache Hit Latency:             4.80 ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📄 License
Licensed under [MIT OR Apache-2.0](LICENSE).
