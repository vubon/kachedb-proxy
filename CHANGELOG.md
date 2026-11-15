# Changelog

All notable changes to the `kachedb-proxy` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0-alpha] - 2026-11-15

### 🚀 Added
- **Google EmbeddingGemma 2 Inference Engine**:
  - Direct local ONNX Runtime inference engine using `ort 2.0.0-rc.9`, `tokenizers 0.21`, and `ndarray 0.16`.
  - Lock-free concurrent session execution with bounded concurrency permits.
  - Automatic download and caching of model weights (`model.onnx`, `tokenizer.json`) to `.fastembed_cache/embeddinggemma-2-onnx`.
- **Matryoshka Representation Learning (MRL) Truncation**:
  - Support for 384-dimensional truncated embeddings with L2 re-normalization alongside native 768-dimensional embeddings.
  - Configurable via `--embedding-dim 384` or `--embedding-dim 768`.
- **Task-Aware Prompt Formatting**:
  - Automatically prefixes input text with Gemma task prompts (`task: search result | query: ...`) based on routing context (`RetrievalQuery`, `RetrievalDocument`, `SemanticSimilarity`, `Classification`, `Clustering`).
- **Dual-Engine Model Architecture**:
  - CLI and environment flag `--embedding-model` (`bge-small-en-v1.5` default, `embeddinggemma-2`).
  - Graceful fallback preserves 100% backward compatibility for existing deployments.
- **8,192-Token Sequence Guard**:
  - Strictly enforces the 8,192 token context window limit before model execution, rejecting oversized inputs with `HTTP 400 Bad Request` across `/v1/chat/completions`, `/v1/messages`, and `/v1/embeddings`.

---

## [0.1.0-alpha] - 2026-11-08

### 🚀 Added
- **Tier 1 SwissTable Exact-Match Short-Circuit (< 100 µs)**:
  - Canonicalized multi-turn chat messages hashed via `xxh3_64`.
  - Sub-millisecond instant responses (< 1.5 ms total client roundtrip, < 100 µs server latency) with `KacheDB-Tier: 1-Exact`.
- **Tier 2 Multi-Model SIMD Vector Semantic Caching**:
  - Isolated vector namespaces by model identifier (`chat_cache:<sanitized_model>`).
  - Dynamic per-request threshold override via modern `KacheDB-Threshold` header (with legacy `X-KacheDB-Threshold` fallback).
  - High-performance asynchronous connection pooling with auto-reconnect on stale sockets.
- **Anthropic Messages API Gateway (`POST /v1/messages` & `POST /messages`)**:
  - Full compatibility with **Claude Code** and official `@anthropic-ai/sdk` and `anthropic-python`.
  - Seamless handling of string and structured block arrays for both `system` and message `content`.
  - Universal gateway translation: forwards to OpenAI-compatible upstreams (e.g. local `llama.cpp` or OpenAI API) and formats completions into compliant Anthropic schemas.
  - Full Anthropic Server-Sent Events (SSE) streaming translation emitting `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, and `message_stop`.
- **Real-Time Token Economics & Efficiency Telemetry**:
  - Lock-free atomic accounting of prompt, completion, and total tokens saved alongside Tier-1/Tier-2 hit ratios and latency speedups.
  - New `GET /v1/savings` and `GET /savings` JSON endpoints.
  - Interactive ASCII terminal report upon shutdown or on demand via `kill -USR1 $(pgrep kachedb-proxy)`.
- **Packaging, Developer Tooling & Verification**:
  - Shifted default port to `6385` within KacheDB's dedicated `638x` port family.
  - Standardized modern RFC 6648 response headers (`KacheDB-*`).
  - Comprehensive developer `Makefile` (`make build`, `make test`, `make test-e2e`, `make coverage`, `make lint`).
  - Multi-container `docker-compose.ai.yml` orchestration stack with optional `llama-server` profile.
  - Automated test suite achieving **95.19% line coverage**, 94.50% region coverage, and 96.27% function coverage.

---

## [0.1.0-alpha.1] - 2026-08-31

### 🚀 Added
- **OpenAI Semantic Reverse Proxy Core**: High-performance Rust HTTP service implementing standard OpenAI `/v1` endpoints.
- **Dual-Mode Response Engine**:
  - `stream: false` serves standard `application/json` payloads for batch tools, linters, and non-SSE IDEs.
  - `stream: true` multiplexes `text/event-stream` chunks for interactive chat tools (**Cursor Chat**, **Aider**).
- **Zero-Cost On-Device Vector Embeddings**:
  - `fastembed-rs` local ONNX Runtime inference (~2–4 ms) with support for `bge-small-en-v1.5`, `bge-base-en-v1.5`, and `all-minilm-l6-v2`.
  - Deterministic `MockEmbedder` for rapid unit tests and CI pipelines.
- **Asynchronous In-Memory Cache Client**: Non-blocking `RESP3` wire client communicating with KacheDB vector indexes (`openai_chat_cache`) using raw little-endian f32 embeddings.
- **Telemetry & Health Endpoints**:
  - `GET /health`: Service health and model status.
  - `GET /metrics`: Real-time request counts, cache hits, hit ratio %, tokens saved, and latency percentiles.
  - `GET /v1/models` & `POST /v1/embeddings`: Full OpenAI compatibility.
- **Comprehensive Test Suite & Benchmarks**:
  - Rust unit tests with 100% pass rate.
  - Python E2E integration test suite (`tests/test_proxy_e2e.py`).
  - Concurrent throughput load test harness (`benchmarks/bench_proxy_throughput.py`).
  - Criterion micro-benchmarks (`benches/proxy_microbenches.rs`).
- **DevOps & Packaging**:
  - Multi-arch `Dockerfile` (`linux/amd64`, `linux/arm64`).
  - GitHub Actions CI/CD workflows for multi-OS formatting, linting, testing, and container publishing.
