//! OpenAI /v1/chat/completions handler supporting dual-layer caching (SwissTable exact + SIMD vector) and SSE streaming.

use crate::cache::{DynKacheDbClient, canonicalize_messages, sanitize_model_name};
use crate::embed::gemma::DEFAULT_MAX_TOKENS as MAX_SEQUENCE_TOKENS;
use crate::embed::task::{EmbeddingTask, PREFIX_SEARCH_QUERY};
use crate::error::ProxyError;
use crate::http::sse::{extract_sse_delta, generate_cached_sse_stream};
use crate::state::AppState;
use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::Sse;
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use xxhash_rust::xxh3::xxh3_64;

pub async fn chat_completions_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Result<Response, ProxyError> {
    let t0 = Instant::now();
    let model = payload
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("o3-mini")
        .to_string();

    let stream = payload
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    let messages = payload
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or_else(|| ProxyError::InvalidRequest("Missing 'messages' array".to_string()))?;

    let mut canonical_prompt = canonicalize_messages(messages);
    if canonical_prompt.is_empty() {
        return Err(ProxyError::InvalidRequest(
            "Empty prompt messages".to_string(),
        ));
    }

    // Include tools and response_format in canonical representation if present
    // so structured-output or tool-calling requests never collide with plain text completions.
    if let Some(tools) = payload.get("tools")
        && !tools.is_null()
    {
        canonical_prompt.push_str(&format!("\n[tools:{tools}]"));
    }
    if let Some(rf) = payload.get("response_format")
        && !rf.is_null()
    {
        canonical_prompt.push_str(&format!("\n[response_format:{rf}]"));
    }

    // Enforce 8,192-token context window guard before cache/upstream operations
    let token_count = canonical_prompt.split_whitespace().count();
    if token_count > MAX_SEQUENCE_TOKENS {
        return Err(ProxyError::InvalidRequest(format!(
            "Prompt exceeds maximum sequence length of {MAX_SEQUENCE_TOKENS} tokens (got {token_count})"
        )));
    }

    // Scoped multi-model and optional tenant namespace isolation identifiers
    let namespace = headers
        .get("KacheDB-Namespace")
        .or_else(|| headers.get("X-KacheDB-Namespace"))
        .and_then(|h| h.to_str().ok())
        .map(sanitize_model_name)
        .filter(|s| !s.is_empty() && s != "default");

    let sanitized_model = match namespace {
        Some(ns) => format!("{ns}:{}", sanitize_model_name(&model)),
        None => sanitize_model_name(&model),
    };
    let prompt_hash = xxh3_64(canonical_prompt.trim().to_lowercase().as_bytes());
    let exact_key = format!("exact_chat:{}:{:016x}", sanitized_model, prompt_hash);
    let index_name = format!("chat_cache:{}", sanitized_model);

    // Dynamic per-request threshold override (modern KacheDB-Threshold, legacy X-KacheDB-Threshold fallback)
    let effective_threshold = headers
        .get("KacheDB-Threshold")
        .or_else(|| headers.get("X-KacheDB-Threshold"))
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|&t| (0.0..=1.0).contains(&t))
        .unwrap_or(state.config.similarity_threshold);

    // =========================================================================
    // Tier 1: SwissTable Exact Hash Short-Circuit (< 100 µs)
    // =========================================================================
    if let Ok(Some(cached_payload)) = state.kachedb.get(&exact_key).await {
        let elapsed_us = t0.elapsed().as_micros() as f64;
        info!(
            "⚡ [TIER-1 EXACT HIT] Model: {}, Latency: {:.1}µs, Prompt: '{}'",
            sanitized_model,
            elapsed_us,
            truncate(&canonical_prompt, 40)
        );

        let prompt_tokens = (canonical_prompt.len() / 4).max(1) as u64;
        let completion_tokens = (cached_payload.len() / 4).max(1) as u64;
        state
            .telemetry
            .record_exact_hit(&model, prompt_tokens, completion_tokens, elapsed_us);

        return Ok(build_cached_chat_response(
            model,
            cached_payload,
            prompt_tokens,
            completion_tokens,
            stream,
            "EXACT",
            "1-Exact",
            "1.000",
            0.0,
            0.0,
            elapsed_us / 1000.0,
        ));
    }

    // =========================================================================
    // Tier 2: SIMD Vector Cosine Semantic Search (< 5 ms)
    // =========================================================================
    let embed_t0 = Instant::now();
    let prompt_vec = encode_search_query(&state, &canonical_prompt)?;
    let embed_elapsed_ms = embed_t0.elapsed().as_secs_f64() * 1000.0;

    let cache_hit = state
        .kachedb
        .vsearch(&index_name, &prompt_vec, 1, effective_threshold)
        .await?;

    if let Some(hit) = cache_hit {
        let elapsed_us = t0.elapsed().as_micros() as f64;
        info!(
            "🎯 [TIER-2 SEMANTIC HIT] Model: {}, Score: {:.3}, Latency: {:.1}µs, Prompt: '{}'",
            sanitized_model,
            hit.score,
            elapsed_us,
            truncate(&canonical_prompt, 40)
        );

        let prompt_tokens = (canonical_prompt.len() / 4).max(1) as u64;
        let completion_tokens = (hit.payload.len() / 4).max(1) as u64;
        state
            .telemetry
            .record_semantic_hit(&model, prompt_tokens, completion_tokens, elapsed_us);

        // Asynchronously register prompt shortcut into Tier 1 SwissTable for subsequent sub-microsecond hits
        let kachedb_clone = state.kachedb.clone();
        let exact_key_clone = exact_key.clone();
        let payload_clone = hit.payload.clone();
        let ttl = state.config.ttl_seconds;
        tokio::spawn(async move {
            if let Err(e) = kachedb_clone
                .set(&exact_key_clone, &payload_clone, Some(ttl))
                .await
            {
                warn!("Failed to promote semantic hit to Tier-1 cache: {e}");
            }
        });

        let score_str = format!("{:.3}", hit.score);
        let distance = (1.0 - hit.score).max(0.0);
        return Ok(build_cached_chat_response(
            model,
            hit.payload,
            prompt_tokens,
            completion_tokens,
            stream,
            "SEMANTIC",
            "2-Semantic",
            &score_str,
            distance,
            embed_elapsed_ms,
            elapsed_us / 1000.0,
        ));
    }

    // =========================================================================
    // Cache MISS -> Forward to Upstream LLM Server
    // =========================================================================
    debug!(
        "💨 [CACHE MISS] Model: {}, Forwarding prompt to upstream: {}",
        sanitized_model, state.config.upstream_url
    );

    let upstream_endpoint = format!(
        "{}/chat/completions",
        state.config.upstream_url.trim_end_matches('/')
    );

    let mut req_builder = state.http_client.post(&upstream_endpoint).json(&payload);

    // Forward auth header if present, else fallback to config API key
    if let Some(auth) = headers.get("Authorization") {
        req_builder = req_builder.header("Authorization", auth);
    } else if let Some(ref key) = state.config.upstream_api_key {
        req_builder = req_builder.header("Authorization", format!("Bearer {key}"));
    }

    let upstream_res = match req_builder.send().await {
        Ok(res) => res,
        Err(e) => {
            error!("Failed to connect to upstream LLM: {e}");
            return Err(ProxyError::Upstream(e));
        }
    };

    let status = upstream_res.status();
    if !status.is_success() {
        let err_text = upstream_res.text().await.unwrap_or_default();
        return Ok((status, [("Content-Type", "application/json")], err_text).into_response());
    }

    // Non-streaming Miss Path
    if !stream {
        let resp_json: Value = upstream_res.json().await?;
        let elapsed_us = t0.elapsed().as_micros() as f64;
        state.telemetry.record_miss(elapsed_us);

        // Extract assistant text content to store in both Tier 1 and Tier 2 KacheDB
        let assistant_text = resp_json
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|msg| msg.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        if !assistant_text.is_empty() {
            spawn_cache_persist(
                state.kachedb.clone(),
                exact_key.clone(),
                index_name.clone(),
                format!("chat:{}", Uuid::new_v4().simple()),
                prompt_vec,
                assistant_text,
                state.config.ttl_seconds,
            );
        }

        let mut response = (StatusCode::OK, Json(resp_json)).into_response();
        let elapsed_ms = elapsed_us / 1000.0;
        let headers = response.headers_mut();
        headers.insert("KacheDB-Cache", HeaderValue::from_static("MISS"));
        headers.insert("Kache-Hit", HeaderValue::from_static("false"));
        headers.insert("Kache-Distance", HeaderValue::from_static("1.0000"));
        if let Ok(val) = HeaderValue::from_str(&format!("{:.2}", embed_elapsed_ms)) {
            headers.insert("Kache-Embedding-Time-Ms", val);
        }
        if let Ok(val) = HeaderValue::from_str(&format!("{:.2}", elapsed_ms)) {
            headers.insert("KacheDB-Latency-Ms", val);
        }
        return Ok(response);
    }

    // Streaming Miss Path (Multiplexed SSE Tee Stream)
    let elapsed_us = t0.elapsed().as_micros() as f64;
    state.telemetry.record_miss(elapsed_us);
    let elapsed_ms = elapsed_us / 1000.0;

    let kachedb_clone = state.kachedb.clone();
    let query_id = format!("chat:{}", Uuid::new_v4().simple());
    let ttl = state.config.ttl_seconds;
    let exact_key_clone = exact_key.clone();
    let index_clone = index_name.clone();
    let mut upstream_stream = upstream_res.bytes_stream();

    let output_stream = async_stream::stream! {
        let mut accumulated_text = String::new();
        let mut saved = false;

        while let Some(chunk_result) = upstream_stream.next().await {
            match chunk_result {
                Ok(bytes) => {
                    let text_chunk = String::from_utf8_lossy(&bytes);
                    for line in text_chunk.lines() {
                        if let Some(delta) = extract_sse_delta(line) {
                            accumulated_text.push_str(&delta);
                        }
                    }

                    // If upstream signals [DONE], persist before yielding to prevent
                    // loss if downstream client disconnects immediately upon receiving [DONE].
                    if !saved && text_chunk.contains("[DONE]") && !accumulated_text.is_empty() {
                        saved = true;
                        spawn_cache_persist(
                            kachedb_clone.clone(),
                            exact_key_clone.clone(),
                            index_clone.clone(),
                            query_id.clone(),
                            prompt_vec.clone(),
                            accumulated_text.clone(),
                            ttl,
                        );
                    }

                    yield Ok::<_, axum::Error>(bytes);
                }
                Err(e) => {
                    error!("Stream read error from upstream: {e}");
                    break;
                }
            }
        }

        // Stream completed naturally without [DONE] tag: Store in KacheDB if not already saved
        if !saved && !accumulated_text.is_empty() {
            spawn_cache_persist(
                kachedb_clone,
                exact_key_clone,
                index_clone,
                query_id,
                prompt_vec,
                accumulated_text,
                ttl,
            );
        }
    };

    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .header("KacheDB-Cache", "MISS")
        .header("Kache-Hit", "false")
        .header("Kache-Distance", "1.0000");
    if let Ok(val) = HeaderValue::from_str(&format!("{:.2}", embed_elapsed_ms)) {
        builder = builder.header("Kache-Embedding-Time-Ms", val);
    }
    if let Ok(val) = HeaderValue::from_str(&format!("{:.2}", elapsed_ms)) {
        builder = builder.header("KacheDB-Latency-Ms", val);
    }
    let response = builder
        .body(axum::body::Body::from_stream(output_stream))
        .map_err(|e| ProxyError::Internal(format!("Failed to build SSE response: {e}")))?;

    Ok(response)
}

fn encode_search_query(state: &AppState, canonical_prompt: &str) -> Result<Vec<f32>, ProxyError> {
    if state.config.search_prefix == PREFIX_SEARCH_QUERY {
        state
            .embedder
            .encode_with_task(canonical_prompt, EmbeddingTask::SearchQuery)
    } else if canonical_prompt.starts_with(&state.config.search_prefix) {
        state.embedder.encode(canonical_prompt)
    } else {
        let prefixed = format!("{}{}", state.config.search_prefix, canonical_prompt);
        state.embedder.encode(&prefixed)
    }
}

#[allow(clippy::too_many_arguments)]
fn build_cached_chat_response(
    model: String,
    payload: String,
    prompt_tokens: u64,
    completion_tokens: u64,
    stream: bool,
    match_type: &'static str,
    tier: &'static str,
    score: &str,
    distance: f32,
    embed_elapsed_ms: f64,
    elapsed_ms: f64,
) -> Response {
    let mut response = if stream {
        let sse_stream = generate_cached_sse_stream(model, payload);
        Sse::new(sse_stream).into_response()
    } else {
        let completion_id = format!("chatcmpl-kache-{}", Uuid::new_v4().simple());
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let resp_json = json!({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": payload
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (prompt_tokens + completion_tokens).max(2)
            }
        });
        (StatusCode::OK, Json(resp_json)).into_response()
    };

    let headers = response.headers_mut();
    headers.insert("KacheDB-Cache", HeaderValue::from_static("HIT"));
    headers.insert("KacheDB-Match", HeaderValue::from_static(match_type));
    headers.insert("KacheDB-Tier", HeaderValue::from_static(tier));
    headers.insert("Kache-Hit", HeaderValue::from_static("true"));
    if let Ok(val) = HeaderValue::from_str(score) {
        headers.insert("KacheDB-Score", val);
    }
    if let Ok(val) = HeaderValue::from_str(&format!("{:.4}", distance)) {
        headers.insert("Kache-Distance", val);
    }
    if let Ok(val) = HeaderValue::from_str(&format!("{:.2}", embed_elapsed_ms)) {
        headers.insert("Kache-Embedding-Time-Ms", val);
    }
    if let Ok(val) = HeaderValue::from_str(&format!("{:.2}", elapsed_ms)) {
        headers.insert("KacheDB-Latency-Ms", val);
    }
    response
}

fn spawn_cache_persist(
    kachedb: DynKacheDbClient,
    exact_key: String,
    index: String,
    query_id: String,
    prompt_vec: Vec<f32>,
    text: String,
    ttl: u64,
) {
    tokio::spawn(async move {
        if let Err(e) = kachedb.set(&exact_key, &text, Some(ttl)).await {
            warn!("Tier-1 cache write failed for '{exact_key}': {e}");
        }
        if let Err(e) = kachedb
            .vadd(&index, &query_id, &prompt_vec, &text, ttl)
            .await
        {
            warn!("Tier-2 vector write failed for '{index}': {e}");
        }
    });
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(max_chars).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KacheDbClient;
    use crate::config::Config;
    use crate::embed::EmbeddingEngine;
    use crate::embed::mock::MockEmbedder;
    use crate::telemetry::TelemetryTracker;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn spawn_mock_kachedb(
        exact_hit: Option<String>,
        vector_hit: Option<(String, f32, String)>,
    ) -> (String, u16) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let exact_hit = exact_hit.clone();
                let vector_hit = vector_hit.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    while let Ok(n) = socket.read(&mut buf).await {
                        if n == 0 {
                            break;
                        }
                        let req = String::from_utf8_lossy(&buf[..n]);
                        if req.contains("GET") {
                            if let Some(ref val) = exact_hit {
                                let resp = format!("${}\r\n{}\r\n", val.len(), val);
                                let _ = socket.write_all(resp.as_bytes()).await;
                            } else {
                                let _ = socket.write_all(b"$-1\r\n").await;
                            }
                        } else if req.contains("SET") {
                            let _ = socket.write_all(b"+OK\r\n").await;
                        } else if req.contains("VSEARCH") {
                            if let Some((ref id, score, ref payload)) = vector_hit {
                                let resp = format!(
                                    "*1\r\n*3\r\n${}\r\n{}\r\n,{:.3}\r\n${}\r\n{}\r\n",
                                    id.len(),
                                    id,
                                    score,
                                    payload.len(),
                                    payload
                                );
                                let _ = socket.write_all(resp.as_bytes()).await;
                            } else {
                                let _ = socket.write_all(b"*0\r\n").await;
                            }
                        } else if req.contains("VADD") {
                            let _ = socket.write_all(b"+OK\r\n").await;
                        } else {
                            let _ = socket.write_all(b"-ERR\r\n").await;
                        }
                    }
                });
            }
        });

        ("127.0.0.1".to_string(), port)
    }

    async fn spawn_mock_upstream(streaming: bool, error: bool) -> (String, u16) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let n = socket.read(&mut buf).await.unwrap_or(0);
                    if n > 0 {
                        if error {
                            let resp = "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: 26\r\n\r\n{\"error\":\"internal failure\"}";
                            let _ = socket.write_all(resp.as_bytes()).await;
                        } else if streaming {
                            let chunks = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n34\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"Chunk1\"}}]}\n\n\r\n0e\r\ndata: [DONE]\n\n\r\n0\r\n\r\n";
                            let _ = socket.write_all(chunks.as_bytes()).await;
                        } else {
                            let body = json!({
                                "id": "chatcmpl-mock",
                                "object": "chat.completion",
                                "created": 1234567,
                                "model": "mock-model",
                                "choices": [{
                                    "index": 0,
                                    "message": { "role": "assistant", "content": "Mock upstream answer" },
                                    "finish_reason": "stop"
                                }],
                                "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
                            }).to_string();
                            let resp = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                                body.len(),
                                body
                            );
                            let _ = socket.write_all(resp.as_bytes()).await;
                        }
                    }
                });
            }
        });

        ("127.0.0.1".to_string(), port)
    }

    fn make_state(kachedb_port: u16, upstream_port: u16) -> AppState {
        let config = Config {
            port: 6385,
            host: "127.0.0.1".to_string(),
            upstream_url: format!("http://127.0.0.1:{upstream_port}/v1"),
            upstream_api_key: Some("test-key".to_string()),
            kachedb_host: "127.0.0.1".to_string(),
            kachedb_port,
            similarity_threshold: 0.85,
            ttl_seconds: 3600,
            embedding_model: "bge-small".to_string(),
            mock_embedder: true,
            ..Config::default()
        };
        AppState {
            config: Arc::new(config),
            embedder: Arc::new(MockEmbedder::default()),
            kachedb: Arc::new(KacheDbClient::new("127.0.0.1".to_string(), kachedb_port)),
            http_client: reqwest::Client::new(),
            telemetry: Arc::new(TelemetryTracker::new()),
        }
    }

    #[tokio::test]
    async fn test_tier1_exact_hit_non_streaming() {
        let (_, kport) = spawn_mock_kachedb(Some("Exact cached hit".to_string()), None).await;
        let (_, uport) = spawn_mock_upstream(false, false).await;
        let state = make_state(kport, uport);

        let payload = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        });

        let mut headers = HeaderMap::new();
        headers.insert("KacheDB-Threshold", HeaderValue::from_static("0.80"));
        let res = chat_completions_handler(State(state), headers, Json(payload))
            .await
            .unwrap();

        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers().get("KacheDB-Cache").unwrap(), "HIT");
        assert_eq!(res.headers().get("KacheDB-Match").unwrap(), "EXACT");
        assert_eq!(res.headers().get("KacheDB-Tier").unwrap(), "1-Exact");
        assert_eq!(res.headers().get("Kache-Hit").unwrap(), "true");
        assert_eq!(res.headers().get("Kache-Distance").unwrap(), "0.0000");
        assert_eq!(
            res.headers().get("Kache-Embedding-Time-Ms").unwrap(),
            "0.00"
        );
    }

    #[tokio::test]
    async fn test_tier1_exact_hit_streaming() {
        let (_, kport) = spawn_mock_kachedb(Some("Stream exact hit".to_string()), None).await;
        let (_, uport) = spawn_mock_upstream(false, false).await;
        let state = make_state(kport, uport);

        let payload = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        });

        let res = chat_completions_handler(State(state), HeaderMap::new(), Json(payload))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers().get("KacheDB-Cache").unwrap(), "HIT");
        assert_eq!(res.headers().get("Kache-Hit").unwrap(), "true");
    }

    #[tokio::test]
    async fn test_tier2_semantic_hit_non_streaming() {
        let (_, kport) = spawn_mock_kachedb(
            None,
            Some(("id1".to_string(), 0.945, "Semantic cached hit".to_string())),
        )
        .await;
        let (_, uport) = spawn_mock_upstream(false, false).await;
        let state = make_state(kport, uport);

        let payload = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello there"}],
            "stream": false
        });

        let res = chat_completions_handler(State(state), HeaderMap::new(), Json(payload))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers().get("KacheDB-Cache").unwrap(), "HIT");
        assert_eq!(res.headers().get("KacheDB-Match").unwrap(), "SEMANTIC");
        assert_eq!(res.headers().get("KacheDB-Tier").unwrap(), "2-Semantic");
        assert_eq!(res.headers().get("Kache-Hit").unwrap(), "true");
        assert_eq!(res.headers().get("Kache-Distance").unwrap(), "0.0550");
        assert!(res.headers().get("Kache-Embedding-Time-Ms").is_some());
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    }

    #[tokio::test]
    async fn test_tier2_semantic_hit_streaming() {
        let (_, kport) = spawn_mock_kachedb(
            None,
            Some(("id1".to_string(), 0.945, "Semantic cached hit".to_string())),
        )
        .await;
        let (_, uport) = spawn_mock_upstream(false, false).await;
        let state = make_state(kport, uport);

        let payload = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello there"}],
            "stream": true
        });

        let res = chat_completions_handler(State(state), HeaderMap::new(), Json(payload))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers().get("KacheDB-Cache").unwrap(), "HIT");
        assert_eq!(res.headers().get("Kache-Hit").unwrap(), "true");
    }

    #[tokio::test]
    async fn test_cache_miss_forwarding_non_streaming() {
        let (_, kport) = spawn_mock_kachedb(None, None).await;
        let (_, uport) = spawn_mock_upstream(false, false).await;
        let state = make_state(kport, uport);

        let payload = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "New question"}],
            "stream": false
        });

        let mut headers = HeaderMap::new();
        headers.insert(
            "Authorization",
            HeaderValue::from_static("Bearer test-token"),
        );
        let res = chat_completions_handler(State(state), headers, Json(payload))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers().get("KacheDB-Cache").unwrap(), "MISS");
        assert_eq!(res.headers().get("Kache-Hit").unwrap(), "false");
        assert_eq!(res.headers().get("Kache-Distance").unwrap(), "1.0000");
        assert!(res.headers().get("Kache-Embedding-Time-Ms").is_some());
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    }

    #[tokio::test]
    async fn test_cache_miss_forwarding_streaming() {
        let (_, kport) = spawn_mock_kachedb(None, None).await;
        let (_, uport) = spawn_mock_upstream(true, false).await;
        let state = make_state(kport, uport);

        let payload = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "New stream question"}],
            "stream": true
        });

        let res = chat_completions_handler(State(state), HeaderMap::new(), Json(payload))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers().get("KacheDB-Cache").unwrap(), "MISS");
        assert_eq!(res.headers().get("Kache-Hit").unwrap(), "false");
        assert_eq!(res.headers().get("Kache-Distance").unwrap(), "1.0000");
        assert!(res.headers().get("Kache-Embedding-Time-Ms").is_some());
        let body_bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body_bytes);
        assert!(body_str.contains("Chunk1"));
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    }

    #[tokio::test]
    async fn test_cache_miss_upstream_error() {
        let (_, kport) = spawn_mock_kachedb(None, None).await;
        let (_, uport) = spawn_mock_upstream(false, true).await;
        let state = make_state(kport, uport);

        let payload = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Error question"}],
            "stream": false
        });

        let res = chat_completions_handler(State(state), HeaderMap::new(), Json(payload))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("this is a very long string", 10), "this is a ...");
    }

    #[tokio::test]
    async fn test_empty_messages_validation() {
        let (_, kport) = spawn_mock_kachedb(None, None).await;
        let (_, uport) = spawn_mock_upstream(false, false).await;
        let state = make_state(kport, uport);

        // Missing messages
        let payload1 = json!({"model": "gpt-4o"});
        let err1 = chat_completions_handler(State(state.clone()), HeaderMap::new(), Json(payload1))
            .await
            .unwrap_err();
        assert!(matches!(err1, ProxyError::InvalidRequest(_)));

        // Empty messages array
        let payload2 = json!({"model": "gpt-4o", "messages": []});
        let err2 = chat_completions_handler(State(state.clone()), HeaderMap::new(), Json(payload2))
            .await
            .unwrap_err();
        assert!(matches!(err2, ProxyError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn test_threshold_headers_and_config_api_key() {
        let (_, kport) = spawn_mock_kachedb(None, None).await;
        let (_, uport) = spawn_mock_upstream(false, false).await;
        let mut state = make_state(kport, uport);
        let mut cfg = (*state.config).clone();
        cfg.upstream_api_key = Some("sk-config-key-1234".to_string());
        state.config = Arc::new(cfg);

        let mut headers = HeaderMap::new();
        headers.insert("KacheDB-Threshold", HeaderValue::from_static("0.90"));
        headers.insert("KacheDB-Namespace", HeaderValue::from_static("team-beta"));

        let payload = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Question with custom threshold"}],
            "tools": [{"type": "function", "function": {"name": "search"}}],
            "response_format": {"type": "json_object"}
        });

        let res = chat_completions_handler(State(state), headers, Json(payload))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_chat_prompt_exceeds_8192_tokens_returns_400() {
        let (_, kport) = spawn_mock_kachedb(None, None).await;
        let (_, uport) = spawn_mock_upstream(false, false).await;
        let state = make_state(kport, uport);

        let oversized_content = vec!["token"; 8195].join(" ");
        let payload = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": oversized_content}]
        });

        let err = chat_completions_handler(State(state), HeaderMap::new(), Json(payload))
            .await
            .unwrap_err();
        match err {
            ProxyError::InvalidRequest(msg) => {
                assert!(msg.contains("8192"));
            }
            other => panic!("Expected InvalidRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_chat_rfc6648_kache_headers_on_hit_and_miss() {
        let (_, kport_hit) = spawn_mock_kachedb(
            None,
            Some(("id-hit".to_string(), 0.900, "Cached reply".to_string())),
        )
        .await;
        let (_, uport) = spawn_mock_upstream(false, false).await;
        let state_hit = make_state(kport_hit, uport);

        let payload = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Test RFC 6648 headers"}]
        });
        let res_hit =
            chat_completions_handler(State(state_hit), HeaderMap::new(), Json(payload.clone()))
                .await
                .unwrap();
        assert_eq!(res_hit.headers().get("Kache-Hit").unwrap(), "true");
        assert_eq!(res_hit.headers().get("Kache-Distance").unwrap(), "0.1000");
        assert!(res_hit.headers().get("Kache-Embedding-Time-Ms").is_some());

        let (_, kport_miss) = spawn_mock_kachedb(None, None).await;
        let state_miss = make_state(kport_miss, uport);
        let res_miss = chat_completions_handler(State(state_miss), HeaderMap::new(), Json(payload))
            .await
            .unwrap();
        assert_eq!(res_miss.headers().get("Kache-Hit").unwrap(), "false");
        assert_eq!(res_miss.headers().get("Kache-Distance").unwrap(), "1.0000");
        assert!(res_miss.headers().get("Kache-Embedding-Time-Ms").is_some());
    }

    struct RecordingEmbedder {
        recorded: Arc<Mutex<Vec<String>>>,
    }

    impl EmbeddingEngine for RecordingEmbedder {
        fn encode(&self, text: &str) -> Result<Vec<f32>, ProxyError> {
            self.recorded.lock().unwrap().push(text.to_string());
            Ok(vec![0.0; 384])
        }

        fn dimension(&self) -> usize {
            384
        }
    }

    #[tokio::test]
    async fn test_chat_custom_search_prefix_propagation() {
        let (_, kport) = spawn_mock_kachedb(None, None).await;
        let (_, uport) = spawn_mock_upstream(false, false).await;
        let mut state = make_state(kport, uport);

        let recorded = Arc::new(Mutex::new(Vec::new()));
        state.embedder = Arc::new(RecordingEmbedder {
            recorded: recorded.clone(),
        });
        let mut cfg = (*state.config).clone();
        cfg.search_prefix = "custom_search_prefix: ".to_string();
        state.config = Arc::new(cfg);

        let payload = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Explain rust borrow checker"}]
        });
        let res = chat_completions_handler(State(state), HeaderMap::new(), Json(payload))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        let captured = recorded.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert!(captured[0].starts_with("custom_search_prefix: "));
    }
}
