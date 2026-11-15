//! Anthropic Messages API handler (`/v1/messages`).
//!
//! Provides bidirectional compatibility with Anthropic SDKs and coding agents
//! (e.g. `claude-code`, Claude Desktop) with dual-layer prompt caching,
//! exact SwissTable short-circuiting, and SSE event streaming.

use crate::cache::DynKacheDbClient;
use crate::cache::canonicalizer::{canonicalize_anthropic_prompt, sanitize_model_name};
use crate::embed::gemma::DEFAULT_MAX_TOKENS as MAX_SEQUENCE_TOKENS;
use crate::embed::task::{EmbeddingTask, PREFIX_SEARCH_QUERY};
use crate::error::ProxyError;
use crate::http::sse::{
    extract_anthropic_sse_delta, extract_sse_delta, generate_cached_anthropic_sse_stream,
};
use crate::state::AppState;
use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response, Sse, sse::Event};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Instant;
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use xxhash_rust::xxh3::xxh3_64;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    String(String),
    Blocks(Vec<ContentBlock>),
}

impl SystemPrompt {
    pub fn as_text(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Blocks(blocks) => {
                let texts: Vec<&str> = blocks.iter().filter_map(|b| b.text.as_deref()).collect();
                texts.join("\n")
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ContentBlock {
    #[serde(rename = "type", default = "default_block_type")]
    pub block_type: String,
    #[serde(default)]
    pub text: Option<String>,
}

fn default_block_type() -> String {
    "text".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl MessageContent {
    pub fn as_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(blocks) => {
                let texts: Vec<&str> = blocks.iter().filter_map(|b| b.text.as_deref()).collect();
                texts.join("\n")
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub tools: Option<Value>,
}

pub async fn messages_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<AnthropicMessagesRequest>,
) -> Result<Response, ProxyError> {
    let t0 = Instant::now();
    let model = payload.model.clone();
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

    let system_text = payload.system.as_ref().map(|s| s.as_text());
    let messages: Vec<(String, String)> = payload
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.as_text()))
        .collect();

    let mut canonical_prompt = canonicalize_anthropic_prompt(system_text.as_deref(), &messages);
    if canonical_prompt.is_empty() {
        return Err(ProxyError::InvalidRequest(
            "Empty prompt messages".to_string(),
        ));
    }
    if let Some(ref tools) = payload.tools
        && !tools.is_null()
    {
        canonical_prompt.push_str(&format!("\n[tools:{tools}]"));
    }

    // Enforce 8,192-token context window guard before cache/upstream operations
    let token_count = canonical_prompt.split_whitespace().count();
    if token_count > MAX_SEQUENCE_TOKENS {
        return Err(ProxyError::InvalidRequest(format!(
            "Prompt exceeds maximum sequence length of {MAX_SEQUENCE_TOKENS} tokens (got {token_count})"
        )));
    }

    let prompt_hash = xxh3_64(canonical_prompt.trim().to_lowercase().as_bytes());
    let exact_key = format!("exact_chat:{}:{:016x}", sanitized_model, prompt_hash);
    let index_name = format!("chat_cache:{}", sanitized_model);

    let effective_threshold = headers
        .get("KacheDB-Threshold")
        .or_else(|| headers.get("X-KacheDB-Threshold"))
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|&t| (0.0..=1.0).contains(&t))
        .unwrap_or(state.config.similarity_threshold);

    let stream = payload.stream.unwrap_or(false);
    let prompt_tokens = (canonical_prompt.len() / 4).max(1) as u64;

    // =========================================================================
    // Tier 1: SwissTable Exact Hash Short-Circuit (< 100 µs)
    // =========================================================================
    if let Ok(Some(cached_payload)) = state.kachedb.get(&exact_key).await {
        let elapsed_us = t0.elapsed().as_micros() as f64;
        let completion_tokens = (cached_payload.len() / 4).max(1) as u64;
        info!(
            "⚡ [TIER-1 EXACT HIT: ANTHROPIC] Model: {}, Latency: {:.1}µs, Prompt: '{}'",
            sanitized_model,
            elapsed_us,
            truncate(&canonical_prompt, 40)
        );

        state
            .telemetry
            .record_exact_hit(&model, prompt_tokens, completion_tokens, elapsed_us);

        return Ok(build_cached_anthropic_response(
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
        let completion_tokens = (hit.payload.len() / 4).max(1) as u64;
        info!(
            "🎯 [TIER-2 SEMANTIC HIT: ANTHROPIC] Model: {}, Score: {:.3}, Latency: {:.1}µs, Prompt: '{}'",
            sanitized_model,
            hit.score,
            elapsed_us,
            truncate(&canonical_prompt, 40)
        );

        state
            .telemetry
            .record_semantic_hit(&model, prompt_tokens, completion_tokens, elapsed_us);

        // Asynchronously register prompt shortcut into Tier 1 SwissTable
        let kachedb_clone = state.kachedb.clone();
        let exact_key_clone = exact_key.clone();
        let payload_clone = hit.payload.clone();
        let ttl = state.config.ttl_seconds;
        tokio::spawn(async move {
            if let Err(e) = kachedb_clone
                .set(&exact_key_clone, &payload_clone, Some(ttl))
                .await
            {
                warn!("Failed to promote Anthropic semantic hit to Tier-1 cache: {e}");
            }
        });

        let score_str = format!("{:.3}", hit.score);
        let distance = (1.0 - hit.score).max(0.0);
        return Ok(build_cached_anthropic_response(
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
        "💨 [CACHE MISS: ANTHROPIC] Model: {}, Forwarding to upstream: {}",
        sanitized_model, state.config.upstream_url
    );

    let is_anthropic_upstream = state.config.upstream_url.contains("anthropic.com");

    if is_anthropic_upstream {
        // Native Anthropic Forwarding
        let upstream_endpoint = format!(
            "{}/messages",
            state.config.upstream_url.trim_end_matches('/')
        );

        let mut req_builder = state.http_client.post(&upstream_endpoint).json(&payload);

        if let Some(auth) = headers.get("Authorization") {
            req_builder = req_builder.header("Authorization", auth);
        } else if let Some(key) = headers.get("x-api-key") {
            req_builder = req_builder.header("x-api-key", key);
        } else if let Some(ref key) = state.config.upstream_api_key {
            req_builder = req_builder.header("x-api-key", key);
        }

        if let Some(ver) = headers.get("anthropic-version") {
            req_builder = req_builder.header("anthropic-version", ver);
        } else {
            req_builder = req_builder.header("anthropic-version", "2023-06-01");
        }

        let upstream_res = match req_builder.send().await {
            Ok(res) => res,
            Err(e) => {
                error!("Failed to connect to upstream Anthropic: {e}");
                return Err(ProxyError::Upstream(e));
            }
        };

        let status = upstream_res.status();
        if !status.is_success() {
            let err_text = upstream_res.text().await.unwrap_or_default();
            return Ok((status, [("Content-Type", "application/json")], err_text).into_response());
        }

        if !stream {
            let resp_json: Value = upstream_res.json().await?;
            let elapsed_us = t0.elapsed().as_micros() as f64;
            state.telemetry.record_miss(elapsed_us);

            let assistant_text = resp_json
                .get("content")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.first())
                .and_then(|block| block.get("text"))
                .and_then(|t| t.as_str())
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
            inject_miss_headers(&mut response, embed_elapsed_ms, elapsed_us / 1000.0);
            return Ok(response);
        } else {
            let elapsed_us = t0.elapsed().as_micros() as f64;
            state.telemetry.record_miss(elapsed_us);

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
                                if let Some(delta) = extract_anthropic_sse_delta(line) {
                                    accumulated_text.push_str(&delta);
                                }
                            }

                            // Persist accumulated completion before yielding message_stop so
                            // early client disconnect upon message_stop cannot cancel the write.
                            if !saved
                                && text_chunk.contains("message_stop")
                                && !accumulated_text.is_empty()
                            {
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
                            error!("Stream read error from Anthropic upstream: {e}");
                            break;
                        }
                    }
                }

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

            let mut builder = axum::response::Response::builder()
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
            if let Ok(val) = HeaderValue::from_str(&format!("{:.2}", elapsed_us / 1000.0)) {
                builder = builder.header("KacheDB-Latency-Ms", val);
            }

            let response = builder
                .body(axum::body::Body::from_stream(output_stream))
                .map_err(|e| {
                    ProxyError::Internal(format!("Failed to build Anthropic SSE response: {e}"))
                })?;

            return Ok(response);
        }
    }

    // =========================================================================
    // Universal Mode: Translate Anthropic Request -> Upstream OpenAI Completions
    // =========================================================================
    let upstream_endpoint = format!(
        "{}/chat/completions",
        state.config.upstream_url.trim_end_matches('/')
    );

    let mut openai_messages = Vec::new();
    if let Some(ref sys) = system_text
        && !sys.trim().is_empty()
    {
        openai_messages.push(json!({ "role": "system", "content": sys.trim() }));
    }
    for (role, content) in &messages {
        openai_messages.push(json!({ "role": role, "content": content }));
    }

    let mut openai_payload = json!({
        "model": model,
        "messages": openai_messages,
        "stream": stream,
    });
    if let Some(mt) = payload.max_tokens {
        openai_payload["max_tokens"] = json!(mt);
    }
    if let Some(temp) = payload.temperature {
        openai_payload["temperature"] = json!(temp);
    }
    if let Some(tp) = payload.top_p {
        openai_payload["top_p"] = json!(tp);
    }
    if let Some(ref tools) = payload.tools
        && !tools.is_null()
    {
        openai_payload["tools"] = tools.clone();
    }

    let mut req_builder = state
        .http_client
        .post(&upstream_endpoint)
        .json(&openai_payload);

    if let Some(auth) = headers.get("Authorization") {
        req_builder = req_builder.header("Authorization", auth);
    } else if let Some(key) = headers.get("x-api-key") {
        req_builder = req_builder.header(
            "Authorization",
            format!("Bearer {}", key.to_str().unwrap_or_default()),
        );
    } else if let Some(ref key) = state.config.upstream_api_key {
        req_builder = req_builder.header("Authorization", format!("Bearer {key}"));
    }

    let upstream_res = match req_builder.send().await {
        Ok(res) => res,
        Err(e) => {
            error!("Failed to connect to OpenAI-compatible upstream: {e}");
            return Err(ProxyError::Upstream(e));
        }
    };

    let status = upstream_res.status();
    if !status.is_success() {
        let err_text = upstream_res.text().await.unwrap_or_default();
        return Ok((status, [("Content-Type", "application/json")], err_text).into_response());
    }

    if !stream {
        let resp_json: Value = upstream_res.json().await?;
        let elapsed_us = t0.elapsed().as_micros() as f64;
        state.telemetry.record_miss(elapsed_us);

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
                assistant_text.clone(),
                state.config.ttl_seconds,
            );
        }

        let completion_tokens = (assistant_text.len() / 4).max(1) as u64;
        let anthropic_resp = json!({
            "id": format!("msg_{}", Uuid::new_v4().simple()),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{
                "type": "text",
                "text": assistant_text
            }],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens
            }
        });

        let mut response = (StatusCode::OK, Json(anthropic_resp)).into_response();
        inject_miss_headers(&mut response, embed_elapsed_ms, elapsed_us / 1000.0);
        return Ok(response);
    }

    // Translating OpenAI SSE Stream -> Anthropic SSE Stream
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
        let msg_id = format!("msg_{}", Uuid::new_v4().simple());
        let mut accumulated_text = String::new();
        let mut started = false;

        while let Some(chunk_result) = upstream_stream.next().await {
            match chunk_result {
                Ok(bytes) => {
                    let text_chunk = String::from_utf8_lossy(&bytes);
                    for line in text_chunk.lines() {
                        let trimmed = line.trim();
                        if trimmed == "data: [DONE]" {
                            continue;
                        }
                        if let Some(delta) = extract_sse_delta(trimmed) {
                            if !started {
                                started = true;
                                let msg_start = json!({
                                    "type": "message_start",
                                    "message": {
                                        "id": msg_id,
                                        "type": "message",
                                        "role": "assistant",
                                        "model": model,
                                        "content": [],
                                        "stop_reason": null,
                                        "stop_sequence": null,
                                        "usage": {
                                            "input_tokens": prompt_tokens,
                                            "output_tokens": 1
                                        }
                                    }
                                });
                                yield Ok::<_, axum::Error>(Event::default().event("message_start").data(msg_start.to_string()));

                                let block_start = json!({
                                    "type": "content_block_start",
                                    "index": 0,
                                    "content_block": { "type": "text", "text": "" }
                                });
                                yield Ok::<_, axum::Error>(Event::default().event("content_block_start").data(block_start.to_string()));
                            }

                            accumulated_text.push_str(&delta);
                            let block_delta = json!({
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": { "type": "text_delta", "text": delta }
                            });
                            yield Ok::<_, axum::Error>(Event::default().event("content_block_delta").data(block_delta.to_string()));
                        }
                    }
                }
                Err(e) => {
                    error!("Stream read error from upstream: {e}");
                    break;
                }
            }
        }

        // Emit closing events
        let block_stop = json!({ "type": "content_block_stop", "index": 0 });
        yield Ok::<_, axum::Error>(Event::default().event("content_block_stop").data(block_stop.to_string()));

        let comp_tokens = (accumulated_text.len() / 4).max(1);
        let msg_delta = json!({
            "type": "message_delta",
            "delta": { "stop_reason": "end_turn", "stop_sequence": null },
            "usage": { "output_tokens": comp_tokens }
        });
        yield Ok::<_, axum::Error>(Event::default().event("message_delta").data(msg_delta.to_string()));

        // Persist accumulated completion to KacheDB before emitting final message_stop
        // to prevent data loss if client disconnects immediately on message_stop.
        if !accumulated_text.is_empty() {
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

        let msg_stop = json!({ "type": "message_stop" });
        yield Ok::<_, axum::Error>(Event::default().event("message_stop").data(msg_stop.to_string()));
    };

    let mut response = Sse::new(output_stream).into_response();
    inject_miss_headers(&mut response, embed_elapsed_ms, elapsed_ms);
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

fn inject_miss_headers(response: &mut Response, embed_elapsed_ms: f64, latency_ms: f64) {
    let headers = response.headers_mut();
    headers.insert("KacheDB-Cache", HeaderValue::from_static("MISS"));
    headers.insert("Kache-Hit", HeaderValue::from_static("false"));
    headers.insert("Kache-Distance", HeaderValue::from_static("1.0000"));
    if let Ok(val) = HeaderValue::from_str(&format!("{:.2}", embed_elapsed_ms)) {
        headers.insert("Kache-Embedding-Time-Ms", val);
    }
    if let Ok(val) = HeaderValue::from_str(&format!("{:.2}", latency_ms)) {
        headers.insert("KacheDB-Latency-Ms", val);
    }
}

#[allow(clippy::too_many_arguments)]
fn build_cached_anthropic_response(
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
        let sse_stream =
            generate_cached_anthropic_sse_stream(model, payload, prompt_tokens, completion_tokens);
        Sse::new(sse_stream).into_response()
    } else {
        let resp_json = json!({
            "id": format!("msg_kache_{}", Uuid::new_v4().simple()),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{
                "type": "text",
                "text": payload
            }],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens
            }
        });
        (StatusCode::OK, Json(resp_json)).into_response()
    };

    inject_cache_headers(
        &mut response,
        "HIT",
        match_type,
        tier,
        score,
        distance,
        embed_elapsed_ms,
        Some(elapsed_ms),
    );
    response
}

#[allow(clippy::too_many_arguments)]
fn inject_cache_headers(
    response: &mut Response,
    cache: &'static str,
    match_type: &'static str,
    tier: &'static str,
    score: &str,
    distance: f32,
    embed_elapsed_ms: f64,
    latency_ms: Option<f64>,
) {
    let headers = response.headers_mut();
    headers.insert("KacheDB-Cache", HeaderValue::from_static(cache));
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
    if let Some(lat) = latency_ms
        && let Ok(val) = HeaderValue::from_str(&format!("{:.2}", lat))
    {
        headers.insert("KacheDB-Latency-Ms", val);
    }
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

fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        return s;
    }
    let mut end = max_len;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KacheDbClient;
    use crate::config::Config;
    use crate::embed::mock::MockEmbedder;
    use crate::telemetry::TelemetryTracker;
    use std::sync::Arc;
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

    async fn spawn_mock_upstream_openai(streaming: bool, error: bool) -> (String, u16) {
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
                            let body = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello from OpenAI stream\"}}]}\n\ndata: [DONE]\n\n";
                            let resp = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
                                body.len(),
                                body
                            );
                            let _ = socket.write_all(resp.as_bytes()).await;
                        } else {
                            let body = json!({
                                "id": "chatcmpl-mock",
                                "object": "chat.completion",
                                "created": 1234567,
                                "model": "mock-model",
                                "choices": [{
                                    "index": 0,
                                    "message": { "role": "assistant", "content": "Hello translated from OpenAI" },
                                    "finish_reason": "stop"
                                }],
                                "usage": { "prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16 }
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

    async fn spawn_mock_upstream_anthropic(streaming: bool, error: bool) -> (String, u16) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let n = socket.read(&mut buf).await.unwrap_or(0);
                    if n > 0 {
                        if error {
                            let resp = "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: 25\r\n\r\n{\"error\":\"anthropic down\"}";
                            let _ = socket.write_all(resp.as_bytes()).await;
                        } else if streaming {
                            let body = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Native Anthropic chunk\"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
                            let resp = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{}",
                                body.len(),
                                body
                            );
                            let _ = socket.write_all(resp.as_bytes()).await;
                        } else {
                            let body = json!({
                                "id": "msg_mock",
                                "type": "message",
                                "role": "assistant",
                                "model": "claude-3-5-sonnet",
                                "content": [{ "type": "text", "text": "Native Anthropic response" }],
                                "stop_reason": "end_turn",
                                "stop_sequence": null,
                                "usage": { "input_tokens": 8, "output_tokens": 4 }
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

    fn make_state(kachedb_port: u16, upstream_url: String) -> AppState {
        let config = Config {
            port: 6385,
            host: "127.0.0.1".to_string(),
            upstream_url,
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
    async fn test_anthropic_tier1_exact_hit_non_streaming() {
        let (_, kport) =
            spawn_mock_kachedb(Some("Exact Anthropic cached answer".to_string()), None).await;
        let (_, uport) = spawn_mock_upstream_openai(false, false).await;
        let state = make_state(kport, format!("http://127.0.0.1:{uport}/v1"));

        let payload = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Text("Hello Claude".to_string()),
            }],
            system: Some(SystemPrompt::String("Be helpful".to_string())),
            max_tokens: Some(1024),
            stream: Some(false),
            temperature: None,
            top_p: None,
            tools: None,
        };

        let mut headers = HeaderMap::new();
        headers.insert("KacheDB-Threshold", HeaderValue::from_static("0.85"));
        let res = messages_handler(State(state), headers, Json(payload))
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
    async fn test_anthropic_tier1_exact_hit_streaming() {
        let (_, kport) =
            spawn_mock_kachedb(Some("Exact Anthropic stream hit".to_string()), None).await;
        let (_, uport) = spawn_mock_upstream_openai(false, false).await;
        let state = make_state(kport, format!("http://127.0.0.1:{uport}/v1"));

        let payload = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Text("Stream hello".to_string()),
            }],
            system: None,
            max_tokens: Some(512),
            stream: Some(true),
            temperature: None,
            top_p: None,
            tools: None,
        };

        let res = messages_handler(State(state), HeaderMap::new(), Json(payload))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers().get("KacheDB-Cache").unwrap(), "HIT");
        assert_eq!(res.headers().get("Kache-Hit").unwrap(), "true");
    }

    #[tokio::test]
    async fn test_anthropic_tier2_semantic_hit() {
        let (_, kport) = spawn_mock_kachedb(
            None,
            Some((
                "id1".to_string(),
                0.941,
                "Semantic Anthropic cached hit".to_string(),
            )),
        )
        .await;
        let (_, uport) = spawn_mock_upstream_openai(false, false).await;
        let state = make_state(kport, format!("http://127.0.0.1:{uport}/v1"));

        let payload = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Text("Semantic query".to_string()),
            }],
            system: None,
            max_tokens: Some(512),
            stream: Some(false),
            temperature: None,
            top_p: None,
            tools: None,
        };

        let res = messages_handler(
            State(state.clone()),
            HeaderMap::new(),
            Json(payload.clone()),
        )
        .await
        .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers().get("KacheDB-Cache").unwrap(), "HIT");
        assert_eq!(res.headers().get("KacheDB-Match").unwrap(), "SEMANTIC");
        assert_eq!(res.headers().get("KacheDB-Tier").unwrap(), "2-Semantic");
        assert_eq!(res.headers().get("Kache-Hit").unwrap(), "true");
        assert_eq!(res.headers().get("Kache-Distance").unwrap(), "0.0590");
        assert!(res.headers().get("Kache-Embedding-Time-Ms").is_some());

        // Now test streaming tier 2
        let mut stream_payload = payload;
        stream_payload.stream = Some(true);
        let res_stream = messages_handler(State(state), HeaderMap::new(), Json(stream_payload))
            .await
            .unwrap();
        assert_eq!(res_stream.status(), StatusCode::OK);
        assert_eq!(res_stream.headers().get("KacheDB-Cache").unwrap(), "HIT");
        assert_eq!(res_stream.headers().get("Kache-Hit").unwrap(), "true");
    }

    #[tokio::test]
    async fn test_anthropic_cache_miss_openai_translation_non_streaming() {
        let (_, kport) = spawn_mock_kachedb(None, None).await;
        let (_, uport) = spawn_mock_upstream_openai(false, false).await;
        let state = make_state(kport, format!("http://127.0.0.1:{uport}/v1"));

        let payload = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Text("Translate this".to_string()),
            }],
            system: Some(SystemPrompt::Blocks(vec![ContentBlock {
                block_type: "text".to_string(),
                text: Some("System rule block".to_string()),
            }])),
            max_tokens: Some(256),
            stream: Some(false),
            temperature: Some(0.7),
            top_p: Some(0.9),
            tools: None,
        };

        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("anthropic-key"));
        let res = messages_handler(State(state), headers, Json(payload))
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
    async fn test_anthropic_cache_miss_openai_translation_streaming() {
        let (_, kport) = spawn_mock_kachedb(None, None).await;
        let (_, uport) = spawn_mock_upstream_openai(true, false).await;
        let state = make_state(kport, format!("http://127.0.0.1:{uport}/v1"));

        let payload = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Blocks(vec![ContentBlock {
                    block_type: "text".to_string(),
                    text: Some("Content block question".to_string()),
                }]),
            }],
            system: None,
            max_tokens: Some(256),
            stream: Some(true),
            temperature: None,
            top_p: None,
            tools: None,
        };

        let res = messages_handler(State(state), HeaderMap::new(), Json(payload))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers().get("KacheDB-Cache").unwrap(), "MISS");
        assert_eq!(res.headers().get("Kache-Hit").unwrap(), "false");
        let body_bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body_bytes);
        assert!(body_str.contains("Hello from OpenAI stream"));
        assert!(body_str.contains("message_stop"));
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    }

    #[tokio::test]
    async fn test_anthropic_cache_miss_native_anthropic_non_streaming() {
        let (_, kport) = spawn_mock_kachedb(None, None).await;
        let (_, uport) = spawn_mock_upstream_anthropic(false, false).await;
        // Upstream contains anthropic.com
        let state = make_state(kport, format!("http://127.0.0.1:{uport}/anthropic.com/v1"));

        let payload = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Text("Native Anthropic prompt".to_string()),
            }],
            system: None,
            max_tokens: Some(256),
            stream: Some(false),
            temperature: None,
            top_p: None,
            tools: None,
        };

        let mut headers = HeaderMap::new();
        headers.insert(
            "Authorization",
            HeaderValue::from_static("Bearer anthropic-token"),
        );
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let res = messages_handler(State(state), headers, Json(payload))
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
    async fn test_anthropic_cache_miss_native_anthropic_streaming() {
        let (_, kport) = spawn_mock_kachedb(None, None).await;
        let (_, uport) = spawn_mock_upstream_anthropic(true, false).await;
        let state = make_state(kport, format!("http://127.0.0.1:{uport}/anthropic.com/v1"));

        let payload = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Text("Native stream prompt".to_string()),
            }],
            system: None,
            max_tokens: Some(256),
            stream: Some(true),
            temperature: None,
            top_p: None,
            tools: None,
        };

        let res = messages_handler(State(state), HeaderMap::new(), Json(payload))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers().get("KacheDB-Cache").unwrap(), "MISS");
        assert_eq!(res.headers().get("Kache-Hit").unwrap(), "false");
        let body_bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body_bytes);
        assert!(body_str.contains("Native Anthropic chunk"));
        assert!(body_str.contains("message_stop"));
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    }

    #[tokio::test]
    async fn test_anthropic_cache_miss_native_anthropic_upstream_error() {
        let (_, kport) = spawn_mock_kachedb(None, None).await;
        let (_, uport) = spawn_mock_upstream_anthropic(false, true).await;
        let state = make_state(kport, format!("http://127.0.0.1:{uport}/anthropic.com/v1"));

        let payload = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Text("Native error prompt".to_string()),
            }],
            system: None,
            max_tokens: Some(256),
            stream: Some(false),
            temperature: None,
            top_p: None,
            tools: None,
        };

        let res = messages_handler(State(state), HeaderMap::new(), Json(payload))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_anthropic_cache_miss_upstream_error() {
        let (_, kport) = spawn_mock_kachedb(None, None).await;
        let (_, uport) = spawn_mock_upstream_openai(false, true).await;
        let state = make_state(kport, format!("http://127.0.0.1:{uport}/v1"));

        let payload = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Text("Error query".to_string()),
            }],
            system: None,
            max_tokens: Some(256),
            stream: Some(false),
            temperature: None,
            top_p: None,
            tools: None,
        };

        let res = messages_handler(State(state), HeaderMap::new(), Json(payload))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_truncate_helper() {
        assert_eq!(truncate("hello world", 5), "hello");
        assert_eq!(truncate("short", 10), "short");
    }

    #[test]
    fn test_content_block_default_type() {
        let block: ContentBlock = serde_json::from_str(r#"{"text":"sample text"}"#).unwrap();
        assert_eq!(block.block_type, "text");
        assert_eq!(block.text.as_deref(), Some("sample text"));
    }

    #[tokio::test]
    async fn test_anthropic_empty_prompt_and_headers() {
        let (_, kport) = spawn_mock_kachedb(None, None).await;
        let (_, uport) = spawn_mock_upstream_openai(false, false).await;
        let state = make_state(kport, format!("http://127.0.0.1:{uport}/v1"));

        // Empty prompt
        let empty_payload = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet".to_string(),
            messages: vec![],
            system: None,
            max_tokens: Some(256),
            stream: Some(false),
            temperature: None,
            top_p: None,
            tools: None,
        };
        let err = messages_handler(State(state.clone()), HeaderMap::new(), Json(empty_payload))
            .await
            .unwrap_err();
        assert!(matches!(err, ProxyError::InvalidRequest(_)));

        // Custom headers
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("ant-test-key"));
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        headers.insert("KacheDB-Threshold", HeaderValue::from_static("0.88"));
        headers.insert("KacheDB-Namespace", HeaderValue::from_static("team-alpha"));

        let payload = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Text("Valid prompt".to_string()),
            }],
            system: None,
            max_tokens: Some(256),
            stream: Some(false),
            temperature: None,
            top_p: None,
            tools: Some(json!([{"name": "get_weather"}])),
        };
        let res = messages_handler(State(state), headers, Json(payload))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_messages_rfc6648_kache_headers_and_8k_guard() {
        let (_, kport) = spawn_mock_kachedb(None, None).await;
        let (_, uport) = spawn_mock_upstream_openai(false, false).await;
        let state = make_state(kport, format!("http://127.0.0.1:{uport}/v1"));

        let oversized_content = vec!["token"; 8195].join(" ");
        let payload = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: MessageContent::Text(oversized_content),
            }],
            system: None,
            max_tokens: Some(256),
            stream: Some(false),
            temperature: None,
            top_p: None,
            tools: None,
        };
        let err = messages_handler(State(state), HeaderMap::new(), Json(payload))
            .await
            .unwrap_err();
        match err {
            ProxyError::InvalidRequest(msg) => assert!(msg.contains("8192")),
            other => panic!("Expected InvalidRequest, got {other:?}"),
        }
    }
}
