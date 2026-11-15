//! OpenAI /v1/embeddings handler using local on-device embedder.

use crate::embed::gemma::DEFAULT_MAX_TOKENS as MAX_SEQUENCE_TOKENS;
use crate::embed::task::{EmbeddingTask, PREFIX_DOCUMENT_INDEXING};
use crate::error::ProxyError;
use crate::state::AppState;
use axum::Json;
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use std::time::Instant;

pub async fn embeddings_handler(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<Response, ProxyError> {
    let model = payload
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(&state.config.embedding_model)
        .to_string();

    let mut input_texts = Vec::new();
    if let Some(input_val) = payload.get("input") {
        if let Some(s) = input_val.as_str() {
            input_texts.push(s.to_string());
        } else if let Some(arr) = input_val.as_array() {
            for item in arr {
                if let Some(s) = item.as_str() {
                    input_texts.push(s.to_string());
                }
            }
        }
    }

    if input_texts.is_empty() {
        return Err(ProxyError::InvalidRequest(
            "Missing 'input' text".to_string(),
        ));
    }

    let mut data = Vec::new();
    let mut total_tokens = 0;
    let embed_t0 = Instant::now();

    for (index, text) in input_texts.iter().enumerate() {
        let token_count = text.split_whitespace().count();
        if token_count > MAX_SEQUENCE_TOKENS {
            return Err(ProxyError::InvalidRequest(format!(
                "Input at index {index} exceeds maximum sequence length of {MAX_SEQUENCE_TOKENS} tokens (got {token_count})"
            )));
        }

        let vec = encode_document_indexing(&state, text)?;
        total_tokens += (text.len() / 4).max(1);
        data.push(json!({
            "object": "embedding",
            "index": index,
            "embedding": vec
        }));
    }

    let embed_elapsed_ms = embed_t0.elapsed().as_secs_f64() * 1000.0;

    let resp_json = json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    });

    let mut response = (StatusCode::OK, Json(resp_json)).into_response();
    if let Ok(val) = HeaderValue::from_str(&format!("{:.2}", embed_elapsed_ms)) {
        response
            .headers_mut()
            .insert("Kache-Embedding-Time-Ms", val);
    }

    Ok(response)
}

fn encode_document_indexing(state: &AppState, text: &str) -> Result<Vec<f32>, ProxyError> {
    if state.config.indexing_prefix == PREFIX_DOCUMENT_INDEXING {
        state
            .embedder
            .encode_with_task(text, EmbeddingTask::DocumentIndexing)
    } else if text.starts_with(&state.config.indexing_prefix) {
        state.embedder.encode(text)
    } else {
        let prefixed = format!("{}{}", state.config.indexing_prefix, text);
        state.embedder.encode(&prefixed)
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

    fn test_state() -> AppState {
        let config = Config {
            port: 6385,
            host: "127.0.0.1".to_string(),
            upstream_url: "http://127.0.0.1:8081/v1".to_string(),
            upstream_api_key: None,
            kachedb_host: "127.0.0.1".to_string(),
            kachedb_port: 6381,
            similarity_threshold: 0.85,
            ttl_seconds: 86400,
            embedding_model: "bge-small".to_string(),
            mock_embedder: true,
            ..Config::default()
        };
        AppState {
            config: Arc::new(config),
            embedder: Arc::new(MockEmbedder::default()),
            kachedb: Arc::new(KacheDbClient::new("127.0.0.1".to_string(), 6381)),
            http_client: reqwest::Client::new(),
            telemetry: Arc::new(TelemetryTracker::new()),
        }
    }

    #[tokio::test]
    async fn test_embeddings_single_string() {
        let state = test_state();
        let payload = json!({
            "model": "text-embedding-3-small",
            "input": "hello world"
        });
        let resp = embeddings_handler(State(state), Json(payload))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get("Kache-Embedding-Time-Ms").is_some());
    }

    #[tokio::test]
    async fn test_embeddings_array_input() {
        let state = test_state();
        let payload = json!({
            "input": ["alpha", "beta"]
        });
        let resp = embeddings_handler(State(state), Json(payload))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_embeddings_missing_input() {
        let state = test_state();
        let payload = json!({
            "model": "test"
        });
        let err = embeddings_handler(State(state.clone()), Json(payload))
            .await
            .unwrap_err();
        match err {
            ProxyError::InvalidRequest(msg) => assert!(msg.contains("Missing 'input'")),
            _ => panic!("Expected InvalidRequest"),
        }

        let payload_empty_arr = json!({
            "model": "test",
            "input": [123, true]
        });
        let err2 = embeddings_handler(State(state), Json(payload_empty_arr))
            .await
            .unwrap_err();
        match err2 {
            ProxyError::InvalidRequest(msg) => assert!(msg.contains("Missing 'input'")),
            _ => panic!("Expected InvalidRequest"),
        }
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
    async fn test_embeddings_indexing_prefix_and_8k_guard() {
        let mut state = test_state();
        let recorded = Arc::new(Mutex::new(Vec::new()));
        state.embedder = Arc::new(RecordingEmbedder {
            recorded: recorded.clone(),
        });
        let mut cfg = (*state.config).clone();
        cfg.indexing_prefix = "custom_indexing_prefix: ".to_string();
        state.config = Arc::new(cfg);

        let payload = json!({
            "model": "embeddinggemma-2",
            "input": "Document content to index"
        });
        let resp = embeddings_handler(State(state.clone()), Json(payload))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        {
            let captured = recorded.lock().unwrap();
            assert_eq!(captured.len(), 1);
            assert_eq!(
                captured[0],
                "custom_indexing_prefix: Document content to index"
            );
        }

        let oversized = vec!["word"; 8195].join(" ");
        let payload_oversized = json!({
            "model": "embeddinggemma-2",
            "input": oversized
        });
        let err = embeddings_handler(State(state), Json(payload_oversized))
            .await
            .unwrap_err();
        match err {
            ProxyError::InvalidRequest(msg) => assert!(msg.contains("8192")),
            other => panic!("Expected InvalidRequest, got {other:?}"),
        }
    }
}
