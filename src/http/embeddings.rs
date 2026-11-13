//! OpenAI /v1/embeddings handler using local on-device embedder.

use crate::error::ProxyError;
use crate::state::AppState;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

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

    for (index, text) in input_texts.iter().enumerate() {
        let vec = state.embedder.encode(text)?;
        total_tokens += (text.len() / 4).max(1);
        data.push(json!({
            "object": "embedding",
            "index": index,
            "embedding": vec
        }));
    }

    let resp_json = json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    });

    Ok((StatusCode::OK, Json(resp_json)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KacheDbClient;
    use crate::config::Config;
    use crate::embed::mock::MockEmbedder;
    use crate::telemetry::TelemetryTracker;
    use std::sync::Arc;

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
}
