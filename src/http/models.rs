//! Model listing, health checks, and telemetry endpoints.

use crate::state::AppState;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

pub async fn models_handler() -> Response {
    let resp = json!({
        "object": "list",
        "data": [
            {
                "id": "claude-sonnet-5-5",
                "object": "model",
                "created": 1740000000,
                "owned_by": "anthropic"
            },
            {
                "id": "claude-opus-5-5",
                "object": "model",
                "created": 1740000000,
                "owned_by": "anthropic"
            },
            {
                "id": "o3-mini",
                "object": "model",
                "created": 1738000000,
                "owned_by": "openai"
            },
            {
                "id": "gpt-4.5",
                "object": "model",
                "created": 1740500000,
                "owned_by": "openai"
            },
            {
                "id": "deepseek-r1",
                "object": "model",
                "created": 1737500000,
                "owned_by": "deepseek"
            },
            {
                "id": "claude-3-5-sonnet",
                "object": "model",
                "created": 1715367049,
                "owned_by": "anthropic"
            },
            {
                "id": "gpt-4o",
                "object": "model",
                "created": 1715367049,
                "owned_by": "openai"
            },
            {
                "id": "gpt-4o-mini",
                "object": "model",
                "created": 1715367049,
                "owned_by": "openai"
            },
            {
                "id": "gemini-2.0-flash",
                "object": "model",
                "created": 1734000000,
                "owned_by": "google"
            }
        ]
    });

    (StatusCode::OK, Json(resp)).into_response()
}

pub async fn health_handler(State(state): State<AppState>) -> Response {
    let resp = json!({
        "status": "HEALTHY",
        "service": "kachedb-proxy",
        "version": env!("CARGO_PKG_VERSION"),
        "kachedb_target": format!("{}:{}", state.config.kachedb_host, state.config.kachedb_port),
        "upstream_target": state.config.upstream_url,
        "embedding_model": state.config.embedding_model,
        "similarity_threshold": state.config.similarity_threshold
    });

    (StatusCode::OK, Json(resp)).into_response()
}

pub async fn metrics_handler(State(state): State<AppState>) -> Response {
    let snapshot = state.telemetry.snapshot();
    (StatusCode::OK, Json(snapshot)).into_response()
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
    async fn test_models_handler() {
        let resp = models_handler().await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_handler() {
        let state = test_state();
        let resp = health_handler(State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_handler() {
        let state = test_state();
        let resp = metrics_handler(State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
