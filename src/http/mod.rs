//! HTTP routing and middleware setup.

pub mod chat;
pub mod embeddings;
pub mod messages;
pub mod models;
pub mod savings;
pub mod sse;

use crate::state::AppState;
use axum::Router;
use axum::routing::{get, post};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

pub fn build_router(state: AppState) -> Router {
    Router::new()
        // OpenAI-Compatible Endpoints
        .route("/v1/chat/completions", post(chat::chat_completions_handler))
        .route("/chat/completions", post(chat::chat_completions_handler))
        .route("/v1/embeddings", post(embeddings::embeddings_handler))
        .route("/embeddings", post(embeddings::embeddings_handler))
        .route("/v1/models", get(models::models_handler))
        .route("/models", get(models::models_handler))
        // Anthropic Messages API Endpoints
        .route("/v1/messages", post(messages::messages_handler))
        .route("/messages", post(messages::messages_handler))
        // Observability & Telemetry
        .route("/health", get(models::health_handler))
        .route("/metrics", get(models::metrics_handler))
        .route("/v1/savings", get(savings::savings_handler))
        .route("/savings", get(savings::savings_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KacheDbClient;
    use crate::config::Config;
    use crate::embed::mock::MockEmbedder;
    use crate::telemetry::TelemetryTracker;
    use std::sync::Arc;

    #[test]
    fn test_build_router() {
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
        let state = AppState {
            config: Arc::new(config),
            embedder: Arc::new(MockEmbedder::default()),
            kachedb: Arc::new(KacheDbClient::new("127.0.0.1".to_string(), 6381)),
            http_client: reqwest::Client::new(),
            telemetry: Arc::new(TelemetryTracker::new()),
        };
        let _router = build_router(state);
    }
}
