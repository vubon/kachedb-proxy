//! Token savings and cache economics endpoint.

use crate::state::AppState;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SavingsResponse {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub hit_ratio_percent: f64,
    pub exact_hits: u64,
    pub semantic_hits: u64,
    pub prompt_tokens_saved: u64,
    pub completion_tokens_saved: u64,
    pub total_tokens_saved: u64,
    pub avg_upstream_latency_ms: f64,
    pub avg_cache_latency_ms: f64,
}

pub async fn savings_handler(State(state): State<AppState>) -> Response {
    let snap = state.telemetry.snapshot();
    let resp = SavingsResponse {
        total_requests: snap.total_requests,
        cache_hits: snap.cache_hits,
        hit_ratio_percent: snap.hit_ratio_percent,
        exact_hits: snap.exact_hits,
        semantic_hits: snap.semantic_hits,
        prompt_tokens_saved: snap.prompt_tokens_saved,
        completion_tokens_saved: snap.completion_tokens_saved,
        total_tokens_saved: snap.total_tokens_saved,
        avg_upstream_latency_ms: snap.avg_upstream_latency_ms,
        avg_cache_latency_ms: snap.avg_cache_latency_ms,
    };
    (StatusCode::OK, Json(resp)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KacheDbClient;
    use crate::config::Config;
    use crate::embed::mock::MockEmbedder;
    use crate::telemetry::TelemetryTracker;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_savings_handler() {
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

        let resp = savings_handler(State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
