//! Shared application state for Axum handlers.

use crate::cache::DynKacheDbClient;
use crate::config::Config;
use crate::embed::DynEmbeddingEngine;
use crate::telemetry::TelemetryTracker;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub embedder: DynEmbeddingEngine,
    pub kachedb: DynKacheDbClient,
    pub http_client: reqwest::Client,
    pub telemetry: Arc<TelemetryTracker>,
}
