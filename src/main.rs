//! kachedb-proxy: High-Performance OpenAI-Compatible Semantic Reverse Proxy.

pub mod cache;
pub mod config;
pub mod embed;
pub mod error;
pub mod http;
pub mod state;
pub mod telemetry;

use cache::KacheDbClient;
use clap::Parser;
use config::Config;
use embed::DynEmbeddingEngine;
use embed::gemma::GemmaOnnxEmbedder;
use embed::local_onnx::LocalOnnxEmbedder;
use embed::mock::MockEmbedder;
use embed::resolver::{ModelResolutionConfig, resolve_model_paths};
use state::AppState;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use telemetry::{TelemetryTracker, generate_ascii_report};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let mut config = Config::parse();
    if config.upstream_url == "https://api.openai.com/v1"
        && let Ok(url) = std::env::var("UPSTREAM_URL")
    {
        config.upstream_url = url;
    }

    print_banner(&config);

    let (app_state, router) = build_app(&config);

    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    log_startup_info(&config, &addr);

    #[cfg(unix)]
    let _sigusr1_handle = spawn_sigusr1_listener(app_state.telemetry.clone());

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("🛑 KacheDB Reverse Proxy shut down gracefully.");
    let final_snapshot = app_state.telemetry.snapshot();
    println!("\n{}\n", generate_ascii_report(&final_snapshot));
    Ok(())
}

pub fn log_startup_info(config: &Config, addr: &SocketAddr) {
    info!("🚀 KacheDB Reverse Proxy listening on http://{addr}");
    info!("🌐 Upstream LLM target: {}", config.upstream_url);
    info!(
        "🗄️ KacheDB backend target: {}:{}",
        config.kachedb_host, config.kachedb_port
    );
    info!(
        "🎯 Semantic threshold: {:.2} | TTL: {}s | Dim: {}",
        config.similarity_threshold, config.ttl_seconds, config.embedding_dim
    );
}

#[cfg(unix)]
pub fn spawn_sigusr1_listener(telemetry: Arc<TelemetryTracker>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        use tokio::signal::unix::{SignalKind, signal};
        if let Ok(mut sigusr1) = signal(SignalKind::user_defined1()) {
            while sigusr1.recv().await.is_some() {
                let snapshot = telemetry.snapshot();
                let report = generate_ascii_report(&snapshot);
                println!("\n{report}\n");
            }
        }
    })
}

fn is_embeddinggemma_model(model_name: &str) -> bool {
    matches!(
        model_name.trim().to_ascii_lowercase().as_str(),
        "embeddinggemma-2" | "embeddinggemma" | "gemma-2" | "embeddinggemma-300m"
    )
}

pub fn setup_app_state(config: &Config) -> AppState {
    let target_dim = config.validate_embedding_dim().unwrap_or(384);

    let embedder: DynEmbeddingEngine = if config.mock_embedder {
        info!(
            "🧪 Initialized MockEmbedder (deterministic hashing mode, dim={})",
            target_dim
        );
        Arc::new(MockEmbedder::new(target_dim))
    } else if is_embeddinggemma_model(&config.embedding_model) {
        info!(
            "🧠 Initializing EmbeddingGemma 2 ONNX engine '{}' (dim={})...",
            config.embedding_model, target_dim
        );
        let resolution_cfg = ModelResolutionConfig::new(
            config.model_path.clone(),
            std::env::var("EMBEDDING_MODEL_PATH").ok(),
            PathBuf::from(".fastembed_cache"),
        );
        match resolve_model_paths(&resolution_cfg)
            .and_then(|resolved| GemmaOnnxEmbedder::try_from_resolved(&resolved, target_dim))
        {
            Ok(engine) => Arc::new(engine),
            Err(e) => {
                tracing::warn!(
                    "⚠️ Could not load EmbeddingGemma 2 ONNX model ({e}). Falling back to MockEmbedder(dim={target_dim})..."
                );
                Arc::new(MockEmbedder::new(target_dim))
            }
        }
    } else {
        info!(
            "🧠 Initializing local ONNX embedder '{}'...",
            config.embedding_model
        );
        match LocalOnnxEmbedder::try_new(&config.embedding_model) {
            Ok(engine) => Arc::new(engine),
            Err(e) => {
                tracing::warn!(
                    "⚠️ Could not load local ONNX model ({e}). Falling back to MockEmbedder(dim={target_dim})..."
                );
                Arc::new(MockEmbedder::new(target_dim))
            }
        }
    };

    let kachedb = Arc::new(KacheDbClient::new(
        config.kachedb_host.clone(),
        config.kachedb_port,
    ));

    let telemetry = Arc::new(TelemetryTracker::new());

    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap_or_default();

    AppState {
        config: Arc::new(config.clone()),
        embedder,
        kachedb,
        http_client,
        telemetry,
    }
}

pub fn build_app(config: &Config) -> (AppState, axum::Router) {
    let state = setup_app_state(config);
    let router = http::build_router(state.clone());
    (state, router)
}

fn print_banner(config: &Config) {
    println!(
        r#"
════════════════════════════════════════════════════════════════════════
  ⚡ KacheDB AI Semantic Reverse Proxy v{}
  Sub-millisecond In-Memory LLM & Vector Cache for AI Developer Tools
════════════════════════════════════════════════════════════════════════
  • Port:             {}
  • Upstream:         {}
  • KacheDB Daemon:   {}:{}
  • Similarity Gate:  {:.2}
  • Embedding Model:  {} ({}-d)
════════════════════════════════════════════════════════════════════════
"#,
        env!("CARGO_PKG_VERSION"),
        config.port,
        config.upstream_url,
        config.kachedb_host,
        config.kachedb_port,
        config.similarity_threshold,
        config.embedding_model,
        config.embedding_dim
    );
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("Received termination signal. Shutting down...");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_print_banner() {
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
        print_banner(&config);
    }

    #[test]
    fn test_build_app_state_mock() {
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
        let (state, _router) = build_app(&config);
        assert_eq!(state.embedder.dimension(), 384);
    }

    #[test]
    fn test_setup_app_state_768_dimension() {
        let config = Config {
            embedding_model: "embeddinggemma-2".to_string(),
            embedding_dim: 768,
            mock_embedder: true,
            ..Config::default()
        };
        let (state, _router) = build_app(&config);
        assert_eq!(state.embedder.dimension(), 768);

        // Also test EmbeddingGemma 2 fallback path when model_path doesn't exist
        let fallback_cfg = Config {
            embedding_model: "embeddinggemma-2".to_string(),
            embedding_dim: 768,
            mock_embedder: false,
            model_path: Some(PathBuf::from("/nonexistent/gemma2/dir")),
            ..Config::default()
        };
        let fallback_state = setup_app_state(&fallback_cfg);
        assert_eq!(fallback_state.embedder.dimension(), 768);
    }

    #[test]
    fn test_build_app_state_onnx_fallback() {
        let config = Config {
            port: 6385,
            host: "127.0.0.1".to_string(),
            upstream_url: "http://127.0.0.1:8081/v1".to_string(),
            upstream_api_key: None,
            kachedb_host: "127.0.0.1".to_string(),
            kachedb_port: 6381,
            similarity_threshold: 0.85,
            ttl_seconds: 86400,
            embedding_model: "nonexistent-model".to_string(),
            mock_embedder: false,
            ..Config::default()
        };
        // Should trigger fallback to mock embedder
        let state = setup_app_state(&config);
        assert_eq!(state.embedder.dimension(), 384);
    }

    #[test]
    fn test_log_startup_info() {
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
        let addr: SocketAddr = "127.0.0.1:6385".parse().unwrap();
        log_startup_info(&config, &addr);
    }

    #[tokio::test]
    async fn test_spawn_sigusr1_listener() {
        let tracker = Arc::new(TelemetryTracker::new());
        let handle = spawn_sigusr1_listener(tracker);
        handle.abort();
    }

    #[tokio::test]
    async fn test_shutdown_signal_wrapper() {
        tokio::select! {
            _ = shutdown_signal() => {},
            _ = tokio::time::sleep(std::time::Duration::from_millis(5)) => {}
        }
    }
}
