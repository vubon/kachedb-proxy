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
use embed::local_onnx::LocalOnnxEmbedder;
use embed::mock::MockEmbedder;
use state::AppState;
use std::net::SocketAddr;
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
        "🎯 Semantic threshold: {:.2} | TTL: {}s",
        config.similarity_threshold, config.ttl_seconds
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

pub fn setup_app_state(config: &Config) -> AppState {
    let embedder: DynEmbeddingEngine = if config.mock_embedder {
        info!("🧪 Initialized MockEmbedder (deterministic hashing mode)");
        Arc::new(MockEmbedder::default())
    } else {
        info!(
            "🧠 Initializing local ONNX embedder '{}'...",
            config.embedding_model
        );
        match LocalOnnxEmbedder::try_new(&config.embedding_model) {
            Ok(engine) => Arc::new(engine),
            Err(e) => {
                tracing::warn!(
                    "⚠️ Could not load local ONNX model ({e}). Falling back to MockEmbedder..."
                );
                Arc::new(MockEmbedder::default())
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
  • Embedding Model:  {}
════════════════════════════════════════════════════════════════════════
"#,
        env!("CARGO_PKG_VERSION"),
        config.port,
        config.upstream_url,
        config.kachedb_host,
        config.kachedb_port,
        config.similarity_threshold,
        config.embedding_model
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
        };
        let (state, _router) = build_app(&config);
        assert_eq!(state.embedder.dimension(), 384);
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
