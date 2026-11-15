//! CLI and runtime configuration for kachedb-proxy.

use crate::embed::task::{PREFIX_DOCUMENT_INDEXING, PREFIX_SEARCH_QUERY};
use crate::error::ProxyError;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "kachedb-proxy",
    author = "Vubon Roy <vubon.roy@gmail.com>",
    version,
    about = "High-performance OpenAI and Anthropic compatible semantic reverse proxy powered by KacheDB SIMD vector cache"
)]
pub struct Config {
    /// Port for the HTTP proxy to listen on
    #[arg(short = 'p', long, env = "KACHEDB_PROXY_PORT", default_value_t = 6385)]
    pub port: u16,

    /// Host IP address to bind to
    #[arg(long, env = "KACHEDB_PROXY_HOST", default_value = "0.0.0.0")]
    pub host: String,

    /// Upstream OpenAI-compatible API base URL (e.g. https://api.openai.com/v1 or http://localhost:11434/v1)
    #[arg(
        short = 'u',
        long,
        visible_alias = "upstream",
        env = "UPSTREAM_BASE_URL",
        default_value = "https://api.openai.com/v1"
    )]
    pub upstream_url: String,

    /// Default upstream API key if not provided by client in Authorization header
    #[arg(short = 'k', long, visible_alias = "api-key", env = "UPSTREAM_API_KEY")]
    pub upstream_api_key: Option<String>,

    /// KacheDB daemon host
    #[arg(long, env = "KACHEDB_HOST", default_value = "127.0.0.1")]
    pub kachedb_host: String,

    /// KacheDB daemon port
    #[arg(long, env = "KACHEDB_PORT", default_value_t = 6379)]
    pub kachedb_port: u16,

    /// Cosine similarity threshold for semantic cache hits (0.0 to 1.0)
    #[arg(
        short = 't',
        long,
        visible_alias = "threshold",
        env = "SIMILARITY_THRESHOLD",
        default_value_t = 0.85
    )]
    pub similarity_threshold: f32,

    /// TTL (time-to-live) in seconds for cached completions (default: 86400 / 24 hours)
    #[arg(
        long,
        visible_alias = "ttl",
        env = "CACHE_TTL_SECONDS",
        default_value_t = 86400
    )]
    pub ttl_seconds: u64,

    /// Local on-device embedding model name (e.g. "bge-small-en-v1.5" or "embeddinggemma-2")
    #[arg(long, env = "EMBEDDING_MODEL", default_value = "bge-small-en-v1.5")]
    pub embedding_model: String,

    /// Enable deterministic mock embedder for rapid unit testing without downloading ONNX models
    #[arg(long, env = "MOCK_EMBEDDER", default_value_t = false)]
    pub mock_embedder: bool,

    /// Target Matryoshka Representation Learning (MRL) embedding dimension (384 or 768)
    #[arg(long, env = "EMBEDDING_DIM", default_value_t = 384)]
    pub embedding_dim: usize,

    /// Explicit directory path to offline ONNX model and tokenizer artifacts
    #[arg(long, env = "EMBEDDING_MODEL_PATH")]
    pub model_path: Option<PathBuf>,

    /// Asymmetric task prefix prepended to chat and message search queries
    #[arg(long, env = "SEARCH_PREFIX", default_value = PREFIX_SEARCH_QUERY)]
    pub search_prefix: String,

    /// Asymmetric task prefix prepended to document indexing requests (/v1/embeddings)
    #[arg(long, env = "INDEXING_PREFIX", default_value = PREFIX_DOCUMENT_INDEXING)]
    pub indexing_prefix: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: 6385,
            host: "0.0.0.0".to_string(),
            upstream_url: "https://api.openai.com/v1".to_string(),
            upstream_api_key: None,
            kachedb_host: "127.0.0.1".to_string(),
            kachedb_port: 6379,
            similarity_threshold: 0.85,
            ttl_seconds: 86400,
            embedding_model: "bge-small-en-v1.5".to_string(),
            mock_embedder: false,
            embedding_dim: 384,
            model_path: None,
            search_prefix: PREFIX_SEARCH_QUERY.to_string(),
            indexing_prefix: PREFIX_DOCUMENT_INDEXING.to_string(),
        }
    }
}

impl Config {
    /// Validates that `embedding_dim` is a supported MRL dimension (`384` or `768`).
    pub fn validate_embedding_dim(&self) -> Result<usize, ProxyError> {
        match self.embedding_dim {
            384 | 768 => Ok(self.embedding_dim),
            other => Err(ProxyError::Embedding(format!(
                "Unsupported embedding dimension {other}: expected 384 or 768"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_flags_and_aliases() {
        let cfg = Config::try_parse_from([
            "kachedb-proxy",
            "--upstream",
            "http://localhost:8081/v1",
            "--api-key",
            "sk-alias-test",
            "--threshold",
            "0.91",
            "--ttl",
            "7200",
        ])
        .unwrap();
        assert_eq!(cfg.upstream_url, "http://localhost:8081/v1");
        assert_eq!(cfg.upstream_api_key.as_deref(), Some("sk-alias-test"));
        assert!((cfg.similarity_threshold - 0.91).abs() < f32::EPSILON);
        assert_eq!(cfg.ttl_seconds, 7200);
        assert_eq!(cfg.embedding_dim, 384);
        assert_eq!(cfg.search_prefix, PREFIX_SEARCH_QUERY);
        assert_eq!(cfg.indexing_prefix, PREFIX_DOCUMENT_INDEXING);
    }

    #[test]
    fn test_cli_embeddinggemma_flags_and_defaults() {
        let cfg = Config::try_parse_from([
            "kachedb-proxy",
            "--embedding-model",
            "embeddinggemma-2",
            "--embedding-dim",
            "768",
            "--model-path",
            "/opt/models/gemma2",
            "--search-prefix",
            "custom_query: ",
            "--indexing-prefix",
            "custom_doc: ",
        ])
        .unwrap();
        assert_eq!(cfg.embedding_model, "embeddinggemma-2");
        assert_eq!(cfg.embedding_dim, 768);
        assert_eq!(
            cfg.model_path.as_deref(),
            Some(std::path::Path::new("/opt/models/gemma2"))
        );
        assert_eq!(cfg.search_prefix, "custom_query: ");
        assert_eq!(cfg.indexing_prefix, "custom_doc: ");
    }

    #[test]
    fn test_config_validate_embedding_dim() {
        let mut cfg = Config::default();
        assert_eq!(cfg.validate_embedding_dim().unwrap(), 384);

        cfg.embedding_dim = 768;
        assert_eq!(cfg.validate_embedding_dim().unwrap(), 768);

        cfg.embedding_dim = 512;
        let err = cfg.validate_embedding_dim().unwrap_err();
        assert!(err.to_string().contains("512"));
    }
}
