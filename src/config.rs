//! CLI and runtime configuration for kachedb-proxy.

use clap::Parser;

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

    /// Local on-device embedding model name
    #[arg(long, env = "EMBEDDING_MODEL", default_value = "bge-small-en-v1.5")]
    pub embedding_model: String,

    /// Enable deterministic mock embedder for rapid unit testing without downloading ONNX models
    #[arg(long, env = "MOCK_EMBEDDER", default_value_t = false)]
    pub mock_embedder: bool,
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
    }
}
