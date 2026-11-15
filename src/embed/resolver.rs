//! Model weights and tokenizer path resolver for EmbeddingGemma 2.
//!
//! Provides dual-mode weights resolution following strict precedence:
//! 1. Explicit CLI override (`--model-path`)
//! 2. Environment variable override (`EMBEDDING_MODEL_PATH`)
//! 3. Default Hugging Face local cache (`.fastembed_cache`)

use crate::error::ProxyError;
use std::path::{Path, PathBuf};

/// The origin source mode of resolved model artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSourceMode {
    /// Explicitly configured via CLI argument `--model-path`.
    ExplicitPath,
    /// Configured via environment variable `EMBEDDING_MODEL_PATH`.
    EnvVar,
    /// Resolved from default Hugging Face local cache directory.
    HuggingFaceCache,
}

/// Configuration inputs for resolving model artifact paths.
#[derive(Debug, Clone)]
pub struct ModelResolutionConfig {
    /// Optional explicit directory path passed via `--model-path`.
    pub cli_path: Option<PathBuf>,
    /// Optional environment variable value from `EMBEDDING_MODEL_PATH`.
    pub env_var: Option<String>,
    /// Default Hugging Face cache directory path (e.g. `~/.fastembed_cache`).
    pub cache_dir: PathBuf,
}

impl ModelResolutionConfig {
    /// Construct a new resolution configuration.
    pub fn new(cli_path: Option<PathBuf>, env_var: Option<String>, cache_dir: PathBuf) -> Self {
        Self {
            cli_path,
            env_var,
            cache_dir,
        }
    }
}

/// Verified artifact paths ready for ONNX Runtime session and tokenizer initialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    /// Path to verified `model.onnx` regular file.
    pub onnx_model_path: PathBuf,
    /// Path to verified `tokenizer.json` regular file.
    pub tokenizer_path: PathBuf,
    /// Origin source mode used to resolve this model.
    pub source_mode: ModelSourceMode,
}

/// Resolve and validate paths for model weights and tokenizer artifacts.
///
/// Precedence hierarchy (BR2.1):
/// 1. If `cli_path` is provided and non-empty, use it.
/// 2. Else if `env_var` is provided and non-empty, use it.
/// 3. Else fallback to `cache_dir`.
///
/// Invariants:
/// - Fails fast with [`ProxyError::Embedding`] if candidate directory or required artifacts are missing (BR2.2, BR2.5).
/// - Guarantees zero network egress when local directory overrides are active (BR2.3).
/// - Pure, zero-panic execution returning typed `Result` (NFR6.1).
pub fn resolve_model_paths(config: &ModelResolutionConfig) -> Result<ResolvedModel, ProxyError> {
    if let Some(ref cli_path) = config.cli_path
        && !cli_path.as_os_str().is_empty()
    {
        return validate_model_artifacts(cli_path, ModelSourceMode::ExplicitPath);
    }

    if let Some(ref env_val) = config.env_var {
        let trimmed = env_val.trim();
        if !trimmed.is_empty() {
            let env_path = PathBuf::from(trimmed);
            return validate_model_artifacts(&env_path, ModelSourceMode::EnvVar);
        }
    }

    validate_model_artifacts(&config.cache_dir, ModelSourceMode::HuggingFaceCache)
}

/// Inspect candidate directory and verify required artifacts exist, are regular files, and are readable.
fn validate_model_artifacts(
    candidate_dir: &Path,
    source_mode: ModelSourceMode,
) -> Result<ResolvedModel, ProxyError> {
    if !candidate_dir.exists() {
        return Err(ProxyError::Embedding(format!(
            "Model directory does not exist: {}",
            candidate_dir.display()
        )));
    }

    if !candidate_dir.is_dir() {
        return Err(ProxyError::Embedding(format!(
            "Model path is not a directory: {}",
            candidate_dir.display()
        )));
    }

    let onnx_path = candidate_dir.join("model.onnx");
    let tokenizer_path = candidate_dir.join("tokenizer.json");

    let onnx_meta = std::fs::metadata(&onnx_path).map_err(|e| {
        ProxyError::Embedding(format!(
            "Required artifact 'model.onnx' missing or unreadable in '{}': {e}",
            candidate_dir.display()
        ))
    })?;

    if !onnx_meta.is_file() {
        return Err(ProxyError::Embedding(format!(
            "Artifact 'model.onnx' in '{}' is not a regular file",
            candidate_dir.display()
        )));
    }

    let tokenizer_meta = std::fs::metadata(&tokenizer_path).map_err(|e| {
        ProxyError::Embedding(format!(
            "Required artifact 'tokenizer.json' missing or unreadable in '{}': {e}",
            candidate_dir.display()
        ))
    })?;

    if !tokenizer_meta.is_file() {
        return Err(ProxyError::Embedding(format!(
            "Artifact 'tokenizer.json' in '{}' is not a regular file",
            candidate_dir.display()
        )));
    }

    Ok(ResolvedModel {
        onnx_model_path: onnx_path,
        tokenizer_path,
        source_mode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new() -> Self {
            let path = std::env::temp_dir()
                .join(format!("kachedb-resolver-test-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&path).expect("Failed to create test tempdir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn create_dummy_artifacts(dir: &Path) {
        File::create(dir.join("model.onnx")).expect("Failed to create dummy model.onnx");
        File::create(dir.join("tokenizer.json")).expect("Failed to create dummy tokenizer.json");
    }

    #[test]
    fn test_resolve_cli_path_precedence() {
        let cli_dir = TestDir::new();
        let env_dir = TestDir::new();
        let cache_dir = TestDir::new();

        create_dummy_artifacts(cli_dir.path());
        create_dummy_artifacts(env_dir.path());
        create_dummy_artifacts(cache_dir.path());

        let config = ModelResolutionConfig::new(
            Some(cli_dir.path().to_path_buf()),
            Some(env_dir.path().to_string_lossy().to_string()),
            cache_dir.path().to_path_buf(),
        );

        let resolved = resolve_model_paths(&config).expect("Resolution should succeed");
        assert_eq!(resolved.source_mode, ModelSourceMode::ExplicitPath);
        assert_eq!(resolved.onnx_model_path, cli_dir.path().join("model.onnx"));
        assert_eq!(
            resolved.tokenizer_path,
            cli_dir.path().join("tokenizer.json")
        );
    }

    #[test]
    fn test_resolve_env_var_precedence() {
        let env_dir = TestDir::new();
        let cache_dir = TestDir::new();

        create_dummy_artifacts(env_dir.path());
        create_dummy_artifacts(cache_dir.path());

        let config = ModelResolutionConfig::new(
            None,
            Some(env_dir.path().to_string_lossy().to_string()),
            cache_dir.path().to_path_buf(),
        );

        let resolved = resolve_model_paths(&config).expect("Resolution should succeed");
        assert_eq!(resolved.source_mode, ModelSourceMode::EnvVar);
        assert_eq!(resolved.onnx_model_path, env_dir.path().join("model.onnx"));
        assert_eq!(
            resolved.tokenizer_path,
            env_dir.path().join("tokenizer.json")
        );
    }

    #[test]
    fn test_resolve_default_cache_fallback() {
        let cache_dir = TestDir::new();
        create_dummy_artifacts(cache_dir.path());

        let config = ModelResolutionConfig::new(None, None, cache_dir.path().to_path_buf());

        let resolved = resolve_model_paths(&config).expect("Resolution should succeed");
        assert_eq!(resolved.source_mode, ModelSourceMode::HuggingFaceCache);
        assert_eq!(
            resolved.onnx_model_path,
            cache_dir.path().join("model.onnx")
        );
        assert_eq!(
            resolved.tokenizer_path,
            cache_dir.path().join("tokenizer.json")
        );
    }

    #[test]
    fn test_resolve_empty_env_var_falls_back_to_cache() {
        let cache_dir = TestDir::new();
        create_dummy_artifacts(cache_dir.path());

        let config = ModelResolutionConfig::new(
            None,
            Some("   ".to_string()),
            cache_dir.path().to_path_buf(),
        );

        let resolved = resolve_model_paths(&config).expect("Resolution should succeed");
        assert_eq!(resolved.source_mode, ModelSourceMode::HuggingFaceCache);
    }

    #[test]
    fn test_resolve_missing_model_onnx_error() {
        let cli_dir = TestDir::new();
        // Create only tokenizer.json
        File::create(cli_dir.path().join("tokenizer.json"))
            .expect("Failed to create tokenizer.json");

        let config = ModelResolutionConfig::new(
            Some(cli_dir.path().to_path_buf()),
            None,
            PathBuf::from("/tmp/unused_cache"),
        );

        let err = resolve_model_paths(&config).expect_err("Should fail due to missing model.onnx");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("model.onnx"),
            "Error should mention model.onnx: {err_msg}"
        );
    }

    #[test]
    fn test_resolve_missing_tokenizer_json_error() {
        let cli_dir = TestDir::new();
        // Create only model.onnx
        File::create(cli_dir.path().join("model.onnx")).expect("Failed to create model.onnx");

        let config = ModelResolutionConfig::new(
            Some(cli_dir.path().to_path_buf()),
            None,
            PathBuf::from("/tmp/unused_cache"),
        );

        let err =
            resolve_model_paths(&config).expect_err("Should fail due to missing tokenizer.json");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("tokenizer.json"),
            "Error should mention tokenizer.json: {err_msg}"
        );
    }

    #[test]
    fn test_resolve_nonexistent_directory_error() {
        let nonexistent = PathBuf::from("/nonexistent/directory/for/model_path_test_12345");
        let config =
            ModelResolutionConfig::new(Some(nonexistent), None, PathBuf::from("/tmp/unused_cache"));

        let err = resolve_model_paths(&config).expect_err("Should fail on non-existent dir");
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("does not exist"),
            "Error message should mention does not exist: {err_msg}"
        );
    }

    #[test]
    fn test_resolve_air_gapped_zero_egress() {
        // In air-gapped environments, invalid paths fail fast immediately without network attempts.
        let invalid_local = PathBuf::from("/nonexistent/air_gapped_model_path");
        let config = ModelResolutionConfig::new(
            Some(invalid_local),
            None,
            PathBuf::from("/tmp/air_gapped_cache"),
        );

        let err = resolve_model_paths(&config).expect_err("Air-gapped invalid path must fail fast");
        assert!(matches!(err, ProxyError::Embedding(_)));
    }
}
