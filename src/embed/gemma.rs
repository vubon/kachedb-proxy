//! In-process EmbeddingGemma 2 ONNX Runtime inference engine (`u3-gemma-engine`).
//!
//! Provides 8,192-token context tokenization via Hugging Face `tokenizers`,
//! bounded concurrency execution via `tokio::sync::Semaphore` (maintaining $< 420\text{ MB}$ RSS),
//! attention-masked mean pooling, and Matryoshka Representation Learning (MRL)
//! dimension slicing (384-d default, 768-d optional) with L2 normalization.

use crate::embed::EmbeddingEngine;
use crate::embed::mrl::slice_and_normalize;
use crate::embed::resolver::ResolvedModel;
use crate::embed::task::{EmbeddingTask, format_task_prompt};
use crate::error::ProxyError;
use ndarray::Array2;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Semaphore;

/// Default maximum token sequence length for EmbeddingGemma 2 (8,192 tokens).
pub const DEFAULT_MAX_TOKENS: usize = 8192;

/// Default target output dimension using Matryoshka slicing (384 dimensions).
pub const DEFAULT_GEMMA_DIMENSION: usize = 384;

/// Native full output dimension of EmbeddingGemma 2 (768 dimensions).
pub const NATIVE_GEMMA_DIMENSION: usize = 768;

/// Default maximum concurrent ONNX forward passes to bound container RSS below 420 MB.
pub const DEFAULT_CONCURRENCY_LIMIT: usize = 2;

/// Default number of intra-op threads per ONNX Runtime session.
pub const DEFAULT_INTRA_THREADS: usize = 4;

/// Runtime configuration for initializing [`GemmaOnnxEmbedder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GemmaEngineConfig {
    /// Verified path to `model.onnx`.
    pub model_path: PathBuf,
    /// Verified path to `tokenizer.json`.
    pub tokenizer_path: PathBuf,
    /// Target output vector dimension (384 default or 768 full).
    pub dimension: usize,
    /// Maximum token sequence length before fail-fast rejection (default: 8,192).
    pub max_tokens: usize,
    /// Maximum concurrent ONNX forward executions (default: 2).
    pub concurrency_limit: usize,
}

impl GemmaEngineConfig {
    /// Create a new [`GemmaEngineConfig`] with canonical defaults (384-d, 8,192 max tokens, 2 permits).
    pub fn new(model_path: PathBuf, tokenizer_path: PathBuf) -> Self {
        Self {
            model_path,
            tokenizer_path,
            dimension: DEFAULT_GEMMA_DIMENSION,
            max_tokens: DEFAULT_MAX_TOKENS,
            concurrency_limit: DEFAULT_CONCURRENCY_LIMIT,
        }
    }

    /// Override target output dimension (must be 384 or 768).
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    /// Override maximum token sequence ceiling.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Override concurrency semaphore limit.
    pub fn with_concurrency_limit(mut self, concurrency_limit: usize) -> Self {
        self.concurrency_limit = concurrency_limit;
        self
    }
}

/// Intermediate value object holding tokenized sequence arrays for ONNX execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedPrompt {
    /// 64-bit integer token identifiers.
    pub input_ids: Vec<i64>,
    /// Binary attention mask (`1` for real tokens, `0` for padding).
    pub attention_mask: Vec<i64>,
    /// Total token sequence length.
    pub sequence_length: usize,
}

/// Raw un-sliced 768-dimensional float vector extracted from the ONNX forward pass.
#[derive(Debug, Clone, PartialEq)]
pub struct RawEmbedding {
    /// Un-sliced raw embedding vector (typically 768-d).
    pub raw_vector: Vec<f32>,
    /// Native dimensionality of the raw vector.
    pub dimension: usize,
}

/// Final unit-normalized float vector sliced to target dimension (384 or 768).
#[derive(Debug, Clone, PartialEq)]
pub struct FinalEmbedding {
    /// Sliced and L2-normalized floating point vector.
    pub normalized_vector: Vec<f32>,
    /// Output dimensionality (`384` or `768`).
    pub dimension: usize,
}

/// Internal backend abstraction allowing unit testing of tokenization, 8K boundary enforcement,
/// semaphore throttling, mean pooling, and MRL slicing without requiring a 190 MB ONNX binary on disk.
pub(crate) enum SessionBackend {
    /// Production ONNX Runtime session executing `model.onnx` on CPU SIMD.
    Onnx(Arc<Session>),
    #[cfg(test)]
    /// Deterministic synthetic forward pass producing `[1, seq_len, 768]` hidden states for unit testing.
    Synthetic,
}

/// Core in-process EmbeddingGemma 2 engine implementing [`EmbeddingEngine`].
pub struct GemmaOnnxEmbedder {
    tokenizer: Tokenizer,
    backend: SessionBackend,
    semaphore: Arc<Semaphore>,
    dimension: usize,
    max_tokens: usize,
}

impl GemmaOnnxEmbedder {
    /// Initialize [`GemmaOnnxEmbedder`] from an explicit [`GemmaEngineConfig`].
    ///
    /// Validates target dimension (384 or 768), loads the Hugging Face `tokenizer.json`
    /// with truncation disabled so >8,192 token prompts are explicitly caught (`BR3.1`),
    /// and initializes the `ort::Session` with Level 3 graph optimization (`BR3.2`).
    pub fn try_new(config: GemmaEngineConfig) -> Result<Self, ProxyError> {
        if config.dimension != 384 && config.dimension != 768 {
            return Err(ProxyError::InvalidRequest(format!(
                "Unsupported EmbeddingGemma 2 dimension {}; must be 384 or 768",
                config.dimension
            )));
        }

        if config.max_tokens == 0 {
            return Err(ProxyError::InvalidRequest(
                "max_tokens must be greater than 0".to_string(),
            ));
        }

        let permits = config.concurrency_limit.max(1);

        let mut tokenizer = Tokenizer::from_file(&config.tokenizer_path).map_err(|e| {
            ProxyError::Embedding(format!(
                "Failed to load tokenizer from '{}': {e}",
                config.tokenizer_path.display()
            ))
        })?;
        // Disable silent truncation so prompts exceeding max_tokens (8,192) trigger BR3.1 HTTP 400 rejection.
        let _ = tokenizer.with_truncation(None);
        tokenizer.with_padding(None);

        let session = Session::builder()
            .map_err(|e| {
                ProxyError::Embedding(format!("Failed to create ONNX session builder: {e}"))
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                ProxyError::Embedding(format!("Failed to set ONNX optimization level: {e}"))
            })?
            .with_intra_threads(DEFAULT_INTRA_THREADS)
            .map_err(|e| ProxyError::Embedding(format!("Failed to set ONNX intra threads: {e}")))?
            .commit_from_file(&config.model_path)
            .map_err(|e| {
                ProxyError::Embedding(format!(
                    "Failed to load ONNX model from '{}': {e}",
                    config.model_path.display()
                ))
            })?;

        Ok(Self {
            tokenizer,
            backend: SessionBackend::Onnx(Arc::new(session)),
            semaphore: Arc::new(Semaphore::new(permits)),
            dimension: config.dimension,
            max_tokens: config.max_tokens,
        })
    }

    /// Initialize [`GemmaOnnxEmbedder`] directly from a Unit 2 [`ResolvedModel`] and target dimension.
    pub fn try_from_resolved(
        resolved: &ResolvedModel,
        dimension: usize,
    ) -> Result<Self, ProxyError> {
        let config = GemmaEngineConfig::new(
            resolved.onnx_model_path.clone(),
            resolved.tokenizer_path.clone(),
        )
        .with_dimension(dimension);
        Self::try_new(config)
    }

    /// Return the configured maximum sequence token limit (default: 8,192).
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Return a reference to the concurrency throttling semaphore (`BR3.3`).
    pub fn semaphore(&self) -> &Arc<Semaphore> {
        &self.semaphore
    }

    /// Tokenize input text and enforce the maximum context length rule (`BR3.1`).
    ///
    /// Fails fast with [`ProxyError::InvalidRequest`] if the token sequence exceeds `self.max_tokens`.
    pub fn tokenize_and_validate(&self, text: &str) -> Result<TokenizedPrompt, ProxyError> {
        if text.trim().is_empty() {
            return Err(ProxyError::InvalidRequest(
                "Input prompt cannot be empty".to_string(),
            ));
        }

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| ProxyError::Embedding(format!("Tokenization failed: {e}")))?;

        let ids = encoding.get_ids();
        let masks = encoding.get_attention_mask();
        let seq_len = ids.len();

        if seq_len == 0 {
            return Err(ProxyError::InvalidRequest(
                "Tokenized prompt produced zero tokens".to_string(),
            ));
        }

        if seq_len > self.max_tokens {
            return Err(ProxyError::InvalidRequest(format!(
                "Prompt exceeds maximum sequence length of {} tokens (got {} tokens)",
                self.max_tokens, seq_len
            )));
        }

        let input_ids: Vec<i64> = ids.iter().map(|&id| i64::from(id)).collect();
        let attention_mask: Vec<i64> = masks.iter().map(|&m| i64::from(m)).collect();

        Ok(TokenizedPrompt {
            input_ids,
            attention_mask,
            sequence_length: seq_len,
        })
    }

    /// Execute the forward pass under bounded concurrency (`BR3.2`, `BR3.3`) and return a 768-d [`RawEmbedding`].
    fn execute_forward(&self, tokenized: &TokenizedPrompt) -> Result<RawEmbedding, ProxyError> {
        // Acquire bounded semaphore permit to enforce < 420 MB RSS ceiling (BR3.3).
        // We spin-wait briefly if all permits are currently held in synchronous trait contexts,
        // or return a typed Embedding error if the semaphore is closed.
        let _permit = self.acquire_permit_bounded()?;

        let seq_len = tokenized.sequence_length;

        match &self.backend {
            SessionBackend::Onnx(session) => {
                let input_ids_arr =
                    Array2::from_shape_vec((1, seq_len), tokenized.input_ids.clone()).map_err(
                        |e| ProxyError::Embedding(format!("Failed to shape input_ids tensor: {e}")),
                    )?;
                let attention_mask_arr =
                    Array2::from_shape_vec((1, seq_len), tokenized.attention_mask.clone())
                        .map_err(|e| {
                            ProxyError::Embedding(format!(
                                "Failed to shape attention_mask tensor: {e}"
                            ))
                        })?;

                let input_ids_val = Value::from_array(input_ids_arr).map_err(|e| {
                    ProxyError::Embedding(format!("Failed to create ONNX input_ids value: {e}"))
                })?;
                let attention_mask_val = Value::from_array(attention_mask_arr).map_err(|e| {
                    ProxyError::Embedding(format!(
                        "Failed to create ONNX attention_mask value: {e}"
                    ))
                })?;

                let session_inputs = ort::inputs![
                    "input_ids" => input_ids_val,
                    "attention_mask" => attention_mask_val,
                ]
                .map_err(|e| {
                    ProxyError::Embedding(format!("Failed to prepare ONNX inputs: {e}"))
                })?;

                let (flat_data, shape_dims) = {
                    let outputs = session.run(session_inputs).map_err(|e| {
                        ProxyError::Embedding(format!("ONNX forward execution failed: {e}"))
                    })?;

                    let first_key = outputs.keys().next().ok_or_else(|| {
                        ProxyError::Embedding(
                            "ONNX session returned zero output tensors".to_string(),
                        )
                    })?;

                    let output_val = outputs.get(first_key).ok_or_else(|| {
                        ProxyError::Embedding(
                            "Failed to retrieve primary ONNX output tensor".to_string(),
                        )
                    })?;

                    let tensor_view = output_val.try_extract_tensor::<f32>().map_err(|e| {
                        ProxyError::Embedding(format!("Failed to extract f32 output tensor: {e}"))
                    })?;

                    let shape: Vec<usize> = tensor_view.shape().to_vec();
                    let data: Vec<f32> = tensor_view.iter().copied().collect();
                    (data, shape)
                };

                // Handle either [1, seq_len, hidden_dim] token embeddings or pre-pooled [1, hidden_dim] output
                if shape_dims.len() == 3 {
                    let hidden_dim = shape_dims[2];
                    let pooled = mean_pool_embeddings(
                        &flat_data,
                        &tokenized.attention_mask,
                        seq_len,
                        hidden_dim,
                    )?;
                    Ok(RawEmbedding {
                        dimension: hidden_dim,
                        raw_vector: pooled,
                    })
                } else if shape_dims.len() == 2 {
                    let hidden_dim = shape_dims[1];
                    if flat_data.len() < hidden_dim {
                        return Err(ProxyError::Embedding(
                            "2D output tensor shorter than declared hidden dimension".to_string(),
                        ));
                    }
                    Ok(RawEmbedding {
                        dimension: hidden_dim,
                        raw_vector: flat_data[..hidden_dim].to_vec(),
                    })
                } else {
                    Err(ProxyError::Embedding(format!(
                        "Unsupported ONNX output tensor rank: {:?}",
                        shape_dims
                    )))
                }
            }
            #[cfg(test)]
            SessionBackend::Synthetic => {
                // Generate deterministic [1, seq_len, 768] token embeddings from token IDs
                let hidden_dim = NATIVE_GEMMA_DIMENSION;
                let mut token_embeddings = Vec::with_capacity(seq_len * hidden_dim);
                for (tok_idx, &tok_id) in tokenized.input_ids.iter().enumerate() {
                    for dim_idx in 0..hidden_dim {
                        let seed = (tok_id as u64)
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add((tok_idx as u64).wrapping_mul(1442695040888963407))
                            .wrapping_add(dim_idx as u64);
                        let val = (((seed % 2000) as f32) / 1000.0) - 1.0 + 0.05;
                        token_embeddings.push(val);
                    }
                }
                let pooled = mean_pool_embeddings(
                    &token_embeddings,
                    &tokenized.attention_mask,
                    seq_len,
                    hidden_dim,
                )?;
                Ok(RawEmbedding {
                    dimension: hidden_dim,
                    raw_vector: pooled,
                })
            }
        }
    }

    /// Acquire a permit from the bounded concurrency semaphore (`BR3.3`).
    fn acquire_permit_bounded(&self) -> Result<tokio::sync::SemaphorePermit<'_>, ProxyError> {
        let mut attempts = 0u32;
        loop {
            match self.semaphore.try_acquire() {
                Ok(permit) => return Ok(permit),
                Err(tokio::sync::TryAcquireError::Closed) => {
                    return Err(ProxyError::Embedding(
                        "Inference concurrency semaphore is closed".to_string(),
                    ));
                }
                Err(tokio::sync::TryAcquireError::NoPermits) => {
                    attempts += 1;
                    if attempts > 5_000 {
                        return Err(ProxyError::Embedding(
                            "Timed out waiting for bounded inference concurrency permit"
                                .to_string(),
                        ));
                    }
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
            }
        }
    }
}

/// Compute attention-mask-weighted mean pooling over a `[seq_len, hidden_dim]` token embedding buffer.
///
/// Pure function with zero panics (`BR3.4`, `NFR6.1`).
pub fn mean_pool_embeddings(
    token_embeddings: &[f32],
    attention_mask: &[i64],
    seq_len: usize,
    hidden_dim: usize,
) -> Result<Vec<f32>, ProxyError> {
    if seq_len == 0 || hidden_dim == 0 {
        return Err(ProxyError::Embedding(
            "Cannot mean-pool empty sequence or zero hidden dimension".to_string(),
        ));
    }

    let expected_len = seq_len.saturating_mul(hidden_dim);
    if token_embeddings.len() < expected_len {
        return Err(ProxyError::Embedding(format!(
            "Token embeddings buffer length ({}) is smaller than seq_len * hidden_dim ({})",
            token_embeddings.len(),
            expected_len
        )));
    }

    if attention_mask.len() < seq_len {
        return Err(ProxyError::Embedding(format!(
            "Attention mask length ({}) is smaller than sequence length ({})",
            attention_mask.len(),
            seq_len
        )));
    }

    let mut pooled = vec![0.0_f32; hidden_dim];
    let mut mask_sum = 0.0_f32;

    for (tok_idx, &mask_val) in attention_mask.iter().take(seq_len).enumerate() {
        if mask_val > 0 {
            let weight = mask_val as f32;
            mask_sum += weight;
            let row_offset = tok_idx * hidden_dim;
            let row = &token_embeddings[row_offset..row_offset + hidden_dim];
            for (acc, &val) in pooled.iter_mut().zip(row.iter()) {
                *acc += val * weight;
            }
        }
    }

    if mask_sum <= 0.0 {
        return Err(ProxyError::Embedding(
            "Attention mask sum is zero; no active tokens to pool".to_string(),
        ));
    }

    let inv_mask_sum = 1.0 / mask_sum;
    for val in pooled.iter_mut() {
        *val *= inv_mask_sum;
    }

    Ok(pooled)
}

impl EmbeddingEngine for GemmaOnnxEmbedder {
    fn encode(&self, text: &str) -> Result<Vec<f32>, ProxyError> {
        let tokenized = self.tokenize_and_validate(text)?;
        let raw = self.execute_forward(&tokenized)?;
        let final_vec = slice_and_normalize(&raw.raw_vector, self.dimension)?;
        let final_embedding = FinalEmbedding {
            dimension: self.dimension,
            normalized_vector: final_vec,
        };
        Ok(final_embedding.normalized_vector)
    }

    fn encode_with_task(&self, text: &str, task: EmbeddingTask) -> Result<Vec<f32>, ProxyError> {
        if text.trim().is_empty() {
            return Err(ProxyError::InvalidRequest(
                "Input prompt cannot be empty".to_string(),
            ));
        }
        let formatted = format_task_prompt(task, text);
        self.encode(&formatted)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    /// Build a lightweight in-memory WordLevel tokenizer from JSON for fast, deterministic unit tests.
    fn build_test_tokenizer() -> Tokenizer {
        let tokenizer_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "[UNK]": 0,
                    "task:": 1,
                    "search": 2,
                    "query": 3,
                    "document": 4,
                    "indexing": 5,
                    "|": 6,
                    "text:": 7,
                    "hello": 8,
                    "world": 9,
                    "rust": 10
                },
                "unk_token": "[UNK]"
            }
        }"#;

        let mut tokenizer =
            Tokenizer::from_str(tokenizer_json).expect("Failed to parse test tokenizer JSON");
        let _ = tokenizer.with_truncation(None);
        tokenizer.with_padding(None);
        tokenizer
    }

    fn build_synthetic_embedder(
        dimension: usize,
        max_tokens: usize,
        permits: usize,
    ) -> GemmaOnnxEmbedder {
        GemmaOnnxEmbedder {
            tokenizer: build_test_tokenizer(),
            backend: SessionBackend::Synthetic,
            semaphore: Arc::new(Semaphore::new(permits)),
            dimension,
            max_tokens,
        }
    }

    #[test]
    fn test_gemma_engine_config_defaults() {
        let cfg = GemmaEngineConfig::new(
            PathBuf::from("/models/gemma/model.onnx"),
            PathBuf::from("/models/gemma/tokenizer.json"),
        );
        assert_eq!(cfg.dimension, 384);
        assert_eq!(cfg.max_tokens, 8192);
        assert_eq!(cfg.concurrency_limit, 2);

        let custom = cfg
            .with_dimension(768)
            .with_max_tokens(4096)
            .with_concurrency_limit(4);
        assert_eq!(custom.dimension, 768);
        assert_eq!(custom.max_tokens, 4096);
        assert_eq!(custom.concurrency_limit, 4);
    }

    #[test]
    fn test_gemma_token_limit_exceeded_error() {
        let embedder = build_synthetic_embedder(384, 8192, 2);

        // Exactly 8,192 tokens must succeed (BR3.1 boundary check)
        let exact_8k = vec!["hello"; 8192].join(" ");
        let ok_res = embedder.encode(&exact_8k);
        assert!(ok_res.is_ok(), "8192 tokens should succeed");

        // 8,193 tokens must fail fast with ProxyError::InvalidRequest (HTTP 400)
        let over_8k = vec!["hello"; 8193].join(" ");
        let err = embedder
            .encode(&over_8k)
            .expect_err("8193 tokens must be rejected");
        assert!(
            matches!(err, ProxyError::InvalidRequest(ref msg) if msg.contains("8192")),
            "Expected InvalidRequest mentioning 8192 limit, got: {err:?}"
        );
    }

    #[test]
    fn test_gemma_semaphore_throttling() {
        let embedder = build_synthetic_embedder(384, 8192, 2);
        assert_eq!(embedder.semaphore().available_permits(), 2);

        // Acquire 1 permit manually; 1 permit remains for encode()
        let _p1 = embedder.semaphore().try_acquire().expect("permit 1");
        assert_eq!(embedder.semaphore().available_permits(), 1);

        let vec = embedder
            .encode("hello world")
            .expect("encode should succeed with 1 remaining permit");
        assert_eq!(vec.len(), 384);
        // Permit acquired inside encode() is immediately released after forward pass
        assert_eq!(embedder.semaphore().available_permits(), 1);
    }

    #[test]
    fn test_gemma_mean_pooling_math() {
        // 3 tokens, hidden_dim = 4; third token is masked out (attention_mask = 0)
        let token_embeddings = vec![
            1.0, 2.0, 3.0, 4.0, // token 0 (mask = 1)
            3.0, 4.0, 5.0, 6.0, // token 1 (mask = 1)
            99.0, 99.0, 99.0, 99.0, // token 2 (mask = 0 -> ignored)
        ];
        let attention_mask = vec![1, 1, 0];

        let pooled = mean_pool_embeddings(&token_embeddings, &attention_mask, 3, 4)
            .expect("mean pooling should succeed");
        assert_eq!(pooled, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_gemma_mrl_slicing_integration() {
        let embedder_384 = build_synthetic_embedder(384, 8192, 2);
        let vec_384 = embedder_384
            .encode("hello rust world")
            .expect("384-d encode should succeed");
        assert_eq!(vec_384.len(), 384);
        let norm_384: f32 = vec_384.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_384 - 1.0).abs() < 1e-4,
            "384-d L2 norm must be 1.0 +- 1e-4, got {norm_384}"
        );

        let embedder_768 = build_synthetic_embedder(768, 8192, 2);
        let vec_768 = embedder_768
            .encode("hello rust world")
            .expect("768-d encode should succeed");
        assert_eq!(vec_768.len(), 768);
        let norm_768: f32 = vec_768.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_768 - 1.0).abs() < 1e-4,
            "768-d L2 norm must be 1.0 +- 1e-4, got {norm_768}"
        );
    }

    #[test]
    fn test_gemma_dimension_accessor() {
        let embedder_384 = build_synthetic_embedder(384, 8192, 2);
        assert_eq!(embedder_384.dimension(), 384);
        assert_eq!(embedder_384.max_tokens(), 8192);

        let embedder_768 = build_synthetic_embedder(768, 8192, 2);
        assert_eq!(embedder_768.dimension(), 768);
    }

    #[test]
    fn test_gemma_encode_with_task_prefix() {
        let embedder = build_synthetic_embedder(384, 8192, 2);
        let query_vec = embedder
            .encode_with_task("hello world", EmbeddingTask::SearchQuery)
            .expect("search query task encoding should succeed");
        let doc_vec = embedder
            .encode_with_task("hello world", EmbeddingTask::DocumentIndexing)
            .expect("document indexing task encoding should succeed");

        assert_eq!(query_vec.len(), 384);
        assert_eq!(doc_vec.len(), 384);
        // Asymmetric prefixes produce distinct token sequences and distinct embeddings
        assert_ne!(query_vec, doc_vec);
    }

    #[test]
    fn test_gemma_zero_panic_guarantee() {
        let embedder = build_synthetic_embedder(384, 8192, 2);

        // Empty or whitespace-only inputs return typed InvalidRequest without panicking
        assert!(matches!(
            embedder.encode(""),
            Err(ProxyError::InvalidRequest(_))
        ));
        assert!(matches!(
            embedder.encode("   "),
            Err(ProxyError::InvalidRequest(_))
        ));
        assert!(matches!(
            embedder.encode_with_task("", EmbeddingTask::SearchQuery),
            Err(ProxyError::InvalidRequest(_))
        ));

        // Zero attention mask in mean pooling returns typed Embedding error
        let mask_zero_err = mean_pool_embeddings(&[1.0, 2.0, 3.0, 4.0], &[0], 1, 4);
        assert!(matches!(mask_zero_err, Err(ProxyError::Embedding(_))));

        // Invalid dimension in try_new returns InvalidRequest
        let bad_cfg = GemmaEngineConfig::new(
            PathBuf::from("/nonexistent/model.onnx"),
            PathBuf::from("/nonexistent/tokenizer.json"),
        )
        .with_dimension(512);
        assert!(matches!(
            GemmaOnnxEmbedder::try_new(bad_cfg),
            Err(ProxyError::InvalidRequest(_))
        ));
    }
}
