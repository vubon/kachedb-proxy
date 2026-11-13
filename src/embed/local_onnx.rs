//! Local ONNX embedding engine using fastembed-rs.

use super::EmbeddingEngine;
use crate::error::ProxyError;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::Mutex;

pub struct LocalOnnxEmbedder {
    model: Mutex<TextEmbedding>,
    dimension: usize,
}

impl LocalOnnxEmbedder {
    pub fn resolve_model(model_name: &str) -> Result<(EmbeddingModel, usize), ProxyError> {
        match model_name.to_lowercase().as_str() {
            "bge-small-en-v1.5" | "bge-small-en" | "bge-small" => {
                Ok((EmbeddingModel::BGESmallENV15, 384))
            }
            "bge-base-en-v1.5" | "bge-base-en" | "bge-base" => {
                Ok((EmbeddingModel::BGEBaseENV15, 768))
            }
            "all-minilm-l6-v2" | "minilm" => Ok((EmbeddingModel::AllMiniLML6V2, 384)),
            "nomic-embed-text-v1.5" | "nomic" => Ok((EmbeddingModel::NomicEmbedTextV15, 768)),
            _ => Err(ProxyError::Embedding(format!(
                "Unsupported embedding model '{model_name}'. Supported: bge-small, bge-base, minilm, nomic"
            ))),
        }
    }

    pub fn try_new(model_name: &str) -> Result<Self, ProxyError> {
        let (embedding_model, dimension) = Self::resolve_model(model_name)?;
        let options = InitOptions::new(embedding_model);
        Self::try_new_with_options(options, dimension)
    }

    pub fn try_new_with_options(
        options: InitOptions,
        dimension: usize,
    ) -> Result<Self, ProxyError> {
        let model = TextEmbedding::try_new(options)
            .map_err(|e| ProxyError::Embedding(format!("Failed to initialize fastembed: {e}")))?;

        Ok(Self {
            model: Mutex::new(model),
            dimension,
        })
    }
}

impl EmbeddingEngine for LocalOnnxEmbedder {
    fn encode(&self, text: &str) -> Result<Vec<f32>, ProxyError> {
        let model = self
            .model
            .lock()
            .map_err(|_| ProxyError::Embedding("Lock poisoned".to_string()))?;

        let embeddings = model
            .embed(vec![text.to_string()], None)
            .map_err(|e| ProxyError::Embedding(format!("Inference error: {e}")))?;

        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| ProxyError::Embedding("No embedding returned".to_string()))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_resolve_model_all_variants() {
        let (m1, d1) = LocalOnnxEmbedder::resolve_model("bge-small").unwrap();
        assert_eq!(d1, 384);
        assert!(matches!(m1, EmbeddingModel::BGESmallENV15));

        let (m2, d2) = LocalOnnxEmbedder::resolve_model("bge-base").unwrap();
        assert_eq!(d2, 768);
        assert!(matches!(m2, EmbeddingModel::BGEBaseENV15));

        let (m3, d3) = LocalOnnxEmbedder::resolve_model("all-minilm-l6-v2").unwrap();
        assert_eq!(d3, 384);
        assert!(matches!(m3, EmbeddingModel::AllMiniLML6V2));

        let (m4, d4) = LocalOnnxEmbedder::resolve_model("nomic").unwrap();
        assert_eq!(d4, 768);
        assert!(matches!(m4, EmbeddingModel::NomicEmbedTextV15));

        let err = LocalOnnxEmbedder::resolve_model("unknown-custom-model");
        assert!(err.is_err());
    }

    #[test]
    fn test_try_new_with_invalid_cache_dir() {
        let (model, dim) = LocalOnnxEmbedder::resolve_model("bge-small").unwrap();
        let options =
            InitOptions::new(model).with_cache_dir(PathBuf::from("/dev/null/forbidden/invalid"));
        let res = LocalOnnxEmbedder::try_new_with_options(options, dim);
        assert!(res.is_err());
    }
}
