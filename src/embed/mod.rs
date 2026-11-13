//! Embedding engine abstractions and implementations.

pub mod local_onnx;
pub mod mock;

use crate::error::ProxyError;
use std::sync::Arc;

pub trait EmbeddingEngine: Send + Sync {
    /// Generate a normalized float vector embedding for the input text.
    fn encode(&self, text: &str) -> Result<Vec<f32>, ProxyError>;

    /// Return the embedding dimension (e.g. 384 for bge-small).
    fn dimension(&self) -> usize;
}

pub type DynEmbeddingEngine = Arc<dyn EmbeddingEngine>;
