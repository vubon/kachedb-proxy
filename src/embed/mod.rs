//! Embedding engine abstractions and implementations.

pub mod gemma;
pub mod local_onnx;
pub mod mock;
pub mod mrl;
pub mod resolver;
pub mod task;

pub use gemma::{GemmaEngineConfig, GemmaOnnxEmbedder};
pub use mrl::{l2_normalize_in_place, slice_and_normalize};
pub use resolver::{ModelResolutionConfig, ModelSourceMode, ResolvedModel, resolve_model_paths};
pub use task::{EmbeddingTask, format_task_prompt};

use crate::error::ProxyError;
use std::sync::Arc;

pub trait EmbeddingEngine: Send + Sync {
    /// Generate a normalized float vector embedding for the input text.
    fn encode(&self, text: &str) -> Result<Vec<f32>, ProxyError>;

    /// Generate a normalized float vector embedding with task prefix formatting.
    ///
    /// By default, this prepends the task prompt prefix via [`format_task_prompt`]
    /// and delegates to [`EmbeddingEngine::encode`].
    fn encode_with_task(&self, text: &str, task: EmbeddingTask) -> Result<Vec<f32>, ProxyError> {
        let formatted = format_task_prompt(task, text);
        self.encode(&formatted)
    }

    /// Return the embedding dimension (e.g. 384 for bge-small or sliced embeddinggemma-2, 768 for full embeddinggemma-2).
    fn dimension(&self) -> usize;
}

pub type DynEmbeddingEngine = Arc<dyn EmbeddingEngine>;
