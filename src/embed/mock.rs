//! Deterministic Mock Embedder for testing and benchmarks without downloading ONNX weights.

use super::EmbeddingEngine;
use crate::error::ProxyError;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct MockEmbedder {
    dimension: usize,
}

impl MockEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Default for MockEmbedder {
    fn default() -> Self {
        Self::new(384)
    }
}

impl EmbeddingEngine for MockEmbedder {
    fn encode(&self, text: &str) -> Result<Vec<f32>, ProxyError> {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        let mut vec = Vec::with_capacity(self.dimension);
        for i in 0..self.dimension {
            // Generate deterministic pseudo-random float based on text hash and index
            let mul = (i as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
            let val = (((hash ^ mul) % 1000) as f32 / 1000.0) - 0.5;
            vec.push(val);
        }

        // Normalize vector to unit length L2 norm = 1.0
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }

        Ok(vec)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::EmbeddingTask;

    #[test]
    fn test_mock_embedder_dimension_and_norm() {
        let embedder = MockEmbedder::default();
        assert_eq!(embedder.dimension(), 384);

        let vec = embedder.encode("Hello, KacheDB proxy!").unwrap();
        assert_eq!(vec.len(), 384);

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mock_embedder_encode_with_task() {
        let embedder = MockEmbedder::default();
        let prompt = "sample query";

        let query_vec = embedder
            .encode_with_task(prompt, EmbeddingTask::SearchQuery)
            .unwrap();
        assert_eq!(query_vec.len(), 384);
        let norm_q: f32 = query_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_q - 1.0).abs() < 1e-5);

        let doc_vec = embedder
            .encode_with_task(prompt, EmbeddingTask::DocumentIndexing)
            .unwrap();
        assert_eq!(doc_vec.len(), 384);
        let norm_d: f32 = doc_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_d - 1.0).abs() < 1e-5);

        // Different tasks prepend different prefixes, so the hashes and vectors must differ
        assert_ne!(query_vec, doc_vec);
    }

    #[test]
    fn test_mock_embedder_768_dimension() {
        let embedder = MockEmbedder::new(768);
        assert_eq!(embedder.dimension(), 768);

        let vec = embedder
            .encode_with_task("test prompt", EmbeddingTask::SearchQuery)
            .unwrap();
        assert_eq!(vec.len(), 768);
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
