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

    #[test]
    fn test_mock_embedder_dimension_and_norm() {
        let embedder = MockEmbedder::default();
        assert_eq!(embedder.dimension(), 384);

        let vec = embedder.encode("Hello, KacheDB proxy!").unwrap();
        assert_eq!(vec.len(), 384);

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
