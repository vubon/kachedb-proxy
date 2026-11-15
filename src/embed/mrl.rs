//! Matryoshka Representation Learning (MRL) dimension slicing and L2 normalization.

use crate::error::ProxyError;

/// Numerical stability threshold below which vector norms are rejected.
pub const EPSILON: f32 = 1e-12;

/// Slices a high-dimensional embedding vector to the specified target dimension and applies L2 normalization.
///
/// Supported target dimensions are strictly 384 and 768 per architectural specifications.
///
/// # Errors
/// Returns [`ProxyError::InvalidRequest`] if `target_dim` is neither 384 nor 768.
/// Returns [`ProxyError::Embedding`] if `vector.len() < target_dim` or if the sliced vector
/// has a norm below [`EPSILON`].
pub fn slice_and_normalize(vector: &[f32], target_dim: usize) -> Result<Vec<f32>, ProxyError> {
    if target_dim != 384 && target_dim != 768 {
        return Err(ProxyError::InvalidRequest(format!(
            "Unsupported MRL target dimension {target_dim}; must be 384 or 768"
        )));
    }

    if vector.len() < target_dim {
        return Err(ProxyError::Embedding(format!(
            "Source vector length ({}) is smaller than target dimension ({})",
            vector.len(),
            target_dim
        )));
    }

    let mut sliced = Vec::with_capacity(target_dim);
    sliced.extend_from_slice(&vector[..target_dim]);

    l2_normalize_in_place(&mut sliced)?;

    Ok(sliced)
}

/// Normalizes a float vector in-place such that its Euclidean norm equals 1.0.
///
/// # Errors
/// Returns [`ProxyError::Embedding`] if the vector is empty or if its Euclidean norm is below [`EPSILON`].
pub fn l2_normalize_in_place(vector: &mut [f32]) -> Result<(), ProxyError> {
    if vector.is_empty() {
        return Err(ProxyError::Embedding(
            "Cannot normalize an empty vector".to_string(),
        ));
    }

    let sum_sq: f32 = vector.iter().map(|&x| x * x).sum();
    let norm = sum_sq.sqrt();

    if norm < EPSILON || norm.is_nan() {
        return Err(ProxyError::Embedding(
            "Vector norm is zero or below epsilon threshold; cannot normalize".to_string(),
        ));
    }

    let inv_norm = 1.0 / norm;
    for x in vector.iter_mut() {
        *x *= inv_norm;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compute_norm(v: &[f32]) -> f32 {
        v.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn test_slice_768_to_384_and_normalize() {
        // Construct a 768-d synthetic vector
        let full_vec: Vec<f32> = (0..768).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let sliced = slice_and_normalize(&full_vec, 384).expect("slicing to 384 must succeed");

        assert_eq!(sliced.len(), 384);
        let norm = compute_norm(&sliced);
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Norm must be 1.0 +- 1e-4, got {}",
            norm
        );

        // Verify elements maintain proportional ratios of original prefix
        assert!(sliced[0] > 0.0);
        assert!(sliced[383] > sliced[0]);
    }

    #[test]
    fn test_identity_slice_768_to_768() {
        let full_vec: Vec<f32> = (0..768).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let normalized = slice_and_normalize(&full_vec, 768).expect("slicing to 768 must succeed");

        assert_eq!(normalized.len(), 768);
        let norm = compute_norm(&normalized);
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_slice_invalid_target_dim() {
        let full_vec: Vec<f32> = vec![1.0; 768];
        for invalid_dim in [128, 256, 512, 1024] {
            let res = slice_and_normalize(&full_vec, invalid_dim);
            assert!(
                matches!(res, Err(ProxyError::InvalidRequest(_))),
                "Target dim {invalid_dim} must return InvalidRequest"
            );
        }
    }

    #[test]
    fn test_slice_source_shorter_than_target() {
        let short_vec: Vec<f32> = vec![1.0; 300];
        let res = slice_and_normalize(&short_vec, 384);
        assert!(
            matches!(res, Err(ProxyError::Embedding(_))),
            "Short source vector must return Embedding error"
        );
    }

    #[test]
    fn test_zero_vector_rejection() {
        let zero_vec = vec![0.0; 768];
        let res = slice_and_normalize(&zero_vec, 384);
        assert!(
            matches!(res, Err(ProxyError::Embedding(_))),
            "Zero vector must return Embedding error"
        );
    }

    #[test]
    fn test_sub_epsilon_vector_rejection() {
        let sub_eps_vec = vec![1e-15; 768];
        let res = slice_and_normalize(&sub_eps_vec, 384);
        assert!(
            matches!(res, Err(ProxyError::Embedding(_))),
            "Sub-epsilon vector must return Embedding error"
        );
    }
}
