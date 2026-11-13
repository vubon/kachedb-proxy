//! Error types for kachedb-proxy.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("KacheDB connection error: {0}")]
    KacheDb(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Upstream HTTP error: {0}")]
    Upstream(#[from] reqwest::Error),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Internal server error: {0}")]
    Internal(String),
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let (status, message, err_type) = match self {
            ProxyError::InvalidRequest(msg) => {
                (StatusCode::BAD_REQUEST, msg, "invalid_request_error")
            }
            ProxyError::Upstream(e) => (
                StatusCode::BAD_GATEWAY,
                format!("Upstream provider error: {e}"),
                "upstream_error",
            ),
            ProxyError::KacheDb(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("KacheDB backend error: {msg}"),
                "kachedb_error",
            ),
            ProxyError::Embedding(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Embedding model error: {msg}"),
                "embedding_error",
            ),
            ProxyError::Serialization(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("JSON serialization error: {e}"),
                "serialization_error",
            ),
            ProxyError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg, "internal_error"),
        };

        let body = Json(json!({
            "error": {
                "message": message,
                "type": err_type,
                "param": null,
                "code": status.as_u16()
            }
        }));

        (status, body).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_into_response_mappings() {
        let err_bad_req = ProxyError::InvalidRequest("Missing prompt".to_string()).into_response();
        assert_eq!(err_bad_req.status(), StatusCode::BAD_REQUEST);

        let err_kachedb = ProxyError::KacheDb("Connection refused".to_string()).into_response();
        assert_eq!(err_kachedb.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let err_embed = ProxyError::Embedding("ONNX failure".to_string()).into_response();
        assert_eq!(err_embed.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let json_err = serde_json::from_str::<serde_json::Value>("{invalid}").unwrap_err();
        let err_ser = ProxyError::Serialization(json_err).into_response();
        assert_eq!(err_ser.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let err_internal = ProxyError::Internal("Unexpected crash".to_string()).into_response();
        assert_eq!(err_internal.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_upstream_error_into_response() {
        let err = reqwest::get("http://127.0.0.1:1").await.unwrap_err();
        let resp = ProxyError::Upstream(err).into_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }
}
