//! Server-Sent Events (SSE) streaming helpers and chunk generator for OpenAI format.

use axum::response::sse::Event;
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Generate simulated SSE stream events from a cached static text response.
pub fn generate_cached_sse_stream(
    model: String,
    content: String,
) -> impl futures_util::Stream<Item = Result<Event, axum::Error>> {
    let completion_id = format!("chatcmpl-kache-{}", Uuid::new_v4().simple());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    async_stream::stream! {
        // 1. Initial role chunk
        let initial_chunk = json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant", "content": "" },
                "finish_reason": null
            }]
        });
        yield Ok(Event::default().data(initial_chunk.to_string()));

        // 2. Content delta chunk
        let content_chunk = json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "content": content },
                "finish_reason": null
            }]
        });
        yield Ok(Event::default().data(content_chunk.to_string()));

        // 3. Final stop chunk
        let final_chunk = json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        });
        yield Ok(Event::default().data(final_chunk.to_string()));

        // 4. [DONE] signal
        yield Ok(Event::default().data("[DONE]"));
    }
}

/// Extract content delta from an upstream OpenAI SSE line.
pub fn extract_sse_delta(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if !trimmed.starts_with("data:") {
        return None;
    }
    let data = trimmed[5..].trim();
    if data == "[DONE]" {
        return None;
    }

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(data)
        && let Some(content) = v
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|first| first.get("delta"))
            .and_then(|delta| delta.get("content"))
            .and_then(|c| c.as_str())
    {
        return Some(content.to_string());
    }
    None
}

/// Generate simulated Anthropic SSE stream events from a cached static text response.
pub fn generate_cached_anthropic_sse_stream(
    model: String,
    content: String,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> impl futures_util::Stream<Item = Result<Event, axum::Error>> {
    let msg_id = format!("msg_kache_{}", Uuid::new_v4().simple());

    async_stream::stream! {
        // 1. message_start
        let msg_start = json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": 1
                }
            }
        });
        yield Ok(Event::default().event("message_start").data(msg_start.to_string()));

        // 2. content_block_start
        let block_start = json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": ""
            }
        });
        yield Ok(Event::default().event("content_block_start").data(block_start.to_string()));

        // 3. content_block_delta
        let block_delta = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": content
            }
        });
        yield Ok(Event::default().event("content_block_delta").data(block_delta.to_string()));

        // 4. content_block_stop
        let block_stop = json!({
            "type": "content_block_stop",
            "index": 0
        });
        yield Ok(Event::default().event("content_block_stop").data(block_stop.to_string()));

        // 5. message_delta
        let msg_delta = json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn",
                "stop_sequence": null
            },
            "usage": {
                "output_tokens": completion_tokens
            }
        });
        yield Ok(Event::default().event("message_delta").data(msg_delta.to_string()));

        // 6. message_stop
        let msg_stop = json!({
            "type": "message_stop"
        });
        yield Ok(Event::default().event("message_stop").data(msg_stop.to_string()));
    }
}

/// Extract content delta from an upstream Anthropic SSE line.
pub fn extract_anthropic_sse_delta(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if !trimmed.starts_with("data:") {
        return None;
    }
    let data = trimmed[5..].trim();
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(data)
        && v.get("type").and_then(|t| t.as_str()) == Some("content_block_delta")
        && let Some(text) = v
            .get("delta")
            .and_then(|d| d.get("text"))
            .and_then(|t| t.as_str())
    {
        return Some(text.to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    #[tokio::test]
    async fn test_generate_cached_sse_stream() {
        let stream = generate_cached_sse_stream("gpt-4o".to_string(), "Hello world".to_string());
        tokio::pin!(stream);

        let mut chunks = Vec::new();
        while let Some(Ok(event)) = stream.next().await {
            chunks.push(event);
        }
        assert_eq!(chunks.len(), 4);
    }

    #[tokio::test]
    async fn test_generate_cached_anthropic_sse_stream() {
        let stream = generate_cached_anthropic_sse_stream(
            "claude-3-5-sonnet".to_string(),
            "Anthropic answer".to_string(),
            10,
            25,
        );
        tokio::pin!(stream);

        let mut chunks = Vec::new();
        while let Some(Ok(event)) = stream.next().await {
            chunks.push(event);
        }
        assert_eq!(chunks.len(), 6);
    }

    #[test]
    fn test_extract_sse_delta_openai() {
        let valid = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}";
        assert_eq!(extract_sse_delta(valid), Some("Hello".to_string()));

        let done = "data: [DONE]";
        assert_eq!(extract_sse_delta(done), None);

        let empty_delta = "data: {\"choices\":[{\"delta\":{}}]}";
        assert_eq!(extract_sse_delta(empty_delta), None);

        let empty_choices = "data: {\"choices\":[]}";
        assert_eq!(extract_sse_delta(empty_choices), None);

        let not_data = "event: ping";
        assert_eq!(extract_sse_delta(not_data), None);

        let malformed = "data: {bad json}";
        assert_eq!(extract_sse_delta(malformed), None);
    }

    #[test]
    fn test_extract_anthropic_sse_delta() {
        let valid = "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Bonjour\"}}";
        assert_eq!(
            extract_anthropic_sse_delta(valid),
            Some("Bonjour".to_string())
        );

        let other_type = "data: {\"type\":\"message_stop\"}";
        assert_eq!(extract_anthropic_sse_delta(other_type), None);

        let delta_no_text = "data: {\"type\":\"content_block_delta\",\"delta\":{}}";
        assert_eq!(extract_anthropic_sse_delta(delta_no_text), None);

        let not_data = "event: message_delta";
        assert_eq!(extract_anthropic_sse_delta(not_data), None);

        let malformed = "data: bad_json";
        assert_eq!(extract_anthropic_sse_delta(malformed), None);
    }
}
