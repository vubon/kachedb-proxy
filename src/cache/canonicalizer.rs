//! Prompt canonicalization and message extraction for semantic caching.

use serde_json::Value;

/// Extract a canonical prompt string from OpenAI-format messages array.
pub fn canonicalize_messages(messages: &[Value]) -> String {
    if messages.is_empty() {
        return String::new();
    }

    let mut parts = Vec::new();
    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = if let Some(c) = msg.get("content") {
            if let Some(s) = c.as_str() {
                s.to_string()
            } else if let Some(arr) = c.as_array() {
                // Multi-modal array with text parts
                let mut text_parts = Vec::new();
                for item in arr {
                    if item.get("type").and_then(|t| t.as_str()) == Some("text")
                        && let Some(t) = item.get("text").and_then(|t| t.as_str())
                    {
                        text_parts.push(t);
                    }
                }
                text_parts.join(" ")
            } else {
                c.to_string()
            }
        } else {
            String::new()
        };

        if !content.trim().is_empty() {
            parts.push(format!("{role}: {}", content.trim()));
        }
    }

    // If only one message and it's user, return plain text for maximum similarity match
    if parts.len() == 1 && parts[0].starts_with("user: ") {
        return parts[0][6..].trim().to_string();
    }

    parts.join("\n")
}

/// Extract a canonical prompt string from an Anthropic Messages request.
pub fn canonicalize_anthropic_prompt(
    system: Option<&str>,
    messages: &[(String, String)],
) -> String {
    let mut parts = Vec::new();

    if let Some(sys) = system {
        let trimmed = sys.trim();
        if !trimmed.is_empty() {
            parts.push(format!("system: {trimmed}"));
        }
    }

    for (role, content) in messages {
        let trimmed = content.trim();
        if !trimmed.is_empty() {
            parts.push(format!("{role}: {trimmed}"));
        }
    }

    // If only one user message and no system prompt, return plain text
    if parts.len() == 1 && parts[0].starts_with("user: ") {
        return parts[0][6..].trim().to_string();
    }

    parts.join("\n")
}

/// Sanitize a model name into a safe identifier for cache keys and vector index namespaces.
pub fn sanitize_model_name(raw_model: &str) -> String {
    let clean = raw_model.trim();
    if clean.is_empty() {
        return "default".to_string();
    }

    // If it's a file path like /Users/vubon/.../qwen2.5-0.5b-instruct-q4_k_m.gguf, extract filename
    let name = if let Some(pos) = clean.rfind('/') {
        &clean[pos + 1..]
    } else {
        clean
    };

    let stripped = name.strip_suffix(".gguf").unwrap_or(name);

    let mut out = String::with_capacity(stripped.len());
    for c in stripped.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
            out.push(c.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }

    if out.is_empty() {
        "default".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_single_user_message() {
        let messages = vec![json!({"role": "user", "content": "How do I reset a password?"})];
        assert_eq!(
            canonicalize_messages(&messages),
            "How do I reset a password?"
        );
    }

    #[test]
    fn test_multi_turn_messages() {
        let messages = vec![
            json!({"role": "system", "content": "You are an assistant."}),
            json!({"role": "user", "content": "Hello"}),
            json!({"role": "assistant", "content": "Hi there"}),
            json!({"role": "user", "content": "Help me code"}),
        ];
        let canonical = canonicalize_messages(&messages);
        assert!(canonical.contains("system: You are an assistant."));
        assert!(canonical.contains("user: Help me code"));
    }

    #[test]
    fn test_canonicalize_anthropic_prompt() {
        let single_msg = vec![("user".to_string(), "How do I reverse a list?".to_string())];
        assert_eq!(
            canonicalize_anthropic_prompt(None, &single_msg),
            "How do I reverse a list?"
        );

        let multi = vec![
            ("user".to_string(), "Hello".to_string()),
            ("assistant".to_string(), "Hi!".to_string()),
            ("user".to_string(), "Help me".to_string()),
        ];
        let prompt = canonicalize_anthropic_prompt(Some("Be concise"), &multi);
        assert!(prompt.starts_with("system: Be concise"));
        assert!(prompt.contains("user: Hello"));
        assert!(prompt.contains("assistant: Hi!"));
        assert!(prompt.ends_with("user: Help me"));
    }

    #[test]
    fn test_sanitize_model_name() {
        assert_eq!(sanitize_model_name("gpt-4o"), "gpt-4o");
        assert_eq!(
            sanitize_model_name("claude-3-5-sonnet@20241022"),
            "claude-3-5-sonnet_20241022"
        );
        assert_eq!(
            sanitize_model_name("/path/to/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
            "qwen2.5-0.5b-instruct-q4_k_m"
        );
        assert_eq!(sanitize_model_name(""), "default");
        assert_eq!(sanitize_model_name("   "), "default");
    }

    #[test]
    fn test_canonicalize_empty_and_multimodal() {
        assert_eq!(canonicalize_messages(&[]), "");

        let multi_modal = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image:"},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                {"type": "text", "text": "in detail."}
            ]
        })];
        assert_eq!(
            canonicalize_messages(&multi_modal),
            "Describe this image: in detail."
        );

        let non_string_content = vec![
            json!({"role": "user", "content": 12345}),
            json!({"role": "assistant"}),
        ];
        let out = canonicalize_messages(&non_string_content);
        assert_eq!(out, "12345");
    }
}
