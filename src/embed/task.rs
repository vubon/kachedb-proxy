//! Task prompt formatting for asymmetric embedding models (e.g. EmbeddingGemma 2).

use std::borrow::Cow;

/// Canonical task prompt prefixes for EmbeddingGemma 2.
pub const PREFIX_SEARCH_QUERY: &str = "task: search query | text: ";
pub const PREFIX_DOCUMENT_INDEXING: &str = "task: document indexing | text: ";

/// Embedding task category defining the downstream retrieval context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingTask {
    /// Query embedding context for vector search operations.
    SearchQuery,
    /// Document chunk embedding context for corpus indexing and cache storage.
    DocumentIndexing,
}

impl EmbeddingTask {
    /// Return the canonical prefix string slice for this task.
    #[inline]
    pub fn prefix(&self) -> &'static str {
        match self {
            EmbeddingTask::SearchQuery => PREFIX_SEARCH_QUERY,
            EmbeddingTask::DocumentIndexing => PREFIX_DOCUMENT_INDEXING,
        }
    }
}

/// Prepend the required task prompt prefix to the input text in an idempotent manner.
///
/// If `text` already begins with the required task prefix, returns `Cow::Borrowed(text)`
/// without performing any heap allocation. Otherwise, returns `Cow::Owned(formatted_string)`.
pub fn format_task_prompt<'a>(task: EmbeddingTask, text: &'a str) -> Cow<'a, str> {
    let prefix = task.prefix();
    if text.starts_with(prefix) {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(format!("{prefix}{text}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_search_query_prefix() {
        let input = "what is the capital of france?";
        let formatted = format_task_prompt(EmbeddingTask::SearchQuery, input);
        assert_eq!(
            formatted,
            "task: search query | text: what is the capital of france?"
        );
        assert!(matches!(formatted, Cow::Owned(_)));
    }

    #[test]
    fn test_format_document_indexing_prefix() {
        let input = "Paris is the capital of France.";
        let formatted = format_task_prompt(EmbeddingTask::DocumentIndexing, input);
        assert_eq!(
            formatted,
            "task: document indexing | text: Paris is the capital of France."
        );
        assert!(matches!(formatted, Cow::Owned(_)));
    }

    #[test]
    fn test_format_prompt_idempotent_borrowed() {
        let pre_prefixed = "task: search query | text: already formatted query";
        let formatted = format_task_prompt(EmbeddingTask::SearchQuery, pre_prefixed);
        assert_eq!(formatted, pre_prefixed);
        assert!(matches!(formatted, Cow::Borrowed(_)));
        assert_eq!(formatted.as_ptr(), pre_prefixed.as_ptr());

        let doc_pre_prefixed = "task: document indexing | text: already indexed doc";
        let doc_formatted = format_task_prompt(EmbeddingTask::DocumentIndexing, doc_pre_prefixed);
        assert_eq!(doc_formatted, doc_pre_prefixed);
        assert!(matches!(doc_formatted, Cow::Borrowed(_)));
        assert_eq!(doc_formatted.as_ptr(), doc_pre_prefixed.as_ptr());
    }

    #[test]
    fn test_format_prompt_empty_string() {
        let formatted_query = format_task_prompt(EmbeddingTask::SearchQuery, "");
        assert_eq!(formatted_query, "task: search query | text: ");
        assert!(matches!(formatted_query, Cow::Owned(_)));

        let formatted_doc = format_task_prompt(EmbeddingTask::DocumentIndexing, "");
        assert_eq!(formatted_doc, "task: document indexing | text: ");
        assert!(matches!(formatted_doc, Cow::Owned(_)));
    }

    #[test]
    fn test_format_prompt_whitespace() {
        let input = "  spaced query text  ";
        let formatted = format_task_prompt(EmbeddingTask::SearchQuery, input);
        assert_eq!(
            formatted,
            "task: search query | text:   spaced query text  "
        );
    }
}
