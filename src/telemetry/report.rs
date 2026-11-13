//! Terminal ASCII banner and reporting for token economics.

use super::tracker::TelemetrySnapshot;

/// Generate styled ASCII terminal banner summarizing token economics and cache efficiency.
pub fn generate_ascii_report(snapshot: &TelemetrySnapshot) -> String {
    let exact_pct = if snapshot.total_requests > 0 {
        (snapshot.exact_hits as f64 / snapshot.total_requests as f64) * 100.0
    } else {
        0.0
    };

    let semantic_pct = if snapshot.total_requests > 0 {
        (snapshot.semantic_hits as f64 / snapshot.total_requests as f64) * 100.0
    } else {
        0.0
    };

    format!(
        r#"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ KacheDB Proxy Token Savings & Efficiency Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Requests Handled:      {:>10}
Exact Cache Hits (Tier 1):   {:>10} ({:.1}%)
Semantic Cache Hits (Tier 2):{:>10} ({:.1}%)
Total Cache Hit Ratio:       {:>9.1}%
Prompt Tokens Saved:         {:>10}
Completion Tokens Saved:     {:>10}
Total Tokens Saved:          {:>10}
Avg Upstream Latency:        {:>9.1} ms
Avg Cache Hit Latency:       {:>9.2} ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"#,
        snapshot.total_requests,
        snapshot.exact_hits,
        exact_pct,
        snapshot.semantic_hits,
        semantic_pct,
        snapshot.hit_ratio_percent,
        snapshot.prompt_tokens_saved,
        snapshot.completion_tokens_saved,
        snapshot.total_tokens_saved,
        snapshot.avg_upstream_latency_ms,
        snapshot.avg_cache_latency_ms
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_report_generation() {
        let snapshot = TelemetrySnapshot {
            total_requests: 100,
            cache_hits: 80,
            exact_hits: 50,
            semantic_hits: 30,
            cache_misses: 20,
            hit_ratio_percent: 80.0,
            prompt_tokens_saved: 120_000,
            completion_tokens_saved: 45_000,
            tokens_saved: 165_000,
            total_tokens_saved: 165_000,
            avg_cache_latency_ms: 1.85,
            avg_upstream_latency_ms: 340.5,
        };

        let report = generate_ascii_report(&snapshot);
        assert!(report.contains("⚡ KacheDB Proxy Token Savings & Efficiency Report"));
        assert!(report.contains("Total Requests Handled:             100"));
        assert!(report.contains("Exact Cache Hits (Tier 1):           50 (50.0%)"));
        assert!(report.contains("Semantic Cache Hits (Tier 2):        30 (30.0%)"));
        assert!(report.contains("Prompt Tokens Saved:             120000"));
        assert!(report.contains("Total Tokens Saved:              165000"));
    }

    #[test]
    fn test_ascii_report_zero_requests() {
        let snapshot = TelemetrySnapshot::default();
        let report = generate_ascii_report(&snapshot);
        assert!(report.contains("Total Requests Handled:               0"));
        assert!(report.contains("(0.0%)"));
    }
}
