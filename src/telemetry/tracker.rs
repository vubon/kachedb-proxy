//! Real-time telemetry tracking for token economics, hit ratio, and latency.

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Default)]
pub struct TelemetryTracker {
    total_requests: AtomicU64,
    cache_hits: AtomicU64,
    exact_hits: AtomicU64,
    semantic_hits: AtomicU64,
    cache_misses: AtomicU64,
    prompt_tokens_saved: AtomicU64,
    completion_tokens_saved: AtomicU64,
    total_cache_latency_us: AtomicU64,
    total_upstream_latency_us: AtomicU64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct TelemetrySnapshot {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub exact_hits: u64,
    pub semantic_hits: u64,
    pub cache_misses: u64,
    pub hit_ratio_percent: f64,
    pub prompt_tokens_saved: u64,
    pub completion_tokens_saved: u64,
    pub tokens_saved: u64,
    pub total_tokens_saved: u64,
    pub avg_cache_latency_ms: f64,
    pub avg_upstream_latency_ms: f64,
}

impl TelemetryTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record Tier-1 SwissTable exact cache hit (< 100 µs).
    pub fn record_exact_hit(
        &self,
        _model: &str,
        prompt_tokens: u64,
        completion_tokens: u64,
        latency_us: f64,
    ) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
        self.exact_hits.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens_saved
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.completion_tokens_saved
            .fetch_add(completion_tokens, Ordering::Relaxed);
        self.total_cache_latency_us
            .fetch_add(latency_us as u64, Ordering::Relaxed);
    }

    /// Record Tier-2 SIMD semantic vector cache hit (< 5 ms).
    pub fn record_semantic_hit(
        &self,
        _model: &str,
        prompt_tokens: u64,
        completion_tokens: u64,
        latency_us: f64,
    ) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
        self.semantic_hits.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens_saved
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.completion_tokens_saved
            .fetch_add(completion_tokens, Ordering::Relaxed);
        self.total_cache_latency_us
            .fetch_add(latency_us as u64, Ordering::Relaxed);
    }

    /// Legacy record_hit forwarding with character estimation.
    pub fn record_hit(&self, chars_saved: usize, latency_us: f64) {
        let tokens = (chars_saved / 4).max(1) as u64;
        self.record_semantic_hit("gpt-4o", 1, tokens, latency_us);
    }

    /// Record cache miss forwarded to upstream LLM server.
    pub fn record_miss(&self, latency_us: f64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.total_upstream_latency_us
            .fetch_add(latency_us as u64, Ordering::Relaxed);
    }

    /// Retrieve an instantaneous point-in-time snapshot of telemetry metrics.
    pub fn snapshot(&self) -> TelemetrySnapshot {
        let reqs = self.total_requests.load(Ordering::Relaxed);
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let exact = self.exact_hits.load(Ordering::Relaxed);
        let semantic = self.semantic_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let prompt_tokens = self.prompt_tokens_saved.load(Ordering::Relaxed);
        let completion_tokens = self.completion_tokens_saved.load(Ordering::Relaxed);
        let total_tokens = prompt_tokens + completion_tokens;
        let cache_lat = self.total_cache_latency_us.load(Ordering::Relaxed);
        let up_lat = self.total_upstream_latency_us.load(Ordering::Relaxed);

        let hit_ratio = if reqs > 0 {
            (hits as f64 / reqs as f64) * 100.0
        } else {
            0.0
        };

        let avg_cache_ms = if hits > 0 {
            (cache_lat as f64 / hits as f64) / 1000.0
        } else {
            0.0
        };

        let avg_up_ms = if misses > 0 {
            (up_lat as f64 / misses as f64) / 1000.0
        } else {
            0.0
        };

        TelemetrySnapshot {
            total_requests: reqs,
            cache_hits: hits,
            exact_hits: exact,
            semantic_hits: semantic,
            cache_misses: misses,
            hit_ratio_percent: (hit_ratio * 100.0).round() / 100.0,
            prompt_tokens_saved: prompt_tokens,
            completion_tokens_saved: completion_tokens,
            tokens_saved: total_tokens,
            total_tokens_saved: total_tokens,
            avg_cache_latency_ms: (avg_cache_ms * 100.0).round() / 100.0,
            avg_upstream_latency_ms: (avg_up_ms * 100.0).round() / 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_tracker_accumulation() {
        let tracker = TelemetryTracker::new();

        tracker.record_miss(350_000.0); // 350ms miss
        let snap1 = tracker.snapshot();
        assert_eq!(snap1.total_requests, 1);
        assert_eq!(snap1.cache_misses, 1);
        assert_eq!(snap1.cache_hits, 0);

        // Record exact hit on gpt-4o
        tracker.record_exact_hit("gpt-4o", 100, 50, 80.0);
        let snap2 = tracker.snapshot();
        assert_eq!(snap2.total_requests, 2);
        assert_eq!(snap2.cache_hits, 1);
        assert_eq!(snap2.exact_hits, 1);
        assert_eq!(snap2.prompt_tokens_saved, 100);
        assert_eq!(snap2.completion_tokens_saved, 50);
        assert_eq!(snap2.total_tokens_saved, 150);
        assert_eq!(snap2.hit_ratio_percent, 50.0);

        // Record semantic hit on claude-3-5-sonnet
        tracker.record_semantic_hit("claude-3-5-sonnet", 1_000_000, 0, 4_500.0);
        let snap3 = tracker.snapshot();
        assert_eq!(snap3.total_requests, 3);
        assert_eq!(snap3.cache_hits, 2);
        assert_eq!(snap3.semantic_hits, 1);
        assert_eq!(snap3.prompt_tokens_saved, 1_000_100);
        assert_eq!(snap3.total_tokens_saved, 1_000_150);

        // Legacy record_hit
        tracker.record_hit(400, 1_200.0);
        let snap4 = tracker.snapshot();
        assert_eq!(snap4.total_requests, 4);
        assert_eq!(snap4.cache_hits, 3);
    }
}
