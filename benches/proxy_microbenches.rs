use criterion::{Criterion, black_box, criterion_group, criterion_main};
use kachedb_proxy::cache::canonicalize_messages;
use kachedb_proxy::embed::EmbeddingEngine;
use kachedb_proxy::embed::mock::MockEmbedder;
use serde_json::json;

fn bench_prompt_canonicalization(c: &mut Criterion) {
    let messages = vec![
        json!({"role": "system", "content": "You are an expert Rust systems programmer."}),
        json!({"role": "user", "content": "How do I implement an in-memory slab allocator in Rust?"}),
        json!({"role": "assistant", "content": "Here is an explanation of slab allocation with megaslabs..."}),
        json!({"role": "user", "content": "Can you show how free lists work?"}),
    ];

    c.bench_function("prompt_canonicalization_4_turns", |b| {
        b.iter(|| {
            canonicalize_messages(black_box(&messages));
        });
    });
}

fn bench_mock_embedding(c: &mut Criterion) {
    let embedder = MockEmbedder::default();
    let text = "How do I implement an in-memory slab allocator in Rust with zero copy?";

    c.bench_function("mock_embedding_384d", |b| {
        b.iter(|| {
            embedder.encode(black_box(text)).unwrap();
        });
    });
}

criterion_group!(benches, bench_prompt_canonicalization, bench_mock_embedding);
criterion_main!(benches);
