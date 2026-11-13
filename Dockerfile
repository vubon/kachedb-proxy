# Multi-stage Dockerfile for kachedb-proxy
# Builds high-performance, minimal container image supporting linux/amd64 & linux/arm64

FROM rust:1.97-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    binutils \
    ca-certificates \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests
COPY Cargo.toml Cargo.lock ./

# Build dependencies cache
RUN mkdir src && echo "fn main() {}" > src/main.rs && echo "" > src/lib.rs
RUN cargo build --release || true

# Copy full source tree
COPY . .
RUN touch src/main.rs src/lib.rs

# Build release binary, inspect dynamic linkage, and strip debug symbols
RUN cargo build --release && \
    strip /app/target/release/kachedb-proxy && \
    (ldd /app/target/release/kachedb-proxy || true)

# Collect any dynamic runtime libraries (e.g. ONNX runtime) into /app/lib
RUN mkdir -p /app/lib && \
    (find /root/.cache /app/target/release -name "libonnxruntime*.so*" -exec cp {} /app/lib/ \; || true)

# ── Runtime Stage ─────────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd -r -m -u 10001 kache && \
    mkdir -p /app/.fastembed_cache && \
    chown -R kache:kache /app

COPY --from=builder /app/target/release/kachedb-proxy /usr/local/bin/kachedb-proxy
COPY --from=builder /app/lib/ /usr/local/lib/
RUN ldconfig

USER kache

EXPOSE 6385

ENV KACHEDB_PROXY_PORT=6385 \
    KACHEDB_PROXY_HOST=0.0.0.0 \
    KACHEDB_HOST=127.0.0.1 \
    KACHEDB_PORT=6379 \
    SIMILARITY_THRESHOLD=0.85 \
    CACHE_TTL_SECONDS=86400

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -fs http://127.0.0.1:6385/health || exit 1

ENTRYPOINT ["kachedb-proxy"]
