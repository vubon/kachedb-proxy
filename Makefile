# ==============================================================================
# ⚡ KacheDB OpenAI & Anthropic Semantic Reverse Proxy (`kachedb-proxy`)
# ==============================================================================

SHELL := /bin/bash
CARGO ?= cargo
PYTHON ?= python3

# Target configurations
PORT ?= 6385
UPSTREAM ?= http://127.0.0.1:8081/v1
KACHEDB_PORT ?= 6381
THRESHOLD ?= 0.85

.PHONY: all help build release run run-release \
        test test-unit test-e2e test-all \
        coverage coverage-html coverage-lcov \
        lint fmt fmt-check clippy check bench clean

all: lint test

# ------------------------------------------------------------------------------
# 📖 Help & Discovery
# ------------------------------------------------------------------------------
help:
	@echo "⚡ KacheDB Semantic Reverse Proxy (kachedb-proxy) Makefile"
	@echo "=================================================================="
	@echo "Build & Execution:"
	@echo "  make build         - Compile debug binary"
	@echo "  make release       - Compile production release binary"
	@echo "  make run           - Run debug proxy (port: $(PORT), upstream: $(UPSTREAM))"
	@echo "  make run-release   - Run release proxy"
	@echo ""
	@echo "Testing & Validation:"
	@echo "  make test          - Run all Rust unit & integration tests"
	@echo "  make test-unit     - Run Rust unit tests only"
	@echo "  make test-e2e      - Run Python E2E integration test suite with local llama.cpp"
	@echo "  make test-all      - Run both Rust unit tests and Python E2E suite"
	@echo ""
	@echo "Code Coverage:"
	@echo "  make coverage      - Generate terminal code coverage report (cargo llvm-cov)"
	@echo "  make coverage-html - Generate interactive HTML coverage report"
	@echo "  make coverage-lcov - Generate lcov.info file for CI/Codecov"
	@echo ""
	@echo "Linting & Quality Gates:"
	@echo "  make lint          - Run all linters (cargo fmt, clippy, ruff)"
	@echo "  make fmt           - Auto-format Rust and Python codebases"
	@echo "  make clippy        - Run strict clippy with -D warnings"
	@echo "  make check         - Run fast cargo check"
	@echo ""
	@echo "Benchmarks & Utility:"
	@echo "  make chat          - Launch interactive streaming terminal chat connected to proxy"
	@echo "  make bench         - Run Criterion micro-benchmarks"
	@echo "  make clean         - Remove build artifacts, target dir, and test caches"
	@echo "=================================================================="

# ------------------------------------------------------------------------------
# 🛠️ Build Targets
# ------------------------------------------------------------------------------
build:
	$(CARGO) build

release:
	$(CARGO) build --release

check:
	$(CARGO) check --all-targets

run:
	$(CARGO) run -- --port $(PORT) --upstream-url $(UPSTREAM) --kachedb-port $(KACHEDB_PORT) --similarity-threshold $(THRESHOLD)

run-release:
	$(CARGO) run --release -- --port $(PORT) --upstream-url $(UPSTREAM) --kachedb-port $(KACHEDB_PORT) --similarity-threshold $(THRESHOLD)

# ------------------------------------------------------------------------------
# 🧪 Testing Targets
# ------------------------------------------------------------------------------
test: test-unit

test-unit:
	$(CARGO) test --all-targets

test-e2e:
	@if [ -f "./scripts/run_local_llama_test.sh" ]; then \
		./scripts/run_local_llama_test.sh; \
	else \
		$(PYTHON) -m pytest -v tests/test_proxy_e2e.py; \
	fi

test-all: test-unit test-e2e

# ------------------------------------------------------------------------------
# 📊 Code Coverage
# ------------------------------------------------------------------------------
coverage:
	@if command -v cargo-llvm-cov >/dev/null 2>&1; then \
		$(CARGO) llvm-cov --all-targets; \
	elif command -v cargo-tarpaulin >/dev/null 2>&1; then \
		$(CARGO) tarpaulin --all-targets; \
	else \
		echo "❌ Neither cargo-llvm-cov nor cargo-tarpaulin found."; \
		echo "Install with: brew install cargo-llvm-cov (or cargo install cargo-llvm-cov)"; \
		exit 1; \
	fi

coverage-html:
	@if command -v cargo-llvm-cov >/dev/null 2>&1; then \
		$(CARGO) llvm-cov --all-targets --html --output-dir target/llvm-cov/html; \
		echo "✅ Interactive HTML coverage report generated at: target/llvm-cov/html/index.html"; \
	else \
		echo "❌ cargo-llvm-cov not found. Install with: brew install cargo-llvm-cov"; \
		exit 1; \
	fi

coverage-lcov:
	@if command -v cargo-llvm-cov >/dev/null 2>&1; then \
		$(CARGO) llvm-cov --all-targets --lcov --output-path target/lcov.info; \
		echo "✅ LCOV coverage file generated at: target/lcov.info"; \
	else \
		echo "❌ cargo-llvm-cov not found."; \
		exit 1; \
	fi

# ------------------------------------------------------------------------------
# 🧹 Linting & Formatting
# ------------------------------------------------------------------------------
lint: clippy fmt-check
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check tests/ benchmarks/ scripts/; \
	elif $(PYTHON) -m ruff --version >/dev/null 2>&1; then \
		$(PYTHON) -m ruff check tests/ benchmarks/ scripts/; \
	fi

clippy:
	$(CARGO) clippy --all-targets -- -D warnings

fmt-check:
	$(CARGO) fmt --all -- --check
	@if command -v ruff >/dev/null 2>&1; then \
		ruff format --check tests/ benchmarks/ scripts/; \
	elif $(PYTHON) -m ruff --version >/dev/null 2>&1; then \
		$(PYTHON) -m ruff format --check tests/ benchmarks/ scripts/; \
	fi

fmt:
	$(CARGO) fmt --all
	@if command -v ruff >/dev/null 2>&1; then \
		ruff format tests/ benchmarks/ scripts/; \
		ruff check --fix tests/ benchmarks/ scripts/; \
	elif $(PYTHON) -m ruff --version >/dev/null 2>&1; then \
		$(PYTHON) -m ruff format tests/ benchmarks/ scripts/; \
		$(PYTHON) -m ruff check --fix tests/ benchmarks/ scripts/; \
	fi

# ------------------------------------------------------------------------------
# 🐳 Docker Compose Stack Targets
# ------------------------------------------------------------------------------
compose-up:
	docker compose -f docker-compose.ai.yml up -d

compose-down:
	docker compose -f docker-compose.ai.yml down

compose-local-llm:
	docker compose -f docker-compose.ai.yml --profile local-llm up -d

compose-logs:
	docker compose -f docker-compose.ai.yml logs -f

# ------------------------------------------------------------------------------
# 💬 Interactive Chat
# ------------------------------------------------------------------------------
chat:
	$(PYTHON) scripts/interactive_chat.py

# ------------------------------------------------------------------------------
# ⚡ Benchmarks
# ------------------------------------------------------------------------------
bench:
	$(CARGO) bench

# ------------------------------------------------------------------------------
# 🧼 Clean
# ------------------------------------------------------------------------------
clean:
	$(CARGO) clean
	rm -rf .pytest_cache .coverage target/llvm-cov target/lcov.info .ruff_cache .fastembed_cache

