#!/usr/bin/env bash
set -euo pipefail

# run_local_llama_test.sh — End-to-end local zero-cost validation for kachedb-proxy
#
# Prerequisite:
#   - KacheDB running on 127.0.0.1:6381 (or 6379)
#   - llama-server installed (via brew install llama.cpp)
#   - GGUF model in database/models/qwen2.5-0.5b-instruct-q4_k_m.gguf

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${ROOT_DIR}/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
LLAMA_PORT=8081
PROXY_PORT=6385
KACHEDB_PORT=6381

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "❌ Model not found at: ${MODEL_PATH}"
  echo "Run: curl -L -o \"${MODEL_PATH}\" https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
  exit 1
fi

echo "=========================================================="
echo "⚡ KacheDB Proxy Local Llama.cpp E2E Validation"
echo "=========================================================="
echo "• Model:      ${MODEL_PATH}"
echo "• Upstream:   http://127.0.0.1:${LLAMA_PORT}/v1"
echo "• Proxy:      http://127.0.0.1:${PROXY_PORT}"
echo "• KacheDB:    127.0.0.1:${KACHEDB_PORT}"
echo "=========================================================="

echo "🧪 Running Pytest E2E Suite..."
PROXY_BASE_URL="http://127.0.0.1:${PROXY_PORT}" python3 -m pytest -v "${ROOT_DIR}/kachedb-proxy/tests/test_proxy_e2e.py"

echo "✅ All tests passed successfully!"

