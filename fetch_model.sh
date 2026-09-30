#!/usr/bin/env bash
# VQ Agent Core: one-command download of the small local-LLM GGUF.
# Free, offline after download. Verified 2026-09-27.
#
# Model: Qwen2.5-0.5B-Instruct, Q4_K_M GGUF
# Repo:  Qwen/Qwen2.5-0.5B-Instruct-GGUF
# File:  qwen2.5-0.5b-instruct-q4_k_m.gguf  (~491 MB, 491400032 bytes)
# SHA-256: 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
#
# Usage:
#   ./fetch_model.sh                # downloads qwen2.5-0.5b-instruct-q4_k_m.gguf into ./models/
#   ./fetch_model.sh qwen2.5-1.5b   # downloads the bigger 1.5B Q4_K_M (~1.0 GB) instead
#
# Start the server (port 11434 matches config.yaml):
#   ./bin/llama/llama-b11146/llama-server -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf --port 11434
#
# NOTE: needs network egress through your proxy (see workspace AGENTS.md lesson).
# Do NOT put *.gguf files into vq-agent-core.zip (too big) -- the zip excludes models/.
set -euo pipefail

HF_BASE="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main"

declare -A FILES=(
  [qwen2.5-0.5b]="qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
declare -A SHAS=(
  [qwen2.5-0.5b]="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
)

# 1.5B alternative (step up if 0.5B proves too weak for tool-routing JSON).
# Repo: Qwen/Qwen2.5-1.5B-Instruct-GGUF, file qwen2.5-1.5b-instruct-q4_k_m.gguf (~986 MB).
declare -A ALT_FILES=( [qwen2.5-1.5b]="qwen2.5-1.5b-instruct-q4_k_m.gguf" )
declare -A ALT_BASE=( [qwen2.5-1.5b]="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main" )

WHICH="${1:-qwen2.5-0.5b}"
MODELS_DIR="$(cd "$(dirname "$0")" && pwd)/models"
mkdir -p "$MODELS_DIR"

if [[ "$WHICH" == "qwen2.5-1.5b" ]]; then
  BASE="${ALT_BASE[qwen2.5-1.5b]}"; FILE="${ALT_FILES[qwen2.5-1.5b]}"; SHA=""
else
  BASE="$HF_BASE"; FILE="${FILES[$WHICH]}"; SHA="${SHAS[$WHICH]:-}"
fi

DEST="$MODELS_DIR/$FILE"
echo "Downloading $FILE -> $DEST"
unset no_proxy NO_PROXY  # sandbox proxy quirk: keep *_proxy, drop no_proxy/NO_PROXY
curl -L --retry 3 --retry-delay 5 -o "$DEST" "$BASE/$FILE"

if [[ -n "$SHA" ]]; then
  echo "Verifying SHA-256..."
  echo "$SHA  $DEST" | sha256sum -c -
fi
echo "Done: $DEST ($(du -h "$DEST" | cut -f1))"
