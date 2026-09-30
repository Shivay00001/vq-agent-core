#!/usr/bin/env bash
# VQ Agent Core: start the local LLM server the agent's THINK layer talks to.
# Free, offline. Matches config.yaml: base_url http://localhost:11434/v1,
# model qwen2.5-0.5b-instruct.
#
# Usage:
#   ./start_server.sh              # foreground (Ctrl+C to stop)
#   nohup ./start_server.sh > server.log 2>&1 &   # background
#
# Needs: ./fetch_model.sh must have been run first (models/*.gguf).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GGUF="$HERE/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
if [[ ! -f "$GGUF" ]]; then
  echo "ERROR: $GGUF not found. Run ./fetch_model.sh first." >&2
  exit 1
fi
exec "$HERE/bin/llama/llama-b11146/llama-server" -m "$GGUF" --port 11434 --threads 2
