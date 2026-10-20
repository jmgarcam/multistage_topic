#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"

exec python3 "$SCRIPT_DIR/evaluate_llama_top_1.py" "$@"
