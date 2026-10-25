#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATASET_PATH="${DATASET_PATH:-${DATASET_DIR:-$PROJECT_DIR/tagged_data/test_tagged_data_no_none.json}}"
TOPICS_FILE="${TOPICS_FILE:-$PROJECT_DIR/temas.json}"
OLLAMA_MODEL="${OLLAMA_MODEL:-ministral-3:3b-instruct-2512-q4_K_M}"
DEFAULT_SLM_OUTPUT="$SCRIPT_DIR/results_ministral_3_3b_instruct_two_prompts/ministral_3_3b_instruct_2512_q4_k_m_direct_results.json"
SLM_ONLY_OUTPUT="${SLM_ONLY_OUTPUT:-${LLAMA_ONLY_OUTPUT:-$DEFAULT_SLM_OUTPUT}}"

exec python3 \
  "$PROJECT_DIR/evaluation/Llama_top_1/evaluate_llama_top_1.py" \
  --model "$OLLAMA_MODEL" \
  --data-dir "$DATASET_PATH" \
  --labelled-data-dir "$DATASET_PATH" \
  --topics "$TOPICS_FILE" \
  --output "$SLM_ONLY_OUTPUT" \
  "$@"
