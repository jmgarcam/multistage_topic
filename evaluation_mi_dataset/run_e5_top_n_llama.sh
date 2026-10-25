#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATASET_PATH="${DATASET_PATH:-${DATASET_DIR:-$PROJECT_DIR/tagged_data/test_tagged_data_no_none.json}}"
TOPICS_FILE="${TOPICS_FILE:-$PROJECT_DIR/temas.json}"
E5_RESULTS_FILE="${E5_RESULTS_FILE:-$SCRIPT_DIR/e5_results/e5_top_n_results.json}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results_e5_ministral_3_3b_instruct_two_prompts_no_margin}"
OLLAMA_MODEL="${OLLAMA_MODEL:-ministral-3:3b-instruct-2512-q4_K_M}"
RESULT_SLUG="${RESULT_SLUG:-ministral_3_3b_instruct_2512_q4_k_m}"

python3 "$PROJECT_DIR/evaluation/E5_top_N/evaluate_e5_top_n.py" \
  --data-dir "$DATASET_PATH" \
  --labelled-data-dir "$DATASET_PATH" \
  --topics "$TOPICS_FILE" \
  --e5-results "$E5_RESULTS_FILE" \
  --output "$E5_RESULTS_FILE"

exec python3 \
  "$PROJECT_DIR/evaluation/E5_top_N_llama/evaluate_e5_top_n_llama.py" \
  --model "$OLLAMA_MODEL" \
  --data-dir "$DATASET_PATH" \
  --labelled-data-dir "$DATASET_PATH" \
  --topics "$TOPICS_FILE" \
  --e5-results "$E5_RESULTS_FILE" \
  --results-dir "$RESULTS_DIR" \
  --result-slug "$RESULT_SLUG" \
  --disable-similarity-margin-gate \
  "$@"
