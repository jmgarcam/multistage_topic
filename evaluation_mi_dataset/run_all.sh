#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ITERATIONS="${ITERATIONS:-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/5_iter}"
DATASET_PATH="${DATASET_PATH:-${DATASET_DIR:-$PROJECT_DIR/tagged_data/test_tagged_data_no_none.json}}"

if [[ ! "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'ITERATIONS debe ser un entero mayor que cero; recibido: %s\n' \
    "$ITERATIONS" >&2
  exit 2
fi

mkdir -p -- "$OUTPUT_ROOT"

run_model() {
  local ollama_model="$1"
  local result_dir_slug="$2"
  local result_file_slug="$3"
  local iteration_dir="$4"
  local e5_results_file="$iteration_dir/e5_results/e5_top_n_results.json"
  local pipeline_results_dir
  local direct_result_file

  pipeline_results_dir="$iteration_dir/results_e5_${result_dir_slug}_two_prompts_no_margin"
  direct_result_file="$iteration_dir/results_${result_dir_slug}_two_prompts/${result_file_slug}_direct_results.json"

  printf '[%s] Iniciando pipeline E5 -> %s\n' \
    "$(date --iso-8601=seconds)" "$ollama_model"
  OLLAMA_MODEL="$ollama_model" \
  DATASET_PATH="$DATASET_PATH" \
  RESULT_SLUG="$result_file_slug" \
  E5_RESULTS_FILE="$e5_results_file" \
  RESULTS_DIR="$pipeline_results_dir" \
    "$SCRIPT_DIR/run_e5_top_n_llama.sh"

  printf '[%s] Iniciando evaluación directa de %s\n' \
    "$(date --iso-8601=seconds)" "$ollama_model"
  OLLAMA_MODEL="$ollama_model" \
  DATASET_PATH="$DATASET_PATH" \
  SLM_ONLY_OUTPUT="$direct_result_file" \
    "$SCRIPT_DIR/run_llama_only.sh"

  printf '[%s] Finalizada la evaluación de %s\n' \
    "$(date --iso-8601=seconds)" "$ollama_model"
}

for ((iteration = 1; iteration <= ITERATIONS; iteration++)); do
  printf -v iteration_name 'iter_%02d' "$iteration"
  iteration_dir="$OUTPUT_ROOT/$iteration_name"
  mkdir -p -- "$iteration_dir"

  printf '[%s] Iniciando iteración %d/%d: %s\n' \
    "$(date --iso-8601=seconds)" "$iteration" "$ITERATIONS" \
    "$iteration_dir"

  run_model \
    'llama3.2:1b' \
    'llama3_2_1b' \
    'llama3_2_1b' \
    "$iteration_dir"
  run_model \
    'llama3.2:3b' \
    'llama3_2_3b' \
    'llama3_2_3b' \
    "$iteration_dir"
  run_model \
    'llama3.1:8b' \
    'llama3_1_8b' \
    'llama3_1_8b' \
    "$iteration_dir"
  run_model \
    'qwen2.5:1.5b-instruct' \
    'qwen2_5_1_5b_instruct' \
    'qwen2_5_1_5b_instruct' \
    "$iteration_dir"
  run_model \
    'qwen2.5:3b-instruct' \
    'qwen2_5_3b_instruct' \
    'qwen2_5_3b_instruct' \
    "$iteration_dir"
  run_model \
    'qwen2.5:7b-instruct' \
    'qwen2_5_7b_instruct' \
    'qwen2_5_7b_instruct' \
    "$iteration_dir"
  run_model \
    'ministral-3:3b-instruct-2512-q4_K_M' \
    'ministral_3_3b_instruct' \
    'ministral_3_3b_instruct_2512_q4_k_m' \
    "$iteration_dir"
  run_model \
    'ministral-3:8b-instruct-2512-q4_K_M' \
    'ministral_3_8b_instruct' \
    'ministral_3_8b_instruct_2512_q4_k_m' \
    "$iteration_dir"

  printf '[%s] Finalizada la iteración %d/%d\n' \
    "$(date --iso-8601=seconds)" "$iteration" "$ITERATIONS"
done

printf '[%s] Finalizadas las %d iteraciones. Resultados: %s\n' \
  "$(date --iso-8601=seconds)" "$ITERATIONS" "$OUTPUT_ROOT"
