#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

TEST_DATA="${TEST_DATA:-$PROJECT_DIR/tagged_data/test_tagged_data_no_none.json}"
TOPICS_FILE="${TOPICS_FILE:-$PROJECT_DIR/temas.json}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/intfloat_model_sizes}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVALUATOR="$PROJECT_DIR/evaluation/E5_top_N/evaluate_e5_top_n.py"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
  elif command -v pipenv >/dev/null 2>&1; then
    project_venv="$(
      PIPENV_PIPFILE="$PROJECT_DIR/Pipfile" pipenv --venv 2>/dev/null || true
    )"
    if [[ -n "$project_venv" && -x "$project_venv/bin/python" ]]; then
      PYTHON_BIN="$project_venv/bin/python"
    else
      PYTHON_BIN=python3
    fi
  else
    PYTHON_BIN=python3
  fi
fi

if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  printf 'BATCH_SIZE debe ser un entero mayor que cero; recibido: %s\n' \
    "$BATCH_SIZE" >&2
  exit 2
fi

for required_file in "$TEST_DATA" "$TOPICS_FILE" "$EVALUATOR"; do
  if [[ ! -f "$required_file" ]]; then
    printf 'No se encontró el archivo requerido: %s\n' "$required_file" >&2
    exit 2
  fi
done

if ! "$PYTHON_BIN" -B -c 'import sentence_transformers, torch' 2>/dev/null; then
  printf 'El Python seleccionado no contiene sentence-transformers y torch: %s\n' \
    "$PYTHON_BIN" >&2
  printf 'Activa el entorno del proyecto o define PYTHON_BIN explícitamente.\n' >&2
  exit 2
fi

test_size="$("$PYTHON_BIN" -B -c \
  'import json, sys; data=json.load(open(sys.argv[1], encoding="utf-8")); print(len(data) if isinstance(data, list) else -1)' \
  "$TEST_DATA")"

if [[ "$test_size" != 600 ]]; then
  printf 'El conjunto de test debe contener exactamente 600 noticias; contiene %s.\n' \
    "$test_size" >&2
  exit 2
fi

mkdir -p -- "$RESULTS_DIR"

models=(
  'small|intfloat/multilingual-e5-small|multilingual_e5_small_results.json'
  'base|intfloat/multilingual-e5-base|multilingual_e5_base_results.json'
  'large|intfloat/multilingual-e5-large|multilingual_e5_large_results.json'
)

printf '[%s] Dataset: %s (%s noticias)\n' \
  "$(date --iso-8601=seconds)" "$TEST_DATA" "$test_size"
printf '[%s] Python: %s\n' "$(date --iso-8601=seconds)" "$PYTHON_BIN"
printf '[%s] Resultados: %s\n' \
  "$(date --iso-8601=seconds)" "$RESULTS_DIR"

for entry in "${models[@]}"; do
  IFS='|' read -r size model output_name <<< "$entry"
  output_file="$RESULTS_DIR/$output_name"

  printf '\n[%s] Iniciando E5 %s: %s\n' \
    "$(date --iso-8601=seconds)" "$size" "$model"

  "$PYTHON_BIN" -B "$EVALUATOR" \
    --data-dir "$TEST_DATA" \
    --labelled-data-dir "$TEST_DATA" \
    --topics "$TOPICS_FILE" \
    --model "$model" \
    --batch-size "$BATCH_SIZE" \
    --output "$output_file" \
    --recompute-e5

  printf '[%s] Finalizado E5 %s: %s\n' \
    "$(date --iso-8601=seconds)" "$size" "$output_file"
done

summary_file="$RESULTS_DIR/comparison.csv"
"$PYTHON_BIN" -B - "$RESULTS_DIR" "$summary_file" <<'PY'
import csv
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
summary_file = Path(sys.argv[2])
paths = (
    results_dir / 'multilingual_e5_small_results.json',
    results_dir / 'multilingual_e5_base_results.json',
    results_dir / 'multilingual_e5_large_results.json',
)
rows = []
for path in paths:
    with path.open(encoding='utf-8') as handle:
        report = json.load(handle)
    configuration = report['configuration']
    dataset = report['dataset']
    metrics = report['metrics']
    timing = report['timing']
    rows.append(
        {
            'model': configuration['model'],
            'total_news': dataset['total_news'],
            'top_1_hits': metrics['top_1_hits'],
            'top_1_accuracy_percent': metrics['top_1_accuracy_percent'],
            'mean_first_correct_rank': metrics['mean_first_correct_rank'],
            'mean_reciprocal_rank': metrics['mean_reciprocal_rank'],
            'total_e5_execution_time_seconds': timing[
                'total_e5_execution_time_seconds'
            ],
            'device': configuration['device'],
            'result_file': str(path.resolve()),
        }
    )

with summary_file.open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)

print(f'Resumen comparativo: {summary_file.resolve()}')
PY

printf '[%s] Finalizada la comparación de los tres tamaños.\n' \
  "$(date --iso-8601=seconds)"
