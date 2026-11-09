#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for plot in \
  accuracy_vs_candidates \
  accuracy_vs_time \
  accuracy_vs_time_by_model \
  corrections_and_harms \
  n_vs_time \
  recall_and_selection_accuracy
do
  python3 "$SCRIPT_DIR/$plot/plot_${plot}.py"
done

python3 \
  "$SCRIPT_DIR/accuracy_vs_candidates/plot_accuracy_vs_candidates_by_family.py"

python3 \
  "$SCRIPT_DIR/accuracy_and_time_statistics/plot_accuracy_and_time_statistics.py"
