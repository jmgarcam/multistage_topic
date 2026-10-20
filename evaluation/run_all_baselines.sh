#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAX_ATTEMPTS=${BASELINE_MAX_ATTEMPTS:-3}
RETRY_DELAY_SECONDS=${BASELINE_RETRY_DELAY_SECONDS:-60}
RESUME_RESULTS=${BASELINE_RESUME_RESULTS:-1}

if ! [[ $MAX_ATTEMPTS =~ ^[1-9][0-9]*$ ]]; then
    printf 'BASELINE_MAX_ATTEMPTS debe ser un entero positivo.\n' >&2
    exit 2
fi
if ! [[ $RETRY_DELAY_SECONDS =~ ^[0-9]+$ ]]; then
    printf 'BASELINE_RETRY_DELAY_SECONDS debe ser un entero no negativo.\n' >&2
    exit 2
fi
if ! [[ $RESUME_RESULTS =~ ^[01]$ ]]; then
    printf 'BASELINE_RESUME_RESULTS debe valer 0 o 1.\n' >&2
    exit 2
fi

BASELINE_NAMES=(
    'E5 Top-1'
    'E5 Top-N'
    'BGE Top-1'
    'BGE Top-N'
    'Llama Top-1'
    'E5 Top-N + Llama'
    'BGE Top-N + Llama'
)
BASELINE_SLUGS=(
    e5_top_1
    e5_top_n
    bge_top_1
    bge_top_n
    llama_top_1
    e5_top_n_llama
    bge_top_n_llama
)
BASELINE_SCRIPTS=(
    E5_top_1/calculate_top_1.sh
    E5_top_N/run_e5_top_n.sh
    BGE_top_1/calculate_top_1.sh
    BGE_top_N/run_bge_top_n.sh
    Llama_top_1/run_llama_top_1.sh
    E5_top_N_llama/run_e5_top_n_llama.sh
    BGE_top_N_llama/run_bge_top_n_llama.sh
)

for index in "${!BASELINE_SCRIPTS[@]}"; do
    BASELINE_SCRIPTS[$index]="$SCRIPT_DIR/${BASELINE_SCRIPTS[$index]}"
    if [[ ! -x ${BASELINE_SCRIPTS[$index]} ]]; then
        printf 'No existe o no es ejecutable: %s\n' \
            "${BASELINE_SCRIPTS[$index]}" >&2
        exit 2
    fi
done

set_launcher_args() {
    LAUNCHER_ARGS=()
    if ((RESUME_RESULTS == 1)); then
        case $1 in
            e5_top_1|bge_top_1)
                LAUNCHER_ARGS=(--reuse-existing)
                ;;
        esac
    fi
}

if [[ ${1:-} == '--help' || ${1:-} == '-h' ]]; then
    printf '%s\n' \
        'Uso: evaluation/run_all_baselines.sh [--dry-run]' \
        'Ejecuta consecutivamente los siete baselines.' \
        'Configurable con BASELINE_MAX_ATTEMPTS,' \
        'BASELINE_RETRY_DELAY_SECONDS y BASELINE_RESUME_RESULTS.'
    exit 0
fi

if [[ ${1:-} == '--dry-run' ]]; then
    printf 'Orden de ejecución (%d baselines):\n' "${#BASELINE_NAMES[@]}"
    for index in "${!BASELINE_NAMES[@]}"; do
        set_launcher_args "${BASELINE_SLUGS[$index]}"
        printf '%d. %s -> %s' \
            "$((index + 1))" \
            "${BASELINE_NAMES[$index]}" \
            "${BASELINE_SCRIPTS[$index]}"
        if ((${#LAUNCHER_ARGS[@]} > 0)); then
            printf ' %s' "${LAUNCHER_ARGS[@]}"
        fi
        printf '\n'
    done
    exit 0
fi

if (($# > 0)); then
    printf 'Argumento desconocido: %s\n' "$1" >&2
    exit 2
fi

RUN_ID=${BASELINE_RUN_ID:-$(date '+%Y%m%d_%H%M%S')}
LOG_DIR=${BASELINE_LOG_DIR:-$SCRIPT_DIR/run_logs/$RUN_ID}
mkdir -p "$LOG_DIR"
MASTER_LOG=$LOG_DIR/all_baselines.log
SUMMARY_FILE=$LOG_DIR/summary.tsv
exec > >(tee -a "$MASTER_LOG") 2>&1

printf 'baseline\tstatus\texit_code\tattempts\tduration_seconds\tlog\n' \
    > "$SUMMARY_FILE"
printf 'Inicio: %s\n' "$(date --iso-8601=seconds)"
printf 'Directorio de logs: %s\n' "$LOG_DIR"
printf 'Intentos máximos por baseline: %s\n' "$MAX_ATTEMPTS"
printf 'Espera entre intentos: %s segundos\n' "$RETRY_DELAY_SECONDS"
printf 'Reutilizar resultados Top-1 validados: %s\n' "$RESUME_RESULTS"
printf 'Modelo Ollama: %s\n' "${OLLAMA_MODEL:-llama3.1:8b}"

RESULT_NAMES=()
RESULT_STATUSES=()
RESULT_CODES=()
FAILURES=0

for index in "${!BASELINE_NAMES[@]}"; do
    name=${BASELINE_NAMES[$index]}
    slug=${BASELINE_SLUGS[$index]}
    launcher=${BASELINE_SCRIPTS[$index]}
    set_launcher_args "$slug"
    baseline_log=$LOG_DIR/${slug}.log
    start_epoch=$(date +%s)
    attempt=1
    exit_code=1

    printf '\n[%s] Iniciando %d/%d: %s\n' \
        "$(date --iso-8601=seconds)" \
        "$((index + 1))" \
        "${#BASELINE_NAMES[@]}" \
        "$name"

    while ((attempt <= MAX_ATTEMPTS)); do
        {
            printf '[%s] Intento %d/%d\n' \
                "$(date --iso-8601=seconds)" \
                "$attempt" \
                "$MAX_ATTEMPTS"
            "$launcher" "${LAUNCHER_ARGS[@]}"
        } 2>&1 | tee -a "$baseline_log"
        exit_code=${PIPESTATUS[0]}

        if ((exit_code == 0)); then
            break
        fi
        if ((attempt == MAX_ATTEMPTS)); then
            break
        fi
        printf '[%s] %s falló con código %d; reintento en %d s.\n' \
            "$(date --iso-8601=seconds)" \
            "$name" \
            "$exit_code" \
            "$RETRY_DELAY_SECONDS" | tee -a "$baseline_log"
        sleep "$RETRY_DELAY_SECONDS"
        attempt=$((attempt + 1))
    done

    end_epoch=$(date +%s)
    duration=$((end_epoch - start_epoch))
    if ((exit_code == 0)); then
        status=OK
    else
        status=FAILED
        FAILURES=$((FAILURES + 1))
    fi

    RESULT_NAMES+=("$name")
    RESULT_STATUSES+=("$status")
    RESULT_CODES+=("$exit_code")
    printf '%s\t%s\t%d\t%d\t%d\t%s\n' \
        "$name" \
        "$status" \
        "$exit_code" \
        "$attempt" \
        "$duration" \
        "$baseline_log" >> "$SUMMARY_FILE"
    printf '[%s] Finalizado %s: %s (%d s).\n' \
        "$(date --iso-8601=seconds)" \
        "$name" \
        "$status" \
        "$duration"
done

printf '\nResumen:\n'
for index in "${!RESULT_NAMES[@]}"; do
    printf '  %-20s %-6s código=%s\n' \
        "${RESULT_NAMES[$index]}" \
        "${RESULT_STATUSES[$index]}" \
        "${RESULT_CODES[$index]}"
done
printf 'Fin: %s\n' "$(date --iso-8601=seconds)"
printf 'Resumen tabulado: %s\n' "$SUMMARY_FILE"
printf 'Log completo: %s\n' "$MASTER_LOG"

if ((FAILURES > 0)); then
    exit 1
fi
