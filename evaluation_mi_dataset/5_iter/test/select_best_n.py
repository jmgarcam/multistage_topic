#!/usr/bin/env python3
"""Selecciona el N con mayor accuracy media para cada modelo evaluado."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
STATS_MODULE_DIR = (
    TEST_DIR.parent / 'plots' / 'accuracy_and_time_statistics'
)
sys.path.insert(0, str(STATS_MODULE_DIR))

from plot_accuracy_and_time_statistics import (  # noqa: E402
    EXPECTED_ITERATIONS,
    EXPECTED_N,
    MODEL_SPECS,
    aggregate,
    load_all_values,
)


BEST_OUTPUT = TEST_DIR / 'best_n_by_model.csv'
DETAIL_OUTPUT = TEST_DIR / 'accuracy_by_n.csv'
SUMMARY_OUTPUT = TEST_DIR / 'best_n_summary.md'
TIE_EPSILON = 1e-12


def rounded(value: float) -> float:
    return round(value, 6)


def family_and_size(label: str) -> tuple[str, str]:
    try:
        family, size = label.split(' · ', maxsplit=1)
    except ValueError as exc:
        raise SystemExit(f'No se pudo separar familia y tamaño: {label!r}.') from exc
    return family, size


def main() -> None:
    values = load_all_values()
    statistics_rows = aggregate(values)

    pipeline_rows = [row for row in statistics_rows if row.kind == 'pipeline']
    e5_rows = [row for row in statistics_rows if row.kind == 'e5_only']
    if len(e5_rows) != 1:
        raise SystemExit(f'Se esperaba un único resultado E5 y hay {len(e5_rows)}.')
    e5_accuracy = e5_rows[0].accuracy_mean

    rows_by_model = defaultdict(list)
    for row in pipeline_rows:
        rows_by_model[row.model].append(row)

    runs_by_key = defaultdict(dict)
    for value in values:
        if value.kind == 'pipeline':
            runs_by_key[(value.model, value.n)][value.iteration] = (
                value.accuracy_percent
            )

    detail_fields = (
        'family', 'size', 'model', 'n', 'runs',
        'accuracy_mean_percent', 'accuracy_sd_pp',
        'accuracy_min_percent', 'accuracy_max_percent',
        'accuracy_ci95_low_percent', 'accuracy_ci95_high_percent',
        'mean_total_time_seconds', 'selected_as_best',
        *[f'{iteration}_accuracy_percent' for iteration in EXPECTED_ITERATIONS],
    )
    best_fields = (
        'family', 'size', 'model', 'selected_n', 'all_best_n', 'runs',
        'accuracy_mean_percent', 'accuracy_sd_pp',
        'accuracy_min_percent', 'accuracy_max_percent',
        'accuracy_ci95_low_percent', 'accuracy_ci95_high_percent',
        'mean_total_time_seconds', 'delta_vs_e5_pp',
        *[f'{iteration}_accuracy_percent' for iteration in EXPECTED_ITERATIONS],
    )

    detail_payloads: list[dict[str, object]] = []
    best_payloads: list[dict[str, object]] = []

    for spec in MODEL_SPECS:
        family, size = family_and_size(spec.label)
        model_rows = sorted(
            rows_by_model.get(spec.ollama_model, []), key=lambda row: row.n or 0
        )
        found_n = tuple(row.n for row in model_rows)
        if found_n != EXPECTED_N:
            raise SystemExit(
                f'{spec.label}: N encontrados {found_n}; se esperaban {EXPECTED_N}.'
            )

        maximum = max(row.accuracy_mean for row in model_rows)
        best_rows = [
            row
            for row in model_rows
            if abs(row.accuracy_mean - maximum) <= TIE_EPSILON
        ]
        selected = min(best_rows, key=lambda row: row.n or 0)
        all_best_n = '|'.join(str(row.n) for row in best_rows)

        for row in model_rows:
            iterations = runs_by_key[(row.model, row.n)]
            if tuple(sorted(iterations)) != EXPECTED_ITERATIONS:
                raise SystemExit(
                    f'{spec.label}, N={row.n}: iteraciones incompletas.'
                )
            detail_payloads.append(
                {
                    'family': family,
                    'size': size,
                    'model': row.model,
                    'n': row.n,
                    'runs': row.runs,
                    'accuracy_mean_percent': rounded(row.accuracy_mean),
                    'accuracy_sd_pp': rounded(row.accuracy_sd),
                    'accuracy_min_percent': rounded(row.accuracy_min),
                    'accuracy_max_percent': rounded(row.accuracy_max),
                    'accuracy_ci95_low_percent': rounded(row.accuracy_ci95_low),
                    'accuracy_ci95_high_percent': rounded(row.accuracy_ci95_high),
                    'mean_total_time_seconds': rounded(row.time_mean),
                    'selected_as_best': row in best_rows,
                    **{
                        f'{iteration}_accuracy_percent': rounded(
                            iterations[iteration]
                        )
                        for iteration in EXPECTED_ITERATIONS
                    },
                }
            )

        selected_iterations = runs_by_key[(selected.model, selected.n)]
        best_payloads.append(
            {
                'family': family,
                'size': size,
                'model': selected.model,
                'selected_n': selected.n,
                'all_best_n': all_best_n,
                'runs': selected.runs,
                'accuracy_mean_percent': rounded(selected.accuracy_mean),
                'accuracy_sd_pp': rounded(selected.accuracy_sd),
                'accuracy_min_percent': rounded(selected.accuracy_min),
                'accuracy_max_percent': rounded(selected.accuracy_max),
                'accuracy_ci95_low_percent': rounded(
                    selected.accuracy_ci95_low
                ),
                'accuracy_ci95_high_percent': rounded(
                    selected.accuracy_ci95_high
                ),
                'mean_total_time_seconds': rounded(selected.time_mean),
                'delta_vs_e5_pp': rounded(selected.accuracy_mean - e5_accuracy),
                **{
                    f'{iteration}_accuracy_percent': rounded(
                        selected_iterations[iteration]
                    )
                    for iteration in EXPECTED_ITERATIONS
                },
            }
        )

    with DETAIL_OUTPUT.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_payloads)

    with BEST_OUTPUT.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=best_fields)
        writer.writeheader()
        writer.writerows(best_payloads)

    summary = [
        '# N óptimo por modelo y tamaño',
        '',
        'Criterio: maximizar la **accuracy media** de las cinco iteraciones. '
        'Si varios valores de N tienen exactamente la misma media, se muestran '
        'todos y se selecciona el N más pequeño.',
        '',
        '| Familia | Tamaño | N seleccionado | Todos los N máximos | '
        'Accuracy media (%) | SD (pp) | Diferencia frente a E5 (pp) |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in best_payloads:
        summary.append(
            f"| {row['family']} | {row['size']} | {row['selected_n']} | "
            f"{row['all_best_n']} | {row['accuracy_mean_percent']:.3f} | "
            f"{row['accuracy_sd_pp']:.3f} | {row['delta_vs_e5_pp']:+.3f} |"
        )
    summary.extend(
        (
            '',
            f'Accuracy media de E5 Top-1: **{e5_accuracy:.3f}%**.',
            '',
            '> Nota metodológica: estos N se han seleccionado usando los '
            'resultados del conjunto de test. Para estimar rendimiento no '
            'sesgado en un paper, N debería seleccionarse con validation y '
            'evaluarse una sola vez en test.',
            '',
        )
    )
    SUMMARY_OUTPUT.write_text('\n'.join(summary), encoding='utf-8')

    print(f'Modelos analizados: {len(best_payloads)}')
    print(f'Configuraciones N analizadas: {len(detail_payloads)}')
    print(f'Resultado: {BEST_OUTPUT}')
    for row in best_payloads:
        print(
            f"- {row['family']} {row['size']}: N={row['selected_n']}, "
            f"accuracy={row['accuracy_mean_percent']:.3f}%"
        )


if __name__ == '__main__':
    main()
