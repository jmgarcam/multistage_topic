#!/usr/bin/env python3
'''Representa el tiempo medio de reranking y el recall para cada valor de k.'''

from __future__ import annotations

import json
import math
import statistics
from numbers import Real
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


INPUT_FILE = '../reranked_topic_results.json'
POSITIONS_FILE = '../semantic_similarity_validation/top_19_positions.json'
OUTPUT_FILE = 'rerank_time_recall.pdf'


def resolve_path(configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def load_json(path: Path) -> Any:
    try:
        with path.open('r', encoding='utf-8') as file_handle:
            return json.load(file_handle)
    except FileNotFoundError as exc:
        raise SystemExit(f'No se encontró el archivo: {path}') from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f'JSON no válido en {path}: {exc}') from exc


def require_non_negative_time(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SystemExit(f'{location}: rerank_time_seconds debe ser un número.')

    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < 0:
        raise SystemExit(
            f'{location}: rerank_time_seconds debe ser un número finito '
            'no negativo.'
        )
    return numeric_value


def calculate_mean_times(payload: Any) -> list[dict[str, int | float]]:
    if not isinstance(payload, list) or not payload:
        raise SystemExit('El JSON de entrada debe contener una lista no vacía.')

    times_by_k: dict[int, list[float]] = {}
    expected_k_values: set[int] | None = None

    for record_index, record in enumerate(payload):
        location = f'Registro {record_index}'
        if not isinstance(record, dict):
            raise SystemExit(f'{location}: se esperaba un objeto JSON.')

        rerank_results = record.get('rerank_results')
        if not isinstance(rerank_results, list) or not rerank_results:
            raise SystemExit(
                f'{location}: rerank_results debe ser una lista no vacía.'
            )

        record_k_values: set[int] = set()
        for result_index, result in enumerate(rerank_results):
            result_location = f'{location}, rerank_result {result_index}'
            if not isinstance(result, dict):
                raise SystemExit(
                    f'{result_location}: se esperaba un objeto JSON.'
                )

            k = result.get('k')
            if isinstance(k, bool) or not isinstance(k, int) or k < 1:
                raise SystemExit(
                    f'{result_location}: k debe ser un entero positivo.'
                )
            if k in record_k_values:
                raise SystemExit(f'{result_location}: valor k duplicado: {k}.')
            record_k_values.add(k)

            elapsed_seconds = require_non_negative_time(
                result.get('rerank_time_seconds'),
                result_location,
            )
            times_by_k.setdefault(k, []).append(elapsed_seconds)

        if expected_k_values is None:
            expected_k_values = record_k_values
        elif record_k_values != expected_k_values:
            missing = sorted(expected_k_values - record_k_values)
            extra = sorted(record_k_values - expected_k_values)
            raise SystemExit(
                f'{location}: conjunto de k inconsistente; '
                f'faltan {missing} y sobran {extra}.'
            )

    return [
        {
            'k': k,
            'mean_seconds': statistics.fmean(times_by_k[k]),
            'news_count': len(times_by_k[k]),
        }
        for k in sorted(times_by_k)
    ]


def calculate_recall_by_k(
    payload: Any, k_values: list[int]
) -> list[dict[str, int | float]]:
    if not isinstance(payload, dict):
        raise SystemExit('El informe de posiciones debe contener un objeto JSON.')

    summary = payload.get('resumen')
    if not isinstance(summary, dict):
        raise SystemExit('El informe de posiciones no contiene un resumen válido.')

    evaluable_news = summary.get('noticias_evaluables')
    if (
        isinstance(evaluable_news, bool)
        or not isinstance(evaluable_news, int)
        or evaluable_news < 1
    ):
        raise SystemExit('resumen.noticias_evaluables debe ser un entero positivo.')

    position_counts = summary.get('cantidad_por_posicion')
    if not isinstance(position_counts, dict) or not position_counts:
        raise SystemExit(
            'resumen.cantidad_por_posicion debe ser un objeto no vacío.'
        )

    normalized_counts: dict[int, int] = {}
    for raw_position, count in position_counts.items():
        try:
            position = int(raw_position)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f'Posición no válida en cantidad_por_posicion: {raw_position!r}.'
            ) from exc
        if position < 1 or str(position) != str(raw_position):
            raise SystemExit(
                f'Posición no válida en cantidad_por_posicion: {raw_position!r}.'
            )
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise SystemExit(
                f'La cantidad de la posición {position} debe ser un entero '
                'no negativo.'
            )
        normalized_counts[position] = count

    found_news = sum(normalized_counts.values())
    outside_top_19 = summary.get('topic_correcto_fuera_del_top_19')
    if (
        isinstance(outside_top_19, bool)
        or not isinstance(outside_top_19, int)
        or outside_top_19 < 0
    ):
        raise SystemExit(
            'resumen.topic_correcto_fuera_del_top_19 debe ser un entero '
            'no negativo.'
        )
    if found_news + outside_top_19 != evaluable_news:
        raise SystemExit(
            'El resumen de posiciones es inconsistente: los aciertos dentro y '
            'fuera de top_19 no suman las noticias evaluables.'
        )

    max_position = max(normalized_counts)
    if any(k > max_position for k in k_values):
        raise SystemExit(
            f'Los valores de k no pueden superar la posición máxima '
            f'disponible ({max_position}).'
        )

    recall_results: list[dict[str, int | float]] = []
    for k in k_values:
        hits = sum(
            count
            for position, count in normalized_counts.items()
            if position <= k
        )
        recall_results.append(
            {
                'k': k,
                'hits': hits,
                'evaluable_news': evaluable_news,
                'recall': hits / evaluable_news,
            }
        )
    return recall_results


def plot_time_and_recall(
    mean_times: list[dict[str, int | float]],
    recall_results: list[dict[str, int | float]],
    output_path: Path,
) -> list[Path]:
    k_values = [int(result['k']) for result in mean_times]
    recall_k_values = [int(result['k']) for result in recall_results]
    if recall_k_values != k_values:
        raise SystemExit(
            'Los valores de k de los tiempos y del recall no coinciden.'
        )

    x_positions = list(range(len(k_values)))
    averages = [float(result['mean_seconds']) for result in mean_times]
    recall_percentages = [
        float(result['recall']) * 100 for result in recall_results
    ]
    time_color = '#3E5266'
    time_fill_color = '#8293A3'
    recall_color = '#7A3E48'
    axis_color = '#262626'
    paper_style = {
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'],
        'font.size': 7.5,
        'axes.labelsize': 8,
        'axes.linewidth': 0.6,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }

    with plt.rc_context(paper_style):
        figure, axis = plt.subplots(figsize=(3.5, 2.55))
        bars = axis.bar(
            x_positions,
            averages,
            width=0.52,
            color=time_fill_color,
            edgecolor=time_color,
            linewidth=0.5,
        )
        axis.bar_label(
            bars,
            labels=[f'{average:.3f}' for average in averages],
            padding=1.5,
            fontsize=6.5,
            color=time_color,
        )
        axis.set_xlabel('k')
        axis.set_ylabel('Mean reranking time (s)')
        axis.yaxis.label.set_color(axis_color)
        axis.set_xticks(x_positions, [str(k) for k in k_values])
        axis.set_xlim(-0.55, len(x_positions) - 0.45)
        axis.set_ylim(0, max(averages) * 1.18 if any(averages) else 1)
        axis.spines['top'].set_visible(False)
        axis.spines['left'].set_color(axis_color)
        axis.tick_params(
            axis='x', direction='out', length=2.5, width=0.6
        )
        axis.tick_params(
            axis='y',
            colors=axis_color,
            direction='out',
            length=2.5,
            width=0.6,
        )
        axis.yaxis.grid(True, color='#D9D9D9', linestyle=':', linewidth=0.45)
        axis.set_axisbelow(True)

        recall_axis = axis.twinx()
        recall_line = recall_axis.plot(
            x_positions,
            recall_percentages,
            color=recall_color,
            marker='o',
            markerfacecolor='white',
            markeredgecolor=recall_color,
            markeredgewidth=0.7,
            markersize=3.5,
            linewidth=0.9,
            zorder=4,
        )[0]
        for position, recall in zip(x_positions, recall_percentages):
            recall_axis.annotate(
                f'{recall:.1f}%',
                (position, recall),
                xytext=(0, 4),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=6.5,
                color=recall_color,
            )
        recall_axis.set_ylabel('Recall@k (%)')
        recall_axis.yaxis.label.set_color(axis_color)
        recall_axis.set_ylim(0, 106)
        recall_axis.set_yticks([0, 25, 50, 75, 100])
        recall_axis.spines['top'].set_visible(False)
        recall_axis.spines['right'].set_color(axis_color)
        recall_axis.tick_params(
            axis='y',
            colors=axis_color,
            direction='out',
            length=2.5,
            width=0.6,
        )
        axis.legend(
            [bars, recall_line],
            ['Mean time', 'Recall@k'],
            loc='upper left',
            ncol=2,
            fontsize=5,
            frameon=False,
            handlelength=1.0,
            handleheight=0.7,
            handletextpad=0.3,
            columnspacing=0.6,
            borderaxespad=0.2,
            markerscale=0.75,
        )
        figure.tight_layout(pad=0.3)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_paths = [
            output_path.with_suffix('.pdf'),
            output_path.with_suffix('.png'),
        ]
        for figure_path in output_paths:
            figure.savefig(figure_path, dpi=600, bbox_inches='tight')
        plt.close(figure)

    return output_paths


def main() -> None:
    input_path = resolve_path(INPUT_FILE)
    positions_path = resolve_path(POSITIONS_FILE)
    output_path = resolve_path(OUTPUT_FILE)
    payload = load_json(input_path)
    mean_times = calculate_mean_times(payload)
    k_values = [int(result['k']) for result in mean_times]
    recall_results = calculate_recall_by_k(
        load_json(positions_path),
        k_values,
    )
    output_paths = plot_time_and_recall(
        mean_times,
        recall_results,
        output_path,
    )

    for result in mean_times:
        k = result['k']
        mean_seconds = result['mean_seconds']
        news_count = result['news_count']
        print(f'k={k}: {mean_seconds:.6f} s ({news_count} noticias)')
    for result in recall_results:
        k = result['k']
        recall = float(result['recall']) * 100
        hits = result['hits']
        evaluable_news = result['evaluable_news']
        print(f'Recall@{k}: {recall:.2f}% ({hits}/{evaluable_news})')
    for figure_path in output_paths:
        print(f'Gráfica guardada: {figure_path}')


if __name__ == '__main__':
    main()
