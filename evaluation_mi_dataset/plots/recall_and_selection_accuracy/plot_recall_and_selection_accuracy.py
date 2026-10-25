#!/usr/bin/env python3
"""Compara recall@N, accuracy final y accuracy de selección del pipeline."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


PLOT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = PLOT_DIR.parent.parent
E5_RESULTS = EVALUATION_DIR / 'e5_results' / 'e5_top_n_results.json'
OUTPUT_SVG = PLOT_DIR / 'recall_and_selection_accuracy.svg'
OUTPUT_CSV = PLOT_DIR / 'recall_and_selection_accuracy_data.csv'
EXPECTED_N = tuple(range(1, 16))


@dataclass(frozen=True)
class ModelSpec:
    label: str
    ollama_model: str
    colour: str
    results_dir: Path


@dataclass(frozen=True)
class Point:
    final_hits: int
    candidate_hits: int
    total_news: int

    @property
    def final_accuracy_percent(self) -> float:
        return 100 * self.final_hits / self.total_news

    @property
    def selection_accuracy_percent(self) -> float:
        return 100 * self.final_hits / self.candidate_hits


@dataclass
class ModelData:
    spec: ModelSpec
    points: dict[int, Point]
    incomplete: dict[int, tuple[int, int]]
    missing: list[int]


def model_result_slug(model: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', model.casefold()).strip('_')


MODEL_SPECS = (
    ModelSpec(
        label='Llama 3.2 · 1B',
        ollama_model='llama3.2:1b',
        colour='#2563eb',
        results_dir=(
            EVALUATION_DIR
            / 'results_e5_llama3_2_1b_two_prompts_no_margin'
        ),
    ),
    ModelSpec(
        label='Llama 3.2 · 3B',
        ollama_model='llama3.2:3b',
        colour='#ea580c',
        results_dir=(
            EVALUATION_DIR
            / 'results_e5_llama3_2_3b_two_prompts_no_margin'
        ),
    ),
    ModelSpec(
        label='Llama 3.1 · 8B',
        ollama_model='llama3.1:8b',
        colour='#7c3aed',
        results_dir=(
            EVALUATION_DIR
            / 'results_e5_llama3_1_8b_two_prompts_no_margin'
        ),
    ),
    ModelSpec(
        label='Qwen 2.5 · 1.5B',
        ollama_model='qwen2.5:1.5b-instruct',
        colour='#0891b2',
        results_dir=(
            EVALUATION_DIR
            / 'results_e5_qwen2_5_1_5b_instruct_two_prompts_no_margin'
        ),
    ),
    ModelSpec(
        label='Qwen 2.5 · 3B',
        ollama_model='qwen2.5:3b-instruct',
        colour='#16a34a',
        results_dir=(
            EVALUATION_DIR
            / 'results_e5_qwen2_5_3b_instruct_two_prompts_no_margin'
        ),
    ),
    ModelSpec(
        label='Qwen 2.5 · 7B',
        ollama_model='qwen2.5:7b-instruct',
        colour='#db2777',
        results_dir=(
            EVALUATION_DIR
            / 'results_e5_qwen2_5_7b_instruct_two_prompts_no_margin'
        ),
    ),
)


def load_json(path: Path) -> Any:
    try:
        with path.open('r', encoding='utf-8') as file_handle:
            return json.load(file_handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f'JSON no válido en {path}: {exc}') from exc


def integer_metric(metrics: dict[str, Any], name: str, path: Path) -> int:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f'{path}: falta la métrica entera {name}.')
    return value


def load_e5() -> tuple[int, dict[int, tuple[int, float]]]:
    if not E5_RESULTS.is_file():
        raise SystemExit(f'No se encontró el resultado E5: {E5_RESULTS}')
    report = load_json(E5_RESULTS)
    if not isinstance(report, dict) or report.get('status') != 'complete':
        raise SystemExit(f'El resultado E5 no está completo: {E5_RESULTS}')

    dataset = report.get('dataset')
    metrics = report.get('metrics')
    total = dataset.get('total_news') if isinstance(dataset, dict) else None
    recall_block = (
        metrics.get('candidate_recall_at_n')
        if isinstance(metrics, dict)
        else None
    )
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or not isinstance(recall_block, dict)
    ):
        raise SystemExit(f'Estructura E5 no válida en {E5_RESULTS}.')

    recall: dict[int, tuple[int, float]] = {}
    for n in EXPECTED_N:
        values = recall_block.get(str(n), recall_block.get(n))
        if not isinstance(values, dict):
            raise SystemExit(f'{E5_RESULTS}: falta Recall@{n}.')
        hits = values.get('hits')
        if isinstance(hits, bool) or not isinstance(hits, int):
            raise SystemExit(f'{E5_RESULTS}: hits no válidos para Recall@{n}.')
        recall[n] = (hits, 100 * hits / total)
    return total, recall


def validate_model(report: dict[str, Any], spec: ModelSpec, path: Path) -> None:
    configuration = report.get('configuration')
    model = None
    if isinstance(configuration, dict):
        model = configuration.get('model', configuration.get('slm_model'))
    if model != spec.ollama_model:
        raise SystemExit(
            f'{path}: modelo {model!r}; se esperaba {spec.ollama_model!r}.'
        )


def progress(report: dict[str, Any], path: Path) -> tuple[int, int]:
    dataset = report.get('dataset')
    if not isinstance(dataset, dict):
        raise SystemExit(f'{path}: falta el bloque dataset.')
    completed = dataset.get('completed_predictions')
    expected = dataset.get('expected_predictions')
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or isinstance(expected, bool)
        or not isinstance(expected, int)
    ):
        raise SystemExit(f'{path}: progreso no válido.')
    return completed, expected


def load_model_data(
    spec: ModelSpec,
    total_news: int,
    e5_recall: dict[int, tuple[int, float]],
) -> ModelData:
    points: dict[int, Point] = {}
    incomplete: dict[int, tuple[int, int]] = {}
    missing: list[int] = []

    for n in EXPECTED_N:
        result_slug = model_result_slug(spec.ollama_model)
        path = spec.results_dir / f'e5_top_{n}_{result_slug}_results.json'
        if not path.is_file():
            missing.append(n)
            continue
        report = load_json(path)
        if not isinstance(report, dict):
            raise SystemExit(f'{path}: se esperaba un objeto JSON.')
        validate_model(report, spec, path)
        completed, expected = progress(report, path)
        if expected != total_news:
            raise SystemExit(
                f'{path}: contiene {expected} noticias; se esperaban '
                f'{total_news}.'
            )
        if report.get('status') != 'complete' or completed != expected:
            incomplete[n] = (completed, expected)
            continue

        metrics = report.get('metrics')
        if not isinstance(metrics, dict):
            raise SystemExit(f'{path}: falta el bloque metrics.')
        final_hits = integer_metric(metrics, 'final_hits', path)
        candidate_hits = integer_metric(
            metrics, 'candidate_recall_at_n_hits', path
        )
        expected_candidate_hits = e5_recall[n][0]
        if candidate_hits != expected_candidate_hits:
            raise SystemExit(
                f'{path}: candidate_recall_at_n_hits={candidate_hits}; '
                f'E5 contiene {expected_candidate_hits} para N={n}.'
            )
        if not 0 <= final_hits <= candidate_hits <= total_news:
            raise SystemExit(
                f'{path}: conteos incoherentes '
                f'({final_hits=}, {candidate_hits=}, {total_news=}).'
            )
        points[n] = Point(final_hits, candidate_hits, total_news)

    return ModelData(spec, points, incomplete, missing)


def availability_text(data: ModelData) -> str:
    complete = sorted(data.points)
    if complete == list(EXPECTED_N):
        return f'{data.spec.label}: N=1–15'
    if not complete:
        return f'{data.spec.label}: sin resultados completos'
    if complete == list(range(1, max(complete) + 1)):
        description = f'N=1–{max(complete)}'
    else:
        description = 'N=' + ','.join(map(str, complete))
    if data.incomplete:
        n = min(data.incomplete)
        completed, expected = data.incomplete[n]
        description += f'; N={n} en curso ({completed}/{expected})'
    return f'{data.spec.label}: {description}'


def write_csv(
    total_news: int,
    e5_recall: dict[int, tuple[int, float]],
    model_data: list[ModelData],
) -> None:
    with OUTPUT_CSV.open('w', encoding='utf-8', newline='') as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=(
                'model',
                'ollama_model',
                'n',
                'e5_recall_hits',
                'e5_recall_at_n_percent',
                'final_hits',
                'final_accuracy_percent',
                'selection_denominator',
                'conditional_selection_accuracy_percent',
                'status',
                'evaluated_news',
            ),
        )
        writer.writeheader()
        for data in model_data:
            for n in EXPECTED_N:
                point = data.points.get(n)
                if point is not None:
                    status = 'complete'
                    evaluated: int | str = total_news
                elif n in data.incomplete:
                    status = 'in_progress'
                    evaluated = data.incomplete[n][0]
                else:
                    status = 'missing'
                    evaluated = ''
                writer.writerow(
                    {
                        'model': data.spec.label,
                        'ollama_model': data.spec.ollama_model,
                        'n': n,
                        'e5_recall_hits': e5_recall[n][0],
                        'e5_recall_at_n_percent': round(e5_recall[n][1], 6),
                        'final_hits': point.final_hits if point else '',
                        'final_accuracy_percent': (
                            round(point.final_accuracy_percent, 6)
                            if point
                            else ''
                        ),
                        'selection_denominator': (
                            point.candidate_hits if point else ''
                        ),
                        'conditional_selection_accuracy_percent': (
                            round(point.selection_accuracy_percent, 6)
                            if point
                            else ''
                        ),
                        'status': status,
                        'evaluated_news': evaluated,
                    }
                )


def x_position(n: int, left: float, width: float) -> float:
    return left + (n - 1) * width / (len(EXPECTED_N) - 1)


def y_position(value: float, top: float, height: float) -> float:
    return top + (100 - value) * height / 100


def consecutive_segments(values: dict[int, float]) -> list[list[tuple[int, float]]]:
    segments: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    previous_n: int | None = None
    for n, value in sorted(values.items()):
        if previous_n is not None and n != previous_n + 1:
            segments.append(current)
            current = []
        current.append((n, value))
        previous_n = n
    if current:
        segments.append(current)
    return segments


def add_series(
    svg: list[str],
    values: dict[int, float],
    colour: str,
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    dashed: bool = False,
) -> None:
    dash = ' stroke-dasharray="10 7"' if dashed else ''
    for segment in consecutive_segments(values):
        points = ' '.join(
            f'{x_position(n, left, width):.1f},'
            f'{y_position(value, top, height):.1f}'
            for n, value in segment
        )
        if len(segment) > 1:
            svg.append(
                f'<polyline points="{points}" fill="none" '
                f'stroke="{colour}" stroke-width="3.4"{dash} '
                'stroke-linejoin="round" stroke-linecap="round"/>'
            )
        for n, value in segment:
            svg.append(
                f'<circle cx="{x_position(n, left, width):.1f}" '
                f'cy="{y_position(value, top, height):.1f}" r="4.1" '
                f'fill="#ffffff" stroke="{colour}" stroke-width="2.5"/>'
            )


def add_panel(
    svg: list[str],
    title: str,
    top: float,
    plot_left: float,
    plot_width: float,
    plot_height: float,
    *,
    show_x_label: bool,
) -> None:
    panel_x = 44
    panel_y = top - 48
    panel_width = 1472
    panel_height = plot_height + (112 if show_x_label else 86)
    bottom = top + plot_height
    svg.append(
        f'<rect x="{panel_x}" y="{panel_y:.1f}" width="{panel_width}" '
        f'height="{panel_height:.1f}" rx="13" fill="#ffffff" '
        'stroke="#dbe3ee"/>'
    )
    svg.append(
        f'<text x="{plot_left:.1f}" y="{top - 18:.1f}" fill="#0f172a" '
        'font-family="Arial, Helvetica, sans-serif" font-size="21" '
        f'font-weight="700">{escape(title)}</text>'
    )

    for tick in range(0, 101, 20):
        y = y_position(tick, top, plot_height)
        svg.append(
            f'<line x1="{plot_left:.1f}" y1="{y:.1f}" '
            f'x2="{plot_left + plot_width:.1f}" y2="{y:.1f}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{plot_left - 13:.1f}" y="{y + 5:.1f}" '
            'text-anchor="end" fill="#64748b" '
            'font-family="Arial, Helvetica, sans-serif" '
            f'font-size="13">{tick}</text>'
        )

    for n in EXPECTED_N:
        x = x_position(n, plot_left, plot_width)
        svg.append(
            f'<line x1="{x:.1f}" y1="{bottom:.1f}" x2="{x:.1f}" '
            f'y2="{bottom + 5:.1f}" stroke="#64748b"/>'
        )
        svg.append(
            f'<text x="{x:.1f}" y="{bottom + 22:.1f}" '
            'text-anchor="middle" fill="#475569" '
            'font-family="Arial, Helvetica, sans-serif" '
            f'font-size="12">{n}</text>'
        )

    svg.extend(
        [
            (
                f'<line x1="{plot_left:.1f}" y1="{bottom:.1f}" '
                f'x2="{plot_left + plot_width:.1f}" y2="{bottom:.1f}" '
                'stroke="#64748b"/>'
            ),
            (
                f'<line x1="{plot_left:.1f}" y1="{top:.1f}" '
                f'x2="{plot_left:.1f}" y2="{bottom:.1f}" '
                'stroke="#64748b"/>'
            ),
            (
                f'<text x="72" y="{top + plot_height / 2:.1f}" '
                f'transform="rotate(-90 72 {top + plot_height / 2:.1f})" '
                'text-anchor="middle" fill="#334155" '
                'font-family="Arial, Helvetica, sans-serif" '
                'font-size="16">Porcentaje (%)</text>'
            ),
        ]
    )
    if show_x_label:
        svg.append(
            f'<text x="{plot_left + plot_width / 2:.1f}" '
            f'y="{bottom + 53:.1f}" text-anchor="middle" fill="#334155" '
            'font-family="Arial, Helvetica, sans-serif" font-size="16">'
            'Número de candidatos (N)</text>'
        )


def generate_svg(
    total_news: int,
    e5_recall: dict[int, tuple[int, float]],
    model_data: list[ModelData],
) -> str:
    width = 1560
    height = 1195
    plot_left = 132
    plot_width = 1320
    plot_height = 310
    upper_top = 220
    lower_top = 685

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title description">'
        ),
        '<title id="title">Recall, accuracy final y accuracy de selección</title>',
        (
            '<desc id="description">Dos paneles alineados muestran Recall@N '
            'de E5, accuracy final del pipeline y accuracy del SLM condicionada '
            'a que una etiqueta correcta esté entre los candidatos.</desc>'
        ),
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        (
            '<text x="48" y="45" fill="#0f172a" '
            'font-family="Arial, Helvetica, sans-serif" font-size="31" '
            'font-weight="700">Cobertura de E5 y capacidad de selección del SLM</text>'
        ),
        (
            f'<text x="48" y="76" fill="#475569" '
            'font-family="Arial, Helvetica, sans-serif" font-size="17">'
            f'Dataset: {total_news} noticias · A_selección = aciertos finales / '
            'noticias con etiqueta correcta en Top-N</text>'
        ),
    ]

    legend_items = [('#0f172a', '10 7', 'Recall@N de E5')]
    legend_items.extend(
        (data.spec.colour, '', data.spec.label) for data in model_data
    )
    for index, (colour, dash, label) in enumerate(legend_items):
        legend_row, legend_column = divmod(index, 4)
        x = 132 + legend_column * 350
        y = 108 + legend_row * 30
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ''
        svg.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 48}" y2="{y}" '
            f'stroke="{colour}" stroke-width="3.4"{dash_attribute}/>'
        )
        svg.append(
            f'<text x="{x + 60}" y="{y + 6}" fill="#334155" '
            'font-family="Arial, Helvetica, sans-serif" font-size="16">'
            f'{escape(label)}</text>'
        )

    add_panel(
        svg,
        'Superior · Recall@N de E5 y accuracy final del pipeline',
        upper_top,
        plot_left,
        plot_width,
        plot_height,
        show_x_label=False,
    )
    add_panel(
        svg,
        'Inferior · Accuracy de selección condicionada a gold ∈ Top-N',
        lower_top,
        plot_left,
        plot_width,
        plot_height,
        show_x_label=True,
    )

    add_series(
        svg,
        {n: values[1] for n, values in e5_recall.items()},
        '#0f172a',
        plot_left,
        upper_top,
        plot_width,
        plot_height,
        dashed=True,
    )
    for data in model_data:
        add_series(
            svg,
            {
                n: point.final_accuracy_percent
                for n, point in data.points.items()
            },
            data.spec.colour,
            plot_left,
            upper_top,
            plot_width,
            plot_height,
        )
        add_series(
            svg,
            {
                n: point.selection_accuracy_percent
                for n, point in data.points.items()
            },
            data.spec.colour,
            plot_left,
            lower_top,
            plot_width,
            plot_height,
        )

    availability = [availability_text(data) for data in model_data]
    for line_index, start in enumerate(range(0, len(availability), 3)):
        line = ' · '.join(availability[start : start + 3])
        prefix = 'Solo ejecuciones completas · ' if line_index == 0 else ''
        svg.append(
            f'<text x="48" y="{1157 + line_index * 19}" fill="#64748b" '
            'font-family="Arial, Helvetica, sans-serif" font-size="13">'
            f'{escape(prefix + line)}</text>'
        )
    svg.append('</svg>')
    return '\n'.join(svg) + '\n'


def main() -> None:
    total_news, e5_recall = load_e5()
    model_data = [
        load_model_data(spec, total_news, e5_recall) for spec in MODEL_SPECS
    ]
    write_csv(total_news, e5_recall, model_data)
    OUTPUT_SVG.write_text(
        generate_svg(total_news, e5_recall, model_data),
        encoding='utf-8',
    )

    for data in model_data:
        complete = ', '.join(map(str, sorted(data.points))) or 'ninguno'
        print(f'{data.spec.label}: N completos = {complete}')
    print(f'Gráfica: {OUTPUT_SVG.resolve()}')
    print(f'Datos: {OUTPUT_CSV.resolve()}')


if __name__ == '__main__':
    main()
