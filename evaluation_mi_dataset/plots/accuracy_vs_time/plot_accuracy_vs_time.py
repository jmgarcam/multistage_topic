#!/usr/bin/env python3
"""Grafica el compromiso entre accuracy y tiempo medio por noticia."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


PLOT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = PLOT_DIR.parent.parent
E5_RESULTS = EVALUATION_DIR / 'e5_results' / 'e5_top_n_results.json'
OUTPUT_SVG = PLOT_DIR / 'accuracy_vs_time.svg'
OUTPUT_CSV = PLOT_DIR / 'accuracy_vs_time_data.csv'
OUTPUT_PARETO = PLOT_DIR / 'pareto_frontier.md'
EXPECTED_N = tuple(range(1, 16))


@dataclass(frozen=True)
class ModelSpec:
    label: str
    ollama_model: str
    colour: str
    pipeline_dir: Path
    direct_file: Path


@dataclass(frozen=True)
class ConfigPoint:
    key: str
    model_label: str
    ollama_model: str
    colour: str
    kind: str
    n: int | None
    total_news: int
    hits: int
    e5_seconds_per_news: float
    slm_seconds_per_news: float
    source: Path

    @property
    def accuracy_percent(self) -> float:
        return 100 * self.hits / self.total_news

    @property
    def total_seconds_per_news(self) -> float:
        return self.e5_seconds_per_news + self.slm_seconds_per_news


@dataclass
class ModelAvailability:
    spec: ModelSpec
    pipeline_complete: list[int]
    pipeline_incomplete: dict[int, tuple[int, int]]
    direct_complete: bool
    direct_incomplete: tuple[int, int] | None


def model_result_slug(model: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', model.casefold()).strip('_')


MODEL_SPECS = (
    ModelSpec(
        label='Llama 3.2 · 1B',
        ollama_model='llama3.2:1b',
        colour='#2563eb',
        pipeline_dir=(
            EVALUATION_DIR
            / 'results_e5_llama3_2_1b_two_prompts_no_margin'
        ),
        direct_file=(
            EVALUATION_DIR
            / 'results_llama3_2_1b_two_prompts'
            / 'llama3_2_1b_direct_results.json'
        ),
    ),
    ModelSpec(
        label='Llama 3.2 · 3B',
        ollama_model='llama3.2:3b',
        colour='#ea580c',
        pipeline_dir=(
            EVALUATION_DIR
            / 'results_e5_llama3_2_3b_two_prompts_no_margin'
        ),
        direct_file=(
            EVALUATION_DIR
            / 'results_llama3_2_3b_two_prompts'
            / 'llama3_2_3b_direct_results.json'
        ),
    ),
    ModelSpec(
        label='Llama 3.1 · 8B',
        ollama_model='llama3.1:8b',
        colour='#7c3aed',
        pipeline_dir=(
            EVALUATION_DIR
            / 'results_e5_llama3_1_8b_two_prompts_no_margin'
        ),
        direct_file=(
            EVALUATION_DIR
            / 'results_llama3_1_8b_two_prompts'
            / 'llama3_1_8b_direct_results.json'
        ),
    ),
    ModelSpec(
        label='Qwen 2.5 · 1.5B',
        ollama_model='qwen2.5:1.5b-instruct',
        colour='#0891b2',
        pipeline_dir=(
            EVALUATION_DIR
            / 'results_e5_qwen2_5_1_5b_instruct_two_prompts_no_margin'
        ),
        direct_file=(
            EVALUATION_DIR
            / 'results_qwen2_5_1_5b_instruct_two_prompts'
            / 'qwen2_5_1_5b_instruct_direct_results.json'
        ),
    ),
    ModelSpec(
        label='Qwen 2.5 · 3B',
        ollama_model='qwen2.5:3b-instruct',
        colour='#16a34a',
        pipeline_dir=(
            EVALUATION_DIR
            / 'results_e5_qwen2_5_3b_instruct_two_prompts_no_margin'
        ),
        direct_file=(
            EVALUATION_DIR
            / 'results_qwen2_5_3b_instruct_two_prompts'
            / 'qwen2_5_3b_instruct_direct_results.json'
        ),
    ),
    ModelSpec(
        label='Qwen 2.5 · 7B',
        ollama_model='qwen2.5:7b-instruct',
        colour='#db2777',
        pipeline_dir=(
            EVALUATION_DIR
            / 'results_e5_qwen2_5_7b_instruct_two_prompts_no_margin'
        ),
        direct_file=(
            EVALUATION_DIR
            / 'results_qwen2_5_7b_instruct_two_prompts'
            / 'qwen2_5_7b_instruct_direct_results.json'
        ),
    ),
)


def load_json(path: Path) -> Any:
    try:
        with path.open('r', encoding='utf-8') as file_handle:
            return json.load(file_handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f'JSON no válido en {path}: {exc}') from exc


def number(mapping: dict[str, Any], name: str, path: Path) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f'{path}: falta el valor numérico {name}.')
    return float(value)


def integer(mapping: dict[str, Any], name: str, path: Path) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f'{path}: falta el entero {name}.')
    return value


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


def configured_model(report: dict[str, Any]) -> Any:
    configuration = report.get('configuration')
    if not isinstance(configuration, dict):
        return None
    return configuration.get('model', configuration.get('slm_model'))


def validate_two_prompts(report: dict[str, Any], path: Path) -> None:
    configuration = report.get('configuration')
    if not isinstance(configuration, dict):
        raise SystemExit(f'{path}: falta el bloque configuration.')
    calls = configuration.get(
        'slm_calls_per_valid_routed_prediction',
        configuration.get('slm_calls_per_valid_prediction'),
    )
    if calls != 2:
        raise SystemExit(
            f'{path}: se esperaban dos llamadas por noticia; contiene {calls!r}.'
        )


def load_e5_point() -> ConfigPoint:
    if not E5_RESULTS.is_file():
        raise SystemExit(f'No se encontró el resultado E5: {E5_RESULTS}')
    report = load_json(E5_RESULTS)
    if not isinstance(report, dict) or report.get('status') != 'complete':
        raise SystemExit(f'El resultado E5 no está completo: {E5_RESULTS}')
    dataset = report.get('dataset')
    metrics = report.get('metrics')
    timing = report.get('timing')
    if not all(isinstance(block, dict) for block in (dataset, metrics, timing)):
        raise SystemExit(f'Estructura no válida en {E5_RESULTS}.')
    total = integer(dataset, 'total_news', E5_RESULTS)
    hits = integer(metrics, 'top_1_hits', E5_RESULTS)
    e5_total = number(timing, 'total_e5_execution_time_seconds', E5_RESULTS)
    return ConfigPoint(
        key='e5_only',
        model_label='E5',
        ollama_model='intfloat/multilingual-e5-base',
        colour='#334155',
        kind='e5_only',
        n=None,
        total_news=total,
        hits=hits,
        e5_seconds_per_news=e5_total / total,
        slm_seconds_per_news=0.0,
        source=E5_RESULTS,
    )


def load_pipeline_point(
    path: Path,
    spec: ModelSpec,
    n: int,
    expected_total: int,
) -> ConfigPoint:
    report = load_json(path)
    if not isinstance(report, dict):
        raise SystemExit(f'{path}: se esperaba un objeto JSON.')
    if configured_model(report) != spec.ollama_model:
        raise SystemExit(
            f'{path}: modelo {configured_model(report)!r}; se esperaba '
            f'{spec.ollama_model!r}.'
        )
    validate_two_prompts(report, path)
    completed, total = progress(report, path)
    if total != expected_total or completed != total or report.get('status') != 'complete':
        raise SystemExit(f'{path}: el resultado no es completo o comparable.')
    configuration = report.get('configuration')
    metrics = report.get('metrics')
    timing = report.get('timing')
    if not all(isinstance(block, dict) for block in (configuration, metrics, timing)):
        raise SystemExit(f'{path}: estructura no válida.')
    if configuration.get('n') != n:
        raise SystemExit(f'{path}: configuration.n no coincide con N={n}.')
    e5_timing = timing.get('e5_ranking')
    if not isinstance(e5_timing, dict):
        raise SystemExit(f'{path}: falta timing.e5_ranking.')
    e5_total = number(e5_timing, 'total_e5_execution_time_seconds', path)
    slm_total = number(timing, 'slm_total_inference_time_seconds', path)
    hits = integer(metrics, 'final_hits', path)
    return ConfigPoint(
        key=f'{spec.ollama_model}_pipeline_n{n}',
        model_label=spec.label,
        ollama_model=spec.ollama_model,
        colour=spec.colour,
        kind='pipeline',
        n=n,
        total_news=total,
        hits=hits,
        e5_seconds_per_news=e5_total / total,
        slm_seconds_per_news=slm_total / total,
        source=path,
    )


def load_direct_point(
    path: Path, spec: ModelSpec, expected_total: int
) -> ConfigPoint:
    report = load_json(path)
    if not isinstance(report, dict):
        raise SystemExit(f'{path}: se esperaba un objeto JSON.')
    if configured_model(report) != spec.ollama_model:
        raise SystemExit(
            f'{path}: modelo {configured_model(report)!r}; se esperaba '
            f'{spec.ollama_model!r}.'
        )
    validate_two_prompts(report, path)
    completed, total = progress(report, path)
    if total != expected_total or completed != total or report.get('status') != 'complete':
        raise SystemExit(f'{path}: el resultado no es completo o comparable.')
    metrics = report.get('metrics')
    timing = report.get('timing')
    if not isinstance(metrics, dict) or not isinstance(timing, dict):
        raise SystemExit(f'{path}: estructura no válida.')
    hits = integer(metrics, 'slm_final_hits', path)
    slm_total = number(timing, 'total_slm_inference_time_seconds', path)
    return ConfigPoint(
        key=f'{spec.ollama_model}_direct',
        model_label=spec.label,
        ollama_model=spec.ollama_model,
        colour=spec.colour,
        kind='slm_direct',
        n=None,
        total_news=total,
        hits=hits,
        e5_seconds_per_news=0.0,
        slm_seconds_per_news=slm_total / total,
        source=path,
    )


def load_all_points() -> tuple[list[ConfigPoint], list[ModelAvailability]]:
    e5_point = load_e5_point()
    points = [e5_point]
    availability: list[ModelAvailability] = []

    for spec in MODEL_SPECS:
        complete: list[int] = []
        incomplete: dict[int, tuple[int, int]] = {}
        for n in EXPECTED_N:
            result_slug = model_result_slug(spec.ollama_model)
            path = spec.pipeline_dir / f'e5_top_{n}_{result_slug}_results.json'
            if not path.is_file():
                continue
            report = load_json(path)
            if not isinstance(report, dict):
                raise SystemExit(f'{path}: se esperaba un objeto JSON.')
            completed, expected = progress(report, path)
            if report.get('status') == 'complete' and completed == expected:
                points.append(
                    load_pipeline_point(path, spec, n, e5_point.total_news)
                )
                complete.append(n)
            else:
                incomplete[n] = (completed, expected)

        direct_complete = False
        direct_incomplete: tuple[int, int] | None = None
        if spec.direct_file.is_file():
            report = load_json(spec.direct_file)
            if not isinstance(report, dict):
                raise SystemExit(f'{spec.direct_file}: objeto JSON esperado.')
            completed, expected = progress(report, spec.direct_file)
            if report.get('status') == 'complete' and completed == expected:
                points.append(
                    load_direct_point(
                        spec.direct_file, spec, e5_point.total_news
                    )
                )
                direct_complete = True
            else:
                direct_incomplete = (completed, expected)

        availability.append(
            ModelAvailability(
                spec=spec,
                pipeline_complete=complete,
                pipeline_incomplete=incomplete,
                direct_complete=direct_complete,
                direct_incomplete=direct_incomplete,
            )
        )
    return points, availability


def dominates(candidate: ConfigPoint, point: ConfigPoint) -> bool:
    no_slower = candidate.total_seconds_per_news <= point.total_seconds_per_news
    no_less_accurate = candidate.hits / candidate.total_news >= point.hits / point.total_news
    strictly_better = (
        candidate.total_seconds_per_news < point.total_seconds_per_news
        or candidate.hits / candidate.total_news > point.hits / point.total_news
    )
    return no_slower and no_less_accurate and strictly_better


def pareto_keys(points: list[ConfigPoint]) -> set[str]:
    return {
        point.key
        for point in points
        if not any(
            other.key != point.key and dominates(other, point)
            for other in points
        )
    }


def write_csv(points: list[ConfigPoint], pareto: set[str]) -> None:
    with OUTPUT_CSV.open('w', encoding='utf-8', newline='') as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=(
                'configuration',
                'model',
                'ollama_model',
                'type',
                'n',
                'total_news',
                'hits',
                'accuracy_percent',
                'e5_seconds_per_news',
                'slm_two_prompts_seconds_per_news',
                'total_seconds_per_news',
                'pareto_nondominated',
                'source',
            ),
        )
        writer.writeheader()
        for point in sorted(
            points, key=lambda item: item.total_seconds_per_news
        ):
            writer.writerow(
                {
                    'configuration': point.key,
                    'model': point.model_label,
                    'ollama_model': point.ollama_model,
                    'type': point.kind,
                    'n': point.n if point.n is not None else '',
                    'total_news': point.total_news,
                    'hits': point.hits,
                    'accuracy_percent': round(point.accuracy_percent, 6),
                    'e5_seconds_per_news': round(
                        point.e5_seconds_per_news, 9
                    ),
                    'slm_two_prompts_seconds_per_news': round(
                        point.slm_seconds_per_news, 9
                    ),
                    'total_seconds_per_news': round(
                        point.total_seconds_per_news, 9
                    ),
                    'pareto_nondominated': point.key in pareto,
                    'source': point.source,
                }
            )


def point_description(point: ConfigPoint) -> str:
    if point.kind == 'e5_only':
        return 'E5 solo'
    if point.kind == 'slm_direct':
        return f'{point.model_label} directo'
    return f'{point.model_label}, N={point.n}'


def write_pareto_table(points: list[ConfigPoint], pareto: set[str]) -> None:
    frontier = sorted(
        (point for point in points if point.key in pareto),
        key=lambda point: point.total_seconds_per_news,
    )
    lines = [
        '# Configuraciones no dominadas',
        '',
        'Una configuración es no dominada cuando ninguna otra alcanza una '
        'accuracy igual o superior con un tiempo igual o inferior, siendo al '
        'menos una de esas dos comparaciones estrictamente mejor.',
        '',
        '| Configuración | Accuracy | Tiempo medio total/noticia |',
        '|---|---:|---:|',
    ]
    for point in frontier:
        lines.append(
            f'| {point_description(point)} | {point.accuracy_percent:.2f}% | '
            f'{point.total_seconds_per_news:.4f} s |'
        )
    OUTPUT_PARETO.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def nice_time_axis(maximum: float) -> tuple[float, float]:
    target_step = maximum * 1.08 / 6
    candidates = (0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5)
    step = next((value for value in candidates if value >= target_step), candidates[-1])
    limit = math.ceil(maximum * 1.08 / step) * step
    return limit, step


def x_position(value: float, left: float, width: float, maximum: float) -> float:
    return left + value * width / maximum


def y_position(value: float, top: float, height: float) -> float:
    return top + (100 - value) * height / 100


def add_shape(
    svg: list[str], point: ConfigPoint, x: float, y: float, size: float = 6.5
) -> None:
    if point.kind == 'pipeline':
        svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" '
            f'fill="{point.colour}" stroke="#ffffff" stroke-width="1.5"/>'
        )
    elif point.kind == 'slm_direct':
        vertices = (
            f'{x:.1f},{y - size - 1:.1f} '
            f'{x - size - 1:.1f},{y + size:.1f} '
            f'{x + size + 1:.1f},{y + size:.1f}'
        )
        svg.append(
            f'<polygon points="{vertices}" fill="{point.colour}" '
            'stroke="#ffffff" stroke-width="1.5"/>'
        )
    else:
        svg.append(
            f'<rect x="{x - size:.1f}" y="{y - size:.1f}" '
            f'width="{2 * size:.1f}" height="{2 * size:.1f}" '
            f'rx="1.5" fill="{point.colour}" stroke="#ffffff" '
            'stroke-width="1.5"/>'
        )


def relevant_labels(
    points: list[ConfigPoint], pareto: set[str]
) -> set[str]:
    labels = set(pareto)
    labels.update(
        point.key for point in points if point.kind in {'e5_only', 'slm_direct'}
    )
    for spec in MODEL_SPECS:
        pipelines = [
            point
            for point in points
            if point.ollama_model == spec.ollama_model
            and point.kind == 'pipeline'
        ]
        if not pipelines:
            continue
        labels.add(min(pipelines, key=lambda point: point.n or 0).key)
        labels.add(max(pipelines, key=lambda point: point.n or 0).key)
        labels.add(
            max(
                pipelines,
                key=lambda point: (
                    point.accuracy_percent,
                    -point.total_seconds_per_news,
                ),
            ).key
        )
    return labels


def availability_summary(item: ModelAvailability) -> str:
    values = item.pipeline_complete
    if values == list(EXPECTED_N):
        pipeline = 'pipeline N=1–15'
    elif values and values == list(range(1, max(values) + 1)):
        pipeline = f'pipeline N=1–{max(values)}'
    elif values:
        pipeline = 'pipeline N=' + ','.join(map(str, values))
    else:
        pipeline = 'sin pipeline completo'
    direct = 'directo disponible' if item.direct_complete else 'sin directo'
    return f'{item.spec.label}: {pipeline}, {direct}'


def generate_svg(
    points: list[ConfigPoint],
    pareto: set[str],
    availability: list[ModelAvailability],
) -> str:
    width = 1560
    height = 900
    plot_left = 135
    plot_top = 205
    plot_width = 1345
    plot_height = 535
    plot_bottom = plot_top + plot_height
    max_time, time_step = nice_time_axis(
        max(point.total_seconds_per_news for point in points)
    )

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title description">'
        ),
        '<title id="title">Accuracy frente a tiempo medio por noticia</title>',
        (
            '<desc id="description">Diagrama de dispersión de accuracy frente '
            'al tiempo medio total por noticia. Se destacan las configuraciones '
            'no dominadas.</desc>'
        ),
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        (
            '<text x="48" y="45" fill="#0f172a" '
            'font-family="Arial, Helvetica, sans-serif" font-size="32" '
            'font-weight="700">Accuracy frente a tiempo: ¿qué configuración '
            'compensa?</text>'
        ),
        (
            '<text x="48" y="75" fill="#475569" '
            'font-family="Arial, Helvetica, sans-serif" font-size="17">'
            'Tiempo amortizado por noticia; el pipeline incluye E5 y las dos '
            'llamadas al SLM</text>'
        ),
    ]

    colour_legend = [
        ('#334155', 'E5'),
        *[(spec.colour, spec.label) for spec in MODEL_SPECS],
    ]
    for index, (colour, label) in enumerate(colour_legend):
        legend_row, legend_column = divmod(index, 4)
        x = 135 + legend_column * 330
        y = 108 + legend_row * 28
        svg.append(
            f'<circle cx="{x}" cy="{y}" r="6" fill="{colour}"/>'
        )
        svg.append(
            f'<text x="{x + 13}" y="{y + 6}" fill="#334155" '
            'font-family="Arial, Helvetica, sans-serif" font-size="15">'
            f'{escape(label)}</text>'
        )

    shape_y = 170
    svg.extend(
        [
            '<rect x="135" y="163" width="13" height="13" rx="1" fill="#64748b"/>',
            '<text x="157" y="175" fill="#334155" font-family="Arial, Helvetica, sans-serif" font-size="14">E5 solo</text>',
            '<polygon points="335,162 327,176 343,176" fill="#64748b"/>',
            '<text x="352" y="175" fill="#334155" font-family="Arial, Helvetica, sans-serif" font-size="14">SLM directo</text>',
            f'<circle cx="558" cy="{shape_y}" r="7" fill="#64748b"/>',
            '<text x="574" y="175" fill="#334155" font-family="Arial, Helvetica, sans-serif" font-size="14">E5 → SLM</text>',
            '<circle cx="746" cy="170" r="10" fill="none" stroke="#ca8a04" stroke-width="3"/>',
            '<text x="764" y="175" fill="#334155" font-family="Arial, Helvetica, sans-serif" font-size="14">No dominada</text>',
        ]
    )

    svg.append(
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" '
        f'height="{plot_height}" fill="#ffffff" stroke="#cbd5e1"/>'
    )
    for accuracy in range(0, 101, 10):
        y = y_position(accuracy, plot_top, plot_height)
        svg.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" '
            f'x2="{plot_left + plot_width}" y2="{y:.1f}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{plot_left - 13}" y="{y + 5:.1f}" '
            'text-anchor="end" fill="#64748b" '
            'font-family="Arial, Helvetica, sans-serif" '
            f'font-size="13">{accuracy}</text>'
        )

    tick_count = int(round(max_time / time_step))
    for index in range(tick_count + 1):
        value = index * time_step
        x = x_position(value, plot_left, plot_width, max_time)
        svg.append(
            f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" '
            f'y2="{plot_bottom}" stroke="#eef2f7" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 23}" '
            'text-anchor="middle" fill="#64748b" '
            'font-family="Arial, Helvetica, sans-serif" font-size="13">'
            f'{value:.2f}</text>'
        )

    svg.extend(
        [
            (
                f'<text x="{plot_left + plot_width / 2:.1f}" '
                f'y="{plot_bottom + 57}" text-anchor="middle" fill="#334155" '
                'font-family="Arial, Helvetica, sans-serif" font-size="16">'
                'Tiempo medio total por noticia (s)</text>'
            ),
            (
                f'<text x="29" y="{plot_top + plot_height / 2:.1f}" '
                f'transform="rotate(-90 29 {plot_top + plot_height / 2:.1f})" '
                'text-anchor="middle" fill="#334155" '
                'font-family="Arial, Helvetica, sans-serif" font-size="16">'
                'Accuracy (%)</text>'
            ),
            (
                f'<text x="{plot_left + 20}" y="{plot_top + 28}" '
                'fill="#15803d" font-family="Arial, Helvetica, sans-serif" '
                'font-size="15" font-weight="700">↖ Zona deseable</text>'
            ),
        ]
    )

    frontier = sorted(
        (point for point in points if point.key in pareto),
        key=lambda point: point.total_seconds_per_news,
    )
    if len(frontier) > 1:
        frontier_points = ' '.join(
            f'{x_position(point.total_seconds_per_news, plot_left, plot_width, max_time):.1f},'
            f'{y_position(point.accuracy_percent, plot_top, plot_height):.1f}'
            for point in frontier
        )
        svg.append(
            f'<polyline points="{frontier_points}" fill="none" '
            'stroke="#ca8a04" stroke-width="2.2" stroke-dasharray="7 6" '
            'opacity="0.8"/>'
        )

    labels = relevant_labels(points, pareto)
    for point in points:
        x = x_position(
            point.total_seconds_per_news, plot_left, plot_width, max_time
        )
        y = y_position(point.accuracy_percent, plot_top, plot_height)
        add_shape(svg, point, x, y)
        if point.key in pareto:
            svg.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10.5" fill="none" '
                'stroke="#ca8a04" stroke-width="2.8"/>'
            )
        if point.key not in labels:
            continue
        if point.kind == 'e5_only':
            label = 'E5 solo'
            dx, dy, anchor = 12, -11, 'start'
        elif point.kind == 'slm_direct':
            label = 'Directo'
            dx, dy, anchor = -12, 23, 'end'
        else:
            label = f'N={point.n}'
            dx, dy, anchor = 10, -10, 'start'
        svg.append(
            f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" '
            f'text-anchor="{anchor}" fill="#1e293b" '
            'font-family="Arial, Helvetica, sans-serif" font-size="13" '
            f'font-weight="600">{escape(label)}</text>'
        )

    summaries = [availability_summary(item) for item in availability]
    for line_index, start in enumerate(range(0, len(summaries), 3)):
        line = ' · '.join(summaries[start : start + 3])
        prefix = 'Solo resultados completos · ' if line_index == 0 else ''
        svg.append(
            f'<text x="48" y="{862 + line_index * 19}" fill="#64748b" '
            'font-family="Arial, Helvetica, sans-serif" font-size="12">'
            f'{escape(prefix + line)}</text>'
        )
    svg.append('</svg>')
    return '\n'.join(svg) + '\n'


def main() -> None:
    points, availability = load_all_points()
    pareto = pareto_keys(points)
    write_csv(points, pareto)
    write_pareto_table(points, pareto)
    OUTPUT_SVG.write_text(
        generate_svg(points, pareto, availability), encoding='utf-8'
    )

    print(f'Configuraciones completas: {len(points)}')
    print('No dominadas:')
    for point in sorted(
        (item for item in points if item.key in pareto),
        key=lambda item: item.total_seconds_per_news,
    ):
        print(
            f'  {point_description(point)}: {point.accuracy_percent:.2f}% · '
            f'{point.total_seconds_per_news:.4f} s/noticia'
        )
    print(f'Gráfica: {OUTPUT_SVG.resolve()}')
    print(f'Datos: {OUTPUT_CSV.resolve()}')
    print(f'Frontera: {OUTPUT_PARETO.resolve()}')


if __name__ == '__main__':
    main()
