#!/usr/bin/env python3
"""Resume accuracy y tiempo de cinco iteraciones para todos los modelos."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape


PLOT_DIR = Path(__file__).resolve().parent
ITERATIONS_ROOT = PLOT_DIR.parent.parent
OUTPUT_SVG = PLOT_DIR / 'accuracy_and_time_statistics.svg'
OUTPUT_DETAILED_CSV = PLOT_DIR / 'iteration_values.csv'
OUTPUT_AGGREGATE_CSV = PLOT_DIR / 'aggregate_statistics.csv'
OUTPUT_SUMMARY = PLOT_DIR / 'statistics_summary.md'
EXPECTED_ITERATIONS = tuple(f'iter_{index:02d}' for index in range(1, 6))
EXPECTED_N = tuple(range(1, 16))


@dataclass(frozen=True)
class ModelSpec:
    label: str
    ollama_model: str
    directory_slug: str
    file_slug: str
    colour: str
    marker: str


@dataclass(frozen=True)
class RunValue:
    iteration: str
    model: str
    model_label: str
    kind: str
    n: int | None
    accuracy_percent: float
    hits: int
    total_news: int
    total_time_seconds: float
    e5_time_seconds: float
    slm_time_seconds: float
    source: Path


@dataclass(frozen=True)
class Statistic:
    model: str
    model_label: str
    kind: str
    n: int | None
    runs: int
    accuracy_mean: float
    accuracy_sd: float
    accuracy_min: float
    accuracy_max: float
    accuracy_ci95_low: float
    accuracy_ci95_high: float
    time_mean: float
    time_sd: float
    time_min: float
    time_max: float
    time_ci95_low: float
    time_ci95_high: float


MODEL_SPECS = (
    ModelSpec(
        'Llama 3.2 · 1B', 'llama3.2:1b', 'llama3_2_1b',
        'llama3_2_1b', '#60a5fa', 'circle'
    ),
    ModelSpec(
        'Llama 3.2 · 3B', 'llama3.2:3b', 'llama3_2_3b',
        'llama3_2_3b', '#2563eb', 'square'
    ),
    ModelSpec(
        'Llama 3.1 · 8B', 'llama3.1:8b', 'llama3_1_8b',
        'llama3_1_8b', '#1e3a8a', 'triangle'
    ),
    ModelSpec(
        'Qwen 2.5 · 1.5B', 'qwen2.5:1.5b-instruct',
        'qwen2_5_1_5b_instruct', 'qwen2_5_1_5b_instruct',
        '#34d399', 'circle'
    ),
    ModelSpec(
        'Qwen 2.5 · 3B', 'qwen2.5:3b-instruct',
        'qwen2_5_3b_instruct', 'qwen2_5_3b_instruct',
        '#059669', 'square'
    ),
    ModelSpec(
        'Qwen 2.5 · 7B', 'qwen2.5:7b-instruct',
        'qwen2_5_7b_instruct', 'qwen2_5_7b_instruct',
        '#065f46', 'triangle'
    ),
    ModelSpec(
        'Ministral 3 · 3B', 'ministral-3:3b-instruct-2512-q4_K_M',
        'ministral_3_3b_instruct',
        'ministral_3_3b_instruct_2512_q4_k_m', '#f59e0b', 'square'
    ),
    ModelSpec(
        'Ministral 3 · 8B', 'ministral-3:8b-instruct-2512-q4_K_M',
        'ministral_3_8b_instruct',
        'ministral_3_8b_instruct_2512_q4_k_m', '#dc2626', 'triangle'
    ),
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open('r', encoding='utf-8') as file_handle:
            value = json.load(file_handle)
    except FileNotFoundError as exc:
        raise SystemExit(f'No se encontró el resultado: {path}') from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f'JSON no válido en {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise SystemExit(f'{path}: se esperaba un objeto JSON.')
    return value


def mapping(value: Any, name: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f'{path}: falta el bloque {name}.')
    return value


def integer(value: Any, name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f'{path}: {name} no es un entero válido.')
    return value


def number(value: Any, name: str, path: Path) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SystemExit(f'{path}: {name} no es un número válido.')
    return float(value)


def validate_complete(
    report: dict[str, Any], path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    configuration = mapping(report.get('configuration'), 'configuration', path)
    dataset = mapping(report.get('dataset'), 'dataset', path)
    metrics = mapping(report.get('metrics'), 'metrics', path)
    timing = mapping(report.get('timing'), 'timing', path)
    total = integer(dataset.get('total_news'), 'dataset.total_news', path)
    completed = dataset.get('completed_predictions', total)
    expected = dataset.get('expected_predictions', total)
    if (
        report.get('status', 'complete') != 'complete'
        or completed != expected
        or expected != total
    ):
        raise SystemExit(f'{path}: el resultado no está completo.')
    return configuration, metrics, timing, total


def accuracy_from_hits(hits: int, total: int) -> float:
    return 100.0 * hits / total if total else 0.0


def load_e5(iteration: Path) -> RunValue:
    path = iteration / 'e5_results' / 'e5_top_n_results.json'
    report = load_json(path)
    configuration, metrics, timing, total = validate_complete(report, path)
    model = configuration.get('model')
    if model != 'intfloat/multilingual-e5-base':
        raise SystemExit(f'{path}: modelo E5 inesperado: {model!r}.')
    hits = integer(metrics.get('top_1_hits'), 'metrics.top_1_hits', path)
    elapsed = number(
        timing.get('total_e5_execution_time_seconds'),
        'timing.total_e5_execution_time_seconds',
        path,
    )
    return RunValue(
        iteration.name, model, 'E5 multilingual-base', 'e5_only', None,
        accuracy_from_hits(hits, total), hits, total, elapsed, elapsed, 0.0,
        path,
    )


def load_pipeline(iteration: Path, spec: ModelSpec, n: int) -> RunValue:
    path = (
        iteration
        / f'results_e5_{spec.directory_slug}_two_prompts_no_margin'
        / f'e5_top_{n}_{spec.file_slug}_results.json'
    )
    report = load_json(path)
    configuration, metrics, timing, total = validate_complete(report, path)
    if configuration.get('slm_model') != spec.ollama_model:
        raise SystemExit(f'{path}: modelo SLM inesperado.')
    if configuration.get('n') != n:
        raise SystemExit(f'{path}: configuration.n no coincide con N={n}.')
    hits = integer(metrics.get('final_hits'), 'metrics.final_hits', path)
    slm_time = number(
        timing.get('slm_total_inference_time_seconds'),
        'timing.slm_total_inference_time_seconds',
        path,
    )
    e5_timing = mapping(timing.get('e5_ranking'), 'timing.e5_ranking', path)
    e5_time = number(
        e5_timing.get('total_e5_execution_time_seconds'),
        'timing.e5_ranking.total_e5_execution_time_seconds',
        path,
    )
    return RunValue(
        iteration.name, spec.ollama_model, spec.label, 'pipeline', n,
        accuracy_from_hits(hits, total), hits, total, e5_time + slm_time,
        e5_time, slm_time, path,
    )


def load_direct(iteration: Path, spec: ModelSpec) -> RunValue:
    path = (
        iteration
        / f'results_{spec.directory_slug}_two_prompts'
        / f'{spec.file_slug}_direct_results.json'
    )
    report = load_json(path)
    configuration, metrics, timing, total = validate_complete(report, path)
    if configuration.get('model') != spec.ollama_model:
        raise SystemExit(f'{path}: modelo directo inesperado.')
    hits = integer(metrics.get('slm_final_hits'), 'metrics.slm_final_hits', path)
    slm_time = number(
        timing.get('total_slm_inference_time_seconds'),
        'timing.total_slm_inference_time_seconds',
        path,
    )
    return RunValue(
        iteration.name, spec.ollama_model, spec.label, 'slm_direct', 15,
        accuracy_from_hits(hits, total), hits, total, slm_time, 0.0,
        slm_time, path,
    )


def load_all_values() -> list[RunValue]:
    iterations = [ITERATIONS_ROOT / name for name in EXPECTED_ITERATIONS]
    missing = [path for path in iterations if not path.is_dir()]
    if missing:
        raise SystemExit(
            'Faltan iteraciones: ' + ', '.join(path.name for path in missing)
        )
    values: list[RunValue] = []
    for iteration in iterations:
        values.append(load_e5(iteration))
        for spec in MODEL_SPECS:
            values.extend(
                load_pipeline(iteration, spec, n) for n in EXPECTED_N
            )
            values.append(load_direct(iteration, spec))

    totals = {value.total_news for value in values}
    if len(totals) != 1:
        raise SystemExit(f'Las iteraciones usan tamaños distintos: {totals}.')
    return values


def t_critical_95(runs: int) -> float:
    values = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    return values.get(runs, 1.96)


def mean_sd_ci(values: list[float]) -> tuple[float, float, float, float, float, float]:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = t_critical_95(len(values)) * sd / math.sqrt(len(values))
    return mean, sd, min(values), max(values), mean - margin, mean + margin


def aggregate(values: list[RunValue]) -> list[Statistic]:
    grouped: dict[tuple[str, str, int | None], list[RunValue]] = defaultdict(list)
    for value in values:
        grouped[(value.model, value.kind, value.n)].append(value)

    statistics_rows: list[Statistic] = []
    for (model, kind, n), runs in grouped.items():
        if len(runs) != len(EXPECTED_ITERATIONS):
            raise SystemExit(
                f'{model}, {kind}, N={n}: hay {len(runs)} iteraciones.'
            )
        accuracy = mean_sd_ci([run.accuracy_percent for run in runs])
        elapsed = mean_sd_ci([run.total_time_seconds for run in runs])
        statistics_rows.append(
            Statistic(
                model=model,
                model_label=runs[0].model_label,
                kind=kind,
                n=n,
                runs=len(runs),
                accuracy_mean=accuracy[0],
                accuracy_sd=accuracy[1],
                accuracy_min=accuracy[2],
                accuracy_max=accuracy[3],
                accuracy_ci95_low=max(0.0, accuracy[4]),
                accuracy_ci95_high=min(100.0, accuracy[5]),
                time_mean=elapsed[0],
                time_sd=elapsed[1],
                time_min=elapsed[2],
                time_max=elapsed[3],
                time_ci95_low=max(0.0, elapsed[4]),
                time_ci95_high=elapsed[5],
            )
        )
    kind_order = {'e5_only': 0, 'pipeline': 1, 'slm_direct': 2}
    return sorted(
        statistics_rows,
        key=lambda row: (
            kind_order[row.kind], row.model_label, row.n or 0
        ),
    )


def write_detailed_csv(values: list[RunValue]) -> None:
    with OUTPUT_DETAILED_CSV.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                'iteration', 'model', 'model_label', 'kind', 'n',
                'accuracy_percent', 'hits', 'total_news',
                'total_time_seconds', 'e5_time_seconds',
                'slm_time_seconds', 'source',
            ),
        )
        writer.writeheader()
        for value in values:
            writer.writerow(
                {
                    'iteration': value.iteration,
                    'model': value.model,
                    'model_label': value.model_label,
                    'kind': value.kind,
                    'n': value.n if value.n is not None else '',
                    'accuracy_percent': round(value.accuracy_percent, 6),
                    'hits': value.hits,
                    'total_news': value.total_news,
                    'total_time_seconds': round(value.total_time_seconds, 6),
                    'e5_time_seconds': round(value.e5_time_seconds, 6),
                    'slm_time_seconds': round(value.slm_time_seconds, 6),
                    'source': value.source,
                }
            )


def write_aggregate_csv(rows: list[Statistic]) -> None:
    fields = tuple(Statistic.__dataclass_fields__)
    with OUTPUT_AGGREGATE_CSV.open(
        'w', encoding='utf-8', newline=''
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = {
                field: getattr(row, field) if field != 'n' or row.n is not None else ''
                for field in fields
            }
            for field, value in payload.items():
                if isinstance(value, float):
                    payload[field] = round(value, 6)
            writer.writerow(payload)


def nice_axis(maximum: float, ticks: int = 5) -> tuple[float, float]:
    raw_step = maximum / ticks if maximum else 1.0
    exponent = math.floor(math.log10(raw_step))
    magnitude = 10**exponent
    fraction = raw_step / magnitude
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    step = nice_fraction * magnitude
    return math.ceil(maximum / step) * step, step


def marker(svg: list[str], shape: str, x: float, y: float, colour: str) -> None:
    if shape == 'square':
        svg.append(
            f'<rect x="{x - 3.2:.1f}" y="{y - 3.2:.1f}" width="6.4" '
            f'height="6.4" fill="{colour}" stroke="#ffffff" stroke-width="0.7"/>'
        )
    elif shape == 'triangle':
        svg.append(
            f'<polygon points="{x:.1f},{y - 4:.1f} {x - 3.8:.1f},{y + 3.3:.1f} '
            f'{x + 3.8:.1f},{y + 3.3:.1f}" fill="{colour}" '
            'stroke="#ffffff" stroke-width="0.7"/>'
        )
    else:
        svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{colour}" '
            'stroke="#ffffff" stroke-width="0.7"/>'
        )


def diamond(svg: list[str], x: float, y: float, colour: str) -> None:
    svg.append(
        f'<polygon points="{x:.1f},{y - 7:.1f} {x - 7:.1f},{y:.1f} '
        f'{x:.1f},{y + 7:.1f} {x + 7:.1f},{y:.1f}" fill="#ffffff" '
        f'stroke="{colour}" stroke-width="2.5"/>'
    )


def path_points(points: Iterable[tuple[float, float]]) -> str:
    return ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)


def draw_panel(
    svg: list[str],
    rows_by_model: dict[str, list[Statistic]],
    direct_by_model: dict[str, Statistic],
    e5: Statistic,
    *,
    top: float,
    height: float,
    left: float,
    width: float,
    metric: str,
    y_min: float,
    y_max: float,
    y_step: float,
    title: str,
    subtitle: str,
) -> None:
    bottom = top + height

    def x_pos(n: float) -> float:
        return left + (n - 1) * width / 14

    def y_pos(value: float) -> float:
        return top + (y_max - value) * height / (y_max - y_min)

    svg.append(
        f'<rect x="{left - 72:.1f}" y="{top - 72:.1f}" '
        f'width="{width + 108:.1f}" height="{height + 125:.1f}" rx="14" '
        'fill="#ffffff" stroke="#dbe3ee"/>'
    )
    svg.append(
        f'<text x="{left - 50:.1f}" y="{top - 39:.1f}" fill="#0f172a" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="23" '
        f'font-weight="700">{escape(title)}</text>'
    )
    svg.append(
        f'<text x="{left - 50:.1f}" y="{top - 15:.1f}" fill="#64748b" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="14">'
        f'{escape(subtitle)}</text>'
    )

    tick = math.ceil(y_min / y_step) * y_step
    while tick <= y_max + y_step / 10:
        y = y_pos(tick)
        svg.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" '
            'y2="{:.1f}" stroke="#e2e8f0"/>'.format(y)
        )
        label = f'{tick:.0f}'
        svg.append(
            f'<text x="{left - 12:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#64748b" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="12">{label}</text>'
        )
        tick += y_step

    for n in EXPECTED_N:
        x = x_pos(n)
        svg.append(
            f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" '
            f'y2="{bottom:.1f}" stroke="#f1f5f9"/>'
        )
        svg.append(
            f'<text x="{x:.1f}" y="{bottom + 22:.1f}" text-anchor="middle" '
            f'fill="#64748b" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="11">{n}</text>'
        )
    svg.append(
        f'<text x="{left + width / 2:.1f}" y="{bottom + 45:.1f}" '
        'text-anchor="middle" fill="#334155" '
        'font-family="Arial, Helvetica, sans-serif" font-size="14">N candidatos</text>'
    )

    e5_mean = e5.accuracy_mean if metric == 'accuracy' else e5.time_mean
    e5_min = e5.accuracy_min if metric == 'accuracy' else e5.time_min
    e5_max = e5.accuracy_max if metric == 'accuracy' else e5.time_max
    y_e5_min = y_pos(e5_min)
    y_e5_max = y_pos(e5_max)
    svg.append(
        f'<rect x="{left:.1f}" y="{min(y_e5_min, y_e5_max):.1f}" '
        f'width="{width:.1f}" height="{abs(y_e5_min - y_e5_max):.1f}" '
        'fill="#0f172a" opacity="0.07"/>'
    )
    y_e5 = y_pos(e5_mean)
    svg.append(
        f'<line x1="{left:.1f}" y1="{y_e5:.1f}" x2="{left + width:.1f}" '
        f'y2="{y_e5:.1f}" stroke="#0f172a" stroke-width="1.8" '
        'stroke-dasharray="8 6"/>'
    )

    for spec in MODEL_SPECS:
        rows = rows_by_model[spec.ollama_model]
        minimums = [
            (x_pos(row.n or 1), y_pos(row.accuracy_min if metric == 'accuracy' else row.time_min))
            for row in rows
        ]
        maximums = [
            (x_pos(row.n or 1), y_pos(row.accuracy_max if metric == 'accuracy' else row.time_max))
            for row in rows
        ]
        polygon = maximums + list(reversed(minimums))
        svg.append(
            f'<polygon points="{path_points(polygon)}" fill="{spec.colour}" '
            'opacity="0.09"/>'
        )

        means: list[tuple[float, float]] = []
        for row in rows:
            value = row.accuracy_mean if metric == 'accuracy' else row.time_mean
            means.append((x_pos(row.n or 1), y_pos(value)))
        svg.append(
            f'<polyline points="{path_points(means)}" fill="none" '
            f'stroke="{spec.colour}" stroke-width="2.2" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )

        for row, (x, y) in zip(rows, means):
            if (row.n or 0) % 2 == 1:
                sd = row.accuracy_sd if metric == 'accuracy' else row.time_sd
                mean = row.accuracy_mean if metric == 'accuracy' else row.time_mean
                y_low = y_pos(max(y_min, mean - sd))
                y_high = y_pos(min(y_max, mean + sd))
                svg.append(
                    f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" '
                    f'y2="{y_high:.1f}" stroke="{spec.colour}" '
                    'stroke-width="1" opacity="0.75"/>'
                )
                for cap_y in (y_low, y_high):
                    svg.append(
                        f'<line x1="{x - 3:.1f}" y1="{cap_y:.1f}" '
                        f'x2="{x + 3:.1f}" y2="{cap_y:.1f}" '
                        f'stroke="{spec.colour}" stroke-width="1" opacity="0.75"/>'
                    )
            marker(svg, spec.marker, x, y, spec.colour)

        direct = direct_by_model[spec.ollama_model]
        direct_value = (
            direct.accuracy_mean if metric == 'accuracy' else direct.time_mean
        )
        plotted_direct = min(y_max, max(y_min, direct_value))
        direct_x = x_pos(15)
        direct_y = y_pos(plotted_direct)
        direct_sd = (
            direct.accuracy_sd if metric == 'accuracy' else direct.time_sd
        )
        direct_min = (
            direct.accuracy_min if metric == 'accuracy' else direct.time_min
        )
        direct_max = (
            direct.accuracy_max if metric == 'accuracy' else direct.time_max
        )
        direct_range_low = y_pos(min(y_max, max(y_min, direct_min)))
        direct_range_high = y_pos(min(y_max, max(y_min, direct_max)))
        direct_sd_low = y_pos(
            min(y_max, max(y_min, direct_value - direct_sd))
        )
        direct_sd_high = y_pos(
            min(y_max, max(y_min, direct_value + direct_sd))
        )
        svg.append(
            f'<line x1="{direct_x:.1f}" y1="{direct_range_low:.1f}" '
            f'x2="{direct_x:.1f}" y2="{direct_range_high:.1f}" '
            f'stroke="{spec.colour}" stroke-width="1.2" opacity="0.55"/>'
        )
        svg.append(
            f'<line x1="{direct_x:.1f}" y1="{direct_sd_low:.1f}" '
            f'x2="{direct_x:.1f}" y2="{direct_sd_high:.1f}" '
            f'stroke="{spec.colour}" stroke-width="3" opacity="0.8"/>'
        )
        diamond(svg, direct_x, direct_y, spec.colour)
        if direct_value < y_min or direct_value > y_max:
            arrow = '↓' if direct_value < y_min else '↑'
            label_y = bottom - 10 if direct_value < y_min else top + 17
            svg.append(
                f'<text x="{direct_x - 12:.1f}" y="{label_y:.1f}" '
                f'text-anchor="end" fill="{spec.colour}" '
                'font-family="Arial, Helvetica, sans-serif" font-size="11" '
                f'font-weight="700">{direct_value:.1f}% {arrow}</text>'
            )


def generate_svg(rows: list[Statistic], total_news: int) -> str:
    pipeline = [row for row in rows if row.kind == 'pipeline']
    direct = {
        row.model: row for row in rows if row.kind == 'slm_direct'
    }
    e5 = next(row for row in rows if row.kind == 'e5_only')
    by_model: dict[str, list[Statistic]] = defaultdict(list)
    for row in pipeline:
        by_model[row.model].append(row)
    for model_rows in by_model.values():
        model_rows.sort(key=lambda row: row.n or 0)

    accuracy_axis_rows = pipeline + [e5]
    accuracy_values = [
        value
        for row in accuracy_axis_rows
        for value in (row.accuracy_min, row.accuracy_max)
    ]
    accuracy_min = max(0.0, 5 * math.floor((min(accuracy_values) - 1) / 5))
    accuracy_max = min(100.0, 5 * math.ceil((max(accuracy_values) + 1) / 5))
    time_maximum = max(row.time_max for row in rows) * 1.05
    time_max, time_step = nice_axis(time_maximum, 6)

    width = 1780
    height = 1120
    plot_left = 115
    plot_width = 1210
    panel_height = 340
    accuracy_top = 185
    time_top = 695
    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title description">'
        ),
        '<title id="title">Accuracy y tiempo en cinco iteraciones</title>',
        (
            '<desc id="description">Media, desviación estándar y rango de '
            'accuracy y tiempo total para ocho modelos y quince valores N.</desc>'
        ),
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        (
            '<text x="55" y="48" fill="#0f172a" '
            'font-family="Arial, Helvetica, sans-serif" font-size="32" '
            'font-weight="700">Cinco iteraciones: accuracy y coste temporal</text>'
        ),
        (
            '<text x="55" y="78" fill="#475569" '
            'font-family="Arial, Helvetica, sans-serif" font-size="16">'
            f'{total_news} noticias · 8 modelos · N=1…15 · dos prompts por noticia</text>'
        ),
    ]

    draw_panel(
        svg, by_model, direct, e5,
        top=accuracy_top, height=panel_height, left=plot_left,
        width=plot_width, metric='accuracy', y_min=accuracy_min,
        y_max=accuracy_max, y_step=5,
        title='Accuracy frente al número de candidatos',
        subtitle='Curva: media · barras: ±1 DE · banda: mínimo–máximo entre iteraciones',
    )
    draw_panel(
        svg, by_model, direct, e5,
        top=time_top, height=panel_height, left=plot_left,
        width=plot_width, metric='time', y_min=0,
        y_max=time_max, y_step=time_step,
        title='Tiempo total frente al número de candidatos',
        subtitle='Segundos para todo el test; el pipeline incluye E5 y las dos llamadas al SLM',
    )

    legend_x = 1390
    svg.append(
        f'<rect x="{legend_x - 24}" y="112" width="360" height="825" '
        'rx="14" fill="#ffffff" stroke="#dbe3ee"/>'
    )
    svg.append(
        f'<text x="{legend_x}" y="150" fill="#0f172a" '
        'font-family="Arial, Helvetica, sans-serif" font-size="20" '
        'font-weight="700">Modelos</text>'
    )
    for index, spec in enumerate(MODEL_SPECS):
        y = 190 + index * 49
        svg.append(
            f'<line x1="{legend_x:.1f}" y1="{y:.1f}" '
            f'x2="{legend_x + 45:.1f}" y2="{y:.1f}" '
            f'stroke="{spec.colour}" stroke-width="3"/>'
        )
        marker(svg, spec.marker, legend_x + 22.5, y, spec.colour)
        svg.append(
            f'<text x="{legend_x + 58:.1f}" y="{y + 5:.1f}" '
            'fill="#334155" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="15">{escape(spec.label)}</text>'
        )

    y = 600
    svg.append(
        f'<line x1="{legend_x:.1f}" y1="{y:.1f}" '
        f'x2="{legend_x + 45:.1f}" y2="{y:.1f}" stroke="#0f172a" '
        'stroke-width="2" stroke-dasharray="8 6"/>'
    )
    svg.append(
        f'<text x="{legend_x + 58:.1f}" y="{y + 5:.1f}" fill="#334155" '
        'font-family="Arial, Helvetica, sans-serif" font-size="15">E5 solo</text>'
    )
    diamond(svg, legend_x + 22, 647, '#64748b')
    svg.append(
        f'<text x="{legend_x + 58:.1f}" y="652" fill="#334155" '
        'font-family="Arial, Helvetica, sans-serif" font-size="15">'
        'SLM directo (15 temas)</text>'
    )

    svg.extend(
        [
            f'<text x="{legend_x}" y="710" fill="#0f172a" '
            'font-family="Arial, Helvetica, sans-serif" font-size="18" '
            'font-weight="700">Estadísticos</text>',
            f'<text x="{legend_x}" y="742" fill="#475569" '
            'font-family="Arial, Helvetica, sans-serif" font-size="14">'
            'n = 5 ejecuciones por punto</text>',
            f'<text x="{legend_x}" y="769" fill="#475569" '
            'font-family="Arial, Helvetica, sans-serif" font-size="14">'
            'DE = desviación estándar muestral</text>',
            f'<text x="{legend_x}" y="796" fill="#475569" '
            'font-family="Arial, Helvetica, sans-serif" font-size="14">'
            'Banda = rango observado</text>',
            f'<text x="{legend_x}" y="823" fill="#475569" '
            'font-family="Arial, Helvetica, sans-serif" font-size="14">'
            'IC 95% disponible en el CSV</text>',
            f'<text x="{legend_x}" y="875" fill="#64748b" '
            'font-family="Arial, Helvetica, sans-serif" font-size="13">'
            'Los rombos son ejecuciones directas;</text>',
            f'<text x="{legend_x}" y="897" fill="#64748b" '
            'font-family="Arial, Helvetica, sans-serif" font-size="13">'
            'no forman parte de las curvas Top-N.</text>',
        ]
    )
    svg.append('</svg>')
    return '\n'.join(svg) + '\n'


def write_summary(rows: list[Statistic]) -> None:
    pipeline = [row for row in rows if row.kind == 'pipeline']
    direct = {row.model: row for row in rows if row.kind == 'slm_direct'}
    lines = [
        '# Resumen estadístico de las cinco iteraciones',
        '',
        '| Modelo | Mejor N | Accuracy media ± DE | Tiempo medio ± DE | Accuracy directa | Tiempo directo |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for spec in MODEL_SPECS:
        candidates = [row for row in pipeline if row.model == spec.ollama_model]
        best = max(candidates, key=lambda row: (row.accuracy_mean, -int(row.n or 0)))
        direct_row = direct[spec.ollama_model]
        lines.append(
            f'| {spec.label} | {best.n} | '
            f'{best.accuracy_mean:.2f}% ± {best.accuracy_sd:.2f} | '
            f'{best.time_mean:.1f}s ± {best.time_sd:.1f} | '
            f'{direct_row.accuracy_mean:.2f}% ± '
            f'{direct_row.accuracy_sd:.2f} | '
            f'{direct_row.time_mean:.1f}s ± {direct_row.time_sd:.1f} |'
        )
    lines.extend(
        [
            '',
            'El mejor N se elige por mayor accuracy media; en caso de empate, '
            'se conserva el menor N. Los tiempos del pipeline incluyen E5 y '
            'las dos llamadas al SLM.',
            '',
        ]
    )
    OUTPUT_SUMMARY.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    values = load_all_values()
    rows = aggregate(values)
    total_news = values[0].total_news
    write_detailed_csv(values)
    write_aggregate_csv(rows)
    write_summary(rows)
    OUTPUT_SVG.write_text(generate_svg(rows, total_news), encoding='utf-8')
    print(f'Iteraciones: {len(EXPECTED_ITERATIONS)}')
    print(f'Modelos: {len(MODEL_SPECS)}')
    print(f'Valores individuales: {len(values)}')
    print(f'Estadísticos agregados: {len(rows)}')
    print(f'Gráfica: {OUTPUT_SVG}')
    print(f'Datos agregados: {OUTPUT_AGGREGATE_CSV}')
    print(f'Datos por iteración: {OUTPUT_DETAILED_CSV}')
    print(f'Resumen: {OUTPUT_SUMMARY}')


if __name__ == '__main__':
    main()
