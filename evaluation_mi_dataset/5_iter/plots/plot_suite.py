#!/usr/bin/env python3
"""Generadores compartidos de las gráficas agregadas de cinco iteraciones."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable
from xml.sax.saxutils import escape


PLOTS_ROOT = Path(__file__).resolve().parent
SUMMARY_PLOT_DIR = PLOTS_ROOT / 'accuracy_and_time_statistics'
sys.path.insert(0, str(SUMMARY_PLOT_DIR))

import plot_accuracy_and_time_statistics as base  # noqa: E402


FONT = 'Arial, Helvetica, sans-serif'


def svg_document(
    width: int, height: int, title: str, description: str
) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title description">'
        ),
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="description">{escape(description)}</desc>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        (
            f'<text x="55" y="48" fill="#0f172a" font-family="{FONT}" '
            f'font-size="32" font-weight="700">{escape(title)}</text>'
        ),
    ]


def text(
    svg: list[str], x: float, y: float, value: str, *, size: int = 12,
    colour: str = '#475569', anchor: str = 'start', weight: str = '400'
) -> None:
    svg.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'fill="{colour}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}">{escape(value)}</text>'
    )


def polyline(
    svg: list[str], points: Iterable[tuple[float, float]], colour: str,
    *, width: float = 2.2, dash: str | None = None, opacity: float = 1.0
) -> None:
    coords = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
    dash_attribute = f' stroke-dasharray="{dash}"' if dash else ''
    svg.append(
        f'<polyline points="{coords}" fill="none" stroke="{colour}" '
        f'stroke-width="{width}" stroke-linejoin="round" '
        f'stroke-linecap="round" opacity="{opacity}"{dash_attribute}/>'
    )


def range_band(
    svg: list[str], upper: list[tuple[float, float]],
    lower: list[tuple[float, float]], colour: str, opacity: float = 0.1
) -> None:
    points = upper + list(reversed(lower))
    coords = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
    svg.append(
        f'<polygon points="{coords}" fill="{colour}" opacity="{opacity}"/>'
    )


def error_bar(
    svg: list[str], x: float, low: float, high: float, colour: str,
    *, width: float = 1.0, cap: float = 3.0, opacity: float = 0.75
) -> None:
    svg.append(
        f'<line x1="{x:.1f}" y1="{low:.1f}" x2="{x:.1f}" '
        f'y2="{high:.1f}" stroke="{colour}" stroke-width="{width}" '
        f'opacity="{opacity}"/>'
    )
    for y in (low, high):
        svg.append(
            f'<line x1="{x - cap:.1f}" y1="{y:.1f}" '
            f'x2="{x + cap:.1f}" y2="{y:.1f}" stroke="{colour}" '
            f'stroke-width="{width}" opacity="{opacity}"/>'
        )


def circle(
    svg: list[str], x: float, y: float, colour: str, radius: float = 3.0,
    *, fill: str | None = None, width: float = 1.0
) -> None:
    svg.append(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
        f'fill="{fill or colour}" stroke="{colour}" stroke-width="{width}"/>'
    )


def triangle(svg: list[str], x: float, y: float, colour: str) -> None:
    svg.append(
        f'<polygon points="{x:.1f},{y - 7:.1f} {x - 6.5:.1f},{y + 5.5:.1f} '
        f'{x + 6.5:.1f},{y + 5.5:.1f}" fill="#ffffff" '
        f'stroke="{colour}" stroke-width="2.2"/>'
    )


def model_marker(
    svg: list[str], shape: str, x: float, y: float, colour: str
) -> None:
    """Dibuja el marcador asociado a un tamaño de modelo."""
    if shape == 'square':
        svg.append(
            f'<rect x="{x - 3.2:.1f}" y="{y - 3.2:.1f}" width="6.4" '
            f'height="6.4" fill="{colour}" stroke="#ffffff" '
            'stroke-width="0.7"/>'
        )
    elif shape == 'triangle':
        svg.append(
            f'<polygon points="{x:.1f},{y - 4:.1f} '
            f'{x - 3.8:.1f},{y + 3.3:.1f} '
            f'{x + 3.8:.1f},{y + 3.3:.1f}" fill="{colour}" '
            'stroke="#ffffff" stroke-width="0.7"/>'
        )
    else:
        circle(svg, x, y, colour, 3.2, width=0.7)


def nice_axis(maximum: float, ticks: int = 5) -> tuple[float, float]:
    raw = maximum / ticks if maximum else 1.0
    exponent = math.floor(math.log10(raw))
    magnitude = 10**exponent
    fraction = raw / magnitude
    if fraction <= 1:
        nice = 1
    elif fraction <= 2:
        nice = 2
    elif fraction <= 2.5:
        nice = 2.5
    elif fraction <= 5:
        nice = 5
    else:
        nice = 10
    step = nice * magnitude
    return math.ceil(maximum / step) * step, step


def draw_axes(
    svg: list[str], left: float, top: float, width: float, height: float,
    *, y_min: float, y_max: float, y_step: float,
    x_values: Iterable[int] = base.EXPECTED_N, x_label: str = 'N'
) -> tuple[Callable[[float], float], Callable[[float], float]]:
    x_values = tuple(x_values)

    def xpos(value: float) -> float:
        return left + (value - min(x_values)) * width / (
            max(x_values) - min(x_values)
        )

    def ypos(value: float) -> float:
        return top + (y_max - value) * height / (y_max - y_min)

    tick = math.ceil(y_min / y_step) * y_step
    while tick <= y_max + y_step / 10:
        y = ypos(tick)
        svg.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" '
            f'y2="{y:.1f}" stroke="#e2e8f0"/>'
        )
        text(svg, left - 8, y + 4, f'{tick:g}', anchor='end', size=10)
        tick += y_step
    for value in x_values:
        x = xpos(value)
        svg.append(
            f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" '
            f'y2="{top + height:.1f}" stroke="#f1f5f9"/>'
        )
        text(svg, x, top + height + 17, str(value), anchor='middle', size=9)
    text(
        svg, left + width / 2, top + height + 38, x_label,
        anchor='middle', size=12, colour='#334155'
    )
    return xpos, ypos


def stat_payload(row: base.Statistic) -> dict[str, Any]:
    return {
        name: getattr(row, name)
        for name in base.Statistic.__dataclass_fields__
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit(f'No hay filas para {path}.')
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def base_statistics() -> tuple[list[base.RunValue], list[base.Statistic]]:
    values = base.load_all_values()
    return values, base.aggregate(values)


def pipeline_by_model(
    rows: list[base.Statistic]
) -> dict[str, list[base.Statistic]]:
    grouped: dict[str, list[base.Statistic]] = defaultdict(list)
    for row in rows:
        if row.kind == 'pipeline':
            grouped[row.model].append(row)
    for model_rows in grouped.values():
        model_rows.sort(key=lambda row: row.n or 0)
    return grouped


def e5_statistic(rows: list[base.Statistic]) -> base.Statistic:
    return next(row for row in rows if row.kind == 'e5_only')


def direct_statistics(
    rows: list[base.Statistic]
) -> dict[str, base.Statistic]:
    return {row.model: row for row in rows if row.kind == 'slm_direct'}


def hybrid_report_path(iteration: str, spec: base.ModelSpec, n: int) -> Path:
    return (
        base.ITERATIONS_ROOT
        / iteration
        / f'results_e5_{spec.directory_slug}_two_prompts_no_margin'
        / f'e5_top_{n}_{spec.file_slug}_results.json'
    )


def summarize(values: list[float]) -> dict[str, float]:
    mean, sd, minimum, maximum, ci_low, ci_high = base.mean_sd_ci(values)
    return {
        'mean': mean,
        'sd': sd,
        'min': minimum,
        'max': maximum,
        'ci95_low': ci_low,
        'ci95_high': ci_high,
    }


def aggregate_hybrid_metrics(
    extractor: Callable[[dict[str, Any], Path], dict[str, float]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    labels: dict[str, str] = {}
    for iteration in base.EXPECTED_ITERATIONS:
        for spec in base.MODEL_SPECS:
            labels[spec.ollama_model] = spec.label
            for n in base.EXPECTED_N:
                path = hybrid_report_path(iteration, spec, n)
                report = base.load_json(path)
                base.validate_complete(report, path)
                for name, value in extractor(report, path).items():
                    grouped[(spec.ollama_model, n)][name].append(float(value))

    rows: list[dict[str, Any]] = []
    for (model, n), metrics in grouped.items():
        row: dict[str, Any] = {
            'model': model,
            'model_label': labels[model],
            'n': n,
            'runs': len(base.EXPECTED_ITERATIONS),
        }
        for metric_name, metric_values in metrics.items():
            for suffix, value in summarize(metric_values).items():
                row[f'{metric_name}_{suffix}'] = value
        rows.append(row)
    return sorted(rows, key=lambda row: (row['model_label'], row['n']))


def draw_model_legend(
    svg: list[str], x: float, y: float, *, columns: int = 4,
    row_height: float = 28, column_width: float = 360
) -> None:
    for index, spec in enumerate(base.MODEL_SPECS):
        column = index % columns
        row = index // columns
        item_x = x + column * column_width
        item_y = y + row * row_height
        svg.append(
            f'<line x1="{item_x:.1f}" y1="{item_y:.1f}" '
            f'x2="{item_x + 34:.1f}" y2="{item_y:.1f}" '
            f'stroke="{spec.colour}" stroke-width="3"/>'
        )
        circle(svg, item_x + 17, item_y, spec.colour, 3.2)
        text(svg, item_x + 43, item_y + 5, spec.label, size=13)


def generate_accuracy_vs_candidates(output_dir: Path) -> None:
    _, rows = base_statistics()
    by_model = pipeline_by_model(rows)
    direct = direct_statistics(rows)
    e5 = e5_statistic(rows)
    width, height = 1680, 1340
    svg = svg_document(
        width, height, 'Accuracy frente al número de candidatos',
        'Ocho paneles con media y variabilidad de cinco iteraciones.'
    )
    text(
        svg, 55, 78,
        'Media de 5 iteraciones · banda mínimo–máximo · barras ±1 DE',
        size=16,
    )
    panel_width, panel_height = 780, 270
    for index, spec in enumerate(base.MODEL_SPECS):
        row_index, column = divmod(index, 2)
        panel_x = 55 + column * 815
        panel_y = 115 + row_index * 297
        svg.append(
            f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width}" '
            f'height="{panel_height}" rx="12" fill="#ffffff" '
            'stroke="#dbe3ee"/>'
        )
        text(
            svg, panel_x + panel_width / 2, panel_y + 27, spec.label,
            anchor='middle', size=17, colour='#0f172a', weight='700'
        )
        left, top = panel_x + 55, panel_y + 45
        plot_width, plot_height = panel_width - 80, 175
        xpos, ypos = draw_axes(
            svg, left, top, plot_width, plot_height,
            y_min=0, y_max=100, y_step=20, x_label='N candidatos'
        )
        model_rows = by_model[spec.ollama_model]
        upper = [(xpos(row.n or 1), ypos(row.accuracy_max)) for row in model_rows]
        lower = [(xpos(row.n or 1), ypos(row.accuracy_min)) for row in model_rows]
        range_band(svg, upper, lower, spec.colour)
        means = [(xpos(row.n or 1), ypos(row.accuracy_mean)) for row in model_rows]
        polyline(svg, means, spec.colour, width=2.5)
        for row, (x, y) in zip(model_rows, means):
            if (row.n or 0) % 2 == 1:
                error_bar(
                    svg, x, ypos(row.accuracy_mean - row.accuracy_sd),
                    ypos(row.accuracy_mean + row.accuracy_sd), spec.colour
                )
            circle(svg, x, y, spec.colour, 2.8)
        e5_y = ypos(e5.accuracy_mean)
        svg.append(
            f'<line x1="{left}" y1="{e5_y:.1f}" x2="{left + plot_width}" '
            f'y2="{e5_y:.1f}" stroke="#0f172a" stroke-dasharray="7 5"/>'
        )
        direct_row = direct[spec.ollama_model]
        direct_y = ypos(direct_row.accuracy_mean)
        direct_low = ypos(direct_row.accuracy_max)
        direct_high = ypos(direct_row.accuracy_min)
        svg.append(
            f'<rect x="{left}" y="{direct_low:.1f}" width="{plot_width}" '
            f'height="{max(0.5, direct_high - direct_low):.1f}" '
            f'fill="{spec.colour}" opacity="0.07"/>'
        )
        svg.append(
            f'<line x1="{left}" y1="{direct_y:.1f}" '
            f'x2="{left + plot_width}" y2="{direct_y:.1f}" '
            f'stroke="{spec.colour}" stroke-dasharray="3 4"/>'
        )
    text(
        svg, 55, 1322,
        'Discontinua negra: E5 Top-1 · Punteada de color: SLM directo con 15 temas',
        size=13,
    )
    svg.append('</svg>')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'accuracy_vs_candidates.svg').write_text(
        '\n'.join(svg) + '\n', encoding='utf-8'
    )
    write_csv(
        output_dir / 'accuracy_vs_candidates_data.csv',
        [stat_payload(row) for row in rows],
    )


def generate_accuracy_vs_candidates_by_family(output_dir: Path) -> None:
    """Agrupa los modelos por familia y representa una curva por tamaño."""
    _, rows = base_statistics()
    by_model = pipeline_by_model(rows)
    e5 = e5_statistic(rows)
    families = (
        (
            'Llama',
            tuple(
                spec for spec in base.MODEL_SPECS
                if spec.label.startswith('Llama')
            ),
        ),
        (
            'Qwen 2.5',
            tuple(
                spec for spec in base.MODEL_SPECS
                if spec.label.startswith('Qwen')
            ),
        ),
        (
            'Ministral 3',
            tuple(
                spec for spec in base.MODEL_SPECS
                if spec.label.startswith('Ministral')
            ),
        ),
    )

    width, height = 1830, 720
    svg = svg_document(
        width, height, 'Accuracy frente al número de candidatos por familia',
        'Tres paneles por familia; cada curva corresponde a un tamaño de modelo.'
    )
    text(
        svg, 55, 78,
        'Media de 5 iteraciones · banda mínimo–máximo · barras ±1 DE',
        size=16,
    )
    svg.append(
        '<line x1="1350" y1="73" x2="1390" y2="73" '
        'stroke="#0f172a" stroke-width="1.8" stroke-dasharray="7 5"/>'
    )
    text(
        svg, 1400, 78, f'E5 Top-1: {e5.accuracy_mean:.2f}%',
        size=14, colour='#0f172a'
    )

    panel_width, panel_height = 570, 550
    for panel_index, (family, specs) in enumerate(families):
        panel_x = 35 + panel_index * 600
        panel_y = 110
        svg.append(
            f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width}" '
            f'height="{panel_height}" rx="12" fill="#ffffff" '
            'stroke="#dbe3ee"/>'
        )
        text(
            svg, panel_x + panel_width / 2, panel_y + 32, family,
            anchor='middle', size=20, colour='#0f172a', weight='700'
        )

        legend_width = 145
        legend_start = panel_x + (
            panel_width - legend_width * len(specs)
        ) / 2
        for legend_index, spec in enumerate(specs):
            item_x = legend_start + legend_index * legend_width
            item_y = panel_y + 60
            svg.append(
                f'<line x1="{item_x:.1f}" y1="{item_y:.1f}" '
                f'x2="{item_x + 30:.1f}" y2="{item_y:.1f}" '
                f'stroke="{spec.colour}" stroke-width="2.5"/>'
            )
            model_marker(
                svg, spec.marker, item_x + 15, item_y, spec.colour
            )
            size_label = spec.label.split(' · ', maxsplit=1)[1]
            text(svg, item_x + 38, item_y + 4, size_label, size=12)

        left, top = panel_x + 58, panel_y + 88
        plot_width, plot_height = panel_width - 82, 390
        xpos, ypos = draw_axes(
            svg, left, top, plot_width, plot_height,
            y_min=0, y_max=100, y_step=20, x_label='N candidatos'
        )

        e5_y = ypos(e5.accuracy_mean)
        svg.append(
            f'<line x1="{left}" y1="{e5_y:.1f}" '
            f'x2="{left + plot_width}" y2="{e5_y:.1f}" '
            'stroke="#0f172a" stroke-width="1.8" '
            'stroke-dasharray="7 5"/>'
        )

        for spec in specs:
            model_rows = by_model[spec.ollama_model]
            upper = [
                (xpos(row.n or 1), ypos(row.accuracy_max))
                for row in model_rows
            ]
            lower = [
                (xpos(row.n or 1), ypos(row.accuracy_min))
                for row in model_rows
            ]
            range_band(svg, upper, lower, spec.colour, 0.09)
            means = [
                (xpos(row.n or 1), ypos(row.accuracy_mean))
                for row in model_rows
            ]
            polyline(svg, means, spec.colour, width=2.5)
            for row, (x, y) in zip(model_rows, means):
                if (row.n or 0) % 2 == 1:
                    error_bar(
                        svg, x,
                        ypos(row.accuracy_mean - row.accuracy_sd),
                        ypos(row.accuracy_mean + row.accuracy_sd),
                        spec.colour,
                    )
                model_marker(svg, spec.marker, x, y, spec.colour)

    text(
        svg, 55, 700,
        'Curvas: pipeline E5 → SLM · referencia discontinua: E5 Top-1',
        size=13,
    )
    svg.append('</svg>')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'accuracy_vs_candidates_by_family.svg').write_text(
        '\n'.join(svg) + '\n', encoding='utf-8'
    )

    data_rows: list[dict[str, Any]] = [
        {'family': 'E5', 'size': '', **stat_payload(e5)}
    ]
    for family, specs in families:
        for spec in specs:
            size = spec.label.split(' · ', maxsplit=1)[1]
            data_rows.extend(
                {
                    'family': family,
                    'size': size,
                    **stat_payload(row),
                }
                for row in by_model[spec.ollama_model]
            )
    write_csv(
        output_dir / 'accuracy_vs_candidates_by_family_data.csv',
        data_rows,
    )


def generate_n_vs_time(output_dir: Path) -> None:
    values, rows = base_statistics()
    total_news = values[0].total_news
    by_model = pipeline_by_model(rows)
    direct = direct_statistics(rows)
    e5 = e5_statistic(rows)
    families = (
        ('Llama', tuple(spec for spec in base.MODEL_SPECS if spec.label.startswith('Llama'))),
        ('Qwen', tuple(spec for spec in base.MODEL_SPECS if spec.label.startswith('Qwen'))),
        ('Ministral', tuple(spec for spec in base.MODEL_SPECS if spec.label.startswith('Ministral'))),
    )
    max_time = max(row.time_max for row in rows) * 1.08
    y_max, y_step = nice_axis(max_time, 6)
    width, height = 1810, 710
    svg = svg_document(
        width, height, 'Tiempo total frente al número de candidatos',
        'Tres paneles muestran media y variabilidad temporal por familia.'
    )
    text(
        svg, 55, 78,
        f'{total_news} noticias · media de 5 iteraciones · tiempo E5 + dos prompts',
        size=16,
    )
    panel_width, panel_height = 550, 540
    for panel_index, (family, specs) in enumerate(families):
        panel_x = 45 + panel_index * 590
        panel_y = 115
        svg.append(
            f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width}" '
            f'height="{panel_height}" rx="12" fill="#ffffff" '
            'stroke="#dbe3ee"/>'
        )
        text(
            svg, panel_x + panel_width / 2, panel_y + 31, family,
            anchor='middle', size=20, colour='#0f172a', weight='700'
        )
        left, top = panel_x + 60, panel_y + 55
        plot_width, plot_height = panel_width - 85, 390
        xpos, ypos = draw_axes(
            svg, left, top, plot_width, plot_height,
            y_min=0, y_max=y_max, y_step=y_step,
            x_label='N candidatos'
        )
        e5_y = ypos(e5.time_mean)
        svg.append(
            f'<line x1="{left}" y1="{e5_y:.1f}" x2="{left + plot_width}" '
            f'y2="{e5_y:.1f}" stroke="#0f172a" stroke-dasharray="7 5"/>'
        )
        for legend_index, spec in enumerate(specs):
            model_rows = by_model[spec.ollama_model]
            upper = [(xpos(row.n or 1), ypos(row.time_max)) for row in model_rows]
            lower = [(xpos(row.n or 1), ypos(row.time_min)) for row in model_rows]
            range_band(svg, upper, lower, spec.colour, 0.08)
            means = [(xpos(row.n or 1), ypos(row.time_mean)) for row in model_rows]
            polyline(svg, means, spec.colour)
            for row, (x, y) in zip(model_rows, means):
                if (row.n or 0) % 2 == 1:
                    error_bar(
                        svg, x, ypos(row.time_mean - row.time_sd),
                        ypos(row.time_mean + row.time_sd), spec.colour
                    )
                circle(svg, x, y, spec.colour, 2.7)
            direct_row = direct[spec.ollama_model]
            direct_x, direct_y = xpos(15), ypos(direct_row.time_mean)
            error_bar(
                svg, direct_x, ypos(direct_row.time_min),
                ypos(direct_row.time_max), spec.colour, width=1.4
            )
            triangle(svg, direct_x, direct_y, spec.colour)
            legend_y = panel_y + 490 + legend_index * 22
            svg.append(
                f'<line x1="{panel_x + 30}" y1="{legend_y}" '
                f'x2="{panel_x + 58}" y2="{legend_y}" '
                f'stroke="{spec.colour}" stroke-width="2.5"/>'
            )
            text(svg, panel_x + 67, legend_y + 4, spec.label, size=11)
    svg.append('</svg>')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'n_vs_time.svg').write_text(
        '\n'.join(svg) + '\n', encoding='utf-8'
    )
    write_csv(
        output_dir / 'n_vs_time_data.csv',
        [stat_payload(row) for row in rows],
    )


def pareto_frontier(rows: list[base.Statistic], total_news: int) -> list[base.Statistic]:
    def x(row: base.Statistic) -> float:
        return row.time_mean / total_news

    frontier: list[base.Statistic] = []
    for candidate in rows:
        dominated = any(
            other is not candidate
            and x(other) <= x(candidate)
            and other.accuracy_mean >= candidate.accuracy_mean
            and (
                x(other) < x(candidate)
                or other.accuracy_mean > candidate.accuracy_mean
            )
            for other in rows
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=x)


def scatter_rows(rows: list[base.Statistic]) -> list[base.Statistic]:
    return [row for row in rows if row.kind in {'e5_only', 'pipeline', 'slm_direct'}]


def scatter_marker(
    svg: list[str], row: base.Statistic, x: float, y: float, colour: str
) -> None:
    if row.kind == 'slm_direct':
        triangle(svg, x, y, colour)
    elif row.kind == 'e5_only':
        svg.append(
            f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" '
            f'height="10" fill="#0f172a"/>'
        )
    else:
        circle(svg, x, y, colour, 3.7)


def model_colour(model: str) -> str:
    if model == 'intfloat/multilingual-e5-base':
        return '#0f172a'
    return next(
        spec.colour for spec in base.MODEL_SPECS
        if spec.ollama_model == model
    )


def generate_accuracy_vs_time(output_dir: Path) -> None:
    values, all_rows = base_statistics()
    total_news = values[0].total_news
    rows = scatter_rows(all_rows)
    frontier = pareto_frontier(rows, total_news)
    max_x = max(row.time_max / total_news for row in rows) * 1.08
    x_max, x_step = nice_axis(max_x, 7)
    width, height = 1650, 870
    svg = svg_document(
        width, height, 'Accuracy frente a tiempo',
        'Dispersión de medias con incertidumbre y frontera de Pareto.'
    )
    text(
        svg, 55, 78,
        'Puntos: media de 5 iteraciones · barras: ±1 DE · dorado: frontera de Pareto',
        size=16,
    )
    left, top, plot_width, plot_height = 105, 135, 1180, 620

    def xpos(value: float) -> float:
        return left + value * plot_width / x_max

    def ypos(value: float) -> float:
        return top + (100 - value) * plot_height / 100

    tick = 0.0
    while tick <= x_max + x_step / 10:
        x = xpos(tick)
        svg.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
            f'y2="{top + plot_height}" stroke="#e2e8f0"/>'
        )
        text(svg, x, top + plot_height + 20, f'{tick:.2f}', anchor='middle', size=10)
        tick += x_step
    for accuracy in range(0, 101, 20):
        y = ypos(accuracy)
        svg.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" '
            f'y2="{y:.1f}" stroke="#e2e8f0"/>'
        )
        text(svg, left - 9, y + 4, str(accuracy), anchor='end', size=10)
    text(
        svg, left + plot_width / 2, top + plot_height + 48,
        'Tiempo medio total por noticia (s)', anchor='middle', size=14,
        colour='#334155'
    )
    frontier_points = [
        (xpos(row.time_mean / total_news), ypos(row.accuracy_mean))
        for row in frontier
    ]
    polyline(svg, frontier_points, '#d97706', width=2.2, dash='7 5')
    for row in rows:
        colour = model_colour(row.model)
        x_value = row.time_mean / total_news
        x_sd = row.time_sd / total_news
        x, y = xpos(x_value), ypos(row.accuracy_mean)
        error_bar(
            svg, x, ypos(row.accuracy_mean - row.accuracy_sd),
            ypos(row.accuracy_mean + row.accuracy_sd), colour
        )
        svg.append(
            f'<line x1="{xpos(max(0, x_value - x_sd)):.1f}" y1="{y:.1f}" '
            f'x2="{xpos(x_value + x_sd):.1f}" y2="{y:.1f}" '
            f'stroke="{colour}" opacity="0.75"/>'
        )
        scatter_marker(svg, row, x, y, colour)
        if row.kind != 'pipeline' or row in frontier:
            label = 'E5' if row.kind == 'e5_only' else row.model_label
            if row.kind == 'pipeline':
                label += f' N={row.n}'
            text(svg, x + 7, y - 7, label, size=10, colour=colour)
    draw_model_legend(svg, 1320, 155, columns=1, row_height=42, column_width=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'accuracy_vs_time.svg').write_text(
        '\n'.join(svg + ['</svg>']) + '\n', encoding='utf-8'
    )
    write_csv(
        output_dir / 'accuracy_vs_time_data.csv',
        [stat_payload(row) for row in rows],
    )
    lines = ['# Frontera de Pareto sobre las medias de cinco iteraciones', '']
    for row in frontier:
        label = 'E5' if row.kind == 'e5_only' else row.model_label
        if row.kind == 'pipeline':
            label += f', N={row.n}'
        elif row.kind == 'slm_direct':
            label += ', directo'
        lines.append(
            f'- {label}: {row.accuracy_mean:.2f}% y '
            f'{row.time_mean / total_news:.4f} s/noticia.'
        )
    (output_dir / 'pareto_frontier.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8'
    )


def generate_accuracy_vs_time_by_model(output_dir: Path) -> None:
    values, rows = base_statistics()
    total_news = values[0].total_news
    by_model = pipeline_by_model(rows)
    direct = direct_statistics(rows)
    e5 = e5_statistic(rows)
    frontier = set(pareto_frontier(scatter_rows(rows), total_news))
    width, height = 1680, 1340
    svg = svg_document(
        width, height, 'Accuracy frente a tiempo por modelo',
        'Ocho paneles con media y desviación estándar de cinco iteraciones.'
    )
    text(svg, 55, 78, 'Escala temporal local · accuracy común 0–100%', size=16)
    panel_width, panel_height = 780, 270
    for index, spec in enumerate(base.MODEL_SPECS):
        row_index, column = divmod(index, 2)
        px, py = 55 + column * 815, 115 + row_index * 297
        svg.append(
            f'<rect x="{px}" y="{py}" width="{panel_width}" '
            f'height="{panel_height}" rx="12" fill="#ffffff" '
            'stroke="#dbe3ee"/>'
        )
        text(
            svg, px + panel_width / 2, py + 27, spec.label,
            anchor='middle', size=17, colour='#0f172a', weight='700'
        )
        model_rows = by_model[spec.ollama_model]
        direct_row = direct[spec.ollama_model]
        local_rows = model_rows + [direct_row, e5]
        x_values = [row.time_mean / total_news for row in local_rows]
        x_low = max(0.0, min(x_values) * 0.9)
        x_high = max(x_values) * 1.08
        left, top, pw, ph = px + 58, py + 45, panel_width - 83, 175

        def xpos(value: float) -> float:
            return left + (value - x_low) * pw / (x_high - x_low)

        def ypos(value: float) -> float:
            return top + (100 - value) * ph / 100

        for accuracy in range(0, 101, 25):
            y = ypos(accuracy)
            svg.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" '
                f'y2="{y:.1f}" stroke="#e2e8f0"/>'
            )
            text(svg, left - 8, y + 4, str(accuracy), anchor='end', size=9)
        for fraction in (0.0, 0.5, 1.0):
            value = x_low + fraction * (x_high - x_low)
            x = xpos(value)
            text(svg, x, top + ph + 17, f'{value:.3f}', anchor='middle', size=9)
        points = [
            (xpos(row.time_mean / total_news), ypos(row.accuracy_mean))
            for row in model_rows
        ]
        polyline(svg, points, spec.colour)
        e5_y = ypos(e5.accuracy_mean)
        svg.append(
            f'<line x1="{left}" y1="{e5_y:.1f}" x2="{left + pw}" '
            f'y2="{e5_y:.1f}" stroke="#dc2626" stroke-dasharray="7 5"/>'
        )
        for row, (x, y) in zip(model_rows, points):
            error_bar(
                svg, x, ypos(row.accuracy_mean - row.accuracy_sd),
                ypos(row.accuracy_mean + row.accuracy_sd), spec.colour
            )
            time_mean = row.time_mean / total_news
            time_sd = row.time_sd / total_news
            svg.append(
                f'<line x1="{xpos(max(x_low, time_mean - time_sd)):.1f}" '
                f'y1="{y:.1f}" x2="{xpos(min(x_high, time_mean + time_sd)):.1f}" '
                f'y2="{y:.1f}" stroke="{spec.colour}" opacity="0.75"/>'
            )
            circle(svg, x, y, spec.colour, 3.0)
            if row.n in {1, 15} or row == max(model_rows, key=lambda item: item.accuracy_mean):
                text(svg, x + 5, y - 5, f'N={row.n}', size=9, colour=spec.colour)
            if row in frontier:
                circle(svg, x, y, '#d97706', 6.3, fill='none', width=1.8)
        dx, dy = xpos(direct_row.time_mean / total_news), ypos(direct_row.accuracy_mean)
        error_bar(
            svg, dx, ypos(direct_row.accuracy_mean - direct_row.accuracy_sd),
            ypos(direct_row.accuracy_mean + direct_row.accuracy_sd),
            spec.colour,
        )
        direct_time_mean = direct_row.time_mean / total_news
        direct_time_sd = direct_row.time_sd / total_news
        svg.append(
            f'<line x1="{xpos(max(x_low, direct_time_mean - direct_time_sd)):.1f}" '
            f'y1="{dy:.1f}" '
            f'x2="{xpos(min(x_high, direct_time_mean + direct_time_sd)):.1f}" '
            f'y2="{dy:.1f}" stroke="{spec.colour}" opacity="0.75"/>'
        )
        triangle(svg, dx, dy, spec.colour)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'accuracy_vs_time_by_model.svg').write_text(
        '\n'.join(svg + ['</svg>']) + '\n', encoding='utf-8'
    )
    write_csv(
        output_dir / 'accuracy_vs_time_by_model_data.csv',
        [stat_payload(row) for row in rows],
    )


def recall_rows() -> tuple[list[dict[str, Any]], dict[int, dict[str, float]]]:
    def extractor(report: dict[str, Any], path: Path) -> dict[str, float]:
        metrics = base.mapping(report.get('metrics'), 'metrics', path)
        return {
            'final_accuracy_percent': base.number(
                metrics.get('final_accuracy_percent'),
                'metrics.final_accuracy_percent', path
            ),
            'selection_accuracy_percent': base.number(
                metrics.get('conditional_selection_accuracy_percent'),
                'metrics.conditional_selection_accuracy_percent', path
            ),
        }

    model_rows = aggregate_hybrid_metrics(extractor)
    recall_values: dict[int, list[float]] = defaultdict(list)
    for iteration in base.EXPECTED_ITERATIONS:
        path = (
            base.ITERATIONS_ROOT / iteration / 'e5_results'
            / 'e5_top_n_results.json'
        )
        report = base.load_json(path)
        metrics = base.mapping(report.get('metrics'), 'metrics', path)
        block = base.mapping(
            metrics.get('candidate_recall_at_n'),
            'metrics.candidate_recall_at_n', path
        )
        for n in base.EXPECTED_N:
            item = base.mapping(block.get(str(n)), f'Recall@{n}', path)
            recall_values[n].append(
                base.number(
                    item.get('candidate_recall_percent'),
                    f'Recall@{n}.candidate_recall_percent', path
                )
            )
    recall = {n: summarize(values) for n, values in recall_values.items()}
    return model_rows, recall


def generate_recall_and_selection(output_dir: Path) -> None:
    rows, recall = recall_rows()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row['model']].append(row)
    width, height = 1660, 980
    svg = svg_document(
        width, height, 'Recall y accuracy de selección',
        'Recall E5, accuracy final y accuracy condicionada con cinco iteraciones.'
    )
    text(
        svg, 55, 78,
        'Curvas: medias · bandas: mínimo–máximo · barras: ±1 DE', size=16
    )
    draw_model_legend(svg, 80, 112, columns=4, row_height=27, column_width=380)

    def draw_metric_panel(
        top: float, title_value: str, metric_prefix: str,
        include_recall: bool
    ) -> None:
        left, pw, ph = 100, 1200, 310
        text(svg, left, top - 22, title_value, size=20, colour='#0f172a', weight='700')
        xpos, ypos = draw_axes(
            svg, left, top, pw, ph, y_min=0, y_max=100, y_step=20,
            x_label='N candidatos'
        )
        if include_recall:
            recall_points = [(xpos(n), ypos(recall[n]['mean'])) for n in base.EXPECTED_N]
            polyline(svg, recall_points, '#0f172a', width=2.5, dash='8 5')
            text(svg, 1330, top + 25, 'Recall@N E5', size=13, colour='#0f172a')
        for spec in base.MODEL_SPECS:
            model_rows = sorted(grouped[spec.ollama_model], key=lambda row: row['n'])
            means = [
                (xpos(row['n']), ypos(row[f'{metric_prefix}_mean']))
                for row in model_rows
            ]
            upper = [
                (xpos(row['n']), ypos(row[f'{metric_prefix}_max']))
                for row in model_rows
            ]
            lower = [
                (xpos(row['n']), ypos(row[f'{metric_prefix}_min']))
                for row in model_rows
            ]
            range_band(svg, upper, lower, spec.colour, 0.07)
            polyline(svg, means, spec.colour)
            for row, (x, y) in zip(model_rows, means):
                if row['n'] % 2:
                    error_bar(
                        svg, x,
                        ypos(row[f'{metric_prefix}_mean'] - row[f'{metric_prefix}_sd']),
                        ypos(row[f'{metric_prefix}_mean'] + row[f'{metric_prefix}_sd']),
                        spec.colour,
                    )
                circle(svg, x, y, spec.colour, 2.5)

    draw_metric_panel(205, 'Recall de E5 y accuracy final del pipeline', 'final_accuracy_percent', True)
    draw_metric_panel(625, 'Accuracy condicionada a que la etiqueta correcta esté en Top-N', 'selection_accuracy_percent', False)
    svg.append('</svg>')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'recall_and_selection_accuracy.svg').write_text(
        '\n'.join(svg) + '\n', encoding='utf-8'
    )
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        csv_rows.append({'recall_mean': recall[row['n']]['mean'], **row})
    write_csv(output_dir / 'recall_and_selection_accuracy_data.csv', csv_rows)


def correction_rows() -> list[dict[str, Any]]:
    def extractor(report: dict[str, Any], path: Path) -> dict[str, float]:
        metrics = base.mapping(report.get('metrics'), 'metrics', path)
        dataset = base.mapping(report.get('dataset'), 'dataset', path)
        total = base.integer(dataset.get('total_news'), 'dataset.total_news', path)
        predictions = report.get('predictions')
        if not isinstance(predictions, list) or len(predictions) != total:
            raise SystemExit(f'{path}: predictions incompleto.')
        wrong_to_wrong = 0
        correct_to_correct = 0
        unchanged = 0
        for prediction in predictions:
            if not isinstance(prediction, dict):
                raise SystemExit(f'{path}: predicción no válida.')
            gold = prediction.get('topics_correctos')
            e5_topic = prediction.get('e5_top_1_topic')
            final_topic = prediction.get('final_decision')
            if not isinstance(gold, list):
                raise SystemExit(f'{path}: topics_correctos no válido.')
            if final_topic == e5_topic:
                unchanged += 1
                continue
            e5_correct = e5_topic in gold
            final_correct = final_topic in gold
            if not e5_correct and not final_correct:
                wrong_to_wrong += 1
            elif e5_correct and final_correct:
                correct_to_correct += 1
        corrections = base.integer(
            metrics.get('slm_helped_vs_e5_top_1'),
            'metrics.slm_helped_vs_e5_top_1', path
        )
        harms = base.integer(
            metrics.get('slm_harmed_vs_e5_top_1'),
            'metrics.slm_harmed_vs_e5_top_1', path
        )
        changes = base.integer(
            metrics.get('slm_valid_topic_changes'),
            'metrics.slm_valid_topic_changes', path
        )
        if corrections + harms + wrong_to_wrong + correct_to_correct != changes:
            raise SystemExit(f'{path}: los cambios de tema no cuadran.')
        return {
            'corrections_percent': 100 * corrections / total,
            'harms_percent': 100 * harms / total,
            'net_percent': 100 * (corrections - harms) / total,
            'wrong_to_wrong_percent': 100 * wrong_to_wrong / total,
            'correct_to_correct_percent': 100 * correct_to_correct / total,
            'unchanged_percent': 100 * unchanged / total,
        }

    return aggregate_hybrid_metrics(extractor)


def generate_corrections_and_harms(output_dir: Path) -> None:
    rows = correction_rows()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row['model']].append(row)
    max_value = max(
        max(row['corrections_percent_max'], row['harms_percent_max'])
        for row in rows
    ) * 1.12
    axis_max, axis_step = nice_axis(max_value, 4)
    width, height = 1680, 1340
    svg = svg_document(
        width, height, 'Correcciones y errores introducidos por el SLM',
        'Barras divergentes agregadas sobre cinco iteraciones.'
    )
    text(
        svg, 55, 78,
        'Verde: errores corregidos · rojo: aciertos estropeados · negro: balance neto',
        size=16,
    )
    panel_width, panel_height = 780, 270
    for index, spec in enumerate(base.MODEL_SPECS):
        row_index, column = divmod(index, 2)
        px, py = 55 + column * 815, 115 + row_index * 297
        svg.append(
            f'<rect x="{px}" y="{py}" width="{panel_width}" '
            f'height="{panel_height}" rx="12" fill="#ffffff" '
            'stroke="#dbe3ee"/>'
        )
        text(
            svg, px + panel_width / 2, py + 27, spec.label,
            anchor='middle', size=17, colour='#0f172a', weight='700'
        )
        left, top, pw, ph = px + 55, py + 47, panel_width - 80, 170

        def xpos(n: int) -> float:
            return left + (n - 1) * pw / 14

        def ypos(value: float) -> float:
            return top + (axis_max - value) * ph / (2 * axis_max)

        zero_y = ypos(0)
        svg.append(
            f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + pw}" '
            f'y2="{zero_y:.1f}" stroke="#64748b" stroke-width="1.2"/>'
        )
        for tick in (-axis_max, 0, axis_max):
            text(svg, left - 8, ypos(tick) + 4, f'{tick:g}', anchor='end', size=9)
        model_rows = sorted(grouped[spec.ollama_model], key=lambda row: row['n'])
        net_points: list[tuple[float, float]] = []
        for row in model_rows:
            x = xpos(row['n'])
            correction = row['corrections_percent_mean']
            harm = row['harms_percent_mean']
            bar_width = 12
            svg.append(
                f'<rect x="{x - bar_width / 2:.1f}" y="{ypos(correction):.1f}" '
                f'width="{bar_width}" height="{zero_y - ypos(correction):.1f}" '
                'fill="#16a34a" opacity="0.8"/>'
            )
            svg.append(
                f'<rect x="{x - bar_width / 2:.1f}" y="{zero_y:.1f}" '
                f'width="{bar_width}" height="{ypos(-harm) - zero_y:.1f}" '
                'fill="#dc2626" opacity="0.8"/>'
            )
            error_bar(
                svg, x,
                ypos(correction - row['corrections_percent_sd']),
                ypos(correction + row['corrections_percent_sd']), '#166534'
            )
            error_bar(
                svg, x,
                ypos(-(harm - row['harms_percent_sd'])),
                ypos(-(harm + row['harms_percent_sd'])), '#991b1b'
            )
            net_points.append((x, ypos(row['net_percent_mean'])))
            text(svg, x, top + ph + 15, str(row['n']), anchor='middle', size=8)
        polyline(svg, net_points, '#111827', width=1.5)
        for x, y in net_points:
            circle(svg, x, y, '#111827', 2.7)
    svg.append('</svg>')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'corrections_and_harms.svg').write_text(
        '\n'.join(svg) + '\n', encoding='utf-8'
    )
    write_csv(output_dir / 'corrections_and_harms_data.csv', rows)
    lines = [
        '# Cambios que no modifican la accuracy', '',
        '| Modelo | N | Incorrecta → incorrecta | Correcta → correcta |',
        '|---|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {row['n']} | "
            f"{row['wrong_to_wrong_percent_mean']:.2f}% ± "
            f"{row['wrong_to_wrong_percent_sd']:.2f} | "
            f"{row['correct_to_correct_percent_mean']:.2f}% ± "
            f"{row['correct_to_correct_percent_sd']:.2f} |"
        )
    (output_dir / 'neutral_changes_table.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8'
    )


GENERATORS: dict[str, Callable[[Path], None]] = {
    'accuracy_vs_candidates': generate_accuracy_vs_candidates,
    'accuracy_vs_candidates_by_family': (
        generate_accuracy_vs_candidates_by_family
    ),
    'accuracy_vs_time': generate_accuracy_vs_time,
    'accuracy_vs_time_by_model': generate_accuracy_vs_time_by_model,
    'corrections_and_harms': generate_corrections_and_harms,
    'n_vs_time': generate_n_vs_time,
    'recall_and_selection_accuracy': generate_recall_and_selection,
}


def generate(name: str, output_dir: Path) -> None:
    try:
        generator = GENERATORS[name]
    except KeyError as exc:
        raise SystemExit(f'Gráfica desconocida: {name}') from exc
    generator(output_dir)
    print(f'Generada {name} en {output_dir}')
