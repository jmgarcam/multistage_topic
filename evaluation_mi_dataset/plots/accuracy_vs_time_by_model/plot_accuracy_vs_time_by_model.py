#!/usr/bin/env python3
"""Muestra accuracy frente a tiempo en pequeños múltiples por modelo."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape


PLOT_DIR = Path(__file__).resolve().parent
SOURCE_PLOT_DIR = PLOT_DIR.parent / 'accuracy_vs_time'
sys.path.insert(0, str(SOURCE_PLOT_DIR))

from plot_accuracy_vs_time import (  # noqa: E402
    MODEL_SPECS,
    ConfigPoint,
    load_all_points,
    pareto_keys,
)


OUTPUT_SVG = PLOT_DIR / 'accuracy_vs_time_by_model.svg'
OUTPUT_CSV = PLOT_DIR / 'accuracy_vs_time_by_model_data.csv'


def nice_number(value: float) -> float:
    """Redondea hacia arriba a 1, 2, 2.5, 5 o 10 por potencia de diez."""
    if value <= 0:
        return 0.01
    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)
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
    return nice_fraction * (10**exponent)


def local_time_axis(points: list[ConfigPoint]) -> tuple[float, float, float]:
    values = [point.total_seconds_per_news for point in points]
    low = min(values)
    high = max(values)
    spread = high - low
    padding = max(spread * 0.10, high * 0.008, 0.001)
    step = nice_number((spread + 2 * padding) / 5)
    axis_low = max(0.0, math.floor((low - padding) / step) * step)
    axis_high = math.ceil((high + padding) / step) * step
    if axis_high <= axis_low:
        axis_high = axis_low + 5 * step
    return axis_low, axis_high, step


def x_position(
    value: float, left: float, width: float, low: float, high: float
) -> float:
    return left + (value - low) * width / (high - low)


def y_position(value: float, top: float, height: float) -> float:
    return top + (100 - value) * height / 100


def relevant_pipeline_keys(
    points: list[ConfigPoint], pareto: set[str]
) -> set[str]:
    pipelines = [point for point in points if point.kind == 'pipeline']
    labels = {point.key for point in pipelines if point.key in pareto}
    if not pipelines:
        return labels
    labels.add(min(pipelines, key=lambda point: point.n or 0).key)
    labels.add(max(pipelines, key=lambda point: point.n or 0).key)
    best = max(
        pipelines,
        key=lambda point: (
            point.accuracy_percent,
            -point.total_seconds_per_news,
        ),
    )
    labels.add(best.key)
    return labels


def draw_pipeline_point(
    svg: list[str], x: float, y: float, colour: str
) -> None:
    svg.append(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" '
        f'fill="{colour}" stroke="#ffffff" stroke-width="1.2"/>'
    )


def draw_direct_point(
    svg: list[str], x: float, y: float, colour: str
) -> None:
    svg.append(
        f'<polygon points="{x:.1f},{y - 8:.1f} '
        f'{x - 7.5:.1f},{y + 6.5:.1f} {x + 7.5:.1f},{y + 6.5:.1f}" '
        f'fill="{colour}" stroke="#ffffff" stroke-width="1.4"/>'
    )


def write_csv(
    points: list[ConfigPoint],
    pareto: set[str],
    labelled: set[str],
    axes: dict[str, tuple[float, float, float]],
) -> None:
    with OUTPUT_CSV.open('w', encoding='utf-8', newline='') as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=(
                'configuration',
                'model',
                'type',
                'n',
                'accuracy_percent',
                'total_seconds_per_news',
                'pareto_nondominated',
                'labelled_in_plot',
                'panel_time_axis_min',
                'panel_time_axis_max',
            ),
        )
        writer.writeheader()
        for point in points:
            axis = axes.get(point.ollama_model)
            writer.writerow(
                {
                    'configuration': point.key,
                    'model': point.model_label,
                    'type': point.kind,
                    'n': point.n if point.n is not None else '',
                    'accuracy_percent': round(point.accuracy_percent, 6),
                    'total_seconds_per_news': round(
                        point.total_seconds_per_news, 9
                    ),
                    'pareto_nondominated': point.key in pareto,
                    'labelled_in_plot': point.key in labelled,
                    'panel_time_axis_min': round(axis[0], 6) if axis else '',
                    'panel_time_axis_max': round(axis[1], 6) if axis else '',
                }
            )


def generate_svg(
    points: list[ConfigPoint], pareto: set[str]
) -> tuple[str, set[str], dict[str, tuple[float, float, float]]]:
    width = 1740
    height = 1185
    outer_left = 54
    outer_right = 42
    gap_x = 34
    gap_y = 25
    columns = 3
    panel_top_start = 155
    panel_height = 470
    panel_width = (
        width - outer_left - outer_right - (columns - 1) * gap_x
    ) / columns
    plot_left_offset = 69
    plot_right_padding = 24
    plot_top_offset = 57
    plot_height = 330

    e5 = next(point for point in points if point.kind == 'e5_only')
    labelled: set[str] = {e5.key}
    axes: dict[str, tuple[float, float, float]] = {}

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title description">'
        ),
        '<title id="title">Accuracy frente a tiempo por modelo</title>',
        (
            '<desc id="description">Seis paneles separan los modelos para '
            'mostrar con claridad sus pipelines, ejecución directa y puntos '
            'globalmente no dominados.</desc>'
        ),
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        (
            '<text x="54" y="46" fill="#0f172a" '
            'font-family="Arial, Helvetica, sans-serif" font-size="32" '
            'font-weight="700">Accuracy frente a tiempo · detalle por modelo</text>'
        ),
        (
            '<text x="54" y="76" fill="#475569" '
            'font-family="Arial, Helvetica, sans-serif" font-size="17">'
            'Escala temporal propia en cada panel · escala de accuracy común '
            '(0–100%)</text>'
        ),
        '<line x1="500" y1="112" x2="548" y2="112" stroke="#64748b" stroke-width="2.5"/>',
        '<circle cx="524" cy="112" r="5" fill="#64748b"/>',
        '<text x="558" y="118" fill="#334155" font-family="Arial, Helvetica, sans-serif" font-size="15">Pipeline E5 → SLM</text>',
        '<polygon points="790,104 782,119 798,119" fill="#64748b"/>',
        '<text x="808" y="118" fill="#334155" font-family="Arial, Helvetica, sans-serif" font-size="15">SLM directo</text>',
        '<circle cx="1014" cy="112" r="10" fill="none" stroke="#ca8a04" stroke-width="3"/>',
        '<text x="1032" y="118" fill="#334155" font-family="Arial, Helvetica, sans-serif" font-size="15">No dominada global</text>',
        '<line x1="1267" y1="112" x2="1315" y2="112" stroke="#dc2626" stroke-width="2.3" stroke-dasharray="8 6"/>',
        '<text x="1325" y="118" fill="#334155" font-family="Arial, Helvetica, sans-serif" font-size="15">E5 Top-1</text>',
    ]

    for panel_index, spec in enumerate(MODEL_SPECS):
        model_points = [
            point for point in points if point.ollama_model == spec.ollama_model
        ]
        pipelines = sorted(
            (point for point in model_points if point.kind == 'pipeline'),
            key=lambda point: point.n or 0,
        )
        direct = next(
            (point for point in model_points if point.kind == 'slm_direct'),
            None,
        )
        if not model_points:
            continue

        axis_low, axis_high, axis_step = local_time_axis(model_points)
        axes[spec.ollama_model] = (axis_low, axis_high, axis_step)
        labelled_for_model = relevant_pipeline_keys(model_points, pareto)
        labelled.update(labelled_for_model)
        if direct is not None:
            labelled.add(direct.key)

        row, column = divmod(panel_index, columns)
        panel_x = outer_left + column * (panel_width + gap_x)
        panel_y = panel_top_start + row * (panel_height + gap_y)
        plot_left = panel_x + plot_left_offset
        plot_top = panel_y + plot_top_offset
        plot_width = panel_width - plot_left_offset - plot_right_padding
        plot_bottom = plot_top + plot_height

        svg.append(
            f'<rect x="{panel_x:.1f}" y="{panel_y:.1f}" '
            f'width="{panel_width:.1f}" height="{panel_height}" rx="12" '
            'fill="#ffffff" stroke="#dbe3ee"/>'
        )
        svg.append(
            f'<circle cx="{panel_x + 19:.1f}" cy="{panel_y + 26:.1f}" '
            f'r="6" fill="{spec.colour}"/>'
        )
        svg.append(
            f'<text x="{panel_x + 32:.1f}" y="{panel_y + 32:.1f}" '
            'fill="#0f172a" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="21" font-weight="700">{escape(spec.label)}</text>'
        )

        for accuracy in range(0, 101, 20):
            y = y_position(accuracy, plot_top, plot_height)
            svg.append(
                f'<line x1="{plot_left:.1f}" y1="{y:.1f}" '
                f'x2="{plot_left + plot_width:.1f}" y2="{y:.1f}" '
                'stroke="#e2e8f0" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{plot_left - 10:.1f}" y="{y + 5:.1f}" '
                'text-anchor="end" fill="#64748b" '
                'font-family="Arial, Helvetica, sans-serif" '
                f'font-size="12">{accuracy}</text>'
            )

        tick_count = int(round((axis_high - axis_low) / axis_step))
        for tick_index in range(tick_count + 1):
            value = axis_low + tick_index * axis_step
            x = x_position(value, plot_left, plot_width, axis_low, axis_high)
            svg.append(
                f'<line x1="{x:.1f}" y1="{plot_top:.1f}" '
                f'x2="{x:.1f}" y2="{plot_bottom:.1f}" '
                'stroke="#eef2f7" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{x:.1f}" y="{plot_bottom + 20:.1f}" '
                'text-anchor="middle" fill="#64748b" '
                'font-family="Arial, Helvetica, sans-serif" font-size="11">'
                f'{value:.3f}</text>'
            )

        e5_y = y_position(e5.accuracy_percent, plot_top, plot_height)
        svg.append(
            f'<line x1="{plot_left:.1f}" y1="{e5_y:.1f}" '
            f'x2="{plot_left + plot_width:.1f}" y2="{e5_y:.1f}" '
            'stroke="#dc2626" stroke-width="2.0" stroke-dasharray="8 6"/>'
        )
        svg.append(
            f'<text x="{plot_left + plot_width - 4:.1f}" y="{e5_y - 7:.1f}" '
            'text-anchor="end" fill="#b91c1c" '
            'font-family="Arial, Helvetica, sans-serif" font-size="11">'
            f'E5 {e5.accuracy_percent:.2f}%</text>'
        )

        if len(pipelines) > 1:
            trajectory = ' '.join(
                f'{x_position(point.total_seconds_per_news, plot_left, plot_width, axis_low, axis_high):.1f},'
                f'{y_position(point.accuracy_percent, plot_top, plot_height):.1f}'
                for point in pipelines
            )
            svg.append(
                f'<polyline points="{trajectory}" fill="none" '
                f'stroke="{spec.colour}" stroke-width="2.2" opacity="0.68" '
                'stroke-linejoin="round" stroke-linecap="round"/>'
            )

        for point in pipelines:
            x = x_position(
                point.total_seconds_per_news,
                plot_left,
                plot_width,
                axis_low,
                axis_high,
            )
            y = y_position(point.accuracy_percent, plot_top, plot_height)
            draw_pipeline_point(svg, x, y, spec.colour)
            if point.key in pareto:
                svg.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9.2" '
                    'fill="none" stroke="#ca8a04" stroke-width="2.5"/>'
                )
            if point.key in labelled_for_model:
                dy = -9 if (point.n or 0) % 2 else 17
                svg.append(
                    f'<text x="{x + 7:.1f}" y="{y + dy:.1f}" '
                    'fill="#1e293b" font-family="Arial, Helvetica, sans-serif" '
                    f'font-size="11" font-weight="600">N={point.n}</text>'
                )

        if direct is not None:
            direct_x = x_position(
                direct.total_seconds_per_news,
                plot_left,
                plot_width,
                axis_low,
                axis_high,
            )
            direct_y = y_position(
                direct.accuracy_percent, plot_top, plot_height
            )
            draw_direct_point(svg, direct_x, direct_y, spec.colour)
            if direct.key in pareto:
                svg.append(
                    f'<circle cx="{direct_x:.1f}" cy="{direct_y:.1f}" '
                    'r="10.5" fill="none" stroke="#ca8a04" '
                    'stroke-width="2.5"/>'
                )
            svg.append(
                f'<text x="{direct_x - 9:.1f}" y="{direct_y + 20:.1f}" '
                'text-anchor="end" fill="#334155" '
                'font-family="Arial, Helvetica, sans-serif" font-size="11">'
                'Directo</text>'
            )

        svg.extend(
            [
                (
                    f'<line x1="{plot_left:.1f}" y1="{plot_top:.1f}" '
                    f'x2="{plot_left:.1f}" y2="{plot_bottom:.1f}" '
                    'stroke="#64748b"/>'
                ),
                (
                    f'<line x1="{plot_left:.1f}" y1="{plot_bottom:.1f}" '
                    f'x2="{plot_left + plot_width:.1f}" '
                    f'y2="{plot_bottom:.1f}" stroke="#64748b"/>'
                ),
                (
                    f'<text x="{plot_left + plot_width / 2:.1f}" '
                    f'y="{plot_bottom + 46:.1f}" text-anchor="middle" '
                    'fill="#334155" font-family="Arial, Helvetica, sans-serif" '
                    'font-size="13">Tiempo medio total/noticia (s)</text>'
                ),
            ]
        )

        if column == 0:
            centre_y = plot_top + plot_height / 2
            svg.append(
                f'<text x="18" y="{centre_y:.1f}" '
                f'transform="rotate(-90 18 {centre_y:.1f})" '
                'text-anchor="middle" fill="#334155" '
                'font-family="Arial, Helvetica, sans-serif" '
                'font-size="15">Accuracy (%)</text>'
            )

    svg.append(
        f'<text x="{width - 48}" y="{height - 22}" text-anchor="end" '
        'fill="#64748b" font-family="Arial, Helvetica, sans-serif" '
        'font-size="12">Las escalas X locales mejoran la lectura; compare '
        'los valores numéricos de los ejes para contrastar velocidades.</text>'
    )
    svg.append('</svg>')
    return '\n'.join(svg) + '\n', labelled, axes


def main() -> None:
    points, _availability = load_all_points()
    pareto = pareto_keys(points)
    svg, labelled, axes = generate_svg(points, pareto)
    write_csv(points, pareto, labelled, axes)
    OUTPUT_SVG.write_text(svg, encoding='utf-8')

    print(f'Modelos: {len(MODEL_SPECS)}')
    print(f'Configuraciones: {len(points)}')
    print(f'No dominadas: {len(pareto)}')
    print(f'Gráfica: {OUTPUT_SVG.resolve()}')
    print(f'Datos: {OUTPUT_CSV.resolve()}')


if __name__ == '__main__':
    main()
