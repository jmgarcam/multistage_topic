#!/usr/bin/env python3
"""Grafica el número de candidatos frente al tiempo medio por noticia."""

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
    ConfigPoint,
    load_all_points,
)


OUTPUT_SVG = PLOT_DIR / 'n_vs_time.svg'
OUTPUT_CSV = PLOT_DIR / 'n_vs_time_data.csv'


def nice_time_axis(maximum: float) -> tuple[float, float]:
    target = maximum * 1.08 / 6
    exponent = math.floor(math.log10(target)) if target > 0 else 0
    magnitude = 10**exponent
    fraction = target / magnitude
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
    limit = math.ceil(maximum * 1.08 / step) * step
    return limit, step


def x_position(n: int, left: float, width: float) -> float:
    return left + (n - 1) * width / 14


def y_position(
    value: float, top: float, height: float, maximum: float
) -> float:
    return top + (maximum - value) * height / maximum


def model_family(point: ConfigPoint) -> str | None:
    if point.model_label.startswith('Llama'):
        return 'Llama'
    if point.model_label.startswith('Qwen'):
        return 'Qwen'
    return None


def total_execution_seconds(point: ConfigPoint) -> float:
    return point.total_seconds_per_news * point.total_news


def write_csv(points: list[ConfigPoint]) -> None:
    with OUTPUT_CSV.open('w', encoding='utf-8', newline='') as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=(
                'configuration',
                'family',
                'model',
                'type',
                'n',
                'candidate_topics_shown',
                'total_execution_seconds',
                'total_seconds_per_news',
                'e5_seconds_per_news',
                'slm_two_prompts_seconds_per_news',
                'source',
            ),
        )
        writer.writeheader()
        for point in points:
            family = model_family(point)
            if point.kind == 'pipeline':
                shown: int | str = point.n if point.n is not None else ''
            elif point.kind == 'slm_direct':
                shown = 15
            else:
                shown = ''
            writer.writerow(
                {
                    'configuration': point.key,
                    'family': family or '',
                    'model': point.model_label,
                    'type': point.kind,
                    'n': point.n if point.n is not None else '',
                    'candidate_topics_shown': shown,
                    'total_execution_seconds': round(
                        total_execution_seconds(point), 6
                    ),
                    'total_seconds_per_news': round(
                        point.total_seconds_per_news, 9
                    ),
                    'e5_seconds_per_news': round(
                        point.e5_seconds_per_news, 9
                    ),
                    'slm_two_prompts_seconds_per_news': round(
                        point.slm_seconds_per_news, 9
                    ),
                    'source': point.source,
                }
            )


def draw_direct_triangle(
    svg: list[str], x: float, y: float, colour: str
) -> None:
    svg.append(
        f'<polygon points="{x:.1f},{y - 9:.1f} '
        f'{x - 8.5:.1f},{y + 7:.1f} {x + 8.5:.1f},{y + 7:.1f}" '
        f'fill="{colour}" stroke="#ffffff" stroke-width="1.5"/>'
    )


def generate_svg(points: list[ConfigPoint]) -> str:
    width = 1640
    height = 760
    outer_left = 55
    outer_right = 42
    gap = 42
    panel_width = (width - outer_left - outer_right - gap) / 2
    panel_top = 165
    panel_height = 545
    plot_left_offset = 68
    plot_right_padding = 25
    plot_top = 205
    plot_height = 390
    plot_bottom = plot_top + plot_height

    plotted = [point for point in points if model_family(point) is not None]
    e5 = next(point for point in points if point.kind == 'e5_only')
    max_time, time_step = nice_time_axis(
        max(total_execution_seconds(point) for point in plotted)
    )

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title description">'
        ),
        '<title id="title">Tiempo total frente al número de candidatos</title>',
        (
            '<desc id="description">Dos paneles comparan cómo evoluciona el '
            'tiempo total de procesar todo el dataset al aumentar N para '
            'Llama y Qwen, con N en el '
            'eje horizontal.</desc>'
        ),
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        (
            '<text x="55" y="47" fill="#0f172a" '
            'font-family="Arial, Helvetica, sans-serif" font-size="33" '
            'font-weight="700">Tiempo total frente al número de candidatos</text>'
        ),
        (
            '<text x="55" y="78" fill="#475569" '
            'font-family="Arial, Helvetica, sans-serif" font-size="17">'
            f'Tiempo acumulado para {e5.total_news} noticias: E5 + dos '
            'llamadas al SLM por noticia</text>'
        ),
        '<line x1="500" y1="113" x2="548" y2="113" stroke="#64748b" stroke-width="2.5"/>',
        '<circle cx="524" cy="113" r="5" fill="#64748b"/>',
        '<text x="558" y="119" fill="#334155" font-family="Arial, Helvetica, sans-serif" font-size="15">Pipeline E5 → SLM</text>',
        '<polygon points="790,104 781,120 799,120" fill="#64748b"/>',
        '<text x="809" y="119" fill="#334155" font-family="Arial, Helvetica, sans-serif" font-size="15">SLM directo (15 temas)</text>',
        '<line x1="1065" y1="113" x2="1113" y2="113" stroke="#0f172a" stroke-width="2" stroke-dasharray="7 6"/>',
        '<text x="1123" y="119" fill="#334155" font-family="Arial, Helvetica, sans-serif" font-size="15">Tiempo de E5 solo</text>',
    ]

    for panel_index, family in enumerate(('Llama', 'Qwen')):
        panel_x = outer_left + panel_index * (panel_width + gap)
        plot_left = panel_x + plot_left_offset
        plot_width = panel_width - plot_left_offset - plot_right_padding

        svg.append(
            f'<rect x="{panel_x:.1f}" y="{panel_top}" '
            f'width="{panel_width:.1f}" height="{panel_height}" rx="12" '
            'fill="#ffffff" stroke="#dbe3ee"/>'
        )
        svg.append(
            f'<text x="{panel_x + panel_width / 2:.1f}" y="194" '
            'text-anchor="middle" fill="#0f172a" '
            'font-family="Arial, Helvetica, sans-serif" font-size="22" '
            f'font-weight="700">{family}</text>'
        )

        tick_count = int(round(max_time / time_step))
        for tick_index in range(tick_count + 1):
            value = tick_index * time_step
            y = y_position(value, plot_top, plot_height, max_time)
            svg.append(
                f'<line x1="{plot_left:.1f}" y1="{y:.1f}" '
                f'x2="{plot_left + plot_width:.1f}" y2="{y:.1f}" '
                'stroke="#e2e8f0" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{plot_left - 10:.1f}" y="{y + 4:.1f}" '
                'text-anchor="end" fill="#64748b" '
                'font-family="Arial, Helvetica, sans-serif" '
                f'font-size="11">{value:.2f}</text>'
            )

        for n in range(1, 16):
            x = x_position(n, plot_left, plot_width)
            svg.append(
                f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" '
                f'y2="{plot_bottom}" stroke="#edf2f7" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{x:.1f}" y="{plot_bottom + 22}" '
                'text-anchor="middle" fill="#64748b" '
                'font-family="Arial, Helvetica, sans-serif" font-size="12">'
                f'{n}</text>'
            )

        e5_y = y_position(
            total_execution_seconds(e5),
            plot_top,
            plot_height,
            max_time,
        )
        svg.append(
            f'<line x1="{plot_left:.1f}" y1="{e5_y:.1f}" '
            f'x2="{plot_left + plot_width:.1f}" y2="{e5_y:.1f}" '
            'stroke="#0f172a" '
            'stroke-width="2" stroke-dasharray="7 6"/>'
        )

        family_points = [
            point for point in plotted if model_family(point) == family
        ]
        model_labels = list(
            dict.fromkeys(point.model_label for point in family_points)
        )
        for model_index, model_label in enumerate(model_labels):
            model_points = [
                point for point in family_points if point.model_label == model_label
            ]
            colour = model_points[0].colour
            pipelines = sorted(
                (point for point in model_points if point.kind == 'pipeline'),
                key=lambda point: point.n or 0,
            )
            direct = next(
                (point for point in model_points if point.kind == 'slm_direct'),
                None,
            )
            trajectory = ' '.join(
                f'{x_position(point.n or 1, plot_left, plot_width):.1f},'
                f'{y_position(total_execution_seconds(point), plot_top, plot_height, max_time):.1f}'
                for point in pipelines
            )
            svg.append(
                f'<polyline points="{trajectory}" fill="none" '
                f'stroke="{colour}" stroke-width="2.5" opacity="0.82" '
                'stroke-linejoin="round" stroke-linecap="round"/>'
            )
            for point in pipelines:
                x = x_position(point.n or 1, plot_left, plot_width)
                y = y_position(
                    total_execution_seconds(point),
                    plot_top,
                    plot_height,
                    max_time,
                )
                svg.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" '
                    f'fill="{colour}" stroke="#ffffff" stroke-width="1.2"/>'
                )
            if direct is not None:
                x = x_position(15, plot_left, plot_width)
                y = y_position(
                    total_execution_seconds(direct),
                    plot_top,
                    plot_height,
                    max_time,
                )
                draw_direct_triangle(svg, x, y, colour)

            legend_x = panel_x + 78 + model_index * 190
            svg.append(
                f'<line x1="{legend_x:.1f}" y1="641" '
                f'x2="{legend_x + 31:.1f}" y2="641" '
                f'stroke="{colour}" stroke-width="2.5"/>'
            )
            svg.append(
                f'<text x="{legend_x + 39:.1f}" y="646" fill="#334155" '
                'font-family="Arial, Helvetica, sans-serif" font-size="12">'
                f'{escape(model_label.replace(family + " ", ""))}</text>'
            )

        svg.extend(
            [
                (
                    f'<line x1="{plot_left:.1f}" y1="{plot_top}" '
                    f'x2="{plot_left:.1f}" y2="{plot_bottom}" '
                    'stroke="#64748b"/>'
                ),
                (
                    f'<line x1="{plot_left:.1f}" y1="{plot_bottom}" '
                    f'x2="{plot_left + plot_width:.1f}" y2="{plot_bottom}" '
                    'stroke="#64748b"/>'
                ),
                (
                    f'<text x="{plot_left + plot_width / 2:.1f}" y="690" '
                    'text-anchor="middle" fill="#334155" '
                    'font-family="Arial, Helvetica, sans-serif" font-size="15">'
                    'Número de candidatos (N)</text>'
                ),
            ]
        )

        if panel_index == 0:
            centre_y = plot_top + plot_height / 2
            svg.append(
                f'<text x="18" y="{centre_y:.1f}" '
                f'transform="rotate(-90 18 {centre_y:.1f})" '
                'text-anchor="middle" fill="#334155" '
                'font-family="Arial, Helvetica, sans-serif" '
                'font-size="16">Tiempo total de ejecución (s)</text>'
            )

    svg.append(
        f'<text x="{width - 44}" y="{height - 22}" text-anchor="end" '
        'fill="#64748b" font-family="Arial, Helvetica, sans-serif" '
        'font-size="12">Escalas compartidas en ambos paneles · El punto '
        'directo se sitúa en N=15 porque recibe el catálogo completo.</text>'
    )
    svg.append('</svg>')
    return '\n'.join(svg) + '\n'


def main() -> None:
    points, _availability = load_all_points()
    write_csv(points)
    OUTPUT_SVG.write_text(generate_svg(points), encoding='utf-8')
    print('Eje X: N')
    print(f'Eje Y: tiempo total para {points[0].total_news} noticias')
    print(f'Configuraciones: {len(points)}')
    print(f'Gráfica: {OUTPUT_SVG.resolve()}')
    print(f'Datos: {OUTPUT_CSV.resolve()}')


if __name__ == '__main__':
    main()
