#!/usr/bin/env python3
"""Genera una gráfica SVG con el número de noticias por categoría."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_DIR / 'tagged_data' / 'tagged_data_no_none.json'
DEFAULT_TOPICS = PROJECT_DIR / 'temas.json'
DEFAULT_OUTPUT = PROJECT_DIR / 'tagged_data' / 'news_by_category.svg'
NONE_TOPIC = 'ninguno'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Cuenta las etiquetas de las noticias y genera una gráfica de '
            'barras horizontal en formato SVG.'
        )
    )
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--topics', type=Path, default=DEFAULT_TOPICS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        with path.open('r', encoding='utf-8') as file_handle:
            return json.load(file_handle)
    except FileNotFoundError as exc:
        raise SystemExit(f'No se encontró el archivo: {path}') from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f'JSON no válido en {path}: {exc}') from exc


def count_topics(
    records: Any, topic_names: list[str]
) -> tuple[Counter[str], int, int]:
    if not isinstance(records, list):
        raise SystemExit('El archivo de entrada debe contener una lista JSON.')

    allowed = set(topic_names)
    counts: Counter[str] = Counter({topic: 0 for topic in topic_names})
    labelled_records = 0
    multi_topic_records = 0

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SystemExit(f'Elemento {index}: se esperaba un objeto JSON.')
        labels = record.get('tema')
        if isinstance(labels, str):
            labels = [labels]
        if not isinstance(labels, list) or not labels:
            raise SystemExit(f'Elemento {index}: tema no es una lista válida.')

        real_labels = list(dict.fromkeys(
            label for label in labels if label != NONE_TOPIC
        ))
        unknown = [label for label in real_labels if label not in allowed]
        if unknown:
            raise SystemExit(
                f'Elemento {index}: categorías desconocidas: {unknown}.'
            )
        if not real_labels:
            continue

        labelled_records += 1
        multi_topic_records += len(real_labels) > 1
        counts.update(real_labels)

    return counts, labelled_records, multi_topic_records


def rounded_axis_max(maximum: int) -> tuple[int, int]:
    if maximum <= 0:
        return 1, 1
    rough_step = maximum / 4
    magnitude = 10 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    nice = next(value for value in (1, 2, 5, 10) if normalized <= value)
    step = max(1, int(nice * magnitude))
    return math.ceil(maximum / step) * step, step


def generate_svg(
    counts: Counter[str],
    record_count: int,
    multi_topic_count: int,
    input_path: Path,
) -> str:
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    assignment_count = sum(counts.values())

    width = 1400
    top = 178
    bottom = 82
    left = 420
    right = 125
    row_height = 42
    bar_height = 25
    height = top + len(rows) * row_height + bottom
    chart_width = width - left - right
    axis_max, tick_step = rounded_axis_max(max(counts.values(), default=0))

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title description">'
        ),
        '<title id="title">Número de noticias por categoría</title>',
        (
            '<desc id="description">Gráfica de barras horizontal con el '
            'número de noticias asignadas a cada categoría.</desc>'
        ),
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        (
            '<text x="56" y="67" fill="#0f172a" font-family="Arial, '
            'Helvetica, sans-serif" font-size="34" font-weight="700">'
            'Número de noticias por categoría</text>'
        ),
        (
            f'<text x="56" y="105" fill="#475569" font-family="Arial, '
            f'Helvetica, sans-serif" font-size="19">{record_count} noticias '
            f'· {assignment_count} asignaciones · {multi_topic_count} '
            'noticias multietiqueta</text>'
        ),
        (
            '<text x="56" y="136" fill="#64748b" font-family="Arial, '
            'Helvetica, sans-serif" font-size="16">Las noticias '
            'multietiqueta cuentan una vez en cada categoría asignada.</text>'
        ),
    ]

    for tick in range(0, axis_max + 1, tick_step):
        x = left + chart_width * tick / axis_max
        svg.append(
            f'<line x1="{x:.1f}" y1="{top - 15}" x2="{x:.1f}" '
            f'y2="{top + len(rows) * row_height - 10}" '
            'stroke="#dbe3ee" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x:.1f}" y="{top - 27}" text-anchor="middle" '
            'fill="#64748b" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="15">{tick}</text>'
        )

    palette = (
        '#1d4ed8', '#2563eb', '#3b82f6', '#0f766e', '#0d9488',
        '#0891b2', '#0284c7', '#4f46e5', '#7c3aed', '#9333ea',
        '#c026d3', '#db2777', '#e11d48', '#ea580c', '#ca8a04',
    )
    for index, (topic, count) in enumerate(rows):
        y = top + index * row_height
        bar_width = chart_width * count / axis_max
        label_y = y + bar_height / 2 + 6
        svg.extend(
            [
                (
                    f'<text x="{left - 18}" y="{label_y:.1f}" '
                    'text-anchor="end" fill="#1e293b" font-family="Arial, '
                    f'Helvetica, sans-serif" font-size="17">{escape(topic)}</text>'
                ),
                (
                    f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" '
                    f'height="{bar_height}" rx="5" fill="{palette[index]}"/>'
                ),
                (
                    f'<text x="{left + bar_width + 12:.1f}" '
                    f'y="{label_y:.1f}" fill="#0f172a" '
                    'font-family="Arial, Helvetica, sans-serif" '
                    f'font-size="17" font-weight="700">{count}</text>'
                ),
            ]
        )

    source = escape(input_path.name)
    svg.append(
        f'<text x="56" y="{height - 30}" fill="#64748b" '
        'font-family="Arial, Helvetica, sans-serif" font-size="14">'
        f'Fuente: {source}</text>'
    )
    svg.append('</svg>')
    return '\n'.join(svg) + '\n'


def main() -> None:
    args = parse_args()
    records = load_json(args.input)
    topics = load_json(args.topics)
    if not isinstance(topics, dict) or not topics:
        raise SystemExit('temas.json debe contener un objeto no vacío.')

    counts, record_count, multi_topic_count = count_topics(
        records, list(topics)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        generate_svg(counts, record_count, multi_topic_count, args.input),
        encoding='utf-8',
    )

    print(f'Noticias representadas: {record_count}')
    print(f'Noticias multietiqueta: {multi_topic_count}')
    print(f'Asignaciones representadas: {sum(counts.values())}')
    print(f'Gráfica: {args.output.resolve()}')


if __name__ == '__main__':
    main()
