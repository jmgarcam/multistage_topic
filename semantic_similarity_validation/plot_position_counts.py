#!/usr/bin/env python3
'''Representa la cantidad de noticias por posición para elegir k.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description='Genera una gráfica con la cantidad de aciertos por posición.'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=script_dir / 'top_19_positions.json',
        help='Informe JSON generado por validate_top_10_positions.py.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=script_dir / 'position_counts.pdf',
        help='Ruta de la figura que se generará.',
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        with path.open('r', encoding='utf-8') as file_handle:
            return json.load(file_handle)
    except FileNotFoundError as exc:
        raise SystemExit(f'No se encontró el archivo: {path}') from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f'JSON no válido en {path}: {exc}') from exc


def read_position_data(path: Path) -> list[int]:
    report = load_json(path)
    if not isinstance(report, dict):
        raise SystemExit(f'{path} debe contener un objeto JSON.')

    news = report.get('noticias')
    if not isinstance(news, list):
        raise SystemExit(f'{path} debe contener una lista noticias.')

    position_fields = {
        key
        for record in news
        if isinstance(record, dict)
        for key in record
        if key.startswith('posicion_top_')
    }
    if len(position_fields) != 1:
        raise SystemExit(
            'No se pudo identificar un único campo posicion_top_k en las noticias.'
        )

    position_field = position_fields.pop()
    try:
        max_k = int(position_field.rsplit('_', maxsplit=1)[1])
    except ValueError as exc:
        raise SystemExit(f'Campo de posición no válido: {position_field}') from exc

    counts = [0] * max_k
    for index, record in enumerate(news):
        if not isinstance(record, dict):
            raise SystemExit(f'Noticia {index}: se esperaba un objeto JSON.')
        position = record.get(position_field)
        if position is None:
            continue
        if isinstance(position, bool) or not isinstance(position, int):
            raise SystemExit(f'Noticia {index}: posición no válida.')
        if not 1 <= position <= max_k:
            raise SystemExit(
                f'Noticia {index}: posición fuera del intervalo 1-{max_k}.'
            )
        counts[position - 1] += 1

    return counts


def plot_positions(counts: list[int], output_path: Path) -> None:
    positions = list(range(1, len(counts) + 1))
    paper_style = {
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.linewidth': 0.7,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }

    with plt.rc_context(paper_style):
        figure, axis = plt.subplots(figsize=(7.0, 3.2))
        bars = axis.bar(
            positions,
            counts,
            width=0.72,
            color='0.35',
            edgecolor='black',
            linewidth=0.45,
        )
        axis.bar_label(bars, padding=2, fontsize=7)
        axis.set_xlabel('Rank position')
        axis.set_ylabel('Number of articles')
        axis.set_xticks(positions)
        axis.set_xlim(0.35, len(positions) + 0.65)
        axis.set_ylim(0, max(counts) * 1.12 if counts else 1)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.tick_params(direction='out', length=3, width=0.7)
        axis.yaxis.grid(True, color='0.85', linestyle='--', linewidth=0.5)
        axis.set_axisbelow(True)

        figure.tight_layout(pad=0.4)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(figure)


def main() -> None:
    args = parse_args()
    counts = read_position_data(args.input)
    plot_positions(counts, args.output)

    print(f'Gráfica guardada: {args.output}')


if __name__ == '__main__':
    main()
