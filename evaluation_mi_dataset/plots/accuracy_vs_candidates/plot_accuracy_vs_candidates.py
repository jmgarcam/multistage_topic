#!/usr/bin/env python3
"""Grafica la accuracy de E5 -> SLM frente al número de candidatos."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


PLOTS_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = PLOTS_DIR.parent.parent
E5_RESULTS = EVALUATION_DIR / 'e5_results' / 'e5_top_n_results.json'
OUTPUT_SVG = PLOTS_DIR / 'accuracy_vs_candidates.svg'
OUTPUT_CSV = PLOTS_DIR / 'accuracy_vs_candidates_data.csv'
EXPECTED_N = tuple(range(1, 16))


@dataclass(frozen=True)
class ModelSpec:
    label: str
    ollama_model: str
    hybrid_dir: Path
    llama_only_file: Path


@dataclass
class ModelData:
    spec: ModelSpec
    accuracies: dict[int, float]
    incomplete: dict[int, tuple[int, int]]
    missing: list[int]
    slm_15_accuracy: float | None
    slm_15_source: str | None


def model_result_slug(model: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', model.casefold()).strip('_')


MODEL_SPECS = (
    ModelSpec(
        label='Llama 3.2 · 1B',
        ollama_model='llama3.2:1b',
        hybrid_dir=(
            EVALUATION_DIR
            / 'results_e5_llama3_2_1b_two_prompts_no_margin'
        ),
        llama_only_file=(
            EVALUATION_DIR
            / 'results_llama3_2_1b_two_prompts'
            / 'llama3_2_1b_direct_results.json'
        ),
    ),
    ModelSpec(
        label='Llama 3.2 · 3B',
        ollama_model='llama3.2:3b',
        hybrid_dir=(
            EVALUATION_DIR
            / 'results_e5_llama3_2_3b_two_prompts_no_margin'
        ),
        llama_only_file=(
            EVALUATION_DIR
            / 'results_llama3_2_3b_two_prompts'
            / 'llama3_2_3b_direct_results.json'
        ),
    ),
    ModelSpec(
        label='Llama 3.1 · 8B',
        ollama_model='llama3.1:8b',
        hybrid_dir=(
            EVALUATION_DIR
            / 'results_e5_llama3_1_8b_two_prompts_no_margin'
        ),
        llama_only_file=(
            EVALUATION_DIR
            / 'results_llama3_1_8b_two_prompts'
            / 'llama3_1_8b_direct_results.json'
        ),
    ),
    ModelSpec(
        label='Qwen 2.5 · 1.5B',
        ollama_model='qwen2.5:1.5b-instruct',
        hybrid_dir=(
            EVALUATION_DIR
            / 'results_e5_qwen2_5_1_5b_instruct_two_prompts_no_margin'
        ),
        llama_only_file=(
            EVALUATION_DIR
            / 'results_qwen2_5_1_5b_instruct_two_prompts'
            / 'qwen2_5_1_5b_instruct_direct_results.json'
        ),
    ),
    ModelSpec(
        label='Qwen 2.5 · 3B',
        ollama_model='qwen2.5:3b-instruct',
        hybrid_dir=(
            EVALUATION_DIR
            / 'results_e5_qwen2_5_3b_instruct_two_prompts_no_margin'
        ),
        llama_only_file=(
            EVALUATION_DIR
            / 'results_qwen2_5_3b_instruct_two_prompts'
            / 'qwen2_5_3b_instruct_direct_results.json'
        ),
    ),
    ModelSpec(
        label='Qwen 2.5 · 7B',
        ollama_model='qwen2.5:7b-instruct',
        hybrid_dir=(
            EVALUATION_DIR
            / 'results_e5_qwen2_5_7b_instruct_two_prompts_no_margin'
        ),
        llama_only_file=(
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


def accuracy_percent(report: dict[str, Any], path: Path) -> float:
    metrics = report.get('metrics')
    value = (
        metrics.get('slm_final_accuracy_percent')
        if isinstance(metrics, dict)
        else None
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f'{path}: falta slm_final_accuracy_percent.')
    return float(value)


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


def is_complete(report: dict[str, Any], path: Path) -> bool:
    completed, expected = progress(report, path)
    return report.get('status') == 'complete' and completed == expected


def load_e5_baseline() -> tuple[float, int]:
    if not E5_RESULTS.is_file():
        raise SystemExit(f'No se encontró el resultado E5: {E5_RESULTS}')
    report = load_json(E5_RESULTS)
    if not isinstance(report, dict) or report.get('status') != 'complete':
        raise SystemExit(f'El resultado E5 no está completo: {E5_RESULTS}')
    metrics = report.get('metrics')
    dataset = report.get('dataset')
    accuracy = metrics.get('top_1_accuracy_percent') if isinstance(metrics, dict) else None
    total = dataset.get('total_news') if isinstance(dataset, dict) else None
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or isinstance(total, bool)
        or not isinstance(total, int)
    ):
        raise SystemExit(f'Métricas E5 no válidas en {E5_RESULTS}.')
    return float(accuracy), total


def load_model_data(spec: ModelSpec, expected_total: int) -> ModelData:
    accuracies: dict[int, float] = {}
    incomplete: dict[int, tuple[int, int]] = {}
    missing: list[int] = []

    for n in EXPECTED_N:
        result_slug = model_result_slug(spec.ollama_model)
        path = spec.hybrid_dir / f'e5_top_{n}_{result_slug}_results.json'
        if not path.is_file():
            missing.append(n)
            continue
        report = load_json(path)
        if not isinstance(report, dict):
            raise SystemExit(f'{path}: se esperaba un objeto JSON.')
        validate_model(report, spec, path)
        completed, expected = progress(report, path)
        if expected != expected_total:
            raise SystemExit(
                f'{path}: contiene {expected} noticias; se esperaban '
                f'{expected_total}.'
            )
        if is_complete(report, path):
            accuracies[n] = accuracy_percent(report, path)
        else:
            incomplete[n] = (completed, expected)

    slm_15_accuracy: float | None = None
    slm_15_source: str | None = None
    if spec.llama_only_file.is_file():
        report = load_json(spec.llama_only_file)
        if not isinstance(report, dict):
            raise SystemExit(
                f'{spec.llama_only_file}: se esperaba un objeto JSON.'
            )
        validate_model(report, spec, spec.llama_only_file)
        completed, expected = progress(report, spec.llama_only_file)
        if expected != expected_total:
            raise SystemExit(
                f'{spec.llama_only_file}: contiene {expected} noticias; '
                f'se esperaban {expected_total}.'
            )
        if is_complete(report, spec.llama_only_file):
            slm_15_accuracy = accuracy_percent(report, spec.llama_only_file)
            slm_15_source = 'llama_only'

    # Con N=15, E5 entrega el catálogo completo en el orden de temas.json.
    # Por tanto es el mismo experimento que Llama solo con los 15 temas.
    if 15 in accuracies:
        if (
            slm_15_accuracy is not None
            and abs(slm_15_accuracy - accuracies[15]) > 0.011
        ):
            raise SystemExit(
                f'{spec.label}: Llama solo ({slm_15_accuracy:.2f}%) no '
                f'coincide con E5 -> SLM para N=15 ({accuracies[15]:.2f}%).'
            )
        if slm_15_accuracy is None:
            slm_15_accuracy = accuracies[15]
            slm_15_source = 'hybrid_n15'

    return ModelData(
        spec=spec,
        accuracies=accuracies,
        incomplete=incomplete,
        missing=missing,
        slm_15_accuracy=slm_15_accuracy,
        slm_15_source=slm_15_source,
    )


def summarize_availability(data: ModelData) -> list[str]:
    complete_n = sorted(data.accuracies)
    if len(complete_n) == len(EXPECTED_N) and data.slm_15_accuracy is not None:
        return ['Serie completa: N=1–15.']
    if not complete_n:
        return ['Sin resultados completos comparables.']

    if complete_n == list(range(1, max(complete_n) + 1)):
        lines = [f'Serie parcial: N=1–{max(complete_n)} completos.']
    else:
        joined = ', '.join(map(str, complete_n))
        lines = [f'Valores N completos: {joined}.']
    if data.incomplete:
        n = min(data.incomplete)
        completed, expected = data.incomplete[n]
        lines.append(f'N={n} en curso: {completed}/{expected}.')
    if data.slm_15_accuracy is None:
        lines.append('Baseline SLM-15 no disponible.')
    return lines


def write_csv(
    baseline: float, total_news: int, model_data: list[ModelData]
) -> None:
    with OUTPUT_CSV.open('w', encoding='utf-8', newline='') as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=(
                'model',
                'ollama_model',
                'n',
                'e5_to_slm_accuracy_percent',
                'e5_top1_accuracy_percent',
                'slm_15_accuracy_percent',
                'status',
                'evaluated_news',
            ),
        )
        writer.writeheader()
        for data in model_data:
            for n in EXPECTED_N:
                if n in data.accuracies:
                    status = 'complete'
                    evaluated = total_news
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
                        'e5_to_slm_accuracy_percent': data.accuracies.get(n, ''),
                        'e5_top1_accuracy_percent': baseline,
                        'slm_15_accuracy_percent': (
                            data.slm_15_accuracy
                            if data.slm_15_accuracy is not None
                            else ''
                        ),
                        'status': status,
                        'evaluated_news': evaluated,
                    }
                )


def polyline_points(
    accuracies: dict[int, float],
    plot_left: float,
    plot_top: float,
    plot_width: float,
    plot_height: float,
) -> str:
    return ' '.join(
        f'{plot_left + (n - 1) * plot_width / 14:.1f},'
        f'{plot_top + (100 - accuracy) * plot_height / 100:.1f}'
        for n, accuracy in sorted(accuracies.items())
    )


def generate_svg(
    baseline: float, total_news: int, model_data: list[ModelData]
) -> str:
    width = 1740
    height = 1245
    outer_left = 58
    outer_right = 42
    gap = 34
    columns = 3
    row_stride = 528
    first_panel_top = 112
    panel_width = (
        width - outer_left - outer_right - (columns - 1) * gap
    ) / columns
    plot_height = 390
    plot_left_offset = 67
    plot_right_padding = 23

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title description">'
        ),
        '<title id="title">Accuracy frente al número de candidatos</title>',
        (
            '<desc id="description">Seis paneles comparan la accuracy de '
            'E5 seguido de cada SLM para N entre 1 y 15.</desc>'
        ),
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        (
            '<text x="58" y="54" fill="#0f172a" '
            'font-family="Arial, Helvetica, sans-serif" font-size="34" '
            'font-weight="700">Accuracy frente al número de candidatos</text>'
        ),
        (
            f'<text x="58" y="88" fill="#475569" '
            'font-family="Arial, Helvetica, sans-serif" font-size="18">'
            f'Dataset etiquetado: {total_news} noticias · E5 Top-1: '
            f'{baseline:.2f}%</text>'
        ),
    ]

    for panel_index, data in enumerate(model_data):
        row, column = divmod(panel_index, columns)
        panel_x = outer_left + column * (panel_width + gap)
        panel_top = first_panel_top + row * row_stride
        plot_top = panel_top + 56
        plot_left = panel_x + plot_left_offset
        plot_width = panel_width - plot_left_offset - plot_right_padding
        plot_bottom = plot_top + plot_height

        svg.append(
            f'<rect x="{panel_x:.1f}" y="{panel_top:.1f}" '
            f'width="{panel_width:.1f}" '
            'height="500" rx="12" fill="#ffffff" stroke="#dbe3ee"/>'
        )
        svg.append(
            f'<text x="{panel_x + panel_width / 2:.1f}" '
            f'y="{panel_top + 33:.1f}" '
            'text-anchor="middle" fill="#0f172a" '
            'font-family="Arial, Helvetica, sans-serif" font-size="22" '
            f'font-weight="700">{escape(data.spec.label)}</text>'
        )

        for tick in range(0, 101, 20):
            y = plot_top + (100 - tick) * plot_height / 100
            svg.append(
                f'<line x1="{plot_left:.1f}" y1="{y:.1f}" '
                f'x2="{plot_left + plot_width:.1f}" y2="{y:.1f}" '
                'stroke="#e2e8f0" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{plot_left - 10:.1f}" y="{y + 5:.1f}" '
                'text-anchor="end" fill="#64748b" '
                'font-family="Arial, Helvetica, sans-serif" '
                f'font-size="13">{tick}</text>'
            )

        for n in EXPECTED_N:
            x = plot_left + (n - 1) * plot_width / 14
            svg.append(
                f'<line x1="{x:.1f}" y1="{plot_bottom:.1f}" '
                f'x2="{x:.1f}" y2="{plot_bottom + 5:.1f}" '
                'stroke="#64748b"/>'
            )
            svg.append(
                f'<text x="{x:.1f}" y="{plot_bottom + 22:.1f}" '
                'text-anchor="middle" fill="#475569" '
                'font-family="Arial, Helvetica, sans-serif" '
                f'font-size="11">{n}</text>'
            )

        svg.extend(
            [
                (
                    f'<line x1="{plot_left:.1f}" y1="{plot_bottom:.1f}" '
                    f'x2="{plot_left + plot_width:.1f}" '
                    f'y2="{plot_bottom:.1f}" stroke="#64748b"/>'
                ),
                (
                    f'<line x1="{plot_left:.1f}" y1="{plot_top:.1f}" '
                    f'x2="{plot_left:.1f}" y2="{plot_bottom:.1f}" '
                    'stroke="#64748b"/>'
                ),
                (
                    f'<text x="{plot_left + plot_width / 2:.1f}" '
                    f'y="{plot_bottom + 48:.1f}" text-anchor="middle" '
                    'fill="#334155" font-family="Arial, Helvetica, '
                    'sans-serif" font-size="15">Número de candidatos (N)</text>'
                ),
            ]
        )

        baseline_y = plot_top + (100 - baseline) * plot_height / 100
        svg.append(
            f'<line x1="{plot_left:.1f}" y1="{baseline_y:.1f}" '
            f'x2="{plot_left + plot_width:.1f}" y2="{baseline_y:.1f}" '
            'stroke="#dc2626" stroke-width="2.3" stroke-dasharray="8 6"/>'
        )

        if data.slm_15_accuracy is not None:
            slm_y = (
                plot_top
                + (100 - data.slm_15_accuracy) * plot_height / 100
            )
            svg.append(
                f'<line x1="{plot_left:.1f}" y1="{slm_y:.1f}" '
                f'x2="{plot_left + plot_width:.1f}" y2="{slm_y:.1f}" '
                'stroke="#059669" stroke-width="2.3"/>'
            )

        if data.accuracies:
            points = polyline_points(
                data.accuracies,
                plot_left,
                plot_top,
                plot_width,
                plot_height,
            )
            svg.append(
                f'<polyline points="{points}" fill="none" '
                'stroke="#2563eb" stroke-width="3.2" '
                'stroke-linejoin="round" stroke-linecap="round"/>'
            )
            for n, accuracy in sorted(data.accuracies.items()):
                x = plot_left + (n - 1) * plot_width / 14
                y = plot_top + (100 - accuracy) * plot_height / 100
                svg.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" '
                    'fill="#ffffff" stroke="#2563eb" stroke-width="2.5"/>'
                )

        notes = summarize_availability(data)
        if notes != ['Serie completa: N=1–15.']:
            note_height = 24 + 20 * len(notes)
            svg.append(
                f'<rect x="{plot_left + 10:.1f}" y="{plot_top + 12:.1f}" '
                f'width="{plot_width - 20:.1f}" height="{note_height}" '
                'rx="7" fill="#fff7ed" stroke="#fdba74"/>'
            )
            for line_index, line in enumerate(notes):
                svg.append(
                    f'<text x="{plot_left + 22:.1f}" '
                    f'y="{plot_top + 38 + line_index * 20:.1f}" '
                    'fill="#9a3412" font-family="Arial, Helvetica, '
                    f'sans-serif" font-size="14">{escape(line)}</text>'
                )

        if column == 0:
            centre_y = plot_top + plot_height / 2
            svg.append(
                f'<text x="17" y="{centre_y:.1f}" '
                f'transform="rotate(-90 17 {centre_y:.1f})" '
                'text-anchor="middle" fill="#334155" '
                'font-family="Arial, Helvetica, sans-serif" '
                'font-size="16">Accuracy (%)</text>'
            )

    legend_y = 1187
    legend_items = (
        ('#2563eb', '', 'E5 → SLM'),
        ('#dc2626', '8 6', 'E5 Top-1'),
        ('#059669', '', 'SLM con los 15 temas'),
    )
    legend_start = 470
    for index, (colour, dash, label) in enumerate(legend_items):
        x = legend_start + index * 285
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ''
        svg.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 44}" '
            f'y2="{legend_y}" stroke="{colour}" stroke-width="3"'
            f'{dash_attribute}/>'
        )
        svg.append(
            f'<text x="{x + 55}" y="{legend_y + 5}" fill="#334155" '
            'font-family="Arial, Helvetica, sans-serif" '
            f'font-size="16">{escape(label)}</text>'
        )

    svg.append(
        f'<text x="{width - 48}" y="{height - 24}" text-anchor="end" '
        'fill="#64748b" font-family="Arial, Helvetica, sans-serif" '
        'font-size="13">Solo se representan ejecuciones completas para '
        'cada N.</text>'
    )
    svg.append('</svg>')
    return '\n'.join(svg) + '\n'


def main() -> None:
    baseline, total_news = load_e5_baseline()
    model_data = [
        load_model_data(spec, total_news) for spec in MODEL_SPECS
    ]
    write_csv(baseline, total_news, model_data)
    OUTPUT_SVG.write_text(
        generate_svg(baseline, total_news, model_data),
        encoding='utf-8',
    )

    print(f'E5 Top-1: {baseline:.2f}%')
    for data in model_data:
        complete_values = ', '.join(map(str, sorted(data.accuracies))) or 'ninguno'
        print(f'{data.spec.label}: N completos = {complete_values}')
    print(f'Gráfica: {OUTPUT_SVG.resolve()}')
    print(f'Datos: {OUTPUT_CSV.resolve()}')


if __name__ == '__main__':
    main()
