#!/usr/bin/env python3
"""Grafica qué errores corrige y qué aciertos estropea cada SLM respecto a E5."""

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
OUTPUT_SVG = PLOT_DIR / 'corrections_and_harms.svg'
OUTPUT_CSV = PLOT_DIR / 'corrections_and_harms_data.csv'
OUTPUT_TABLE = PLOT_DIR / 'neutral_changes_table.md'
EXPECTED_N = tuple(range(1, 16))


@dataclass(frozen=True)
class ModelSpec:
    label: str
    ollama_model: str
    results_dir: Path


@dataclass(frozen=True)
class TransitionCounts:
    total: int
    corrections: int
    harms: int
    wrong_to_wrong: int
    correct_to_correct: int
    unchanged: int

    @property
    def changed(self) -> int:
        return (
            self.corrections
            + self.harms
            + self.wrong_to_wrong
            + self.correct_to_correct
        )

    @property
    def corrections_percent(self) -> float:
        return 100 * self.corrections / self.total

    @property
    def harms_percent(self) -> float:
        return 100 * self.harms / self.total

    @property
    def net_percent(self) -> float:
        return 100 * (self.corrections - self.harms) / self.total


@dataclass
class ModelData:
    spec: ModelSpec
    points: dict[int, TransitionCounts]
    incomplete: dict[int, tuple[int, int]]
    missing: list[int]


def model_result_slug(model: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', model.casefold()).strip('_')


MODEL_SPECS = (
    ModelSpec(
        label='Llama 3.2 · 1B',
        ollama_model='llama3.2:1b',
        results_dir=(
            EVALUATION_DIR
            / 'results_e5_llama3_2_1b_two_prompts_no_margin'
        ),
    ),
    ModelSpec(
        label='Llama 3.2 · 3B',
        ollama_model='llama3.2:3b',
        results_dir=(
            EVALUATION_DIR
            / 'results_e5_llama3_2_3b_two_prompts_no_margin'
        ),
    ),
    ModelSpec(
        label='Llama 3.1 · 8B',
        ollama_model='llama3.1:8b',
        results_dir=(
            EVALUATION_DIR
            / 'results_e5_llama3_1_8b_two_prompts_no_margin'
        ),
    ),
    ModelSpec(
        label='Qwen 2.5 · 1.5B',
        ollama_model='qwen2.5:1.5b-instruct',
        results_dir=(
            EVALUATION_DIR
            / 'results_e5_qwen2_5_1_5b_instruct_two_prompts_no_margin'
        ),
    ),
    ModelSpec(
        label='Qwen 2.5 · 3B',
        ollama_model='qwen2.5:3b-instruct',
        results_dir=(
            EVALUATION_DIR
            / 'results_e5_qwen2_5_3b_instruct_two_prompts_no_margin'
        ),
    ),
    ModelSpec(
        label='Qwen 2.5 · 7B',
        ollama_model='qwen2.5:7b-instruct',
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


def integer_value(mapping: dict[str, Any], name: str, path: Path) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f'{path}: falta el entero {name}.')
    return value


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


def classify_transitions(
    report: dict[str, Any], total: int, path: Path
) -> TransitionCounts:
    predictions = report.get('predictions')
    metrics = report.get('metrics')
    if not isinstance(predictions, list) or len(predictions) != total:
        raise SystemExit(f'{path}: predictions no contiene {total} elementos.')
    if not isinstance(metrics, dict):
        raise SystemExit(f'{path}: falta el bloque metrics.')

    corrections = 0
    harms = 0
    wrong_to_wrong = 0
    correct_to_correct = 0
    unchanged = 0

    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise SystemExit(f'{path}: predicción {index} no válida.')
        gold = prediction.get('topics_correctos')
        e5_topic = prediction.get('e5_top_1_topic')
        final_topic = prediction.get('final_decision')
        if (
            not isinstance(gold, list)
            or not gold
            or not all(isinstance(topic, str) for topic in gold)
            or not isinstance(e5_topic, str)
            or not isinstance(final_topic, str)
        ):
            raise SystemExit(f'{path}: campos no válidos en predicción {index}.')

        if final_topic == e5_topic:
            unchanged += 1
            continue
        e5_correct = e5_topic in gold
        final_correct = final_topic in gold
        if not e5_correct and final_correct:
            corrections += 1
        elif e5_correct and not final_correct:
            harms += 1
        elif not e5_correct and not final_correct:
            wrong_to_wrong += 1
        else:
            correct_to_correct += 1

    counts = TransitionCounts(
        total=total,
        corrections=corrections,
        harms=harms,
        wrong_to_wrong=wrong_to_wrong,
        correct_to_correct=correct_to_correct,
        unchanged=unchanged,
    )

    expected_values = {
        'slm_helped_vs_e5_top_1': counts.corrections,
        'slm_harmed_vs_e5_top_1': counts.harms,
        'slm_valid_topic_changes': counts.changed,
        'net_hit_change_vs_e5_top_1': counts.corrections - counts.harms,
    }
    for metric_name, calculated in expected_values.items():
        stored = integer_value(metrics, metric_name, path)
        if stored != calculated:
            raise SystemExit(
                f'{path}: {metric_name}={stored}, pero las predicciones '
                f'producen {calculated}.'
            )

    retriever_hits = integer_value(metrics, 'retriever_top_1_hits', path)
    final_hits = integer_value(metrics, 'final_hits', path)
    if final_hits - retriever_hits != counts.corrections - counts.harms:
        raise SystemExit(f'{path}: el balance no coincide con la accuracy final.')
    return counts


def load_model_data(spec: ModelSpec) -> ModelData:
    points: dict[int, TransitionCounts] = {}
    incomplete: dict[int, tuple[int, int]] = {}
    missing: list[int] = []
    expected_total: int | None = None

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
        if expected_total is None:
            expected_total = expected
        elif expected != expected_total:
            raise SystemExit(
                f'{path}: contiene {expected} noticias; se esperaban '
                f'{expected_total}.'
            )
        if report.get('status') != 'complete' or completed != expected:
            incomplete[n] = (completed, expected)
            continue
        points[n] = classify_transitions(report, expected, path)

    return ModelData(spec, points, incomplete, missing)


def availability_text(data: ModelData) -> str:
    complete = sorted(data.points)
    if complete == list(EXPECTED_N):
        return 'Serie completa: N=1–15.'
    if not complete:
        return 'Sin resultados completos.'
    if complete == list(range(1, max(complete) + 1)):
        text = f'Serie parcial: N=1–{max(complete)}.'
    else:
        text = 'N completos: ' + ', '.join(map(str, complete)) + '.'
    if data.incomplete:
        n = min(data.incomplete)
        completed, expected = data.incomplete[n]
        text += f' N={n} en curso: {completed}/{expected}.'
    return text


def write_csv(model_data: list[ModelData]) -> None:
    with OUTPUT_CSV.open('w', encoding='utf-8', newline='') as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=(
                'model',
                'ollama_model',
                'n',
                'total_news',
                'corrections',
                'corrections_percent',
                'harms',
                'harms_percent',
                'net_accuracy_points',
                'wrong_to_wrong_changes',
                'wrong_to_wrong_percent',
                'correct_to_correct_changes',
                'correct_to_correct_percent',
                'total_topic_changes',
                'unchanged',
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
                    evaluated: int | str = point.total
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
                        'total_news': point.total if point else '',
                        'corrections': point.corrections if point else '',
                        'corrections_percent': (
                            round(point.corrections_percent, 6) if point else ''
                        ),
                        'harms': point.harms if point else '',
                        'harms_percent': (
                            round(point.harms_percent, 6) if point else ''
                        ),
                        'net_accuracy_points': (
                            round(point.net_percent, 6) if point else ''
                        ),
                        'wrong_to_wrong_changes': (
                            point.wrong_to_wrong if point else ''
                        ),
                        'wrong_to_wrong_percent': (
                            round(100 * point.wrong_to_wrong / point.total, 6)
                            if point
                            else ''
                        ),
                        'correct_to_correct_changes': (
                            point.correct_to_correct if point else ''
                        ),
                        'correct_to_correct_percent': (
                            round(100 * point.correct_to_correct / point.total, 6)
                            if point
                            else ''
                        ),
                        'total_topic_changes': point.changed if point else '',
                        'unchanged': point.unchanged if point else '',
                        'status': status,
                        'evaluated_news': evaluated,
                    }
                )


def write_neutral_table(model_data: list[ModelData]) -> None:
    lines = [
        '# Cambios que no modifican la accuracy',
        '',
        'Los cambios `incorrecta → incorrecta` no corrigen el error de E5. '
        'También se separan los posibles cambios `correcta → correcta` de las '
        'noticias multietiqueta.',
        '',
        '| Modelo | N | Incorrecta → incorrecta | % del total | '
        'Correcta → correcta | % del total |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for data in model_data:
        for n, point in sorted(data.points.items()):
            wrong_percent = 100 * point.wrong_to_wrong / point.total
            correct_percent = 100 * point.correct_to_correct / point.total
            lines.append(
                f'| {data.spec.label} | {n} | {point.wrong_to_wrong} | '
                f'{wrong_percent:.2f}% | {point.correct_to_correct} | '
                f'{correct_percent:.2f}% |'
            )
    OUTPUT_TABLE.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def y_limit(model_data: list[ModelData]) -> int:
    maximum = max(
        (
            max(
                point.corrections_percent,
                point.harms_percent,
                abs(point.net_percent),
            )
            for data in model_data
            for point in data.points.values()
        ),
        default=10,
    )
    step = 5 if maximum <= 20 else 10
    return max(10, int(math.ceil(maximum / step) * step))


def x_position(n: int, left: float, width: float) -> float:
    return left + (n - 1) * width / (len(EXPECTED_N) - 1)


def y_position(
    value: float, top: float, height: float, maximum: float
) -> float:
    return top + (maximum - value) * height / (2 * maximum)


def generate_svg(model_data: list[ModelData]) -> str:
    width = 1740
    height = 1235
    outer_left = 54
    outer_right = 42
    gap = 34
    columns = 3
    row_stride = 500
    first_panel_top = 143
    panel_width = (
        width - outer_left - outer_right - (columns - 1) * gap
    ) / columns
    plot_height = 375
    plot_left_offset = 68
    plot_right_padding = 23
    maximum = y_limit(model_data)
    tick_step = 5 if maximum <= 20 else 10

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title description">'
        ),
        '<title id="title">Correcciones y errores introducidos por el SLM</title>',
        (
            '<desc id="description">Barras divergentes muestran errores de E5 '
            'corregidos, aciertos de E5 estropeados y su balance neto para cada '
            'modelo y número de candidatos.</desc>'
        ),
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        (
            '<text x="54" y="48" fill="#0f172a" '
            'font-family="Arial, Helvetica, sans-serif" font-size="33" '
            'font-weight="700">¿Qué aporta el SLM a E5?</text>'
        ),
        (
            '<text x="54" y="80" fill="#475569" '
            'font-family="Arial, Helvetica, sans-serif" font-size="17">'
            'Porcentaje sobre todas las noticias · ΔA = correcciones − aciertos '
            'estropeados</text>'
        ),
    ]

    legend = (
        ('#16a34a', 'Errores de E5 corregidos'),
        ('#dc2626', 'Aciertos de E5 estropeados'),
    )
    for index, (colour, label) in enumerate(legend):
        x = 510 + index * 360
        svg.append(
            f'<rect x="{x}" y="105" width="28" height="15" '
            f'rx="2" fill="{colour}" opacity="0.88"/>'
        )
        svg.append(
            f'<text x="{x + 40}" y="118" fill="#334155" '
            'font-family="Arial, Helvetica, sans-serif" font-size="16">'
            f'{escape(label)}</text>'
        )
    svg.extend(
        [
            '<circle cx="1229" cy="112" r="5" fill="#111827"/>',
            (
                '<text x="1243" y="118" fill="#334155" '
                'font-family="Arial, Helvetica, sans-serif" font-size="16">'
                'Balance neto (ΔA)</text>'
            ),
        ]
    )

    for panel_index, data in enumerate(model_data):
        row, column = divmod(panel_index, columns)
        panel_x = outer_left + column * (panel_width + gap)
        panel_top = first_panel_top + row * row_stride
        plot_top = panel_top + 41
        plot_left = panel_x + plot_left_offset
        plot_width = panel_width - plot_left_offset - plot_right_padding
        plot_bottom = plot_top + plot_height
        zero_y = y_position(0, plot_top, plot_height, maximum)

        svg.append(
            f'<rect x="{panel_x:.1f}" y="{panel_top:.1f}" '
            f'width="{panel_width:.1f}" '
            'height="478" rx="12" fill="#ffffff" stroke="#dbe3ee"/>'
        )
        svg.append(
            f'<text x="{panel_x + panel_width / 2:.1f}" '
            f'y="{panel_top + 29:.1f}" '
            'text-anchor="middle" fill="#0f172a" '
            'font-family="Arial, Helvetica, sans-serif" font-size="21" '
            f'font-weight="700">{escape(data.spec.label)}</text>'
        )

        for tick in range(-maximum, maximum + 1, tick_step):
            y = y_position(tick, plot_top, plot_height, maximum)
            stroke = '#64748b' if tick == 0 else '#e2e8f0'
            stroke_width = '1.8' if tick == 0 else '1'
            svg.append(
                f'<line x1="{plot_left:.1f}" y1="{y:.1f}" '
                f'x2="{plot_left + plot_width:.1f}" y2="{y:.1f}" '
                f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
            )
            svg.append(
                f'<text x="{plot_left - 10:.1f}" y="{y + 5:.1f}" '
                'text-anchor="end" fill="#64748b" '
                'font-family="Arial, Helvetica, sans-serif" '
                f'font-size="12">{tick}</text>'
            )

        for n in EXPECTED_N:
            x = x_position(n, plot_left, plot_width)
            svg.append(
                f'<line x1="{x:.1f}" y1="{plot_bottom:.1f}" '
                f'x2="{x:.1f}" y2="{plot_bottom + 5:.1f}" '
                'stroke="#64748b"/>'
            )
            svg.append(
                f'<text x="{x:.1f}" y="{plot_bottom + 21:.1f}" '
                'text-anchor="middle" fill="#475569" '
                'font-family="Arial, Helvetica, sans-serif" '
                f'font-size="11">{n}</text>'
            )

        bar_width = min(18, plot_width / 20)
        for n, point in sorted(data.points.items()):
            x = x_position(n, plot_left, plot_width)
            positive_y = y_position(
                point.corrections_percent, plot_top, plot_height, maximum
            )
            negative_y = y_position(
                -point.harms_percent, plot_top, plot_height, maximum
            )
            net_y = y_position(
                point.net_percent, plot_top, plot_height, maximum
            )
            svg.append(
                f'<rect x="{x - bar_width / 2:.1f}" y="{positive_y:.1f}" '
                f'width="{bar_width:.1f}" height="{zero_y - positive_y:.1f}" '
                'rx="2" fill="#16a34a" opacity="0.88"/>'
            )
            svg.append(
                f'<rect x="{x - bar_width / 2:.1f}" y="{zero_y:.1f}" '
                f'width="{bar_width:.1f}" height="{negative_y - zero_y:.1f}" '
                'rx="2" fill="#dc2626" opacity="0.88"/>'
            )
            svg.append(
                f'<circle cx="{x:.1f}" cy="{net_y:.1f}" r="4.5" '
                'fill="#111827" stroke="#ffffff" stroke-width="1.2"/>'
            )

        svg.extend(
            [
                (
                    f'<line x1="{plot_left:.1f}" y1="{plot_top:.1f}" '
                    f'x2="{plot_left:.1f}" y2="{plot_bottom:.1f}" '
                    'stroke="#64748b"/>'
                ),
                (
                    f'<text x="{plot_left + plot_width / 2:.1f}" '
                    f'y="{plot_bottom + 47:.1f}" text-anchor="middle" '
                    'fill="#334155" font-family="Arial, Helvetica, sans-serif" '
                    'font-size="15">Número de candidatos (N)</text>'
                ),
            ]
        )

        note = availability_text(data)
        if note != 'Serie completa: N=1–15.':
            note_y = (
                plot_top + plot_height / 2
                if not data.points
                else plot_top + 20
            )
            note_size = 15 if not data.points else 12
            svg.append(
                f'<text x="{panel_x + panel_width / 2:.1f}" '
                f'y="{note_y:.1f}" '
                'text-anchor="middle" fill="#9a3412" '
                'font-family="Arial, Helvetica, sans-serif" '
                f'font-size="{note_size}">'
                f'{escape(note)}</text>'
            )

        if column == 0:
            centre_y = plot_top + plot_height / 2
            svg.append(
                f'<text x="18" y="{centre_y:.1f}" '
                f'transform="rotate(-90 18 {centre_y:.1f})" '
                'text-anchor="middle" fill="#334155" '
                'font-family="Arial, Helvetica, sans-serif" '
                'font-size="16">Porcentaje del total (%)</text>'
            )

    svg.append(
        f'<text x="{width - 48}" y="{height - 26}" text-anchor="end" '
        'fill="#64748b" font-family="Arial, Helvetica, sans-serif" '
        'font-size="13">Escala vertical común y simétrica · Solo ejecuciones '
        'completas.</text>'
    )
    svg.append('</svg>')
    return '\n'.join(svg) + '\n'


def main() -> None:
    model_data = [load_model_data(spec) for spec in MODEL_SPECS]
    write_csv(model_data)
    write_neutral_table(model_data)
    OUTPUT_SVG.write_text(generate_svg(model_data), encoding='utf-8')

    for data in model_data:
        complete = ', '.join(map(str, sorted(data.points))) or 'ninguno'
        print(f'{data.spec.label}: N completos = {complete}')
    print(f'Gráfica: {OUTPUT_SVG.resolve()}')
    print(f'Datos: {OUTPUT_CSV.resolve()}')
    print(f'Tabla neutral: {OUTPUT_TABLE.resolve()}')


if __name__ == '__main__':
    main()
