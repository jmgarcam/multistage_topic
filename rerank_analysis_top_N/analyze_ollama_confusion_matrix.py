#!/usr/bin/env python3
'''Calcula y representa matrices de confusión para la evaluación top-N.'''

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_FILE = SCRIPT_DIR / 'ollama_llama_3_1_8b_top_n_results.json'
DEFAULT_VALIDATION_DIR = SCRIPT_DIR.parent / 'data_validation'
DEFAULT_TOPICS_FILE = SCRIPT_DIR.parent / 'temas.json'
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'confusion_matrices'
DEFAULT_DECISION_FIELD = 'ollama_decision'
NONE_TOPIC = 'ninguno'
INVALID_LABEL = 'invalid'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Calcula matrices de confusión entre las etiquetas de '
            'data_validation y las predicciones de Ollama.'
        )
    )
    parser.add_argument(
        '--results',
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help='Informe JSON con las predicciones de Ollama.',
    )
    parser.add_argument(
        '--validation-dir',
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
        help='Directorio con los JSON usados como ground truth.',
    )
    parser.add_argument(
        '--topics',
        type=Path,
        default=DEFAULT_TOPICS_FILE,
        help='Catálogo temas.json que fija el orden de las clases.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Directorio donde se guardarán tablas y figuras.',
    )
    parser.add_argument(
        '--n',
        nargs='+',
        type=int,
        help='Valores de N que se representarán; por defecto, todos.',
    )
    parser.add_argument(
        '--decision-field',
        default=DEFAULT_DECISION_FIELD,
        help='Campo que contiene la decisión del modelo.',
    )
    parser.add_argument(
        '--top-confusions',
        type=int,
        default=15,
        help='Número de confusiones principales que se muestran por consola.',
    )
    args = parser.parse_args()
    if args.n is not None and any(n < 1 for n in args.n):
        parser.error('Todos los valores de --n deben ser positivos.')
    if args.top_confusions < 0:
        parser.error('--top-confusions no puede ser negativo.')
    return args


def load_json(path: Path) -> Any:
    try:
        with path.open('r', encoding='utf-8') as file_handle:
            return json.load(file_handle)
    except FileNotFoundError as exc:
        raise SystemExit(f'No se encontró el archivo: {path}') from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f'JSON no válido en {path}: {exc}') from exc


def required_string(record: dict[str, Any], field: str, location: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f'{location}: falta un campo {field} no vacío.')
    return value.strip()


def load_topic_names(path: Path) -> list[str]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise SystemExit(f'{path} debe contener un objeto JSON no vacío.')
    topic_names: list[str] = []
    for topic, definition in payload.items():
        if (
            not isinstance(topic, str)
            or not topic.strip()
            or not isinstance(definition, str)
            or not definition.strip()
        ):
            raise SystemExit(
                f'{path}: todos los temas y definiciones deben ser textos '
                'no vacíos.'
            )
        topic = topic.strip()
        if topic.casefold() in {NONE_TOPIC, INVALID_LABEL}:
            raise SystemExit(
                f'{path}: {topic!r} está reservado y no puede ser un tema.'
            )
        if topic in topic_names:
            raise SystemExit(f'{path}: tema duplicado {topic!r}.')
        topic_names.append(topic)
    return topic_names


def normalize_ground_truth(
    value: Any, location: str, topic_catalog: set[str]
) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list) and value:
        values = value
    else:
        raise SystemExit(
            f'{location}: tema debe ser una cadena o una lista no vacía.'
        )

    normalized: list[str] = []
    for topic_index, topic in enumerate(values):
        if not isinstance(topic, str) or not topic.strip():
            raise SystemExit(
                f'{location}, tema {topic_index}: se esperaba un texto no vacío.'
            )
        topic = topic.strip()
        if topic.casefold() == NONE_TOPIC:
            topic = NONE_TOPIC
        if topic in normalized:
            raise SystemExit(f'{location}: tema duplicado {topic!r}.')
        normalized.append(topic)

    if NONE_TOPIC in normalized:
        if len(normalized) != 1:
            raise SystemExit(
                f'{location}: {NONE_TOPIC!r} no puede combinarse con otros temas.'
            )
        return normalized

    unknown = [topic for topic in normalized if topic not in topic_catalog]
    if unknown:
        raise SystemExit(f'{location}: temas desconocidos: {unknown}.')
    return normalized


def load_ground_truth(
    directory: Path, topic_catalog: set[str]
) -> dict[str, list[str]]:
    if not directory.is_dir():
        raise SystemExit(f'No se encontró el directorio: {directory}')
    paths = sorted(directory.glob('*.json'))
    if not paths:
        raise SystemExit(f'No se encontraron archivos JSON en {directory}.')

    truth_by_id: dict[str, list[str]] = {}
    locations: dict[str, str] = {}
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, list):
            raise SystemExit(f'{path} debe contener una lista JSON.')
        for record_index, record in enumerate(payload):
            location = f'{path}, elemento {record_index}'
            if not isinstance(record, dict):
                raise SystemExit(f'{location}: se esperaba un objeto JSON.')
            news_id = required_string(record, '_id', location)
            if news_id in truth_by_id:
                raise SystemExit(
                    f'{location}: _id duplicado; ya aparece en '
                    f'{locations[news_id]}.'
                )
            locations[news_id] = location
            truth_by_id[news_id] = normalize_ground_truth(
                record.get('tema'), location, topic_catalog
            )
    return truth_by_id


def load_predictions(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = load_json(path)
    if not isinstance(report, dict):
        raise SystemExit(f'{path} debe contener un objeto JSON.')
    predictions = report.get('predictions')
    if not isinstance(predictions, list):
        raise SystemExit(f'{path}: predictions debe ser una lista.')

    normalized: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()
    for prediction_index, prediction in enumerate(predictions):
        location = f'{path}, predicción {prediction_index}'
        if not isinstance(prediction, dict):
            raise SystemExit(f'{location}: se esperaba un objeto JSON.')
        news_id = required_string(prediction, '_id', location)
        n = prediction.get('n')
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise SystemExit(f'{location}: n debe ser un entero positivo.')
        key = (news_id, n)
        if key in seen_keys:
            raise SystemExit(f'{location}: combinación _id/N duplicada.')
        seen_keys.add(key)
        normalized.append(prediction)
    return report, normalized


def normalize_prediction(value: Any, location: str, labels: set[str]) -> str:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str):
        raise SystemExit(f'{location}: la decisión debe ser una cadena.')
    if value not in labels:
        raise SystemExit(f'{location}: decisión desconocida {value!r}.')
    return value


def build_confusion_matrix(
    predictions: list[dict[str, Any]],
    truth_by_id: dict[str, list[str]],
    topic_names: list[str],
    decision_field: str,
    n: int,
) -> tuple[
    list[list[float]],
    dict[str, int | float],
    Counter[tuple[str, str]],
    Counter[tuple[str, str]],
]:
    true_labels = [*topic_names, NONE_TOPIC]
    predicted_labels = [*true_labels, INVALID_LABEL]
    true_index = {label: index for index, label in enumerate(true_labels)}
    predicted_index = {
        label: index for index, label in enumerate(predicted_labels)
    }
    prediction_catalog = set(predicted_labels)
    topic_catalog = set(topic_names)
    matrix = [
        [0.0 for _ in predicted_labels]
        for _ in true_labels
    ]
    weighted_errors: Counter[tuple[str, str]] = Counter()
    affected_news: Counter[tuple[str, str]] = Counter()

    selected = [prediction for prediction in predictions if prediction['n'] == n]
    if not selected:
        raise SystemExit(f'No existen predicciones para N={n}.')

    correct = 0
    none_ground_truth = 0
    correct_none = 0
    predicted_none = 0
    invalid_predictions = 0
    multi_label_news = 0
    for prediction_index, prediction in enumerate(selected):
        location = f'N={n}, predicción {prediction_index}'
        news_id = prediction['_id']
        if news_id not in truth_by_id:
            raise SystemExit(
                f'{location}: el _id {news_id!r} no existe en data_validation.'
            )
        true_topics = truth_by_id[news_id]
        if len(true_topics) > 1:
            multi_label_news += 1

        saved_truth = prediction.get('topics_correctos')
        if saved_truth is not None:
            normalized_saved_truth = normalize_ground_truth(
                saved_truth, location, topic_catalog
            )
            if set(normalized_saved_truth) != set(true_topics):
                raise SystemExit(
                    f'{location}: topics_correctos no coincide con el ground '
                    'truth actual de data_validation.'
                )

        predicted = normalize_prediction(
            prediction.get(decision_field), location, prediction_catalog
        )
        is_correct = predicted in true_topics
        if is_correct:
            correct += 1
            matrix[true_index[predicted]][predicted_index[predicted]] += 1.0
        else:
            weight = 1.0 / len(true_topics)
            for true_topic in true_topics:
                matrix[true_index[true_topic]][predicted_index[predicted]] += weight
                weighted_errors[(true_topic, predicted)] += weight
                affected_news[(true_topic, predicted)] += 1

        if true_topics == [NONE_TOPIC]:
            none_ground_truth += 1
            if is_correct:
                correct_none += 1
        if predicted == NONE_TOPIC:
            predicted_none += 1
        if predicted == INVALID_LABEL:
            invalid_predictions += 1

    matrix_total = sum(sum(row) for row in matrix)
    if not math.isclose(matrix_total, len(selected), abs_tol=1e-9):
        raise RuntimeError(
            f'N={n}: la matriz suma {matrix_total}, pero existen '
            f'{len(selected)} predicciones.'
        )

    false_none = predicted_none - correct_none
    summary: dict[str, int | float] = {
        'n': n,
        'evaluated_news': len(selected),
        'ground_truth_news': len(truth_by_id),
        'missing_news': len(truth_by_id) - len(selected),
        'correct_predictions': correct,
        'incorrect_predictions': len(selected) - correct,
        'accuracy': correct / len(selected),
        'multi_label_news': multi_label_news,
        'ground_truth_none_news': none_ground_truth,
        'predicted_none_news': predicted_none,
        'correct_none_predictions': correct_none,
        'false_none_predictions': false_none,
        'none_recall': (
            correct_none / none_ground_truth if none_ground_truth else 0.0
        ),
        'none_precision': (
            correct_none / predicted_none if predicted_none else 0.0
        ),
        'invalid_predictions': invalid_predictions,
    }
    return matrix, summary, weighted_errors, affected_news


def normalize_matrix_rows(matrix: list[list[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in matrix:
        row_total = sum(row)
        normalized.append(
            [100.0 * value / row_total for value in row]
            if row_total
            else [0.0 for value in row]
        )
    return normalized


def format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f'{value:.6f}'.rstrip('0').rstrip('.')


def write_matrix_csv(
    path: Path,
    matrix: list[list[float]],
    true_labels: list[str],
    predicted_labels: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(['ground_truth / prediction', *predicted_labels])
        for label, row in zip(true_labels, matrix):
            writer.writerow([label, *(format_number(value) for value in row)])


def write_confusions_csv(
    path: Path,
    weighted_errors: Counter[tuple[str, str]],
    affected_news: Counter[tuple[str, str]],
) -> None:
    ordered_pairs = sorted(
        weighted_errors,
        key=lambda pair: (
            -weighted_errors[pair],
            pair[0].casefold(),
            pair[1].casefold(),
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(
            [
                'ground_truth',
                'prediction',
                'weighted_count',
                'affected_news',
            ]
        )
        for pair in ordered_pairs:
            writer.writerow(
                [
                    pair[0],
                    pair[1],
                    format_number(weighted_errors[pair]),
                    affected_news[pair],
                ]
            )


def heatmap_color(value: float, maximum: float) -> str:
    ratio = min(max(value / maximum if maximum else 0.0, 0.0), 1.0)
    start = (247, 251, 255)
    end = (8, 81, 156)
    red, green, blue = (
        round(start[index] + ratio * (end[index] - start[index]))
        for index in range(3)
    )
    return f'#{red:02x}{green:02x}{blue:02x}'


def plot_confusion_matrix(
    output_stem: Path,
    matrix: list[list[float]],
    true_labels: list[str],
    predicted_labels: list[str],
    summary: dict[str, int | float],
) -> list[Path]:
    percentages = normalize_matrix_rows(matrix)
    maximum = max(
        (value for row in percentages for value in row),
        default=0.0,
    )
    color_maximum = max(maximum, 1.0)
    cell_size = 52
    left_margin = 245
    top_margin = 245
    right_margin = 125
    bottom_margin = 80
    grid_width = len(predicted_labels) * cell_size
    grid_height = len(true_labels) * cell_size
    width = left_margin + grid_width + right_margin
    height = top_margin + grid_height + bottom_margin
    n = int(summary['n'])
    evaluated = int(summary['evaluated_news'])
    accuracy = float(summary['accuracy']) * 100
    font = 'DejaVu Sans, Arial, sans-serif'
    lines = [
        f'''<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>''',
        '''<rect width='100%' height='100%' fill='white'/>''',
        f'''<text x='{width / 2:.1f}' y='30' text-anchor='middle' font-family='{font}' font-size='18' font-weight='bold'>Matriz de confusión normalizada por tema real — N={n}</text>''',
        f'''<text x='{width / 2:.1f}' y='54' text-anchor='middle' font-family='{font}' font-size='13'>{evaluated} noticias; accuracy={accuracy:.2f}%</text>''',
        f'''<text x='{left_margin + grid_width / 2:.1f}' y='82' text-anchor='middle' font-family='{font}' font-size='14' font-weight='bold'>Predicción de Ollama</text>''',
        f'''<text x='22' y='{top_margin + grid_height / 2:.1f}' text-anchor='middle' font-family='{font}' font-size='14' font-weight='bold' transform='rotate(-90 22 {top_margin + grid_height / 2:.1f})'>Ground truth de data_validation</text>''',
    ]

    for column_index, label in enumerate(predicted_labels):
        x = left_margin + column_index * cell_size + cell_size / 2
        y = top_margin - 12
        lines.append(
            f'''<text x='{x:.1f}' y='{y}' text-anchor='start' font-family='{font}' font-size='10' transform='rotate(-52 {x:.1f} {y})'>{escape(label)}</text>'''
        )

    threshold = color_maximum * 0.55
    for row_index, (label, row) in enumerate(zip(true_labels, percentages)):
        y = top_margin + row_index * cell_size
        lines.append(
            f'''<text x='{left_margin - 10}' y='{y + cell_size / 2 + 4:.1f}' text-anchor='end' font-family='{font}' font-size='10'>{escape(label)}</text>'''
        )
        for column_index, value in enumerate(row):
            x = left_margin + column_index * cell_size
            color = heatmap_color(value, color_maximum)
            lines.append(
                f'''<rect x='{x}' y='{y}' width='{cell_size}' height='{cell_size}' fill='{color}' stroke='#d0d7de' stroke-width='0.7'/>'''
            )
            if value <= 0:
                continue
            annotation = f'{value:.0f}' if value >= 10 else f'{value:.1f}'
            text_color = 'white' if value >= threshold else '#202020'
            lines.append(
                f'''<text x='{x + cell_size / 2:.1f}' y='{y + cell_size / 2 + 3:.1f}' text-anchor='middle' font-family='{font}' font-size='9' fill='{text_color}'>{annotation}</text>'''
            )

    legend_x = left_margin + grid_width + 32
    legend_y = top_margin
    legend_height = grid_height
    legend_steps = 40
    for step in range(legend_steps):
        fraction = 1.0 - step / (legend_steps - 1)
        value = fraction * color_maximum
        y = legend_y + step * legend_height / legend_steps
        lines.append(
            f'''<rect x='{legend_x}' y='{y:.2f}' width='18' height='{legend_height / legend_steps + 0.5:.2f}' fill='{heatmap_color(value, color_maximum)}'/>'''
        )
    lines.extend(
        [
            f'''<text x='{legend_x + 26}' y='{legend_y + 8}' font-family='{font}' font-size='9'>{color_maximum:.1f}%</text>''',
            f'''<text x='{legend_x + 26}' y='{legend_y + legend_height}' font-family='{font}' font-size='9'>0%</text>''',
            f'''<text x='{width / 2:.1f}' y='{height - 20}' text-anchor='middle' font-family='{font}' font-size='10' fill='#404040'>En noticias multietiqueta, los aciertos se asignan a la etiqueta elegida y los errores se reparten entre las etiquetas válidas.</text>''',
            '''</svg>''',
        ]
    )

    output_path = output_stem.with_suffix('.svg')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file_handle:
        file_handle.write('\n'.join(lines))
        file_handle.write('\n')
    return [output_path]


def print_top_confusions(
    n: int,
    weighted_errors: Counter[tuple[str, str]],
    limit: int,
) -> None:
    if limit == 0:
        return
    ordered = sorted(
        weighted_errors.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )
    print(f'Principales confusiones para N={n}:')
    for (true_topic, predicted_topic), weight in ordered[:limit]:
        print(
            f'  {true_topic} -> {predicted_topic}: '
            f'{format_number(weight)}'
        )
    if not ordered:
        print('  Ninguna.')


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)
        file_handle.write('\n')


def main() -> None:
    args = parse_args()
    topic_names = load_topic_names(args.topics)
    true_labels = [*topic_names, NONE_TOPIC]
    predicted_labels = [*true_labels, INVALID_LABEL]
    truth_by_id = load_ground_truth(args.validation_dir, set(topic_names))
    report, predictions = load_predictions(args.results)
    available_n = sorted({prediction['n'] for prediction in predictions})
    requested_n = sorted(set(args.n)) if args.n is not None else available_n
    missing_n = [n for n in requested_n if n not in available_n]
    if missing_n:
        raise SystemExit(
            f'No existen predicciones para los valores de N: {missing_n}.'
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, int | float]] = []
    generated_files: list[str] = []
    for n in requested_n:
        matrix, summary, weighted_errors, affected_news = (
            build_confusion_matrix(
                predictions,
                truth_by_id,
                topic_names,
                args.decision_field,
                n,
            )
        )
        percentages = normalize_matrix_rows(matrix)
        stem = args.output_dir / f'confusion_matrix_n_{n}'
        counts_path = args.output_dir / f'confusion_matrix_n_{n}_counts.csv'
        percentages_path = (
            args.output_dir / f'confusion_matrix_n_{n}_row_percent.csv'
        )
        confusions_path = args.output_dir / f'confusions_n_{n}.csv'
        write_matrix_csv(counts_path, matrix, true_labels, predicted_labels)
        write_matrix_csv(
            percentages_path,
            percentages,
            true_labels,
            predicted_labels,
        )
        write_confusions_csv(
            confusions_path, weighted_errors, affected_news
        )
        figure_paths = plot_confusion_matrix(
            stem, matrix, true_labels, predicted_labels, summary
        )
        summaries.append(summary)
        generated_files.extend(
            str(path)
            for path in [
                counts_path,
                percentages_path,
                confusions_path,
                *figure_paths,
            ]
        )
        correct_count = int(summary['correct_predictions'])
        evaluated_count = int(summary['evaluated_news'])
        accuracy_percent = float(summary['accuracy']) * 100
        print(
            f'N={n}: {correct_count}/{evaluated_count} aciertos '
            f'({accuracy_percent:.2f}%).'
        )
        print_top_confusions(n, weighted_errors, args.top_confusions)

    summary_path = args.output_dir / 'confusion_summary.json'
    save_json(
        summary_path,
        {
            'source_results': str(args.results.resolve()),
            'source_ground_truth': str(args.validation_dir.resolve()),
            'results_status': report.get('status'),
            'ground_truth_news': len(truth_by_id),
            'true_labels': true_labels,
            'predicted_labels': predicted_labels,
            'multi_label_policy': (
                'Los aciertos se asignan a la etiqueta predicha válida; '
                'los errores reparten peso uniforme entre las etiquetas reales.'
            ),
            'matrices': summaries,
            'generated_files': generated_files,
        },
    )
    print(f'Resumen guardado: {summary_path}')
    if report.get('status') != 'complete':
        print(
            'AVISO: el informe de Ollama está incompleto; las matrices reflejan '
            'solo las predicciones disponibles.'
        )


if __name__ == '__main__':
    main()
