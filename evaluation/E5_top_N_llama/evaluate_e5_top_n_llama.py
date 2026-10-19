#!/usr/bin/env python3
'''Evalúa Llama con los Top-N candidatos producidos directamente por E5.'''

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EVALUATION_DIR.parent
DATASET_DIR = PROJECT_DIR / 'dataset_mteb'
RESULTS_DIR = SCRIPT_DIR / 'results'
E5_MODEL_NAME = 'intfloat/multilingual-e5-base'
DEFAULT_N_VALUES = list(range(2, 13))
DEFAULT_CHECKPOINT_EVERY = 10
PROMPT_VERSION = 'mteb_forced_topic_selection_v1_e5_top_n_v1'

if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

import evaluate_ollama_top_n_no_none as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Ejecuta Llama con los primeros N temas del ranking de similitud '
            'E5. Sin valores N procesa del 2 al 12.'
        )
    )
    parser.add_argument(
        'n_values',
        metavar='N',
        type=int,
        nargs='*',
        help='Tamaños Top-N que se evaluarán (por defecto: 2..12).',
    )
    parser.add_argument(
        '--model',
        default=os.environ.get('OLLAMA_MODEL', 'llama3.2:1b'),
        help='Modelo disponible en Ollama.',
    )
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help='Guarda el informe después de este número de predicciones.',
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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)
        file_handle.write('\n')
    temporary_path.replace(path)


def load_records(
    topics: dict[str, str], truth_by_id: dict[str, list[str] | None]
) -> list[dict[str, Any]]:
    input_dir = DATASET_DIR / 'semantic_similarity_results'
    input_paths = sorted(input_dir.glob('*.json'))
    if not input_paths:
        raise SystemExit(f'No se encontraron resultados E5 en {input_dir}.')

    topic_count = len(topics)
    top_key = f'top_{topic_count}'
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for input_path in input_paths:
        payload = load_json(input_path)
        if not isinstance(payload, list):
            raise SystemExit(f'{input_path} debe contener una lista JSON.')
        for record_index, record in enumerate(payload):
            location = f'{input_path}, elemento {record_index}'
            if not isinstance(record, dict):
                raise SystemExit(f'{location}: se esperaba un objeto JSON.')
            news_id = base.required_string(record, '_id', location)
            if news_id in seen_ids:
                raise SystemExit(f'{location}: _id duplicado {news_id!r}.')
            seen_ids.add(news_id)
            if news_id not in truth_by_id:
                continue

            source_file = base.required_string(record, 'file', location)
            text = base.required_string(record, 'text', location)
            candidates = record.get(top_key)
            if not isinstance(candidates, list) or len(candidates) != topic_count:
                raise SystemExit(
                    f'{location}: {top_key} debe contener {topic_count} temas.'
                )

            prepared_candidates: list[dict[str, str]] = []
            seen_topics: set[str] = set()
            for candidate_index, candidate in enumerate(candidates):
                candidate_location = f'{location}, candidato {candidate_index}'
                if not isinstance(candidate, dict):
                    raise SystemExit(
                        f'{candidate_location}: se esperaba un objeto JSON.'
                    )
                topic = base.required_string(
                    candidate, 'tema', candidate_location
                )
                if topic not in topics or topic in seen_topics:
                    raise SystemExit(
                        f'{candidate_location}: tema desconocido o duplicado '
                        f'{topic!r}.'
                    )
                seen_topics.add(topic)
                prepared_candidates.append({'tema': topic})
            if seen_topics != set(topics):
                raise SystemExit(
                    f'{location}: {top_key} no contiene todo el catálogo.'
                )

            correct_topics = truth_by_id[news_id]
            if correct_topics is None:
                raise SystemExit(
                    f'{location}: el dataset MTEB no admite "ninguno".'
                )
            records.append(
                {
                    '_id': news_id,
                    'file': source_file,
                    'text': text,
                    'topics_correctos': correct_topics,
                    'candidates': prepared_candidates,
                }
            )

    missing_ids = set(truth_by_id) - seen_ids
    if missing_ids:
        raise SystemExit(
            f'Faltan resultados E5 para {len(missing_ids)} noticias.'
        )
    return records


def wilson_interval(hits: int, total: int) -> dict[str, Any] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = hits / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return {
        'lower': round(lower, 6),
        'upper': round(upper, 6),
        'lower_percent': round(lower * 100, 2),
        'upper_percent': round(upper * 100, 2),
        'method': 'Wilson score interval',
    }


def mean_time_summary(elapsed_times: list[float]) -> dict[str, Any]:
    sample_size = len(elapsed_times)
    if sample_size:
        mean = statistics.fmean(elapsed_times)
        if sample_size > 1:
            margin = (
                1.959963984540054
                * statistics.stdev(elapsed_times)
                / math.sqrt(sample_size)
            )
        else:
            margin = 0.0
        interval = {
            'lower': round(max(0.0, mean - margin), 6),
            'upper': round(mean + margin, 6),
        }
    else:
        mean = 0.0
        interval = {'lower': 0.0, 'upper': 0.0}
    return {
        'sample_size': sample_size,
        'mean_seconds': round(mean, 6),
        'confidence_interval_95_seconds': interval,
        'method': 'Normal approximation for the sample mean',
    }


def build_report(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    topics: dict[str, str],
    n: int,
) -> dict[str, Any]:
    hits = sum(prediction['matches_real_topic'] for prediction in predictions)
    evaluated = len(predictions)
    accuracy = hits / evaluated if evaluated else 0.0
    candidate_recall_hits = sum(
        any(
            topic in {candidate['tema'] for candidate in record['candidates'][:n]}
            for topic in record['topics_correctos']
        )
        for record in records
    )
    candidate_recall = candidate_recall_hits / len(records) if records else 0.0
    elapsed_times = [
        float(prediction['inference_time_seconds']) for prediction in predictions
    ]
    return {
        'status': 'complete' if evaluated == len(records) else 'in_progress',
        'configuration': {
            'e5_model': E5_MODEL_NAME,
            'llama_model': base.MODEL_NAME,
            'candidate_source': 'E5 cosine-similarity ranking',
            'n': n,
            'topic_count': len(topics),
            'prompt_version': PROMPT_VERSION,
            'decision_strategy': 'forced_topic_selection',
            'prediction_format': base.PREDICTION_FORMAT,
            'generation': 'Ollama API (temperature=0, seed=0)',
            'api_base_url': base.get_base_url(),
        },
        'dataset': {
            'total_news': len(records),
            'expected_predictions': len(records),
            'completed_predictions': evaluated,
        },
        'metrics': {
            'evaluated_news': evaluated,
            'top_1_hits': hits,
            'top_1_misses': evaluated - hits,
            'top_1_accuracy_mean': round(accuracy, 6),
            'top_1_accuracy_mean_percent': round(accuracy * 100, 2),
            'top_1_accuracy_confidence_interval_95': wilson_interval(
                hits, evaluated
            ),
            'candidate_recall_hits': candidate_recall_hits,
            'candidate_recall': round(candidate_recall, 6),
            'candidate_recall_percent': round(candidate_recall * 100, 2),
        },
        'timing': {
            'total_inference_time_seconds': round(sum(elapsed_times), 6),
            'per_news_inference_time': mean_time_summary(elapsed_times),
            'scope': (
                'Tiempo medido alrededor de cada selección mediante la API '
                'de Ollama; incluye la llamada HTTP y la inferencia.'
            ),
        },
        'predictions': predictions,
    }


def load_checkpoint(
    path: Path,
    records: list[dict[str, Any]],
    n: int,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    report = load_json(path)
    if not isinstance(report, dict):
        raise SystemExit(f'Checkpoint no válido: {path}')
    configuration = report.get('configuration')
    expected = {
        'e5_model': E5_MODEL_NAME,
        'llama_model': base.MODEL_NAME,
        'candidate_source': 'E5 cosine-similarity ranking',
        'n': n,
        'prompt_version': PROMPT_VERSION,
    }
    if not isinstance(configuration, dict) or any(
        configuration.get(key) != value for key, value in expected.items()
    ):
        raise SystemExit(
            f'{path} usa otra configuración. Muévelo antes de ejecutar '
            'este experimento.'
        )
    predictions = report.get('predictions')
    if not isinstance(predictions, list):
        raise SystemExit(f'{path} no contiene una lista predictions.')

    records_by_id = {record['_id']: record for record in records}
    seen_ids: set[str] = set()
    for index, prediction in enumerate(predictions):
        location = f'{path}, predicción {index}'
        if not isinstance(prediction, dict):
            raise SystemExit(f'{location}: se esperaba un objeto JSON.')
        news_id = base.required_string(prediction, '_id', location)
        if news_id not in records_by_id or news_id in seen_ids:
            raise SystemExit(f'{location}: _id desconocido o duplicado.')
        seen_ids.add(news_id)
        record = records_by_id[news_id]
        candidate_names = [
            candidate['tema'] for candidate in record['candidates'][:n]
        ]
        if prediction.get('n') != n or prediction.get(
            'candidate_topics'
        ) != candidate_names:
            raise SystemExit(f'{location}: candidatos inconsistentes.')
        decision = base.validate_saved_decision(
            prediction.get(base.DECISION_FIELD), candidate_names, location
        )
        prediction[base.DECISION_FIELD] = decision
        base.validate_prediction_outputs(
            prediction, candidate_names, decision, location
        )
        prediction['topics_correctos'] = list(record['topics_correctos'])
        prediction['candidate_contains_real_topic'] = any(
            topic in candidate_names for topic in record['topics_correctos']
        )
        prediction['matches_real_topic'] = base.decision_matches_topics(
            decision, record['topics_correctos']
        )
        elapsed = prediction.get('inference_time_seconds')
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise SystemExit(f'{location}: tiempo de inferencia inválido.')
    return predictions


def evaluate_n(
    records: list[dict[str, Any]],
    topics: dict[str, str],
    n: int,
    checkpoint_every: int,
    client_holder: list[base.OllamaClient],
) -> None:
    output_path = RESULTS_DIR / f'e5_top_{n}_llama_results.json'
    predictions = load_checkpoint(output_path, records, n)
    completed_ids = {prediction['_id'] for prediction in predictions}
    pending = [record for record in records if record['_id'] not in completed_ids]
    order = {record['_id']: index for index, record in enumerate(records)}

    def checkpoint() -> dict[str, Any]:
        predictions.sort(key=lambda prediction: order[prediction['_id']])
        report = build_report(records, predictions, topics, n)
        save_json(output_path, report)
        return report

    if not pending:
        report = checkpoint()
        print(
            f"N={n}: completo, {report['metrics']['top_1_accuracy_mean_percent']:.2f}%"
        )
        return

    if not client_holder:
        client, _, _ = base.load_model()
        client_holder.append(client)
    client = client_holder[0]
    generated_since_checkpoint = 0

    try:
        for pending_index, record in enumerate(pending, start=1):
            candidates = record['candidates'][:n]
            start = time.perf_counter()
            decision, validity_raw_output, raw_output = base.generate_decision(
                client, None, None, record['text'], candidates, topics
            )
            elapsed = time.perf_counter() - start
            candidate_names = [candidate['tema'] for candidate in candidates]
            predictions.append(
                {
                    '_id': record['_id'],
                    'file': record['file'],
                    'topics_correctos': record['topics_correctos'],
                    'n': n,
                    'candidate_topics': candidate_names,
                    'candidate_contains_real_topic': any(
                        topic in candidate_names
                        for topic in record['topics_correctos']
                    ),
                    base.DECISION_FIELD: decision,
                    base.VALIDITY_RAW_OUTPUT_FIELD: validity_raw_output,
                    base.RAW_OUTPUT_FIELD: raw_output,
                    'matches_real_topic': base.decision_matches_topics(
                        decision, record['topics_correctos']
                    ),
                    'inference_time_seconds': round(elapsed, 6),
                }
            )
            generated_since_checkpoint += 1
            if generated_since_checkpoint >= checkpoint_every:
                checkpoint()
                print(f'N={n}: {pending_index}/{len(pending)} pendientes')
                generated_since_checkpoint = 0
    except BaseException:
        checkpoint()
        print(f'N={n}: checkpoint guardado tras una interrupción.')
        raise

    report = checkpoint()
    print(
        f"N={n}: completo, {report['metrics']['top_1_accuracy_mean_percent']:.2f}%"
    )


def main() -> None:
    args = parse_args()
    if args.checkpoint_every < 1:
        raise SystemExit('--checkpoint-every debe ser mayor que cero.')

    topics = base.load_temas(DATASET_DIR / 'temas.json')
    n_values = args.n_values or DEFAULT_N_VALUES
    if n_values != sorted(set(n_values)):
        raise SystemExit('Los valores N deben estar ordenados y no repetirse.')
    if any(n < 1 or n > len(topics) for n in n_values):
        raise SystemExit(f'N debe estar entre 1 y {len(topics)}.')

    base.MODEL_NAME = args.model
    base.MODEL_LABEL = f'Ollama {args.model}'
    base.ALLOW_NONE_DECISION = False
    base.PROMPT_VERSION = 'mteb_forced_topic_selection_v1'

    truth = base.load_ground_truth(
        DATASET_DIR / 'data_validation', set(topics)
    )
    records = load_records(topics, truth)
    print(f'Noticias: {len(records)}; valores N: {n_values}')

    client_holder: list[base.OllamaClient] = []
    for n in n_values:
        evaluate_n(
            records,
            topics,
            n,
            args.checkpoint_every,
            client_holder,
        )


if __name__ == '__main__':
    main()
