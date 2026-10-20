#!/usr/bin/env python3
'''Evalúa Llama directamente con todos los temas del catálogo.'''

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_OUTPUT = (
    SCRIPT_DIR / 'results_llama3_1_8b' / 'llama_top_1_results.json'
)
DEFAULT_CHECKPOINT_EVERY = 10
PROMPT_VERSION = 'mteb_forced_topic_selection_v1_all_topics_direct_v1'
CANDIDATE_MODE = 'complete_topic_catalog'
CANDIDATE_SOURCE = 'temas.json complete catalog'

if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

import evaluate_ollama_top_n_no_none as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Entrega directamente a Llama todos los temas de temas.json para '
            'cada noticia de test y obliga al modelo a devolver uno.'
        )
    )
    parser.add_argument(
        '--model',
        default=os.environ.get('OLLAMA_MODEL', 'llama3.1:8b'),
        help='Modelo disponible en Ollama.',
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DATASET_DIR / 'data_test',
        help='Directorio con las noticias de test sin etiquetas.',
    )
    parser.add_argument(
        '--labelled-data-dir',
        type=Path,
        default=DATASET_DIR / 'data_test_labelled',
        help='Copia etiquetada de test, usada únicamente para métricas.',
    )
    parser.add_argument(
        '--topics',
        type=Path,
        default=DATASET_DIR / 'temas.json',
        help='Catálogo completo de temas y definiciones.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help='Informe y checkpoint de salida.',
    )
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help='Guarda un checkpoint después de este número de noticias.',
    )
    return parser.parse_args()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_test_records(data_dir: Path) -> list[dict[str, str]]:
    if not data_dir.is_dir():
        raise SystemExit(f'No se encontró el directorio: {data_dir}')
    paths = sorted(data_dir.glob('*.json'))
    if not paths:
        raise SystemExit(f'No se encontraron archivos JSON en {data_dir}.')

    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for path in paths:
        payload = base.load_json(path)
        source_records = payload if isinstance(payload, list) else [payload]
        if not isinstance(payload, (dict, list)):
            raise SystemExit(
                f'{path} debe contener un objeto o una lista JSON.'
            )
        for record_index, record in enumerate(source_records):
            location = f'{path}, elemento {record_index}'
            if not isinstance(record, dict):
                raise SystemExit(f'{location}: se esperaba un objeto JSON.')
            news_id = base.required_string(record, '_id', location)
            if news_id in seen_ids:
                raise SystemExit(f'{location}: _id duplicado {news_id!r}.')
            seen_ids.add(news_id)
            records.append(
                {
                    '_id': news_id,
                    'file': path.name,
                    'text': base.required_string(record, 'text', location),
                }
            )
    return records


def load_labelled_records(
    data_dir: Path, topic_catalog: set[str]
) -> dict[str, dict[str, Any]]:
    if not data_dir.is_dir():
        raise SystemExit(f'No se encontró el directorio: {data_dir}')
    paths = sorted(data_dir.glob('*.json'))
    if not paths:
        raise SystemExit(f'No se encontraron archivos JSON en {data_dir}.')

    labelled_by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = base.load_json(path)
        if not isinstance(payload, list):
            raise SystemExit(f'{path} debe contener una lista JSON.')
        for record_index, record in enumerate(payload):
            location = f'{path}, elemento {record_index}'
            if not isinstance(record, dict):
                raise SystemExit(f'{location}: se esperaba un objeto JSON.')
            news_id = base.required_string(record, '_id', location)
            if news_id in labelled_by_id:
                raise SystemExit(f'{location}: _id duplicado {news_id!r}.')
            topics = base.normalize_real_topics(
                record.get('tema'), location, topic_catalog
            )
            if topics is None:
                raise SystemExit(
                    f'{location}: test debe tener una etiqueta del catálogo.'
                )
            labelled_by_id[news_id] = {
                'text': base.required_string(record, 'text', location),
                'topics': topics,
            }
    return labelled_by_id


def prepare_records(
    test_records: list[dict[str, str]],
    labelled_by_id: dict[str, dict[str, Any]],
    topics: dict[str, str],
) -> list[dict[str, Any]]:
    test_by_id = {record['_id']: record for record in test_records}
    if set(test_by_id) != set(labelled_by_id):
        missing = set(labelled_by_id) - set(test_by_id)
        unexpected = set(test_by_id) - set(labelled_by_id)
        raise SystemExit(
            'Los IDs de data_test y data_test_labelled no coinciden: '
            f'{len(missing)} ausentes y {len(unexpected)} inesperados.'
        )

    all_candidates = [{'tema': topic} for topic in topics]
    prepared: list[dict[str, Any]] = []
    for record in test_records:
        news_id = record['_id']
        truth = labelled_by_id[news_id]
        if record['text'] != truth['text']:
            raise SystemExit(
                'El texto de data_test no coincide con la copia etiquetada '
                f'para {news_id!r}.'
            )
        prepared.append(
            {
                '_id': record['_id'],
                'file': record['file'],
                'text': record['text'],
                'topics_correctos': list(truth['topics']),
                'candidates': [dict(candidate) for candidate in all_candidates],
            }
        )
    return prepared


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
    return {
        'lower': round(max(0.0, centre - margin), 6),
        'upper': round(min(1.0, centre + margin), 6),
        'method': 'Wilson score interval',
    }


def build_report(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    topics: dict[str, str],
    context: dict[str, Any],
) -> dict[str, Any]:
    hits = sum(prediction['matches_real_topic'] for prediction in predictions)
    evaluated = len(predictions)
    accuracy = hits / evaluated if evaluated else 0.0
    elapsed_times = [
        float(prediction['inference_time_seconds'])
        for prediction in predictions
    ]
    return {
        'status': 'complete' if evaluated == len(records) else 'in_progress',
        'configuration': {
            'model': base.MODEL_NAME,
            'prompt_version': PROMPT_VERSION,
            'decision_strategy': 'forced_topic_selection',
            'prediction_format': base.PREDICTION_FORMAT,
            'candidate_mode': CANDIDATE_MODE,
            'candidate_source': CANDIDATE_SOURCE,
            'topic_count': len(topics),
            'uses_retriever': False,
            'uses_e5_candidates': False,
            'uses_bge_candidates': False,
            'uses_precomputed_candidates': False,
            'ground_truth_source': 'data_test_labelled',
            'labels_used_for': 'metrics only',
            'generation': 'Ollama API (temperature=0, seed=0)',
            'metric_definitions': {
                'llama_final_accuracy': (
                    'fraction of final Llama decisions matching the correct '
                    'topic'
                ),
            },
            'api_base_url': base.get_base_url(),
            'data_dir': context['data_dir'],
            'labelled_data_dir': context['labelled_data_dir'],
            'input_data_sha256': context['input_data_sha256'],
            'topic_catalog_sha256': context['topic_catalog_sha256'],
            'labelled_test_sha256': context['labelled_test_sha256'],
        },
        'dataset': {
            'total_news': len(records),
            'candidate_topics_per_news': len(topics),
            'expected_predictions': len(records),
            'completed_predictions': evaluated,
        },
        'metrics': {
            'evaluated_news': evaluated,
            'llama_final_hits': hits,
            'llama_final_misses': evaluated - hits,
            'llama_final_accuracy': round(accuracy, 6),
            'llama_final_accuracy_percent': round(accuracy * 100, 2),
            'llama_final_accuracy_confidence_interval_95': wilson_interval(
                hits, evaluated
            ),
            'invalid_decisions': sum(
                prediction[base.DECISION_FIELD] == 'invalid'
                for prediction in predictions
            ),
        },
        'timing': {
            'total_llama_inference_time_seconds': round(
                sum(elapsed_times), 6
            ),
            'mean_llama_inference_time_seconds': (
                round(statistics.fmean(elapsed_times), 6)
                if elapsed_times
                else None
            ),
            'scope': 'Una llamada de clasificación a Llama por noticia.',
        },
        'predictions': predictions,
    }


def load_checkpoint(
    path: Path,
    records: list[dict[str, Any]],
    topic_names: list[str],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    report = base.load_json(path)
    if not isinstance(report, dict):
        raise SystemExit(f'Checkpoint no válido: {path}')
    configuration = report.get('configuration')
    expected = {
        'model': base.MODEL_NAME,
        'prompt_version': PROMPT_VERSION,
        'candidate_mode': CANDIDATE_MODE,
        'topic_count': len(topic_names),
        'input_data_sha256': context['input_data_sha256'],
        'topic_catalog_sha256': context['topic_catalog_sha256'],
        'labelled_test_sha256': context['labelled_test_sha256'],
    }
    if not isinstance(configuration, dict) or any(
        configuration.get(key) != value for key, value in expected.items()
    ):
        raise SystemExit(
            f'{path} usa otra configuración o dataset. Muévelo antes de '
            'ejecutar este experimento.'
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
        if prediction.get('candidate_topics') != topic_names:
            raise SystemExit(
                f'{location}: los candidatos no coinciden con temas.json.'
            )
        decision = base.validate_saved_decision(
            prediction.get(base.DECISION_FIELD), topic_names, location
        )
        if not isinstance(decision, str):
            raise SystemExit(f'{location}: Llama debe devolver un único tema.')
        prediction[base.DECISION_FIELD] = decision
        base.validate_prediction_outputs(
            prediction, topic_names, decision, location
        )
        record = records_by_id[news_id]
        prediction['file'] = record['file']
        prediction['topics_correctos'] = list(record['topics_correctos'])
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


def main() -> None:
    args = parse_args()
    if args.checkpoint_every < 1:
        raise SystemExit('--checkpoint-every debe ser mayor que cero.')

    base.MODEL_NAME = args.model
    base.MODEL_LABEL = f'Ollama {args.model}'
    base.ALLOW_NONE_DECISION = False
    base.PROMPT_VERSION = 'mteb_forced_topic_selection_v1'

    topics = base.load_temas(args.topics)
    topic_names = list(topics)
    test_records = load_test_records(args.data_dir)
    labelled_by_id = load_labelled_records(
        args.labelled_data_dir, set(topic_names)
    )
    records = prepare_records(test_records, labelled_by_id, topics)
    normalized_labels = [
        {
            '_id': record['_id'],
            'text': labelled_by_id[record['_id']]['text'],
            'topics': labelled_by_id[record['_id']]['topics'],
        }
        for record in test_records
    ]
    context = {
        'data_dir': str(args.data_dir.resolve()),
        'labelled_data_dir': str(args.labelled_data_dir.resolve()),
        'input_data_sha256': sha256_json(test_records),
        'topic_catalog_sha256': sha256_json(topics),
        'labelled_test_sha256': sha256_json(normalized_labels),
    }

    predictions = load_checkpoint(
        args.output, records, topic_names, context
    )
    completed_ids = {prediction['_id'] for prediction in predictions}
    pending = [
        record for record in records if record['_id'] not in completed_ids
    ]
    order = {record['_id']: index for index, record in enumerate(records)}

    def checkpoint() -> dict[str, Any]:
        predictions.sort(key=lambda prediction: order[prediction['_id']])
        report = build_report(records, predictions, topics, context)
        base.save_report(args.output, report)
        return report

    if not pending:
        report = checkpoint()
        accuracy = report['metrics']['llama_final_accuracy_percent']
        print(f'Resultados ya completos: {accuracy:.2f}%')
        return

    print(f'Noticias: {len(records)}')
    print(f'Candidatos por noticia: {len(topic_names)}')
    print(f'Decisiones pendientes: {len(pending)}')
    client, _, _ = base.load_model()
    generated_since_checkpoint = 0

    try:
        for pending_index, record in enumerate(pending, start=1):
            start = time.perf_counter()
            decision, validity_raw_output, raw_output = base.generate_decision(
                client,
                None,
                None,
                record['text'],
                record['candidates'],
                topics,
            )
            elapsed = time.perf_counter() - start
            if decision != 'invalid' and decision not in topic_names:
                raise RuntimeError(
                    f'Ollama devolvió un tema fuera del catálogo: {decision!r}.'
                )
            predictions.append(
                {
                    '_id': record['_id'],
                    'file': record['file'],
                    'topics_correctos': record['topics_correctos'],
                    'candidate_topics': topic_names,
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
            if generated_since_checkpoint >= args.checkpoint_every:
                checkpoint()
                print(f'Progreso: {pending_index}/{len(pending)} pendientes')
                generated_since_checkpoint = 0
    except BaseException:
        checkpoint()
        print(f'Checkpoint guardado tras una interrupción: {args.output}')
        raise

    report = checkpoint()
    accuracy = report['metrics']['llama_final_accuracy_percent']
    print(f'Resultados guardados: {args.output}')
    print(f'Llama final accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    main()
