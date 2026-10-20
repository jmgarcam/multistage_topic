#!/usr/bin/env python3
'''Evalúa BGE Top-1 directamente contra todo el catálogo de temas.'''

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from numbers import Real
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EVALUATION_DIR.parent
DATASET_DIR = PROJECT_DIR / 'dataset_mteb'
RESULTS_DIR = SCRIPT_DIR / 'results'
MODEL_NAME = 'BAAI/bge-reranker-v2-m3'
BATCH_SIZE = 16
MAX_LENGTH = 512
TOPIC_TEMPLATE = 'Tema: {topic}. Definición: {definition}'

if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

import evaluate_ollama_top_n_no_none as evaluation_base  # noqa: E402


ScoreFunction = Callable[[list[list[str]]], Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Evalúa cada noticia directamente contra todos los temas mediante '
            'BAAI/bge-reranker-v2-m3 y conserva el tema con mayor score.'
        )
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DATASET_DIR / 'data_test',
        help='Directorio con las noticias de entrada sin etiquetas.',
    )
    parser.add_argument(
        '--topics',
        type=Path,
        default=DATASET_DIR / 'temas.json',
        help='Catálogo completo de temas y definiciones.',
    )
    parser.add_argument(
        '--labelled-data-dir',
        '--ground-truth-dir',
        dest='labelled_data_dir',
        type=Path,
        default=DATASET_DIR / 'data_test_labelled',
        help='Copia etiquetada de test, utilizada únicamente para métricas.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=RESULTS_DIR / 'bge_top_1_results.json',
        help='Ruta del informe JSON de salida.',
    )
    parser.add_argument(
        '--model',
        default=MODEL_NAME,
        help='Modelo cross-encoder de FlagEmbedding.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help='Tamaño de lote para puntuar los pares texto/tema.',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=MAX_LENGTH,
        help='Longitud máxima de los pares tokenizados.',
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


def load_topics(path: Path) -> dict[str, str]:
    topics = load_json(path)
    if not isinstance(topics, dict) or not topics:
        raise SystemExit(f'{path} debe contener un objeto JSON no vacío.')
    for topic, definition in topics.items():
        if (
            not isinstance(topic, str)
            or not topic.strip()
            or not isinstance(definition, str)
            or not definition.strip()
        ):
            raise SystemExit(
                'Todos los temas y sus definiciones deben ser cadenas no vacías.'
            )
    return topics


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_texts(data_dir: Path) -> list[dict[str, str]]:
    if not data_dir.is_dir():
        raise SystemExit(f'No se encontró el directorio: {data_dir}')
    paths = sorted(data_dir.glob('*.json'))
    if not paths:
        raise SystemExit(f'No se encontraron archivos JSON en {data_dir}.')

    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for path in paths:
        payload = load_json(path)
        if isinstance(payload, dict):
            source_records = [payload]
        elif isinstance(payload, list):
            source_records = payload
        else:
            raise SystemExit(
                f'{path} debe contener un objeto o una lista JSON.'
            )
        for record_index, record in enumerate(source_records):
            location = f'{path}, elemento {record_index}'
            if not isinstance(record, dict):
                raise SystemExit(f'{location}: se esperaba un objeto JSON.')
            news_id = evaluation_base.required_string(record, '_id', location)
            if news_id in seen_ids:
                raise SystemExit(f'{location}: _id duplicado {news_id!r}.')
            seen_ids.add(news_id)
            text = evaluation_base.required_string(record, 'text', location)
            records.append({'_id': news_id, 'file': path.name, 'text': text})
    return records


def load_ground_truth(
    directory: Path, topic_catalog: set[str]
) -> dict[str, dict[str, Any]]:
    if not directory.is_dir():
        raise SystemExit(f'No se encontró el directorio: {directory}')
    paths = sorted(directory.glob('*.json'))
    if not paths:
        raise SystemExit(f'No se encontraron archivos JSON en {directory}.')

    truth_by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, list):
            raise SystemExit(f'{path} debe contener una lista JSON.')
        for record_index, record in enumerate(payload):
            location = f'{path}, elemento {record_index}'
            if not isinstance(record, dict):
                raise SystemExit(f'{location}: se esperaba un objeto JSON.')
            news_id = evaluation_base.required_string(record, '_id', location)
            if news_id in truth_by_id:
                raise SystemExit(f'{location}: _id duplicado {news_id!r}.')
            text = evaluation_base.required_string(record, 'text', location)
            topics = evaluation_base.normalize_real_topics(
                record.get('tema'), location, topic_catalog
            )
            if topics is None:
                raise SystemExit(
                    f'{location}: este dataset no debe contener ninguno.'
                )
            truth_by_id[news_id] = {'text': text, 'topics': topics}
    return truth_by_id


def validate_dataset(
    records: list[dict[str, str]],
    truth_by_id: dict[str, dict[str, Any]],
) -> None:
    records_by_id = {record['_id']: record for record in records}
    input_ids = set(records_by_id)
    truth_ids = set(truth_by_id)
    if input_ids != truth_ids:
        missing = truth_ids - input_ids
        unexpected = input_ids - truth_ids
        raise SystemExit(
            'Los IDs de entrada no coinciden con el ground truth: '
            f'{len(missing)} ausentes y {len(unexpected)} inesperados.'
        )
    for news_id, truth in truth_by_id.items():
        if records_by_id[news_id]['text'] != truth['text']:
            raise SystemExit(
                f'El texto de entrada no coincide con el ground truth para '
                f'{news_id!r}.'
            )


def normalize_scores(raw_scores: Any, expected_count: int) -> list[float]:
    if isinstance(raw_scores, Real):
        values = [raw_scores]
    else:
        try:
            values = list(raw_scores)
        except TypeError as exc:
            raise RuntimeError(
                'BGE devolvió scores con un formato no iterable.'
            ) from exc
    scores = [float(score) for score in values]
    if len(scores) != expected_count:
        raise RuntimeError(
            'BGE devolvió un número de scores distinto al número de temas: '
            f'{len(scores)} frente a {expected_count}.'
        )
    if not all(math.isfinite(score) for score in scores):
        raise RuntimeError('BGE devolvió uno o más scores no finitos.')
    return scores


def score_record(
    record: dict[str, str],
    topics: dict[str, str],
    score_pairs: ScoreFunction,
    synchronize: Callable[[], None] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    topic_names = list(topics)
    pairs = [
        [
            record['text'],
            TOPIC_TEMPLATE.format(
                topic=topic_name,
                definition=topics[topic_name],
            ),
        ]
        for topic_name in topic_names
    ]
    if synchronize is not None:
        synchronize()
    start = time.perf_counter()
    raw_scores = score_pairs(pairs)
    if synchronize is not None:
        synchronize()
    elapsed = time.perf_counter() - start

    scores = normalize_scores(raw_scores, len(topic_names))
    ranking = [
        {'tema': topic_name, 'cross_encoder_score': round(score, 6)}
        for topic_name, score in zip(topic_names, scores)
    ]
    ranking.sort(
        key=lambda candidate: candidate['cross_encoder_score'],
        reverse=True,
    )
    return ranking, elapsed


def evaluate_records(
    records: list[dict[str, str]],
    truth_by_id: dict[str, dict[str, Any]],
    topics: dict[str, str],
    score_pairs: ScoreFunction,
    synchronize: Callable[[], None] | None = None,
) -> tuple[list[dict[str, Any]], list[float]]:
    predictions: list[dict[str, Any]] = []
    inference_times: list[float] = []
    total_records = len(records)
    for record_index, record in enumerate(records, start=1):
        ranking, elapsed = score_record(
            record,
            topics,
            score_pairs,
            synchronize,
        )
        correct_topics = list(truth_by_id[record['_id']]['topics'])
        winner = ranking[0]
        predictions.append(
            {
                '_id': record['_id'],
                'file': record['file'],
                'topics_correctos': correct_topics,
                'topic_predicho': winner['tema'],
                'cross_encoder_score': winner['cross_encoder_score'],
                'topic_scores': ranking,
                'inference_time_seconds': round(elapsed, 6),
                'match': winner['tema'] in correct_topics,
            }
        )
        inference_times.append(elapsed)
        if record_index % 100 == 0 or record_index == total_records:
            print(f'Progreso: {record_index}/{total_records} noticias')
    return predictions, inference_times


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


def timing_summary(elapsed_times: list[float]) -> dict[str, Any]:
    sample_size = len(elapsed_times)
    mean = statistics.fmean(elapsed_times) if elapsed_times else 0.0
    if sample_size > 1:
        margin = (
            1.959963984540054
            * statistics.stdev(elapsed_times)
            / math.sqrt(sample_size)
        )
    else:
        margin = 0.0
    return {
        'sample_size': sample_size,
        'mean_seconds': round(mean, 6),
        'confidence_interval_95_seconds': {
            'lower': round(max(0.0, mean - margin), 6),
            'upper': round(mean + margin, 6),
        },
        'method': 'Normal approximation for the sample mean',
    }


def build_report(
    *,
    records: list[dict[str, str]],
    truth_by_id: dict[str, dict[str, Any]],
    topics: dict[str, str],
    predictions: list[dict[str, Any]],
    inference_times: list[float],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
    data_dir: Path,
) -> dict[str, Any]:
    hits = sum(prediction['match'] for prediction in predictions)
    total = len(predictions)
    accuracy = hits / total if total else 0.0
    normalized_truth = [
        {
            '_id': news_id,
            'text': truth['text'],
            'topics': truth['topics'],
        }
        for news_id, truth in truth_by_id.items()
    ]
    return {
        'status': 'complete',
        'configuration': {
            'model': model_name,
            'candidate_source': 'full topic catalog',
            'uses_e5_candidates': False,
            'uses_precomputed_results': False,
            'candidate_count': len(topics),
            'batch_size': batch_size,
            'max_length': max_length,
            'device': device,
            'use_fp16': device.startswith('cuda'),
            'score_normalization': True,
            'input_pair_order': [
                'news_text',
                'topic_name_and_definition',
            ],
            'topic_representation': TOPIC_TEMPLATE,
            'ranking_key': 'cross_encoder_score descending',
            'tie_breaker': 'topic catalog order',
            'metric_definitions': {
                'top_1_accuracy': (
                    'fraction of news whose highest-scoring topic is correct; '
                    'equal to micro-precision in this single-label dataset'
                ),
            },
            'data_dir': str(data_dir.resolve()),
            'input_data_sha256': sha256_json(records),
            'topic_catalog_sha256': sha256_json(topics),
            'ground_truth_sha256': sha256_json(normalized_truth),
        },
        'dataset': {
            'total_news': len(truth_by_id),
            'evaluable_news': total,
            'non_evaluable_news': len(truth_by_id) - total,
            'multi_topic_news': sum(
                len(truth['topics']) > 1 for truth in truth_by_id.values()
            ),
            'topic_count': len(topics),
        },
        'metrics': {
            'top_1_hits': hits,
            'top_1_misses': total - hits,
            'top_1_accuracy': round(accuracy, 6),
            'top_1_accuracy_percent': round(accuracy * 100, 2),
            'top_1_accuracy_confidence_interval_95': wilson_interval(
                hits, total
            ),
        },
        'timing': {
            'total_inference_time_seconds': round(sum(inference_times), 6),
            'per_news_inference_time': timing_summary(inference_times),
            'scope': (
                f'Tiempo medido alrededor de compute_score para los '
                f'{len(topics)} pares texto/tema de cada noticia, incluyendo '
                'sincronización CUDA cuando se utiliza GPU.'
            ),
        },
        'predictions': predictions,
    }


def load_reranker(
    model_name: str,
) -> tuple[Any, str, Callable[[], None] | None]:
    try:
        import torch
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise SystemExit(
            'Faltan dependencias. Instálalas con: '
            'pip install torch FlagEmbedding'
        ) from exc

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    reranker = FlagReranker(
        model_name,
        devices=device,
        use_fp16=device.startswith('cuda'),
    )
    synchronize = torch.cuda.synchronize if device.startswith('cuda') else None
    return reranker, device, synchronize


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit('--batch-size debe ser mayor que cero.')
    if args.max_length < 1:
        raise SystemExit('--max-length debe ser mayor que cero.')

    topics = load_topics(args.topics)
    records = load_texts(args.data_dir)
    truth_by_id = load_ground_truth(args.labelled_data_dir, set(topics))
    validate_dataset(records, truth_by_id)

    reranker, device, synchronize = load_reranker(args.model)
    print(f'Dispositivo: {device}')
    print(f'Modelo: {args.model}')
    print(f'Noticias: {len(records)}; temas por noticia: {len(topics)}')

    def score_pairs(pairs: list[list[str]]) -> Any:
        return reranker.compute_score(
            pairs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            normalize=True,
        )

    predictions, inference_times = evaluate_records(
        records,
        truth_by_id,
        topics,
        score_pairs,
        synchronize,
    )
    report = build_report(
        records=records,
        truth_by_id=truth_by_id,
        topics=topics,
        predictions=predictions,
        inference_times=inference_times,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        data_dir=args.data_dir,
    )
    save_json(args.output, report)

    metrics = report['metrics']
    hits = metrics['top_1_hits']
    total = hits + metrics['top_1_misses']
    accuracy_percent = metrics['top_1_accuracy_percent']
    print(f'Resultados guardados: {args.output.resolve()}')
    print(f'BGE Top-1: {hits}/{total} ({accuracy_percent:.2f}%)')


if __name__ == '__main__':
    main()
