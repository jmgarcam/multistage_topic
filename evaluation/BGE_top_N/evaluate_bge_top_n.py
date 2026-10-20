#!/usr/bin/env python3
'''Evalúa las métricas Top-1 y Top-N de BGE.'''

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EVALUATION_DIR.parent
DATASET_DIR = PROJECT_DIR / 'dataset_mteb'
BGE_TOP_1_PATH = EVALUATION_DIR / 'BGE_top_1' / 'calculate_top_1.py'
DEFAULT_BGE_RESULTS = (
    EVALUATION_DIR / 'BGE_top_1' / 'results' / 'bge_top_1_results.json'
)
DEFAULT_OUTPUT = SCRIPT_DIR / 'results' / 'bge_top_n_results.json'


def load_bge_top_1_module() -> Any:
    module_name = 'bge_top_1_for_top_n'
    specification = importlib.util.spec_from_file_location(
        module_name, BGE_TOP_1_PATH
    )
    if specification is None or specification.loader is None:
        raise SystemExit(f'No se pudo cargar {BGE_TOP_1_PATH}.')
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


bge = load_bge_top_1_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Puntúa directamente todos los temas con BGE y calcula las métricas '
            'Top-1 y Top-N. Sin valores N evalúa desde 1 hasta el total '
            'de temas.'
        )
    )
    parser.add_argument(
        'n_values',
        metavar='N',
        type=int,
        nargs='*',
        help='Valores de N; por defecto se evalúan todos los posibles.',
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
        help='Copia etiquetada de test, utilizada únicamente para métricas.',
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
        help='Ruta del informe JSON de salida.',
    )
    parser.add_argument(
        '--model',
        default=bge.MODEL_NAME,
        help='Modelo cross-encoder de FlagEmbedding.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=bge.BATCH_SIZE,
        help='Tamaño de lote para puntuar los pares texto/tema.',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=bge.MAX_LENGTH,
        help='Longitud máxima de los pares tokenizados.',
    )
    parser.add_argument(
        '--bge-results',
        type=Path,
        default=DEFAULT_BGE_RESULTS,
        help=(
            'Informe BGE Top-1 cuyo ranking completo se reutiliza después '
            'de validarlo.'
        ),
    )
    parser.add_argument(
        '--recompute-bge',
        action='store_true',
        help='Ignora el informe BGE Top-1 y vuelve a ejecutar el reranker.',
    )
    return parser.parse_args()


def resolve_n_values(requested: list[int], topic_count: int) -> list[int]:
    n_values = requested or list(range(1, topic_count + 1))
    if n_values != sorted(set(n_values)):
        raise SystemExit('Los valores N deben estar ordenados y no repetirse.')
    if any(n < 1 or n > topic_count for n in n_values):
        raise SystemExit(f'N debe estar entre 1 y {topic_count}.')
    return n_values


def prepare_predictions(
    records: list[dict[str, str]],
    bge_predictions: list[dict[str, Any]],
    n_values: list[int],
    topic_count: int,
) -> list[dict[str, Any]]:
    by_id = {prediction['_id']: prediction for prediction in bge_predictions}
    if set(by_id) != {record['_id'] for record in records}:
        raise RuntimeError('El ranking BGE no cubre exactamente data_test.')

    predictions: list[dict[str, Any]] = []
    for record in records:
        source = by_id[record['_id']]
        ranking = [dict(candidate) for candidate in source['topic_scores']]
        names = [candidate['tema'] for candidate in ranking]
        if len(ranking) != topic_count or len(set(names)) != topic_count:
            raise RuntimeError(
                'El ranking BGE no contiene exactamente todos los temas.'
            )
        correct_topics = list(source['topics_correctos'])
        correct_ranks = [
            index
            for index, topic in enumerate(names, start=1)
            if topic in correct_topics
        ]
        if not correct_ranks:
            raise RuntimeError(
                'Ninguna etiqueta correcta aparece en el ranking BGE.'
            )
        first_correct_rank = min(correct_ranks)
        relevant_topic_count_by_n = {
            str(n): sum(topic in correct_topics for topic in names[:n])
            for n in n_values
        }
        predictions.append(
            {
                '_id': record['_id'],
                'file': record['file'],
                'topics_correctos': correct_topics,
                'first_correct_rank': first_correct_rank,
                'matches_by_n': {
                    str(n): first_correct_rank <= n for n in n_values
                },
                'relevant_topic_count_by_n': relevant_topic_count_by_n,
                'topic_scores': ranking,
                'inference_time_seconds': source[
                    'inference_time_seconds'
                ],
            }
        )
    return predictions


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
        'lower_percent': round(max(0.0, centre - margin) * 100, 2),
        'upper_percent': round(min(1.0, centre + margin) * 100, 2),
        'method': 'Wilson score interval',
    }


def candidate_recall_metrics(
    predictions: list[dict[str, Any]], n_values: list[int]
) -> dict[str, dict[str, Any]]:
    total = len(predictions)
    metrics: dict[str, dict[str, Any]] = {}
    for n in n_values:
        hits = sum(
            prediction['matches_by_n'][str(n)]
            for prediction in predictions
        )
        recall = hits / total if total else 0.0
        metrics[str(n)] = {
            'hits': hits,
            'misses': total - hits,
            'candidate_recall': round(recall, 6),
            'candidate_recall_percent': round(recall * 100, 2),
            'confidence_interval_95': wilson_interval(hits, total),
        }
    return metrics


def retrieval_precision_metrics(
    predictions: list[dict[str, Any]], n_values: list[int]
) -> dict[str, dict[str, Any]]:
    total_news = len(predictions)
    metrics: dict[str, dict[str, Any]] = {}
    for n in n_values:
        relevant_retrieved = sum(
            prediction['relevant_topic_count_by_n'][str(n)]
            for prediction in predictions
        )
        retrieved = total_news * n
        precision = relevant_retrieved / retrieved if retrieved else 0.0
        metrics[str(n)] = {
            'relevant_topics_retrieved': relevant_retrieved,
            'retrieved_topics': retrieved,
            'retrieval_precision': round(precision, 6),
            'retrieval_precision_percent': round(precision * 100, 2),
        }
    return metrics


def build_report(
    *,
    records: list[dict[str, str]],
    truth_by_id: dict[str, dict[str, Any]],
    topics: dict[str, str],
    predictions: list[dict[str, Any]],
    inference_times: list[float],
    n_values: list[int],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
    data_dir: Path,
    labelled_data_dir: Path,
    uses_precomputed_results: bool,
    precomputed_results_source: Path | None,
) -> dict[str, Any]:
    normalized_truth = [
        {
            '_id': record['_id'],
            'text': truth_by_id[record['_id']]['text'],
            'topics': truth_by_id[record['_id']]['topics'],
        }
        for record in records
    ]
    first_ranks = [
        prediction['first_correct_rank'] for prediction in predictions
    ]
    top_1_hits = sum(rank == 1 for rank in first_ranks)
    top_1_accuracy = top_1_hits / len(predictions) if predictions else 0.0
    return {
        'status': 'complete',
        'configuration': {
            'model': model_name,
            'architecture': 'cross-encoder',
            'candidate_source': (
                'validated complete ranking from BGE Top-1 report'
                if uses_precomputed_results
                else 'full topic catalog scored directly by BGE'
            ),
            'uses_precomputed_results': uses_precomputed_results,
            'precomputed_results_source': (
                str(precomputed_results_source.resolve())
                if precomputed_results_source is not None
                else None
            ),
            'uses_e5': False,
            'uses_llama': False,
            'n_values': n_values,
            'candidate_count': len(topics),
            'batch_size': batch_size,
            'max_length': max_length,
            'device': device,
            'use_fp16': device.startswith('cuda'),
            'score_normalization': True,
            'input_pair_order': ['news_text', 'topic_name_and_definition'],
            'topic_representation': bge.TOPIC_TEMPLATE,
            'ranking_key': 'cross_encoder_score descending',
            'tie_breaker': 'topic catalog order',
            'metric_definitions': {
                'top_1_accuracy': (
                    'fraction of news whose highest-scoring topic is correct; '
                    'equal to micro-precision in this single-label dataset'
                ),
                'candidate_recall_at_n': (
                    'fraction of news with the correct topic among the first N'
                ),
                'retrieval_precision_at_n': (
                    'correct topics retrieved divided by all N retrieved '
                    'topics; kept separate from Top-1 accuracy'
                ),
            },
            'data_dir': str(data_dir.resolve()),
            'labelled_data_dir': str(labelled_data_dir.resolve()),
            'labels_used_for': 'metrics only',
            'input_data_sha256': bge.sha256_json(records),
            'topic_catalog_sha256': bge.sha256_json(topics),
            'labelled_test_sha256': bge.sha256_json(normalized_truth),
        },
        'dataset': {
            'total_news': len(records),
            'topic_count': len(topics),
            'multi_topic_news': sum(
                len(truth['topics']) > 1 for truth in truth_by_id.values()
            ),
        },
        'metrics': {
            'top_1_hits': top_1_hits,
            'top_1_misses': len(predictions) - top_1_hits,
            'top_1_accuracy': round(top_1_accuracy, 6),
            'top_1_accuracy_percent': round(top_1_accuracy * 100, 2),
            'top_1_accuracy_confidence_interval_95': wilson_interval(
                top_1_hits, len(predictions)
            ),
            'candidate_recall_at_n': candidate_recall_metrics(
                predictions, n_values
            ),
            'retrieval_precision_at_n': retrieval_precision_metrics(
                predictions, n_values
            ),
            'mean_first_correct_rank': (
                round(statistics.fmean(first_ranks), 6)
                if first_ranks
                else None
            ),
            'median_first_correct_rank': (
                statistics.median(first_ranks) if first_ranks else None
            ),
            'mean_reciprocal_rank': (
                round(
                    statistics.fmean(1 / rank for rank in first_ranks),
                    6,
                )
                if first_ranks
                else None
            ),
        },
        'timing': {
            'total_inference_time_seconds': round(sum(inference_times), 6),
            'per_news_inference_time': bge.timing_summary(inference_times),
            'scope': (
                'Tiempos originales del informe BGE Top-1 reutilizado; las '
                'métricas Top-1 y Top-N no vuelven a ejecutar el modelo.'
                if uses_precomputed_results
                else 'BGE puntúa una vez los pares de los 12 temas por '
                'noticia; las métricas Top-1 y Top-N usan el mismo ranking.'
            ),
        },
        'predictions': predictions,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit('--batch-size debe ser mayor que cero.')
    if args.max_length < 1:
        raise SystemExit('--max-length debe ser mayor que cero.')

    topics = bge.load_topics(args.topics)
    n_values = resolve_n_values(args.n_values, len(topics))
    records = bge.load_texts(args.data_dir)
    truth_by_id = bge.load_ground_truth(
        args.labelled_data_dir, set(topics)
    )
    bge.validate_dataset(records, truth_by_id)

    print(f'Modelo: {args.model}')
    print(f'Noticias: {len(records)}; N: {n_values}')
    uses_precomputed_results = (
        not args.recompute_bge and args.bge_results.is_file()
    )
    precomputed_results_source: Path | None = None
    if uses_precomputed_results:
        print(f'Validando ranking BGE Top-1: {args.bge_results.resolve()}')
        raw_predictions, inference_times, device = (
            bge.load_validated_predictions(
                args.bge_results,
                records,
                truth_by_id,
                topics,
                args.model,
                args.batch_size,
                args.max_length,
            )
        )
        precomputed_results_source = args.bge_results
        print('Ranking validado; no se vuelve a cargar el modelo BGE.')
    else:
        if not args.recompute_bge:
            print(
                f'No existe {args.bge_results}; se calculará el ranking BGE.'
            )
        reranker, device, synchronize = bge.load_reranker(args.model)

        def score_pairs(pairs: list[list[str]]) -> Any:
            return reranker.compute_score(
                pairs,
                batch_size=args.batch_size,
                max_length=args.max_length,
                normalize=True,
            )

        raw_predictions, inference_times = bge.evaluate_records(
            records,
            truth_by_id,
            topics,
            score_pairs,
            synchronize,
        )
    print(f'Dispositivo del ranking BGE: {device}')
    predictions = prepare_predictions(
        records, raw_predictions, n_values, len(topics)
    )
    report = build_report(
        records=records,
        truth_by_id=truth_by_id,
        topics=topics,
        predictions=predictions,
        inference_times=inference_times,
        n_values=n_values,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        data_dir=args.data_dir,
        labelled_data_dir=args.labelled_data_dir,
        uses_precomputed_results=uses_precomputed_results,
        precomputed_results_source=precomputed_results_source,
    )
    bge.save_json(args.output, report)

    print(f'Resultados guardados: {args.output.resolve()}')
    print(
        'BGE Top-1 accuracy: '
        f"{report['metrics']['top_1_accuracy_percent']:.2f}%"
    )
    for n, metric in report['metrics']['candidate_recall_at_n'].items():
        hits = metric['hits']
        recall_percent = metric['candidate_recall_percent']
        precision_percent = report['metrics']['retrieval_precision_at_n'][n][
            'retrieval_precision_percent'
        ]
        print(
            f'BGE Candidate Recall@{n}: {hits}/{len(predictions)} '
            f'({recall_percent:.2f}%); '
            f'Retrieval Precision@{n}: {precision_percent:.2f}%'
        )


if __name__ == '__main__':
    main()
