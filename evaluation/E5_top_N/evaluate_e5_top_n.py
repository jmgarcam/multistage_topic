#!/usr/bin/env python3
'''Evalúa las métricas Top-1 y Top-N de E5.'''

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EVALUATION_DIR.parent
DATASET_DIR = PROJECT_DIR / 'dataset_mteb'
DEFAULT_E5_RESULTS = (
    EVALUATION_DIR / 'E5_top_1' / 'results' / 'e5_top_1_results.json'
)
DEFAULT_OUTPUT = SCRIPT_DIR / 'results' / 'e5_top_n_results.json'

if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

import e5_topic_ranker as e5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Puntúa directamente todos los temas con E5 y calcula las métricas '
            'Top-1 y Top-N. Sin valores N evalúa todos los posibles.'
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
        default=e5.DEFAULT_MODEL,
        help='Modelo bi-encoder de sentence-transformers.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=e5.DEFAULT_BATCH_SIZE,
        help='Número de textos codificados en cada lote.',
    )
    parser.add_argument(
        '--e5-results',
        type=Path,
        default=DEFAULT_E5_RESULTS,
        help=(
            'Informe E5 Top-1 cuyo ranking completo se reutiliza después '
            'de validarlo.'
        ),
    )
    parser.add_argument(
        '--recompute-e5',
        action='store_true',
        help='Ignora el informe E5 Top-1 y vuelve a ejecutar el bi-encoder.',
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
    labelled_by_id: dict[str, dict[str, Any]],
    rankings: list[dict[str, Any]],
    n_values: list[int],
    topic_count: int,
) -> list[dict[str, Any]]:
    by_id = {ranking['_id']: ranking for ranking in rankings}
    if set(by_id) != {record['_id'] for record in records}:
        raise RuntimeError('El ranking E5 no cubre exactamente data_test.')

    predictions: list[dict[str, Any]] = []
    for record in records:
        ranking = [dict(candidate) for candidate in by_id[record['_id']][
            'candidates'
        ]]
        names = [candidate['tema'] for candidate in ranking]
        if len(ranking) != topic_count or len(set(names)) != topic_count:
            raise RuntimeError(
                'El ranking E5 no contiene exactamente todos los temas.'
            )
        correct_topics = list(labelled_by_id[record['_id']]['topics'])
        correct_ranks = [
            index
            for index, topic in enumerate(names, start=1)
            if topic in correct_topics
        ]
        if not correct_ranks:
            raise RuntimeError(
                'Ninguna etiqueta correcta aparece en el ranking E5.'
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
    labelled_by_id: dict[str, dict[str, Any]],
    topics: dict[str, str],
    rankings: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    n_values: list[int],
    model_name: str,
    batch_size: int,
    device: str,
    timing: dict[str, Any],
    data_dir: Path,
    labelled_data_dir: Path,
    uses_precomputed_results: bool,
    precomputed_results_source: Path | None,
) -> dict[str, Any]:
    normalized_labels = [
        {
            '_id': record['_id'],
            'text': labelled_by_id[record['_id']]['text'],
            'topics': labelled_by_id[record['_id']]['topics'],
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
            'architecture': 'bi-encoder',
            'candidate_source': (
                'validated complete ranking from E5 Top-1 report'
                if uses_precomputed_results
                else 'full topic catalog scored directly by E5'
            ),
            'uses_precomputed_results': uses_precomputed_results,
            'precomputed_results_source': (
                str(precomputed_results_source.resolve())
                if precomputed_results_source is not None
                else None
            ),
            'uses_bge': False,
            'uses_llama': False,
            'n_values': n_values,
            'candidate_count': len(topics),
            'batch_size': batch_size,
            'device': device,
            'normalize_embeddings': True,
            'similarity': 'cosine',
            'news_prefix': e5.E5_PREFIX.strip(),
            'topic_prefix': e5.E5_PREFIX.strip(),
            'topic_representation': e5.TOPIC_TEMPLATE,
            'ranking_key': 'cosine similarity descending',
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
            'input_data_sha256': e5.sha256_json(records),
            'topic_catalog_sha256': e5.sha256_json(topics),
            'labelled_test_sha256': e5.sha256_json(normalized_labels),
            'e5_ranking_sha256': e5.ranking_sha256(rankings),
        },
        'dataset': {
            'total_news': len(records),
            'topic_count': len(topics),
            'multi_topic_news': sum(
                len(labelled['topics']) > 1
                for labelled in labelled_by_id.values()
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
        'timing': timing,
        'predictions': predictions,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit('--batch-size debe ser mayor que cero.')

    topics = e5.load_topics(args.topics)
    n_values = resolve_n_values(args.n_values, len(topics))
    records = e5.load_test_records(args.data_dir)
    labelled_by_id = e5.load_labelled_test(
        args.labelled_data_dir, set(topics)
    )
    e5.validate_dataset(records, labelled_by_id)

    print(f'Modelo: {args.model}')
    print(f'Noticias: {len(records)}; N: {n_values}')
    uses_precomputed_results = (
        not args.recompute_e5 and args.e5_results.is_file()
    )
    precomputed_results_source: Path | None = None
    if uses_precomputed_results:
        print(f'Validando ranking E5 Top-1: {args.e5_results.resolve()}')
        rankings, runtime = e5.load_validated_rankings(
            args.e5_results,
            records,
            labelled_by_id,
            topics,
            args.model,
            args.batch_size,
        )
        precomputed_results_source = args.e5_results
        runtime['timing']['scope'] = (
            'Tiempos originales del informe E5 Top-1 reutilizado; este '
            'baseline no vuelve a ejecutar el modelo.'
        )
        print('Ranking validado; no se vuelve a cargar el modelo E5.')
    else:
        if not args.recompute_e5:
            print(
                f'No existe {args.e5_results}; se calculará el ranking E5.'
            )
        rankings, runtime = e5.compute_rankings(
            records,
            topics,
            model_name=args.model,
            batch_size=args.batch_size,
        )
    print(f'Dispositivo del ranking E5: {runtime["device"]}')
    predictions = prepare_predictions(
        records, labelled_by_id, rankings, n_values, len(topics)
    )
    report = build_report(
        records=records,
        labelled_by_id=labelled_by_id,
        topics=topics,
        rankings=rankings,
        predictions=predictions,
        n_values=n_values,
        model_name=args.model,
        batch_size=args.batch_size,
        device=runtime['device'],
        timing=runtime['timing'],
        data_dir=args.data_dir,
        labelled_data_dir=args.labelled_data_dir,
        uses_precomputed_results=uses_precomputed_results,
        precomputed_results_source=precomputed_results_source,
    )
    e5.save_json(args.output, report)

    print(f'Resultados guardados: {args.output.resolve()}')
    print(
        'E5 Top-1 accuracy: '
        f"{report['metrics']['top_1_accuracy_percent']:.2f}%"
    )
    for n, metric in report['metrics']['candidate_recall_at_n'].items():
        hits = metric['hits']
        recall_percent = metric['candidate_recall_percent']
        precision_percent = report['metrics']['retrieval_precision_at_n'][n][
            'retrieval_precision_percent'
        ]
        print(
            f'E5 Candidate Recall@{n}: {hits}/{len(predictions)} '
            f'({recall_percent:.2f}%); '
            f'Retrieval Precision@{n}: {precision_percent:.2f}%'
        )


if __name__ == '__main__':
    main()
