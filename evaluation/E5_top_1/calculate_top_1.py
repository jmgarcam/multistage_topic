#!/usr/bin/env python3
'''Evalúa E5 Top-1 directamente contra todo el catálogo de temas.'''

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EVALUATION_DIR.parent
DATASET_DIR = PROJECT_DIR / 'dataset_mteb'
RESULTS_DIR = SCRIPT_DIR / 'results'

if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

import e5_topic_ranker as e5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Clasifica cada noticia de test con el bi-encoder E5, puntuando '
            'directamente todos los temas y conservando el de mayor '
            'similitud coseno.'
        )
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DATASET_DIR / 'data_test',
        help='Directorio con las noticias de test usadas como entrada.',
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
        default=RESULTS_DIR / 'e5_top_1_results.json',
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
        '--reuse-existing',
        action='store_true',
        help=(
            'Si el informe de salida existe, lo valida y lo reutiliza sin '
            'volver a cargar E5.'
        ),
    )
    return parser.parse_args()


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


def build_report(
    *,
    records: list[dict[str, str]],
    labelled_by_id: dict[str, dict[str, Any]],
    topics: dict[str, str],
    rankings: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
    device: str,
    timing: dict[str, Any],
    data_dir: Path,
    labelled_data_dir: Path,
) -> dict[str, Any]:
    rankings_by_id = {ranking['_id']: ranking for ranking in rankings}
    record_ids = [record['_id'] for record in records]
    if set(rankings_by_id) != set(record_ids):
        raise RuntimeError('El ranking E5 no cubre exactamente data_test.')

    predictions: list[dict[str, Any]] = []
    for record in records:
        ranking = rankings_by_id[record['_id']]['candidates']
        if len(ranking) != len(topics):
            raise RuntimeError(
                'El ranking E5 no contiene exactamente todos los temas.'
            )
        winner = ranking[0]
        correct_topics = list(labelled_by_id[record['_id']]['topics'])
        predictions.append(
            {
                '_id': record['_id'],
                'file': record['file'],
                'topics_correctos': correct_topics,
                'topic_predicho': winner['tema'],
                'similarity': winner['similarity'],
                'topic_scores': ranking,
                'match': winner['tema'] in correct_topics,
            }
        )

    hits = sum(prediction['match'] for prediction in predictions)
    total = len(predictions)
    accuracy = hits / total if total else 0.0
    normalized_labels = [
        {
            '_id': record['_id'],
            'text': labelled_by_id[record['_id']]['text'],
            'topics': labelled_by_id[record['_id']]['topics'],
        }
        for record in records
    ]
    return {
        'status': 'complete',
        'configuration': {
            'model': model_name,
            'architecture': 'bi-encoder',
            'candidate_source': 'full topic catalog scored directly by E5',
            'uses_precomputed_results': False,
            'uses_bge': False,
            'uses_llama': False,
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
            'evaluable_news': total,
            'topic_count': len(topics),
            'multi_topic_news': sum(
                len(labelled['topics']) > 1
                for labelled in labelled_by_id.values()
            ),
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
        'timing': timing,
        'predictions': predictions,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit('--batch-size debe ser mayor que cero.')

    topics = e5.load_topics(args.topics)
    records = e5.load_test_records(args.data_dir)
    labelled_by_id = e5.load_labelled_test(
        args.labelled_data_dir, set(topics)
    )
    e5.validate_dataset(records, labelled_by_id)

    print(f'Modelo: {args.model}')
    print(f'Noticias: {len(records)}; temas por noticia: {len(topics)}')
    if args.reuse_existing and args.output.is_file():
        rankings, runtime = e5.load_validated_rankings(
            args.output,
            records,
            labelled_by_id,
            topics,
            args.model,
            args.batch_size,
        )
        hits = sum(
            ranking['candidates'][0]['tema']
            in labelled_by_id[ranking['_id']]['topics']
            for ranking in rankings
        )
        total = len(rankings)
        accuracy_percent = hits / total * 100 if total else 0.0
        print(f'Dispositivo del informe existente: {runtime["device"]}')
        print(f'Informe completo y validado: {args.output.resolve()}')
        print(f'E5 Top-1: {hits}/{total} ({accuracy_percent:.2f}%)')
        return
    if args.reuse_existing:
        print(f'No existe {args.output}; se calculará E5 Top-1.')

    rankings, runtime = e5.compute_rankings(
        records,
        topics,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    device = runtime['device']
    print(f'Dispositivo: {device}')

    report = build_report(
        records=records,
        labelled_by_id=labelled_by_id,
        topics=topics,
        rankings=rankings,
        model_name=args.model,
        batch_size=args.batch_size,
        device=runtime['device'],
        timing=runtime['timing'],
        data_dir=args.data_dir,
        labelled_data_dir=args.labelled_data_dir,
    )
    e5.save_json(args.output, report)

    metrics = report['metrics']
    hits = metrics['top_1_hits']
    total = hits + metrics['top_1_misses']
    accuracy_percent = metrics['top_1_accuracy_percent']
    print(f'Resultados guardados: {args.output.resolve()}')
    print(
        f'E5 Top-1: {hits}/{total} '
        f'({accuracy_percent:.2f}%)'
    )


if __name__ == '__main__':
    main()
