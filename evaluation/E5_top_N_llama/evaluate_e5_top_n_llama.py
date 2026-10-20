#!/usr/bin/env python3
'''Calcula E5 sobre test y evalúa Llama con cada Top-N posible.'''

from __future__ import annotations

import argparse
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
DEFAULT_E5_RESULTS = (
    EVALUATION_DIR / 'E5_top_N' / 'results' / 'e5_top_n_results.json'
)
DEFAULT_RESULTS_DIR = SCRIPT_DIR / 'results_llama3_1_8b'
DEFAULT_CHECKPOINT_EVERY = 10
PROMPT_VERSION = 'mteb_forced_topic_selection_v1_e5_top_n_v1'
DIRECT_CANDIDATE_SOURCE = 'E5 computed directly from data_test'
PRECOMPUTED_CANDIDATE_SOURCE = (
    'validated complete ranking from E5 Top-N report'
)
LEGACY_CANDIDATE_SOURCE = 'E5 cosine-similarity ranking'

if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

import e5_topic_ranker as e5  # noqa: E402
import evaluate_ollama_top_n_no_none as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Calcula el ranking E5 directamente desde data_test y ejecuta '
            'Llama con los primeros N temas. Sin valores N evalúa desde 1 '
            'hasta el número total de temas.'
        )
    )
    parser.add_argument(
        '--n_values',
        metavar='N',
        type=int,
        nargs='*',
        help='Tamaños Top-N; por defecto se evalúan todos los posibles.',
    )
    parser.add_argument(
        '--model',
        default=os.environ.get('OLLAMA_MODEL', 'llama3.1:8b'),
        help='Modelo disponible en Ollama.',
    )
    parser.add_argument(
        '--e5-model',
        default=e5.DEFAULT_MODEL,
        help='Modelo bi-encoder usado para generar los candidatos.',
    )
    parser.add_argument(
        '--e5-batch-size',
        type=int,
        default=e5.DEFAULT_BATCH_SIZE,
        help='Número de textos codificados por lote con E5.',
    )
    parser.add_argument(
        '--e5-results',
        type=Path,
        default=DEFAULT_E5_RESULTS,
        help=(
            'Informe E5 Top-N cuyo ranking completo se reutiliza después '
            'de validarlo.'
        ),
    )
    parser.add_argument(
        '--recompute-e5',
        action='store_true',
        help='Ignora el informe E5 Top-N y vuelve a ejecutar el bi-encoder.',
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DATASET_DIR / 'data_test',
        help='Directorio con las noticias de test.',
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
        '--results-dir',
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help='Directorio de informes y checkpoints por valor de N.',
    )
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help='Guarda el informe después de este número de predicciones.',
    )
    return parser.parse_args()


def resolve_n_values(requested: list[int], topic_count: int) -> list[int]:
    n_values = requested or list(range(1, topic_count + 1))
    if n_values != sorted(set(n_values)):
        raise SystemExit('Los valores N deben estar ordenados y no repetirse.')
    if any(n < 1 or n > topic_count for n in n_values):
        raise SystemExit(f'N debe estar entre 1 y {topic_count}.')
    return n_values


def prepare_records(
    rankings: list[dict[str, Any]],
    labelled_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ranking in rankings:
        news_id = ranking['_id']
        labelled = labelled_by_id[news_id]
        records.append(
            {
                '_id': news_id,
                'file': ranking['file'],
                'text': ranking['text'],
                'topics_correctos': list(labelled['topics']),
                'candidates': [
                    dict(candidate) for candidate in ranking['candidates']
                ],
            }
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
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    topics: dict[str, str],
    n: int,
    e5_context: dict[str, Any],
) -> dict[str, Any]:
    llama_final_hits = sum(
        prediction['matches_real_topic'] for prediction in predictions
    )
    evaluated = len(predictions)
    llama_final_accuracy = (
        llama_final_hits / evaluated if evaluated else 0.0
    )
    available = sum(
        prediction['candidate_contains_real_topic']
        for prediction in predictions
    )
    conditional_selection_accuracy = (
        llama_final_hits / available if available else 0.0
    )
    retriever_top_1_hits = sum(
        record['candidates'][0]['tema'] in record['topics_correctos']
        for record in records
    )
    retriever_top_1_accuracy = (
        retriever_top_1_hits / len(records) if records else 0.0
    )
    candidate_recall_hits = sum(
        any(
            topic in {candidate['tema'] for candidate in record['candidates'][:n]}
            for topic in record['topics_correctos']
        )
        for record in records
    )
    candidate_recall = candidate_recall_hits / len(records) if records else 0.0
    elapsed_times = [
        float(prediction['inference_time_seconds'])
        for prediction in predictions
    ]
    return {
        'status': 'complete' if evaluated == len(records) else 'in_progress',
        'configuration': {
            'e5_model': e5_context['model'],
            'e5_architecture': 'bi-encoder',
            'e5_batch_size': e5_context['batch_size'],
            'e5_device': e5_context['device'],
            'e5_normalize_embeddings': True,
            'e5_similarity': 'cosine',
            'e5_news_prefix': e5.E5_PREFIX.strip(),
            'e5_topic_prefix': e5.E5_PREFIX.strip(),
            'e5_topic_representation': e5.TOPIC_TEMPLATE,
            'e5_ranking_sha256': e5_context['ranking_sha256'],
            'llama_model': base.MODEL_NAME,
            'candidate_source': e5_context['candidate_source'],
            'uses_precomputed_e5_results': e5_context[
                'uses_precomputed_results'
            ],
            'precomputed_e5_results_source': e5_context[
                'precomputed_results_source'
            ],
            'n': n,
            'topic_count': len(topics),
            'prompt_version': PROMPT_VERSION,
            'decision_strategy': 'forced_topic_selection',
            'prediction_format': base.PREDICTION_FORMAT,
            'generation': 'Ollama API (temperature=0, seed=0)',
            'metric_definitions': {
                'retriever_top_1_accuracy': (
                    'fraction of news whose highest-scoring retriever topic '
                    'is correct; equal to micro-precision for this dataset'
                ),
                'candidate_recall_at_n': (
                    'fraction of news with the correct topic among the N '
                    'candidates sent to Llama'
                ),
                'llama_final_accuracy': (
                    'fraction of final Llama decisions matching the correct '
                    'topic'
                ),
                'conditional_selection_accuracy': (
                    'Llama accuracy restricted to news where the correct '
                    'topic was available among the candidates'
                ),
            },
            'api_base_url': base.get_base_url(),
            'data_dir': e5_context['data_dir'],
            'labelled_data_dir': e5_context['labelled_data_dir'],
            'labels_used_for': 'metrics only',
            'input_data_sha256': e5_context['input_data_sha256'],
            'topic_catalog_sha256': e5_context['topic_catalog_sha256'],
            'labelled_test_sha256': e5_context['labelled_test_sha256'],
        },
        'dataset': {
            'total_news': len(records),
            'expected_predictions': len(records),
            'completed_predictions': evaluated,
        },
        'metrics': {
            'evaluated_news': evaluated,
            'retriever_top_1_hits': retriever_top_1_hits,
            'retriever_top_1_misses': len(records) - retriever_top_1_hits,
            'retriever_top_1_accuracy': round(
                retriever_top_1_accuracy, 6
            ),
            'retriever_top_1_accuracy_percent': round(
                retriever_top_1_accuracy * 100, 2
            ),
            'llama_final_hits': llama_final_hits,
            'llama_final_misses': evaluated - llama_final_hits,
            'llama_final_accuracy': round(llama_final_accuracy, 6),
            'llama_final_accuracy_percent': round(
                llama_final_accuracy * 100, 2
            ),
            'llama_final_accuracy_confidence_interval_95': wilson_interval(
                llama_final_hits, evaluated
            ),
            'candidate_recall_at_n_hits': candidate_recall_hits,
            'candidate_recall_at_n': round(candidate_recall, 6),
            'candidate_recall_at_n_percent': round(
                candidate_recall * 100, 2
            ),
            'retrieval_errors': evaluated - available,
            'selection_opportunities': available,
            'selection_errors': available - llama_final_hits,
            'conditional_selection_accuracy': round(
                conditional_selection_accuracy, 6
            ),
            'conditional_selection_accuracy_percent': round(
                conditional_selection_accuracy * 100, 2
            ),
        },
        'timing': {
            'e5_ranking': e5_context['timing'],
            'llama_total_inference_time_seconds': round(
                sum(elapsed_times), 6
            ),
            'llama_per_news_inference_time': mean_time_summary(elapsed_times),
            'scope': (
                'El ranking E5 se obtiene una vez, por cálculo o desde un '
                'informe validado. Para cada N se mide por separado la '
                'llamada HTTP y la inferencia de Llama.'
            ),
        },
        'predictions': predictions,
    }


def load_checkpoint(
    path: Path,
    records: list[dict[str, Any]],
    n: int,
    e5_context: dict[str, Any],
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    report = e5.load_json(path)
    if not isinstance(report, dict):
        raise SystemExit(f'Checkpoint no válido: {path}')
    configuration = report.get('configuration')
    common_expected = {
        'e5_model': e5_context['model'],
        'llama_model': base.MODEL_NAME,
        'n': n,
        'prompt_version': PROMPT_VERSION,
    }
    if not isinstance(configuration, dict) or any(
        configuration.get(key) != value
        for key, value in common_expected.items()
    ):
        raise SystemExit(
            f'{path} usa otra configuración. Muévelo antes de ejecutar '
            'este experimento.'
        )
    candidate_source = configuration.get('candidate_source')
    if candidate_source in {
        DIRECT_CANDIDATE_SOURCE,
        PRECOMPUTED_CANDIDATE_SOURCE,
    }:
        if (
            configuration.get('e5_ranking_sha256')
            != e5_context['ranking_sha256']
            or configuration.get('e5_batch_size')
            != e5_context['batch_size']
        ):
            raise SystemExit(f'{path} usa otro ranking o configuración E5.')
    elif candidate_source != LEGACY_CANDIDATE_SOURCE:
        raise SystemExit(f'{path} tiene una fuente de candidatos desconocida.')

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
        candidates = record['candidates'][:n]
        candidate_names = [candidate['tema'] for candidate in candidates]
        if prediction.get('n') != n or prediction.get(
            'candidate_topics'
        ) != candidate_names:
            raise SystemExit(
                f'{location}: los candidatos guardados no coinciden con '
                'el ranking E5 calculado directamente.'
            )
        decision = base.validate_saved_decision(
            prediction.get(base.DECISION_FIELD), candidate_names, location
        )
        prediction[base.DECISION_FIELD] = decision
        base.validate_prediction_outputs(
            prediction, candidate_names, decision, location
        )
        prediction['file'] = record['file']
        prediction['topics_correctos'] = list(record['topics_correctos'])
        prediction['candidate_scores'] = [dict(candidate) for candidate in candidates]
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
    results_dir: Path,
    e5_context: dict[str, Any],
) -> None:
    output_path = results_dir / f'e5_top_{n}_llama_results.json'
    predictions = load_checkpoint(output_path, records, n, e5_context)
    completed_ids = {prediction['_id'] for prediction in predictions}
    pending = [record for record in records if record['_id'] not in completed_ids]
    order = {record['_id']: index for index, record in enumerate(records)}

    def checkpoint() -> dict[str, Any]:
        predictions.sort(key=lambda prediction: order[prediction['_id']])
        report = build_report(records, predictions, topics, n, e5_context)
        e5.save_json(output_path, report)
        return report

    if not pending:
        report = checkpoint()
        accuracy = report['metrics']['llama_final_accuracy_percent']
        print(f'N={n}: completo, {accuracy:.2f}%')
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
                    'candidate_scores': [
                        dict(candidate) for candidate in candidates
                    ],
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
    accuracy = report['metrics']['llama_final_accuracy_percent']
    print(f'N={n}: completo, {accuracy:.2f}%')


def main() -> None:
    args = parse_args()
    if args.checkpoint_every < 1:
        raise SystemExit('--checkpoint-every debe ser mayor que cero.')
    if args.e5_batch_size < 1:
        raise SystemExit('--e5-batch-size debe ser mayor que cero.')

    topics = e5.load_topics(args.topics)
    n_values = resolve_n_values(args.n_values, len(topics))
    test_records = e5.load_test_records(args.data_dir)
    labelled_by_id = e5.load_labelled_test(
        args.labelled_data_dir, set(topics)
    )
    e5.validate_dataset(test_records, labelled_by_id)

    print(f'Modelo E5: {args.e5_model}')
    print(f'Noticias: {len(test_records)}; valores N: {n_values}')
    uses_precomputed_results = (
        not args.recompute_e5 and args.e5_results.is_file()
    )
    precomputed_results_source: Path | None = None
    if uses_precomputed_results:
        print(f'Validando ranking E5 Top-N: {args.e5_results.resolve()}')
        rankings, runtime = e5.load_validated_rankings(
            args.e5_results,
            test_records,
            labelled_by_id,
            topics,
            args.e5_model,
            args.e5_batch_size,
        )
        precomputed_results_source = args.e5_results
        candidate_source = PRECOMPUTED_CANDIDATE_SOURCE
        runtime['timing']['scope'] = (
            'Tiempos originales del informe E5 reutilizado; este baseline '
            'no vuelve a ejecutar el modelo.'
        )
        print('Ranking validado; no se vuelve a cargar el modelo E5.')
    else:
        if not args.recompute_e5:
            print(
                f'No existe {args.e5_results}; se calculará el ranking E5.'
            )
        rankings, runtime = e5.compute_rankings(
            test_records,
            topics,
            model_name=args.e5_model,
            batch_size=args.e5_batch_size,
        )
        candidate_source = DIRECT_CANDIDATE_SOURCE
    device = runtime['device']
    print(f'Dispositivo del ranking E5: {device}')

    normalized_labels = [
        {
            '_id': record['_id'],
            'text': labelled_by_id[record['_id']]['text'],
            'topics': labelled_by_id[record['_id']]['topics'],
        }
        for record in test_records
    ]
    e5_context = {
        'model': args.e5_model,
        'batch_size': args.e5_batch_size,
        'device': device,
        'timing': runtime['timing'],
        'ranking_sha256': e5.ranking_sha256(rankings),
        'input_data_sha256': e5.sha256_json(test_records),
        'topic_catalog_sha256': e5.sha256_json(topics),
        'labelled_test_sha256': e5.sha256_json(normalized_labels),
        'data_dir': str(args.data_dir.resolve()),
        'labelled_data_dir': str(args.labelled_data_dir.resolve()),
        'candidate_source': candidate_source,
        'uses_precomputed_results': uses_precomputed_results,
        'precomputed_results_source': (
            str(precomputed_results_source.resolve())
            if precomputed_results_source is not None
            else None
        ),
    }
    records = prepare_records(rankings, labelled_by_id)

    base.MODEL_NAME = args.model
    base.MODEL_LABEL = f'Ollama {args.model}'
    base.ALLOW_NONE_DECISION = False
    base.PROMPT_VERSION = 'mteb_forced_topic_selection_v1'

    client_holder: list[base.OllamaClient] = []
    for n in n_values:
        evaluate_n(
            records,
            topics,
            n,
            args.checkpoint_every,
            client_holder,
            args.results_dir,
            e5_context,
        )


if __name__ == '__main__':
    main()
