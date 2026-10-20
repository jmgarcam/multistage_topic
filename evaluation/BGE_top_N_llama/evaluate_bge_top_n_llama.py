#!/usr/bin/env python3
'''Calcula BGE sobre test y evalúa Llama con cada Top-N posible.'''

from __future__ import annotations

import argparse
import importlib.util
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
BGE_TOP_1_PATH = EVALUATION_DIR / 'BGE_top_1' / 'calculate_top_1.py'
DEFAULT_RESULTS_DIR = SCRIPT_DIR / 'results_llama3_2_3b'
DEFAULT_CHECKPOINT_EVERY = 10
PROMPT_VERSION = 'mteb_forced_topic_selection_v1_bge_top_n_v1'
CANDIDATE_SOURCE = 'BGE computed directly from data_test'

if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

import evaluate_ollama_top_n_no_none as base  # noqa: E402


def load_bge_top_1_module() -> Any:
    module_name = 'bge_top_1_runtime'
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
            'Calcula el ranking BGE directamente desde data_test y ejecuta '
            'Llama con los primeros N temas. Sin valores N evalúa desde 1 '
            'hasta el número total de temas.'
        )
    )
    parser.add_argument(
        'n_values',
        metavar='N',
        type=int,
        nargs='*',
        help='Tamaños Top-N; por defecto se evalúan todos los posibles.',
    )
    parser.add_argument(
        '--model',
        default=os.environ.get('OLLAMA_MODEL', 'llama3.2:3b'),
        help='Modelo disponible en Ollama.',
    )
    parser.add_argument(
        '--bge-model',
        default=bge.MODEL_NAME,
        help='Modelo cross-encoder usado para generar los candidatos.',
    )
    parser.add_argument(
        '--bge-batch-size',
        type=int,
        default=bge.BATCH_SIZE,
        help='Tamaño de lote para puntuar pares con BGE.',
    )
    parser.add_argument(
        '--bge-max-length',
        type=int,
        default=bge.MAX_LENGTH,
        help='Longitud máxima de los pares tokenizados por BGE.',
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
    test_records: list[dict[str, str]],
    bge_predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    predictions_by_id = {
        prediction['_id']: prediction for prediction in bge_predictions
    }
    if set(predictions_by_id) != {record['_id'] for record in test_records}:
        raise RuntimeError('El ranking BGE no cubre exactamente data_test.')
    records: list[dict[str, Any]] = []
    for record in test_records:
        prediction = predictions_by_id[record['_id']]
        records.append(
            {
                '_id': record['_id'],
                'file': record['file'],
                'text': record['text'],
                'topics_correctos': list(prediction['topics_correctos']),
                'candidates': [
                    dict(candidate)
                    for candidate in prediction['topic_scores']
                ],
            }
        )
    return records


def ranking_sha256(records: list[dict[str, Any]]) -> str:
    payload = [
        {'_id': record['_id'], 'candidates': record['candidates']}
        for record in records
    ]
    return bge.sha256_json(payload)


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
    bge_context: dict[str, Any],
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
            'bge_model': bge_context['model'],
            'bge_architecture': 'cross-encoder',
            'bge_batch_size': bge_context['batch_size'],
            'bge_max_length': bge_context['max_length'],
            'bge_device': bge_context['device'],
            'bge_use_fp16': bge_context['device'].startswith('cuda'),
            'bge_score_normalization': True,
            'bge_topic_representation': bge.TOPIC_TEMPLATE,
            'bge_ranking_sha256': bge_context['ranking_sha256'],
            'llama_model': base.MODEL_NAME,
            'candidate_source': CANDIDATE_SOURCE,
            'uses_precomputed_bge_results': False,
            'uses_e5_candidates': False,
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
            'data_dir': bge_context['data_dir'],
            'labelled_data_dir': bge_context['labelled_data_dir'],
            'labels_used_for': 'metrics only',
            'input_data_sha256': bge_context['input_data_sha256'],
            'topic_catalog_sha256': bge_context['topic_catalog_sha256'],
            'labelled_test_sha256': bge_context['labelled_test_sha256'],
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
            'bge_ranking': bge_context['timing'],
            'llama_total_inference_time_seconds': round(
                sum(elapsed_times), 6
            ),
            'llama_per_news_inference_time': mean_time_summary(elapsed_times),
            'scope': (
                'BGE se ejecuta una vez sobre todo test. Para cada N se mide '
                'por separado la llamada HTTP y la inferencia de Llama.'
            ),
        },
        'predictions': predictions,
    }


def load_checkpoint(
    path: Path,
    records: list[dict[str, Any]],
    n: int,
    bge_context: dict[str, Any],
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    report = bge.load_json(path)
    if not isinstance(report, dict):
        raise SystemExit(f'Checkpoint no válido: {path}')
    configuration = report.get('configuration')
    expected = {
        'bge_model': bge_context['model'],
        'bge_batch_size': bge_context['batch_size'],
        'bge_max_length': bge_context['max_length'],
        'bge_ranking_sha256': bge_context['ranking_sha256'],
        'llama_model': base.MODEL_NAME,
        'candidate_source': CANDIDATE_SOURCE,
        'n': n,
        'prompt_version': PROMPT_VERSION,
    }
    if not isinstance(configuration, dict) or any(
        configuration.get(key) != value for key, value in expected.items()
    ):
        raise SystemExit(
            f'{path} usa otra configuración o ranking. Muévelo antes de '
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
        record = records_by_id[news_id]
        candidates = record['candidates'][:n]
        candidate_names = [candidate['tema'] for candidate in candidates]
        if prediction.get('n') != n or prediction.get(
            'candidate_topics'
        ) != candidate_names:
            raise SystemExit(
                f'{location}: los candidatos guardados no coinciden con '
                'el ranking BGE calculado directamente.'
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
        prediction['candidate_scores'] = [
            dict(candidate) for candidate in candidates
        ]
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
    bge_context: dict[str, Any],
) -> None:
    output_path = results_dir / f'bge_top_{n}_llama_results.json'
    predictions = load_checkpoint(output_path, records, n, bge_context)
    completed_ids = {prediction['_id'] for prediction in predictions}
    pending = [record for record in records if record['_id'] not in completed_ids]
    order = {record['_id']: index for index, record in enumerate(records)}

    def checkpoint() -> dict[str, Any]:
        predictions.sort(key=lambda prediction: order[prediction['_id']])
        report = build_report(records, predictions, topics, n, bge_context)
        bge.save_json(output_path, report)
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
    if args.bge_batch_size < 1:
        raise SystemExit('--bge-batch-size debe ser mayor que cero.')
    if args.bge_max_length < 1:
        raise SystemExit('--bge-max-length debe ser mayor que cero.')

    topics = bge.load_topics(args.topics)
    n_values = resolve_n_values(args.n_values, len(topics))
    test_records = bge.load_texts(args.data_dir)
    labelled_by_id = bge.load_ground_truth(
        args.labelled_data_dir, set(topics)
    )
    bge.validate_dataset(test_records, labelled_by_id)

    reranker, device, synchronize = bge.load_reranker(args.bge_model)
    print(f'Modelo BGE: {args.bge_model}')
    print(f'Dispositivo BGE: {device}')
    print(f'Noticias: {len(test_records)}; valores N: {n_values}')

    def score_pairs(pairs: list[list[str]]) -> Any:
        return reranker.compute_score(
            pairs,
            batch_size=args.bge_batch_size,
            max_length=args.bge_max_length,
            normalize=True,
        )

    bge_predictions, inference_times = bge.evaluate_records(
        test_records,
        labelled_by_id,
        topics,
        score_pairs,
        synchronize,
    )
    records = prepare_records(test_records, bge_predictions)
    normalized_labels = [
        {
            '_id': record['_id'],
            'text': labelled_by_id[record['_id']]['text'],
            'topics': labelled_by_id[record['_id']]['topics'],
        }
        for record in test_records
    ]
    bge_context = {
        'model': args.bge_model,
        'batch_size': args.bge_batch_size,
        'max_length': args.bge_max_length,
        'device': device,
        'ranking_sha256': ranking_sha256(records),
        'input_data_sha256': bge.sha256_json(test_records),
        'topic_catalog_sha256': bge.sha256_json(topics),
        'labelled_test_sha256': bge.sha256_json(normalized_labels),
        'data_dir': str(args.data_dir.resolve()),
        'labelled_data_dir': str(args.labelled_data_dir.resolve()),
        'timing': {
            'total_cross_encoder_time_seconds': round(
                sum(inference_times), 6
            ),
            'per_news_cross_encoder_time': bge.timing_summary(
                inference_times
            ),
            'scope': (
                'Tiempo de compute_score para los pares de los 12 temas de '
                'cada noticia, con sincronización CUDA cuando corresponde.'
            ),
        },
    }

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
            bge_context,
        )


if __name__ == '__main__':
    main()
