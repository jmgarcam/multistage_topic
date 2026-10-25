#!/usr/bin/env python3
'''Calcula E5 sobre test y evalúa un SLM con cada Top-N posible.'''

from __future__ import annotations

import argparse
import math
import os
import re
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
DEFAULT_LLAMA_MODEL = 'llama3.2:3b'
DEFAULT_SIMILARITY_MARGIN_THRESHOLD = 0.0025
DEFAULT_RESULTS_DIR = (
    SCRIPT_DIR
    / 'results_llama3_2_3b_temas_order_two_prompts_e5_margin_0_0025'
)
DEFAULT_CHECKPOINT_EVERY = 10
PROMPT_VERSION = (
    'mteb_forced_topic_selection_v1_e5_top_n_v5_margin_gate_sentence_id'
)
NO_MARGIN_PROMPT_VERSION = (
    'mteb_forced_topic_selection_v1_e5_top_n_v5_no_margin_sentence_id'
)
CANDIDATE_ORDER = 'temas.json catalog order after E5 margin filtering'
NO_MARGIN_CANDIDATE_ORDER = 'temas.json catalog order of E5 Top-N'
ROUTING_STRATEGY = (
    'preserve_e5_top_1_when_margin_gte_threshold_else_slm'
)
NO_MARGIN_ROUTING_STRATEGY = 'always_route_e5_top_n_to_slm'
NEAR_CANDIDATE_RULE = (
    'top_1_similarity - candidate_similarity < threshold, capped at N'
)
NO_MARGIN_CANDIDATE_RULE = 'first N topics of E5 cosine-similarity ranking'
E5_DECISION_SOURCE = 'e5_top_1'
SLM_DECISION_SOURCE = 'slm'
CONFIDENT_MARGIN_REASON = 'margin_gte_threshold'
SINGLE_CANDIDATE_REASON = 'single_candidate_preserves_e5_top_1'
AMBIGUOUS_MARGIN_REASON = 'margin_lt_threshold'
NO_MARGIN_REASON = 'margin_gate_disabled'
FINAL_DECISION_FIELD = 'final_decision'
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
            'Calcula el ranking E5 desde data_test, conserva E5 Top-1 en '
            'casos de confianza y ejecuta el SLM únicamente para noticias '
            'ambiguas. Sin valores N evalúa desde 1 hasta el número total '
            'de temas.'
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
        default=os.environ.get('OLLAMA_MODEL', DEFAULT_LLAMA_MODEL),
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
        help='Archivo JSON o directorio con las noticias de test.',
    )
    parser.add_argument(
        '--labelled-data-dir',
        type=Path,
        default=DATASET_DIR / 'data_test_labelled',
        help='Archivo JSON o directorio etiquetado, usado solo para métricas.',
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
        '--result-slug',
        default=os.environ.get('RESULT_SLUG'),
        help=(
            'Identificador del modelo usado en los nombres de los JSON. '
            'Por defecto se deriva de --model.'
        ),
    )
    parser.add_argument(
        '--similarity-margin-threshold',
        type=float,
        default=DEFAULT_SIMILARITY_MARGIN_THRESHOLD,
        help=(
            'Conserva E5 Top-1 cuando similitud_top1 - similitud_top2 '
            'es mayor o igual que este umbral. En casos ambiguos el SLM '
            'recibe los candidatos situados dentro del mismo margen '
            '(por defecto: %(default)s).'
        ),
    )
    parser.add_argument(
        '--disable-similarity-margin-gate',
        action='store_true',
        help=(
            'Desactiva el enrutado por margen: todas las noticias se envían '
            'al SLM con los primeros N candidatos E5.'
        ),
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


def model_result_slug(model: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', model.casefold()).strip('_')
    if not slug:
        raise SystemExit('No se pudo derivar un identificador desde --model.')
    return slug


def resolve_result_slug(requested: str | None, model: str) -> str:
    slug = requested or model_result_slug(model)
    if not re.fullmatch(r'[a-z0-9]+(?:_[a-z0-9]+)*', slug):
        raise SystemExit(
            '--result-slug solo admite minúsculas, números y guiones bajos.'
        )
    return slug


def validate_similarity_margin_threshold(value: float) -> float:
    if not math.isfinite(value) or value < 0 or value > 2:
        raise SystemExit(
            '--similarity-margin-threshold debe estar entre 0 y 2.'
        )
    return value


def similarity_margin(higher: float, lower: float) -> float:
    return round(float(higher) - float(lower), 12)


def route_prediction(
    record: dict[str, Any],
    topics: dict[str, str],
    n: int,
    similarity_margin_threshold: float | None,
) -> dict[str, Any]:
    ranked_candidates = record['candidates']
    if len(ranked_candidates) < 2:
        raise RuntimeError(
            'El enrutado por margen necesita al menos dos candidatos E5.'
        )

    top_1 = ranked_candidates[0]
    top_2 = ranked_candidates[1]
    margin = similarity_margin(
        top_1['similarity'], top_2['similarity']
    )
    if margin < 0:
        raise RuntimeError('El ranking E5 no está ordenado por similitud.')

    common = {
        'e5_top_1_topic': top_1['tema'],
        'e5_top_1_similarity': float(top_1['similarity']),
        'e5_top_2_topic': top_2['tema'],
        'e5_top_2_similarity': float(top_2['similarity']),
        'e5_top_1_top_2_margin': margin,
    }
    if similarity_margin_threshold is None:
        return {
            **common,
            'decision_source': SLM_DECISION_SOURCE,
            'routing_reason': NO_MARGIN_REASON,
            'candidates': base.order_candidates_by_catalog(
                [dict(candidate) for candidate in ranked_candidates[:n]],
                topics,
            ),
        }
    if n == 1:
        return {
            **common,
            'decision_source': E5_DECISION_SOURCE,
            'routing_reason': SINGLE_CANDIDATE_REASON,
            'candidates': [dict(top_1)],
        }
    if margin >= similarity_margin_threshold:
        return {
            **common,
            'decision_source': E5_DECISION_SOURCE,
            'routing_reason': CONFIDENT_MARGIN_REASON,
            'candidates': [dict(top_1)],
        }

    nearby_ranked_candidates = [
        dict(candidate)
        for candidate in ranked_candidates[:n]
        if (
            similarity_margin(
                top_1['similarity'], candidate['similarity']
            )
            < similarity_margin_threshold
        )
    ]
    if len(nearby_ranked_candidates) < 2:
        raise RuntimeError(
            'Un caso ambiguo debe contener al menos los dos primeros temas.'
        )
    return {
        **common,
        'decision_source': SLM_DECISION_SOURCE,
        'routing_reason': AMBIGUOUS_MARGIN_REASON,
        'candidates': base.order_candidates_by_catalog(
            nearby_ranked_candidates, topics
        ),
    }


def routing_configuration(
    similarity_margin_threshold: float | None,
) -> dict[str, Any]:
    if similarity_margin_threshold is None:
        return {
            'candidate_order': NO_MARGIN_CANDIDATE_ORDER,
            'routing_strategy': NO_MARGIN_ROUTING_STRATEGY,
            'similarity_margin_threshold': None,
            'similarity_margin_gate_enabled': False,
            'near_candidate_rule': NO_MARGIN_CANDIDATE_RULE,
            'prompt_version': NO_MARGIN_PROMPT_VERSION,
        }
    return {
        'candidate_order': CANDIDATE_ORDER,
        'routing_strategy': ROUTING_STRATEGY,
        'similarity_margin_threshold': similarity_margin_threshold,
        'similarity_margin_gate_enabled': True,
        'near_candidate_rule': NEAR_CANDIDATE_RULE,
        'prompt_version': PROMPT_VERSION,
    }


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
    similarity_margin_threshold: float | None,
) -> dict[str, Any]:
    routing = routing_configuration(similarity_margin_threshold)
    final_hits = sum(
        prediction['matches_real_topic'] for prediction in predictions
    )
    evaluated = len(predictions)
    final_accuracy = final_hits / evaluated if evaluated else 0.0
    llama_predictions = [
        prediction
        for prediction in predictions
        if prediction['decision_source'] == SLM_DECISION_SOURCE
    ]
    e5_predictions = [
        prediction
        for prediction in predictions
        if prediction['decision_source'] == E5_DECISION_SOURCE
    ]
    llama_routed = len(llama_predictions)
    e5_direct = len(e5_predictions)
    llama_routed_hits = sum(
        prediction['matches_real_topic']
        for prediction in llama_predictions
    )
    e5_direct_hits = sum(
        prediction['matches_real_topic'] for prediction in e5_predictions
    )
    llama_routed_accuracy = (
        llama_routed_hits / llama_routed if llama_routed else 0.0
    )
    e5_direct_accuracy = e5_direct_hits / e5_direct if e5_direct else 0.0
    llama_candidate_available = sum(
        prediction['candidate_contains_real_topic']
        for prediction in llama_predictions
    )
    conditional_selection_accuracy = (
        llama_routed_hits / llama_candidate_available
        if llama_candidate_available
        else 0.0
    )
    llama_invalid_decisions = sum(
        prediction[base.DECISION_FIELD] == 'invalid'
        for prediction in llama_predictions
    )
    llama_topic_changes = sum(
        prediction[base.DECISION_FIELD] != 'invalid'
        and prediction[base.DECISION_FIELD]
        != prediction['e5_top_1_topic']
        for prediction in llama_predictions
    )
    llama_helped = sum(
        prediction['e5_top_1_topic'] not in prediction['topics_correctos']
        and prediction['matches_real_topic']
        for prediction in llama_predictions
    )
    llama_harmed = sum(
        prediction['e5_top_1_topic'] in prediction['topics_correctos']
        and not prediction['matches_real_topic']
        for prediction in llama_predictions
    )
    evaluated_e5_hits = sum(
        prediction['e5_top_1_topic'] in prediction['topics_correctos']
        for prediction in predictions
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
        for prediction in llama_predictions
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
            'slm_model': base.MODEL_NAME,
            'candidate_source': e5_context['candidate_source'],
            'uses_precomputed_e5_results': e5_context[
                'uses_precomputed_results'
            ],
            'precomputed_e5_results_source': e5_context[
                'precomputed_results_source'
            ],
            'n': n,
            'topic_count': len(topics),
            'candidate_order': routing['candidate_order'],
            'routing_strategy': routing['routing_strategy'],
            'similarity_margin_threshold': routing[
                'similarity_margin_threshold'
            ],
            'similarity_margin_gate_enabled': routing[
                'similarity_margin_gate_enabled'
            ],
            'near_candidate_rule': routing['near_candidate_rule'],
            'n_is_maximum_slm_candidate_count': True,
            'prompt_version': routing['prompt_version'],
            'prompt_strategy': base.TWO_STAGE_PROMPT_STRATEGY,
            'event_evidence_strategy': base.EVENT_EVIDENCE_STRATEGY,
            'slm_calls_per_valid_routed_prediction': 2,
            'slm_calls_per_e5_direct_prediction': 0,
            'event_extraction_candidates_visible': False,
            'decision_strategy': 'forced_topic_selection',
            'final_decision_field': FINAL_DECISION_FIELD,
            'legacy_decision_field': base.DECISION_FIELD,
            'prediction_format': base.PREDICTION_FORMAT,
            'generation': 'Ollama API (temperature=0, seed=0)',
            'metric_definitions': {
                'retriever_top_1_accuracy': (
                    'fraction of news whose highest-scoring retriever topic '
                    'is correct; equal to micro-precision for this dataset'
                ),
                'candidate_recall_at_n': (
                    'fraction of news with the correct topic among the first '
                    'N topics of the complete E5 ranking'
                ),
                'final_accuracy': (
                    'fraction of routed final decisions matching the correct '
                    'topic, whether the source was E5 or the SLM'
                ),
                'slm_final_accuracy': (
                    'alias of final_accuracy; it includes '
                    'decisions preserved directly from E5'
                ),
                'conditional_selection_accuracy': (
                    'SLM accuracy among routed news where the correct topic '
                    'was available in the candidate set'
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
            'final_hits': final_hits,
            'final_misses': evaluated - final_hits,
            'final_accuracy': round(final_accuracy, 6),
            'final_accuracy_percent': round(final_accuracy * 100, 2),
            'final_accuracy_confidence_interval_95': wilson_interval(
                final_hits, evaluated
            ),
            'slm_final_hits': final_hits,
            'slm_final_misses': evaluated - final_hits,
            'slm_final_accuracy': round(final_accuracy, 6),
            'slm_final_accuracy_percent': round(
                final_accuracy * 100, 2
            ),
            'slm_final_accuracy_confidence_interval_95': wilson_interval(
                final_hits, evaluated
            ),
            'e5_direct_decisions': e5_direct,
            'e5_direct_hits': e5_direct_hits,
            'e5_direct_accuracy': round(e5_direct_accuracy, 6),
            'e5_direct_accuracy_percent': round(
                e5_direct_accuracy * 100, 2
            ),
            'slm_routed_decisions': llama_routed,
            'slm_routed_percent': round(
                llama_routed / evaluated * 100 if evaluated else 0.0, 2
            ),
            'slm_routed_hits': llama_routed_hits,
            'slm_routed_accuracy': round(llama_routed_accuracy, 6),
            'slm_routed_accuracy_percent': round(
                llama_routed_accuracy * 100, 2
            ),
            'slm_invalid_decisions': llama_invalid_decisions,
            'slm_valid_topic_changes': llama_topic_changes,
            'slm_helped_vs_e5_top_1': llama_helped,
            'slm_harmed_vs_e5_top_1': llama_harmed,
            'net_hit_change_vs_e5_top_1': final_hits - evaluated_e5_hits,
            'candidate_recall_at_n_hits': candidate_recall_hits,
            'candidate_recall_at_n': round(candidate_recall, 6),
            'candidate_recall_at_n_percent': round(
                candidate_recall * 100, 2
            ),
            'retrieval_errors_at_n': len(records) - candidate_recall_hits,
            'slm_candidate_recall_routed_hits': (
                llama_candidate_available
            ),
            'slm_candidate_recall_routed': round(
                llama_candidate_available / llama_routed
                if llama_routed
                else 0.0,
                6,
            ),
            'selection_opportunities': llama_candidate_available,
            'selection_errors': (
                llama_candidate_available - llama_routed_hits
            ),
            'conditional_selection_accuracy': round(
                conditional_selection_accuracy, 6
            ),
            'conditional_selection_accuracy_percent': round(
                conditional_selection_accuracy * 100, 2
            ),
        },
        'timing': {
            'e5_ranking': e5_context['timing'],
            'slm_total_inference_time_seconds': round(
                sum(elapsed_times), 6
            ),
            'slm_per_routed_news_inference_time': mean_time_summary(
                elapsed_times
            ),
            'scope': (
                'El ranking E5 se obtiene una vez, por cálculo o desde un '
                'informe validado. '
                + (
                    'Todas las noticias ejecutan extracción y clasificación '
                    'en dos llamadas con el Top-N completo de E5.'
                    if similarity_margin_threshold is None
                    else 'Las noticias con margen suficiente conservan E5 '
                    'Top-1 sin llamar al SLM. Solo los casos ambiguos '
                    'ejecutan extracción y clasificación en dos llamadas.'
                )
            ),
        },
        'predictions': predictions,
    }


def load_checkpoint(
    path: Path,
    records: list[dict[str, Any]],
    topics: dict[str, str],
    n: int,
    e5_context: dict[str, Any],
    similarity_margin_threshold: float | None,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    report = e5.load_json(path)
    if not isinstance(report, dict):
        raise SystemExit(f'Checkpoint no válido: {path}')
    configuration = report.get('configuration')
    if base.uses_legacy_literal_evidence(configuration):
        print(
            f'{path}: resultado anterior con copia literal; se reiniciará '
            'en el mismo archivo con identificadores de frase.'
        )
        return []
    routing = routing_configuration(similarity_margin_threshold)
    common_expected = {
        'e5_model': e5_context['model'],
        'slm_model': base.MODEL_NAME,
        'n': n,
        'candidate_order': routing['candidate_order'],
        'prompt_version': routing['prompt_version'],
        'prompt_strategy': base.TWO_STAGE_PROMPT_STRATEGY,
        'event_evidence_strategy': base.EVENT_EVIDENCE_STRATEGY,
        'routing_strategy': routing['routing_strategy'],
        'similarity_margin_threshold': routing[
            'similarity_margin_threshold'
        ],
        'similarity_margin_gate_enabled': routing[
            'similarity_margin_gate_enabled'
        ],
        'near_candidate_rule': routing['near_candidate_rule'],
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
        route = route_prediction(
            record, topics, n, similarity_margin_threshold
        )
        candidates = route['candidates']
        candidate_names = [candidate['tema'] for candidate in candidates]
        expected_route_fields = {
            'decision_source': route['decision_source'],
            'routing_reason': route['routing_reason'],
            'slm_called': (
                route['decision_source'] == SLM_DECISION_SOURCE
            ),
            'e5_top_1_topic': route['e5_top_1_topic'],
            'e5_top_1_similarity': route['e5_top_1_similarity'],
            'e5_top_2_topic': route['e5_top_2_topic'],
            'e5_top_2_similarity': route['e5_top_2_similarity'],
            'e5_top_1_top_2_margin': route['e5_top_1_top_2_margin'],
        }
        if (
            prediction.get('n') != n
            or prediction.get('candidate_topics') != candidate_names
            or any(
                prediction.get(key) != value
                for key, value in expected_route_fields.items()
            )
        ):
            raise SystemExit(
                f'{location}: el enrutado o los candidatos guardados no '
                'coinciden con el ranking E5 y la configuración de enrutado.'
            )
        decision = base.validate_saved_decision(
            prediction.get(base.DECISION_FIELD), candidate_names, location
        )
        prediction[base.DECISION_FIELD] = decision
        if prediction.get(FINAL_DECISION_FIELD) != decision:
            raise SystemExit(
                f'{location}: final_decision no coincide con la decisión '
                'guardada.'
            )
        if route['decision_source'] == E5_DECISION_SOURCE:
            if decision != route['e5_top_1_topic']:
                raise SystemExit(
                    f'{location}: una decisión directa debe conservar E5 '
                    'Top-1.'
                )
            if any(
                prediction.get(field) is not None
                for field in (
                    base.MAIN_EVENT_FIELD,
                    base.EVIDENCE_SENTENCE_ID_FIELD,
                    base.EVIDENCE_SENTENCE_FIELD,
                    base.EVENT_EXTRACTION_RAW_OUTPUT_FIELD,
                    base.RAW_OUTPUT_FIELD,
                )
            ):
                raise SystemExit(
                    f'{location}: una decisión directa no debe contener '
                    'salidas del SLM.'
                )
        else:
            base.validate_two_stage_prediction_outputs(
                prediction,
                record['text'],
                candidate_names,
                decision,
                location,
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
    e5_context: dict[str, Any],
    similarity_margin_threshold: float | None,
    result_slug: str,
) -> None:
    output_path = results_dir / f'e5_top_{n}_{result_slug}_results.json'
    predictions = load_checkpoint(
        output_path,
        records,
        topics,
        n,
        e5_context,
        similarity_margin_threshold,
    )
    completed_ids = {prediction['_id'] for prediction in predictions}
    pending = [record for record in records if record['_id'] not in completed_ids]
    order = {record['_id']: index for index, record in enumerate(records)}

    def checkpoint() -> dict[str, Any]:
        predictions.sort(key=lambda prediction: order[prediction['_id']])
        report = build_report(
            records,
            predictions,
            topics,
            n,
            e5_context,
            similarity_margin_threshold,
        )
        e5.save_json(output_path, report)
        return report

    if not pending:
        report = checkpoint()
        accuracy = report['metrics']['final_accuracy_percent']
        print(f'N={n}: completo, {accuracy:.2f}%')
        return

    generated_since_checkpoint = 0

    try:
        for pending_index, record in enumerate(pending, start=1):
            route = route_prediction(
                record, topics, n, similarity_margin_threshold
            )
            candidates = route['candidates']
            if route['decision_source'] == E5_DECISION_SOURCE:
                decision = route['e5_top_1_topic']
                extraction = None
                event_raw_output = None
                raw_output = None
                elapsed = 0.0
            else:
                if not client_holder:
                    client, _, _ = base.load_model()
                    client_holder.append(client)
                start = time.perf_counter()
                (
                    decision,
                    extraction,
                    event_raw_output,
                    raw_output,
                ) = base.generate_two_stage_decision(
                    client_holder[0], record['text'], candidates, topics
                )
                elapsed = time.perf_counter() - start
            candidate_names = [candidate['tema'] for candidate in candidates]
            predictions.append(
                {
                    '_id': record['_id'],
                    'file': record['file'],
                    'topics_correctos': record['topics_correctos'],
                    'n': n,
                    'decision_source': route['decision_source'],
                    'routing_reason': route['routing_reason'],
                    'slm_called': (
                        route['decision_source'] == SLM_DECISION_SOURCE
                    ),
                    'e5_top_1_topic': route['e5_top_1_topic'],
                    'e5_top_1_similarity': route['e5_top_1_similarity'],
                    'e5_top_2_topic': route['e5_top_2_topic'],
                    'e5_top_2_similarity': route['e5_top_2_similarity'],
                    'e5_top_1_top_2_margin': route[
                        'e5_top_1_top_2_margin'
                    ],
                    'candidate_topics': candidate_names,
                    'candidate_scores': [
                        dict(candidate) for candidate in candidates
                    ],
                    'candidate_contains_real_topic': any(
                        topic in candidate_names
                        for topic in record['topics_correctos']
                    ),
                    base.DECISION_FIELD: decision,
                    FINAL_DECISION_FIELD: decision,
                    base.MAIN_EVENT_FIELD: (
                        extraction[base.MAIN_EVENT_FIELD]
                        if extraction is not None
                        else None
                    ),
                    base.EVIDENCE_SENTENCE_ID_FIELD: (
                        extraction[base.EVIDENCE_SENTENCE_ID_FIELD]
                        if extraction is not None
                        else None
                    ),
                    base.EVIDENCE_SENTENCE_FIELD: (
                        extraction[base.EVIDENCE_SENTENCE_FIELD]
                        if extraction is not None
                        else None
                    ),
                    base.EVENT_EXTRACTION_RAW_OUTPUT_FIELD: event_raw_output,
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
    metrics = report['metrics']
    accuracy = metrics['final_accuracy_percent']
    routed = metrics['slm_routed_decisions']
    print(
        f'N={n}: completo, {accuracy:.2f}%; '
        f'{routed}/{len(records)} noticias enviadas al SLM'
    )


def main() -> None:
    args = parse_args()
    if args.checkpoint_every < 1:
        raise SystemExit('--checkpoint-every debe ser mayor que cero.')
    if args.e5_batch_size < 1:
        raise SystemExit('--e5-batch-size debe ser mayor que cero.')
    similarity_margin_threshold = (
        None
        if args.disable_similarity_margin_gate
        else validate_similarity_margin_threshold(
            args.similarity_margin_threshold
        )
    )

    topics = e5.load_topics(args.topics)
    n_values = resolve_n_values(args.n_values, len(topics))
    test_records = e5.load_test_records(args.data_dir)
    labelled_by_id = e5.load_labelled_test(
        args.labelled_data_dir, set(topics)
    )
    e5.validate_dataset(test_records, labelled_by_id)

    print(f'Modelo E5: {args.e5_model}')
    print(f'Noticias: {len(test_records)}; valores N: {n_values}')
    if similarity_margin_threshold is None:
        print('Enrutado por margen E5: desactivado')
    else:
        print(
            'Umbral de margen E5 Top-1/Top-2: '
            f'{similarity_margin_threshold:g}'
        )
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
    result_slug = resolve_result_slug(args.result_slug, args.model)
    print(f'Identificador de resultados: {result_slug}')
    for n in n_values:
        evaluate_n(
            records,
            topics,
            n,
            args.checkpoint_every,
            client_holder,
            args.results_dir,
            e5_context,
            similarity_margin_threshold,
            result_slug,
        )


if __name__ == '__main__':
    main()
