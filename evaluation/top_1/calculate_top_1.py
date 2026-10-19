#!/usr/bin/env python3
'''Calcula Top-1 para E5 y BGE, y evalúa Llama con todo el catálogo.'''

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EVALUATION_DIR.parent
DATASET_DIR = PROJECT_DIR / 'dataset_mteb'
RESULTS_DIR = SCRIPT_DIR / 'results'

if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

import evaluate_ollama_all_topics as llama_all  # noqa: E402
import evaluate_ollama_top_n_no_none as llama_base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Calcula los informes Top-1 de E5 y BGE sobre resultados '
            'precomputados y ejecuta Llama con los 12 temas.'
        )
    )
    parser.add_argument(
        '--llama-model',
        default=os.environ.get('OLLAMA_MODEL', 'llama3.2:1b'),
        help='Modelo de Ollama para el informe de Llama.',
    )
    parser.add_argument(
        '--skip-llama',
        action='store_true',
        help='Calcula únicamente E5 y BGE, sin iniciar ni reanudar Llama.',
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


def timing_summary(
    total_seconds: float, per_news_seconds: list[float], scope: str
) -> dict[str, Any]:
    sample_size = len(per_news_seconds)
    if sample_size:
        mean = statistics.fmean(per_news_seconds)
        if sample_size > 1:
            margin = (
                1.959963984540054
                * statistics.stdev(per_news_seconds)
                / math.sqrt(sample_size)
            )
        else:
            margin = 0.0
        interval = {
            'lower': round(max(0.0, mean - margin), 9),
            'upper': round(mean + margin, 9),
        }
    else:
        mean = 0.0
        interval = {'lower': 0.0, 'upper': 0.0}
    return {
        'total_execution_time_seconds': round(total_seconds, 6),
        'per_news_execution_time': {
            'sample_size': sample_size,
            'mean_seconds': round(mean, 9),
            'confidence_interval_95_seconds': interval,
            'method': 'Normal approximation for the sample mean',
        },
        'scope': scope,
    }


def load_truth(topic_catalog: set[str]) -> dict[str, list[str]]:
    raw_truth = llama_base.load_ground_truth(
        DATASET_DIR / 'data_validation', topic_catalog
    )
    truth: dict[str, list[str]] = {}
    for news_id, topics in raw_truth.items():
        if topics is None:
            raise SystemExit(
                'El dataset MTEB no debe contener la etiqueta "ninguno".'
            )
        truth[news_id] = topics
    return truth


def make_report(
    *,
    model: str,
    source: Path,
    candidate_count: int,
    truth: dict[str, list[str]],
    payload: list[Any],
    predict: Callable[[dict[str, Any], int], tuple[str, float]],
    score_field: str,
) -> dict[str, Any]:
    start_total = time.perf_counter()
    predictions: list[dict[str, Any]] = []
    per_news_seconds: list[float] = []
    seen_ids: set[str] = set()

    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise SystemExit(f'Registro {index}: se esperaba un objeto JSON.')
        news_id = llama_base.required_string(record, '_id', f'Registro {index}')
        if news_id in seen_ids:
            raise SystemExit(f'Registro {index}: _id duplicado {news_id!r}.')
        seen_ids.add(news_id)
        if news_id not in truth:
            continue

        start_news = time.perf_counter()
        predicted_topic, score = predict(record, index)
        elapsed = time.perf_counter() - start_news
        per_news_seconds.append(elapsed)
        correct_topics = truth[news_id]
        predictions.append(
            {
                '_id': news_id,
                'topics_correctos': correct_topics,
                'topic_predicho': predicted_topic,
                score_field: round(score, 6),
                'match': predicted_topic in correct_topics,
            }
        )

    missing_ids = set(truth) - seen_ids
    if missing_ids:
        raise SystemExit(
            f'Faltan resultados para {len(missing_ids)} noticias de validación.'
        )

    total_seconds = time.perf_counter() - start_total
    hits = sum(prediction['match'] for prediction in predictions)
    total = len(predictions)
    accuracy = hits / total if total else 0.0
    return {
        'configuration': {
            'model': model,
            'candidate_count': candidate_count,
            'source': str(source.resolve()),
        },
        'dataset': {
            'total_news': len(truth),
            'evaluable_news': total,
            'non_evaluable_news': len(truth) - total,
        },
        'timing': timing_summary(
            total_seconds,
            per_news_seconds,
            'Carga, validación y cálculo de Top-1 sobre resultados '
            'precomputados. La media por noticia mide su validación y '
            'selección individual; no incluye la inferencia original del '
            'modelo ni la carga inicial del JSON.',
        ),
        'metrics': {
            'top_1_hits': hits,
            'top_1_misses': total - hits,
            'top_1_accuracy': round(accuracy, 6),
            'top_1_accuracy_percent': round(accuracy * 100, 2),
            'top_1_accuracy_mean': round(accuracy, 6),
            'top_1_accuracy_mean_percent': round(accuracy * 100, 2),
            'top_1_accuracy_confidence_interval_95': wilson_interval(hits, total),
        },
        'predictions': predictions,
    }


def load_similarity_payload(directory: Path) -> list[Any]:
    paths = sorted(directory.glob('*.json'))
    if not paths:
        raise SystemExit(f'No se encontraron resultados E5 en {directory}.')
    payload: list[Any] = []
    for path in paths:
        contents = load_json(path)
        if not isinstance(contents, list):
            raise SystemExit(f'{path} debe contener una lista JSON.')
        payload.extend(contents)
    return payload


def calculate_e5(
    topics: dict[str, str], truth: dict[str, list[str]]
) -> dict[str, Any]:
    source = DATASET_DIR / 'semantic_similarity_results'
    top_key = f'top_{len(topics)}'

    def predict(record: dict[str, Any], index: int) -> tuple[str, float]:
        candidates = record.get(top_key)
        if not isinstance(candidates, list) or len(candidates) != len(topics):
            raise SystemExit(
                f'Registro E5 {index}: {top_key} debe contener '
                f'{len(topics)} candidatos.'
            )
        first = candidates[0]
        if not isinstance(first, dict):
            raise SystemExit(f'Registro E5 {index}: candidato Top-1 inválido.')
        topic = llama_base.required_string(first, 'tema', f'Registro E5 {index}')
        score = first.get('similarity')
        if topic not in topics or isinstance(score, bool) or not isinstance(
            score, (int, float)
        ) or not math.isfinite(float(score)):
            raise SystemExit(f'Registro E5 {index}: candidato Top-1 inválido.')
        return topic, float(score)

    return make_report(
        model='intfloat/multilingual-e5-base',
        source=source,
        candidate_count=len(topics),
        truth=truth,
        payload=load_similarity_payload(source),
        predict=predict,
        score_field='similarity',
    )


def calculate_bge(
    topics: dict[str, str], truth: dict[str, list[str]]
) -> dict[str, Any]:
    source = DATASET_DIR / 'reranked_topic_results.json'
    payload = load_json(source)
    if not isinstance(payload, list):
        raise SystemExit(f'{source} debe contener una lista JSON.')

    def predict(record: dict[str, Any], index: int) -> tuple[str, float]:
        results = record.get('rerank_results')
        if not isinstance(results, list):
            raise SystemExit(f'Registro BGE {index}: rerank_results inválido.')
        selected = next(
            (
                result
                for result in results
                if isinstance(result, dict) and result.get('k') == len(topics)
            ),
            None,
        )
        if selected is None:
            raise SystemExit(
                f'Registro BGE {index}: falta el reranking con K={len(topics)}.'
            )
        candidates = selected.get('reranked_topics')
        if not isinstance(candidates, list) or len(candidates) != len(topics):
            raise SystemExit(f'Registro BGE {index}: candidatos inválidos.')
        first = candidates[0]
        if not isinstance(first, dict):
            raise SystemExit(f'Registro BGE {index}: candidato Top-1 inválido.')
        topic = llama_base.required_string(first, 'tema', f'Registro BGE {index}')
        score = first.get('cross_encoder_score')
        if topic not in topics or isinstance(score, bool) or not isinstance(
            score, (int, float)
        ) or not math.isfinite(float(score)):
            raise SystemExit(f'Registro BGE {index}: candidato Top-1 inválido.')
        return topic, float(score)

    return make_report(
        model='BAAI/bge-reranker-v2-m3',
        source=source,
        candidate_count=len(topics),
        truth=truth,
        payload=payload,
        predict=predict,
        score_field='cross_encoder_score',
    )


def calculate_llama(model: str, topic_count: int) -> dict[str, Any]:
    output_path = RESULTS_DIR / 'llama_top_1_results.json'
    llama_base.MODEL_NAME = model
    llama_base.MODEL_LABEL = f'Ollama {model}'
    llama_base.ALLOW_NONE_DECISION = False
    llama_base.PROMPT_VERSION = 'mteb_forced_topic_selection_v1'
    llama_all.PROMPT_VERSION = f'{llama_base.PROMPT_VERSION}_all_topics_v1'
    llama_all.CANDIDATE_MODE = 'all_topics_no_none'
    llama_all.OUTPUT_FILE = str(output_path)
    llama_all.main()

    report = load_json(output_path)
    if not isinstance(report, dict):
        raise SystemExit(f'Informe de Llama no válido: {output_path}')
    predictions = report.get('predictions')
    metrics = report.get('metrics')
    configuration = report.get('configuration')
    if not isinstance(predictions, list) or not isinstance(metrics, dict):
        raise SystemExit(f'Informe de Llama incompleto: {output_path}')
    if not isinstance(configuration, dict):
        raise SystemExit(f'Configuración de Llama inválida: {output_path}')

    hits = sum(bool(prediction.get('matches_real_topic')) for prediction in predictions)
    total = len(predictions)
    accuracy = hits / total if total else 0.0
    elapsed_times = [
        float(prediction['inference_time_seconds'])
        for prediction in predictions
        if isinstance(prediction, dict)
        and isinstance(prediction.get('inference_time_seconds'), (int, float))
        and not isinstance(prediction.get('inference_time_seconds'), bool)
    ]
    configuration['candidate_count'] = topic_count
    metrics.update(
        {
            'top_1_hits': hits,
            'top_1_misses': total - hits,
            'top_1_accuracy': round(accuracy, 6),
            'top_1_accuracy_percent': round(accuracy * 100, 2),
            'top_1_accuracy_mean': round(accuracy, 6),
            'top_1_accuracy_mean_percent': round(accuracy * 100, 2),
            'top_1_accuracy_confidence_interval_95': wilson_interval(hits, total),
        }
    )
    report['timing'] = {
        'total_inference_time_seconds': round(sum(elapsed_times), 6),
        'per_news_inference_time': timing_summary(
            sum(elapsed_times), elapsed_times, ''
        )['per_news_execution_time'],
        'scope': (
            'Tiempo medido alrededor de cada selección de Llama mediante la '
            'API de Ollama; incluye la llamada HTTP y la inferencia.'
        ),
    }
    save_json(output_path, report)
    return report


def print_accuracy(label: str, report: dict[str, Any]) -> None:
    metrics = report['metrics']
    print(
        f"{label}: {metrics['top_1_hits']}/"
        f"{metrics['top_1_hits'] + metrics['top_1_misses']} "
        f"({metrics['top_1_accuracy_percent']:.2f}%)"
    )


def main() -> None:
    args = parse_args()
    topics = llama_base.load_temas(DATASET_DIR / 'temas.json')
    truth = load_truth(set(topics))

    e5_report = calculate_e5(topics, truth)
    save_json(RESULTS_DIR / 'e5_top_1_results.json', e5_report)
    print_accuracy('E5 Top-1', e5_report)

    bge_report = calculate_bge(topics, truth)
    save_json(RESULTS_DIR / 'bge_top_1_results.json', bge_report)
    print_accuracy('BGE Top-1', bge_report)

    if not args.skip_llama:
        llama_report = calculate_llama(args.llama_model, len(topics))
        print_accuracy('Llama Top-1', llama_report)


if __name__ == '__main__':
    main()
