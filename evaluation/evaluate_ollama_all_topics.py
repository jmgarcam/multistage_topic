#!/usr/bin/env python3
'''Evalúa cada noticia con todo el catálogo de temas mediante Ollama.'''

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import evaluate_ollama_top_n_no_none as base


DATASET_DIR = Path(__file__).resolve().parent.parent / 'dataset_mteb'
SOURCE_RESULTS_FILE = 'reranked_topic_results.json'
VALIDATION_DIR = 'data_validation'
TOPICS_FILE = 'temas.json'
OUTPUT_FILE = 'ollama_llama_3_2_3b_all_topics_no_none_results.json'
PROMPT_VERSION = f'{base.PROMPT_VERSION}_all_topics_v1'
CANDIDATE_MODE = 'all_topics_no_none'
CHECKPOINT_EVERY = 10


def resolve_path(configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else DATASET_DIR / path


def prepare_records(
    payload: Any,
    truth_by_id: dict[str, list[str] | None],
    topics: dict[str, str],
) -> list[dict[str, Any]]:
    '''Une textos y ground truth; ignora por completo el reranking.'''
    if not isinstance(payload, list) or not payload:
        raise SystemExit(
            'reranked_topic_results.json debe contener una lista no vacía.'
        )

    topic_candidates = [{'tema': topic} for topic in topics]
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record_index, record in enumerate(payload):
        location = f'Resultado {record_index}'
        if not isinstance(record, dict):
            raise SystemExit(f'{location}: se esperaba un objeto JSON.')
        news_id = base.required_string(record, '_id', location)
        if news_id in seen_ids:
            raise SystemExit(f'{location}: _id duplicado {news_id}.')
        seen_ids.add(news_id)
        source_file = base.required_string(record, 'file', location)
        text = base.required_string(record, 'text', location)
        if news_id not in truth_by_id:
            continue
        correct_topics = truth_by_id[news_id]
        if correct_topics is None:
            correct_topics = [base.NONE_TOPIC]
        records.append(
            {
                '_id': news_id,
                'file': source_file,
                'text': text,
                'topics_correctos': correct_topics,
                'candidates': [dict(candidate) for candidate in topic_candidates],
            }
        )

    truth_ids = set(truth_by_id)
    missing_ids = truth_ids - seen_ids
    if missing_ids:
        raise SystemExit(
            'Faltan textos para '
            f'{len(missing_ids)} noticias del conjunto de validación.'
        )
    return records


def validate_configuration() -> None:
    if CHECKPOINT_EVERY < 1:
        raise SystemExit('CHECKPOINT_EVERY debe ser mayor que cero.')
    if base.MAX_NEW_TOKENS < 1:
        raise SystemExit('MAX_NEW_TOKENS debe ser mayor que cero.')
    if base.VALIDITY_MAX_NEW_TOKENS < 1:
        raise SystemExit(
            'VALIDITY_MAX_NEW_TOKENS debe ser mayor que cero.'
        )


def load_checkpoint(
    path: Path, records: list[dict[str, Any]], topic_names: list[str]
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []

    payload = base.load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f'Checkpoint no válido: {path}')
    configuration = payload.get('configuration')
    expected_configuration = {
        'model': base.MODEL_NAME,
        'prompt_version': PROMPT_VERSION,
        'candidate_mode': CANDIDATE_MODE,
        'topic_count': len(topic_names),
    }
    if not isinstance(configuration, dict) or any(
        configuration.get(field) != value
        for field, value in expected_configuration.items()
    ):
        raise SystemExit(
            'El checkpoint existente usa otra configuración. Muévelo o '
            'elimínalo antes de iniciar este experimento.'
        )

    predictions = payload.get('predictions')
    if not isinstance(predictions, list):
        raise SystemExit('El checkpoint no contiene una lista predictions.')

    records_by_id = {record['_id']: record for record in records}
    seen_ids: set[str] = set()
    for prediction_index, prediction in enumerate(predictions):
        location = f'Checkpoint, predicción {prediction_index}'
        if not isinstance(prediction, dict):
            raise SystemExit(f'{location}: se esperaba un objeto JSON.')
        news_id = base.required_string(prediction, '_id', location)
        if news_id not in records_by_id:
            raise SystemExit(f'{location}: _id desconocido {news_id!r}.')
        if news_id in seen_ids:
            raise SystemExit(f'{location}: _id duplicado {news_id!r}.')
        seen_ids.add(news_id)
        if prediction.get('candidate_topics') != topic_names:
            raise SystemExit(
                f'{location}: candidate_topics no coincide con temas.json.'
            )

        decision = base.validate_saved_decision(
            prediction.get(base.DECISION_FIELD), topic_names, location
        )
        prediction[base.DECISION_FIELD] = decision
        base.validate_prediction_outputs(
            prediction, topic_names, decision, location
        )

        record = records_by_id[news_id]
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
            raise SystemExit(f'{location}: tiempo de inferencia no válido.')
    return predictions


def build_report(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    total_news: int,
    topic_names: list[str],
) -> dict[str, Any]:
    correct = sum(result['matches_real_topic'] for result in predictions)
    none_decisions = sum(
        base.decision_is_none(result[base.DECISION_FIELD])
        for result in predictions
    )
    invalid_decisions = sum(
        result[base.DECISION_FIELD] == 'invalid' for result in predictions
    )
    none_results = [
        result
        for result in predictions
        if result['topics_correctos'] == [base.NONE_TOPIC]
    ]
    correct_none = sum(result['matches_real_topic'] for result in none_results)
    elapsed_times = [
        float(result['inference_time_seconds']) for result in predictions
    ]

    return {
        'status': (
            'complete' if len(predictions) == len(records) else 'in_progress'
        ),
        'configuration': {
            'model': base.MODEL_NAME,
            'max_new_tokens': base.MAX_NEW_TOKENS,
            'validity_max_new_tokens': (
                base.VALIDITY_MAX_NEW_TOKENS
                if base.ALLOW_NONE_DECISION
                else None
            ),
            'prompt_version': PROMPT_VERSION,
            'decision_strategy': (
                'validity_gate_then_topic_selection'
                if base.ALLOW_NONE_DECISION
                else 'forced_topic_selection'
            ),
            'candidate_mode': CANDIDATE_MODE,
            'topic_count': len(topic_names),
            'none_ground_truth_excluded': False,
            'ground_truth_source': 'data_validation',
            'prediction_format': base.PREDICTION_FORMAT,
            'multi_topic_match': 'any',
            'generation': 'Ollama API (temperature=0, seed=0)',
            'api_base_url': base.get_base_url(),
        },
        'dataset': {
            'total_news': total_news,
            'evaluable_news': len(records),
            'excluded_none_news': total_news - len(records),
            'evaluated_none_news': sum(
                record['topics_correctos'] == [base.NONE_TOPIC]
                for record in records
            ),
            'multi_topic_news': sum(
                len(record['topics_correctos']) > 1 for record in records
            ),
            'expected_predictions': len(records),
            'completed_predictions': len(predictions),
        },
        'metrics': {
            'evaluated_news': len(predictions),
            'correct_predictions': correct,
            'incorrect_predictions': len(predictions) - correct,
            'none_decisions': none_decisions,
            'invalid_decisions': invalid_decisions,
            'accuracy': (
                round(correct / len(predictions), 6) if predictions else None
            ),
            'accuracy_percent': (
                round(correct * 100 / len(predictions), 2)
                if predictions
                else None
            ),
            'ground_truth_none_news': len(none_results),
            'correct_none_predictions': correct_none,
            'none_accuracy': (
                round(correct_none / len(none_results), 6)
                if none_results
                else None
            ),
            'false_none_decisions': none_decisions - correct_none,
            'mean_inference_time_seconds': (
                round(statistics.fmean(elapsed_times), 6)
                if elapsed_times
                else None
            ),
        },
        'predictions': predictions,
    }


def print_summary(report: dict[str, Any]) -> None:
    dataset = report['dataset']
    metrics = report['metrics']
    print('Estado: ' + str(report['status']))
    completed = dataset['completed_predictions']
    expected = dataset['expected_predictions']
    accuracy = metrics['accuracy_percent']
    if accuracy is None:
        print(f'0/{expected} decisiones completadas')
    else:
        print(
            f"{metrics['correct_predictions']}/{completed} coincidencias "
            f'({accuracy:.2f}%); progreso {completed}/{expected}'
        )


def main() -> None:
    validate_configuration()
    topics = base.load_temas(resolve_path(TOPICS_FILE))
    topic_names = list(topics)
    truth_by_id = base.load_ground_truth(
        resolve_path(VALIDATION_DIR), set(topic_names)
    )
    records = prepare_records(
        base.load_json(resolve_path(SOURCE_RESULTS_FILE)), truth_by_id, topics
    )
    if not records:
        raise SystemExit(
            f'No hay noticias evaluables para ejecutar {base.MODEL_LABEL}.'
        )

    output_path = resolve_path(OUTPUT_FILE)
    predictions = load_checkpoint(output_path, records, topic_names)
    completed_ids = {prediction['_id'] for prediction in predictions}
    pending = [record for record in records if record['_id'] not in completed_ids]

    record_order = {
        record['_id']: index for index, record in enumerate(records)
    }

    def checkpoint() -> dict[str, Any]:
        predictions.sort(key=lambda prediction: record_order[prediction['_id']])
        report = build_report(
            records, predictions, len(truth_by_id), topic_names
        )
        base.save_report(output_path, report)
        return report

    if not pending:
        report = checkpoint()
        print(f'Resultados ya completos: {output_path}')
        print_summary(report)
        return

    print(f'Noticias evaluables: {len(records)}')
    print(f'Temas candidatos por noticia: {len(topic_names)}')
    print(f'Decisiones pendientes: {len(pending)}')
    client, tokenizer, device = base.load_model()
    generated_since_checkpoint = 0

    try:
        for pending_index, record in enumerate(pending, start=1):
            start_time = time.perf_counter()
            decision, validity_raw_output, raw_output = base.generate_decision(
                client,
                tokenizer,
                device,
                record['text'],
                record['candidates'],
                topics,
            )
            elapsed_seconds = time.perf_counter() - start_time
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
                    'inference_time_seconds': round(elapsed_seconds, 6),
                }
            )
            generated_since_checkpoint += 1
            if generated_since_checkpoint >= CHECKPOINT_EVERY:
                checkpoint()
                print(
                    f'Progreso: {pending_index}/{len(pending)} decisiones '
                    'pendientes procesadas.'
                )
                generated_since_checkpoint = 0
    except BaseException:
        checkpoint()
        print(f'Checkpoint guardado tras una interrupción: {output_path}')
        raise

    report = checkpoint()
    print(f'Resultados guardados: {output_path}')
    print_summary(report)


if __name__ == '__main__':
    main()
