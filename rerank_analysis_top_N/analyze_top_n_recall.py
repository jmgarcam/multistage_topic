#!/usr/bin/env python3
'''Calcula Recall@N sobre los candidatos producidos por el reranker.'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RERANK_RESULTS_FILE = '../reranked_topic_results.json'
POSITIONS_FILE = '../semantic_similarity_validation/top_14_positions.json'
OUTPUT_FILE = 'top_n_recall_results.json'
K = 10
N_VALUES = [1, 3, 5, 7, 10]
NONE_TOPIC = 'ninguno'


def resolve_path(configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def load_json(path: Path) -> Any:
    try:
        with path.open('r', encoding='utf-8') as file_handle:
            return json.load(file_handle)
    except FileNotFoundError as exc:
        raise SystemExit(f'No se encontró el archivo: {path}') from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f'JSON no válido en {path}: {exc}') from exc


def required_string(record: dict[str, Any], field: str, location: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f'{location}: falta un campo {field} no vacío.')
    return value


def load_ground_truth(payload: Any) -> dict[str, str | None]:
    if not isinstance(payload, dict):
        raise SystemExit('top_14_positions.json debe contener un objeto JSON.')

    summary = payload.get('resumen')
    news = payload.get('noticias')
    if not isinstance(summary, dict):
        raise SystemExit('top_14_positions.json no contiene un resumen válido.')
    if not isinstance(news, list) or not news:
        raise SystemExit(
            'top_14_positions.json debe contener una lista noticias no vacía.'
        )

    truth_by_id: dict[str, str | None] = {}
    for record_index, record in enumerate(news):
        location = f'Etiqueta {record_index}'
        if not isinstance(record, dict):
            raise SystemExit(f'{location}: se esperaba un objeto JSON.')

        news_id = required_string(record, '_id', location)
        if news_id in truth_by_id:
            raise SystemExit(f'{location}: _id duplicado {news_id}.')

        correct_topic = required_string(record, 'topic_correcto', location)
        truth_by_id[news_id] = (
            None if correct_topic.casefold() == NONE_TOPIC else correct_topic
        )

    evaluable_count = sum(topic is not None for topic in truth_by_id.values())
    none_count = len(truth_by_id) - evaluable_count
    expected_counts = {
        'total_noticias': len(truth_by_id),
        'noticias_evaluables': evaluable_count,
        'noticias_no_evaluables': none_count,
    }
    for field, expected_value in expected_counts.items():
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f'resumen.{field} debe ser un entero.')
        if value != expected_value:
            raise SystemExit(
                f'resumen.{field}={value} no coincide con {expected_value} '
                'registros.'
            )

    return truth_by_id


def load_reranked_topics(payload: Any, k: int) -> dict[str, list[str]]:
    if not isinstance(payload, list) or not payload:
        raise SystemExit(
            'reranked_topic_results.json debe contener una lista no vacía.'
        )

    topics_by_id: dict[str, list[str]] = {}
    for record_index, record in enumerate(payload):
        location = f'Resultado {record_index}'
        if not isinstance(record, dict):
            raise SystemExit(f'{location}: se esperaba un objeto JSON.')

        news_id = required_string(record, '_id', location)
        if news_id in topics_by_id:
            raise SystemExit(f'{location}: _id duplicado {news_id}.')

        rerank_results = record.get('rerank_results')
        if not isinstance(rerank_results, list) or not rerank_results:
            raise SystemExit(
                f'{location}: rerank_results debe ser una lista no vacía.'
            )

        selected_result: dict[str, Any] | None = None
        seen_k_values: set[int] = set()
        for result_index, result in enumerate(rerank_results):
            result_location = f'{location}, rerank_result {result_index}'
            if not isinstance(result, dict):
                raise SystemExit(
                    f'{result_location}: se esperaba un objeto JSON.'
                )

            result_k = result.get('k')
            if (
                isinstance(result_k, bool)
                or not isinstance(result_k, int)
                or result_k < 1
            ):
                raise SystemExit(
                    f'{result_location}: k debe ser un entero positivo.'
                )
            if result_k in seen_k_values:
                raise SystemExit(
                    f'{result_location}: valor k duplicado: {result_k}.'
                )
            seen_k_values.add(result_k)
            if result_k == k:
                selected_result = result

        if selected_result is None:
            raise SystemExit(f'{location}: no contiene resultados para k={k}.')

        candidates = selected_result.get('reranked_topics')
        if not isinstance(candidates, list) or len(candidates) != k:
            raise SystemExit(
                f'{location}: reranked_topics para k={k} debe contener '
                f'exactamente {k} temas.'
            )

        topic_names: list[str] = []
        for candidate_index, candidate in enumerate(candidates):
            candidate_location = (
                f'{location}, k={k}, candidato {candidate_index}'
            )
            if not isinstance(candidate, dict):
                raise SystemExit(
                    f'{candidate_location}: se esperaba un objeto JSON.'
                )
            topic = required_string(candidate, 'tema', candidate_location)
            if topic in topic_names:
                raise SystemExit(
                    f'{candidate_location}: tema duplicado {topic!r}.'
                )
            topic_names.append(topic)

        topics_by_id[news_id] = topic_names

    return topics_by_id


def validate_configuration() -> None:
    if isinstance(K, bool) or not isinstance(K, int) or K < 1:
        raise SystemExit('K debe ser un entero positivo.')
    if not isinstance(N_VALUES, list) or not N_VALUES:
        raise SystemExit('N_VALUES debe ser una lista no vacía.')
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n < 1
        for n in N_VALUES
    ):
        raise SystemExit(
            'Todos los elementos de N_VALUES deben ser enteros positivos.'
        )
    if N_VALUES != sorted(set(N_VALUES)):
        raise SystemExit(
            'N_VALUES debe estar ordenada de menor a mayor y no tener duplicados.'
        )
    if N_VALUES[-1] > K:
        raise SystemExit('N_VALUES no puede contener valores mayores que K.')


def build_report(
    truth_by_id: dict[str, str | None],
    topics_by_id: dict[str, list[str]],
    k: int,
    n_values: list[int],
) -> dict[str, Any]:
    truth_ids = set(truth_by_id)
    result_ids = set(topics_by_id)
    missing_ids = truth_ids - result_ids
    extra_ids = result_ids - truth_ids
    if missing_ids or extra_ids:
        raise SystemExit(
            'Los conjuntos de _id no coinciden: '
            f'{len(missing_ids)} sin reranking y '
            f'{len(extra_ids)} sin etiqueta.'
        )

    evaluable_results: list[dict[str, Any]] = []
    for news_id, correct_topic in truth_by_id.items():
        if correct_topic is None:
            continue

        candidates = topics_by_id[news_id]
        try:
            reranked_position = candidates.index(correct_topic) + 1
        except ValueError:
            reranked_position = None

        evaluable_results.append(
            {
                '_id': news_id,
                'topic_correcto': correct_topic,
                f'posicion_rerank_k_{k}': reranked_position,
            }
        )

    evaluable_count = len(evaluable_results)
    position_field = f'posicion_rerank_k_{k}'
    recall_by_n = []
    for n in n_values:
        hits = sum(
            result[position_field] is not None
            and result[position_field] <= n
            for result in evaluable_results
        )
        recall = hits / evaluable_count if evaluable_count else 0.0
        recall_by_n.append(
            {
                'n': n,
                'hits': hits,
                'misses': evaluable_count - hits,
                'evaluable_news': evaluable_count,
                'recall': round(recall, 6),
                'recall_percent': round(recall * 100, 2),
            }
        )

    found_in_k = sum(
        result[position_field] is not None for result in evaluable_results
    )
    return {
        'configuration': {
            'k': k,
            'n_values': n_values,
            'recall_definition': (
                'Evaluable news whose correct topic appears in the first N '
                'reranked candidates divided by all evaluable news.'
            ),
        },
        'dataset': {
            'total_news': len(truth_by_id),
            'evaluable_news': evaluable_count,
            'non_evaluable_news': len(truth_by_id) - evaluable_count,
            'correct_topic_in_reranked_k': found_in_k,
            'correct_topic_outside_reranked_k': evaluable_count - found_in_k,
        },
        'recall_by_n': recall_by_n,
        'evaluable_news': evaluable_results,
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)
        file_handle.write('\n')


def main() -> None:
    validate_configuration()
    truth_by_id = load_ground_truth(load_json(resolve_path(POSITIONS_FILE)))
    topics_by_id = load_reranked_topics(
        load_json(resolve_path(RERANK_RESULTS_FILE)),
        K,
    )
    report = build_report(truth_by_id, topics_by_id, K, N_VALUES)
    output_path = resolve_path(OUTPUT_FILE)
    save_json(output_path, report)

    print(f'Resultados guardados: {output_path}')
    print(f'K seleccionado: {K}')
    for result in report['recall_by_n']:
        n = result['n']
        recall_percent = result['recall_percent']
        hits = result['hits']
        evaluable_news = result['evaluable_news']
        print(f'Recall@{n}: {recall_percent:.2f}% ({hits}/{evaluable_news})')


if __name__ == '__main__':
    main()
