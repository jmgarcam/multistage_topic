#!/usr/bin/env python3
'''Reordena temas candidatos con un cross-encoder multilingüe.'''

from __future__ import annotations

import json
import math
import time
from numbers import Real
from pathlib import Path
from typing import Any


INPUT_DIR = 'semantic_similarity_results'
TEMAS_JSON = 'temas.json'
OUTPUT_FILE = 'reranked_topic_results.json'
K_VALUES = [3, 5, 7, 10, 13, 15, 19]
MODEL_NAME = 'BAAI/bge-reranker-v2-m3'
BATCH_SIZE = 16
MAX_LENGTH = 512
DATASET_DIR = Path(__file__).resolve().parent.parent / 'dataset_mteb'


def resolve_path(configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return DATASET_DIR / path


def load_json(path: Path) -> Any:
    try:
        with path.open('r', encoding='utf-8') as file_handle:
            return json.load(file_handle)
    except FileNotFoundError as exc:
        raise SystemExit(f'No se encontró el archivo: {path}') from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f'JSON no válido en {path}: {exc}') from exc


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


def load_similarity_results(directory: Path) -> tuple[list[Any], int]:
    if not directory.is_dir():
        raise SystemExit(f'No se encontró el directorio de entrada: {directory}')

    input_paths = sorted(directory.glob('*.json'))
    if not input_paths:
        raise SystemExit(f'No se encontraron archivos JSON en {directory}.')

    records: list[Any] = []
    for input_path in input_paths:
        payload = load_json(input_path)
        if not isinstance(payload, list):
            raise SystemExit(f'{input_path} debe contener una lista JSON.')
        records.extend(payload)

    return records, len(input_paths)


def require_finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SystemExit(f'{location} debe ser un número.')
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise SystemExit(f'{location} debe ser un número finito.')
    return numeric_value


def prepare_records(
    payload: Any, topics: dict[str, str]
) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise SystemExit('El JSON de entrada debe contener una lista.')

    prepared_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for record_index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise SystemExit(
                f'Registro {record_index}: se esperaba un objeto JSON.'
            )

        news_id = record.get('_id')
        if not isinstance(news_id, str) or not news_id.strip():
            raise SystemExit(
                f'Registro {record_index}: falta un campo _id no vacío.'
            )
        if news_id in seen_ids:
            raise SystemExit(f'Registro {record_index}: _id duplicado {news_id}.')
        seen_ids.add(news_id)

        source_file = record.get('file')
        text = record.get('text')
        if not isinstance(source_file, str) or not source_file.strip():
            raise SystemExit(
                f'Registro {record_index}: falta un campo file no vacío.'
            )
        if not isinstance(text, str) or not text.strip():
            raise SystemExit(
                f'Registro {record_index}: falta un campo text no vacío.'
            )

        number_of_topics = len(topics)
        top_key = f'top_{number_of_topics}'
        candidates = record.get(top_key)
        if not isinstance(candidates, list) or len(candidates) != number_of_topics:
            raise SystemExit(
                f'Registro {record_index}: {top_key} debe contener exactamente '
                f'{number_of_topics} temas.'
            )

        prepared_candidates: list[dict[str, Any]] = []
        seen_candidates: set[str] = set()
        for candidate_index, candidate in enumerate(candidates):
            location = f'Registro {record_index}, candidato {candidate_index}'
            if not isinstance(candidate, dict):
                raise SystemExit(f'{location}: se esperaba un objeto JSON.')

            topic = candidate.get('tema')
            if not isinstance(topic, str) or not topic.strip():
                raise SystemExit(f'{location}: falta un campo tema no vacío.')
            if topic not in topics:
                raise SystemExit(f'{location}: el tema {topic!r} no existe en temas.json.')
            if topic in seen_candidates:
                raise SystemExit(f'{location}: tema duplicado {topic!r}.')
            seen_candidates.add(topic)

            embedding_similarity = require_finite_number(
                candidate.get('similarity'), f'{location}: similarity'
            )
            prepared_candidates.append(
                {
                    'tema': topic,
                    'embedding_similarity': embedding_similarity,
                    'definition': topics[topic],
                }
            )

        if seen_candidates != set(topics):
            raise SystemExit(
                f'Registro {record_index}: {top_key} no contiene exactamente '
                'los temas definidos en temas.json.'
            )

        prepared_records.append(
            {
                '_id': news_id,
                'file': source_file,
                'text': text,
                'candidates': prepared_candidates,
            }
        )

    return prepared_records


def rerank_records(
    records: list[dict[str, Any]],
    k_values: list[int],
    model_name: str,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    try:
        import torch
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise SystemExit(
            'Faltan dependencias. Instálalas con: pip install torch FlagEmbedding'
        ) from exc

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Dispositivo: {device}')
    print(f'Modelo: {model_name}')

    reranker = FlagReranker(
        model_name,
        devices=device,
        use_fp16=device.startswith('cuda'),
    )

    output_records: list[dict[str, Any]] = []
    for record in records:
        rerank_results: list[dict[str, Any]] = []
        for k in k_values:
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            candidates = record['candidates'][:k]
            pairs = [
                [record['text'], candidate['definition']]
                for candidate in candidates
            ]

            raw_scores = reranker.compute_score(
                pairs,
                batch_size=batch_size,
                max_length=max_length,
                normalize=True,
            )
            if device.startswith('cuda'):
                torch.cuda.synchronize()

            if isinstance(raw_scores, Real):
                raw_scores = [raw_scores]
            scores = [float(score) for score in raw_scores]
            if len(scores) != len(pairs):
                raise SystemExit(
                    'El reranker devolvió un número de scores distinto al '
                    f'número de pares para k={k}.'
                )
            if not all(math.isfinite(score) for score in scores):
                raise SystemExit(
                    f'El reranker devolvió uno o más scores no finitos para k={k}.'
                )

            reranked_topics = [
                {
                    'tema': candidate['tema'],
                    'embedding_similarity': candidate['embedding_similarity'],
                    'cross_encoder_score': round(score, 6),
                }
                for candidate, score in zip(candidates, scores)
            ]
            reranked_topics.sort(
                key=lambda candidate: candidate['cross_encoder_score'],
                reverse=True,
            )
            elapsed_seconds = time.perf_counter() - start_time
            rerank_results.append(
                {
                    'k': k,
                    'rerank_time_seconds': round(elapsed_seconds, 6),
                    'reranked_topics': reranked_topics,
                }
            )

        output_records.append(
            {
                '_id': record['_id'],
                'file': record['file'],
                'text': record['text'],
                'rerank_results': rerank_results,
            }
        )

    return output_records


def save_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file_handle:
        json.dump(results, file_handle, ensure_ascii=False, indent=2)
        file_handle.write('\n')


def select_k_values(
    configured_values: list[int], number_of_topics: int
) -> list[int]:
    k_values = [k for k in configured_values if k <= number_of_topics]
    if number_of_topics not in k_values:
        k_values.append(number_of_topics)
    return sorted(k_values)


def main() -> None:
    if BATCH_SIZE < 1:
        raise SystemExit('BATCH_SIZE debe ser mayor que cero.')
    if MAX_LENGTH < 1:
        raise SystemExit('MAX_LENGTH debe ser mayor que cero.')
    if not isinstance(K_VALUES, list) or not K_VALUES:
        raise SystemExit('K_VALUES debe ser una lista no vacía.')
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in K_VALUES):
        raise SystemExit('Todos los elementos de K_VALUES deben ser enteros positivos.')
    if len(set(K_VALUES)) != len(K_VALUES):
        raise SystemExit('Los elementos de K_VALUES no deben estar duplicados.')

    input_dir = resolve_path(INPUT_DIR)
    temas_path = resolve_path(TEMAS_JSON)
    output_path = resolve_path(OUTPUT_FILE)
    if output_path.resolve().parent == input_dir.resolve():
        raise SystemExit(
            'El archivo de salida debe estar fuera del directorio de entrada.'
        )

    topics = load_topics(temas_path)
    k_values = select_k_values(K_VALUES, len(topics))
    payload, input_file_count = load_similarity_results(input_dir)
    records = prepare_records(payload, topics)

    if records:
        output_records = rerank_records(
            records,
            k_values,
            MODEL_NAME,
            BATCH_SIZE,
            MAX_LENGTH,
        )
    else:
        output_records = []

    save_results(output_path, output_records)
    print(f'Resultados guardados: {output_path}')
    print(f'Archivos de entrada: {input_file_count}')
    print(f'Noticias reordenadas: {len(output_records)}')
    print(f'Valores K utilizados: {k_values}')


if __name__ == '__main__':
    main()
