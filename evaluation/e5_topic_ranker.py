#!/usr/bin/env python3
'''Utilidades compartidas para clasificar noticias directamente con E5.'''

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL = 'intfloat/multilingual-e5-base'
DEFAULT_BATCH_SIZE = 32
E5_PREFIX = 'query: '
TOPIC_TEMPLATE = 'Tema: {topic}. Definición: {definition}'


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


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def required_string(record: dict[str, Any], field: str, location: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f'{location}: falta un campo {field} no vacío.')
    return value


def load_topics(path: Path) -> dict[str, str]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise SystemExit(f'{path} debe contener un objeto JSON no vacío.')
    for topic, definition in payload.items():
        if (
            not isinstance(topic, str)
            or not topic.strip()
            or not isinstance(definition, str)
            or not definition.strip()
        ):
            raise SystemExit(
                'Todos los temas y sus definiciones deben ser cadenas no vacías.'
            )
    return payload


def _load_json_records(path: Path, allow_single: bool) -> list[Any]:
    payload = load_json(path)
    if isinstance(payload, list):
        return payload
    if allow_single and isinstance(payload, dict):
        return [payload]
    expected = 'un objeto o una lista' if allow_single else 'una lista'
    raise SystemExit(f'{path} debe contener {expected} de objetos JSON.')


def load_test_records(data_dir: Path) -> list[dict[str, str]]:
    if not data_dir.is_dir():
        raise SystemExit(f'No se encontró el directorio: {data_dir}')
    paths = sorted(data_dir.glob('*.json'))
    if not paths:
        raise SystemExit(f'No se encontraron archivos JSON en {data_dir}.')

    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for path in paths:
        for index, record in enumerate(_load_json_records(path, True)):
            location = f'{path}, elemento {index}'
            if not isinstance(record, dict):
                raise SystemExit(f'{location}: se esperaba un objeto JSON.')
            news_id = required_string(record, '_id', location)
            if news_id in seen_ids:
                raise SystemExit(f'{location}: _id duplicado {news_id!r}.')
            seen_ids.add(news_id)
            text = required_string(record, 'text', location)
            records.append({'_id': news_id, 'file': path.name, 'text': text})
    return records


def normalize_topic_labels(
    value: Any, location: str, topic_catalog: set[str]
) -> list[str]:
    if isinstance(value, str):
        labels = [value]
    elif isinstance(value, list):
        labels = value
    else:
        raise SystemExit(f'{location}: tema debe ser una cadena o una lista.')
    if not labels:
        raise SystemExit(f'{location}: tema no puede ser una lista vacía.')
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise SystemExit(f'{location}: tema contiene una etiqueta no válida.')
    normalized = list(dict.fromkeys(labels))
    unknown = [label for label in normalized if label not in topic_catalog]
    if unknown:
        raise SystemExit(f'{location}: temas inexistentes: {unknown}.')
    return normalized


def load_labelled_test(
    data_dir: Path, topic_catalog: set[str]
) -> dict[str, dict[str, Any]]:
    if not data_dir.is_dir():
        raise SystemExit(f'No se encontró el directorio: {data_dir}')
    paths = sorted(data_dir.glob('*.json'))
    if not paths:
        raise SystemExit(f'No se encontraron archivos JSON en {data_dir}.')

    labelled: dict[str, dict[str, Any]] = {}
    for path in paths:
        for index, record in enumerate(_load_json_records(path, False)):
            location = f'{path}, elemento {index}'
            if not isinstance(record, dict):
                raise SystemExit(f'{location}: se esperaba un objeto JSON.')
            news_id = required_string(record, '_id', location)
            if news_id in labelled:
                raise SystemExit(f'{location}: _id duplicado {news_id!r}.')
            labelled[news_id] = {
                'text': required_string(record, 'text', location),
                'topics': normalize_topic_labels(
                    record.get('tema'), location, topic_catalog
                ),
            }
    return labelled


def validate_dataset(
    records: list[dict[str, str]],
    labelled_by_id: dict[str, dict[str, Any]],
) -> None:
    records_by_id = {record['_id']: record for record in records}
    input_ids = set(records_by_id)
    labelled_ids = set(labelled_by_id)
    if input_ids != labelled_ids:
        missing = labelled_ids - input_ids
        unexpected = input_ids - labelled_ids
        raise SystemExit(
            'Los IDs de data_test no coinciden con data_test_labelled: '
            f'{len(missing)} ausentes y {len(unexpected)} inesperados.'
        )
    for news_id, labelled in labelled_by_id.items():
        if records_by_id[news_id]['text'] != labelled['text']:
            raise SystemExit(
                'El texto de data_test no coincide con data_test_labelled '
                f'para {news_id!r}.'
            )


def topic_sentences(
    topics: dict[str, str], model_name: str
) -> list[str]:
    sentences = [
        TOPIC_TEMPLATE.format(topic=topic, definition=definition)
        for topic, definition in topics.items()
    ]
    if 'e5' in model_name.lower():
        return [E5_PREFIX + sentence for sentence in sentences]
    return sentences


def news_sentences(
    records: list[dict[str, str]], model_name: str
) -> list[str]:
    texts = [record['text'] for record in records]
    if 'e5' in model_name.lower():
        return [E5_PREFIX + text for text in texts]
    return texts


def rankings_from_similarity_rows(
    records: list[dict[str, str]],
    topics: dict[str, str],
    similarity_rows: Any,
) -> list[dict[str, Any]]:
    rows = list(similarity_rows)
    if len(rows) != len(records):
        raise RuntimeError(
            'E5 devolvió un número de filas distinto al número de noticias.'
        )
    topic_names = list(topics)
    rankings: list[dict[str, Any]] = []
    for record, raw_row in zip(records, rows):
        row = [float(score) for score in raw_row]
        if len(row) != len(topic_names):
            raise RuntimeError(
                'E5 devolvió un número de scores distinto al número de temas.'
            )
        if not all(math.isfinite(score) for score in row):
            raise RuntimeError('E5 devolvió uno o más scores no finitos.')
        indices = sorted(range(len(topic_names)), key=lambda index: (-row[index], index))
        rankings.append(
            {
                '_id': record['_id'],
                'file': record['file'],
                'text': record['text'],
                'candidates': [
                    {
                        'tema': topic_names[index],
                        'similarity': round(row[index], 6),
                    }
                    for index in indices
                ],
            }
        )
    return rankings


def compute_rankings(
    records: list[dict[str, str]],
    topics: dict[str, str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if batch_size < 1:
        raise SystemExit('--batch-size debe ser mayor que cero.')
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            'Faltan dependencias. Instálalas con: '
            'pip install torch sentence-transformers'
        ) from exc

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    synchronize = torch.cuda.synchronize if device == 'cuda' else None

    load_start = time.perf_counter()
    model = SentenceTransformer(model_name, device=device)
    if synchronize is not None:
        synchronize()
    model_load_seconds = time.perf_counter() - load_start

    topic_start = time.perf_counter()
    topic_embeddings = model.encode(
        topic_sentences(topics, model_name),
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if synchronize is not None:
        synchronize()
    topic_embedding_seconds = time.perf_counter() - topic_start

    news_start = time.perf_counter()
    news_embeddings = model.encode(
        news_sentences(records, model_name),
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    if synchronize is not None:
        synchronize()
    news_embedding_seconds = time.perf_counter() - news_start

    ranking_start = time.perf_counter()
    similarities = news_embeddings @ topic_embeddings.T
    similarity_rows = similarities.detach().cpu().tolist()
    rankings = rankings_from_similarity_rows(records, topics, similarity_rows)
    similarity_and_ranking_seconds = time.perf_counter() - ranking_start

    total_seconds = (
        model_load_seconds
        + topic_embedding_seconds
        + news_embedding_seconds
        + similarity_and_ranking_seconds
    )
    per_news_seconds = (
        (news_embedding_seconds + similarity_and_ranking_seconds) / len(records)
        if records
        else 0.0
    )
    timing = {
        'total_e5_execution_time_seconds': round(total_seconds, 6),
        'model_load_seconds': round(model_load_seconds, 6),
        'topic_embedding_seconds': round(topic_embedding_seconds, 6),
        'news_embedding_seconds': round(news_embedding_seconds, 6),
        'similarity_and_ranking_seconds': round(
            similarity_and_ranking_seconds, 6
        ),
        'mean_news_embedding_and_ranking_seconds': round(per_news_seconds, 9),
        'scope': (
            'E5 se carga en memoria, codifica una vez el catálogo completo, '
            'codifica todas las noticias de test y calcula similitud coseno.'
        ),
    }
    return rankings, {'device': device, 'timing': timing}


def ranking_sha256(rankings: list[dict[str, Any]]) -> str:
    payload = [
        {'_id': ranking['_id'], 'candidates': ranking['candidates']}
        for ranking in rankings
    ]
    return sha256_json(payload)
