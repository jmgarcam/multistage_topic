#!/usr/bin/env python3
'''Utilidades compartidas para clasificar noticias directamente con E5.'''

from __future__ import annotations

import hashlib
import json
import math
import time
from numbers import Real
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


def load_validated_rankings(
    path: Path,
    records: list[dict[str, str]],
    labelled_by_id: dict[str, dict[str, Any]],
    topics: dict[str, str],
    model_name: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Carga rankings completos solo si pertenecen al experimento actual."""
    report = load_json(path)
    if not isinstance(report, dict) or report.get('status') != 'complete':
        raise SystemExit(f'{path} no contiene un informe E5 completo.')
    configuration = report.get('configuration')
    if not isinstance(configuration, dict):
        raise SystemExit(f'{path} no contiene una configuración válida.')

    normalized_labels = [
        {
            '_id': record['_id'],
            'text': labelled_by_id[record['_id']]['text'],
            'topics': labelled_by_id[record['_id']]['topics'],
        }
        for record in records
    ]
    expected_configuration = {
        'model': model_name,
        'architecture': 'bi-encoder',
        'uses_bge': False,
        'uses_llama': False,
        'candidate_count': len(topics),
        'batch_size': batch_size,
        'normalize_embeddings': True,
        'similarity': 'cosine',
        'news_prefix': E5_PREFIX.strip(),
        'topic_prefix': E5_PREFIX.strip(),
        'topic_representation': TOPIC_TEMPLATE,
        'input_data_sha256': sha256_json(records),
        'topic_catalog_sha256': sha256_json(topics),
        'labelled_test_sha256': sha256_json(normalized_labels),
    }
    mismatches = [
        key
        for key, expected_value in expected_configuration.items()
        if configuration.get(key) != expected_value
    ]
    if mismatches:
        raise SystemExit(
            f'{path} no corresponde a los datos o parámetros actuales '
            f'(campos distintos: {", ".join(mismatches)}).'
        )

    device = configuration.get('device')
    if not isinstance(device, str) or not device:
        raise SystemExit(f'{path}: dispositivo E5 inválido.')
    timing = report.get('timing')
    if not isinstance(timing, dict):
        raise SystemExit(f'{path}: información de tiempos inválida.')
    saved_predictions = report.get('predictions')
    if not isinstance(saved_predictions, list):
        raise SystemExit(f'{path} no contiene una lista predictions.')
    if len(saved_predictions) != len(records):
        raise SystemExit(
            f'{path} contiene {len(saved_predictions)} predicciones; '
            f'se esperaban {len(records)}.'
        )

    by_id: dict[str, dict[str, Any]] = {}
    for index, prediction in enumerate(saved_predictions):
        location = f'{path}, predicción {index}'
        if not isinstance(prediction, dict):
            raise SystemExit(f'{location}: se esperaba un objeto JSON.')
        news_id = prediction.get('_id')
        if not isinstance(news_id, str) or not news_id or news_id in by_id:
            raise SystemExit(f'{location}: _id ausente, inválido o duplicado.')
        by_id[news_id] = prediction
    if set(by_id) != {record['_id'] for record in records}:
        raise SystemExit(
            f'{path}: las predicciones no cubren exactamente data_test.'
        )

    topic_names = set(topics)
    rankings: list[dict[str, Any]] = []
    for record in records:
        prediction = by_id[record['_id']]
        location = f'{path}, predicción {record["_id"]!r}'
        if prediction.get('file') != record['file']:
            raise SystemExit(f'{location}: el archivo de origen no coincide.')
        correct_topics = list(labelled_by_id[record['_id']]['topics'])
        if prediction.get('topics_correctos') != correct_topics:
            raise SystemExit(f'{location}: las etiquetas no coinciden.')

        saved_candidates = prediction.get('topic_scores')
        if not isinstance(saved_candidates, list) or len(
            saved_candidates
        ) != len(topics):
            raise SystemExit(f'{location}: ranking E5 incompleto.')
        candidates: list[dict[str, Any]] = []
        previous_similarity = math.inf
        for candidate_index, candidate in enumerate(saved_candidates):
            candidate_location = f'{location}, candidato {candidate_index}'
            if not isinstance(candidate, dict):
                raise SystemExit(
                    f'{candidate_location}: se esperaba un objeto JSON.'
                )
            topic = candidate.get('tema')
            similarity = candidate.get('similarity')
            if not isinstance(topic, str) or topic not in topic_names:
                raise SystemExit(f'{candidate_location}: tema inválido.')
            if (
                isinstance(similarity, bool)
                or not isinstance(similarity, Real)
                or not math.isfinite(float(similarity))
            ):
                raise SystemExit(
                    f'{candidate_location}: similitud inválida.'
                )
            numeric_similarity = float(similarity)
            if numeric_similarity > previous_similarity:
                raise SystemExit(
                    f'{location}: el ranking no está ordenado por similitud.'
                )
            previous_similarity = numeric_similarity
            candidates.append(
                {'tema': topic, 'similarity': numeric_similarity}
            )
        if {candidate['tema'] for candidate in candidates} != topic_names:
            raise SystemExit(
                f'{location}: el ranking no contiene exactamente el catálogo.'
            )
        rankings.append(
            {
                '_id': record['_id'],
                'file': record['file'],
                'text': record['text'],
                'candidates': candidates,
            }
        )

    saved_ranking_hash = configuration.get('e5_ranking_sha256')
    calculated_ranking_hash = ranking_sha256(rankings)
    if saved_ranking_hash != calculated_ranking_hash:
        raise SystemExit(f'{path}: el hash del ranking E5 no coincide.')
    return rankings, {'device': device, 'timing': dict(timing)}


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
