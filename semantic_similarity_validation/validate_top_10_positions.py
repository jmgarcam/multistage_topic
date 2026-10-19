#!/usr/bin/env python3
'''Comprueba la posición de los temas correctos en el ranking semántico.'''

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


NONE_TOPIC = 'ninguno'


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    parser = argparse.ArgumentParser(
        description=(
            'Localiza las etiquetas correctas de cada noticia en el ranking '
            'completo de temas.'
        )
    )
    parser.add_argument(
        '--validation-dir',
        type=Path,
        default=project_dir / 'data_validation',
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=project_dir / 'semantic_similarity_results',
    )
    parser.add_argument(
        '--topics',
        type=Path,
        default=project_dir / 'temas.json',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help=(
            'Archivo de salida. Por defecto se usa top_<numero>_positions.json.'
        ),
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


def load_json_lists(directory: Path) -> list[tuple[Path, list[Any]]]:
    if not directory.is_dir():
        raise SystemExit(f'No se encontró el directorio: {directory}')
    paths = sorted(directory.glob('*.json'))
    if not paths:
        raise SystemExit(f'No se encontraron archivos JSON en {directory}.')

    payloads = []
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, list):
            raise SystemExit(f'{path} debe contener una lista JSON.')
        payloads.append((path, payload))
    return payloads


def required_string(record: dict[str, Any], field: str, location: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f'{location}: falta un campo {field} no vacío.')
    return value


def required_topics(value: Any, location: str) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list) and value:
        values = value
    else:
        raise SystemExit(
            f'{location}: tema debe ser una cadena o una lista no vacía.'
        )

    topics: list[str] = []
    for index, topic in enumerate(values):
        if not isinstance(topic, str) or not topic.strip():
            raise SystemExit(
                f'{location}, tema {index}: se esperaba una cadena no vacía.'
            )
        topic = topic.strip()
        if topic in topics:
            raise SystemExit(f'{location}: tema duplicado {topic!r}.')
        topics.append(topic)
    if NONE_TOPIC in topics and len(topics) != 1:
        raise SystemExit(
            f'{location}: {NONE_TOPIC!r} no puede combinarse con otros temas.'
        )
    return topics


def load_topic_catalog(path: Path) -> set[str]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise SystemExit(f'{path} debe contener un objeto JSON no vacío.')
    if any(not isinstance(topic, str) or not topic.strip() for topic in payload):
        raise SystemExit(f'{path} contiene nombres de tema no válidos.')
    return set(payload)


def load_validation_records(directory: Path) -> list[dict[str, Any]]:
    records = []
    seen_ids: dict[str, str] = {}
    for path, payload in load_json_lists(directory):
        for index, record in enumerate(payload):
            location = f'{path}, elemento {index}'
            if not isinstance(record, dict):
                raise SystemExit(f'{location}: se esperaba un objeto JSON.')
            news_id = required_string(record, '_id', location)
            if news_id in seen_ids:
                raise SystemExit(
                    f'{location}: _id duplicado; ya aparece en {seen_ids[news_id]}.'
                )
            seen_ids[news_id] = location
            records.append(
                {
                    '_id': news_id,
                    'topics': required_topics(record.get('tema'), location),
                    'text': required_string(record, 'text', location),
                    'source': required_string(record, 'account', location),
                    'validation_file': path.name,
                }
            )
    return records


def normalize_top_k(
    value: Any, location: str, topic_catalog: set[str]
) -> list[dict[str, Any]]:
    number_of_topics = len(topic_catalog)
    if not isinstance(value, list) or len(value) != number_of_topics:
        raise SystemExit(
            f'{location}: top_{number_of_topics} debe contener exactamente '
            f'{number_of_topics} temas.'
        )

    normalized = []
    seen_topics: set[str] = set()
    for index, candidate in enumerate(value):
        item_location = f'{location}, top_{number_of_topics}[{index}]'
        if not isinstance(candidate, dict):
            raise SystemExit(f'{item_location}: se esperaba un objeto JSON.')
        topic = required_string(candidate, 'tema', item_location)
        similarity = candidate.get('similarity')
        if (
            isinstance(similarity, bool)
            or not isinstance(similarity, (int, float))
            or not math.isfinite(float(similarity))
        ):
            raise SystemExit(
                f'{item_location}: similarity debe ser un número finito.'
            )
        if topic in seen_topics:
            raise SystemExit(f'{item_location}: tema duplicado {topic!r}.')
        if topic not in topic_catalog:
            raise SystemExit(
                f'{item_location}: {topic!r} no existe en temas.json.'
            )
        seen_topics.add(topic)
        normalized.append({'tema': topic, 'similarity': float(similarity)})
    if seen_topics != topic_catalog:
        raise SystemExit(
            f'{location}: el ranking no contiene exactamente el catálogo.'
        )
    return normalized


def load_similarity_results(
    directory: Path, topic_catalog: set[str]
) -> dict[str, dict[str, Any]]:
    records_by_id = {}
    seen_locations: dict[str, str] = {}
    top_key = f'top_{len(topic_catalog)}'
    for path, payload in load_json_lists(directory):
        for index, record in enumerate(payload):
            location = f'{path}, elemento {index}'
            if not isinstance(record, dict):
                raise SystemExit(f'{location}: se esperaba un objeto JSON.')
            news_id = required_string(record, '_id', location)
            if news_id in records_by_id:
                raise SystemExit(
                    f'{location}: _id duplicado; ya aparece en '
                    f'{seen_locations[news_id]}.'
                )
            seen_locations[news_id] = location
            records_by_id[news_id] = {
                'result_file': required_string(record, 'file', location),
                'top_k': normalize_top_k(
                    record.get(top_key), location, topic_catalog
                ),
            }
    return records_by_id


def summarize(
    records: list[dict[str, Any]], number_of_topics: int
) -> dict[str, Any]:
    total = len(records)
    evaluable = [record for record in records if record['evaluable']]
    found = [record for record in records if record['top_k_position'] is not None]
    position_counts = {
        str(position): sum(
            record['top_k_position'] == position for record in records
        )
        for position in range(1, number_of_topics + 1)
    }
    top_name = f'top_{number_of_topics}'

    return {
        'total_noticias': total,
        'noticias_evaluables': len(evaluable),
        'noticias_no_evaluables': total - len(evaluable),
        f'topic_correcto_en_{top_name}': len(found),
        f'topic_correcto_fuera_del_{top_name}': len(evaluable) - len(found),
        f'porcentaje_acierto_{top_name}': (
            round(len(found) * 100 / len(evaluable), 2) if evaluable else None
        ),
        'cantidad_por_posicion': position_counts,
    }


def build_report(
    validation_records: list[dict[str, Any]],
    similarity_by_id: dict[str, dict[str, Any]],
    topic_catalog: set[str],
) -> dict[str, Any]:
    number_of_topics = len(topic_catalog)
    validation_ids = {record['_id'] for record in validation_records}
    result_ids = set(similarity_by_id)
    missing_ids = validation_ids - result_ids
    extra_ids = result_ids - validation_ids
    if missing_ids or extra_ids:
        raise SystemExit(
            'Los conjuntos de _id no coinciden: '
            f'{len(missing_ids)} sin resultado y '
            f'{len(extra_ids)} sin etiqueta de validación.'
        )

    results = []
    summary_records = []
    for tagged_record in validation_records:
        ranked_record = similarity_by_id[tagged_record['_id']]
        top_k = ranked_record['top_k']
        correct_topics = tagged_record['topics']
        evaluable_topics = [
            topic for topic in correct_topics if topic in topic_catalog
        ]
        positions = [
            index
            for index, candidate in enumerate(top_k, start=1)
            if candidate['tema'] in evaluable_topics
        ]
        position = min(positions) if positions else None
        evaluable = bool(evaluable_topics)

        results.append(
            {
                '_id': tagged_record['_id'],
                'fuente': tagged_record['source'],
                'temas_correctos': correct_topics,
                f'posicion_top_{number_of_topics}': position,
            }
        )
        summary_records.append(
            {
                'evaluable': evaluable,
                'top_k_position': position,
            }
        )

    return {
        'resumen': summarize(summary_records, number_of_topics),
        'noticias': results,
    }


def main() -> None:
    args = parse_args()
    topic_catalog = load_topic_catalog(args.topics)
    number_of_topics = len(topic_catalog)
    validation_records = load_validation_records(args.validation_dir)
    similarity_by_id = load_similarity_results(
        args.results_dir, topic_catalog
    )
    report = build_report(validation_records, similarity_by_id, topic_catalog)

    output_path = args.output
    if output_path is None:
        output_path = (
            Path(__file__).resolve().parent
            / f'top_{number_of_topics}_positions.json'
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file_handle:
        json.dump(report, file_handle, ensure_ascii=False, indent=2)
        file_handle.write('\n')

    summary = report['resumen']
    top_accuracy = summary[f'porcentaje_acierto_top_{number_of_topics}']
    total_news = summary['total_noticias']
    not_evaluable = summary['noticias_no_evaluables']
    print(f'Informe guardado: {output_path}')
    print(f'Noticias validadas: {total_news}')
    if top_accuracy is not None:
        print(
            f'Top-{number_of_topics} accuracy (etiquetas evaluables): '
            f'{top_accuracy:.2f}%'
        )
    print(
        'Noticias no evaluables: '
        f'{not_evaluable}'
    )


if __name__ == '__main__':
    main()
