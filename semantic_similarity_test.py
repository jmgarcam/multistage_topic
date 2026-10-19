#!/usr/bin/env python3
'''Clasifica textos por similitud semántica con las definiciones de temas.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer, util


DEFAULT_MODEL = 'intfloat/multilingual-e5-base'


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            'Ordena todos los temas por similitud con cada texto mediante '
            'embeddings y similitud coseno.'
        )
    )
    parser.add_argument(
        '--temas',
        type=Path,
        default=script_dir / 'temas.json',
        help='JSON que relaciona cada tema con su definición.',
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=script_dir / 'data_test',
        help='Directorio que contiene los JSON de entrada.',
    )
    parser.add_argument(
        '--output-dir',
        '--output',
        dest='output_dir',
        type=Path,
        default=script_dir / 'semantic_similarity_results',
        help=(
            'Directorio de salida. Se crea un JSON por cada archivo fuente '
            '(--output se mantiene como alias por compatibilidad).'
        ),
    )
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL,
        help='Modelo de sentence-transformers que se utilizará.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Número de textos codificados en cada lote.',
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


def load_topics(path: Path) -> tuple[list[str], list[str]]:
    topics = load_json(path)
    if not isinstance(topics, dict) or not topics:
        raise SystemExit(f'{path} debe contener un objeto JSON no vacío.')

    invalid_topics = [
        topic
        for topic, definition in topics.items()
        if not isinstance(topic, str)
        or not isinstance(definition, str)
        or not topic.strip()
        or not definition.strip()
    ]
    if invalid_topics:
        raise SystemExit(
            'Todos los temas y sus definiciones deben ser cadenas no vacías.'
        )

    topic_names = list(topics)
    # Incluir el nombre mejora la representación cuando aporta términos que no
    # aparecen literalmente en la definición.
    topic_sentences = [
        f'Tema: {topic}. Definición: {topics[topic]}' for topic in topic_names
    ]
    return topic_names, topic_sentences


def load_texts(data_dir: Path) -> list[dict[str, str]]:
    if not data_dir.is_dir():
        raise SystemExit(f'No se encontró el directorio: {data_dir}')

    inputs: list[dict[str, str]] = []

    for json_path in sorted(data_dir.glob('*.json')):
        payload = load_json(json_path)
        if isinstance(payload, dict):
            records = [payload]
        elif isinstance(payload, list):
            records = payload
        else:
            raise SystemExit(
                f'{json_path} debe contener un objeto o una lista de objetos JSON.'
            )

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise SystemExit(
                    f'{json_path}, elemento {index}: se esperaba un objeto JSON.'
                )
            news_id = record.get('_id')
            if not isinstance(news_id, str) or not news_id.strip():
                raise SystemExit(
                    f'{json_path}, elemento {index}: falta un campo _id no vacío.'
                )
            text = record.get('text')
            if not isinstance(text, str) or not text.strip():
                raise SystemExit(
                    f'{json_path}, elemento {index}: falta un campo text no vacío.'
                )
            inputs.append({'_id': news_id, 'file': json_path.name, 'text': text})

    if not inputs:
        raise SystemExit(f'No se encontraron textos en archivos JSON de {data_dir}.')
    return inputs


def save_results_by_source(
    results_by_source: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_file, source_results in results_by_source.items():
        output_path = output_dir / source_file
        with output_path.open('w', encoding='utf-8') as file_handle:
            json.dump(source_results, file_handle, ensure_ascii=False, indent=2)
            file_handle.write('\n')

        print(
            f'Resultados guardados: {output_path} '
            f'({len(source_results)} textos)'
        )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit('--batch-size debe ser mayor que cero.')

    topic_names, topic_sentences = load_topics(args.temas)
    number_of_topics = len(topic_names)

    if args.data_dir.resolve() == args.output_dir.resolve():
        raise SystemExit(
            'El directorio de salida debe ser diferente del directorio de entrada.'
        )

    inputs = load_texts(args.data_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Dispositivo: {device}')
    print(f'Modelo: {args.model}')

    model = SentenceTransformer(args.model, device=device)

    texts_to_encode = [entry['text'] for entry in inputs]
    if 'e5' in args.model.lower():
        # Para esta comparación simétrica, tanto las noticias como los temas
        # utilizan el prefijo query recomendado por E5.
        topic_sentences = ['query: ' + sentence for sentence in topic_sentences]
        texts_to_encode = ['query: ' + text for text in texts_to_encode]

    # Los embeddings de temas se calculan una sola vez para toda la ejecución.
    topic_embeddings = model.encode(
        topic_sentences,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    text_embeddings = model.encode(
        texts_to_encode,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    similarities = util.cos_sim(text_embeddings, topic_embeddings)
    top_scores, top_indices = torch.topk(
        similarities, k=number_of_topics, dim=1
    )

    results_by_source: dict[str, list[dict[str, Any]]] = {}
    for entry, scores, indices in zip(inputs, top_scores, top_indices):
        source_results = results_by_source.setdefault(entry['file'], [])
        source_results.append(
            {
                '_id': entry['_id'],
                'file': entry['file'],
                'text': entry['text'],
                f'top_{number_of_topics}': [
                    {
                        'tema': topic_names[index],
                        'similarity': round(float(score), 6),
                    }
                    for score, index in zip(scores.tolist(), indices.tolist())
                ],
            }
        )

    save_results_by_source(results_by_source, args.output_dir)


if __name__ == '__main__':
    main()
