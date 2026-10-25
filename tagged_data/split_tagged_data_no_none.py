#!/usr/bin/env python3
"""Extrae particiones test y validation reproducibles del dataset etiquetado."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR / 'tagged_data_no_none.json'
DEFAULT_TEST_OUTPUT = SCRIPT_DIR / 'test_tagged_data_no_none.json'
DEFAULT_VALIDATION_OUTPUT = (
    SCRIPT_DIR / 'validation_tagged_data_no_none.json'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Selecciona aleatoriamente dos particiones disjuntas y conserva '
            'el orden original de los registros dentro de cada salida.'
        )
    )
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--test-output', type=Path, default=DEFAULT_TEST_OUTPUT)
    parser.add_argument(
        '--validation-output',
        type=Path,
        default=DEFAULT_VALIDATION_OUTPUT,
    )
    parser.add_argument('--test-size', type=int, default=600)
    parser.add_argument('--validation-size', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open('r', encoding='utf-8') as file_handle:
            records = json.load(file_handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f'JSON no válido en {path}: {exc}') from exc
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise SystemExit(f'{path} debe contener una lista de objetos JSON.')
    identifiers = [record.get('_id') for record in records]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise SystemExit('Todos los registros deben tener un _id no vacío.')
    if len(set(identifiers)) != len(identifiers):
        raise SystemExit('El dataset de origen contiene valores _id duplicados.')
    return records


def save_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as file_handle:
            json.dump(value, file_handle, ensure_ascii=False, indent=2)
            file_handle.write('\n')
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    args = parse_args()
    if args.test_size < 1 or args.validation_size < 1:
        raise SystemExit('Los tamaños de test y validation deben ser positivos.')
    records = load_records(args.source)
    selected_size = args.test_size + args.validation_size
    if selected_size > len(records):
        raise SystemExit(
            f'Se solicitan {selected_size} registros, pero solo existen '
            f'{len(records)}.'
        )

    shuffled_indices = list(range(len(records)))
    random.Random(args.seed).shuffle(shuffled_indices)
    test_indices = set(shuffled_indices[: args.test_size])
    validation_indices = set(shuffled_indices[args.test_size : selected_size])

    test_records = [
        record for index, record in enumerate(records) if index in test_indices
    ]
    validation_records = [
        record
        for index, record in enumerate(records)
        if index in validation_indices
    ]
    save_json_atomic(args.test_output, test_records)
    save_json_atomic(args.validation_output, validation_records)

    print(f'Origen: {len(records)} registros')
    print(f'Test: {len(test_records)} -> {args.test_output.resolve()}')
    print(
        f'Validation: {len(validation_records)} -> '
        f'{args.validation_output.resolve()}'
    )
    print(f'No seleccionados: {len(records) - selected_size}')
    print(f'Semilla: {args.seed}')


if __name__ == '__main__':
    main()
