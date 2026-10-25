#!/usr/bin/env python3
"""Selecciona una muestra aleatoria reproducible de los datos limpios."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_DIR / 'clean_data'
DEFAULT_OUTPUT_FILE = (
    PROJECT_DIR / 'candidate_tagged_data' / 'candidate_tagged_data.json'
)
DEFAULT_REPORT_FILE = (
    PROJECT_DIR / 'candidate_tagged_data' / 'selection_report.json'
)
DEFAULT_SAMPLE_SIZE = 2_000
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Selecciona registros al azar de los archivos clean_data.*.json '
            'sin modificar los datos limpios originales.'
        )
    )
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output-file', type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument('--report-file', type=Path, default=DEFAULT_REPORT_FILE)
    parser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def load_records(input_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    input_files = sorted(input_dir.glob('clean_data.*.json'))
    if not input_files:
        raise SystemExit(
            f'No se encontraron archivos clean_data.*.json en {input_dir}'
        )

    records: list[dict[str, Any]] = []
    for input_file in input_files:
        try:
            with input_file.open('r', encoding='utf-8') as file_handle:
                payload = json.load(file_handle)
        except json.JSONDecodeError as exc:
            raise SystemExit(f'JSON no válido en {input_file}: {exc}') from exc

        if not isinstance(payload, list):
            raise SystemExit(f'{input_file} debe contener una lista JSON.')
        for index, record in enumerate(payload):
            if not isinstance(record, dict):
                raise SystemExit(
                    f'{input_file}, elemento {index}: se esperaba un objeto.'
                )
            records.append(record)

    return records, input_files


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)
        file_handle.write('\n')
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise SystemExit('--sample-size debe ser mayor que cero.')

    records, input_files = load_records(args.input_dir)
    if args.sample_size > len(records):
        raise SystemExit(
            f'No se pueden seleccionar {args.sample_size} registros: '
            f'solo hay {len(records)} disponibles.'
        )

    selected = random.Random(args.seed).sample(records, args.sample_size)
    save_json(args.output_file, selected)

    account_counts = Counter(
        str(record.get('account', '<sin cuenta>')) for record in selected
    )
    report = {
        'seed': args.seed,
        'sample_size': args.sample_size,
        'available_records': len(records),
        'input_files': [str(path.resolve()) for path in input_files],
        'output_file': str(args.output_file.resolve()),
        'selected_by_account': dict(sorted(account_counts.items())),
    }
    save_json(args.report_file, report)

    print(f'Registros disponibles: {len(records)}')
    print(f'Registros seleccionados: {len(selected)}')
    print(f'Semilla aleatoria: {args.seed}')
    print(f'Muestra: {args.output_file.resolve()}')
    print(f'Informe: {args.report_file.resolve()}')


if __name__ == '__main__':
    main()
