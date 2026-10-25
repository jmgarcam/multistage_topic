#!/usr/bin/env python3
"""Migra los informes de evaluación del esquema específico de Llama a SLM."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Renombra etiquetas de metadatos llama_* a slm_* sin modificar '
            'los nombres reales de los modelos ni las salidas de Ollama.'
        )
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=SCRIPT_DIR,
        help='Directorio que contiene las carpetas results*.',
    )
    return parser.parse_args()


def migrate_key(key: str) -> str:
    if key.startswith('llama_'):
        return f'slm_{key[len("llama_"):]}'
    return key.replace('_llama_', '_slm_')


def migrate_string(key: str | None, value: str) -> str:
    if key == 'decision_source' and value == 'llama':
        return 'slm'
    if key == 'routing_strategy':
        return {
            'always_route_e5_top_n_to_llama': (
                'always_route_e5_top_n_to_slm'
            ),
            'preserve_e5_top_1_when_margin_gte_threshold_else_llama': (
                'preserve_e5_top_1_when_margin_gte_threshold_else_slm'
            ),
        }.get(value, value)
    if key == 'final_accuracy':
        return value.replace('E5 or Llama', 'E5 or the SLM')
    if key == 'conditional_selection_accuracy':
        return value.replace('Llama accuracy', 'SLM accuracy')
    if key == 'slm_final_accuracy':
        return value.replace('Llama decisions', 'SLM decisions')
    if key == 'scope':
        return value.replace('llamar a Llama', 'llamar al SLM')
    return value


def migrate(value: Any, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        migrated: dict[str, Any] = {}
        for key, item in value.items():
            new_key = migrate_key(key)
            if new_key in migrated:
                raise ValueError(
                    f'Colisión al migrar {key!r} a {new_key!r}.'
                )
            migrated[new_key] = migrate(item, new_key)
        return migrated
    if isinstance(value, list):
        return [migrate(item, parent_key) for item in value]
    if isinstance(value, str):
        return migrate_string(parent_key, value)
    return value


def save_json_atomic(path: Path, value: Any) -> None:
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


def result_files(root: Path) -> list[Path]:
    return sorted(
        path
        for results_dir in root.glob('results*')
        if results_dir.is_dir()
        for path in results_dir.rglob('*.json')
    )


def main() -> None:
    root = parse_args().root.resolve()
    paths = result_files(root)
    changed = 0
    for path in paths:
        with path.open('r', encoding='utf-8') as file_handle:
            original = json.load(file_handle)
        migrated = migrate(original)
        if migrated != original:
            save_json_atomic(path, migrated)
            changed += 1
    print(f'JSON examinados: {len(paths)}')
    print(f'JSON migrados: {changed}')


if __name__ == '__main__':
    main()
