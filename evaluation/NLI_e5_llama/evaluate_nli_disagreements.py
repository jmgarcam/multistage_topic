#!/usr/bin/env python3
'''Aplica NLI únicamente cuando las decisiones de E5 y Llama difieren.'''

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EVALUATION_DIR.parent
DATASET_DIR = PROJECT_DIR / 'dataset_mteb'
DEFAULT_SOURCE_RESULTS = (
    EVALUATION_DIR
    / 'E5_top_N_llama'
    / 'results_llama3_2_3b_temas_order_two_prompts_e5_margin_0_0025'
    / 'e5_top_3_llama_results.json'
)
RESULTS_DIR = SCRIPT_DIR / 'results_mdeberta_v3_base_xnli'
DEFAULT_OUTPUT = RESULTS_DIR / 'e5_top_3_llama_nli_results.json'
MODEL_NAME = (
    'MoritzLaurer/'
    'mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
)
BATCH_SIZE = 16
MAX_LENGTH = 512
NLI_LABELS = {'entailment', 'neutral', 'contradiction'}
PREMISE_TEMPLATE = (
    'Acontecimiento principal: {main_event}\n'
    'Frase de evidencia: {evidence_sentence}'
)
HYPOTHESIS_TEMPLATE = (
    'El acontecimiento principal de la noticia pertenece al tema '
    '«{topic}». Definición del tema: {definition}'
)


PairClassifier = Callable[[list[tuple[str, str]]], list[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Aplica un modelo NLI a las hipótesis de E5 y Llama solo en '
            'las noticias donde sus decisiones difieren. El informe guarda '
            'únicamente las etiquetas NLI, sin scores ni decisión derivada.'
        )
    )
    parser.add_argument(
        '--source-results',
        type=Path,
        default=DEFAULT_SOURCE_RESULTS,
        help='Informe completo de E5 Top-N + Llama que se analizará.',
    )
    parser.add_argument(
        '--topics',
        type=Path,
        default=DATASET_DIR / 'temas.json',
        help='Catálogo de temas y definiciones.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help='Ruta del informe JSON de salida.',
    )
    parser.add_argument(
        '--model',
        default=MODEL_NAME,
        help='Modelo multilingüe de inferencia textual.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help='Número de pares premisa/hipótesis por lote.',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=MAX_LENGTH,
        help='Longitud máxima tokenizada de cada par.',
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='Dispositivo de Transformers: auto, cpu, cuda o cuda:N.',
    )
    parser.add_argument(
        '--reuse-existing',
        action='store_true',
        help='Valida y reutiliza un informe completo ya existente.',
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


def required_string(value: Any, name: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f'{location}: {name} debe ser una cadena no vacía.')
    return value.strip()


def load_topics(path: Path) -> dict[str, str]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise SystemExit(f'{path} debe contener un objeto JSON no vacío.')
    topics: dict[str, str] = {}
    for raw_topic, raw_definition in payload.items():
        topic = required_string(raw_topic, 'tema', str(path))
        definition = required_string(
            raw_definition, f'definición de {topic!r}', str(path)
        )
        topics[topic] = definition
    return topics


def load_source_report(path: Path) -> dict[str, Any]:
    report = load_json(path)
    if not isinstance(report, dict):
        raise SystemExit(f'{path} debe contener un objeto JSON.')
    if report.get('status') != 'complete':
        raise SystemExit(f'{path} no contiene un informe completo.')
    predictions = report.get('predictions')
    if not isinstance(predictions, list) or not predictions:
        raise SystemExit(f'{path} no contiene predicciones.')
    return report


def build_premise(prediction: dict[str, Any], location: str) -> str:
    main_event = required_string(
        prediction.get('main_event'), 'main_event', location
    )
    evidence_sentence = required_string(
        prediction.get('evidence_sentence'), 'evidence_sentence', location
    )
    return PREMISE_TEMPLATE.format(
        main_event=main_event,
        evidence_sentence=evidence_sentence,
    )


def build_hypothesis(topic: str, topics: dict[str, str]) -> str:
    return HYPOTHESIS_TEMPLATE.format(
        topic=topic,
        definition=topics[topic],
    )


def select_disagreements(
    predictions: list[Any], topics: dict[str, str]
) -> list[dict[str, Any]]:
    disagreements: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_prediction in enumerate(predictions):
        location = f'predicción de origen {index}'
        if not isinstance(raw_prediction, dict):
            raise SystemExit(f'{location}: se esperaba un objeto JSON.')
        news_id = required_string(raw_prediction.get('_id'), '_id', location)
        if news_id in seen_ids:
            raise SystemExit(f'{location}: _id duplicado {news_id!r}.')
        seen_ids.add(news_id)

        e5_topic = required_string(
            raw_prediction.get('e5_top_1_topic'),
            'e5_top_1_topic',
            location,
        )
        llama_topic = required_string(
            raw_prediction.get('final_decision'),
            'final_decision',
            location,
        )
        for field_name, topic in (
            ('e5_top_1_topic', e5_topic),
            ('final_decision', llama_topic),
        ):
            if topic not in topics:
                raise SystemExit(
                    f'{location}: {field_name} contiene un tema desconocido '
                    f'{topic!r}.'
                )

        if e5_topic == llama_topic:
            continue
        if raw_prediction.get('llama_called') is not True:
            raise SystemExit(
                f'{location}: E5 y la decisión final difieren, pero '
                'llama_called no es true.'
            )

        premise = build_premise(raw_prediction, location)
        disagreements.append(
            {
                '_id': news_id,
                'file': raw_prediction.get('file'),
                'e5_topic': e5_topic,
                'llama_topic': llama_topic,
                'premise': premise,
                'e5_hypothesis': build_hypothesis(e5_topic, topics),
                'llama_hypothesis': build_hypothesis(llama_topic, topics),
            }
        )
    return disagreements


def normalize_nli_label(raw_label: Any) -> str:
    if not isinstance(raw_label, str):
        raise RuntimeError('El modelo NLI devolvió una etiqueta no textual.')
    normalized = raw_label.strip().lower().replace('-', '_')
    aliases = {
        'entailment': 'entailment',
        'entails': 'entailment',
        'neutral': 'neutral',
        'contradiction': 'contradiction',
        'contradicts': 'contradiction',
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise RuntimeError(
            f'Etiqueta NLI desconocida en model.config.id2label: {raw_label!r}.'
        ) from exc


def normalize_id2label(raw_mapping: Any) -> dict[int, str]:
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise RuntimeError('El modelo NLI no define config.id2label.')
    normalized: dict[int, str] = {}
    for raw_index, raw_label in raw_mapping.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f'Índice NLI no válido en config.id2label: {raw_index!r}.'
            ) from exc
        normalized[index] = normalize_nli_label(raw_label)
    if set(normalized.values()) != NLI_LABELS:
        raise RuntimeError(
            'config.id2label debe contener exactamente entailment, neutral '
            f'y contradiction; contiene {sorted(set(normalized.values()))}.'
        )
    return normalized


def resolve_device(requested_device: str, torch_module: Any) -> str:
    requested = requested_device.strip().lower()
    if requested == 'auto':
        return 'cuda:0' if torch_module.cuda.is_available() else 'cpu'
    if requested == 'cuda':
        requested = 'cuda:0'
    if requested.startswith('cuda') and not torch_module.cuda.is_available():
        raise SystemExit(
            f'Se solicitó {requested_device!r}, pero CUDA no está disponible.'
        )
    if requested != 'cpu' and not requested.startswith('cuda:'):
        raise SystemExit(
            '--device debe ser auto, cpu, cuda o un dispositivo cuda:N.'
        )
    return requested


def load_nli_classifier(
    model_name: str,
    batch_size: int,
    max_length: int,
    requested_device: str,
) -> tuple[PairClassifier, str, dict[int, str]]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            'Faltan dependencias. Instálalas con: pip install torch '
            'transformers sentencepiece'
        ) from exc

    device = resolve_device(requested_device, torch)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    id2label = normalize_id2label(model.config.id2label)
    model.to(device)
    model.eval()

    def classify_pairs(pairs: list[tuple[str, str]]) -> list[str]:
        labels: list[str] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            premises = [premise for premise, _ in batch]
            hypotheses = [hypothesis for _, hypothesis in batch]
            encoded = tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            encoded = {
                name: tensor.to(device) for name, tensor in encoded.items()
            }
            with torch.inference_mode():
                predicted_ids = (
                    model(**encoded).logits.argmax(dim=-1).detach().cpu()
                )
            labels.extend(id2label[int(index)] for index in predicted_ids)
        return labels

    return classify_pairs, device, id2label


def evaluate_disagreements(
    disagreements: list[dict[str, Any]],
    classify_pairs: PairClassifier,
) -> tuple[list[dict[str, Any]], float]:
    pairs: list[tuple[str, str]] = []
    for disagreement in disagreements:
        pairs.extend(
            [
                (
                    disagreement['premise'],
                    disagreement['e5_hypothesis'],
                ),
                (
                    disagreement['premise'],
                    disagreement['llama_hypothesis'],
                ),
            ]
        )

    start = time.perf_counter()
    labels = classify_pairs(pairs)
    elapsed = time.perf_counter() - start
    if len(labels) != len(pairs):
        raise RuntimeError(
            'El clasificador NLI devolvió un número de etiquetas distinto '
            f'al número de pares: {len(labels)} frente a {len(pairs)}.'
        )
    normalized_labels = [normalize_nli_label(label) for label in labels]

    results: list[dict[str, Any]] = []
    for index, disagreement in enumerate(disagreements):
        e5_label, llama_label = normalized_labels[index * 2 : index * 2 + 2]
        results.append(
            {
                **disagreement,
                'nli_results': {
                    'e5_hypothesis': e5_label,
                    'llama_hypothesis': llama_label,
                },
            }
        )
    return results, elapsed


def label_counts(results: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(result['nli_results'][key] for result in results)
    return {label: counts.get(label, 0) for label in sorted(NLI_LABELS)}


def build_report(
    *,
    source_report: dict[str, Any],
    source_path: Path,
    topics: dict[str, str],
    results: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
    id2label: dict[int, str],
    elapsed: float,
) -> dict[str, Any]:
    source_predictions = source_report['predictions']
    llama_called = sum(
        prediction.get('llama_called') is True
        for prediction in source_predictions
        if isinstance(prediction, dict)
    )
    return {
        'status': 'complete',
        'configuration': {
            'model': model_name,
            'architecture': 'multilingual natural language inference',
            'source_results': str(source_path.resolve()),
            'source_results_sha256': sha256_json(source_report),
            'source_n': source_report.get('configuration', {}).get('n'),
            'selection_rule': 'e5_top_1_topic != final_decision',
            'premise_source': 'llama_main_event_and_exact_evidence_sentence',
            'premise_template': PREMISE_TEMPLATE,
            'hypothesis_template': HYPOTHESIS_TEMPLATE,
            'batch_size': batch_size,
            'max_length': max_length,
            'device': device,
            'id2label': {str(index): label for index, label in id2label.items()},
            'stores_nli_scores': False,
            'computes_score_differences': False,
            'derives_final_decision': False,
            'uses_ground_truth': False,
            'topic_catalog_sha256': sha256_json(topics),
        },
        'dataset': {
            'source_predictions': len(source_predictions),
            'source_llama_called': llama_called,
            'e5_llama_disagreements': len(results),
            'nli_pairs_evaluated': len(results) * 2,
        },
        'nli_label_counts': {
            'e5_hypothesis': label_counts(results, 'e5_hypothesis'),
            'llama_hypothesis': label_counts(results, 'llama_hypothesis'),
        },
        'timing': {
            'total_nli_inference_time_seconds': round(elapsed, 6),
        },
        'results': results,
    }


def validate_existing_report(
    report: Any,
    source_report: dict[str, Any],
    source_path: Path,
    topics: dict[str, str],
    model_name: str,
    batch_size: int,
    max_length: int,
) -> None:
    if not isinstance(report, dict) or report.get('status') != 'complete':
        raise SystemExit('El informe NLI existente no está completo.')
    configuration = report.get('configuration')
    if not isinstance(configuration, dict):
        raise SystemExit('El informe NLI existente no tiene configuración.')
    expected = {
        'model': model_name,
        'source_results': str(source_path.resolve()),
        'source_results_sha256': sha256_json(source_report),
        'batch_size': batch_size,
        'max_length': max_length,
        'topic_catalog_sha256': sha256_json(topics),
        'stores_nli_scores': False,
        'computes_score_differences': False,
        'derives_final_decision': False,
        'uses_ground_truth': False,
    }
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise SystemExit(
                f'El informe NLI existente no coincide en {key!r}.'
            )

    disagreements = select_disagreements(source_report['predictions'], topics)
    results = report.get('results')
    if not isinstance(results, list) or len(results) != len(disagreements):
        raise SystemExit(
            'El informe NLI existente no contiene todos los desacuerdos.'
        )
    expected_ids = [item['_id'] for item in disagreements]
    result_ids = [
        result.get('_id') if isinstance(result, dict) else None
        for result in results
    ]
    if result_ids != expected_ids:
        raise SystemExit(
            'Los IDs del informe NLI existente no coinciden con el origen.'
        )
    forbidden_fields = {
        'score',
        'scores',
        'score_difference',
        'nli_decision',
        'final_decision',
    }
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise SystemExit(f'Resultado NLI {index}: se esperaba un objeto.')
        if forbidden_fields.intersection(result):
            raise SystemExit(
                f'Resultado NLI {index}: contiene un campo derivado prohibido.'
            )
        nli_results = result.get('nli_results')
        if not isinstance(nli_results, dict):
            raise SystemExit(
                f'Resultado NLI {index}: falta el objeto nli_results.'
            )
        for key in ('e5_hypothesis', 'llama_hypothesis'):
            if normalize_nli_label(nli_results.get(key)) not in NLI_LABELS:
                raise SystemExit(
                    f'Resultado NLI {index}: etiqueta inválida en {key}.'
                )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit('--batch-size debe ser mayor que cero.')
    if args.max_length < 1:
        raise SystemExit('--max-length debe ser mayor que cero.')

    topics = load_topics(args.topics)
    source_report = load_source_report(args.source_results)
    disagreements = select_disagreements(
        source_report['predictions'], topics
    )
    if not disagreements:
        raise SystemExit('No hay desacuerdos entre E5 y Llama que evaluar.')

    print(f'Informe de origen: {args.source_results.resolve()}')
    print(f'Desacuerdos E5/Llama: {len(disagreements)}')
    print(f'Pares NLI: {len(disagreements) * 2}')
    print(f'Modelo NLI: {args.model}')

    if args.reuse_existing and args.output.is_file():
        existing_report = load_json(args.output)
        validate_existing_report(
            existing_report,
            source_report,
            args.source_results,
            topics,
            args.model,
            args.batch_size,
            args.max_length,
        )
        print(f'Informe NLI completo y validado: {args.output.resolve()}')
        return

    classify_pairs, device, id2label = load_nli_classifier(
        args.model,
        args.batch_size,
        args.max_length,
        args.device,
    )
    print(f'Dispositivo: {device}')
    results, elapsed = evaluate_disagreements(
        disagreements, classify_pairs
    )
    report = build_report(
        source_report=source_report,
        source_path=args.source_results,
        topics=topics,
        results=results,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        id2label=id2label,
        elapsed=elapsed,
    )
    save_json(args.output, report)

    print(f'Resultados NLI guardados: {args.output.resolve()}')
    print('No se han calculado scores, diferencias ni una decisión final.')


if __name__ == '__main__':
    main()
