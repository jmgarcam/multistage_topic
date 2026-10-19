#!/usr/bin/env python3
'''Funciones compartidas para clasificar topics con modelos instruct.'''

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


MODEL_NAME = 'Qwen/Qwen3.5-2B'
MODEL_CLASS = 'multimodal_lm'
DEVICE_MAP: str | None = None
MODEL_DTYPE: str | None = None
TRUST_REMOTE_CODE = False
MAX_NEW_TOKENS = 128


def load_json(path: Path) -> Any:
    try:
        with path.open('r', encoding='utf-8') as file_handle:
            return json.load(file_handle)
    except FileNotFoundError as exc:
        raise SystemExit(f'No se encontro el archivo: {path}') from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f'JSON no valido en {path}: {exc}') from exc


def load_temas(path: Path) -> dict[str, str]:
    temas = load_json(path)
    if not isinstance(temas, dict) or not temas:
        raise SystemExit(f'{path} debe contener un objeto JSON no vacio.')

    for tema, definicion in temas.items():
        if (
            not isinstance(tema, str)
            or not tema.strip()
            or not isinstance(definicion, str)
            or not definicion.strip()
        ):
            raise SystemExit(
                'Todos los temas y sus definiciones deben ser cadenas no vacias.'
            )
    return temas


def build_prompt(
    text: str, candidates: list[dict[str, Any]], temas: dict[str, str]
) -> str:
    candidate_blocks: list[str] = []
    candidate_names: list[str] = []
    for position, candidate in enumerate(candidates, start=1):
        tema = candidate['tema']
        candidate_names.append(tema)
        candidate_blocks.append(
            f'{position}. Tema candidato: {tema}\n'
            f'Definición exacta: {temas[tema]}'
        )

    allowed_options = '\n'.join(
        f'- {option}' for option in [*candidate_names, 'ninguno']
    )
    return (
        f'Noticia:\n{text}\n\n'
        'Temas candidatos y definiciones oficiales:\n'
        f'{chr(10).join(candidate_blocks)}\n\n'
        'Selecciona únicamente uno de los temas candidatos si su definición '
        'describe el tema principal de la noticia.\n'
        'No selecciones un tema solo porque aparezcan palabras relacionadas con él.\n'
        'La relación con el tema debe ser central en la noticia.\n'
        'Si ninguno describe el tema principal, responde ninguno.\n\n'
        f'Valores permitidos:\n{allowed_options}\n\n'
        'Responde exclusivamente con uno de los valores permitidos. '
        'No añadas explicación, razonamiento, puntuación ni texto adicional.'
    )


def normalize_spaces(value: str) -> str:
    return ' '.join(value.split())


def validate_decision(raw_output: str, candidate_names: list[str]) -> str:
    valid_options = [*candidate_names, 'ninguno']
    normalized_options = {
        normalize_spaces(option): option for option in valid_options
    }
    normalized_output = normalize_spaces(raw_output)
    if normalized_output in normalized_options:
        return normalized_options[normalized_output]

    canonical_by_casefold = {
        normalized.casefold(): canonical
        for normalized, canonical in normalized_options.items()
    }
    alternatives = sorted(normalized_options, key=len, reverse=True)
    option_pattern = '|'.join(re.escape(option) for option in alternatives)
    pattern = re.compile(rf'(?<!\w)({option_pattern})(?!\w)', re.IGNORECASE)
    extracted: list[str] = []
    for match in pattern.finditer(normalized_output):
        canonical = canonical_by_casefold.get(match.group(1).casefold())
        if canonical is not None and canonical not in extracted:
            extracted.append(canonical)
    if len(extracted) == 1:
        return extracted[0]
    return 'invalid'


def load_model() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            'Faltan dependencias para cargar el modelo. Instala torch, tqdm '
            'y la version mas reciente de transformers.'
        ) from exc

    tokenizer_class = AutoTokenizer
    if MODEL_CLASS == 'multimodal_lm':
        from transformers import AutoModelForMultimodalLM

        model_class = AutoModelForMultimodalLM
    elif MODEL_CLASS == 'mistral3':
        from transformers import (
            Mistral3ForConditionalGeneration,
            MistralCommonBackend,
        )

        model_class = Mistral3ForConditionalGeneration
        tokenizer_class = MistralCommonBackend
    elif MODEL_CLASS == 'causal_lm':
        from transformers import AutoModelForCausalLM

        model_class = AutoModelForCausalLM
    else:
        raise SystemExit(f'Clase de modelo no soportada: {MODEL_CLASS}')

    tokenizer_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {}
    if TRUST_REMOTE_CODE:
        tokenizer_kwargs['trust_remote_code'] = True
        model_kwargs['trust_remote_code'] = True
    if MODEL_DTYPE is not None:
        model_kwargs['dtype'] = MODEL_DTYPE

    tokenizer = tokenizer_class.from_pretrained(MODEL_NAME, **tokenizer_kwargs)
    if MODEL_CLASS != 'mistral3' and not tokenizer.chat_template:
        raise SystemExit('El tokenizer no contiene el chat template oficial.')

    if DEVICE_MAP is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_class.from_pretrained(MODEL_NAME, **model_kwargs)
        model.to(device)
    else:
        model = model_class.from_pretrained(
            MODEL_NAME,
            device_map=DEVICE_MAP,
            **model_kwargs,
        )
        device = model.device

    print(f'Dispositivo de entrada: {device}')
    print(f'Modelo: {MODEL_NAME}')
    model.eval()
    return model, tokenizer, device


def generate_decision(
    model: Any,
    tokenizer: Any,
    device: Any,
    text: str,
    candidates: list[dict[str, Any]],
    temas: dict[str, str],
) -> str:
    prompt = build_prompt(text, candidates, temas)
    messages = [
        {
            'role': 'system',
            'content': (
                'Eres un clasificador temático. Devuelve solo una etiqueta válida, '
                'sin explicación ni razonamiento.'
            ),
        },
        {'role': 'user', 'content': prompt},
    ]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
    )
    model_inputs = {
        name: tensor.to(device) for name, tensor in model_inputs.items()
    }
    input_length = model_inputs['input_ids'].shape[-1]

    import torch

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    new_token_ids = generated_ids[0, input_length:]
    raw_output = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    candidate_names = [candidate['tema'] for candidate in candidates]
    return validate_decision(raw_output, candidate_names)
