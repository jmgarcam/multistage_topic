#!/usr/bin/env python3
'''Clasificador Qwen3.5-2B con el adaptador de fine-tuning.'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from topic_classifier import load_json, load_temas


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_MODEL_NAME = 'Qwen/Qwen3.5-2B'
ADAPTER_PATH = SCRIPT_DIR.parent / 'FT_models' / 'Qwen3.5-2B'
MODEL_NAME = 'FT_models/Qwen3.5-2B'
DEVICE_MAP: str | None = None
MODEL_DTYPE: str | None = 'auto'
TRUST_REMOTE_CODE = False
MAX_NEW_TOKENS = 64
PREDICTION_FORMAT = 'JSON array'

SYSTEM_PROMPT = (
    'Eres un clasificador tematico de noticias. Selecciona todos los temas '
    'aplicables del catalogo proporcionado. Responde exclusivamente con un '
    'array JSON de cadenas, sin explicaciones ni texto adicional. Usa '
    '[\u0022ninguno\u0022] solo cuando no sea aplicable ningun tema; ninguno '
    'no se puede combinar con otras etiquetas.'
)


def build_prompt(text: str, candidates: list[dict[str, Any]]) -> str:
    candidate_names = [candidate['tema'] for candidate in candidates]
    allowed = '\n'.join(
        f'- {topic}' for topic in [*candidate_names, 'ninguno']
    )
    return f'Temas permitidos:\n{allowed}\n\nNoticia:\n{text}'


def validate_decision(
    raw_output: str, candidate_names: list[str]
) -> list[str] | str:
    try:
        parsed = json.loads(raw_output.strip())
    except (json.JSONDecodeError, TypeError):
        return 'invalid'
    if not isinstance(parsed, list) or not parsed:
        return 'invalid'

    canonical_by_casefold = {
        option.casefold(): option for option in [*candidate_names, 'ninguno']
    }
    normalized: list[str] = []
    for value in parsed:
        if not isinstance(value, str):
            return 'invalid'
        canonical = canonical_by_casefold.get(value.strip().casefold())
        if canonical is None or canonical in normalized:
            return 'invalid'
        normalized.append(canonical)

    if 'ninguno' in normalized and len(normalized) != 1:
        return 'invalid'
    return normalized


def load_model() -> tuple[Any, Any, Any]:
    adapter_config_path = ADAPTER_PATH / 'adapter_config.json'
    adapter_weights_path = ADAPTER_PATH / 'adapter_model.safetensors'
    if not adapter_config_path.is_file() or not adapter_weights_path.is_file():
        raise SystemExit(
            f'No se encontro un adaptador LoRA completo en {ADAPTER_PATH}.'
        )
    adapter_config = load_json(adapter_config_path)
    if not isinstance(adapter_config, dict):
        raise SystemExit(f'Configuracion no valida: {adapter_config_path}')
    if adapter_config.get('base_model_name_or_path') != BASE_MODEL_NAME:
        raise SystemExit(
            'El modelo base del adaptador no coincide con '
            f'{BASE_MODEL_NAME}.'
        )

    try:
        import torch
        import transformers
        from peft import PeftModel
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            'Faltan dependencias. Instala torch, peft, tqdm y la version mas '
            'reciente de transformers.'
        ) from exc

    model_class = None
    for class_name in (
        'Qwen3_5ForConditionalGeneration',
        'AutoModelForMultimodalLM',
        'AutoModelForImageTextToText',
    ):
        model_class = getattr(transformers, class_name, None)
        if model_class is not None:
            break
    if model_class is None:
        raise SystemExit(
            'Transformers no incluye Qwen3.5. Instala una version reciente.'
        )

    tokenizer_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {}
    if TRUST_REMOTE_CODE:
        tokenizer_kwargs['trust_remote_code'] = True
        model_kwargs['trust_remote_code'] = True
    if MODEL_DTYPE is not None:
        model_kwargs['dtype'] = MODEL_DTYPE

    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_PATH), **tokenizer_kwargs
    )
    if not tokenizer.chat_template:
        raise SystemExit('El tokenizer no contiene el chat template oficial.')

    if DEVICE_MAP is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_model = model_class.from_pretrained(
            BASE_MODEL_NAME, **model_kwargs
        )
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
        model.to(device)
    else:
        base_model = model_class.from_pretrained(
            BASE_MODEL_NAME,
            device_map=DEVICE_MAP,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
        device = model.device

    print(f'Dispositivo de entrada: {device}')
    print(f'Modelo base: {BASE_MODEL_NAME}')
    print(f'Adaptador FT: {ADAPTER_PATH}')
    model.eval()
    return model, tokenizer, device


def generate_decision(
    model: Any,
    tokenizer: Any,
    device: Any,
    text: str,
    candidates: list[dict[str, Any]],
    temas: dict[str, str],
) -> list[str] | str:
    del temas
    prompt = build_prompt(text, candidates)
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
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
    raw_output = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    candidate_names = [candidate['tema'] for candidate in candidates]
    return validate_decision(raw_output, candidate_names)
