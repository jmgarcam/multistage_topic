#!/usr/bin/env python3
'''Evalúa los candidatos top-N mediante la API HTTP de Ollama.'''

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DATASET_DIR = Path(__file__).resolve().parent.parent / 'dataset_mteb'
MODEL_NAME = 'llama3.2:3b'
MAX_NEW_TOKENS = 128
VALIDITY_MAX_NEW_TOKENS = 512
PREDICTION_FORMAT = 'structured JSON with one enum label'
PROMPT_VERSION = 'mteb_forced_topic_selection_v1'
RERANK_RESULTS_FILE = 'reranked_topic_results.json'
LABELLED_TEST_DIR = 'data_test_labelled'
TOPICS_FILE = 'temas.json'
OUTPUT_FILE = 'ollama_llama_3_2_3b_top_n_no_none_results.json'
DECISION_FIELD = 'ollama_decision'
RAW_OUTPUT_FIELD = 'ollama_raw_output'
VALIDITY_RAW_OUTPUT_FIELD = 'ollama_validity_raw_output'
MODEL_LABEL = 'Ollama llama3.2:3b'
DEFAULT_BASE_URL = 'http://localhost:11434'
DEFAULT_TIMEOUT_SECONDS = 300.0
NONE_TOPIC = 'ninguno'
ALLOW_NONE_DECISION = False
K = 10
N_VALUES = [1, 3, 5, 7, 10]
CHECKPOINT_EVERY = 10


def resolve_path(configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return DATASET_DIR / path


def load_json(path: Path) -> Any:
    try:
        with path.open('r', encoding='utf-8') as file_handle:
            return json.load(file_handle)
    except FileNotFoundError as exc:
        raise SystemExit(f'No se encontró el archivo: {path}') from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f'JSON no válido en {path}: {exc}') from exc


def load_temas(path: Path) -> dict[str, str]:
    temas = load_json(path)
    if not isinstance(temas, dict) or not temas:
        raise SystemExit(f'{path} debe contener un objeto JSON no vacío.')

    for tema, definicion in temas.items():
        if (
            not isinstance(tema, str)
            or not tema.strip()
            or not isinstance(definicion, str)
            or not definicion.strip()
        ):
            raise SystemExit(
                'Todos los temas y sus definiciones deben ser cadenas no vacías.'
            )
    return temas


def normalize_spaces(value: str) -> str:
    return ' '.join(value.split())


def candidate_options(
    candidate_names: list[str], include_none: bool | None = None
) -> list[str]:
    if include_none is None:
        include_none = ALLOW_NONE_DECISION
    options = list(candidate_names)
    if include_none:
        options.append(NONE_TOPIC)
    return list(dict.fromkeys(options))


def validate_decision(
    raw_output: str,
    candidate_names: list[str],
    include_none: bool | None = None,
) -> str:
    valid_options = candidate_options(candidate_names, include_none)
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


def build_response_schema(
    candidate_names: list[str], include_none: bool | None = None
) -> dict[str, Any]:
    valid_options = candidate_options(candidate_names, include_none)
    return {
        'type': 'object',
        'properties': {
            'tema': {
                'type': 'string',
                'enum': valid_options,
            }
        },
        'required': ['tema'],
        'additionalProperties': False,
    }


def validate_structured_decision(
    raw_output: str,
    candidate_names: list[str],
    include_none: bool | None = None,
) -> str:
    try:
        decoded = json.loads(raw_output)
    except json.JSONDecodeError:
        return 'invalid'
    if not isinstance(decoded, dict) or set(decoded) != {'tema'}:
        return 'invalid'
    tema = decoded.get('tema')
    if not isinstance(tema, str):
        return 'invalid'
    return validate_decision(tema, candidate_names, include_none)


def build_prompt(
    text: str,
    candidates: list[dict[str, Any]],
    temas: dict[str, str],
    include_none: bool | None = None,
) -> str:
    if include_none is None:
        include_none = ALLOW_NONE_DECISION
    candidate_names = [candidate['tema'] for candidate in candidates]
    candidate_text = '\n\n'.join(
        f'{position}. Tema candidato: {tema}\n'
        f'   Definición: {temas[tema]}'
        for position, tema in enumerate(candidate_names, start=1)
    )
    response_schema = build_response_schema(candidate_names, include_none)
    valid_options = response_schema['properties']['tema']['enum']
    valid_options_text = '\n'.join(f'- {option}' for option in valid_options)
    return (
        'Identifica únicamente el tema principal de la noticia.\n\n'
        f'NOTICIA:\n{text}\n\n'
        f'TEMAS CANDIDATOS:\n{candidate_text}\n\n'
        'INSTRUCCIONES:\n\n'
        '- Debes elegir obligatoriamente uno de los temas candidatos.\n'
        '- Selecciona el tema que mejor describa el tema principal o contenido '
        'central de la noticia.\n'
        '- No elijas un tema únicamente porque aparezcan palabras, entidades '
        'o conceptos relacionados con él.\n'
        '- Ignora información secundaria, contexto, causas, consecuencias o '
        'menciones incidentales.\n'
        '- Si varios temas parecen relacionados, elige únicamente el que '
        'represente de forma más directa el foco principal de la noticia.\n'
        '- No puedes responder "ninguno", combinar varios temas ni utilizar '
        'un valor fuera de la lista de candidatos.\n\n'
        f'VALORES PERMITIDOS:\n{valid_options_text}\n\n'
        'Devuelve exclusivamente un objeto JSON que cumpla exactamente este '
        'esquema:\n\n'
        f'{json.dumps(response_schema, ensure_ascii=False, indent=2)}'
    )


def build_validity_schema(candidate_names: list[str]) -> dict[str, Any]:
    return {
        'type': 'object',
        'properties': {
            candidate_name: {
                'type': 'string',
                'enum': ['sí', 'no'],
            }
            for candidate_name in candidate_names
        },
        'required': candidate_names,
        'additionalProperties': False,
    }


def build_validity_prompt(
    text: str, candidates: list[dict[str, Any]], temas: dict[str, str]
) -> str:
    candidate_names = [candidate['tema'] for candidate in candidates]
    candidate_text = '\n\n'.join(
        f'{position}. {candidate_name}\n'
        f'Definición: {temas[candidate_name]}'
        for position, candidate_name in enumerate(candidate_names, start=1)
    )
    expected_output = '{\n' + ',\n'.join(
        f'  {json.dumps(candidate_name, ensure_ascii=False)}: "sí" | "no"'
        for candidate_name in candidate_names
    ) + '\n}'
    return (
        'Determina si cada uno de los temas candidatos representa el tema '
        'PRINCIPAL de la noticia.\n\n'
        f'NOTICIA:\n{text}\n\n'
        f'TEMAS CANDIDATOS:\n\n{candidate_text}\n\n'
        'Para cada candidato decide independientemente si podría asignarse '
        'como etiqueta principal de la noticia.\n\n'
        'Un tema es válido únicamente si representa directamente el contenido '
        'principal de la noticia.\n\n'
        'No es válido si:\n'
        '- aparece únicamente como contexto;\n'
        '- es una causa o consecuencia secundaria;\n'
        '- existe solo una relación indirecta;\n'
        '- comparte palabras o conceptos con la noticia pero no describe su '
        'tema principal;\n'
        '- otro asunto diferente constituye claramente el foco de la '
        'noticia.\n\n'
        'No es necesario seleccionar ningún tema. Si ninguno representa el '
        'tema principal, todos deben marcarse como "no".\n\n'
        f'Devuelve exclusivamente:\n\n{expected_output}'
    )


def validate_validity_response(
    raw_output: str, candidate_names: list[str]
) -> list[str] | None:
    try:
        decoded = json.loads(raw_output)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict) or set(decoded) != set(candidate_names):
        return None
    valid_topics: list[str] = []
    for candidate_name in candidate_names:
        value = decoded.get(candidate_name)
        if not isinstance(value, str):
            return None
        normalized = normalize_spaces(value).casefold()
        if normalized in {'sí', 'si'}:
            valid_topics.append(candidate_name)
        elif normalized != 'no':
            return None
    return valid_topics


def required_string(record: dict[str, Any], field: str, location: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f'{location}: falta un campo {field} no vacío.')
    return value


def validate_saved_decision(
    value: Any, candidate_topics: list[str], location: str
) -> list[str] | str:
    if value == 'invalid':
        return value
    allowed = set(candidate_topics)
    if ALLOW_NONE_DECISION:
        allowed.add(NONE_TOPIC)
    if isinstance(value, str):
        if value not in allowed:
            raise SystemExit(
                f'{location}: decisión fuera de los temas candidatos: {value!r}.'
            )
        return value
    if not isinstance(value, list) or not value:
        raise SystemExit(
            f'{location}: formato no válido para {DECISION_FIELD}.'
        )

    normalized: list[str] = []
    for topic in value:
        if not isinstance(topic, str) or topic not in allowed:
            raise SystemExit(
                f'{location}: decisión fuera de los temas candidatos: {topic!r}.'
            )
        if topic in normalized:
            raise SystemExit(f'{location}: tema predicho duplicado {topic!r}.')
        normalized.append(topic)
    if NONE_TOPIC in normalized and len(normalized) != 1:
        raise SystemExit(
            f'{location}: {NONE_TOPIC!r} no puede combinarse con otros temas.'
        )
    return normalized


def decision_matches_topics(
    decision: list[str] | str, correct_topics: list[str]
) -> bool:
    if isinstance(decision, list):
        return any(topic in correct_topics for topic in decision)
    return decision in correct_topics


def decision_is_none(decision: list[str] | str) -> bool:
    return decision == NONE_TOPIC or decision == [NONE_TOPIC]


def normalize_real_topics(
    value: Any, location: str, topic_catalog: set[str]
) -> list[str] | None:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list) and value:
        values = value
    else:
        raise SystemExit(
            f'{location}: tema debe ser una cadena o una lista no vacía.'
        )

    normalized: list[str] = []
    for topic_index, topic in enumerate(values):
        if not isinstance(topic, str) or not topic.strip():
            raise SystemExit(
                f'{location}, topic {topic_index}: se esperaba una cadena '
                'no vacía.'
            )
        topic = topic.strip()
        if topic in normalized:
            raise SystemExit(f'{location}: tema duplicado {topic!r}.')
        normalized.append(topic)

    none_topics = [
        topic for topic in normalized if topic.casefold() == NONE_TOPIC
    ]
    if none_topics:
        if len(normalized) != 1:
            raise SystemExit(
                f'{location}: {NONE_TOPIC!r} no puede combinarse con otros '
                'topics.'
            )
        return None

    unknown_topics = [
        topic for topic in normalized if topic not in topic_catalog
    ]
    if unknown_topics:
        raise SystemExit(
            f'{location}: temas inexistentes en temas.json: {unknown_topics}.'
        )
    return normalized


def load_ground_truth(
    directory: Path, topic_catalog: set[str]
) -> dict[str, list[str] | None]:
    if not directory.is_dir():
        raise SystemExit(f'No se encontró el directorio: {directory}')
    paths = sorted(directory.glob('*.json'))
    if not paths:
        raise SystemExit(f'No se encontraron archivos JSON en {directory}.')

    truth_by_id: dict[str, list[str] | None] = {}
    seen_locations: dict[str, str] = {}
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, list):
            raise SystemExit(f'{path} debe contener una lista JSON.')
        for record_index, record in enumerate(payload):
            location = f'{path}, elemento {record_index}'
            if not isinstance(record, dict):
                raise SystemExit(f'{location}: se esperaba un objeto JSON.')
            news_id = required_string(record, '_id', location)
            if news_id in truth_by_id:
                raise SystemExit(
                    f'{location}: _id duplicado; ya aparece en '
                    f'{seen_locations[news_id]}.'
                )
            seen_locations[news_id] = location
            truth_by_id[news_id] = normalize_real_topics(
                record.get('tema'), location, topic_catalog
            )

    return truth_by_id


def validate_configuration() -> None:
    if isinstance(K, bool) or not isinstance(K, int) or K < 1:
        raise SystemExit('K debe ser un entero positivo.')
    if not isinstance(N_VALUES, list) or not N_VALUES:
        raise SystemExit('N_VALUES debe ser una lista no vacía.')
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n < 1
        for n in N_VALUES
    ):
        raise SystemExit(
            'Todos los elementos de N_VALUES deben ser enteros positivos.'
        )
    if N_VALUES != sorted(set(N_VALUES)):
        raise SystemExit(
            'N_VALUES debe estar ordenada de menor a mayor y no tener duplicados.'
        )
    if N_VALUES[-1] > K:
        raise SystemExit('N_VALUES no puede contener valores mayores que K.')
    if CHECKPOINT_EVERY < 1:
        raise SystemExit('CHECKPOINT_EVERY debe ser mayor que cero.')
    if VALIDITY_MAX_NEW_TOKENS < 1:
        raise SystemExit(
            'VALIDITY_MAX_NEW_TOKENS debe ser mayor que cero.'
        )


def select_rerank_result(
    record: dict[str, Any], location: str
) -> dict[str, Any]:
    rerank_results = record.get('rerank_results')
    if not isinstance(rerank_results, list) or not rerank_results:
        raise SystemExit(
            f'{location}: rerank_results debe ser una lista no vacía.'
        )

    selected_result = None
    seen_k_values: set[int] = set()
    for result_index, result in enumerate(rerank_results):
        result_location = f'{location}, rerank_result {result_index}'
        if not isinstance(result, dict):
            raise SystemExit(f'{result_location}: se esperaba un objeto JSON.')
        result_k = result.get('k')
        if (
            isinstance(result_k, bool)
            or not isinstance(result_k, int)
            or result_k < 1
        ):
            raise SystemExit(f'{result_location}: k debe ser un entero positivo.')
        if result_k in seen_k_values:
            raise SystemExit(
                f'{result_location}: valor k duplicado: {result_k}.'
            )
        seen_k_values.add(result_k)
        if result_k == K:
            selected_result = result

    if selected_result is None:
        raise SystemExit(f'{location}: no contiene resultados para K={K}.')
    return selected_result


def prepare_records(
    payload: Any,
    truth_by_id: dict[str, list[str] | None],
    topics: dict[str, str],
) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise SystemExit(
            'reranked_topic_results.json debe contener una lista no vacía.'
        )

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record_index, record in enumerate(payload):
        location = f'Resultado {record_index}'
        if not isinstance(record, dict):
            raise SystemExit(f'{location}: se esperaba un objeto JSON.')

        news_id = required_string(record, '_id', location)
        if news_id in seen_ids:
            raise SystemExit(f'{location}: _id duplicado {news_id}.')
        seen_ids.add(news_id)

        source_file = required_string(record, 'file', location)
        text = required_string(record, 'text', location)
        selected_result = select_rerank_result(record, location)
        candidates = selected_result.get('reranked_topics')
        if not isinstance(candidates, list) or len(candidates) != K:
            raise SystemExit(
                f'{location}: reranked_topics para K={K} debe contener '
                f'exactamente {K} temas.'
            )

        prepared_candidates: list[dict[str, str]] = []
        seen_topics: set[str] = set()
        for candidate_index, candidate in enumerate(candidates):
            candidate_location = (
                f'{location}, K={K}, candidato {candidate_index}'
            )
            if not isinstance(candidate, dict):
                raise SystemExit(
                    f'{candidate_location}: se esperaba un objeto JSON.'
                )
            topic = required_string(candidate, 'tema', candidate_location)
            if topic not in topics:
                raise SystemExit(
                    f'{candidate_location}: {topic!r} no existe en temas.json.'
                )
            if topic in seen_topics:
                raise SystemExit(
                    f'{candidate_location}: tema duplicado {topic!r}.'
                )
            seen_topics.add(topic)
            prepared_candidates.append({'tema': topic})

        if news_id not in truth_by_id:
            continue
        correct_topics = truth_by_id[news_id]
        if correct_topics is None:
            correct_topics = [NONE_TOPIC]
        records.append(
            {
                '_id': news_id,
                'file': source_file,
                'text': text,
                'topics_correctos': correct_topics,
                'candidates': prepared_candidates,
            }
        )

    truth_ids = set(truth_by_id)
    missing_ids = truth_ids - seen_ids
    if missing_ids:
        raise SystemExit(
            'Faltan textos o resultados de reranking para '
            f'{len(missing_ids)} noticias del conjunto de validación.'
        )
    return records


def get_base_url() -> str:
    base_url = os.environ.get('OLLAMA_BASE_URL', DEFAULT_BASE_URL).strip()
    base_url = base_url.rstrip('/')
    if base_url.endswith('/api'):
        base_url = base_url[:-4]

    parsed = urlparse(base_url)
    if (
        parsed.scheme not in {'http', 'https'}
        or not parsed.netloc
        or parsed.path not in {'', '/'}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise SystemExit(
            'OLLAMA_BASE_URL debe ser una URL base válida, por ejemplo '
            f'{DEFAULT_BASE_URL}.'
        )
    return base_url


def get_timeout_seconds() -> float:
    raw_value = os.environ.get(
        'OLLAMA_TIMEOUT_SECONDS', str(DEFAULT_TIMEOUT_SECONDS)
    )
    try:
        timeout = float(raw_value)
    except ValueError as exc:
        raise SystemExit(
            'OLLAMA_TIMEOUT_SECONDS debe ser un número positivo.'
        ) from exc
    if timeout <= 0:
        raise SystemExit(
            'OLLAMA_TIMEOUT_SECONDS debe ser un número positivo.'
        )
    return timeout


@dataclass(frozen=True)
class OllamaClient:
    base_url: str
    timeout_seconds: float

    def request_json(
        self, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None
        headers = {'Accept': 'application/json'}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            headers['Content-Type'] = 'application/json'

        request = Request(
            f'{self.base_url}{path}',
            data=data,
            headers=headers,
            method='POST' if payload is not None else 'GET',
        )
        try:
            with urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                raw_response = response.read()
        except HTTPError as exc:
            details = exc.read().decode('utf-8', errors='replace').strip()
            suffix = f': {details[:500]}' if details else ''
            raise RuntimeError(
                f'Ollama respondió HTTP {exc.code} en {path}{suffix}'
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f'No se pudo conectar con Ollama en {self.base_url}: '
                f'{exc.reason}'
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f'Ollama excedió el timeout de {self.timeout_seconds:g} s.'
            ) from exc

        try:
            decoded = json.loads(raw_response)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f'Ollama devolvió una respuesta JSON no válida en {path}.'
            ) from exc
        if not isinstance(decoded, dict):
            raise RuntimeError(
                f'Ollama devolvió un formato inesperado en {path}.'
            )
        return decoded

    def ensure_model_available(self) -> None:
        response = self.request_json('/api/tags')
        models = response.get('models')
        if not isinstance(models, list):
            raise RuntimeError(
                'La respuesta de /api/tags no contiene una lista models.'
            )

        available_names = {
            value
            for model in models
            if isinstance(model, dict)
            for field in ('name', 'model')
            for value in [model.get(field)]
            if isinstance(value, str)
        }
        if MODEL_NAME not in available_names:
            raise SystemExit(
                f'El modelo {MODEL_NAME!r} no está disponible en Ollama. '
                f'Descárgalo con: ollama pull {MODEL_NAME}'
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any],
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        response = self.request_json(
            '/api/chat',
            {
                'model': MODEL_NAME,
                'messages': messages,
                'stream': False,
                'format': response_format,
                'options': {
                    'temperature': 0,
                    'seed': 0,
                    'num_predict': max_new_tokens,
                },
            },
        )
        if response.get('done') is not True:
            raise RuntimeError(
                'Ollama no marcó como terminada la respuesta de /api/chat.'
            )
        message = response.get('message')
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                'Ollama no devolvió contenido en message.content.'
            )
        return content


def load_model() -> tuple[OllamaClient, None, None]:
    '''Comprueba la API y devuelve un cliente; no carga el modelo en Python.'''
    client = OllamaClient(get_base_url(), get_timeout_seconds())
    client.ensure_model_available()
    print(f'API de Ollama: {client.base_url}')
    print(f'Modelo: {MODEL_NAME}')
    return client, None, None


def generate_decision(
    client: OllamaClient,
    _tokenizer: None,
    _device: None,
    text: str,
    candidates: list[dict[str, Any]],
    temas: dict[str, str],
) -> tuple[str, str | None, str | None]:
    candidate_names = [candidate['tema'] for candidate in candidates]

    validity_raw_output: str | None = None
    if ALLOW_NONE_DECISION:
        validity_raw_output = client.chat(
            [
                {
                    'role': 'system',
                    'content': (
                        'Eres un filtro temático. Evalúa independientemente '
                        'cada tema candidato y devuelve únicamente el objeto '
                        'JSON solicitado.'
                    ),
                },
                {
                    'role': 'user',
                    'content': build_validity_prompt(text, candidates, temas),
                },
            ],
            build_validity_schema(candidate_names),
            max_new_tokens=VALIDITY_MAX_NEW_TOKENS,
        )
        valid_topic_names = validate_validity_response(
            validity_raw_output, candidate_names
        )
        if valid_topic_names == []:
            return NONE_TOPIC, validity_raw_output, None
        if valid_topic_names is None:
            return 'invalid', validity_raw_output, None
        candidates = [
            candidate
            for candidate in candidates
            if candidate['tema'] in valid_topic_names
        ]
        candidate_names = valid_topic_names

    prompt = build_prompt(text, candidates, temas, include_none=False)
    response_schema = build_response_schema(
        candidate_names, include_none=False
    )
    raw_output = client.chat(
        [
            {
                'role': 'system',
                'content': (
                    'Eres un clasificador temático. Devuelve únicamente el '
                    'objeto JSON solicitado y usa exactamente una de las '
                    'etiquetas permitidas como valor de tema.'
                ),
            },
            {'role': 'user', 'content': prompt},
        ],
        response_schema,
    )
    decision = validate_structured_decision(
        raw_output, candidate_names, include_none=False
    )
    return decision, validity_raw_output, raw_output


def validate_prediction_outputs(
    prediction: dict[str, Any],
    candidate_names: list[str],
    decision: list[str] | str,
    location: str,
) -> None:
    raw_output = prediction.get(RAW_OUTPUT_FIELD)
    validity_raw_output = prediction.get(VALIDITY_RAW_OUTPUT_FIELD)

    if not ALLOW_NONE_DECISION:
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise SystemExit(
                f'{location}: falta una respuesta bruta de Ollama no vacía.'
            )
        if (
            validate_structured_decision(
                raw_output, candidate_names, include_none=False
            )
            != decision
        ):
            raise SystemExit(
                f'{location}: la respuesta bruta no coincide con la decisión '
                'guardada.'
            )
        return

    if (
        not isinstance(validity_raw_output, str)
        or not validity_raw_output.strip()
    ):
        raise SystemExit(
            f'{location}: falta la respuesta del filtro de validez.'
        )
    valid_topic_names = validate_validity_response(
        validity_raw_output, candidate_names
    )
    if valid_topic_names == []:
        if decision != NONE_TOPIC or raw_output is not None:
            raise SystemExit(
                f'{location}: una respuesta de validez negativa debe producir '
                f'{NONE_TOPIC!r} sin ejecutar la selección.'
            )
        return
    if valid_topic_names is None:
        if decision != 'invalid' or raw_output is not None:
            raise SystemExit(
                f'{location}: una respuesta de validez inválida debe quedar '
                'marcada como invalid.'
            )
        return
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise SystemExit(
            f'{location}: falta la respuesta de selección temática.'
        )
    if (
        validate_structured_decision(
            raw_output, valid_topic_names, include_none=False
        )
        != decision
    ):
        raise SystemExit(
            f'{location}: la selección temática no coincide con la decisión '
            'guardada.'
        )


def build_report(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    total_news: int,
) -> dict[str, Any]:
    expected_predictions = len(records) * len(N_VALUES)
    topic_records = [
        record
        for record in records
        if record['topics_correctos'] != [NONE_TOPIC]
    ]
    none_ground_truth_count = len(records) - len(topic_records)
    accuracy_by_n = []
    for n in N_VALUES:
        results = [result for result in predictions if result['n'] == n]
        correct = sum(result['matches_real_topic'] for result in results)
        none_decisions = sum(
            decision_is_none(result[DECISION_FIELD]) for result in results
        )
        invalid_decisions = sum(
            result[DECISION_FIELD] == 'invalid' for result in results
        )
        candidate_hits = sum(
            any(
                topic in {
                    candidate['tema']
                    for candidate in record['candidates'][:n]
                }
                for topic in record['topics_correctos']
            )
            for record in topic_records
        )
        none_results = [
            result
            for result in results
            if result['topics_correctos'] == [NONE_TOPIC]
        ]
        correct_none_predictions = sum(
            result['matches_real_topic'] for result in none_results
        )
        accuracy_by_n.append(
            {
                'n': n,
                'evaluated_news': len(results),
                'expected_news': len(records),
                'correct_predictions': correct,
                'incorrect_predictions': len(results) - correct,
                'none_decisions': none_decisions,
                'invalid_decisions': invalid_decisions,
                'accuracy': (
                    round(correct / len(results), 6) if results else None
                ),
                'accuracy_percent': (
                    round(correct * 100 / len(results), 2)
                    if results
                    else None
                ),
                'candidate_recall_evaluable_news': len(topic_records),
                'candidate_recall_ceiling': (
                    round(candidate_hits / len(topic_records), 6)
                    if topic_records
                    else None
                ),
                'candidate_recall_ceiling_percent': (
                    round(candidate_hits * 100 / len(topic_records), 2)
                    if topic_records
                    else None
                ),
                'ground_truth_none_news': len(none_results),
                'correct_none_predictions': correct_none_predictions,
                'incorrect_none_predictions': (
                    len(none_results) - correct_none_predictions
                ),
                'none_accuracy': (
                    round(correct_none_predictions / len(none_results), 6)
                    if none_results
                    else None
                ),
                'none_accuracy_percent': (
                    round(
                        correct_none_predictions * 100 / len(none_results), 2
                    )
                    if none_results
                    else None
                ),
                'false_none_decisions': (
                    none_decisions - correct_none_predictions
                ),
            }
        )

    return {
        'status': (
            'complete'
            if len(predictions) == expected_predictions
            else 'in_progress'
        ),
        'configuration': {
            'model': MODEL_NAME,
            'max_new_tokens': MAX_NEW_TOKENS,
            'validity_max_new_tokens': (
                VALIDITY_MAX_NEW_TOKENS if ALLOW_NONE_DECISION else None
            ),
            'prompt_version': PROMPT_VERSION,
            'decision_strategy': (
                'validity_gate_then_topic_selection'
                if ALLOW_NONE_DECISION
                else 'forced_topic_selection'
            ),
            'k': K,
            'n_values': N_VALUES,
            'none_ground_truth_excluded': False,
            'ground_truth_source': 'data_test_labelled',
            'prediction_format': PREDICTION_FORMAT,
            'multi_topic_match': 'any',
            'generation': 'Ollama API (temperature=0, seed=0)',
            'api_base_url': get_base_url(),
        },
        'dataset': {
            'total_news': total_news,
            'evaluable_news': len(records),
            'excluded_none_news': total_news - len(records),
            'evaluated_none_news': none_ground_truth_count,
            'topic_ground_truth_news': len(topic_records),
            'multi_topic_news': sum(
                len(record['topics_correctos']) > 1 for record in records
            ),
            'total_ground_truth_topics': sum(
                len(record['topics_correctos']) for record in records
            ),
            'expected_predictions': expected_predictions,
            'completed_predictions': len(predictions),
        },
        'accuracy_by_n': accuracy_by_n,
        'predictions': predictions,
    }


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as file_handle:
        json.dump(report, file_handle, ensure_ascii=False, indent=2)
        file_handle.write('\n')
    temporary_path.replace(path)


def load_checkpoint(
    path: Path, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []

    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f'Checkpoint no válido: {path}')
    configuration = payload.get('configuration')
    expected_configuration = {
        'model': MODEL_NAME,
        'prompt_version': PROMPT_VERSION,
        'k': K,
        'n_values': N_VALUES,
    }
    if not isinstance(configuration, dict) or any(
        configuration.get(field) != value
        for field, value in expected_configuration.items()
    ):
        raise SystemExit(
            'El checkpoint existente usa otra configuración. Muévelo o '
            'elimínalo antes de iniciar este experimento.'
        )

    predictions = payload.get('predictions')
    if not isinstance(predictions, list):
        raise SystemExit('El checkpoint no contiene una lista predictions.')

    records_by_id = {record['_id']: record for record in records}
    seen_keys: set[tuple[str, int]] = set()
    for prediction_index, prediction in enumerate(predictions):
        location = f'Checkpoint, predicción {prediction_index}'
        if not isinstance(prediction, dict):
            raise SystemExit(f'{location}: se esperaba un objeto JSON.')
        news_id = required_string(prediction, '_id', location)
        n = prediction.get('n')
        if news_id not in records_by_id or n not in N_VALUES:
            raise SystemExit(f'{location}: combinación _id/N no válida.')
        key = (news_id, n)
        if key in seen_keys:
            raise SystemExit(f'{location}: combinación _id/N duplicada.')
        seen_keys.add(key)

        record = records_by_id[news_id]
        expected_candidates = [
            candidate['tema'] for candidate in record['candidates'][:n]
        ]
        if prediction.get('candidate_topics') != expected_candidates:
            raise SystemExit(
                f'{location}: candidate_topics inconsistente. '
                'El checkpoint fue generado con otros candidatos o con otro '
                'orden de reranking.'
            )
        decision = validate_saved_decision(
            prediction.get(DECISION_FIELD), expected_candidates, location
        )
        prediction[DECISION_FIELD] = decision
        validate_prediction_outputs(
            prediction, expected_candidates, decision, location
        )

        correct_topics = record['topics_correctos']
        prediction.pop('topic_correcto', None)
        prediction['topics_correctos'] = list(correct_topics)
        prediction['candidate_contains_real_topic'] = any(
            topic in expected_candidates for topic in correct_topics
        )
        prediction['matches_real_topic'] = decision_matches_topics(
            decision, correct_topics
        )
        elapsed = prediction.get('inference_time_seconds')
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise SystemExit(f'{location}: tiempo de inferencia no válido.')

    return predictions


def sort_predictions(
    predictions: list[dict[str, Any]], records: list[dict[str, Any]]
) -> None:
    record_order = {
        record['_id']: index for index, record in enumerate(records)
    }
    n_order = {n: index for index, n in enumerate(N_VALUES)}
    predictions.sort(
        key=lambda prediction: (
            record_order[prediction['_id']],
            n_order[prediction['n']],
        )
    )


def print_summary(report: dict[str, Any]) -> None:
    print('Estado: ' + str(report['status']))
    for result in report['accuracy_by_n']:
        n = result['n']
        completed = result['evaluated_news']
        expected = result['expected_news']
        correct = result['correct_predictions']
        accuracy = result['accuracy_percent']
        if accuracy is None:
            print(f'N={n}: 0/{expected} decisiones completadas')
        else:
            print(
                f'N={n}: {correct}/{completed} coincidencias '
                f'({accuracy:.2f}%); progreso {completed}/{expected}'
            )


def main() -> None:
    validate_configuration()
    if MAX_NEW_TOKENS < 1:
        raise SystemExit('MAX_NEW_TOKENS debe ser mayor que cero.')

    topics = load_temas(resolve_path(TOPICS_FILE))
    truth_by_id = load_ground_truth(
        resolve_path(LABELLED_TEST_DIR), set(topics)
    )
    records = prepare_records(
        load_json(resolve_path(RERANK_RESULTS_FILE)),
        truth_by_id,
        topics,
    )
    if not records:
        raise SystemExit(
            f'No hay noticias evaluables para ejecutar {MODEL_LABEL}.'
        )

    output_path = resolve_path(OUTPUT_FILE)
    predictions = load_checkpoint(output_path, records)
    completed_keys = {
        (prediction['_id'], prediction['n']) for prediction in predictions
    }
    pending = [
        (record, n)
        for record in records
        for n in N_VALUES
        if (record['_id'], n) not in completed_keys
    ]

    def checkpoint() -> dict[str, Any]:
        sort_predictions(predictions, records)
        report = build_report(records, predictions, len(truth_by_id))
        save_report(output_path, report)
        return report

    if not pending:
        report = checkpoint()
        print(f'Resultados ya completos: {output_path}')
        print_summary(report)
        return

    none_count = sum(
        record['topics_correctos'] == [NONE_TOPIC] for record in records
    )
    print(f'Noticias evaluables: {len(records)}')
    print(f'Noticias con etiqueta {NONE_TOPIC!r} incluidas: {none_count}')
    print(f'Decisiones pendientes: {len(pending)}')
    client, tokenizer, device = load_model()
    generated_since_checkpoint = 0

    try:
        for pending_index, (record, n) in enumerate(pending, start=1):
            candidates = record['candidates'][:n]
            start_time = time.perf_counter()
            decision, validity_raw_output, raw_output = generate_decision(
                client,
                tokenizer,
                device,
                record['text'],
                candidates,
                topics,
            )
            elapsed_seconds = time.perf_counter() - start_time
            candidate_topics = [candidate['tema'] for candidate in candidates]
            predictions.append(
                {
                    '_id': record['_id'],
                    'file': record['file'],
                    'topics_correctos': record['topics_correctos'],
                    'n': n,
                    'candidate_topics': candidate_topics,
                    'candidate_contains_real_topic': any(
                        topic in candidate_topics
                        for topic in record['topics_correctos']
                    ),
                    DECISION_FIELD: decision,
                    VALIDITY_RAW_OUTPUT_FIELD: validity_raw_output,
                    RAW_OUTPUT_FIELD: raw_output,
                    'matches_real_topic': decision_matches_topics(
                        decision, record['topics_correctos']
                    ),
                    'inference_time_seconds': round(elapsed_seconds, 6),
                }
            )
            generated_since_checkpoint += 1
            if generated_since_checkpoint >= CHECKPOINT_EVERY:
                checkpoint()
                print(
                    f'Progreso: {pending_index}/{len(pending)} decisiones '
                    'pendientes procesadas.'
                )
                generated_since_checkpoint = 0
    except BaseException:
        checkpoint()
        print(f'Checkpoint guardado tras una interrupción: {output_path}')
        raise

    report = checkpoint()
    print(f'Resultados guardados: {output_path}')
    print_summary(report)


if __name__ == '__main__':
    main()
