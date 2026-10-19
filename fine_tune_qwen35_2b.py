#!/usr/bin/env python3
'''Ajuste fino supervisado de Qwen3.5-2B para clasificar noticias.

Lee todos los archivos ``*.json`` de ``data_FT/02_etiquetadas`` y selecciona
aleatoriamente hasta 50 noticias de cada tema con una semilla reproducible.
Cada archivo debe ser una lista de objetos con ``text`` y ``tema``. El
entrenamiento usa LoRA y, por defecto, cuantizacion de 4 bits (QLoRA). La
perdida se calcula solo sobre la respuesta. No se usa validacion ni test.

Dependencias (Qwen3.5 requiere Transformers reciente)::

    pip install torch accelerate peft bitsandbytes
    pip install transformers@git+https://github.com/huggingface/transformers.git@main

Uso::

    python fine_tune_qwen35_2b.py --dry-run
    python fine_tune_qwen35_2b.py
'''

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / 'data_FT' / '02_etiquetadas'
DEFAULT_TOPICS_FILE = SCRIPT_DIR / 'data_FT' / 'temas.json'
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'FT_models' / 'Qwen3.5-2B'
DEFAULT_MODEL_NAME = 'Qwen/Qwen3.5-2B'
NONE_TOPIC = 'ninguno'

SYSTEM_PROMPT = (
    'Eres un clasificador tematico de noticias. Selecciona todos los temas '
    'aplicables del catalogo proporcionado. Responde exclusivamente con un '
    'array JSON de cadenas, sin explicaciones ni texto adicional. Usa '
    '[\u0022ninguno\u0022] solo cuando no sea aplicable ningun tema; ninguno '
    'no se puede combinar con otras etiquetas.'
)

# Capas de lenguaje; la torre de vision utiliza nombres distintos.
DEFAULT_LORA_TARGETS = (
    'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj',
    'down_proj', 'in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a',
    'out_proj',
)


@dataclass(frozen=True)
class TrainingExample:
    text: str
    topics: tuple[str, ...]
    source_file: str
    source_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Entrena Qwen3.5-2B con todos los JSON, sin validacion ni test.'
    )
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--topics-file', type=Path, default=DEFAULT_TOPICS_FILE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--model-name', default=DEFAULT_MODEL_NAME)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--epochs', type=float, default=3.0)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('--warmup-ratio', type=float, default=0.03)
    parser.add_argument('--logging-steps', type=int, default=10)
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument('--save-total-limit', type=int, default=2)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--examples-per-topic', type=int, default=50,
        help='Maximo de noticias por tema (50 por defecto).',
    )
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument(
        '--precision', choices=('auto', 'bf16', 'fp16', 'fp32'), default='auto'
    )
    parser.add_argument(
        '--load-in-4bit', action=argparse.BooleanOptionalAction, default=True,
        help='Usa QLoRA de 4 bits (activado por defecto).',
    )
    parser.add_argument(
        '--gradient-checkpointing',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Valida y resume los JSON sin cargar ni entrenar el modelo.',
    )
    args = parser.parse_args()
    validate_cli_args(args, parser)
    return args


def validate_cli_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    positive_values = {
        '--max-length': args.max_length,
        '--epochs': args.epochs,
        '--learning-rate': args.learning_rate,
        '--batch-size': args.batch_size,
        '--gradient-accumulation-steps': args.gradient_accumulation_steps,
        '--logging-steps': args.logging_steps,
        '--save-steps': args.save_steps,
        '--save-total-limit': args.save_total_limit,
        '--lora-r': args.lora_r,
        '--lora-alpha': args.lora_alpha,
        '--examples-per-topic': args.examples_per_topic,
    }
    for option, value in positive_values.items():
        if value <= 0:
            parser.error(f'{option} debe ser mayor que cero.')
    if not 0 <= args.warmup_ratio < 1:
        parser.error('--warmup-ratio debe estar en el intervalo [0, 1).')
    if not 0 <= args.lora_dropout < 1:
        parser.error('--lora-dropout debe estar en el intervalo [0, 1).')
    if args.num_workers < 0:
        parser.error('--num-workers no puede ser negativo.')


def load_json(path: Path) -> Any:
    try:
        with path.open('r', encoding='utf-8-sig') as file_handle:
            return json.load(file_handle)
    except FileNotFoundError as exc:
        raise SystemExit(f'No se encontro el archivo: {path}') from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f'JSON no valido en {path}: {exc}') from exc


def load_topic_catalog(path: Path) -> tuple[str, ...]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise SystemExit(f'{path} debe contener un objeto JSON no vacio.')

    topics: list[str] = []
    for topic, definition in payload.items():
        if not isinstance(topic, str) or not topic.strip():
            raise SystemExit(f'{path}: los nombres de tema deben ser texto.')
        if not isinstance(definition, str) or not definition.strip():
            raise SystemExit(
                f'{path}: la definicion de {topic!r} debe ser texto no vacio.'
            )
        normalized_topic = topic.strip()
        if normalized_topic == NONE_TOPIC:
            raise SystemExit(
                f'{path}: {NONE_TOPIC!r} es una etiqueta reservada y no debe '
                'figurar en el catalogo.'
            )
        topics.append(normalized_topic)

    if len(topics) != len(set(topics)):
        raise SystemExit(f'{path}: contiene nombres de tema duplicados.')
    return tuple(topics)


def normalize_topics(
    raw_topics: Any,
    *,
    allowed_topics: set[str],
    location: str,
) -> tuple[str, ...]:
    if isinstance(raw_topics, str):
        topic_values = [raw_topics]
    elif isinstance(raw_topics, list) and raw_topics:
        topic_values = raw_topics
    else:
        raise SystemExit(
            f'{location}: tema debe ser una cadena o una lista no vacia.'
        )

    normalized: list[str] = []
    for topic_index, topic in enumerate(topic_values):
        if not isinstance(topic, str) or not topic.strip():
            raise SystemExit(
                f'{location}, tema {topic_index}: se esperaba texto no vacio.'
            )
        clean_topic = topic.strip()
        if clean_topic in normalized:
            raise SystemExit(f'{location}: tema duplicado {clean_topic!r}.')
        if clean_topic not in allowed_topics:
            raise SystemExit(
                f'{location}: el tema {clean_topic!r} no existe en el catalogo.'
            )
        normalized.append(clean_topic)

    if NONE_TOPIC in normalized and len(normalized) != 1:
        raise SystemExit(
            f'{location}: {NONE_TOPIC!r} no se puede combinar con otros temas.'
        )
    return tuple(normalized)


def load_training_examples(
    input_dir: Path, catalog_topics: Sequence[str]
) -> tuple[list[TrainingExample], list[Path]]:
    if not input_dir.is_dir():
        raise SystemExit(f'No se encontro el directorio: {input_dir}')

    input_paths = sorted(input_dir.glob('*.json'))
    if not input_paths:
        raise SystemExit(f'No se encontraron archivos JSON en {input_dir}.')

    allowed_topics = {*catalog_topics, NONE_TOPIC}
    examples: list[TrainingExample] = []
    for input_path in input_paths:
        payload = load_json(input_path)
        if not isinstance(payload, list):
            raise SystemExit(f'{input_path} debe contener una lista JSON.')

        for record_index, record in enumerate(payload):
            location = f'{input_path}, elemento {record_index}'
            if not isinstance(record, dict):
                raise SystemExit(f'{location}: se esperaba un objeto JSON.')

            text = record.get('text')
            if not isinstance(text, str) or not text.strip():
                raise SystemExit(
                    f'{location}: falta un campo text de texto no vacio.'
                )
            topics = normalize_topics(
                record.get('tema'),
                allowed_topics=allowed_topics,
                location=location,
            )
            examples.append(
                TrainingExample(
                    text=text.strip(),
                    topics=topics,
                    source_file=input_path.name,
                    source_index=record_index,
                )
            )

    if not examples:
        raise SystemExit(f'Los JSON de {input_dir} no contienen ejemplos.')
    return examples, input_paths


def sample_examples_per_topic(
    examples: Sequence[TrainingExample],
    catalog_topics: Sequence[str],
    limit: int,
    seed: int,
) -> list[TrainingExample]:
    '''Selecciona una muestra multietiqueta con un maximo por tema.'''

    labels = (*catalog_topics, NONE_TOPIC)
    available = Counter(
        topic for example in examples for topic in example.topics
    )
    targets = {topic: min(limit, available[topic]) for topic in labels}

    # Para alcanzar el objetivo de un tema con pocos datos es obligatorio
    # conservar todos sus ejemplos. Se incorporan antes del muestreo aleatorio.
    selected = {
        index
        for index, example in enumerate(examples)
        if any(0 < available[topic] <= limit for topic in example.topics)
    }
    selected_counts = Counter(
        topic
        for index in selected
        for topic in examples[index].topics
    )
    over_limit = {
        topic: count
        for topic, count in selected_counts.items()
        if count > targets[topic]
    }
    if over_limit:
        raise SystemExit(
            'No es posible respetar el limite por tema debido a la '
            f'combinacion de etiquetas: {over_limit}.'
        )

    rng = random.Random(seed)
    while True:
        deficits = {
            topic: targets[topic] - selected_counts[topic]
            for topic in labels
            if selected_counts[topic] < targets[topic]
        }
        if not deficits:
            break

        feasible_by_topic: dict[str, list[int]] = {}
        for topic, required in deficits.items():
            candidates = [
                index
                for index, example in enumerate(examples)
                if index not in selected
                and topic in example.topics
                and all(
                    selected_counts[label] < targets[label]
                    for label in example.topics
                )
            ]
            if len(candidates) < required:
                raise SystemExit(
                    'No se pudo construir una muestra que respete todas las '
                    f'etiquetas. Tema afectado: {topic!r}.'
                )
            feasible_by_topic[topic] = candidates

        # Atiende primero el tema con menos margen de candidatos.
        chosen_topic = min(
            deficits,
            key=lambda topic: (
                len(feasible_by_topic[topic]) - deficits[topic],
                len(feasible_by_topic[topic]),
                topic,
            ),
        )
        candidates = feasible_by_topic[chosen_topic]
        rng.shuffle(candidates)
        # En caso de empate aleatorio, se prefieren noticias que cubren varios
        # deficits a la vez y reducen el numero total de ejemplos necesarios.
        candidates.sort(
            key=lambda index: sum(
                label in deficits for label in examples[index].topics
            ),
            reverse=True,
        )
        chosen_index = candidates[0]
        selected.add(chosen_index)
        selected_counts.update(examples[chosen_index].topics)

    shortages = {
        topic: available[topic]
        for topic in labels
        if available[topic] < limit
    }
    if shortages:
        print(f'Aviso: temas con menos de {limit} noticias; se usan todas:')
        for topic, count in shortages.items():
            print(f'    - {topic}: {count}')

    sampled = [examples[index] for index in sorted(selected)]
    print(
        f'Muestreo aleatorio por tema: maximo={limit}, seed={seed}, '
        f'noticias seleccionadas={len(sampled)}'
    )
    return sampled


def print_dataset_summary(
    examples: Sequence[TrainingExample], input_paths: Sequence[Path]
) -> None:
    label_counts = Counter(
        topic for example in examples for topic in example.topics
    )
    file_counts = Counter(example.source_file for example in examples)
    text_counts = Counter(example.text for example in examples)
    repeated = sum(count - 1 for count in text_counts.values() if count > 1)

    print('Datos de entrenamiento validados')
    print(f'  Directorio: {input_paths[0].parent}')
    print(f'  Archivos JSON: {len(input_paths)}')
    for path in input_paths:
        print(f'    - {path.name}: {file_counts[path.name]} ejemplos')
    print(f'  Ejemplos totales: {len(examples)}')
    multiple = sum(len(example.topics) > 1 for example in examples)
    print(f'  Ejemplos multietiqueta: {multiple}')
    print(f'  Textos repetidos conservados: {repeated}')
    print('  Frecuencia de etiquetas:')
    for topic, count in label_counts.most_common():
        print(f'    - {topic}: {count}')


def build_messages(
    text: str,
    topics: Sequence[str],
    catalog_topics: Sequence[str],
) -> list[dict[str, str]]:
    allowed = '\n'.join(
        f'- {topic}' for topic in (*catalog_topics, NONE_TOPIC)
    )
    user_prompt = f'Temas permitidos:\n{allowed}\n\nNoticia:\n{text}'
    answer = json.dumps(list(topics), ensure_ascii=False, separators=(',', ':'))
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
        {'role': 'assistant', 'content': answer},
    ]


def apply_chat_template(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
        return_dict=False,
    )
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise RuntimeError('El chat template no devolvio una lista de token IDs.')
    return token_ids


def encode_example(
    tokenizer: Any,
    example: TrainingExample,
    catalog_topics: Sequence[str],
    max_length: int,
) -> dict[str, list[int]]:
    '''Tokeniza preservando siempre la respuesta con noticias largas.'''

    text_token_ids = tokenizer.encode(example.text, add_special_tokens=False)

    def render(number_of_text_tokens: int) -> tuple[list[int], list[int]]:
        shortened_text = tokenizer.decode(
            text_token_ids[:number_of_text_tokens],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        messages = build_messages(shortened_text, example.topics, catalog_topics)
        prompt_ids = apply_chat_template(
            tokenizer, messages[:-1], add_generation_prompt=True
        )
        full_ids = apply_chat_template(
            tokenizer, messages, add_generation_prompt=False
        )
        return prompt_ids, full_ids

    prompt_ids, full_ids = render(len(text_token_ids))
    if len(full_ids) > max_length:
        low = 0
        high = len(text_token_ids)
        while low < high:
            middle = (low + high + 1) // 2
            _, candidate_ids = render(middle)
            if len(candidate_ids) <= max_length:
                low = middle
            else:
                high = middle - 1
        prompt_ids, full_ids = render(low)

    if len(full_ids) > max_length:
        raise SystemExit(
            f'--max-length={max_length} es insuficiente incluso sin la noticia '
            f'({example.source_file}, elemento {example.source_index}).'
        )
    if len(prompt_ids) >= len(full_ids):
        raise RuntimeError('La plantilla no genero tokens de respuesta.')
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            'Los tokens del prompt no son prefijo de la conversacion completa; '
            'no es posible enmascarar la perdida de forma segura.'
        )

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {
        'input_ids': full_ids,
        'attention_mask': [1] * len(full_ids),
        'labels': labels,
    }


def resolve_precision(torch: Any, precision: str) -> tuple[Any, bool, bool]:
    has_cuda = torch.cuda.is_available()
    if precision == 'auto':
        if has_cuda and torch.cuda.is_bf16_supported():
            precision = 'bf16'
        elif has_cuda:
            precision = 'fp16'
        else:
            precision = 'fp32'

    if precision == 'bf16':
        if not has_cuda or not torch.cuda.is_bf16_supported():
            raise SystemExit('La GPU disponible no admite precision bf16.')
        return torch.bfloat16, True, False
    if precision == 'fp16':
        if not has_cuda:
            raise SystemExit('La precision fp16 requiere una GPU CUDA.')
        return torch.float16, False, True
    return torch.float32, False, False


def import_training_dependencies() -> tuple[Any, ...]:
    try:
        import torch
        import transformers
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from torch.nn.utils.rnn import pad_sequence
        from transformers import (
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            'Faltan dependencias. Consulta los comandos de instalacion '
            'incluidos al principio del script.'
        ) from exc

    return (
        torch,
        transformers,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        pad_sequence,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )


def select_model_class(transformers: Any) -> Any:
    for class_name in (
        'Qwen3_5ForConditionalGeneration',
        'AutoModelForMultimodalLM',
        'AutoModelForImageTextToText',
    ):
        model_class = getattr(transformers, class_name, None)
        if model_class is not None:
            return model_class
    raise SystemExit(
        'Transformers no incluye Qwen3.5. Instala la rama main como indica '
        'el encabezado del script.'
    )


def run_training(
    args: argparse.Namespace,
    examples: Sequence[TrainingExample],
    catalog_topics: Sequence[str],
    input_paths: Sequence[Path],
) -> None:
    (
        torch,
        transformers,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        pad_sequence,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    ) = import_training_dependencies()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.load_in_4bit and not torch.cuda.is_available():
        raise SystemExit(
            'El entrenamiento de 4 bits requiere GPU CUDA. Usa '
            '--no-load-in-4bit para desactivarlo.'
        )

    compute_dtype, use_bf16, use_fp16 = resolve_precision(
        torch, args.precision
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.chat_template:
        raise SystemExit('El tokenizer no contiene el chat template oficial.')
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise SystemExit('El tokenizer no define pad_token ni eos_token.')
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    encoded_examples = [
        encode_example(tokenizer, example, catalog_topics, args.max_length)
        for example in examples
    ]
    token_lengths = [len(example['input_ids']) for example in encoded_examples]
    print(
        'Longitud tokenizada: '
        f'min={min(token_lengths)}, '
        f'media={sum(token_lengths) / len(token_lengths):.1f}, '
        f'max={max(token_lengths)}'
    )

    class TokenizedDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(encoded_examples)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return encoded_examples[index]

    class CompletionOnlyCollator:
        def __call__(
            self, features: Sequence[dict[str, list[int]]]
        ) -> dict[str, Any]:
            input_ids = [
                torch.tensor(item['input_ids'], dtype=torch.long)
                for item in features
            ]
            masks = [
                torch.tensor(item['attention_mask'], dtype=torch.long)
                for item in features
            ]
            labels = [
                torch.tensor(item['labels'], dtype=torch.long)
                for item in features
            ]
            return {
                'input_ids': pad_sequence(
                    input_ids, batch_first=True,
                    padding_value=tokenizer.pad_token_id,
                ),
                'attention_mask': pad_sequence(
                    masks, batch_first=True, padding_value=0
                ),
                'labels': pad_sequence(
                    labels, batch_first=True, padding_value=-100
                ),
            }

    model_kwargs: dict[str, Any] = {'dtype': compute_dtype}
    if args.load_in_4bit:
        model_kwargs['device_map'] = {'': torch.cuda.current_device()}
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model_class = select_model_class(transformers)
    print(f'Cargando {args.model_name} con {model_class.__name__}...')
    model = model_class.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    available_suffixes = {
        name.rsplit('.', 1)[-1] for name, _ in model.named_modules()
    }
    lora_targets = [
        target for target in DEFAULT_LORA_TARGETS
        if target in available_suffixes
    ]
    if not lora_targets:
        raise SystemExit(
            'No se encontraron capas compatibles con los objetivos LoRA '
            'esperados para Qwen3.5.'
        )
    missing_targets = sorted(set(DEFAULT_LORA_TARGETS) - set(lora_targets))
    if missing_targets:
        print('Aviso: objetivos LoRA ausentes: ' + ', '.join(missing_targets))
    print('Objetivos LoRA: ' + ', '.join(lora_targets))

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias='none',
        target_modules=lora_targets,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    if args.gradient_checkpointing and not args.load_in_4bit:
        enable_input_grads = getattr(model, 'enable_input_require_grads', None)
        if enable_input_grads is not None:
            enable_input_grads()

    use_tf32 = False
    if torch.cuda.is_available():
        major_compute_capability, _ = torch.cuda.get_device_capability()
        use_tf32 = major_compute_capability >= 8

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        do_train=True,
        do_eval=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # En Transformers actual, un float menor que 1 en warmup_steps se
        # interpreta como proporcion del total de pasos.
        warmup_steps=args.warmup_ratio,
        lr_scheduler_type='cosine',
        logging_strategy='steps',
        logging_steps=args.logging_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim='paged_adamw_8bit' if args.load_in_4bit else 'adamw_torch',
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=use_tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        report_to='none',
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenizedDataset(),
        data_collator=CompletionOnlyCollator(),
        processing_class=tokenizer,
    )

    print('Iniciando entrenamiento sin validacion ni test...')
    train_result = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    trainer.save_state()
    trainer.log_metrics('train', train_result.metrics)
    trainer.save_metrics('train', train_result.metrics)

    manifest = {
        'base_model': args.model_name,
        'training_files': [str(path.resolve()) for path in input_paths],
        'number_of_examples': len(examples),
        'catalog_topics': list(catalog_topics),
        'none_topic': NONE_TOPIC,
        'output_format': 'JSON array of topic strings',
        'validation_used': False,
        'test_used': False,
        'arguments': {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        'lora_targets': lora_targets,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'training_manifest.json').open(
        'w', encoding='utf-8'
    ) as file_handle:
        json.dump(manifest, file_handle, ensure_ascii=False, indent=2)
        file_handle.write('\n')

    print(f'Adaptador LoRA y tokenizer guardados en: {args.output_dir}')


def main() -> None:
    args = parse_args()
    catalog_topics = load_topic_catalog(args.topics_file)
    all_examples, input_paths = load_training_examples(
        args.input_dir, catalog_topics
    )
    examples = sample_examples_per_topic(
        all_examples,
        catalog_topics,
        args.examples_per_topic,
        args.seed,
    )
    print_dataset_summary(examples, input_paths)
    if args.dry_run:
        print('Dry run completado: no se ha cargado ni entrenado ningun modelo.')
        return
    run_training(args, examples, catalog_topics, input_paths)


if __name__ == '__main__':
    main()
