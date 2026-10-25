#!/usr/bin/env python3
"""Limpia publicaciones de X sin realizar ningún etiquetado temático."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_DIR / 'raw_data'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'clean_data'
CLEANING_VERSION = 'raw_x_cleaning_v2_no_emojis'

# Bloques Unicode usados por emojis, pictogramas, banderas y otros iconos
# visuales habituales. También se eliminan sus componentes de presentación
# para no dejar restos de secuencias compuestas.
EMOJI_AND_ICON_RANGES = (
    (0x2190, 0x21FF),  # Flechas
    (0x2300, 0x23FF),  # Símbolos técnicos diversos
    (0x2460, 0x24FF),  # Caracteres alfanuméricos encerrados
    (0x25A0, 0x27BF),  # Formas, símbolos y dingbats
    (0x2900, 0x297F),  # Flechas suplementarias
    (0x2B00, 0x2BFF),  # Símbolos y flechas diversos
    (0x1F000, 0x1FAFF),  # Emojis, pictogramas, banderas y modificadores
)
EMOJI_AND_ICON_SINGLETONS = frozenset(
    {
        0x00A9,  # ©
        0x00AE,  # ®
        0x203C,  # ‼
        0x2049,  # ⁉
        0x20E3,  # Combinación de tecla encerrada
        0x2122,  # ™
        0x2139,  # ℹ
        0x3030,
        0x303D,
        0x3297,
        0x3299,
        0xFE0E,  # Selector de presentación como texto
        0xFE0F,  # Selector de presentación como emoji
    }
)

STATUS_URL_PATTERN = re.compile(
    r'https?://(?:www\.)?(?:x\.com|twitter\.com)/'
    r'[A-Za-z0-9_]{1,15}/status/\d+(?:[/?#].*)?',
    re.IGNORECASE,
)
ALPHA_WORD_PATTERN = re.compile(r'[^\W\d_]+', re.UNICODE)
SHOW_MORE_PATTERN = re.compile(
    r'\b(?:mostrar\s+m[aá]s|show\s+more)\s*[.…]*$', re.IGNORECASE
)
TRAILING_ELLIPSIS_PATTERN = re.compile(
    r'(?:\.{3}|…)(?:["»”’\)\]]*)$', re.UNICODE
)
TRAILING_DISPLAY_URL_PATTERN = re.compile(
    r'(?:https?://|www\.|\b[\w-]+\.[a-z]{2,}/)'
    r'\S*(?:\.{3}|…)(?:["»”’\)\]]*)$',
    re.IGNORECASE,
)
TERMINAL_PATTERN = re.compile(r'[.!?](?:["»”’\)\]]*)$')
SOCIAL_OR_URL_END_PATTERN = re.compile(
    r'(?:@[A-Za-z0-9_]+|#[\w]+|https?://\S+|www\.\S+|'
    r'\b[\w-]+\.[a-z]{2,}/\S*)$',
    re.IGNORECASE,
)

INCOMPLETE_FINAL_WORDS = frozenset(
    {
        'a',
        'al',
        'ante',
        'bajo',
        'con',
        'contra',
        'de',
        'del',
        'desde',
        'durante',
        'e',
        'el',
        'en',
        'entre',
        'hacia',
        'hasta',
        'la',
        'las',
        'los',
        'o',
        'para',
        'por',
        'que',
        'según',
        'sin',
        'sobre',
        'su',
        'sus',
        'tras',
        'u',
        'un',
        'una',
        'unos',
        'unas',
        'y',
    }
)

PROMOTIONAL_PATTERNS = (
    (
        'subscription',
        re.compile(
            r'\b(?:suscr[ií]bete|hazte\s+soci[oa]|hazte\s+miembro|'
            r'activa\s+las\s+notificaciones)\b',
            re.IGNORECASE,
        ),
    ),
    (
        'cover',
        re.compile(
            r'\b(?:consulta|consultar|lee|leer|ya\s+puedes\s+consultar)\b'
            r'.{0,50}\bportada\b|'
            r'\b(?:esta\s+es\s+)?la\s+portada\s+(?:de|del)\s+'
            r'(?:hoy|mañana)\b|'
            r'\bla\s+portada\s+y\s+la\s+dos\b',
            re.IGNORECASE,
        ),
    ),
    (
        'follow_on_social_media',
        re.compile(
            r'\b(?:s[ií]guenos|encu[eé]ntranos)\s+en\s+'
            r'(?:nuestras?|otras?)\s+redes\b',
            re.IGNORECASE,
        ),
    ),
    (
        'application_download',
        re.compile(
            r'\bdescarga\s+(?:ya\s+)?(?:nuestra|la)\s+app\b',
            re.IGNORECASE,
        ),
    ),
    (
        'newsletter',
        re.compile(
            r'\b(?:ap[uú]ntate|suscr[ií]bete)\b.{0,80}'
            r'\b(?:newsletter|bolet[ií]n)\b|'
            r'\bnewsletter\b.{0,80}\b(?:recibe|correo|suscr[ií]bete)\b',
            re.IGNORECASE,
        ),
    ),
    (
        'programming_announcement',
        re.compile(
            r'^\s*(?:📺|📻)?\s*(?:hoy|esta\s+noche|mañana)\s+en\s+'
            r'(?:la\s+[12]|rtve|rne|24\s+horas)\b|'
            r'\bno\s+te\s+pierdas\b.{0,100}\b(?:a\s+las|'
            r'\d{1,2}:\d{2}\s*h?)\b',
            re.IGNORECASE,
        ),
    ),
    (
        'affiliate_offer',
        re.compile(
            r'\bamazon\b.{0,120}\b(?:ofertas?|chollos?|descuentazos?)\b|'
            r'\b(?:ofertas?|chollos?|descuentazos?)\b.{0,120}\bamazon\b',
            re.IGNORECASE,
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Limpia los JSON de raw_data sin asignar categorías ni modificar '
            'los archivos originales.'
        )
    )
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open('r', encoding='utf-8') as file_handle:
            payload = json.load(file_handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f'JSON no válido en {path}: {exc}') from exc
    if not isinstance(payload, list):
        raise SystemExit(f'{path} debe contener una lista JSON.')
    records: list[dict[str, Any]] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise SystemExit(
                f'{path}, elemento {index}: se esperaba un objeto JSON.'
            )
        records.append(record)
    return records


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)
        file_handle.write('\n')
    temporary_path.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def is_emoji_or_icon(character: str) -> bool:
    codepoint = ord(character)
    return codepoint in EMOJI_AND_ICON_SINGLETONS or any(
        start <= codepoint <= end for start, end in EMOJI_AND_ICON_RANGES
    )


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize('NFC', html.unescape(value))
    characters: list[str] = []
    for character in normalized:
        if is_emoji_or_icon(character):
            continue
        if character.isspace():
            characters.append(' ')
        elif unicodedata.category(character) in {'Cc', 'Cf'}:
            continue
        else:
            characters.append(character)
    return re.sub(r'\s+', ' ', ''.join(characters)).strip()


def alpha_words(text: str) -> list[str]:
    return ALPHA_WORD_PATTERN.findall(text)


def truncation_reason(text: str) -> str | None:
    if SHOW_MORE_PATTERN.search(text):
        return 'truncated_show_more'
    if (
        TRAILING_ELLIPSIS_PATTERN.search(text)
        and not TRAILING_DISPLAY_URL_PATTERN.search(text)
    ):
        return 'truncated_ellipsis'

    has_terminal = bool(TERMINAL_PATTERN.search(text)) or text.endswith(
        ('"', '»', '”', '’', ')', ']')
    )
    ends_with_social_or_url = bool(SOCIAL_OR_URL_END_PATTERN.search(text))
    words = [word.casefold() for word in alpha_words(text)]
    if (
        len(text) >= 240
        and words
        and words[-1] in INCOMPLETE_FINAL_WORDS
        and not has_terminal
        and not ends_with_social_or_url
    ):
        return 'truncated_incomplete_ending'
    return None


def promotional_reason(text: str) -> str | None:
    for reason, pattern in PROMOTIONAL_PATTERNS:
        if pattern.search(text):
            return f'promotional_{reason}'
    return None


def initial_rejection_reasons(record: dict[str, Any], text: str) -> list[str]:
    reasons: list[str] = []
    news_id = record.get('_id')
    if not isinstance(news_id, str) or not news_id.strip():
        reasons.append('missing_or_invalid_id')

    tweet_url = record.get('tweet_url')
    if (
        not isinstance(tweet_url, str)
        or STATUS_URL_PATTERN.fullmatch(tweet_url.strip()) is None
    ):
        reasons.append('missing_or_invalid_tweet_url')

    words = alpha_words(text)
    if not text:
        reasons.append('empty_text')
    elif len(words) < 2:
        reasons.append('insufficient_informative_text')

    truncated = truncation_reason(text)
    if truncated is not None:
        reasons.append(truncated)

    promotional = promotional_reason(text)
    if promotional is not None:
        reasons.append(promotional)
    return reasons


def rejected_record(
    record: dict[str, Any], reasons: list[str], normalized_text: str
) -> dict[str, Any]:
    rejected = dict(record)
    rejected['_cleaning'] = {
        'reasons': reasons,
        'normalized_text': normalized_text,
    }
    return rejected


def clean_files(
    input_paths: list[Path], output_dir: Path
) -> dict[str, Any]:
    if not input_paths:
        raise SystemExit('No se encontraron archivos raw_data.*.json.')

    source_payloads: dict[Path, list[dict[str, Any]]] = {}
    source_candidates: dict[Path, list[tuple[dict[str, Any], str]]] = {}
    rejected_by_source: dict[Path, list[dict[str, Any]]] = {}
    input_hashes: dict[str, str] = {}

    for path in input_paths:
        records = load_records(path)
        source_payloads[path] = records
        source_candidates[path] = []
        rejected_by_source[path] = []
        input_hashes[path.name] = sha256_file(path)
        for record in records:
            raw_text = record.get('text')
            normalized_text = (
                normalize_text(raw_text) if isinstance(raw_text, str) else ''
            )
            reasons = initial_rejection_reasons(record, normalized_text)
            if reasons:
                rejected_by_source[path].append(
                    rejected_record(record, reasons, normalized_text)
                )
            else:
                source_candidates[path].append((record, normalized_text))

    seen_ids: dict[str, str] = {}
    seen_urls: dict[str, str] = {}
    seen_texts: dict[str, str] = {}
    cleaned_by_source: dict[Path, list[dict[str, Any]]] = {}

    for path in input_paths:
        cleaned_by_source[path] = []
        for record, normalized_text in source_candidates[path]:
            news_id = str(record['_id']).strip()
            tweet_url = str(record['tweet_url']).strip()
            text_key = normalized_text.casefold()
            reasons: list[str] = []
            duplicate_of: dict[str, str] = {}
            if news_id in seen_ids:
                reasons.append('duplicate_id')
                duplicate_of['_id'] = seen_ids[news_id]
            if tweet_url in seen_urls:
                reasons.append('duplicate_tweet_url')
                duplicate_of['tweet_url'] = seen_urls[tweet_url]
            if text_key in seen_texts:
                reasons.append('duplicate_normalized_text')
                duplicate_of['text'] = seen_texts[text_key]

            if reasons:
                rejected = rejected_record(record, reasons, normalized_text)
                rejected['_cleaning']['duplicate_of'] = duplicate_of
                rejected_by_source[path].append(rejected)
                continue

            cleaned = dict(record)
            cleaned['_id'] = news_id
            cleaned['tweet_url'] = tweet_url
            cleaned['text'] = normalized_text
            cleaned_by_source[path].append(cleaned)
            seen_ids[news_id] = news_id
            seen_urls[tweet_url] = news_id
            seen_texts[text_key] = news_id

    output_files: list[str] = []
    rejected_files: list[str] = []
    file_reports: dict[str, dict[str, Any]] = {}
    total_reasons: Counter[str] = Counter()
    total_input = total_cleaned = total_rejected = 0

    for path in input_paths:
        source_name = path.name.removeprefix('raw_data.')
        clean_path = output_dir / f'clean_data.{source_name}'
        reject_path = (
            output_dir / 'rejected' / f'rejected_data.{source_name}'
        )
        clean_records = cleaned_by_source[path]
        reject_records = rejected_by_source[path]
        save_json(clean_path, clean_records)
        save_json(reject_path, reject_records)
        output_files.append(str(clean_path.resolve()))
        rejected_files.append(str(reject_path.resolve()))

        reason_counts = Counter(
            reason
            for record in reject_records
            for reason in record['_cleaning']['reasons']
        )
        total_reasons.update(reason_counts)
        input_count = len(source_payloads[path])
        total_input += input_count
        total_cleaned += len(clean_records)
        total_rejected += len(reject_records)
        file_reports[path.name] = {
            'input_records': input_count,
            'cleaned_records': len(clean_records),
            'rejected_records': len(reject_records),
            'reasons': dict(sorted(reason_counts.items())),
            'input_sha256': input_hashes[path.name],
            'output_file': str(clean_path.resolve()),
            'rejected_file': str(reject_path.resolve()),
        }

    report = {
        'cleaning_version': CLEANING_VERSION,
        'labels_or_classifiers_used': False,
        'input_dir': str(input_paths[0].parent.resolve()),
        'output_dir': str(output_dir.resolve()),
        'criteria': {
            'requires_valid_x_status_url': True,
            'minimum_alphabetic_words': 2,
            'unicode_normalization': 'NFC',
            'html_entities_decoded': True,
            'whitespace_collapsed': True,
            'control_and_format_characters_removed': True,
            'case_accents_numbers_hashtags_and_non_icon_punctuation_preserved': (
                True
            ),
            'emojis_pictograms_and_icons_removed': True,
            'truncated_texts_removed': True,
            'promotional_texts_removed': True,
            'duplicates_removed_globally': (
                '_id, tweet_url and whitespace/case-normalized text'
            ),
            'published_at_modified': False,
            'account_modified': False,
        },
        'totals': {
            'input_records': total_input,
            'cleaned_records': total_cleaned,
            'rejected_records': total_rejected,
            'rejection_reasons': dict(sorted(total_reasons.items())),
        },
        'files': file_reports,
        'output_files': output_files,
        'rejected_files': rejected_files,
    }
    save_json(output_dir / 'audit' / 'cleaning_report.json', report)
    return report


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f'No se encontró el directorio: {args.input_dir}')
    input_paths = sorted(args.input_dir.glob('raw_data.*.json'))
    report = clean_files(input_paths, args.output_dir)
    totals = report['totals']
    print(f"Registros de entrada: {totals['input_records']}")
    print(f"Registros conservados: {totals['cleaned_records']}")
    print(f"Registros eliminados: {totals['rejected_records']}")
    print(
        'Informe: '
        f"{(args.output_dir / 'audit' / 'cleaning_report.json').resolve()}"
    )


if __name__ == '__main__':
    main()
