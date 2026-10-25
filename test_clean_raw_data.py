from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clean_data import clean_raw_data as subject


class CleanRawDataTests(unittest.TestCase):
    def test_normalizes_text_without_losing_semantic_characters(self) -> None:
        text = '  ÁRBOL\u200b  #DANA\n🔥 &amp; lluvia  '

        normalized = subject.normalize_text(text)

        self.assertEqual(normalized, 'ÁRBOL #DANA & lluvia')

    def test_removes_emojis_icons_and_composed_sequences(self) -> None:
        text = 'Alerta 🚨 ✅ 👩🏽\u200d🚒 bandera 🇪🇸 y temperatura 30 °C.'

        normalized = subject.normalize_text(text)

        self.assertEqual(
            normalized,
            'Alerta bandera y temperatura 30 °C.',
        )

    def test_detects_explicit_and_length_truncation(self) -> None:
        self.assertEqual(
            subject.truncation_reason('Una noticia que continúa…'),
            'truncated_ellipsis',
        )
        self.assertEqual(
            subject.truncation_reason('Noticia completa. Mostrar más'),
            'truncated_show_more',
        )
        long_cut = ('Una noticia suficientemente larga ' * 8) + 'de'
        self.assertEqual(
            subject.truncation_reason(long_cut),
            'truncated_incomplete_ending',
        )

    def test_does_not_treat_a_shortened_display_url_as_truncated(self) -> None:
        text = 'Noticia completa eldiario.es/noticia/ejemplo…'

        self.assertIsNone(subject.truncation_reason(text))

    def test_detects_promotional_content(self) -> None:
        self.assertEqual(
            subject.promotional_reason(
                'Ya puedes consultar la portada de mañana.'
            ),
            'promotional_cover',
        )
        self.assertEqual(
            subject.promotional_reason('Suscríbete a nuestra newsletter.'),
            'promotional_subscription',
        )

    def test_keeps_account_and_published_at_unchanged(self) -> None:
        record = {
            '_id': 'one',
            'text': '  Noticia válida sobre una tormenta.  ',
            'tweet_url': 'https://x.com/medio/status/123',
            'account': 'CuentaOriginal',
            'published_at': None,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir) / 'raw_data.test.json'
            output = Path(temporary_dir) / 'output'
            subject.save_json(source, [record])

            subject.clean_files([source], output)
            cleaned = subject.load_records(output / 'clean_data.test.json')

        self.assertEqual(cleaned[0]['account'], 'CuentaOriginal')
        self.assertIsNone(cleaned[0]['published_at'])
        self.assertEqual(
            cleaned[0]['text'], 'Noticia válida sobre una tormenta.'
        )

    def test_removes_duplicate_normalized_text_globally(self) -> None:
        records = [
            {
                '_id': 'one',
                'text': 'Noticia   válida sobre SALUD.',
                'tweet_url': 'https://x.com/medio/status/1',
            },
            {
                '_id': 'two',
                'text': 'noticia válida sobre salud.',
                'tweet_url': 'https://x.com/medio/status/2',
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir) / 'raw_data.test.json'
            output = Path(temporary_dir) / 'output'
            subject.save_json(source, records)

            report = subject.clean_files([source], output)
            rejected = subject.load_records(
                output / 'rejected' / 'rejected_data.test.json'
            )

        self.assertEqual(report['totals']['cleaned_records'], 1)
        self.assertEqual(report['totals']['rejected_records'], 1)
        self.assertIn(
            'duplicate_normalized_text',
            rejected[0]['_cleaning']['reasons'],
        )


if __name__ == '__main__':
    unittest.main()
