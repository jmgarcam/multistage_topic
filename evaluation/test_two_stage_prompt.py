from __future__ import annotations

import json
import unittest

import evaluate_ollama_top_n_no_none as subject


class FakeOllamaClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[dict[str, str]], dict, int]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        response_format: dict,
        max_new_tokens: int = subject.MAX_NEW_TOKENS,
    ) -> str:
        self.calls.append((messages, response_format, max_new_tokens))
        return self.responses.pop(0)


class TwoStagePromptTests(unittest.TestCase):
    def test_sentence_segmentation_preserves_numbered_source_text(self) -> None:
        text = 'Primera frase. «Segunda pregunta?»\nTercera frase'

        sentences = subject.segment_evidence_sentences(text)

        self.assertEqual(
            sentences,
            [
                ('F1', 'Primera frase.'),
                ('F2', '«Segunda pregunta?»'),
                ('F3', 'Tercera frase'),
            ],
        )

    def test_event_extraction_resolves_a_valid_sentence_identifier(self) -> None:
        text = 'El equipo ganó la final. La afición celebró la victoria.'
        valid = json.dumps(
            {
                'acontecimiento_principal': 'El equipo ganó la final.',
                'frase_evidencia_id': 'F1',
            }
        )
        unknown_identifier = json.dumps(
            {
                'acontecimiento_principal': 'El equipo ganó la final.',
                'frase_evidencia_id': 'F3',
            }
        )

        extraction = subject.validate_event_extraction(valid, text)

        self.assertEqual(
            extraction,
            {
                subject.MAIN_EVENT_FIELD: 'El equipo ganó la final.',
                subject.EVIDENCE_SENTENCE_ID_FIELD: 'F1',
                subject.EVIDENCE_SENTENCE_FIELD: (
                    'El equipo ganó la final.'
                ),
            },
        )
        self.assertIsNone(
            subject.validate_event_extraction(unknown_identifier, text)
        )

    def test_two_stage_decision_hides_candidates_from_first_prompt(self) -> None:
        text = 'El equipo ganó la final. La afición celebró la victoria.'
        extraction = json.dumps(
            {
                'acontecimiento_principal': 'El equipo ganó la final.',
                'frase_evidencia_id': 'F1',
            }
        )
        classification = json.dumps({'tema': 'deportes'})
        client = FakeOllamaClient([extraction, classification])
        candidates = [{'tema': 'deportes'}, {'tema': 'economia'}]
        topics = {
            'deportes': 'Actividad deportiva.',
            'economia': 'Actividad económica.',
        }

        decision, event, event_raw, raw = (
            subject.generate_two_stage_decision(
                client, text, candidates, topics
            )
        )

        self.assertEqual(decision, 'deportes')
        self.assertEqual(event_raw, extraction)
        self.assertEqual(raw, classification)
        self.assertEqual(
            event[subject.MAIN_EVENT_FIELD], 'El equipo ganó la final.'
        )
        self.assertEqual(event[subject.EVIDENCE_SENTENCE_ID_FIELD], 'F1')
        self.assertEqual(
            event[subject.EVIDENCE_SENTENCE_FIELD],
            'El equipo ganó la final.',
        )
        self.assertEqual(len(client.calls), 2)
        first_prompt = client.calls[0][0][1]['content']
        second_prompt = client.calls[1][0][1]['content']
        self.assertNotIn('deportes', first_prompt)
        self.assertNotIn('economia', first_prompt)
        self.assertIn('deportes', second_prompt)
        self.assertIn('economia', second_prompt)
        self.assertIn('F1', second_prompt)
        self.assertIn('El equipo ganó la final.', second_prompt)
        extraction_schema = client.calls[0][1]
        self.assertEqual(
            extraction_schema['properties']['frase_evidencia_id']['enum'],
            ['F1', 'F2'],
        )

    def test_invalid_extraction_skips_classification(self) -> None:
        extraction = json.dumps(
            {
                'acontecimiento_principal': 'Acontecimiento inventado.',
                'frase_evidencia_id': 'F99',
            }
        )
        client = FakeOllamaClient([extraction])

        decision, event, event_raw, raw = (
            subject.generate_two_stage_decision(
                client,
                'Texto real de la noticia.',
                [{'tema': 'deportes'}],
                {'deportes': 'Actividad deportiva.'},
            )
        )

        self.assertEqual(decision, 'invalid')
        self.assertIsNone(event)
        self.assertEqual(event_raw, extraction)
        self.assertIsNone(raw)
        self.assertEqual(len(client.calls), 1)


if __name__ == '__main__':
    unittest.main()
