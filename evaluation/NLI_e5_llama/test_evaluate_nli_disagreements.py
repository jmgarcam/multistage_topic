from __future__ import annotations

import unittest

import evaluate_nli_disagreements as subject


class NLIDisagreementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.topics = {
            'economia': 'Actividad económica y sectores productivos.',
            'politica': 'Gobiernos, partidos y decisiones políticas.',
            'ejercito': 'Fuerzas armadas, defensa y operaciones militares.',
        }

    def test_selects_only_e5_llama_disagreements(self) -> None:
        predictions = [
            {
                '_id': 'same',
                'e5_top_1_topic': 'economia',
                'final_decision': 'economia',
                'llama_called': True,
            },
            {
                '_id': 'different',
                'file': 'news.json',
                'e5_top_1_topic': 'economia',
                'final_decision': 'politica',
                'llama_called': True,
                'main_event': 'Una organización protesta ante un ministerio.',
                'evidence_sentence': 'Los agricultores se manifestaron.',
                'topics_correctos': ['economia'],
            },
        ]

        disagreements = subject.select_disagreements(
            predictions, self.topics
        )

        self.assertEqual(len(disagreements), 1)
        self.assertEqual(disagreements[0]['_id'], 'different')
        self.assertEqual(disagreements[0]['e5_topic'], 'economia')
        self.assertEqual(disagreements[0]['llama_topic'], 'politica')
        self.assertNotIn('topics_correctos', disagreements[0])

    def test_evaluation_stores_labels_without_scores_or_decision(self) -> None:
        disagreements = [
            {
                '_id': 'different',
                'file': 'news.json',
                'e5_topic': 'economia',
                'llama_topic': 'politica',
                'premise': 'Premisa',
                'e5_hypothesis': 'Hipótesis E5',
                'llama_hypothesis': 'Hipótesis Llama',
            }
        ]

        def fake_classifier(
            pairs: list[tuple[str, str]],
        ) -> list[str]:
            self.assertEqual(
                pairs,
                [
                    ('Premisa', 'Hipótesis E5'),
                    ('Premisa', 'Hipótesis Llama'),
                ],
            )
            return ['neutral', 'entailment']

        results, _ = subject.evaluate_disagreements(
            disagreements, fake_classifier
        )

        self.assertEqual(
            results[0]['nli_results'],
            {
                'e5_hypothesis': 'neutral',
                'llama_hypothesis': 'entailment',
            },
        )
        serialized_keys = set(results[0])
        self.assertNotIn('score', serialized_keys)
        self.assertNotIn('score_difference', serialized_keys)
        self.assertNotIn('nli_decision', serialized_keys)
        self.assertNotIn('final_decision', serialized_keys)

    def test_report_does_not_use_ground_truth_or_derive_metrics(self) -> None:
        source_report = {
            'status': 'complete',
            'configuration': {'n': 3},
            'predictions': [
                {'_id': 'different', 'llama_called': True},
            ],
        }
        results = [
            {
                '_id': 'different',
                'e5_topic': 'economia',
                'llama_topic': 'politica',
                'premise': 'Premisa',
                'e5_hypothesis': 'Hipótesis E5',
                'llama_hypothesis': 'Hipótesis Llama',
                'nli_results': {
                    'e5_hypothesis': 'neutral',
                    'llama_hypothesis': 'entailment',
                },
            }
        ]

        report = subject.build_report(
            source_report=source_report,
            source_path=subject.DEFAULT_SOURCE_RESULTS,
            topics=self.topics,
            results=results,
            model_name=subject.MODEL_NAME,
            batch_size=16,
            max_length=512,
            device='cpu',
            id2label={0: 'entailment', 1: 'neutral', 2: 'contradiction'},
            elapsed=0.1,
        )

        self.assertNotIn('metrics', report)
        self.assertFalse(report['configuration']['stores_nli_scores'])
        self.assertFalse(
            report['configuration']['computes_score_differences']
        )
        self.assertFalse(report['configuration']['derives_final_decision'])
        self.assertFalse(report['configuration']['uses_ground_truth'])
        self.assertEqual(report['dataset']['e5_llama_disagreements'], 1)
        self.assertEqual(report['dataset']['nli_pairs_evaluated'], 2)

    def test_normalizes_model_label_mapping(self) -> None:
        mapping = subject.normalize_id2label(
            {0: 'ENTAILMENT', 1: 'NEUTRAL', 2: 'CONTRADICTION'}
        )

        self.assertEqual(
            mapping,
            {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
        )


if __name__ == '__main__':
    unittest.main()
