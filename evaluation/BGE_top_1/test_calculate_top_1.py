from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import calculate_top_1 as subject


class FakeScorer:
    def __init__(self, scores: Any) -> None:
        self.scores = scores
        self.received_pairs: list[list[str]] | None = None

    def __call__(self, pairs: list[list[str]]) -> Any:
        self.received_pairs = pairs
        return self.scores


class BgeTop1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.topics = {
            'tema-a': 'Definición A',
            'tema-b': 'Definición B',
            'tema-c': 'Definición C',
        }
        self.records = [
            {
                '_id': 'news-1',
                'file': 'source.json',
                'text': 'Texto de prueba',
            }
        ]
        self.truth = {
            'news-1': {'text': 'Texto de prueba', 'topics': ['tema-b']}
        }

    def test_scores_every_topic_and_selects_the_highest_score(self) -> None:
        scorer = FakeScorer([0.2, 0.9, 0.4])

        predictions, elapsed = subject.evaluate_records(
            self.records,
            self.truth,
            self.topics,
            scorer,
        )

        self.assertEqual(
            scorer.received_pairs,
            [
                [
                    'Texto de prueba',
                    'Tema: tema-a. Definición: Definición A',
                ],
                [
                    'Texto de prueba',
                    'Tema: tema-b. Definición: Definición B',
                ],
                [
                    'Texto de prueba',
                    'Tema: tema-c. Definición: Definición C',
                ],
            ],
        )
        self.assertEqual(predictions[0]['topic_predicho'], 'tema-b')
        self.assertTrue(predictions[0]['match'])
        self.assertEqual(len(predictions[0]['topic_scores']), 3)
        self.assertEqual(len(elapsed), 1)

        report = subject.build_report(
            records=self.records,
            truth_by_id=self.truth,
            topics=self.topics,
            predictions=predictions,
            inference_times=elapsed,
            model_name=subject.MODEL_NAME,
            batch_size=subject.BATCH_SIZE,
            max_length=subject.MAX_LENGTH,
            device='cpu',
            data_dir=Path('data_test'),
        )
        self.assertFalse(report['configuration']['uses_e5_candidates'])
        self.assertFalse(report['configuration']['uses_precomputed_results'])
        self.assertEqual(report['configuration']['candidate_count'], 3)
        self.assertEqual(
            report['configuration']['topic_representation'],
            subject.TOPIC_TEMPLATE,
        )
        self.assertEqual(report['metrics']['top_1_accuracy'], 1.0)

    def test_rejects_an_incorrect_number_of_scores(self) -> None:
        scorer = FakeScorer([0.2, 0.9])

        with self.assertRaisesRegex(RuntimeError, 'número de scores'):
            subject.evaluate_records(
                self.records,
                self.truth,
                self.topics,
                scorer,
            )

    def test_rejects_non_finite_scores(self) -> None:
        scorer = FakeScorer([0.2, float('nan'), 0.4])

        with self.assertRaisesRegex(RuntimeError, 'no finitos'):
            subject.evaluate_records(
                self.records,
                self.truth,
                self.topics,
                scorer,
            )

    def test_rejects_text_that_differs_from_ground_truth(self) -> None:
        self.records[0]['text'] = 'Otro texto'

        with self.assertRaisesRegex(SystemExit, 'no coincide'):
            subject.validate_dataset(self.records, self.truth)

    def test_rejects_incomplete_id_set(self) -> None:
        self.truth['news-2'] = {'text': 'Segundo texto', 'topics': ['tema-a']}

        with self.assertRaisesRegex(SystemExit, '1 ausentes'):
            subject.validate_dataset(self.records, self.truth)

    def test_loads_only_a_complete_matching_top_1_report(self) -> None:
        predictions, elapsed = subject.evaluate_records(
            self.records,
            self.truth,
            self.topics,
            FakeScorer([0.2, 0.9, 0.4]),
        )
        report = subject.build_report(
            records=self.records,
            truth_by_id=self.truth,
            topics=self.topics,
            predictions=predictions,
            inference_times=elapsed,
            model_name=subject.MODEL_NAME,
            batch_size=subject.BATCH_SIZE,
            max_length=subject.MAX_LENGTH,
            device='cuda:0',
            data_dir=Path('data_test'),
        )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'bge_top_1_results.json'
            subject.save_json(path, report)
            loaded, loaded_times, device = subject.load_validated_predictions(
                path,
                self.records,
                self.truth,
                self.topics,
                subject.MODEL_NAME,
                subject.BATCH_SIZE,
                subject.MAX_LENGTH,
            )

            self.assertEqual(loaded, predictions)
            self.assertEqual(
                loaded_times,
                [predictions[0]['inference_time_seconds']],
            )
            self.assertEqual(device, 'cuda:0')

            report['configuration']['input_data_sha256'] = 'otro-dataset'
            subject.save_json(path, report)
            with self.assertRaisesRegex(SystemExit, 'input_data_sha256'):
                subject.load_validated_predictions(
                    path,
                    self.records,
                    self.truth,
                    self.topics,
                    subject.MODEL_NAME,
                    subject.BATCH_SIZE,
                    subject.MAX_LENGTH,
                )


if __name__ == '__main__':
    unittest.main()
