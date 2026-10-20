from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import calculate_top_1 as subject


class E5Top1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {'_id': 'news-1', 'file': 'test.json', 'text': 'Texto uno'}
        ]
        self.labelled = {
            'news-1': {'text': 'Texto uno', 'topics': ['tema-b']}
        }
        self.topics = {
            'tema-a': 'Definición A',
            'tema-b': 'Definición B',
            'tema-c': 'Definición C',
        }

    def test_selects_the_highest_e5_score(self) -> None:
        rankings = subject.e5.rankings_from_similarity_rows(
            self.records,
            self.topics,
            [[0.2, 0.9, 0.4]],
        )
        report = subject.build_report(
            records=self.records,
            labelled_by_id=self.labelled,
            topics=self.topics,
            rankings=rankings,
            model_name=subject.e5.DEFAULT_MODEL,
            batch_size=subject.e5.DEFAULT_BATCH_SIZE,
            device='cpu',
            timing={},
            data_dir=Path('data_test'),
            labelled_data_dir=Path('data_test_labelled'),
        )

        prediction = report['predictions'][0]
        self.assertEqual(prediction['topic_predicho'], 'tema-b')
        self.assertEqual(prediction['similarity'], 0.9)
        self.assertEqual(len(prediction['topic_scores']), 3)
        self.assertTrue(prediction['match'])
        configuration = report['configuration']
        self.assertEqual(configuration['architecture'], 'bi-encoder')
        self.assertFalse(configuration['uses_precomputed_results'])
        self.assertFalse(configuration['uses_bge'])
        self.assertFalse(configuration['uses_llama'])

    def test_ties_follow_topic_catalog_order(self) -> None:
        rankings = subject.e5.rankings_from_similarity_rows(
            self.records,
            self.topics,
            [[0.8, 0.8, 0.1]],
        )

        names = [
            candidate['tema'] for candidate in rankings[0]['candidates']
        ]
        self.assertEqual(names, ['tema-a', 'tema-b', 'tema-c'])

    def test_rejects_mismatched_labelled_text(self) -> None:
        self.labelled['news-1']['text'] = 'Otro texto'

        with self.assertRaisesRegex(SystemExit, 'no coincide'):
            subject.e5.validate_dataset(self.records, self.labelled)

    def test_loads_only_a_complete_matching_e5_report(self) -> None:
        rankings = subject.e5.rankings_from_similarity_rows(
            self.records,
            self.topics,
            [[0.2, 0.9, 0.4]],
        )
        report = subject.build_report(
            records=self.records,
            labelled_by_id=self.labelled,
            topics=self.topics,
            rankings=rankings,
            model_name=subject.e5.DEFAULT_MODEL,
            batch_size=subject.e5.DEFAULT_BATCH_SIZE,
            device='cuda',
            timing={'total_e5_execution_time_seconds': 1.0},
            data_dir=Path('data_test'),
            labelled_data_dir=Path('data_test_labelled'),
        )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'e5_results.json'
            subject.e5.save_json(path, report)
            loaded, runtime = subject.e5.load_validated_rankings(
                path,
                self.records,
                self.labelled,
                self.topics,
                subject.e5.DEFAULT_MODEL,
                subject.e5.DEFAULT_BATCH_SIZE,
            )

            self.assertEqual(loaded, rankings)
            self.assertEqual(runtime['device'], 'cuda')
            self.assertEqual(
                runtime['timing']['total_e5_execution_time_seconds'], 1.0
            )

            report['configuration']['topic_catalog_sha256'] = 'otro-catálogo'
            subject.e5.save_json(path, report)
            with self.assertRaisesRegex(SystemExit, 'topic_catalog_sha256'):
                subject.e5.load_validated_rankings(
                    path,
                    self.records,
                    self.labelled,
                    self.topics,
                    subject.e5.DEFAULT_MODEL,
                    subject.e5.DEFAULT_BATCH_SIZE,
                )


if __name__ == '__main__':
    unittest.main()
