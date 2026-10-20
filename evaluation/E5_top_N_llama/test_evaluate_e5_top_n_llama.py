from __future__ import annotations

import unittest

import evaluate_e5_top_n_llama as subject


class E5TopNLlamaTests(unittest.TestCase):
    def test_default_sweep_covers_every_possible_n(self) -> None:
        self.assertEqual(subject.resolve_n_values([], 4), [1, 2, 3, 4])

    def test_report_separates_retrieval_and_selection(self) -> None:
        topics = {'tema-a': 'A', 'tema-b': 'B'}
        records = [
            {
                '_id': 'news-1',
                'file': 'test.json',
                'text': 'Uno',
                'topics_correctos': ['tema-b'],
                'candidates': [
                    {'tema': 'tema-b', 'similarity': 0.9},
                    {'tema': 'tema-a', 'similarity': 0.1},
                ],
            },
            {
                '_id': 'news-2',
                'file': 'test.json',
                'text': 'Dos',
                'topics_correctos': ['tema-b'],
                'candidates': [
                    {'tema': 'tema-a', 'similarity': 0.8},
                    {'tema': 'tema-b', 'similarity': 0.2},
                ],
            },
        ]
        predictions = [
            {
                'matches_real_topic': True,
                'candidate_contains_real_topic': True,
                'inference_time_seconds': 0.1,
            },
            {
                'matches_real_topic': False,
                'candidate_contains_real_topic': False,
                'inference_time_seconds': 0.1,
            },
        ]
        context = {
            'model': subject.e5.DEFAULT_MODEL,
            'batch_size': 32,
            'device': 'cpu',
            'ranking_sha256': 'ranking',
            'data_dir': 'data_test',
            'labelled_data_dir': 'data_test_labelled',
            'input_data_sha256': 'input',
            'topic_catalog_sha256': 'topics',
            'labelled_test_sha256': 'labels',
            'timing': {},
        }

        report = subject.build_report(
            records, predictions, topics, 1, context
        )

        metrics = report['metrics']
        self.assertEqual(metrics['candidate_recall_at_n'], 0.5)
        self.assertEqual(metrics['retrieval_errors'], 1)
        self.assertEqual(metrics['selection_opportunities'], 1)
        self.assertEqual(metrics['selection_errors'], 0)
        self.assertEqual(metrics['llama_final_accuracy'], 0.5)
        self.assertEqual(metrics['retriever_top_1_accuracy'], 0.5)
        self.assertEqual(
            metrics['conditional_selection_accuracy'], 1.0
        )
        self.assertFalse(
            report['configuration']['uses_precomputed_e5_results']
        )


if __name__ == '__main__':
    unittest.main()
