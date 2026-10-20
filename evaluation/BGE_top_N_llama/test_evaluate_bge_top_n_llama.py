from __future__ import annotations

import unittest

import evaluate_bge_top_n_llama as subject


class BgeTopNLlamaTests(unittest.TestCase):
    def test_default_sweep_covers_every_possible_n(self) -> None:
        self.assertEqual(subject.resolve_n_values([], 4), [1, 2, 3, 4])

    def test_prepare_records_preserves_bge_ranking(self) -> None:
        test_records = [
            {'_id': 'news-1', 'file': 'test.json', 'text': 'Texto'}
        ]
        bge_predictions = [
            {
                '_id': 'news-1',
                'topics_correctos': ['tema-b'],
                'topic_scores': [
                    {'tema': 'tema-b', 'cross_encoder_score': 0.9},
                    {'tema': 'tema-a', 'cross_encoder_score': 0.1},
                ],
            }
        ]

        records = subject.prepare_records(test_records, bge_predictions)

        self.assertEqual(records[0]['topics_correctos'], ['tema-b'])
        self.assertEqual(
            [candidate['tema'] for candidate in records[0]['candidates']],
            ['tema-b', 'tema-a'],
        )

    def test_report_separates_retrieval_and_selection(self) -> None:
        topics = {'tema-a': 'A', 'tema-b': 'B'}
        records = [
            {
                '_id': 'news-1',
                'file': 'test.json',
                'text': 'Uno',
                'topics_correctos': ['tema-b'],
                'candidates': [
                    {'tema': 'tema-b', 'cross_encoder_score': 0.9},
                    {'tema': 'tema-a', 'cross_encoder_score': 0.1},
                ],
            },
            {
                '_id': 'news-2',
                'file': 'test.json',
                'text': 'Dos',
                'topics_correctos': ['tema-b'],
                'candidates': [
                    {'tema': 'tema-a', 'cross_encoder_score': 0.8},
                    {'tema': 'tema-b', 'cross_encoder_score': 0.2},
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
            'model': subject.bge.MODEL_NAME,
            'batch_size': subject.bge.BATCH_SIZE,
            'max_length': subject.bge.MAX_LENGTH,
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
        configuration = report['configuration']
        self.assertFalse(configuration['uses_precomputed_bge_results'])
        self.assertFalse(configuration['uses_e5_candidates'])
        self.assertEqual(
            configuration['bge_topic_representation'],
            subject.bge.TOPIC_TEMPLATE,
        )


if __name__ == '__main__':
    unittest.main()
