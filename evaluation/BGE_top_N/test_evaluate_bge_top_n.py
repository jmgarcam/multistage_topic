from __future__ import annotations

import unittest

import evaluate_bge_top_n as subject


class BgeTopNTests(unittest.TestCase):
    def test_default_n_values_cover_the_catalog(self) -> None:
        self.assertEqual(subject.resolve_n_values([], 4), [1, 2, 3, 4])

    def test_recall_is_calculated_from_one_complete_ranking(self) -> None:
        records = [
            {'_id': 'one', 'file': 'test.json', 'text': 'Uno'},
            {'_id': 'two', 'file': 'test.json', 'text': 'Dos'},
        ]
        raw_predictions = [
            {
                '_id': 'one',
                'topics_correctos': ['b'],
                'topic_scores': [
                    {'tema': 'b', 'cross_encoder_score': 0.9},
                    {'tema': 'a', 'cross_encoder_score': 0.1},
                ],
                'inference_time_seconds': 0.1,
            },
            {
                '_id': 'two',
                'topics_correctos': ['b'],
                'topic_scores': [
                    {'tema': 'a', 'cross_encoder_score': 0.8},
                    {'tema': 'b', 'cross_encoder_score': 0.2},
                ],
                'inference_time_seconds': 0.1,
            },
        ]

        predictions = subject.prepare_predictions(
            records, raw_predictions, [1, 2], 2
        )
        metrics = subject.candidate_recall_metrics(predictions, [1, 2])
        precision = subject.retrieval_precision_metrics(predictions, [1, 2])

        self.assertEqual(metrics['1']['candidate_recall'], 0.5)
        self.assertEqual(metrics['2']['candidate_recall'], 1.0)
        self.assertEqual(precision['1']['retrieval_precision'], 0.5)
        self.assertEqual(precision['1']['retrieval_precision_percent'], 50.0)
        self.assertEqual(precision['2']['retrieval_precision'], 0.5)
        self.assertEqual(precision['2']['retrieval_precision_percent'], 50.0)
        self.assertEqual(precision['2']['relevant_topics_retrieved'], 2)
        self.assertEqual(precision['2']['retrieved_topics'], 4)
        self.assertEqual(predictions[1]['first_correct_rank'], 2)


if __name__ == '__main__':
    unittest.main()
