from __future__ import annotations

import unittest

import evaluate_llama_top_1 as subject


class LlamaTop1Tests(unittest.TestCase):
    def test_prepare_records_uses_the_complete_catalog(self) -> None:
        test_records = [
            {'_id': 'news-1', 'file': 'test.json', 'text': 'Texto'}
        ]
        labelled = {
            'news-1': {'text': 'Texto', 'topics': ['tema-b']}
        }
        topics = {'tema-a': 'A', 'tema-b': 'B', 'tema-c': 'C'}

        records = subject.prepare_records(test_records, labelled, topics)

        self.assertEqual(
            [candidate['tema'] for candidate in records[0]['candidates']],
            ['tema-a', 'tema-b', 'tema-c'],
        )
        self.assertEqual(records[0]['topics_correctos'], ['tema-b'])

    def test_prepare_records_rejects_text_mismatch(self) -> None:
        test_records = [
            {'_id': 'news-1', 'file': 'test.json', 'text': 'Texto A'}
        ]
        labelled = {
            'news-1': {'text': 'Texto B', 'topics': ['tema-a']}
        }

        with self.assertRaises(SystemExit):
            subject.prepare_records(
                test_records, labelled, {'tema-a': 'Definición'}
            )

    def test_report_describes_a_direct_llama_baseline(self) -> None:
        topics = {'tema-a': 'A', 'tema-b': 'B'}
        records = [
            {
                '_id': 'news-1',
                'file': 'test.json',
                'text': 'Texto',
                'topics_correctos': ['tema-b'],
                'candidates': [{'tema': 'tema-a'}, {'tema': 'tema-b'}],
            }
        ]
        predictions = [
            {
                subject.base.DECISION_FIELD: 'tema-b',
                'matches_real_topic': True,
                'inference_time_seconds': 0.1,
            }
        ]
        context = {
            'data_dir': 'data_test',
            'labelled_data_dir': 'data_test_labelled',
            'input_data_sha256': 'input',
            'topic_catalog_sha256': 'topics',
            'labelled_test_sha256': 'labels',
        }

        report = subject.build_report(
            records, predictions, topics, context
        )

        self.assertEqual(report['dataset']['candidate_topics_per_news'], 2)
        self.assertEqual(report['metrics']['llama_final_accuracy'], 1.0)
        configuration = report['configuration']
        self.assertFalse(configuration['uses_retriever'])
        self.assertFalse(configuration['uses_e5_candidates'])
        self.assertFalse(configuration['uses_bge_candidates'])
        self.assertFalse(configuration['uses_precomputed_candidates'])
        self.assertEqual(
            configuration['labels_used_for'], 'metrics only'
        )


if __name__ == '__main__':
    unittest.main()
