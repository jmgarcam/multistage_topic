from __future__ import annotations

import unittest

import evaluate_llama_top_1 as subject


class LlamaTop1Tests(unittest.TestCase):
    def test_defaults_use_ordered_two_prompt_llama_3b_experiment(self) -> None:
        self.assertEqual(subject.DEFAULT_LLAMA_MODEL, 'llama3.2:3b')
        self.assertEqual(
            subject.DEFAULT_OUTPUT.parent.name,
            'results_llama3_2_3b_temas_order_two_prompts',
        )
        self.assertEqual(
            subject.DEFAULT_OUTPUT.name,
            'llama3_2_3b_direct_results.json',
        )

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
        self.assertEqual(report['metrics']['slm_final_accuracy'], 1.0)
        configuration = report['configuration']
        self.assertFalse(configuration['uses_retriever'])
        self.assertFalse(configuration['uses_e5_candidates'])
        self.assertFalse(configuration['uses_bge_candidates'])
        self.assertFalse(configuration['uses_precomputed_candidates'])
        self.assertEqual(
            configuration['candidate_order'], subject.CANDIDATE_ORDER
        )
        self.assertEqual(
            configuration['prompt_strategy'],
            subject.base.TWO_STAGE_PROMPT_STRATEGY,
        )
        self.assertEqual(configuration['slm_calls_per_valid_prediction'], 2)
        self.assertEqual(
            configuration['labels_used_for'], 'metrics only'
        )
        self.assertNotIn('llama_final_accuracy', report['metrics'])
        self.assertIn(
            'total_slm_inference_time_seconds', report['timing']
        )


if __name__ == '__main__':
    unittest.main()
