from __future__ import annotations

import unittest

import evaluate_e5_top_n_llama as subject


class E5TopNLlamaTests(unittest.TestCase):
    def test_model_result_slug_is_safe_and_model_specific(self) -> None:
        self.assertEqual(
            subject.model_result_slug(
                'ministral-3:3b-instruct-2512-q4_K_M'
            ),
            'ministral_3_3b_instruct_2512_q4_k_m',
        )
        self.assertEqual(
            subject.model_result_slug('qwen2.5:7b-instruct'),
            'qwen2_5_7b_instruct',
        )

    def test_default_sweep_covers_every_possible_n(self) -> None:
        self.assertEqual(subject.resolve_n_values([], 4), [1, 2, 3, 4])

    def test_top_n_is_reordered_like_temas_json_before_llama(self) -> None:
        topics = {
            'tema-a': 'A',
            'tema-b': 'B',
            'tema-c': 'C',
            'tema-d': 'D',
        }
        ranked_top_n = [
            {'tema': 'tema-d', 'similarity': 0.9},
            {'tema': 'tema-b', 'similarity': 0.8},
            {'tema': 'tema-a', 'similarity': 0.7},
        ]

        ordered = subject.base.order_candidates_by_catalog(
            ranked_top_n, topics
        )

        self.assertEqual(
            [candidate['tema'] for candidate in ordered],
            ['tema-a', 'tema-b', 'tema-d'],
        )
        self.assertEqual(
            subject.DEFAULT_RESULTS_DIR.name,
            'results_llama3_2_3b_temas_order_two_prompts_e5_margin_0_0025',
        )
        self.assertEqual(subject.DEFAULT_LLAMA_MODEL, 'llama3.2:3b')

    def test_confident_margin_preserves_e5_top_1(self) -> None:
        topics = {'tema-a': 'A', 'tema-b': 'B', 'tema-c': 'C'}
        record = {
            'candidates': [
                {'tema': 'tema-c', 'similarity': 0.9},
                {'tema': 'tema-b', 'similarity': 0.8},
                {'tema': 'tema-a', 'similarity': 0.7},
            ]
        }

        route = subject.route_prediction(record, topics, 3, 0.01)

        self.assertEqual(
            route['decision_source'], subject.E5_DECISION_SOURCE
        )
        self.assertEqual(route['routing_reason'], subject.CONFIDENT_MARGIN_REASON)
        self.assertEqual(
            [candidate['tema'] for candidate in route['candidates']],
            ['tema-c'],
        )

    def test_margin_equal_to_threshold_preserves_e5_top_1(self) -> None:
        topics = {'tema-a': 'A', 'tema-b': 'B'}
        record = {
            'candidates': [
                {'tema': 'tema-a', 'similarity': 0.9000},
                {'tema': 'tema-b', 'similarity': 0.8975},
            ]
        }

        route = subject.route_prediction(record, topics, 2, 0.0025)

        self.assertEqual(route['e5_top_1_top_2_margin'], 0.0025)
        self.assertEqual(
            route['decision_source'], subject.E5_DECISION_SOURCE
        )

    def test_ambiguous_margin_sends_only_nearby_candidates_to_slm(self) -> None:
        topics = {
            'tema-a': 'A',
            'tema-b': 'B',
            'tema-c': 'C',
            'tema-d': 'D',
        }
        record = {
            'candidates': [
                {'tema': 'tema-d', 'similarity': 0.900},
                {'tema': 'tema-b', 'similarity': 0.899},
                {'tema': 'tema-c', 'similarity': 0.898},
                {'tema': 'tema-a', 'similarity': 0.890},
            ]
        }

        route = subject.route_prediction(record, topics, 4, 0.0025)

        self.assertEqual(
            route['decision_source'], subject.SLM_DECISION_SOURCE
        )
        self.assertEqual(route['routing_reason'], subject.AMBIGUOUS_MARGIN_REASON)
        self.assertEqual(
            [candidate['tema'] for candidate in route['candidates']],
            ['tema-b', 'tema-c', 'tema-d'],
        )

    def test_n_one_always_preserves_e5_without_llama(self) -> None:
        topics = {'tema-a': 'A', 'tema-b': 'B'}
        record = {
            'candidates': [
                {'tema': 'tema-b', 'similarity': 0.9000},
                {'tema': 'tema-a', 'similarity': 0.8999},
            ]
        }

        route = subject.route_prediction(record, topics, 1, 0.0025)

        self.assertEqual(
            route['decision_source'], subject.E5_DECISION_SOURCE
        )
        self.assertEqual(route['routing_reason'], subject.SINGLE_CANDIDATE_REASON)

    def test_disabled_margin_routes_complete_top_n_to_slm(self) -> None:
        topics = {
            'tema-a': 'A',
            'tema-b': 'B',
            'tema-c': 'C',
            'tema-d': 'D',
        }
        record = {
            'candidates': [
                {'tema': 'tema-d', 'similarity': 0.900},
                {'tema': 'tema-b', 'similarity': 0.800},
                {'tema': 'tema-c', 'similarity': 0.700},
                {'tema': 'tema-a', 'similarity': 0.600},
            ]
        }

        route = subject.route_prediction(record, topics, 3, None)

        self.assertEqual(
            route['decision_source'], subject.SLM_DECISION_SOURCE
        )
        self.assertEqual(route['routing_reason'], subject.NO_MARGIN_REASON)
        self.assertEqual(
            [candidate['tema'] for candidate in route['candidates']],
            ['tema-b', 'tema-c', 'tema-d'],
        )

    def test_disabled_margin_routes_n_one_to_slm(self) -> None:
        topics = {'tema-a': 'A', 'tema-b': 'B'}
        record = {
            'candidates': [
                {'tema': 'tema-b', 'similarity': 0.9},
                {'tema': 'tema-a', 'similarity': 0.8},
            ]
        }

        route = subject.route_prediction(record, topics, 1, None)

        self.assertEqual(
            route['decision_source'], subject.SLM_DECISION_SOURCE
        )
        self.assertEqual(
            [candidate['tema'] for candidate in route['candidates']],
            ['tema-b'],
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
                'decision_source': subject.SLM_DECISION_SOURCE,
                'e5_top_1_topic': 'tema-b',
                'topics_correctos': ['tema-b'],
                subject.base.DECISION_FIELD: 'tema-b',
                'matches_real_topic': True,
                'candidate_contains_real_topic': True,
                'inference_time_seconds': 0.1,
            },
            {
                'decision_source': subject.E5_DECISION_SOURCE,
                'e5_top_1_topic': 'tema-a',
                'topics_correctos': ['tema-b'],
                subject.base.DECISION_FIELD: 'tema-a',
                'matches_real_topic': False,
                'candidate_contains_real_topic': False,
                'inference_time_seconds': 0.0,
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
            'candidate_source': subject.DIRECT_CANDIDATE_SOURCE,
            'uses_precomputed_results': False,
            'precomputed_results_source': None,
            'timing': {},
        }

        report = subject.build_report(
            records, predictions, topics, 1, context, 0.0025
        )

        metrics = report['metrics']
        self.assertEqual(metrics['candidate_recall_at_n'], 0.5)
        self.assertEqual(metrics['retrieval_errors_at_n'], 1)
        self.assertEqual(metrics['selection_opportunities'], 1)
        self.assertEqual(metrics['selection_errors'], 0)
        self.assertEqual(metrics['final_accuracy'], 0.5)
        self.assertEqual(metrics['retriever_top_1_accuracy'], 0.5)
        self.assertEqual(
            metrics['conditional_selection_accuracy'], 1.0
        )
        self.assertFalse(
            report['configuration']['uses_precomputed_e5_results']
        )
        self.assertEqual(
            report['configuration']['candidate_order'],
            subject.CANDIDATE_ORDER,
        )
        self.assertEqual(
            report['configuration']['prompt_strategy'],
            subject.base.TWO_STAGE_PROMPT_STRATEGY,
        )
        self.assertEqual(
            report['configuration']['routing_strategy'],
            subject.ROUTING_STRATEGY,
        )
        self.assertEqual(
            report['configuration']['similarity_margin_threshold'],
            0.0025,
        )
        self.assertEqual(
            report['configuration']['slm_model'], subject.base.MODEL_NAME
        )
        self.assertEqual(metrics['slm_final_accuracy'], 0.5)
        self.assertNotIn('llama_model', report['configuration'])
        self.assertFalse(
            any(key.startswith('llama_') for key in metrics)
        )


if __name__ == '__main__':
    unittest.main()
