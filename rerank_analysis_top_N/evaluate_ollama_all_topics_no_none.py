#!/usr/bin/env python3
'''Evalúa todo el catálogo obligando a elegir un tema concreto.'''

import evaluate_ollama_all_topics as evaluator


evaluator.base.ALLOW_NONE_DECISION = False
evaluator.base.MODEL_NAME = 'llama3.2:3b'
evaluator.base.MODEL_LABEL = 'Ollama llama3.2:3b'
evaluator.PROMPT_VERSION = 'structured_enum_without_none_all_topics_v1'
evaluator.CANDIDATE_MODE = 'all_topics_no_none'
evaluator.VALIDATION_DIR = '../data_validation_topics_only'
evaluator.OUTPUT_FILE = (
    'ollama_llama_3_2_3b_all_topics_no_none_results.json'
)


if __name__ == '__main__':
    evaluator.main()
