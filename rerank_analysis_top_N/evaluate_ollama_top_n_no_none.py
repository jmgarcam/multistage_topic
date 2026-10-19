#!/usr/bin/env python3
'''Evalúa los candidatos top-N obligando a elegir un tema concreto.'''

import evaluate_ollama_top_n as evaluator


evaluator.ALLOW_NONE_DECISION = False
evaluator.MODEL_NAME = 'llama3.2:3b'
evaluator.MODEL_LABEL = 'Ollama llama3.2:3b'
evaluator.PROMPT_VERSION = 'structured_enum_without_none_v1'
evaluator.VALIDATION_DIR = '../data_validation_topics_only'
evaluator.OUTPUT_FILE = 'ollama_llama_3_2_3b_top_n_no_none_results.json'


if __name__ == '__main__':
    evaluator.main()
