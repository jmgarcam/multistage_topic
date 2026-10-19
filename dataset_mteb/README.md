# Experimento MTEB Spanish News

Este directorio ejecuta la metodología sobre el split oficial `test` de
[`mteb/spanish_news`](https://huggingface.co/datasets/mteb/spanish_news).
Contiene 2.048 noticias clasificadas en 12 temas y no incluye una clase
`ninguno`.

## Datos

- `data_test/spanish_news_test.json`: textos de entrada sin ground truth.
- `data_validation/spanish_news_test.json`: los mismos textos con el nombre de
  la clase en `tema`, traducido a la clave correspondiente de `temas.json`.
- `temas.json`: nombres oficiales de las 12 clases y definiciones en español
  empleadas por la similitud semántica, el reranker y Ollama.

## Ejecución

Desde la raíz del proyecto, ejecutar en orden los scripts situados en
`evaluation/`:

```bash
python3 evaluation/semantic_similarity_test.py
python3 evaluation/rerank_candidate_topics.py
python3 evaluation/evaluate_ollama_top_n_no_none.py
python3 evaluation/evaluate_ollama_all_topics.py
```

Los scripts resuelven sus rutas independientemente del directorio de trabajo y
siguen leyendo y escribiendo los datos dentro de `dataset_mteb/`. La similitud
genera `dataset_mteb/semantic_similarity_results/`; el reranker produce
`dataset_mteb/reranked_topic_results.json`. Los evaluadores usan
`llama3.2:3b` y obligan al modelo a seleccionar un tema candidato.

## Evaluaciones auxiliares

Para calcular los informes Top-1 de E5 y BGE y ejecutar o reanudar el de
Llama:

```bash
evaluation/top_1/calculate_top_1.sh
```

La opción `--skip-llama` limita el cálculo a los resultados precomputados de
E5 y BGE. El modelo de Ollama se puede cambiar con `--llama-model`.

Para evaluar Llama directamente sobre los candidatos del ranking E5:

```bash
evaluation/E5_top_N_llama/run_e5_top_n_llama.sh
```

Sin argumentos procesa `N=2..12`; también acepta una lista concreta, por
ejemplo `2 3 5`. Cada N utiliza su propio resultado reanudable dentro de
`evaluation/E5_top_N_llama/results/`.
