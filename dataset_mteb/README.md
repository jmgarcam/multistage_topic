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
