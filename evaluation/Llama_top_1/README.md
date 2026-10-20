# Llama Top-1

Baseline independiente que entrega directamente a Llama los 12 temas de
`dataset_mteb/temas.json`, con su nombre y definición, para cada noticia de
`dataset_mteb/data_test`. Llama debe devolver exactamente una etiqueta.

No utiliza E5, BGE, recuperación de candidatos ni resultados precalculados.
Las etiquetas de `dataset_mteb/data_test_labelled` se consultan únicamente
para calcular las métricas.

Desde la raíz del repositorio:

```bash
evaluation/Llama_top_1/run_llama_top_1.sh
```

Puede seleccionarse otro modelo disponible en Ollama:

```bash
evaluation/Llama_top_1/run_llama_top_1.sh --model llama3.1:8b
```

El informe y los checkpoints se guardan en
`evaluation/Llama_top_1/results_llama3_1_8b/llama_top_1_results.json`.


La métrica principal se denomina `llama_final_accuracy`: proporción de noticias
en las que la única etiqueta devuelta por Llama coincide con la etiqueta correcta.


El modelo predeterminado es `llama3.1:8b` y se mantiene el prompt v1 original.
Los resultados anteriores permanecen en sus directorios originales.
