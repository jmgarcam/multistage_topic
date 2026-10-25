# Llama Top-1

Baseline independiente que entrega directamente a Llama los 12 temas de
`dataset_mteb/temas.json`, con su nombre y definición, para cada noticia de
`dataset_mteb/data_test`. Llama debe devolver exactamente una etiqueta.

Cada predicción utiliza dos prompts. El programa numera primero las frases de
la noticia como `F1`, `F2`, etc. El primer prompt no recibe categorías: extrae
el acontecimiento principal y devuelve el identificador de una frase de
evidencia. El programa recupera esa frase directamente del texto original. El
segundo prompt recibe la extracción, la frase original y las categorías.

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
`evaluation/Llama_top_1/results_llama3_2_3b_temas_order_two_prompts/llama3_2_3b_direct_results.json`.
Si existe un resultado anterior que copiaba literalmente la evidencia, se
reinicia y se sustituye progresivamente en ese mismo archivo.


La métrica principal se denomina `slm_final_accuracy`: proporción de noticias
en las que la única etiqueta devuelta por Llama coincide con la etiqueta correcta.


El modelo predeterminado es `llama3.2:3b`. Los 12 candidatos se presentan en
el orden de `temas.json`, igual que en las variantes ordenadas E5/BGE Top-N +
Llama. Los resultados 3B anteriores permanecen en `results_llama3_2_3b/` y
`results_llama3_2_3b_temas_order/`; los resultados 1B tampoco se modifican.
