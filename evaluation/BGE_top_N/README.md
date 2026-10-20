# BGE Top-N

Baseline independiente del cross-encoder `BAAI/bge-reranker-v2-m3`. Para cada
noticia de `dataset_mteb/data_test`, BGE puntúa directamente los 12 temas
representados como `Tema: <nombre>. Definición: <definición>`.

El ranking se calcula una sola vez y se evalúan todos los valores `N=1..12`.
`top_1_accuracy` mide si el tema con mayor puntuación coincide con la etiqueta
correcta; en este dataset de etiqueta única equivale a la micro-precision.
`candidate_recall_at_n` mide si el tema correcto aparece entre los N candidatos.
`retrieval_precision_at_n` conserva la precisión de recuperación estándar: qué
fracción de los N candidatos son correctos. No utiliza E5 ni Llama. Por defecto
reutiliza el ranking completo producido por BGE Top-1 después de validar el
modelo, los parámetros, los hashes de los datos y todas las predicciones. Así se
evita cargar por segunda vez el mismo modelo nativo en GPU. Las etiquetas de
`data_test_labelled` se consultan únicamente para métricas.

Desde la raíz del repositorio:

```bash
evaluation/BGE_top_N/run_bge_top_n.sh
```

También se pueden seleccionar valores concretos de N:

```bash
evaluation/BGE_top_N/run_bge_top_n.sh 1 3 5 12
```

Para ignorar el informe de BGE Top-1 y recalcular el ranking:

```bash
evaluation/BGE_top_N/run_bge_top_n.sh --recompute-bge
```

El informe se guarda en `evaluation/BGE_top_N/results/bge_top_n_results.json`.
