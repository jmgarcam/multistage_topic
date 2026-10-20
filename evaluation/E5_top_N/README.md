# E5 Top-N

Baseline independiente del bi-encoder `intfloat/multilingual-e5-base`. Para
cada noticia de `dataset_mteb/data_test`, E5 calcula la similitud coseno con
los 12 temas representados como `Tema: <nombre>. Definición: <definición>`.

El ranking se calcula una sola vez y se evalúan todos los valores `N=1..12`.
`top_1_accuracy` mide si el tema con mayor puntuación coincide con la etiqueta
correcta; en este dataset de etiqueta única equivale a la micro-precision.
`candidate_recall_at_n` mide si el tema correcto aparece entre los N candidatos.
`retrieval_precision_at_n` conserva la precisión de recuperación estándar: qué
fracción de los N candidatos son correctos. No utiliza BGE ni Llama. Por
defecto reutiliza el ranking completo producido por E5 Top-1 después de
validar el modelo, los parámetros, los hashes de los datos y todas las
predicciones. Así se evita cargar por segunda vez el mismo modelo nativo en
GPU. Las etiquetas de `data_test_labelled` se consultan únicamente para
métricas.

Desde la raíz del repositorio:

```bash
evaluation/E5_top_N/run_e5_top_n.sh
```

También se pueden seleccionar valores concretos de N:

```bash
evaluation/E5_top_N/run_e5_top_n.sh 1 3 5 12
```

Para ignorar el informe de E5 Top-1 y recalcular el ranking:

```bash
evaluation/E5_top_N/run_e5_top_n.sh --recompute-e5
```

El informe se guarda en `evaluation/E5_top_N/results/e5_top_n_results.json`.
