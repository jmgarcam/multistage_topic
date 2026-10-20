# E5 Top-1

Baseline independiente del bi-encoder `intfloat/multilingual-e5-base`. Lee las
noticias directamente desde `dataset_mteb/data_test`, calcula la similitud
coseno con los 12 temas de `temas.json` y devuelve la etiqueta con mayor
puntuación. No utiliza BGE, Llama ni resultados precomputados.

Desde la raíz del repositorio:

```bash
evaluation/E5_top_1/calculate_top_1.sh
```

Las etiquetas de `dataset_mteb/data_test_labelled` solo se consultan para
calcular las métricas. El informe se guarda en
`evaluation/E5_top_1/results/e5_top_1_results.json`.


La métrica `top_1_accuracy` comprueba si el tema con mayor puntuación coincide
con la etiqueta correcta. En este dataset de etiqueta única equivale a la
micro-precision Top-1 definida para el estudio.
