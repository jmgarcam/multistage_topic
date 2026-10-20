# BGE Top-N + Llama

Calcula directamente con `BAAI/bge-reranker-v2-m3` el ranking completo de los
temas para cada noticia de `dataset_mteb/data_test`. BGE puntúa los 12 pares
formados por la noticia y `Tema: <nombre>. Definición: <definición>`. Después
entrega a Llama los primeros N candidatos y le obliga a seleccionar uno.

Desde la raíz del repositorio:

```bash
evaluation/BGE_top_N_llama/run_bge_top_n_llama.sh
```

Sin argumentos evalúa todos los valores `N=1..12`. Puede limitarse el barrido
indicando valores concretos:

```bash
evaluation/BGE_top_N_llama/run_bge_top_n_llama.sh 1 3 5 12
```

BGE Top-1 se ejecuta una sola vez y su ranking completo se reutiliza para todos
los N después de validar el modelo, los parámetros, los hashes de los datos y
todas las predicciones. No se leen rankings de E5. Las etiquetas de
`data_test_labelled` se utilizan únicamente para métricas. Cada N se guarda
como un checkpoint independiente dentro de `results_llama3_1_8b/`.

Para ignorar el informe de BGE Top-1 y recalcular el ranking:

```bash
evaluation/BGE_top_N_llama/run_bge_top_n_llama.sh --recompute-bge
```


## Métricas

- `retriever_top_1_accuracy`: el primer tema de BGE coincide con la etiqueta.
- `candidate_recall_at_n`: el tema correcto aparece entre los N candidatos.
- `llama_final_accuracy`: la decisión final de Llama coincide con la etiqueta.
- `conditional_selection_accuracy`: accuracy de Llama cuando el tema correcto
  estaba disponible entre los candidatos.
- `retrieval_errors` y `selection_errors`: separan los fallos de recuperación y
  de selección.


El selector usa `llama3.1:8b` con el prompt v1 original. Los resultados 8B se
guardan en `results_llama3_1_8b/`; los resultados anteriores permanecen en sus
directorios originales.
