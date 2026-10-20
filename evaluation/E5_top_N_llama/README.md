# E5 Top-N + Llama

Calcula directamente con E5 el ranking completo de temas para cada noticia de
`dataset_mteb/data_test`. Después entrega a Llama los primeros N candidatos y
le obliga a seleccionar uno de ellos.

Desde la raíz del repositorio:

```bash
evaluation/E5_top_N_llama/run_e5_top_n_llama.sh
```

Sin argumentos evalúa todos los valores `N=1..12`. Puede limitarse el barrido
indicando valores concretos, por ejemplo:

```bash
evaluation/E5_top_N_llama/run_e5_top_n_llama.sh 1 3 5 12
```

E5 se ejecuta una sola vez y su ranking se reutiliza para todos los N de esa
ejecución. No se leen `semantic_similarity_results` ni resultados de BGE. Las
etiquetas de `data_test_labelled` se utilizan únicamente para métricas.


## Métricas

- `retriever_top_1_accuracy`: el primer tema de E5 coincide con la etiqueta.
- `candidate_recall_at_n`: el tema correcto aparece entre los N candidatos.
- `llama_final_accuracy`: la decisión final de Llama coincide con la etiqueta.
- `conditional_selection_accuracy`: accuracy de Llama cuando el tema correcto
  estaba disponible entre los candidatos.
- `retrieval_errors` y `selection_errors`: separan los fallos de recuperación y
  de selección.


El selector usa `llama3.2:3b` con el prompt v1 original. Los resultados 3B se
guardan en `results_llama3_2_3b/`; los resultados anteriores de 1B permanecen
en `results/`.
