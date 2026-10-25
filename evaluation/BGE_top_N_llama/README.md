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
como un checkpoint independiente.

BGE selecciona el conjunto Top-N por score, pero antes de construir el prompt
esos N temas se reordenan como aparecen en `temas.json`. Así el ranking no
introduce una prioridad posicional adicional ante Llama. Para `N=12`, el orden
de candidatos coincide con el baseline Llama directo.

Cada predicción utiliza dos prompts. El programa numera las frases como `F1`,
`F2`, etc. El primero identifica el acontecimiento principal y selecciona el
identificador de una frase sin recibir los temas candidatos. La frase original
se recupera mediante ese identificador. El segundo recibe la noticia, la
extracción y los candidatos para seleccionar el tema.

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


El selector usa por defecto `llama3.2:3b`. Los resultados con orden controlado
se guardan en `results_llama3_2_3b_temas_order_two_prompts/`. Las variantes 3B
anteriores permanecen en `results_llama3_2_3b/` y
`results_llama3_2_3b_temas_order/`; los resultados 1B tampoco se modifican. Un
resultado previo de dos prompts con copia literal se reinicia y sustituye en
el mismo directorio.
