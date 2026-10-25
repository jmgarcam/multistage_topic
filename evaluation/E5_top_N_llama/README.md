# E5 Top-N + Llama

Calcula con E5 el ranking completo de temas para cada noticia de
`dataset_mteb/data_test` y usa el margen entre Top-1 y Top-2 como puerta de
confianza:

```text
si similitud_top1 - similitud_top2 >= umbral:
    conservar E5 Top-1 sin llamar a Llama
si el margen es menor:
    consultar a Llama con los candidatos cercanos
```

El umbral predeterminado es `0.0025`. En los casos ambiguos solo se incluyen
los primeros N candidatos cuya diferencia respecto a Top-1 sea menor que el
mismo umbral. Por tanto, N es un máximo y el número real de candidatos puede
ser menor.

Desde la raíz del repositorio:

```bash
evaluation/E5_top_N_llama/run_e5_top_n_llama.sh
```

Sin argumentos evalúa todos los valores `N=1..12`. Puede limitarse el barrido
indicando valores concretos, por ejemplo:

```bash
evaluation/E5_top_N_llama/run_e5_top_n_llama.sh --n_values 1 3 5 12
```

Para probar otro umbral y guardar el experimento en otro directorio:

```bash
evaluation/E5_top_N_llama/run_e5_top_n_llama.sh \
  --similarity-margin-threshold 0.005 \
  --results-dir evaluation/E5_top_N_llama/results_margin_0_005 \
  --n_values 2 3 5 12
```

El ranking completo de E5 Top-N se reutiliza para todos los N después de
validar el modelo, los parámetros, los hashes de los datos y todas las
predicciones. No se leen `semantic_similarity_results` ni resultados de BGE.
Las etiquetas de `data_test_labelled` se utilizan únicamente para métricas.

E5 selecciona primero los candidatos cercanos por similitud. Antes de construir
el prompt se reordenan como aparecen en `temas.json`, para que el ranking no
introduzca una prioridad posicional adicional ante Llama. Con `N=1` se conserva
siempre E5 Top-1 y no se carga Llama, ya que no existe ninguna alternativa que
el modelo pueda seleccionar.

Cada noticia ambigua utiliza dos prompts. El programa numera las frases como
`F1`, `F2`, etc. El primero identifica el acontecimiento principal y selecciona
el identificador de una frase sin recibir los temas candidatos. El programa
recupera la frase original usando ese identificador. El segundo recibe la
noticia, la extracción y los candidatos para seleccionar el tema. Las noticias
resueltas directamente por E5 no ejecutan ninguno de los dos prompts.

Para ignorar el informe de E5 Top-N y recalcular el ranking:

```bash
evaluation/E5_top_N_llama/run_e5_top_n_llama.sh --recompute-e5
```


## Métricas

- `retriever_top_1_accuracy`: el primer tema de E5 coincide con la etiqueta.
- `candidate_recall_at_n`: el tema correcto aparece entre los N candidatos.
- `final_accuracy`: la decisión final del enrutador E5/Llama coincide con la
  etiqueta.
- `e5_direct_decisions` y `slm_routed_decisions`: noticias resueltas por cada
  rama.
- `slm_helped_vs_e5_top_1` y `slm_harmed_vs_e5_top_1`: correcciones y
  errores introducidos por Llama respecto a E5 Top-1.
- `net_hit_change_vs_e5_top_1`: cambio neto de aciertos frente a E5 Top-1.
- `conditional_selection_accuracy`: accuracy de Llama cuando el tema correcto
  estaba disponible entre los candidatos cercanos.
- `slm_invalid_decisions`: extracciones o selecciones inválidas entre las
  noticias enviadas a Llama.


El selector usa por defecto `llama3.2:3b`. Los resultados del enrutador se
guardan en
`results_llama3_2_3b_temas_order_two_prompts_e5_margin_0_0025/`. Los resultados
anteriores de un prompt no se modifican. Los informes previos de dos prompts
con copia literal se reinician y sustituyen progresivamente en este mismo
directorio.
