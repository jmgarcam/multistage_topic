# Evaluación sobre `test_tagged_data_no_none.json`

Este directorio ejecuta el pipeline E5 + SLM con los dos prompts sobre la
partición de test etiquetada manualmente disponible en:

```text
tagged_data/test_tagged_data_no_none.json
```

El archivo contiene 600 noticias. Se ha extraído, con semilla 42, del conjunto
de 822 noticias sin la categoría `ninguno`. Otras 200 noticias disjuntas están
reservadas en `tagged_data/validation_tagged_data_no_none.json` y no son
utilizadas por `run_all.sh`. Las 22 restantes no pertenecen a ninguna de estas
dos particiones. La división se puede reproducir con:

```bash
python3 tagged_data/split_tagged_data_no_none.py
```

La configuración predeterminada de `run_all.sh` ejecuta cinco iteraciones del
experimento actual:

- E5: `intfloat/multilingual-e5-base`;
- tamaños máximos de candidatos: todos los valores `N=1..15`;
- modelos generativos: Llama 3.2 1B/3B, Llama 3.1 8B, Qwen 2.5 Instruct
  1.5B/3B/7B y Ministral 3 Instruct 3B/8B;
- sin filtrado ni enrutado por margen E5;
- extracción del acontecimiento y clasificación en dos llamadas;
- el SLM recibe siempre el Top-N completo de E5;
- orden de candidatos según `temas.json`.

## Ejecución

Para ejecutar cinco veces E5 + SLM con todos los valores de N y el SLM
directo, desde la raíz del proyecto:

```bash
evaluation_mi_dataset/run_all.sh
```

El launcher utiliza la partición de test como entrada y como referencia
etiquetada. Esto no filtra las etiquetas al modelo: los cargadores comunes
extraen únicamente `_id` y `text` para generar el ranking y los prompts; el
campo `tema` se usa solo para calcular las métricas.

Todos los resultados nuevos quedan bajo:

```text
evaluation_mi_dataset/5_iter/
```

Cada repetición utiliza un directorio independiente, desde `iter_01/` hasta
`iter_05/`. Dentro se crea un resultado E5 compartido por los ocho modelos:

```text
evaluation_mi_dataset/5_iter/iter_XX/e5_results/e5_top_n_results.json
```

Después se guardan los resultados híbridos para `N=1..15`, cuyos nombres
incluyen el modelo real con el patrón `e5_top_<N>_<modelo>_results.json`, y el
resultado directo de cada uno de los ocho modelos. La estructura de cada
modelo es:

```text
evaluation_mi_dataset/5_iter/iter_XX/
├── e5_results/
├── results_e5_<modelo>_two_prompts_no_margin/
└── results_<modelo>_two_prompts/
```

Son 129 informes por iteración y 645 informes tras completar las cinco. Si la
ejecución se interrumpe, volver a lanzar `run_all.sh` reanuda los checkpoints
de la iteración correspondiente.

También se puede ejecutar cada experimento de forma independiente:

```bash
evaluation_mi_dataset/run_e5_top_n_llama.sh
evaluation_mi_dataset/run_llama_only.sh
```

Sin `--n_values`, el primer launcher evalúa automáticamente todo el catálogo,
desde `N=1` hasta `N=15`. Para limitar una ejecución a valores concretos se
puede usar, por ejemplo,

```bash
evaluation_mi_dataset/run_e5_top_n_llama.sh --n_values 2 3 4
```

Las rutas y los parámetros principales también pueden sustituirse mediante
variables de entorno:

```bash
OLLAMA_MODEL=llama3.2:1b \
RESULTS_DIR=/ruta/a/resultados \
evaluation_mi_dataset/run_e5_top_n_llama.sh
```

`run_all.sh` permite cambiar el número de repeticiones y la raíz de salida con
`ITERATIONS` y `OUTPUT_ROOT`. Los launchers individuales admiten
`DATASET_PATH`, `TOPICS_FILE`, `E5_RESULTS_FILE`, `RESULTS_DIR`, `RESULT_SLUG`,
`SLM_ONLY_OUTPUT` y `OLLAMA_MODEL`. `LLAMA_ONLY_OUTPUT` se conserva como alias
compatible de `SLM_ONLY_OUTPUT`.

Los informes JSON emplean el prefijo neutral `slm_` para las métricas y los
metadatos del modelo generativo (`slm_model`, `slm_final_accuracy`,
`slm_called`, etc.). Los campos `ollama_*` identifican la API de ejecución y
se conservan con ese nombre. Para actualizar informes antiguos se puede usar:

```bash
python3 evaluation_mi_dataset/migrate_result_schema_to_slm.py
```

`DATASET_DIR` se conserva como alias compatible, aunque `DATASET_PATH` permite
indicar tanto un archivo JSON como un directorio.

Si se cambia el modelo o el conjunto de valores N, debe emplearse otro
`RESULTS_DIR` para no mezclar checkpoints incompatibles.
