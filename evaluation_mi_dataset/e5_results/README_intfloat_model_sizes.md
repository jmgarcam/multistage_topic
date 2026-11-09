# Comparación de tamaños de multilingual E5

`run_intfloat_model_sizes.sh` evalúa secuencialmente las 600 noticias de
`tagged_data/test_tagged_data_no_none.json` con los tres tamaños publicados de
la familia multilingüe de intfloat:

- `intfloat/multilingual-e5-small`;
- `intfloat/multilingual-e5-base`;
- `intfloat/multilingual-e5-large`.

No existe un checkpoint oficial `intfloat/multilingual-e5-medium`; el tamaño
intermedio de esta familia se denomina `base`.

Ejecución:

```bash
evaluation_mi_dataset/e5_results/run_intfloat_model_sizes.sh
```

El script utiliza el entorno virtual activo o detecta automáticamente el
entorno de Pipenv del proyecto. También se puede indicar un intérprete de forma
explícita con `PYTHON_BIN=/ruta/al/python`.

Los JSON se guardan en `e5_results/intfloat_model_sizes/`, junto con
`comparison.csv`. El archivo existente `e5_top_n_results.json` no se modifica.

El tamaño de lote se puede ajustar, por ejemplo para reducir memoria con el
modelo large:

```bash
BATCH_SIZE=16 evaluation_mi_dataset/e5_results/run_intfloat_model_sizes.sh
```
