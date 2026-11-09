# Selección del mejor N

Esta carpeta obtiene el número de candidatos `N` que maximiza la accuracy de
cada familia y tamaño de modelo probado en las cinco iteraciones.

El criterio principal es la **media de accuracy de las cinco ejecuciones**. Si
existe un empate exacto, `all_best_n` conserva todos los valores empatados y
`selected_n` toma el menor de ellos.

Para regenerar los resultados:

```bash
python3 evaluation_mi_dataset/5_iter/test/select_best_n.py
```

Archivos generados:

- `best_n_by_model.csv`: configuración seleccionada para cada modelo;
- `accuracy_by_n.csv`: las 120 combinaciones modelo–N y sus estadísticas;
- `best_n_summary.md`: resumen legible de la selección.

## Precaución metodológica

Los resultados de `5_iter` se calcularon sobre
`tagged_data/test_tagged_data_no_none.json`. Elegir `N` con estas accuracies
convierte el test en datos de selección. Para un resultado publicable sin
sesgo de selección, se debe escoger `N` en validation y aplicar después ese
valor fijo al test.
