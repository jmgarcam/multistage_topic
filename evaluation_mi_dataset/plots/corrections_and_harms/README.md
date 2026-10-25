# Correcciones y errores introducidos por el SLM

La figura usa barras divergentes, con un panel por modelo:

- las barras verdes muestran errores de E5 Top-1 que corrige el SLM;
- las barras rojas muestran aciertos de E5 Top-1 que el SLM estropea;
- los puntos negros muestran el balance neto en puntos de accuracy.

Todos los valores se expresan como porcentaje sobre el conjunto completo:

```text
ΔA = (correcciones - aciertos estropeados) / total de noticias
```

Por construcción, `ΔA` coincide con la diferencia entre la accuracy final del
pipeline y la accuracy E5 Top-1.

El CSV contiene también los cambios que no alteran la accuracy. La tabla
Markdown separa `incorrecta → incorrecta` de `correcta → correcta`, posibilidad
que existe porque el dataset contiene algunas noticias multietiqueta.

Solo se representan ejecuciones completas. Para regenerar los artefactos:

```bash
python3 evaluation_mi_dataset/plots/corrections_and_harms/plot_corrections_and_harms.py
```

Salidas:

- `corrections_and_harms.svg`;
- `corrections_and_harms_data.csv`;
- `neutral_changes_table.md`.
