# Accuracy frente a tiempo

El diagrama de dispersión compara todas las configuraciones completas:

- **Eje X:** tiempo medio total amortizado por noticia.
- **Eje Y:** accuracy.
- **Color:** modelo.
- **Forma:** E5 solo, SLM directo o pipeline E5 → SLM.

Para E5 se divide el tiempo total de ejecución entre todas las noticias. Para
el pipeline se suma ese coste amortizado de E5 al tiempo total de inferencia
del SLM dividido entre las noticias. La inferencia del SLM contiene las dos
llamadas: extracción del acontecimiento y clasificación. El SLM directo no
incluye E5, pero sí sus dos prompts.

Una configuración se marca como **no dominada** cuando no existe otra con
tiempo igual o menor y accuracy igual o mayor, siendo al menos una de las dos
comparaciones estrictamente mejor. La línea dorada discontinua une esa
frontera de Pareto; la zona preferible es la superior izquierda.

Solo se etiquetan E5, los SLM directos, los extremos o máximos de cada serie y
los puntos de Pareto. Para regenerar los artefactos:

```bash
python3 evaluation_mi_dataset/plots/accuracy_vs_time/plot_accuracy_vs_time.py
```

Salidas:

- `accuracy_vs_time.svg`;
- `accuracy_vs_time_data.csv`;
- `pareto_frontier.md`.
