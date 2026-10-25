# Recall y accuracy de selección

La figura contiene dos paneles alineados para `N=1..15`:

- **Superior:** Recall@N de E5 y accuracy final de cada pipeline E5 → SLM.
- **Inferior:** accuracy final del SLM condicionada a que exista una etiqueta
  correcta entre los `N` candidatos recuperados por E5.

La métrica condicionada se calcula directamente a partir de los conteos:

```text
A_selección(N) = final_hits / candidate_recall_at_n_hits
```

Por tanto, Recall@N es el techo de accuracy del pipeline cuando la elección
del SLM está restringida a esos candidatos. La gráfica solo incorpora
ejecuciones completas; las parciales o ausentes se documentan en el pie y en
el CSV.

Para regenerar la figura y la tabla de datos:

```bash
python3 evaluation_mi_dataset/plots/recall_and_selection_accuracy/plot_recall_and_selection_accuracy.py
```

Salidas:

- `recall_and_selection_accuracy.svg`;
- `recall_and_selection_accuracy_data.csv`.
