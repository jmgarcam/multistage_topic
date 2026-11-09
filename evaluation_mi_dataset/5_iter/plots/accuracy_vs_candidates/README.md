# Accuracy frente al número de candidatos

Ocho paneles, uno por modelo, con la media de las cinco iteraciones del
pipeline E5 → SLM para `N=1..15`. La banda representa el rango y las barras
una desviación estándar. Incluye E5 Top-1 y el SLM directo como referencias.

Además, `accuracy_vs_candidates_by_family.svg` ofrece una segunda
representación que no sustituye a la anterior. Contiene un panel por familia
(`Llama`, `Qwen 2.5` y `Ministral 3`), una curva por tamaño y una única
referencia E5 Top-1. En esta vista se representan únicamente los pipelines
E5 → SLM, sin las líneas de SLM directo.

```bash
python3 evaluation_mi_dataset/5_iter/plots/accuracy_vs_candidates/plot_accuracy_vs_candidates.py
python3 evaluation_mi_dataset/5_iter/plots/accuracy_vs_candidates/plot_accuracy_vs_candidates_by_family.py
```
