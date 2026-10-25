# Accuracy frente al número de candidatos

La figura compara el pipeline E5 → SLM para `N=1..15` en seis paneles:

- Llama 3.2 1B;
- Llama 3.2 3B;
- Llama 3.1 8B;
- Qwen 2.5 1.5B Instruct;
- Qwen 2.5 3B Instruct;
- Qwen 2.5 7B Instruct.

Incluye como referencias horizontales la accuracy de E5 Top-1 y la del mismo
SLM cuando recibe los 15 temas. Solo se dibujan resultados completos sobre las
822 noticias; las ejecuciones parciales o ausentes aparecen indicadas dentro
del panel y nunca se representan como resultados definitivos.

Para regenerar la gráfica y su tabla de datos:

```bash
python3 evaluation_mi_dataset/plots/accuracy_vs_candidates/plot_accuracy_vs_candidates.py
```

Salidas:

- `accuracy_vs_candidates.svg`;
- `accuracy_vs_candidates_data.csv`.
