# Accuracy frente a tiempo por modelo

Esta es una alternativa más legible al diagrama de dispersión conjunto. Usa
una cuadrícula 3×2 con un panel para cada modelo:

- Llama 3.2 1B, Llama 3.2 3B y Llama 3.1 8B;
- Qwen 2.5 1.5B, Qwen 2.5 3B y Qwen 2.5 7B.

La escala vertical es común (`0–100%`) para que las accuracies sean
comparables. La escala temporal se ajusta localmente en cada panel para evitar
que las configuraciones de un mismo modelo queden superpuestas; por ello, las
velocidades deben compararse mediante los valores numéricos de los ejes X y no
por la posición horizontal entre paneles.

Elementos de la figura:

- círculos unidos: pipeline E5 → SLM para `N=1..15`;
- triángulo: SLM directo con los 15 temas;
- línea roja discontinua: accuracy E5 Top-1;
- anillo dorado: configuración globalmente no dominada.

Solo se etiquetan los extremos, el máximo de cada modelo y los puntos de la
frontera de Pareto. El script reutiliza la carga y validación de resultados del
plot conjunto, por lo que ambas figuras aplican exactamente la misma definición
de tiempo y dominancia.

Para regenerarla:

```bash
python3 evaluation_mi_dataset/plots/accuracy_vs_time_by_model/plot_accuracy_vs_time_by_model.py
```

Salidas:

- `accuracy_vs_time_by_model.svg`;
- `accuracy_vs_time_by_model_data.csv`.
