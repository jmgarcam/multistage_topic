# Accuracy y tiempo en cinco iteraciones

Esta figura agrega las cinco ejecuciones de los ocho modelos sobre las 600
noticias de test.

- La curva representa la media entre iteraciones.
- Las barras muestran una desviación estándar muestral por encima y por debajo
  de la media; se dibujan en los valores N impares para evitar saturar la
  figura.
- La banda translúcida representa el mínimo y el máximo observados.
- Los rombos abiertos en N=15 corresponden a la media del SLM directo con los
  15 temas y también incluyen su rango y desviación estándar verticales.
  Si un resultado directo queda fuera del intervalo ampliado del panel de
  accuracy, el rombo se fija al borde y se anota su valor con una flecha.
- Las líneas discontinuas muestran E5 Top-1 y el tiempo de E5 solo.
- Los intervalos de confianza del 95% para la media se incluyen en el CSV
  agregado y se calculan con la distribución t de Student.

Para regenerar todos los artefactos:

```bash
python3 evaluation_mi_dataset/5_iter/plots/accuracy_and_time_statistics/plot_accuracy_and_time_statistics.py
```

Salidas:

- `accuracy_and_time_statistics.svg`: figura combinada.
- `aggregate_statistics.csv`: media, desviación estándar, rango e IC 95%.
- `iteration_values.csv`: valores originales de cada repetición.
- `statistics_summary.md`: mejor N y resumen por modelo.
