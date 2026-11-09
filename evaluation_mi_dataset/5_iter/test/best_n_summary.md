# N óptimo por modelo y tamaño

Criterio: maximizar la **accuracy media** de las cinco iteraciones. Si varios valores de N tienen exactamente la misma media, se muestran todos y se selecciona el N más pequeño.

| Familia | Tamaño | N seleccionado | Todos los N máximos | Accuracy media (%) | SD (pp) | Diferencia frente a E5 (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3.2 | 1B | 1 | 1 | 81.167 | 0.000 | +0.000 |
| Llama 3.2 | 3B | 5 | 5 | 86.333 | 0.000 | +5.167 |
| Llama 3.1 | 8B | 9 | 9 | 90.133 | 0.075 | +8.967 |
| Qwen 2.5 | 1.5B | 2 | 2 | 85.000 | 0.000 | +3.833 |
| Qwen 2.5 | 3B | 7 | 7 | 89.800 | 0.447 | +8.633 |
| Qwen 2.5 | 7B | 13 | 13 | 91.900 | 0.091 | +10.733 |
| Ministral 3 | 3B | 15 | 15 | 91.367 | 0.506 | +10.200 |
| Ministral 3 | 8B | 11 | 11 | 93.367 | 0.075 | +12.200 |

Accuracy media de E5 Top-1: **81.167%**.

> Nota metodológica: estos N se han seleccionado usando los resultados del conjunto de test. Para estimar rendimiento no sesgado en un paper, N debería seleccionarse con validation y evaluarse una sola vez en test.
