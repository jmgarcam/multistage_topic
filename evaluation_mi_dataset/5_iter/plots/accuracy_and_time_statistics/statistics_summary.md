# Resumen estadístico de las cinco iteraciones

| Modelo | Mejor N | Accuracy media ± DE | Tiempo medio ± DE | Accuracy directa | Tiempo directo |
|---|---:|---:|---:|---:|---:|
| Llama 3.2 · 1B | 1 | 81.17% ± 0.00 | 95.9s ± 2.8 | 12.83% ± 0.00 | 95.4s ± 0.1 |
| Llama 3.2 · 3B | 5 | 86.33% ± 0.00 | 157.8s ± 9.2 | 83.50% ± 0.00 | 191.3s ± 0.4 |
| Llama 3.1 · 8B | 9 | 90.13% ± 0.07 | 270.1s ± 1.4 | 89.00% ± 0.37 | 297.5s ± 0.2 |
| Qwen 2.5 · 1.5B | 2 | 85.00% ± 0.00 | 92.1s ± 1.4 | 80.27% ± 0.97 | 98.9s ± 0.5 |
| Qwen 2.5 · 3B | 7 | 89.80% ± 0.45 | 147.4s ± 1.6 | 87.03% ± 0.07 | 160.8s ± 4.8 |
| Qwen 2.5 · 7B | 13 | 91.90% ± 0.09 | 217.7s ± 1.2 | 91.30% ± 0.18 | 217.3s ± 0.7 |
| Ministral 3 · 3B | 15 | 91.37% ± 0.51 | 247.8s ± 9.0 | 91.23% ± 0.43 | 241.9s ± 9.4 |
| Ministral 3 · 8B | 11 | 93.37% ± 0.07 | 365.2s ± 1.5 | 92.87% ± 0.36 | 379.6s ± 0.3 |

El mejor N se elige por mayor accuracy media; en caso de empate, se conserva el menor N. Los tiempos del pipeline incluyen E5 y las dos llamadas al SLM.
