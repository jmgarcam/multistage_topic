# BGE Top-1

Calcula la accuracy Top-1 de `BAAI/bge-reranker-v2-m3` sobre las 12 clases del
dataset MTEB como baseline independiente. No utiliza rankings ni candidatos de
E5: aplica BGE directamente a los 12 pares formados por el texto de la noticia
y el nombre y definición de cada tema, y conserva el tema con mayor score. La
representación de cada candidato es
`Tema: <nombre>. Definición: <definición>`, igual que en los experimentos E5.

Desde la raíz del repositorio:

```bash
evaluation/BGE_top_1/calculate_top_1.sh
```

El informe se guarda por defecto en
`evaluation/BGE_top_1/results/bge_top_1_results.json`. Las rutas de entrada y
salida pueden cambiarse con `--data-dir`, `--topics`, `--ground-truth-dir` y
`--output`; `--labelled-data-dir` es el nombre preferido para la copia
etiquetada. Los valores por defecto reproducen la configuración de
`evaluation/rerank_candidate_topics.py`: modelo `BAAI/bge-reranker-v2-m3`,
batch 16 y longitud máxima 512.


La métrica `top_1_accuracy` comprueba si el tema con mayor puntuación coincide
con la etiqueta correcta. En este dataset de etiqueta única equivale a la
micro-precision Top-1 definida para el estudio.
