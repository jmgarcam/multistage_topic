# NLI sobre desacuerdos E5/Llama

Este experimento aplica inferencia textual multilingüe exclusivamente a las
noticias donde la decisión final de Llama difiere de E5 Top-1. Por defecto lee
el informe completo de E5 Top-3 + Llama con margen `0.0025`.

Para cada desacuerdo construye una premisa con el acontecimiento principal y
la frase de evidencia exacta ya extraídos sin mostrar candidatos a Llama. A
continuación evalúa por separado dos hipótesis:

```text
El acontecimiento principal de la noticia pertenece al tema «{tema}».
Definición del tema: {definición}
```

El informe guarda únicamente la etiqueta NLI de cada hipótesis:
`entailment`, `neutral` o `contradiction`. No guarda logits ni probabilidades,
no calcula diferencias entre hipótesis, no decide entre E5 y Llama y no utiliza
el ground truth.

## Ejecución

Desde la raíz del proyecto:

```bash
evaluation/NLI_e5_llama/run_nli_disagreements.sh
```

El modelo predeterminado es:

```text
MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
```

El resultado se guarda en:

```text
evaluation/NLI_e5_llama/results_mdeberta_v3_base_xnli/
e5_top_3_llama_nli_results.json
```

Se puede analizar otro informe sin modificarlo:

```bash
evaluation/NLI_e5_llama/run_nli_disagreements.sh \
  --source-results RUTA/AL/INFORME.json \
  --output RUTA/AL/RESULTADO_NLI.json
```

Para validar y reutilizar un resultado existente:

```bash
evaluation/NLI_e5_llama/run_nli_disagreements.sh --reuse-existing
```
