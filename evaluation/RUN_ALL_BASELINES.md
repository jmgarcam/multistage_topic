# Ejecución consecutiva de los siete baselines

`evaluation/run_all_baselines.sh` ejecuta, en este orden:

1. E5 Top-1
2. E5 Top-N
3. BGE Top-1
4. BGE Top-N
5. Llama Top-1
6. E5 Top-N + Llama
7. BGE Top-N + Llama

Desde la raíz del repositorio:

```bash
evaluation/run_all_baselines.sh
```

Cada baseline termina antes de iniciar el siguiente. Por defecto se realizan
hasta tres intentos por baseline, esperando 60 segundos entre ellos. Los
experimentos con Llama pueden así reanudar sus checkpoints después de un fallo
transitorio como HTTP 429. Un fallo definitivo no impide ejecutar los
baselines posteriores.

Para dejarlo ejecutándose al cerrar la terminal:

```bash
nohup evaluation/run_all_baselines.sh >/dev/null 2>&1 &
```

Los logs se guardan en `evaluation/run_logs/<fecha_hora>/`: un log general, un
log por baseline y `summary.tsv` con estados, intentos y tiempos finales.

El orden puede comprobarse sin ejecutar modelos:

```bash
evaluation/run_all_baselines.sh --dry-run
```

Los reintentos son configurables:

```bash
BASELINE_MAX_ATTEMPTS=5 \
BASELINE_RETRY_DELAY_SECONDS=300 \
OLLAMA_MODEL=llama3.2:3b \
nohup evaluation/run_all_baselines.sh >/dev/null 2>&1 &
```


Los tres baselines que usan Llama emplean por defecto `llama3.2:3b` con el
prompt v1 original y guardan sus resultados en `results_llama3_2_3b/`. Los
resultados anteriores de `llama3.2:1b` permanecen en `results/`.
