# Ejecución consecutiva de los siete baselines

`evaluation/run_all_baselines.sh` ejecuta, en este orden:

1. E5 Top-1
2. E5 Top-N
3. BGE Top-1
4. BGE Top-N
5. Llama Top-1
6. E5 Top-N + Llama
7. BGE Top-N + Llama

E5 Top-1 y BGE Top-1 guardan sus rankings completos. Los baselines Top-N y
Top-N + Llama reutilizan esos rankings después de validarlos, por lo que la
ejecución consecutiva carga una sola vez cada modelo de recuperación en GPU.

Desde la raíz del repositorio:

```bash
evaluation/run_all_baselines.sh
```

Cada baseline termina antes de iniciar el siguiente. Por defecto se realizan
hasta tres intentos por baseline, esperando 60 segundos entre ellos. Los
experimentos con Llama pueden así reanudar sus checkpoints después de un fallo
transitorio como HTTP 429. Un fallo definitivo no impide ejecutar los
baselines posteriores.

La ejecución también es reanudable por defecto: E5 Top-1 y BGE Top-1 validan
y reutilizan sus informes completos si ya existen. Si falta alguno, se calcula
normalmente. Para forzar que ambos Top-1 vuelvan a calcularse:

```bash
BASELINE_RESUME_RESULTS=0 evaluation/run_all_baselines.sh
```

El lanzador impide que haya dos ejecuciones simultáneas sobre la misma GPU y
el mismo servidor Ollama. Si ya hay una activa, la segunda termina con código
3 y muestra la ruta del lock.

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


Los tres baselines que usan Llama emplean por defecto `llama3.2:3b`. El programa
numera las frases y el primer prompt extrae el acontecimiento y selecciona un
identificador sin ver las categorías; la frase original se recupera de forma
determinista. El segundo prompt realiza la clasificación. En `E5 Top-N +
Llama`, además, las noticias
con margen Top-1/Top-2 mayor o igual que `0.0025` conservan E5 Top-1 sin llamar
a Llama; solo los casos ambiguos reciben candidatos cercanos. Sus resultados
se guardan en
`results_llama3_2_3b_temas_order_two_prompts_e5_margin_0_0025/`. Llama Top-1 y
BGE Top-N + Llama mantienen sus directorios de dos prompts. Los checkpoints de
dos prompts con copia literal se reinician y sustituyen en esos mismos
directorios. La variable `OLLAMA_MODEL` permite sobrescribir el modelo de los
tres baselines.
