# Ollama con Docker Compose

Inicia el contenedor desde este directorio:

```bash
docker compose up -d
```

La API de Ollama quedara disponible en `http://localhost:11434` y los modelos
se conservaran en el volumen Docker `ollama_data`.

Para descargar y ejecutar un modelo:

```bash
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama run llama3.2
```

Para detener el servicio sin borrar los modelos:

```bash
docker compose down
```

El puerto publicado se puede cambiar mediante `OLLAMA_PORT`. Por ejemplo:

```bash
OLLAMA_PORT=11435 docker compose up -d
```
