# Cinco iteraciones

`evaluation_mi_dataset/run_all.sh` guarda aquí cinco ejecuciones separadas.
Cada `iter_XX/` contiene un resultado E5 y, para cada modelo, los 15
resultados E5 → SLM más el resultado directo con los 15 temas.

Todas las ejecuciones utilizan por defecto las 600 noticias de
`tagged_data/test_tagged_data_no_none.json`; la partición de validation no se
emplea para calcular estos resultados.

Los modelos incluidos son:

- Llama 3.2 1B;
- Llama 3.2 3B;
- Llama 3.1 8B;
- Qwen 2.5 Instruct 1.5B;
- Qwen 2.5 Instruct 3B;
- Qwen 2.5 Instruct 7B;
- Ministral 3 3B Instruct;
- Ministral 3 8B Instruct.

Cada iteración genera 129 informes JSON: uno de E5 y 16 por cada uno de los
ocho modelos. Las cinco iteraciones completas generan 645 informes.

Los checkpoints están separados por iteración, por lo que volver a lanzar el
script reanuda únicamente los resultados incompletos y no mezcla ejecuciones.
