# Tiempo total frente al número de candidatos

Esta figura no representa accuracy. Compara directamente:

- **Eje X:** número de candidatos `N`;
- **Eje Y:** tiempo total para procesar las 822 noticias, en segundos.

Se utilizan dos paneles con escalas compartidas, uno para Llama y otro para
Qwen. Cada línea corresponde a un tamaño de modelo y contiene los pipelines
E5 → SLM para `N=1..15`.

Los triángulos representan el SLM directo y se colocan en `N=15` porque recibe
el catálogo completo de quince temas. La línea horizontal discontinua muestra el
tiempo de E5 solo; no se le asigna ningún valor de `N` en los datos.

El tiempo del pipeline es la suma del tiempo total de E5 y de todas las
inferencias del SLM. Cada noticia realiza dos llamadas: extracción del
acontecimiento y clasificación. En el SLM directo se suma solo su inferencia,
sin añadir E5. Ambos paneles usan exactamente las mismas escalas, por lo que
sus posiciones son comparables directamente.

Para regenerar la figura:

```bash
python3 evaluation_mi_dataset/plots/n_vs_time/plot_n_vs_time.py
```

Salidas:

- `n_vs_time.svg`;
- `n_vs_time_data.csv`.
