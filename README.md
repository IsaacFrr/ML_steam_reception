# ML — Predicción de la recepción de videojuegos en Steam

Proyecto de Machine Learning (Project Break II — The Bridge, Data Science).
Modelo de **clasificación supervisada** que predice si un videojuego tendrá una
**recepción positiva** del público a partir de sus características de lanzamiento.

## Problema de negocio
Un estudio o distribuidor quiere estimar, **antes de lanzar** un juego, la
probabilidad de que tenga buena acogida, para ajustar precio, género, plataformas
e idiomas. El modelo aprende de +122.000 juegos ya publicados en Steam.

## Dataset
- **Steam Games Dataset** (FronkonGames) — datos de la Steam Web API + SteamSpy.
- **122.611 registros × 39 columnas**. Licencia MIT.
- Fuente: https://www.kaggle.com/datasets/fronkongames/steam-games-dataset
  (alternativa sin login: https://huggingface.co/datasets/FronkonGames/steam-games-dataset)
- Los CSV pesados no se versionan (ver `.gitignore`); en `src/data_sample/` hay una muestra.

## Objetivo y variable objetivo
- **Target:** recepción positiva sí/no, derivada de las reseñas `Positive` / `Negative`
  (p. ej. % de reseñas positivas por encima de un umbral, filtrando juegos con pocas reseñas).
- **Tipo:** clasificación binaria. **Métrica:** F1 / ROC-AUC (clases desbalanceadas).

## Estructura del repositorio
```
├── main.ipynb              # Notebook final del pipeline de ML
├── src/
│   ├── data/               # Datos pesados (no versionados)
│   ├── data_sample/        # Muestra ligera del dataset
│   ├── img/                # Figuras del EDA
│   ├── models/             # Modelos guardados (pickle/joblib)
│   ├── notebooks/          # Notebooks de desarrollo (EDA, pruebas)
│   └── utils/              # Funciones auxiliares
├── Presentacion.pdf        # Documento soporte (pendiente)
├── requirements.txt
└── README.md
```

## Reproducción
1. Descarga `games.csv` del enlace de arriba y colócalo en `src/data/games.csv`.
2. Crea el entorno e instala dependencias: `pip install -r requirements.txt`.
3. Ejecuta los notebooks de `src/notebooks/` y, finalmente, `main.ipynb`.

## Estado
- [x] Definición del problema y del dataset
- [ ] EDA dirigido al modelado
- [ ] Preprocesado y feature engineering
- [ ] Modelado, optimización y evaluación
- [ ] Presentación y vídeo

## Autor
- Isaac Frías — [GitHub](https://github.com/IsaacFrr)
