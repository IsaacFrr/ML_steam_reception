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

> Nota: el CSV original tiene un bug de cabecera (junta `Discount` y `DLC count` en un solo nombre).
> `src/utils/data_loader.py` lo repara automáticamente al cargar.

## Objetivo y variable objetivo
- **Target:** `recepcion_positiva` — 1 si el **% de reseñas positivas ≥ 70 %**, 0 en caso contrario.
- **Población:** solo juegos con **≥ 50 reseñas** (para que el % sea fiable) → **30.620 juegos**.
- **Tipo:** clasificación binaria. **Clases desbalanceadas ~75/25** → métrica **F1 / ROC-AUC**
  (no accuracy).

## EDA (`src/notebooks/01_eda.ipynb`)
EDA dirigido al modelado. Hallazgos principales:
- **Balance:** 75 % de los juegos son "bien recibidos" → desbalance a tratar en el modelo.
- **⚠️ Fuga de información:** se excluyen como features `Positive`, `Negative`, `pct_positivas`,
  `User score` y `Recommendations`, porque derivan de las mismas reseñas que definen el target.
- **Modelo de negocio:** los juegos de **pago** (76 %) se reciben mejor que los **F2P** (69 %).
- **Género:** hay señal clara (Adventure ~81 %, Action ~73 %).
- Relaciones feature→target y correlaciones en `src/img/`.

## Preprocesado y encoding (`src/notebooks/02_preprocessing.ipynb`)
Lógica reutilizable en `src/utils/preprocessing.py`:
- **Selección de features sin fuga** y separación `X` / `y` (`get_X_y`).
- **Split train/test 80/20 estratificado** por el target.
- **`ColumnTransformer`** (`build_preprocessor`), ajustado **solo con train**:
  - numéricas de cola larga (precio, playtime, Peak CCU, …): imputar mediana → `log1p` → escalar;
  - resto de numéricas: imputar mediana → escalar;
  - `genero_principal`: imputar → One-Hot (agrupa categorías raras, `min_frequency=50`);
  - flags binarios: passthrough.
- Resultado: **18 features → ~33 columnas** codificadas, sin nulos y con las numéricas centradas.

## Modelado y resultados (`main.ipynb`)
Pipeline de scikit-learn (`preprocesado + clasificador`) para no filtrar información en la CV:
- **Split** train/test 80/20 estratificado (24.496 / 6.124 juegos, 18 features).
- **Comparativa por validación cruzada (5-fold, ROC-AUC):** Dummy (0.50) < Regresión Logística
  (0.70) < Random Forest (0.75) < **HistGradientBoosting (0.76)**.
- **Optimización** del ganador con `GridSearchCV` (`learning_rate=0.1`, `max_iter=300`).
- **Evaluación final en test:** **ROC-AUC = 0.774**, **F1 = 0.868**. Matriz de confusión, curva ROC
  e importancia por permutación en `src/img/`.
- Variables más influyentes: tiempo de juego medio, precio, nº de idiomas, `es_f2p` y el género.
- El modelo entrenado (Pipeline completo) se guarda en `src/models/modelo_recepcion.joblib`.

> El modelo predice la recepción usando **solo datos independientes de las reseñas** (sin fuga). El
> techo de rendimiento lo limita la señal disponible: la acogida real depende también de calidad,
> marketing y momento de lanzamiento, no capturados por el dataset.

## Estructura del repositorio
```
├── main.ipynb              # Pipeline final: comparativa, GridSearch, evaluación y guardado
├── src/
│   ├── data/               # Datos pesados (no versionados)
│   ├── data_sample/        # Muestra ligera del dataset
│   ├── img/                # Figuras del EDA
│   ├── models/             # Modelos guardados (pickle/joblib)
│   ├── notebooks/
│   │   ├── 01_eda.ipynb            # EDA dirigido al modelado
│   │   └── 02_preprocessing.ipynb  # Preprocesado y feature encoding
│   └── utils/
│       ├── data_loader.py          # carga, limpieza, features y target
│       └── preprocessing.py        # split y ColumnTransformer
├── Presentacion.pdf        # Documento soporte (pendiente)
├── requirements.txt
└── README.md
```

## Reproducción
1. Descarga `games.csv` del enlace de arriba y colócalo en `src/data/games.csv`.
2. Crea el entorno e instala dependencias: `pip install -r requirements.txt`.
3. Ejecuta los notebooks de `src/notebooks/` en orden y, finalmente, `main.ipynb`.

## Estado
- [x] Definición del problema y del dataset
- [x] EDA dirigido al modelado
- [x] Preprocesado y feature encoding
- [x] Modelado, optimización y evaluación (HistGB · ROC-AUC test 0.774)
- [ ] Presentación y vídeo

## Autor
- Isaac Frías — [GitHub](https://github.com/IsaacFrr)
