# ML — Predicción de la recepción de videojuegos en Steam

> 🇪🇸 Español (abajo) · 🇬🇧 [English version](#english-version)

Proyecto de Machine Learning (Project Break II — The Bridge, Data Science).
Modelo de **clasificación supervisada** que predice si un videojuego tendrá una
**recepción positiva** del público a partir de sus características de lanzamiento.

**Resultado:** ROC-AUC **0.776** y F1 **0.869** en test (baseline: 0.50), sin usar
ninguna variable derivada de las reseñas.

## Problema de negocio
Lanzar un videojuego cuesta años y dinero, y la señal de si ha gustado —las reseñas—
llega *después* del lanzamiento. Un estudio o distribuidor quiere estimar **antes de
lanzar** la probabilidad de buena acogida, para ajustar precio, género, plataformas e
idiomas. El modelo aprende de +122.000 juegos ya publicados en Steam.

## Dataset
- **Steam Games Dataset** (FronkonGames) — Steam Web API + SteamSpy. Licencia MIT.
- **122.611 registros × 39 columnas**.
- Fuente: https://www.kaggle.com/datasets/fronkongames/steam-games-dataset
  (alternativa sin login: https://huggingface.co/datasets/FronkonGames/steam-games-dataset)
- Los CSV pesados no se versionan (ver `.gitignore`); en `src/data_sample/` hay una muestra.

> **Bug del dataset:** la cabecera del CSV junta `Discount` y `DLC count` en un solo nombre,
> lo que desplaza todas las columnas siguientes. `src/utils/data_loader.py` lo repara al cargar.

## Variable objetivo
- **Target:** `recepcion_positiva` = 1 si el **% de reseñas positivas ≥ 70 %**, 0 en caso contrario.
- **Población:** solo juegos con **≥ 50 reseñas** (para que el % sea fiable) → **30.620 juegos**.
- **Tipo:** clasificación binaria. **Clases desbalanceadas ~75/25** → métrica **ROC-AUC / F1**
  (nunca accuracy).

## Prevención de fuga de información
El target se calcula con las reseñas, así que se **excluyen del modelo** todas las variables
derivadas de ellas: `Positive`, `Negative`, `total_reviews`, `pct_positivas`, `User score` y
`Recommendations`. El modelo solo usa información conocida **antes o independientemente** de
las reseñas (precio, género, idiomas, plataformas, logros, playtime, antigüedad…).

## Resultados
| Modelo | ROC-AUC (CV 5-fold) |
|---|---|
| Dummy (baseline) | 0.50 |
| Regresión Logística | 0.70 |
| Random Forest | 0.75 |
| **HistGradientBoosting** (elegido) | **0.76** |

Optimización con `RandomizedSearchCV` sobre un espacio de **720 combinaciones** (12 muestreadas ×
3 folds) y validación del ganador con 5 folds → **ROC-AUC test 0.776 · F1 test 0.869**.
La ganancia frente al modelo por defecto es de solo +0.0004: el rendimiento lo limita la **señal
disponible**, no los hiperparámetros.

Variables más influyentes: tiempo de juego medio, precio, nº de idiomas, `es_f2p` y género
(importancia por permutación + curvas de dependencia parcial).

## Estructura del repositorio
```
├── main.ipynb              # Notebook principal: EDA, preprocesado, modelado y evaluación
├── Presentacion.pptx/.pdf  # Presentación y documento soporte
├── src/
│   ├── data/               # Datos pesados (no versionados) + caché parquet
│   ├── data_sample/        # Muestra ligera del dataset
│   ├── img/                # Figuras generadas (01-10)
│   ├── models/             # Modelo entrenado (no versionado, se regenera)
│   ├── notebooks/
│   │   ├── 01_eda.ipynb            # EDA extendido dirigido al modelado
│   │   └── 02_preprocessing.ipynb  # Preprocesado y feature encoding
│   └── utils/
│       ├── data_loader.py          # carga, limpieza, features, target y caché
│       └── preprocessing.py        # split y ColumnTransformer
├── requirements.txt
└── README.md
```

## Reproducción
1. Descarga `games.csv` del enlace de arriba y colócalo en `src/data/games.csv`.
2. Instala dependencias: `pip install -r requirements.txt`.
3. Ejecuta `main.ipynb` de principio a fin (los notebooks de `src/notebooks/` amplían EDA y
   preprocesado). La primera ejecución cachea el dataset preparado; las siguientes son inmediatas.

## Estado
- [x] Definición del problema y del dataset
- [x] EDA dirigido al modelado
- [x] Preprocesado y feature encoding
- [x] Modelado, optimización y evaluación (HistGB · ROC-AUC test 0.776)
- [x] Presentación y documento soporte

## Autor
Isaac Frías — [GitHub](https://github.com/IsaacFrr)

---

# English version

Machine Learning project (Project Break II — The Bridge, Data Science).
A **supervised classification** model that predicts whether a video game will be
**positively received** by players, based on its launch characteristics.

**Result:** **0.776 ROC-AUC** and **0.869 F1** on the test set (baseline: 0.50), without using
any review-derived feature.

## Business problem
Shipping a game takes years and money, and the signal of whether players liked it — the reviews —
only arrives *after* launch. A studio or publisher wants to estimate the probability of a good
reception **before launching**, in order to tune price, genre, platforms and languages. The model
learns from 122,000+ games already published on Steam.

## Dataset
- **Steam Games Dataset** (FronkonGames) — Steam Web API + SteamSpy. MIT license.
- **122,611 rows × 39 columns**. Heavy CSVs are not versioned; a sample lives in `src/data_sample/`.
- **Known bug:** the CSV header merges `Discount` and `DLC count` into a single name, shifting every
  following column. `src/utils/data_loader.py` repairs it on load.

## Target variable
- `recepcion_positiva` = 1 if the **share of positive reviews is ≥ 70 %**, else 0.
- **Population:** games with **≥ 50 reviews** only (so the ratio is reliable) → **30,620 games**.
- Binary classification with **imbalanced classes (~75/25)** → **ROC-AUC / F1**, never accuracy.

## Data leakage prevention
The target is computed from the reviews, so every review-derived variable is **excluded** from the
feature set: `Positive`, `Negative`, `total_reviews`, `pct_positivas`, `User score` and
`Recommendations`. The model only sees information available **before or independently of** the
reviews.

## Results
| Model | ROC-AUC (5-fold CV) |
|---|---|
| Dummy (baseline) | 0.50 |
| Logistic Regression | 0.70 |
| Random Forest | 0.75 |
| **HistGradientBoosting** (selected) | **0.76** |

Tuned with `RandomizedSearchCV` over a **720-combination** space (12 sampled × 3 folds), then
validated with 5 folds → **test ROC-AUC 0.776 · test F1 0.869**. The gain over the default model is
only +0.0004: performance is capped by the **available signal**, not by hyperparameters.

Most influential features: average playtime, price, number of languages, free-to-play flag and genre
(permutation importance + partial dependence curves).

## Reproduction
1. Download `games.csv` and place it at `src/data/games.csv`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run `main.ipynb` end to end. The first run caches the prepared dataset; later runs are instant.

## Author
Isaac Frías — [GitHub](https://github.com/IsaacFrr)
