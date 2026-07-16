"""
preprocessing.py
----------------
Preparacion de datos para el modelado: seleccion de features (sin fuga),
separacion X / y y construccion del preprocesador (ColumnTransformer).

El preprocesador se define aqui pero se AJUSTA solo con los datos de train
(en el notebook / pipeline) para no filtrar informacion del test.

    from src.utils.preprocessing import get_X_y, build_preprocessor, FEATURES
    X, y = get_X_y(dft)
    pre = build_preprocessor()
    X_train_t = pre.fit_transform(X_train)   # fit SOLO en train
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer


TARGET = "recepcion_positiva"

# Features numericas legitimas (ninguna deriva de las reseñas -> sin leakage)
NUM_FEATURES = [
    "Price", "Required age", "DLC count", "Achievements",
    "Average playtime forever", "Peak CCU",
    "n_generos", "n_categorias", "n_idiomas", "n_plataformas",
    "antiguedad_anios",
]

# Numericas con cola muy larga -> se transforman con log1p antes de escalar
LOG_FEATURES = ["Price", "Average playtime forever", "Peak CCU", "DLC count", "Achievements"]

# Features categoricas / binarias
CAT_FEATURES = ["genero_principal"]
BIN_FEATURES = ["es_f2p", "tiene_publisher", "tiene_achievements", "Windows", "Mac", "Linux"]

FEATURES = NUM_FEATURES + CAT_FEATURES + BIN_FEATURES


def get_X_y(df, features=FEATURES, target=TARGET):
    """Devuelve X (solo features sin fuga) e y. Castea booleanos a int."""
    X = df[features].copy()
    for c in ["Windows", "Mac", "Linux"]:
        if c in X.columns:
            X[c] = X[c].astype(int)
    y = df[target].astype(int)
    return X, y


def _log1p_df(X):
    """log1p seguro (recorta negativos a 0) para las columnas de cola larga."""
    return np.log1p(np.clip(X, a_min=0, a_max=None))


def build_preprocessor(num=NUM_FEATURES, log=LOG_FEATURES, cat=CAT_FEATURES, binv=BIN_FEATURES):
    """
    ColumnTransformer:
      - log:  imputar(mediana) -> log1p -> escalar
      - resto numericas: imputar(mediana) -> escalar
      - categoricas: imputar(constante) -> OneHot
      - binarias: pasan tal cual
    """
    num_plain = [c for c in num if c not in log]

    pipe_log = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(_log1p_df, feature_names_out="one-to-one")),
        ("sc", StandardScaler()),
    ])
    pipe_num = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    pipe_cat = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Sin genero")),
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("log", pipe_log, log),
            ("num", pipe_num, num_plain),
            ("cat", pipe_cat, cat),
            ("bin", "passthrough", binv),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
