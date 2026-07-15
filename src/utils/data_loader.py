"""
data_loader.py
--------------
Funciones reutilizables para cargar, limpiar y enriquecer el dataset de
Steam Games (FronkonGames, HuggingFace/Kaggle) y para construir la variable
objetivo del proyecto de ML: la recepcion positiva de un juego.

Uso tipico desde un notebook (con la raiz del repo en sys.path):

    from src.utils.data_loader import load_raw, clean_steam, add_features, add_target

    df = add_target(add_features(clean_steam(load_raw())))
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# Ruta por defecto al CSV bruto (relativa a la raiz del repo)
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "games.csv"

# El CSV de HuggingFace/Kaggle tiene un bug conocido: el header lista
# 'DiscountDLC count' como una sola columna pero los datos traen dos
# columnas separadas ('Discount' y 'DLC count'), lo que desalinea todas
# las columnas posteriores. Forzamos los nombres correctos al cargar.
COLUMNAS_REALES = [
    "AppID", "Name", "Release date", "Estimated owners", "Peak CCU",
    "Required age", "Price", "Discount", "DLC count", "About the game",
    "Supported languages", "Full audio languages", "Reviews", "Header image",
    "Website", "Support url", "Support email", "Windows", "Mac", "Linux",
    "Metacritic score", "Metacritic url", "User score", "Positive", "Negative",
    "Score rank", "Achievements", "Recommendations", "Notes",
    "Average playtime forever", "Average playtime two weeks",
    "Median playtime forever", "Median playtime two weeks",
    "Developers", "Publishers", "Categories", "Genres", "Tags",
    "Screenshots", "Movies",
]

# Columnas sin valor para el modelado (texto libre, urls, casi todo nulo)
COLS_DROP = [
    "Movies", "Score rank", "Metacritic url", "Reviews", "Notes",
    "Website", "Support url", "Support email", "Header image",
    "About the game", "Screenshots", "Full audio languages",
]

MIN_REVIEWS_TRACCION = 10   # umbral de "traccion" para el EDA general
MIN_REVIEWS_TARGET = 50     # reseñas minimas para que el % positivas sea fiable
UMBRAL_POSITIVA = 70        # % de reseñas positivas a partir del cual es "bien recibido"


def load_raw(path=DEFAULT_DATA_PATH):
    """Carga el CSV bruto con el header reparado."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encuentra {path}. Ver README -> Reproduccion -> Descargar el dataset."
        )
    return pd.read_csv(
        path, header=None, skiprows=1, names=COLUMNAS_REALES, engine="python",
    )


def clean_steam(df, drop_cols=None, filtrar_traccion=True, min_reviews=MIN_REVIEWS_TRACCION):
    """Limpieza del dataset bruto: tipos, duplicados y filtro de traccion."""
    df = df.copy()
    drop = drop_cols if drop_cols is not None else COLS_DROP
    df.drop(columns=[c for c in drop if c in df.columns], inplace=True)

    df["Release date"] = pd.to_datetime(df["Release date"], errors="coerce", format="mixed")

    num_cols = [
        "Price", "Discount", "DLC count", "Peak CCU", "Positive", "Negative",
        "Metacritic score", "User score", "Recommendations", "Required age",
        "Achievements", "Average playtime forever", "Median playtime forever",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["Windows", "Mac", "Linux"]:
        if c in df.columns:
            df[c] = df[c].astype(bool)

    if "AppID" in df.columns:
        df.drop_duplicates(subset=["AppID"], keep="first", inplace=True)

    if filtrar_traccion:
        total_reviews = df["Positive"].fillna(0) + df["Negative"].fillna(0)
        mask = (total_reviews >= min_reviews) | (df["Peak CCU"].fillna(0) > 0)
        df = df.loc[mask].copy()

    df.reset_index(drop=True, inplace=True)
    return df


def _cuenta_items(serie, sep=","):
    """Cuenta elementos separados por 'sep' en una columna de texto."""
    return (
        serie.fillna("").astype(str)
        .apply(lambda s: len([x for x in s.split(sep) if x.strip()]))
    )


def add_features(df):
    """Crea variables derivadas utiles para el analisis y el modelado."""
    df = df.copy()
    df["es_f2p"] = (df["Price"].fillna(0) == 0).astype(int)
    df["modelo_negocio"] = np.where(df["es_f2p"] == 1, "F2P", "Pago")

    df["total_reviews"] = df["Positive"].fillna(0) + df["Negative"].fillna(0)
    df["pct_positivas"] = np.where(
        df["total_reviews"] > 0, df["Positive"] / df["total_reviews"] * 100, np.nan,
    )

    df["anyo_lanzamiento"] = df["Release date"].dt.year
    df["antiguedad_anios"] = 2025 - df["anyo_lanzamiento"]

    if "Genres" in df.columns:
        df["genero_principal"] = (
            df["Genres"].fillna("Sin genero").astype(str).str.split(",").str[0].str.strip()
        )
        df["n_generos"] = _cuenta_items(df["Genres"])
    if "Categories" in df.columns:
        df["n_categorias"] = _cuenta_items(df["Categories"])
    if "Tags" in df.columns:
        df["n_tags"] = _cuenta_items(df["Tags"])
    if "Supported languages" in df.columns:
        df["n_idiomas"] = _cuenta_items(df["Supported languages"])

    df["n_plataformas"] = df[["Windows", "Mac", "Linux"]].sum(axis=1).astype(int)
    df["tiene_publisher"] = df["Publishers"].fillna("").astype(str).str.strip().ne("").astype(int)
    df["tiene_achievements"] = (df["Achievements"].fillna(0) > 0).astype(int)
    return df


def add_target(df, min_reviews=MIN_REVIEWS_TARGET, umbral=UMBRAL_POSITIVA):
    """
    Construye la variable objetivo de CLASIFICACION.

    Poblacion: solo juegos con al menos `min_reviews` reseñas, para que el
    % de positivas sea estadisticamente fiable.
    Target `recepcion_positiva` = 1 si pct_positivas >= `umbral`, si no 0.
    """
    df = df.copy()
    df = df[df["total_reviews"] >= min_reviews].copy()
    df["recepcion_positiva"] = (df["pct_positivas"] >= umbral).astype(int)
    df.reset_index(drop=True, inplace=True)
    return df


# Columnas que NO deben usarse como features: son la propia base de calculo
# del target (fuga de informacion / data leakage) o su equivalente directo.
COLS_LEAKAGE = [
    "Positive", "Negative", "total_reviews", "pct_positivas",
    "User score", "Recommendations", "recepcion_positiva",
]


def resumen_calidad(df):
    """Resumen rapido de calidad: dtype, nulos, % nulos, unicos y ejemplo."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulos": df.isnull().sum(),
        "pct_nulos": (df.isnull().mean() * 100).round(2),
        "unicos": df.nunique(),
        "ejemplo": [
            df[c].dropna().iloc[0] if df[c].notna().any() else None
            for c in df.columns
        ],
    })
