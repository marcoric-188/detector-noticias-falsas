"""Construcción de modelos a partir de la especificación de ``configs/experimentos.yaml``.

Todos los modelos reciben un ``DataFrame`` con columnas ``titular`` y ``cuerpo``. La conversión a texto y el
TF-IDF están dentro del ``Pipeline``, de modo que en validación cruzada se ajustan solo con el pliegue de
entrenamiento (prevención de fuga).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from .preprocessing import construir_entrada
from .rules import ClasificadorReglas, ReglasTransformer

#: Orden de preferencia (más simple / interpretable primero) usado para desempatar en la selección.
COMPLEJIDAD = {"logreg_word": 1, "logreg_char_wb": 2, "hibrido": 3, "svm": 4}


def seleccionar_texto(X, modo: str = "titular_cuerpo"):  # noqa: N803
    """Convierte el DataFrame (titular, cuerpo) en una lista de textos según el modo de entrada."""
    return [construir_entrada(t, c, modo) for t, c in zip(X["titular"], X["cuerpo"])]


def _tfidf(params: dict[str, Any]) -> TfidfVectorizer:
    p = dict(params)
    p["ngram_range"] = tuple(p.get("ngram_range", (1, 1)))
    return TfidfVectorizer(strip_accents=None, max_features=200_000, dtype=np.float64, **p)


def familia(spec: dict[str, Any]) -> str:
    """Nombre corto de la familia de modelo (para la regla de desempate)."""
    tipo = spec["tipo"]
    if tipo == "logreg":
        return f"logreg_{spec['tfidf']['analyzer']}"
    return tipo


def construir_modelo(spec: dict[str, Any], semilla: int = 42):
    """Crea el estimador (sin entrenar) descrito por ``spec``.

    Args:
        spec: diccionario ``modelo`` de un experimento.
        semilla: semilla aleatoria para los componentes que la usan.
    """
    tipo = spec["tipo"]
    entrada = spec.get("entrada", "titular_cuerpo")
    texto = FunctionTransformer(seleccionar_texto, kw_args={"modo": entrada})

    if tipo == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if tipo == "reglas":
        return ClasificadorReglas()

    clf_params = dict(spec.get("clf", {}))
    if tipo == "nb":
        clf = MultinomialNB(**clf_params)
    elif tipo in ("logreg", "hibrido"):
        clf = LogisticRegression(max_iter=5000, random_state=semilla, **clf_params)
    elif tipo == "svm":
        clf = CalibratedClassifierCV(
            LinearSVC(random_state=semilla, max_iter=20000, **clf_params), cv=3, method="sigmoid"
        )
    else:
        raise ValueError(f"Tipo de modelo desconocido: {tipo}")

    if tipo == "hibrido":
        rasgos = FeatureUnion(
            [
                ("texto", Pipeline([("sel", texto), ("tfidf", _tfidf(spec["tfidf"]))])),
                ("reglas", ReglasTransformer()),
            ]
        )
        modelo = Pipeline([("rasgos", rasgos), ("clf", clf)])
    else:
        modelo = Pipeline([("sel", texto), ("tfidf", _tfidf(spec["tfidf"])), ("clf", clf)])

    if spec.get("busqueda"):
        interna = StratifiedKFold(n_splits=3, shuffle=True, random_state=semilla)
        modelo = GridSearchCV(modelo, spec["busqueda"], cv=interna, scoring="f1_macro", n_jobs=1)
    return modelo
