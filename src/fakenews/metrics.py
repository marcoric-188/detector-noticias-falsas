"""Métricas de clasificación usadas en todo el proyecto (1 = falsa es la clase positiva)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calcular(y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None = None) -> dict[str, float]:
    """Calcula las métricas de clasificación de un conjunto de predicciones.

    Incluye F1-macro, exactitud, recall y precisión de «falsa», especificidad y, si hay probabilidades,
    ROC-AUC, PR-AUC y Brier.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "exactitud": accuracy_score(y, pred),
        "recall_falsa": recall_score(y, pred, pos_label=1, zero_division=0),
        "precision_falsa": precision_score(y, pred, pos_label=1, zero_division=0),
        "especificidad": tn / (tn + fp) if (tn + fp) else 0.0,
        "vp": int(tp), "fp": int(fp), "vn": int(tn), "fn": int(fn),
    }
    if proba is not None and len(np.unique(y)) == 2:
        m["roc_auc"] = roc_auc_score(y, proba)
        m["pr_auc"] = average_precision_score(y, proba)
        m["brier"] = brier_score_loss(y, proba)
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in m.items()}


def bootstrap_ic(
    y: np.ndarray,
    pred: np.ndarray,
    metrica: str = "f1_macro",
    n: int = 2000,
    semilla: int = 42,
    pred_base: np.ndarray | None = None,
) -> dict[str, float]:
    """Intervalo de confianza al 95 % por bootstrap (percentil).

    Si se da ``pred_base``, calcula el IC de la diferencia pareada ``métrica(pred) - métrica(pred_base)``
    sobre los mismos remuestreos.
    """
    rng = np.random.default_rng(semilla)
    y, pred = np.asarray(y), np.asarray(pred)
    valores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        v = calcular(y[idx], pred[idx])[metrica]
        if pred_base is not None:
            v -= calcular(y[idx], np.asarray(pred_base)[idx])[metrica]
        valores.append(v)
    valores = np.array(valores)
    punto = calcular(y, pred)[metrica] - (calcular(y, pred_base)[metrica] if pred_base is not None else 0)
    return {
        "valor": float(punto),
        "ic95_inf": float(np.percentile(valores, 2.5)),
        "ic95_sup": float(np.percentile(valores, 97.5)),
        "remuestreos": int(len(valores)),
    }
