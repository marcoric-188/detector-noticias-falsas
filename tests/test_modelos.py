import numpy as np
import pandas as pd
import pytest

from fakenews import config
from fakenews.metrics import bootstrap_ic, calcular
from fakenews.models import construir_modelo

TEXTOS_F = ["Increíble descubrimiento que nos ocultan comparte ya", "Urgente milagro que cura todo comparte"]
TEXTOS_V = ["El ministerio informó según el comunicado oficial", "La agencia dijo que el índice subió en marzo"]


def datos_juguete(n=12):
    cuerpos = (TEXTOS_F * n)[:n] + (TEXTOS_V * n)[:n]
    return pd.DataFrame({"titular": cuerpos, "cuerpo": cuerpos}), np.array([1] * n + [0] * n)


@pytest.mark.parametrize("eid", list(config.experimentos()))
def test_cada_experimento_construye_entrena_y_predice(eid):
    spec = dict(config.experimentos()[eid]["modelo"])
    if spec.get("busqueda"):  # rejilla mínima para que la prueba sea rápida
        spec["busqueda"] = {"clf__C": [1.0], "tfidf__min_df": [1]}
    if "tfidf" in spec:
        spec["tfidf"] = {**spec["tfidf"], "min_df": 1}
    X, y = datos_juguete()
    modelo = construir_modelo(spec, 0).fit(X, y)
    p = modelo.predict_proba(X)
    assert p.shape == (len(X), 2)
    assert np.allclose(p.sum(1), 1)
    assert set(modelo.predict(X)) <= {0, 1}


def test_metricas_casos_conocidos():
    y = np.array([1, 1, 0, 0])
    m = calcular(y, np.array([1, 0, 0, 1]), np.array([0.9, 0.4, 0.2, 0.6]))
    assert m["recall_falsa"] == 0.5 and m["especificidad"] == 0.5 and m["exactitud"] == 0.5
    assert (m["vp"], m["fn"], m["vn"], m["fp"]) == (1, 1, 1, 1)
    assert m["roc_auc"] == 0.75
    perfecto = calcular(y, y)
    assert perfecto["f1_macro"] == 1.0


def test_bootstrap_contiene_el_valor_puntual():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    pred = np.where(rng.random(200) < 0.8, y, 1 - y)
    ic = bootstrap_ic(y, pred, n=300)
    assert ic["ic95_inf"] <= ic["valor"] <= ic["ic95_sup"]
    dif = bootstrap_ic(y, pred, n=300, pred_base=pred)
    assert dif["valor"] == 0 and dif["ic95_inf"] == 0 == dif["ic95_sup"]
