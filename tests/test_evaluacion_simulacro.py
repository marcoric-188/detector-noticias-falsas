"""Simulacro de la evaluación final con particiones FALSAS tomadas solo de entrenamiento.

Permite verificar que ``fakenews.evaluate`` funciona de principio a fin (y que se niega a repetirse) sin
tocar nunca las particiones de prueba reales.
"""

import json
import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from fakenews import config

TRAIN = config.ruta("data/processed/train.csv")
pytestmark = pytest.mark.skipif(not TRAIN.exists(), reason="Ejecute primero: python -m fakenews.data preparar")


@pytest.fixture(scope="module")
def raiz_simulacro(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("simulacro")
    shutil.copytree(config.ruta("configs"), raiz / "configs")
    base = yaml.safe_load((raiz / "configs/base.yaml").read_text(encoding="utf-8"))
    base["evaluacion"]["bootstrap_remuestreos"] = 50
    (raiz / "configs/base.yaml").write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")

    tr = pd.read_csv(TRAIN, keep_default_na=False).sample(frac=1, random_state=0).reset_index(drop=True)
    proc = raiz / "data/processed"
    proc.mkdir(parents=True)
    tr.iloc[:420].to_csv(proc / "train.csv", index=False)
    tr.iloc[420:550].to_csv(proc / "prueba_dd.csv", index=False)
    tr.iloc[550:].to_csv(proc / "prueba_fd.csv", index=False)

    res = raiz / "results"
    (res / "tablas").mkdir(parents=True)
    spec = config.experimentos()["E-02"]["modelo"]
    (res / "seleccion.json").write_text(json.dumps({"elegido": "E-02", "modelo": spec}), encoding="utf-8")
    pd.DataFrame([
        {"id": "E-02", "esquema": "agrupada", "f1_macro_media": 0.7, "f1_macro_de": 0.1},
        {"id": "E-00b", "esquema": "agrupada", "f1_macro_media": 0.45, "f1_macro_de": 0.1},
    ]).to_csv(res / "bitacora.csv", index=False)
    return raiz


def _evaluar(raiz, *extra):
    env = {**os.environ, "FAKENEWS_ROOT": str(raiz),
           "PYTHONPATH": str(config.RAIZ / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run([sys.executable, "-m", "fakenews.evaluate", "--confirmo-uso-unico", *extra],
                          env=env, capture_output=True, text=True, timeout=900)


def test_evaluacion_completa_y_uso_unico(raiz_simulacro):
    r = _evaluar(raiz_simulacro)
    assert r.returncode == 0, r.stderr[-3000:]
    fin = json.loads((raiz_simulacro / "results/metricas_finales.json").read_text(encoding="utf-8"))
    for clave in ("prueba_dd", "prueba_fd"):
        m = fin[clave]["modelo_final"]["metricas"]
        assert 0 <= m["f1_macro"] <= 1
        ic = fin[clave]["modelo_final"]["ic95"]["f1_macro"]
        assert ic["ic95_inf"] <= ic["ic95_sup"]
    assert len(fin["criterios_exito"]) == 7
    assert fin["app"]["explicaciones_fieles"] == fin["app"]["explicaciones_total"]
    for archivo in ("matrices_confusion.png", "curvas_roc_pr.png", "calibracion.png"):
        assert (raiz_simulacro / "results/figuras" / archivo).exists()
    assert (raiz_simulacro / "models/detector_v1.0.0.joblib").exists()
    assert (raiz_simulacro / "results/tablas/evaluacion_reglas.csv").exists()

    # Segunda ejecución sin --forzar: debe negarse (la prueba se usa una sola vez)
    r2 = _evaluar(raiz_simulacro)
    assert r2.returncode != 0
    assert "ya se usó" in (r2.stderr + r2.stdout)
