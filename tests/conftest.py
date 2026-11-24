import warnings

import joblib
import pandas as pd
import pytest

from fakenews import config
from fakenews.models import construir_modelo
from fakenews.predict import ruta_modelo_por_defecto
from fakenews.train import empaquetar


@pytest.fixture(scope="session")
def ruta_modelo(tmp_path_factory):
    """Modelo versionado real si existe; si no, un modelo rápido (E-02) entrenado solo con entrenamiento."""
    real = ruta_modelo_por_defecto()
    if real.exists():
        return real
    train = config.ruta("data/processed/train.csv")
    if not train.exists():
        pytest.skip("No hay modelo ni datos: ejecute python -m fakenews.data preparar")
    df = pd.read_csv(train, keep_default_na=False)
    spec = config.experimentos()["E-02"]["modelo"]
    modelo = construir_modelo(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(df[["titular", "cuerpo"]], df["etiqueta"].to_numpy())
    ruta = tmp_path_factory.mktemp("modelo") / "detector_prueba.joblib"
    joblib.dump(empaquetar(modelo, spec, "solo pruebas"), ruta)
    return ruta
