"""Verificaciones explícitas de no-fuga (criterio 5 de la rúbrica)."""

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from fakenews import config, data, experiments
from fakenews.data import casi_duplicados, normalizar_fuente
from fakenews.models import construir_modelo

PROCESADOS = config.ruta("data/processed")
hay_datos = pytest.mark.skipif(not (PROCESADOS / "train.csv").exists(),
                               reason="Ejecute primero: python -m fakenews.data preparar")


def test_casi_duplicados_detecta_copias():
    ref = pd.Series(["el gato come pescado en la casa", "la economía crece este año"])
    cand = pd.Series(["El gato come pescado en la casa", "noticia totalmente distinta sobre deportes"])
    r = casi_duplicados(ref, cand, 0.9)
    assert r["es_duplicado"].tolist() == [True, False]


def test_normalizar_fuente_unifica_variantes():
    assert normalizar_fuente("El País ") == normalizar_fuente("EL PAIS") == "el pais"
    assert normalizar_fuente(None) == "desconocida"


def test_tfidf_se_ajusta_solo_con_entrenamiento():
    """Un término que solo aparece en validación no debe estar en el vocabulario aprendido."""
    spec = config.experimentos()["E-03"]["modelo"]
    entren = pd.DataFrame({"titular": ["a"] * 4,
                           "cuerpo": ["uno dos tres", "uno dos cuatro", "cinco seis tres", "cinco seis cuatro"]})
    modelo = construir_modelo(spec).fit(entren, np.array([0, 0, 1, 1]))
    vocab = modelo.named_steps["tfidf"].vocabulary_
    assert "palabraexclusivadeprueba" not in vocab
    modelo.predict_proba(pd.DataFrame({"titular": [""], "cuerpo": ["palabraexclusivadeprueba uno"]}))
    assert "palabraexclusivadeprueba" not in modelo.named_steps["tfidf"].vocabulary_


def test_pliegues_agrupados_no_comparten_fuentes():
    y = np.array([0, 1] * 30)
    grupos = np.repeat(np.arange(12), 5)
    for itr, iva in StratifiedGroupKFold(5, shuffle=True, random_state=1).split(y, y, grupos):
        assert not set(grupos[itr]) & set(grupos[iva])


def test_las_pruebas_exigen_confirmacion():
    with pytest.raises(PermissionError):
        data.cargar_prueba("prueba_dd")
    with pytest.raises(ValueError):
        data.cargar_prueba("train", confirmo_uso_unico=True)


def test_los_experimentos_no_leen_la_prueba():
    """El código de la bitácora solo puede usar la partición de entrenamiento."""
    fuente = inspect.getsource(experiments)
    assert "cargar_prueba" not in fuente
    assert "prueba_dd" not in fuente and "prueba_fd" not in fuente


@hay_datos
def test_no_hay_casi_duplicados_entre_entrenamiento_y_pruebas():
    tr = pd.read_csv(PROCESADOS / "train.csv", keep_default_na=False)
    dd = pd.read_csv(PROCESADOS / "prueba_dd.csv", keep_default_na=False)
    fd = pd.read_csv(PROCESADOS / "prueba_fd.csv", keep_default_na=False)
    umbral = config.base()["datos"]["umbral_casi_duplicado"]
    assert not casi_duplicados(tr["texto"], dd["texto"], umbral)["es_duplicado"].any()
    ref = pd.concat([tr["texto"], dd["texto"]], ignore_index=True)
    assert not casi_duplicados(ref, fd["texto"], umbral)["es_duplicado"].any()


@hay_datos
def test_tamanos_y_balance_de_particiones():
    tr = pd.read_csv(PROCESADOS / "train.csv", keep_default_na=False)
    dd = pd.read_csv(PROCESADOS / "prueba_dd.csv", keep_default_na=False)
    fd = pd.read_csv(PROCESADOS / "prueba_fd.csv", keep_default_na=False)
    assert (len(tr), len(dd), len(fd)) == (676, 290, 568)
    assert tr["etiqueta"].sum() == 338
    assert set(tr["etiqueta"]) == {0, 1}

def test_columnas_que_revelan_la_etiqueta_no_entran_al_modelo():
    """El modelo solo recibe titular y cuerpo: ni fuente, ni enlace, ni tema."""
    from fakenews import models

    fuente = inspect.getsource(models.seleccionar_texto) + inspect.getsource(experiments.validar)
    for columna in ("fuente\"]", "enlace", "\"tema"):
        assert columna not in fuente
    assert 'df[["titular", "cuerpo"]]' in inspect.getsource(experiments.validar)
