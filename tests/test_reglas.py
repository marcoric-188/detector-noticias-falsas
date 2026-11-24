import numpy as np
import pandas as pd
import pytest

from fakenews.config import cargar_yaml
from fakenews.preprocessing import quitar_tildes
from fakenews.rules import ClasificadorReglas, SistemaExperto

NEUTRA_TIT = "Alcaldía anuncia obras en la vía principal"
NEUTRA = ('La Secretaría de Movilidad informó el lunes que habrá cierres. "Planeen sus viajes", dijo la '
          "secretaria en rueda de prensa. " + "La entidad explicó los desvíos previstos para la zona. " * 12)


@pytest.fixture(scope="module")
def se():
    return SistemaExperto()


def ids(resultado):
    return {a.id for a in resultado.activadas}


def test_base_de_conocimiento_bien_formada():
    kb = cargar_yaml("reglas")
    assert kb["version"] and kb["umbral_sospechosa"] > 0
    vistos = set()
    for r in kb["reglas"]:
        assert {"id", "nombre", "descripcion", "fundamento", "campo", "tipo", "peso"} <= set(r)
        assert r["id"] not in vistos
        vistos.add(r["id"])


def test_noticia_neutra_no_es_sospechosa(se):
    r = se.evaluar(NEUTRA_TIT, NEUTRA)
    assert not r.sospechosa
    assert "R10" in ids(r)  # cita textual atribuida (protectora)
    assert not ids(r) & {"R01", "R02", "R03", "R04", "R05", "R07", "R08"}


@pytest.mark.parametrize(
    "titular, cuerpo, regla",
    [
        ("", "Esto pasó y punto. " * 30, "R01"),
        ("ESTO ES UN TITULAR GRITADO EN MAYUSCULAS", NEUTRA, "R02"),
        ("", NEUTRA + " Es URGENTE saberlo.", "R03"),
        ("", NEUTRA + " Compártelo con tus contactos.", "R04"),
        ("", NEUTRA + " Increíble!!", "R05"),
        ("", NEUTRA + " Científicos descubrieron algo.", "R06"),
        ("", "Sucedió algo en la ciudad y la gente comentó. " * 10, "R07"),
        ("", "Texto breve sin mucha sustancia, según dijo alguien.", "R08"),
        ("", NEUTRA + " Este remedio casero funciona.", "R09"),
    ],
)
def test_cada_regla_se_activa_con_su_senal(se, titular, cuerpo, regla):
    assert regla in ids(se.evaluar(titular, cuerpo))


def test_puntaje_es_suma_de_pesos_y_umbral(se):
    r = se.evaluar("¡¡URGENTE!! COMPARTE ESTO YA MISMO", "Compártelo antes de que lo borren!!! " * 5)
    assert r.puntaje == pytest.approx(sum(a.peso for a in r.activadas))
    assert r.sospechosa == (r.puntaje >= se.umbral)
    assert r.sospechosa


def test_evidencia_es_fiel_al_texto(se):
    """Anexo A.2: las 'citas' de la explicación deben existir en el texto (no decorativas)."""
    tit, cue = "¡¡URGENTE!!", "Científicos descubrieron que el limón cura el cáncer. Compártelo!!!"
    plano = quitar_tildes(f"{tit}\n{cue}".lower()).replace("\n", " ")
    for a in se.evaluar(tit, cue).activadas:
        if a.evidencia.startswith("«"):
            frag = a.evidencia.strip("«»").strip("…")
            assert frag in plano or frag in f"{tit} {cue}"


def test_explicacion_en_lenguaje_natural(se):
    r = se.evaluar("", "Compártelo!!! " * 5)
    assert r.explicacion().startswith("Sospechosa")
    assert "Ninguna regla" in SistemaExperto({"version": "x", "umbral_sospechosa": 1, "reglas": []})\
        .evaluar("", "x").explicacion()


def test_entradas_vacias_no_fallan(se):
    r = se.evaluar(None, None)
    assert isinstance(r.puntaje, float)


def test_clasificador_de_reglas_no_aprende_de_los_datos():
    X = pd.DataFrame({"titular": ["", "¡¡URGENTE!!"], "cuerpo": [NEUTRA, "Compártelo!!! " * 5]})
    a = ClasificadorReglas().fit(X, np.array([0, 1]))
    b = ClasificadorReglas().fit(X, np.array([1, 0]))  # etiquetas invertidas -> mismas predicciones
    assert (a.predict(X) == b.predict(X)).all()
    p = a.predict_proba(X)[:, 1]
    assert ((p >= 0.5) == (a.predict(X) == 1)).all()
