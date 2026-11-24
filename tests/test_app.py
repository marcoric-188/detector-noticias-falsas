"""Prueba de la interfaz Streamlit sin abrir el navegador (streamlit.testing)."""

import pytest

from fakenews.predict import ruta_modelo_por_defecto

st_testing = pytest.importorskip("streamlit.testing.v1")
pytestmark = pytest.mark.skipif(not ruta_modelo_por_defecto().exists(),
                                reason="La app necesita el modelo versionado en models/")
APP = "app/app.py"


def test_la_app_carga_sin_errores():
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "Detector de noticias falsas" in at.title[0].value


def test_analiza_un_ejemplo():
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    at.checkbox[0].uncheck().run()  # no escribir registros de monitoreo durante las pruebas
    at.selectbox[0].select_index(1).run()
    at.button[0].click().run()
    assert not at.exception
    textos = " ".join(m.value for m in at.markdown) + " ".join(e.value for e in at.error)
    assert "Prioridad" in textos or "Indeterminada" in " ".join(w.value for w in at.warning)
    assert len(at.metric) == 3


def test_entrada_vacia_muestra_error_claro():
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    at.button[0].click().run()
    assert not at.exception
    assert any("vacío" in e.value for e in at.error)
