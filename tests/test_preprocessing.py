import math

import pytest

from fakenews.preprocessing import (
    construir_entrada,
    normalizar,
    proporcion_espanol,
    quitar_tildes,
    quitar_titular_del_cuerpo,
)


def test_normaliza_cifras_de_ambas_versiones_igual():
    # v1 trae *NUMBER*, v2 trae dígitos: ambos deben quedar como NUM (hallazgo H3)
    assert normalizar("Murieron *NUMBER* personas") == normalizar("Murieron 25 personas") == "Murieron NUM personas"
    assert normalizar("El 3,5% de 1.200") == "El NUM % de NUM"


def test_normaliza_urls_correos_html_y_marcas():
    s = normalizar("Ver <b>esto</b> en https://x.co/a?b=1 o www.y.com, *URL*, escribir a a.b@c.org *PHONE*")
    assert "<b>" not in s and "http" not in s and "www" not in s and "@" not in s
    assert s.count("URL") == 3 and "EMAIL" in s and "NUM" in s


@pytest.mark.parametrize("valor", [None, math.nan, ""])
def test_valores_vacios(valor):
    assert normalizar(valor) == ""


def test_colapsa_espacios_y_lineas():
    assert normalizar("a   b\n\n\n  c ") == "a b\nc"


def test_quita_titular_repetido_al_inicio_del_cuerpo():
    assert quitar_titular_del_cuerpo("TITULO", "titulo\nCuerpo de la nota") == "Cuerpo de la nota"
    assert quitar_titular_del_cuerpo("Otro", "Cuerpo") == "Cuerpo"
    assert quitar_titular_del_cuerpo("", "Cuerpo") == "Cuerpo"


def test_construir_entrada_modos():
    assert construir_entrada("Tit", "Tit\nCuerpo 2020") == "Tit\nCuerpo NUM"
    assert construir_entrada("Tit", "Cuerpo", "titular") == "Tit"
    assert construir_entrada("Tit", "Cuerpo", "cuerpo") == "Cuerpo"
    assert construir_entrada(None, "Cuerpo") == "Cuerpo"
    with pytest.raises(ValueError):
        construir_entrada("a", "b", "otro")


def test_quitar_tildes():
    assert quitar_tildes("Según la información, añadió") == "Segun la informacion, anadio"


def test_deteccion_de_idioma():
    es = "El gobierno informó que la medida será aplicada en todo el país durante los próximos días."
    en = "The government said that the measure will be applied across the country in the coming days."
    assert proporcion_espanol(es) > 0.3
    assert proporcion_espanol(en) < 0.08
    assert proporcion_espanol("") == 0.0
