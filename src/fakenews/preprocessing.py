"""Normalización de texto común a entrenamiento, evaluación y aplicación.

Decisiones (ver ``docs/decisiones_tecnicas.md``):
- v1 del corpus reemplazó cifras por ``*NUMBER*`` y v2 no; se normaliza todo a ``NUM`` (hallazgo H3).
- En v1 el cuerpo repite el titular al inicio; se elimina antes de unirlos (hallazgo H4).
"""

from __future__ import annotations

import re
import unicodedata

_RE_HTML = re.compile(r"<[^>]{1,200}>")
_RE_URL = re.compile(r"(https?://\S+|www\.\S+|\*URL\*)", re.IGNORECASE)
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_RE_MARCAS_NUM = re.compile(r"\*(NUMBER|PHONE)\*")
_RE_DIGITOS = re.compile(r"\d+(?:[.,:/]\d+)*")
_RE_ESPACIOS = re.compile(r"[ \t\r\f\v]+")
_RE_SALTOS = re.compile(r"\n{2,}")

#: Palabras funcionales frecuentes del español, usadas solo para detectar si un texto está en español.
PALABRAS_ESPANOL = frozenset(
    """de la que el en y a los se del las un por con no una su para es al lo como mas pero sus le ya o
    este porque esta entre cuando muy sin sobre tambien me hasta hay donde quien desde todo nos durante
    todos uno les ni contra otros ese eso ante ellos esto antes algunos unos yo otro otras otra el tanto
    esa estos mucho quienes nada muchos cual poco ella estar estas algunas algo nosotros fue ha han son
    era sido ser tiene dijo segun""".split()
)


def quitar_tildes(texto: str) -> str:
    """Elimina tildes y diéresis (``"Según"`` -> ``"Segun"``); conserva la ñ como n."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalizar(texto: object) -> str:
    """Limpia y normaliza un texto.

    - Valores nulos o no textuales -> cadena vacía.
    - Quita etiquetas HTML, reemplaza URLs por ``URL``, correos por ``EMAIL`` y cifras por ``NUM``.
    - Colapsa espacios repetidos y líneas en blanco.
    """
    if texto is None or (isinstance(texto, float) and texto != texto):  # NaN
        return ""
    s = str(texto)
    s = _RE_HTML.sub(" ", s)
    s = _RE_URL.sub(" URL ", s)
    s = _RE_EMAIL.sub(" EMAIL ", s)
    s = _RE_MARCAS_NUM.sub(" NUM ", s)
    s = _RE_DIGITOS.sub(" NUM ", s)
    s = _RE_ESPACIOS.sub(" ", s)
    s = "\n".join(linea.strip() for linea in s.split("\n"))
    s = _RE_SALTOS.sub("\n", s)
    return s.strip()


def quitar_titular_del_cuerpo(titular: str, cuerpo: str) -> str:
    """Si el cuerpo empieza repitiendo el titular (sin importar mayúsculas), lo elimina."""
    t, c = titular.strip(), cuerpo.strip()
    if t and c.lower().startswith(t.lower()):
        return c[len(t):].lstrip(" .:-\n")
    return c


def construir_entrada(titular: object, cuerpo: object, modo: str = "titular_cuerpo") -> str:
    """Construye el texto que recibe el modelo.

    Args:
        titular: titular de la noticia (puede faltar).
        cuerpo: cuerpo de la noticia.
        modo: ``titular_cuerpo`` (por defecto), ``titular`` o ``cuerpo``.
    """
    t = normalizar(titular)
    c = quitar_titular_del_cuerpo(t, normalizar(cuerpo))
    if modo == "titular":
        return t
    if modo == "cuerpo":
        return c
    if modo == "titular_cuerpo":
        return f"{t}\n{c}".strip()
    raise ValueError(f"Modo de entrada desconocido: {modo!r}")


def palabras(texto: str) -> list[str]:
    """Tokenización simple en palabras alfabéticas (minúsculas, sin tildes)."""
    return re.findall(r"[a-zñ]+", quitar_tildes(texto.lower()))


def proporcion_espanol(texto: str) -> float:
    """Proporción de palabras del texto que son palabras funcionales del español (0 a 1)."""
    tokens = palabras(texto)
    if not tokens:
        return 0.0
    return sum(t in PALABRAS_ESPANOL for t in tokens) / len(tokens)
