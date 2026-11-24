"""Uso del modelo: validación de entradas, predicción, explicación y abstención.

Uso desde la terminal::

    python -m fakenews.predict --titular "..." --cuerpo "..."
    python -m fakenews.predict --archivo noticias.csv --salida predicciones.csv   # columnas titular, cuerpo
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config
from .preprocessing import construir_entrada, normalizar, palabras, proporcion_espanol, quitar_tildes
from .rules import SistemaExperto

ADVERTENCIA_USO = (
    "Herramienta educativa de apoyo a la priorización. Detecta patrones de estilo aprendidos de noticias "
    "mexicanas de 2018; NO verifica hechos ni determina si una noticia es verdadera. Una persona debe "
    "revisar siempre la decisión final."
)


class EntradaInvalida(ValueError):
    """La entrada no cumple las condiciones de uso del modelo (mensaje en español para el usuario)."""


def ruta_modelo_por_defecto() -> Path:
    """Ruta del modelo de producción versionado (``models/detector_v<versión>.joblib``)."""
    cfg = config.base()
    return config.ruta(cfg["salidas"]["dir_modelos"]) / f"detector_v{cfg['proyecto']['version_modelo']}.joblib"


# --------------------------------------------------------------------------------------- explicación
def _vectorizador_y_pesos(modelo) -> tuple[Any, np.ndarray]:
    """Extrae el TF-IDF ajustado y el peso lineal de cada término hacia la clase «falsa»."""
    if hasattr(modelo, "best_estimator_"):
        modelo = modelo.best_estimator_
    pasos = modelo.named_steps
    clf = pasos["clf"]
    if "rasgos" in pasos:  # modelo híbrido: TF-IDF + reglas
        vec = pasos["rasgos"].transformer_list[0][1].named_steps["tfidf"]
    else:
        vec = pasos["tfidf"]
    if hasattr(clf, "coef_"):
        pesos = clf.coef_[0]
    elif hasattr(clf, "feature_log_prob_"):
        pesos = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    elif hasattr(clf, "calibrated_classifiers_"):
        pesos = np.mean([c.estimator.coef_[0] for c in clf.calibrated_classifiers_], axis=0)
    else:
        raise TypeError("El clasificador no es lineal; no se puede explicar por términos.")
    return vec, np.asarray(pesos)[: len(vec.vocabulary_)]


def termino_presente(termino: str, texto: str) -> bool:
    """Comprueba que un término (palabra, bigrama o n-grama de caracteres) aparece en el texto."""
    t = texto.lower()
    if " " in termino.strip() and not termino.startswith(" ") and not termino.endswith(" "):
        partes = [re.escape(p) for p in termino.split()]
        return re.search(r"\b" + r"\W+".join(partes) + r"\b", t) is not None
    if termino != termino.strip():  # n-grama de caracteres con borde de palabra
        return termino in f" {t} "
    return re.search(r"\b" + re.escape(termino) + r"\b", t) is not None or termino in t


def explicar_terminos(modelo, texto: str, k_pos: int = 8, k_neg: int = 4) -> dict[str, list[dict]]:
    """Términos del texto que más empujaron la decisión hacia «falsa» y hacia «verdadera».

    La contribución de cada término es ``tfidf(término, texto) × peso(término)``; solo se consideran
    términos presentes en el texto, por lo que la explicación es fiel al modelo y a la entrada.
    """
    vec, pesos = _vectorizador_y_pesos(modelo)
    x = vec.transform([texto]).tocoo()
    if x.nnz == 0:
        return {"hacia_falsa": [], "hacia_verdadera": []}
    nombres = vec.get_feature_names_out()
    contrib = x.data * pesos[x.col]
    orden = np.argsort(contrib)
    pos = [{"termino": nombres[x.col[i]], "contribucion": round(float(contrib[i]), 4)}
           for i in orden[::-1][:k_pos] if contrib[i] > 0]
    neg = [{"termino": nombres[x.col[i]], "contribucion": round(float(contrib[i]), 4)}
           for i in orden[:k_neg] if contrib[i] < 0]
    return {"hacia_falsa": pos, "hacia_verdadera": neg}


# ------------------------------------------------------------------------------------------ detector
class Detector:
    """Carga el modelo versionado y la base de reglas, valida la entrada y devuelve una predicción explicada."""

    def __init__(self, ruta: str | Path | None = None):
        """Carga ``models/detector_v<versión>.joblib`` (o la ruta indicada)."""
        ruta = Path(ruta) if ruta else ruta_modelo_por_defecto()
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {ruta}. Ejecute: python -m fakenews.train"
            )
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            paquete = joblib.load(ruta)
        self.modelo = paquete["modelo"]
        self.metadatos: dict[str, Any] = paquete["metadatos"]
        self.avisos_carga = [str(a.message) for a in avisos if "Version" in type(a.message).__name__]
        if self.metadatos.get("sklearn") != sklearn.__version__:
            self.avisos_carga.append(
                f"El modelo se guardó con scikit-learn {self.metadatos.get('sklearn')} y se está usando "
                f"{sklearn.__version__}. Si hay errores, reconstrúyalo con: python -m fakenews.train"
            )
        self.reglas = SistemaExperto()
        cfg = config.base()
        self.umbral = cfg["decision"]["umbral"]
        self.abst_inf, self.abst_sup = cfg["decision"]["abstencion"]
        self.lim = cfg["entradas_app"]

    def validar(self, titular: object, cuerpo: object) -> tuple[str, str, list[str]]:
        """Valida y normaliza la entrada. Devuelve (titular, cuerpo, advertencias).

        Raises:
            EntradaInvalida: con un mensaje claro si la entrada no es utilizable.
        """
        if cuerpo is None or not isinstance(cuerpo, str) or not cuerpo.strip():
            raise EntradaInvalida("El cuerpo de la noticia está vacío. Pegue el texto completo de la noticia.")
        if titular is not None and not isinstance(titular, str):
            raise EntradaInvalida("El titular debe ser texto.")
        advertencias = []
        if re.search(r"<[^>]{1,200}>", f"{titular or ''} {cuerpo}"):
            advertencias.append("Se eliminaron etiquetas HTML del texto.")
        t, c = normalizar(titular or ""), normalizar(cuerpo)
        n = len(palabras(f"{t} {c}"))
        if n < self.lim["min_palabras"]:
            raise EntradaInvalida(
                f"El texto tiene {n} palabras; se necesitan al menos {self.lim['min_palabras']}. "
                "El modelo no es fiable con mensajes muy cortos (por ejemplo, un trino o solo un titular)."
            )
        if n > self.lim["max_palabras"]:
            raise EntradaInvalida(
                f"El texto tiene {n} palabras; el máximo es {self.lim['max_palabras']}. "
                "Pegue solo una noticia a la vez."
            )
        prop = proporcion_espanol(f"{t} {c}")
        if prop < self.lim["min_proporcion_palabras_espanol"]:
            raise EntradaInvalida(
                "El texto no parece estar en español (muy pocas palabras frecuentes del español). "
                "El modelo solo se entrenó con noticias en español."
            )
        if not t:
            advertencias.append("Sin titular: el modelo funciona mejor con titular y cuerpo.")
        return t, c, advertencias

    def predecir(self, titular: object, cuerpo: object) -> dict[str, Any]:
        """Predice la prioridad de revisión de una noticia y explica la decisión."""
        t0 = time.perf_counter()
        t, c, advertencias = self.validar(titular, cuerpo)
        X = pd.DataFrame({"titular": [t], "cuerpo": [c]})  # noqa: N806
        p = float(self.modelo.predict_proba(X)[0, 1])
        if p >= self.abst_sup:
            etiqueta = "sospechosa"
        elif p <= self.abst_inf:
            etiqueta = "probablemente confiable"
        else:
            etiqueta = "indeterminada"
        reglas = self.reglas.evaluar(t, c)
        texto_modelo = construir_entrada(t, c)
        return {
            "etiqueta": etiqueta,
            "prob_falsa": round(p, 4),
            "decision_binaria": "falsa" if p >= self.umbral else "verdadera",
            "requiere_revision_humana": True,
            "terminos": explicar_terminos(self.modelo, texto_modelo),
            "reglas": {
                "puntaje": reglas.puntaje,
                "umbral": self.reglas.umbral,
                "explicacion": reglas.explicacion(),
                "activadas": [r.__dict__ for r in reglas.activadas],
            },
            "advertencias": advertencias + self.avisos_carga,
            "aviso_de_uso": ADVERTENCIA_USO,
            "version_modelo": self.metadatos.get("version"),
            "version_reglas": self.reglas.version,
            "latencia_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


def evidencia_fiel(resultado: dict[str, Any], titular: str, cuerpo: str) -> bool:
    """Comprueba que cada término y cada evidencia de regla citados existan realmente en el texto (CE6)."""
    texto = construir_entrada(normalizar(titular), normalizar(cuerpo))
    for grupo in resultado["terminos"].values():
        for item in grupo:
            if not termino_presente(item["termino"], texto):
                return False
    plano = quitar_tildes(f"{normalizar(titular)}\n{normalizar(cuerpo)}".lower()).replace("\n", " ")
    for r in resultado["reglas"]["activadas"]:
        ev = r["evidencia"]
        if ev.startswith("«"):
            frag = ev.strip("«»").strip("…")
            if quitar_tildes(frag.lower()) not in plano and frag not in f"{titular} {cuerpo}".replace("\n", " "):
                return False
    return True


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la línea de comandos."""
    p = argparse.ArgumentParser(description="Clasifica noticias (prioridad de revisión)")
    p.add_argument("--titular", default="")
    p.add_argument("--cuerpo")
    p.add_argument("--archivo", help="CSV con columnas 'titular' y 'cuerpo'")
    p.add_argument("--salida", help="CSV de salida para --archivo")
    a = p.parse_args(argv)
    det = Detector()
    if a.archivo:
        df = pd.read_csv(a.archivo, keep_default_na=False)
        filas = []
        for t, c in zip(df.get("titular", [""] * len(df)), df["cuerpo"]):
            try:
                r = det.predecir(t, c)
                filas.append({"etiqueta": r["etiqueta"], "prob_falsa": r["prob_falsa"],
                              "reglas": r["reglas"]["explicacion"], "error": ""})
            except EntradaInvalida as e:
                filas.append({"etiqueta": "", "prob_falsa": np.nan, "reglas": "", "error": str(e)})
        out = pd.concat([df, pd.DataFrame(filas)], axis=1)
        out.to_csv(a.salida or sys.stdout, index=False)
        return 0
    if not a.cuerpo:
        p.error("indique --cuerpo o --archivo")
    try:
        print(json.dumps(det.predecir(a.titular, a.cuerpo), ensure_ascii=False, indent=2))
    except EntradaInvalida as e:
        print(f"Entrada inválida: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
