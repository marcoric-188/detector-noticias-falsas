"""Sistema experto de reglas de verificación (base de conocimiento: ``configs/reglas.yaml``).

Componentes clásicos de un sistema experto:

- **Base de conocimiento**: reglas «SI señal ENTONCES suma peso», derivadas de la infografía de la IFLA.
- **Motor de inferencia**: ``SistemaExperto.evaluar`` aplica todas las reglas y suma los pesos.
- **Explicación**: cada regla activada devuelve su descripción, su fundamento y la *evidencia* textual
  concreta que la activó (fragmento encontrado o ausencia comprobada), para que la explicación sea fiel.

El sistema no aprende de datos: sus pesos y su umbral se fijaron a mano antes de los experimentos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from . import config
from .preprocessing import quitar_tildes


@dataclass
class ReglaActivada:
    """Regla que se activó para un texto, con la evidencia que la justifica."""

    id: str
    nombre: str
    descripcion: str
    fundamento: str
    peso: float
    evidencia: str


@dataclass
class ResultadoReglas:
    """Salida del motor de inferencia."""

    puntaje: float
    sospechosa: bool
    activadas: list[ReglaActivada] = field(default_factory=list)

    def explicacion(self) -> str:
        """Explicación en lenguaje natural, por ejemplo: «Sospechosa porque: no cita fuentes; …»."""
        if not self.activadas:
            return "Ninguna regla de verificación se activó."
        motivos = "; ".join(r.descripcion.rstrip(".").lower() for r in self.activadas if r.peso > 0)
        protectoras = [r for r in self.activadas if r.peso < 0]
        veredicto = "Sospechosa" if self.sospechosa else "Sin alerta suficiente"
        texto = f"{veredicto} según las reglas (puntaje {self.puntaje:g})"
        if motivos:
            texto += f" porque: {motivos}"
        if protectoras:
            texto += ". A favor: " + "; ".join(r.descripcion.rstrip(".").lower() for r in protectoras)
        return texto + "."


def _fragmento(texto: str, inicio: int, fin: int, margen: int = 30) -> str:
    a, b = max(0, inicio - margen), min(len(texto), fin + margen)
    return ("…" if a > 0 else "") + texto[a:b].replace("\n", " ") + ("…" if b < len(texto) else "")


class SistemaExperto:
    """Motor de inferencia que aplica la base de reglas a un titular y un cuerpo."""

    def __init__(self, reglas: dict[str, Any] | None = None):
        """Carga las reglas (por defecto desde ``configs/reglas.yaml``) y compila sus patrones."""
        self.kb = reglas if reglas is not None else config.cargar_yaml("reglas")
        self.version = str(self.kb["version"])
        self.umbral = float(self.kb["umbral_sospechosa"])
        self.reglas = self.kb["reglas"]
        self._patrones = {}
        for r in self.reglas:
            for clave in ("patrones", "requiere_a", "requiere_b"):
                if clave in r:
                    self._patrones[(r["id"], clave)] = [re.compile(p) for p in r[clave]]

    @property
    def ids(self) -> list[str]:
        """Identificadores de las reglas en orden."""
        return [r["id"] for r in self.reglas]

    # -------------------------------------------------------------------------------- reglas individuales
    def _campo(self, regla: dict, titular: str, cuerpo: str) -> str:
        return {"titular": titular, "cuerpo": cuerpo, "todo": f"{titular}\n{cuerpo}"}[regla["campo"]]

    def _aplicar(self, regla: dict, titular: str, cuerpo: str) -> str | None:
        """Devuelve la evidencia si la regla se activa, o ``None``."""
        original = self._campo(regla, titular, cuerpo)
        plano = quitar_tildes(original.lower())  # misma longitud que `original` salvo casos raros
        tipo = regla["tipo"]
        rid = regla["id"]

        if tipo == "presencia":
            for pat in self._patrones[(rid, "patrones")]:
                m = pat.search(plano)
                if m:
                    return f"«{_fragmento(plano, m.start(), m.end())}»"
            return None
        if tipo == "ausencia":
            if not plano.strip():
                return "el campo está vacío"
            if any(p.search(plano) for p in self._patrones[(rid, "patrones")]):
                return None
            return "no se encontró ninguna de las expresiones esperadas"
        if tipo == "mayusculas":
            letras = [c for c in original if c.isalpha()]
            if len(letras) < regla.get("min_letras", 10):
                return None
            prop = sum(c.isupper() for c in letras) / len(letras)
            if prop >= regla["umbral"]:
                return f"{prop:.0%} de las letras del titular están en mayúsculas"
            return None
        if tipo == "puntuacion":
            for pat in self._patrones[(rid, "patrones")]:
                m = pat.search(original)
                if m:
                    return f"«{_fragmento(original, m.start(), m.end(), 20)}»"
            n = original.count("!")
            if n >= regla.get("max_exclamaciones", 3):
                return f"{n} signos de exclamación"
            return None
        if tipo == "longitud":
            n = len(original.split())
            return f"{n} palabras" if n < regla["max_palabras"] else None
        if tipo == "combinada":
            ma = next((m for p in self._patrones[(rid, "requiere_a")] if (m := p.search(plano))), None)
            mb = next((m for p in self._patrones[(rid, "requiere_b")] if (m := p.search(plano))), None)
            if ma and mb:
                return f"«{_fragmento(plano, mb.start(), mb.end())}»"
            return None
        raise ValueError(f"Tipo de regla desconocido: {tipo}")

    # ------------------------------------------------------------------------------- motor de inferencia
    def evaluar(self, titular: str, cuerpo: str) -> ResultadoReglas:
        """Aplica todas las reglas y devuelve puntaje, veredicto y explicación con evidencia."""
        titular, cuerpo = titular or "", cuerpo or ""
        activadas = []
        for r in self.reglas:
            evidencia = self._aplicar(r, titular, cuerpo)
            if evidencia is not None:
                activadas.append(
                    ReglaActivada(r["id"], r["nombre"], r["descripcion"], r["fundamento"],
                                  float(r["peso"]), evidencia)
                )
        puntaje = round(sum(a.peso for a in activadas), 2)
        return ResultadoReglas(puntaje=puntaje, sospechosa=puntaje >= self.umbral, activadas=activadas)

    def senales(self, titular: str, cuerpo: str) -> np.ndarray:
        """Vector binario (una posición por regla) con las reglas activadas."""
        act = {a.id for a in self.evaluar(titular, cuerpo).activadas}
        return np.array([1.0 if i in act else 0.0 for i in self.ids])

    def matriz_senales(self, df: pd.DataFrame) -> np.ndarray:
        """Matriz n × reglas para un DataFrame con columnas ``titular`` y ``cuerpo``."""
        return np.vstack([self.senales(t, c) for t, c in zip(df["titular"], df["cuerpo"])])


# ------------------------------------------------------------------------- adaptadores para scikit-learn
class ReglasTransformer(BaseEstimator, TransformerMixin):
    """Convierte un DataFrame (titular, cuerpo) en la matriz de señales de reglas. No aprende nada."""

    def fit(self, X, y=None):  # noqa: D102, N803
        self.sistema_ = SistemaExperto()
        return self

    def transform(self, X):  # noqa: D102, N803
        return self.sistema_.matriz_senales(X)


class ClasificadorReglas(BaseEstimator, ClassifierMixin):
    """Usa el sistema experto como clasificador (línea base no-ML, E-00b).

    ``fit`` no ajusta ningún parámetro con los datos; solo registra las clases.
    ``predict_proba`` aplica una función logística al puntaje centrada en el umbral, de modo que
    probabilidad >= 0,5 equivale a puntaje >= umbral.
    """

    def fit(self, X, y):  # noqa: D102, N803
        self.sistema_ = SistemaExperto()
        self.classes_ = np.array([0, 1])
        return self

    def _puntajes(self, X):  # noqa: N803
        return np.array([self.sistema_.evaluar(t, c).puntaje for t, c in zip(X["titular"], X["cuerpo"])])

    def predict_proba(self, X):  # noqa: D102, N803
        z = self._puntajes(X) - self.sistema_.umbral
        p = 1.0 / (1.0 + np.exp(-2.0 * z))
        p = np.where(z >= 0, np.maximum(p, 0.5), p)
        return np.column_stack([1 - p, p])

    def predict(self, X):  # noqa: D102, N803
        return (self._puntajes(X) >= self.sistema_.umbral).astype(int)
