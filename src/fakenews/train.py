"""Entrenamiento y empaquetado del modelo de producción versionado.

El modelo de producción usa la configuración elegida por la bitácora (``results/seleccion.json``) con los
hiperparámetros fijados en la evaluación final (``results/metricas_finales.json``) y se entrena con las 971
noticias de v1 (entrenamiento + prueba DD), igual que en FakeDeS 2021.

    python -m fakenews.train

Este comando NO evalúa nada: solo reconstruye ``models/detector_v<versión>.joblib`` (por ejemplo, si la
versión de scikit-learn del entorno es distinta de la usada al guardar el modelo).
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
import sklearn

from . import __version__, config
from .data import cargar_entrenamiento, cargar_prueba, huella_datos
from .models import construir_modelo
from .rules import SistemaExperto
from .tracking import entorno, version_codigo


def spec_final() -> dict[str, Any]:
    """Especificación del modelo final sin búsqueda (hiperparámetros ya fijados)."""
    sel = json.loads(config.ruta("results/seleccion.json").read_text(encoding="utf-8"))
    spec = dict(sel["modelo"])
    ruta_m = config.ruta("results/metricas_finales.json")
    if spec.get("busqueda"):
        if not ruta_m.exists():
            raise RuntimeError("Primero ejecute la evaluación final (python -m fakenews.evaluate ...)")
        fijos = json.loads(ruta_m.read_text(encoding="utf-8"))["hiperparametros_fijados"]
        spec = fijar_hiperparametros(spec, fijos)
    return spec


def fijar_hiperparametros(spec: dict[str, Any], mejores: dict[str, Any]) -> dict[str, Any]:
    """Convierte ``{'clf__C': 3, 'tfidf__min_df': 2}`` en parámetros fijos de la especificación."""
    s = json.loads(json.dumps(spec))
    s.pop("busqueda", None)
    for clave, valor in mejores.items():
        bloque, param = clave.split("__", 1)
        s.setdefault(bloque, {})[param] = valor
    return s


def empaquetar(modelo, spec: dict[str, Any], datos_entrenamiento: str, extra: dict | None = None) -> dict:
    """Paquete serializable: modelo + metadatos para trazabilidad."""
    return {
        "modelo": modelo,
        "metadatos": {
            "version": config.base()["proyecto"]["version_modelo"],
            "paquete_fakenews": __version__,
            "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "spec": spec,
            "datos_entrenamiento": datos_entrenamiento,
            "version_datos": huella_datos(),
            "version_reglas": SistemaExperto().version,
            "version_codigo": version_codigo(),
            "sklearn": sklearn.__version__,
            "entorno": entorno(),
            **(extra or {}),
        },
    }


def entrenar_produccion(spec: dict[str, Any] | None = None, extra: dict | None = None):
    """Entrena con entrenamiento + prueba DD y guarda el modelo versionado."""
    from .predict import ruta_modelo_por_defecto

    spec = spec or spec_final()
    tr = cargar_entrenamiento()
    dd = cargar_prueba("prueba_dd", confirmo_uso_unico=True)  # se usa para ENTRENAR, no para evaluar
    datos = pd.concat([tr, dd], ignore_index=True)
    modelo = construir_modelo(spec, 42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(datos[["titular", "cuerpo"]], datos["etiqueta"].to_numpy())
    ruta = ruta_modelo_por_defecto()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(empaquetar(modelo, spec, f"v1 train + development ({len(datos)} noticias)", extra), ruta,
                compress=3)
    print(f"Modelo guardado en {ruta}")
    return modelo


def main() -> int:
    """Punto de entrada de la línea de comandos."""
    ruta_m = config.ruta("results/metricas_finales.json")
    extra = None
    if ruta_m.exists():
        fin = json.loads(ruta_m.read_text(encoding="utf-8"))
        extra = {"metricas_prueba_fd": fin.get("prueba_fd", {}).get("modelo_final", {}).get("metricas")}
    entrenar_produccion(extra=extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
