"""Lectura de la configuración del proyecto (archivos YAML en ``configs/``)."""

from __future__ import annotations

import hashlib
import json
import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

#: Raíz del repositorio. Se puede forzar con la variable de entorno ``FAKENEWS_ROOT``.
RAIZ = Path(os.environ.get("FAKENEWS_ROOT", Path(__file__).resolve().parents[2]))


def ruta(relativa: str | Path) -> Path:
    """Convierte una ruta relativa a la raíz del repositorio en absoluta."""
    return RAIZ / relativa


@cache
def cargar_yaml(nombre: str) -> dict[str, Any]:
    """Lee ``configs/<nombre>.yaml`` y devuelve su contenido como diccionario."""
    with open(ruta(f"configs/{nombre}.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def base() -> dict[str, Any]:
    """Configuración general (``configs/base.yaml``)."""
    return cargar_yaml("base")


def experimentos() -> dict[str, Any]:
    """Definición de experimentos (``configs/experimentos.yaml``)."""
    return cargar_yaml("experimentos")


def huella(obj: Any) -> str:
    """Hash corto y estable de un objeto serializable (para registrar configuraciones)."""
    texto = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:12]
