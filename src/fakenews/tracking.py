"""Registro ligero de corridas (equivalente local a MLflow).

Cada corrida se agrega como una línea JSON en ``results/corridas.jsonl`` con: fecha, versión del código
(commit de git), versión de los datos, huella de la configuración, parámetros, métricas, duración y versiones
de las bibliotecas. Se eligió este formato en lugar de un servidor MLflow para que el registro quede dentro
del repositorio y se pueda revisar sin instalar nada.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from . import config


def version_codigo() -> str:
    """Commit corto de git (con sufijo ``+cambios`` si hay archivos sin confirmar)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.RAIZ, stderr=subprocess.DEVNULL, text=True
        ).strip()
        sucio = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src", "configs"], cwd=config.RAIZ, text=True
        ).strip()
        return commit + ("+cambios" if sucio else "")
    except (OSError, subprocess.CalledProcessError):
        return "sin-git"


def entorno() -> dict[str, str]:
    """Versiones relevantes del entorno de ejecución."""
    return {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sistema": platform.platform(terse=True),
    }


def registrar_corrida(registro: dict[str, Any], archivo: str = "results/corridas.jsonl") -> dict[str, Any]:
    """Agrega metadatos comunes al registro y lo guarda como una línea JSON."""
    completo = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version_codigo": version_codigo(),
        **registro,
        "entorno": entorno(),
    }
    ruta = config.ruta(archivo)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "a", encoding="utf-8") as f:
        f.write(json.dumps(completo, ensure_ascii=False, default=float) + "\n")
    return completo
