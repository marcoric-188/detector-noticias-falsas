"""Descarga, verificación, armonización y partición de los datos.

Uso desde la terminal::

    python -m fakenews.data descargar   # descarga los .xlsx y verifica SHA-256
    python -m fakenews.data preparar    # armoniza, normaliza y elimina casi duplicados entre particiones

Solo ``cargar_entrenamiento`` es de uso libre. Las particiones de prueba se leen exclusivamente desde
``fakenews.evaluate`` mediante ``cargar_prueba(..., confirmo_uso_unico=True)``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .preprocessing import construir_entrada, normalizar, quitar_titular_del_cuerpo

COLUMNAS = ["id", "etiqueta", "tema", "fuente", "fuente_norm", "titular", "cuerpo", "enlace", "texto"]
PARTICIONES = {"entrenamiento": "train", "prueba_dd": "prueba_dd", "prueba_fd": "prueba_fd"}


# ---------------------------------------------------------------------------------------------- descarga
def sha256_archivo(ruta: Path) -> str:
    """SHA-256 de un archivo."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 16), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(forzar: bool = False) -> dict[str, Path]:
    """Descarga los tres archivos del corpus desde el commit fijado y verifica su SHA-256.

    Raises:
        RuntimeError: si un hash no coincide (archivo alterado o versión distinta).
    """
    cfg = config.base()["datos"]
    destino = config.ruta(cfg["dir_crudos"])
    destino.mkdir(parents=True, exist_ok=True)
    rutas = {}
    for clave, info in cfg["archivos"].items():
        archivo = destino / info["nombre"]
        if forzar or not archivo.exists():
            url = f"{cfg['url_base']}/{info['nombre']}"
            print(f"Descargando {url}")
            urllib.request.urlretrieve(url, archivo)  # noqa: S310 (URL fija de configuración)
        obtenido = sha256_archivo(archivo)
        if obtenido != info["sha256"]:
            raise RuntimeError(
                f"El SHA-256 de {archivo.name} no coincide.\n  esperado: {info['sha256']}\n"
                f"  obtenido: {obtenido}\nBorre el archivo y vuelva a descargar."
            )
        print(f"  OK {info['nombre']} ({archivo.stat().st_size:,} bytes, sha256 {obtenido[:12]}…)")
        rutas[clave] = archivo
    return rutas


# --------------------------------------------------------------------------------------- armonización
def normalizar_fuente(nombre: object) -> str:
    """Nombre de medio comparable: minúsculas, sin tildes ni espacios extra (``El País`` -> ``el pais``)."""
    if nombre is None or (isinstance(nombre, float) and nombre != nombre):
        return "desconocida"
    s = unicodedata.normalize("NFKD", str(nombre).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.split()) or "desconocida"


def armonizar(df: pd.DataFrame) -> pd.DataFrame:
    """Lleva v1 y v2 al mismo esquema (``COLUMNAS``). Etiqueta: 1 = falsa, 0 = verdadera."""
    d = df.rename(columns=lambda c: str(c).strip().lower())
    d = d.rename(columns={"topics": "topic"})
    cat = d["category"]
    if cat.dtype == bool or set(cat.dropna().unique()) <= {True, False}:
        etiqueta = (~cat.astype(bool)).astype(int)  # v2: True = verdadera
    else:
        etiqueta = cat.astype(str).str.strip().str.lower().map({"fake": 1, "true": 0})
    if etiqueta.isna().any():
        raise ValueError("Hay etiquetas que no son True/Fake")
    out = pd.DataFrame(
        {
            "id": d["id"].astype(int),
            "etiqueta": etiqueta.astype(int),
            "tema": d["topic"].fillna("Sin tema").astype(str).str.strip(),
            "fuente": d["source"].fillna("desconocida").astype(str).str.strip(),
            "titular": d["headline"].fillna("").astype(str),
            "cuerpo": d["text"].fillna("").astype(str),
            "enlace": d["link"].fillna("").astype(str),
        }
    )
    out["fuente_norm"] = out["fuente"].map(normalizar_fuente)
    out["texto"] = [construir_entrada(t, c) for t, c in zip(out["titular"], out["cuerpo"])]
    out["titular"] = out["titular"].map(normalizar)
    out["cuerpo"] = [
        quitar_titular_del_cuerpo(t, normalizar(c)) for t, c in zip(out["titular"], out["cuerpo"])
    ]
    return out[COLUMNAS]


def casi_duplicados(referencia: pd.Series, candidatos: pd.Series, umbral: float) -> pd.DataFrame:
    """Para cada candidato, similitud coseno TF-IDF máxima contra la referencia.

    El vectorizador se ajusta SOLO con la referencia (entrenamiento) para no usar información de prueba.
    """
    vec = TfidfVectorizer(min_df=1).fit(referencia)
    sim = cosine_similarity(vec.transform(candidatos), vec.transform(referencia))
    idx = sim.argmax(axis=1)
    return pd.DataFrame(
        {"sim_max": sim.max(axis=1), "idx_ref": idx, "es_duplicado": sim.max(axis=1) >= umbral},
        index=candidatos.index,
    )


def preparar() -> dict:
    """Genera ``data/processed/{train,prueba_dd,prueba_fd}.csv`` y un reporte de deduplicación."""
    cfg = config.base()
    rutas = descargar()
    crudos = {k: armonizar(pd.read_excel(v)) for k, v in rutas.items()}
    tr, dd, fd = crudos["entrenamiento"], crudos["prueba_dd"], crudos["prueba_fd"]
    umbral = cfg["datos"]["umbral_casi_duplicado"]

    dup_dd = casi_duplicados(tr["texto"], dd["texto"], umbral)
    ref_fd = pd.concat([tr["texto"], dd["texto"]], ignore_index=True)
    dup_fd = casi_duplicados(ref_fd, fd["texto"], umbral)

    reporte = {
        "umbral_similitud": umbral,
        "prueba_dd": {
            "antes": len(dd),
            "eliminados": int(dup_dd["es_duplicado"].sum()),
            "ids_eliminados": dd.loc[dup_dd["es_duplicado"], "id"].tolist(),
        },
        "prueba_fd": {
            "antes": len(fd),
            "eliminados": int(dup_fd["es_duplicado"].sum()),
            "ids_eliminados": fd.loc[dup_fd["es_duplicado"], "id"].tolist(),
        },
        "duplicados_exactos_en_entrenamiento": int(tr["texto"].str.lower().duplicated().sum()),
    }
    dd = dd.loc[~dup_dd["es_duplicado"]].reset_index(drop=True)
    fd = fd.loc[~dup_fd["es_duplicado"]].reset_index(drop=True)
    reporte["prueba_dd"]["despues"] = len(dd)
    reporte["prueba_fd"]["despues"] = len(fd)
    reporte["sha256"] = {k: v["sha256"] for k, v in cfg["datos"]["archivos"].items()}

    destino = config.ruta(cfg["datos"]["dir_procesados"])
    destino.mkdir(parents=True, exist_ok=True)
    tr.to_csv(destino / "train.csv", index=False)
    dd.to_csv(destino / "prueba_dd.csv", index=False)
    fd.to_csv(destino / "prueba_fd.csv", index=False)

    tablas = config.ruta(cfg["salidas"]["dir_tablas"])
    tablas.mkdir(parents=True, exist_ok=True)
    with open(tablas / "deduplicacion.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)
    print(json.dumps(reporte, ensure_ascii=False, indent=2))
    return reporte


# ------------------------------------------------------------------------------------------- lectura
def _leer(nombre: str) -> pd.DataFrame:
    archivo = config.ruta(config.base()["datos"]["dir_procesados"]) / f"{nombre}.csv"
    if not archivo.exists():
        raise FileNotFoundError(
            f"No existe {archivo}. Ejecute primero: python -m fakenews.data preparar"
        )
    df = pd.read_csv(archivo, keep_default_na=False, dtype={"titular": str, "cuerpo": str, "texto": str})
    df["etiqueta"] = df["etiqueta"].astype(int)
    return df


def cargar_entrenamiento() -> pd.DataFrame:
    """Partición de entrenamiento (v1 ``train``). Es la única que usan los experimentos."""
    return _leer("train")


def cargar_prueba(nombre: str, *, confirmo_uso_unico: bool = False) -> pd.DataFrame:
    """Lee una partición de prueba (``prueba_dd`` o ``prueba_fd``).

    Exige confirmar explícitamente que se usa para la evaluación final única.
    """
    if nombre not in ("prueba_dd", "prueba_fd"):
        raise ValueError("nombre debe ser 'prueba_dd' o 'prueba_fd'")
    if not confirmo_uso_unico:
        raise PermissionError(
            "Las particiones de prueba solo se leen en la evaluación final (fakenews.evaluate)."
        )
    return _leer(nombre)


def huella_datos() -> str:
    """Identificador corto de la versión de los datos (hash de los tres SHA-256)."""
    return config.huella(config.base()["datos"]["archivos"])


def resumen(df: pd.DataFrame) -> dict:
    """Estadísticas básicas de una partición."""
    n_pal = df["texto"].str.split().str.len()
    return {
        "n": len(df),
        "falsas": int(df["etiqueta"].sum()),
        "verdaderas": int((df["etiqueta"] == 0).sum()),
        "fuentes": int(df["fuente_norm"].nunique()),
        "palabras_mediana": float(np.median(n_pal)),
    }


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la línea de comandos."""
    p = argparse.ArgumentParser(description="Gestión de datos del detector de noticias falsas")
    p.add_argument("accion", choices=["descargar", "preparar"])
    p.add_argument("--forzar", action="store_true", help="volver a descargar aunque exista")
    a = p.parse_args(argv)
    if a.accion == "descargar":
        descargar(forzar=a.forzar)
    else:
        preparar()
    return 0


if __name__ == "__main__":
    sys.exit(main())
