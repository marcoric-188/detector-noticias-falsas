"""Ejecución reproducible de la bitácora de experimentos (solo con la partición de entrenamiento).

Uso::

    python -m fakenews.experiments              # todos los experimentos de configs/experimentos.yaml
    python -m fakenews.experiments E-02 E-03    # solo algunos

Salidas:
    results/bitacora.csv               una fila por experimento y esquema de validación
    results/tablas/pliegues.csv        métricas de cada pliegue (para medir variabilidad)
    results/corridas.jsonl             registro de cada corrida con versión de código, datos y entorno
    results/seleccion.json             modelo elegido con la regla fijada en docs/plan_evaluacion.md
    results/figuras/*.png              comparación de experimentos y curva de aprendizaje
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, learning_curve

from . import config
from .data import cargar_entrenamiento, huella_datos
from .metrics import calcular
from .models import COMPLEJIDAD, construir_modelo, familia
from .tracking import registrar_corrida, version_codigo

ESQUEMAS = ("estratificada", "agrupada")
METRICAS_RESUMEN = ["f1_macro", "exactitud", "recall_falsa", "precision_falsa", "especificidad", "roc_auc"]


def pliegues(esquema: str, y: np.ndarray, grupos: np.ndarray, semilla: int, n: int):
    """Genera los índices de los pliegues para un esquema de validación."""
    if esquema == "estratificada":
        return StratifiedKFold(n_splits=n, shuffle=True, random_state=semilla).split(np.zeros(len(y)), y)
    if esquema == "agrupada":
        return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=semilla).split(
            np.zeros(len(y)), y, grupos
        )
    raise ValueError(esquema)


def validar(spec: dict, df: pd.DataFrame, esquema: str) -> list[dict]:
    """Validación cruzada repetida de un experimento. Devuelve las métricas de cada pliegue."""
    cfg = config.base()["validacion"]
    X = df[["titular", "cuerpo"]]  # noqa: N806
    y = df["etiqueta"].to_numpy()
    grupos = df["fuente_norm"].to_numpy()
    filas = []
    for semilla in cfg["semillas"]:
        for k, (itr, iva) in enumerate(pliegues(esquema, y, grupos, semilla, cfg["n_pliegues"])):
            # Verificación explícita de no-fuga en cada pliegue
            assert not set(itr) & set(iva), "índices compartidos entre entrenamiento y validación"
            if esquema == "agrupada":
                assert not set(grupos[itr]) & set(grupos[iva]), "una fuente aparece en ambos lados"
            modelo = construir_modelo(spec, semilla)
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                modelo.fit(X.iloc[itr], y[itr])
            t_fit = time.perf_counter() - t0
            proba = modelo.predict_proba(X.iloc[iva])[:, 1]
            pred = modelo.predict(X.iloc[iva])
            m = calcular(y[iva], pred, proba)
            m["f1_macro_entrenamiento"] = calcular(y[itr], modelo.predict(X.iloc[itr]))["f1_macro"]
            m.update(semilla=semilla, pliegue=k, esquema=esquema, segundos_ajuste=t_fit, n_val=len(iva))
            if hasattr(modelo, "best_params_"):
                m["mejores_parametros"] = json.dumps(modelo.best_params_)
            filas.append(m)
    return filas


def resumir(eid: str, exp: dict, filas: list[dict], esquema: str, segundos: float) -> dict:
    """Resume los pliegues de un experimento en media ± desviación."""
    d = pd.DataFrame(filas)
    r = {"id": eid, "esquema": esquema, "cambio": exp["cambio"], "compara_con": exp.get("compara_con")}
    for met in METRICAS_RESUMEN + ["f1_macro_entrenamiento"]:
        if met in d:
            r[f"{met}_media"] = round(float(d[met].mean()), 4)
            r[f"{met}_de"] = round(float(d[met].std(ddof=1)), 4)
    r["brecha_sobreajuste"] = round(r["f1_macro_entrenamiento_media"] - r["f1_macro_media"], 4)
    r["n_pliegues"] = len(d)
    r["segundos_total"] = round(segundos, 1)
    if "mejores_parametros" in d:
        r["mejores_parametros_frecuentes"] = d["mejores_parametros"].value_counts().head(3).to_dict()
    return r


def seleccionar(bitacora: pd.DataFrame, exps: dict) -> dict:
    """Aplica la regla de selección fijada a priori (docs/plan_evaluacion.md, sección 5).

    1. Candidatos: modelos de ML que no son línea base (logreg, svm, hibrido).
    2. Mayor F1-macro medio en validación agrupada por fuente.
    3. Entre los que quedan a menos de una desviación estándar del mejor, el más simple (``COMPLEJIDAD``);
       si empatan en simplicidad, el de mayor media.
    """
    ag = bitacora[bitacora["esquema"] == "agrupada"].copy()
    ag = ag[ag["id"].map(lambda i: exps[i]["modelo"]["tipo"] in ("logreg", "svm", "hibrido"))]
    mejor = ag.loc[ag["f1_macro_media"].idxmax()]
    cerca = ag[ag["f1_macro_media"] >= mejor["f1_macro_media"] - mejor["f1_macro_de"]].copy()
    cerca["complejidad"] = cerca["id"].map(lambda i: COMPLEJIDAD[familia(exps[i]["modelo"])])
    elegido = cerca.sort_values(["complejidad", "f1_macro_media"], ascending=[True, False]).iloc[0]
    return {
        "elegido": elegido["id"],
        "mejor_por_media": mejor["id"],
        "f1_macro_agrupada_mejor": float(mejor["f1_macro_media"]),
        "de_mejor": float(mejor["f1_macro_de"]),
        "candidatos_dentro_de_1_de": cerca["id"].tolist(),
        "f1_macro_agrupada_elegido": float(elegido["f1_macro_media"]),
        "modelo": exps[elegido["id"]]["modelo"],
        "regla": "docs/plan_evaluacion.md §5",
    }


def curva_aprendizaje(spec: dict, df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """Curva de aprendizaje con validación agrupada por fuente (diagnóstico de sobreajuste/subajuste)."""
    import matplotlib.pyplot as plt

    from .plots import estilo, guardar

    cfg = config.base()["validacion"]
    X, y = df[["titular", "cuerpo"]], df["etiqueta"].to_numpy()  # noqa: N806
    cv = StratifiedGroupKFold(n_splits=cfg["n_pliegues"], shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tam, tr, va = learning_curve(
            clone(construir_modelo(spec, 42)), X, y, groups=df["fuente_norm"], cv=cv,
            train_sizes=np.linspace(0.15, 1.0, 7), scoring="f1_macro", n_jobs=1,
        )
    tabla = pd.DataFrame({
        "n_entrenamiento": tam,
        "f1_entrenamiento_media": tr.mean(1), "f1_entrenamiento_de": tr.std(1, ddof=1),
        "f1_validacion_media": va.mean(1), "f1_validacion_de": va.std(1, ddof=1),
    })
    tabla.to_csv(config.ruta("results/tablas/curva_aprendizaje.csv"), index=False)

    estilo()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for col, etiqueta, color in [("f1_entrenamiento", "Entrenamiento", "#2a78d6"),
                                 ("f1_validacion", "Validación (agrupada por fuente)", "#eb6834")]:
        m, s = tabla[f"{col}_media"], tabla[f"{col}_de"]
        ax.plot(tam, m, marker="o", lw=2, ms=6, color=color, label=etiqueta)
        ax.fill_between(tam, m - s, m + s, color=color, alpha=0.15, lw=0)
    ax.set_xlabel("Noticias de entrenamiento (cantidad)")
    ax.set_ylabel("F1-macro (0 a 1)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(f"Curva de aprendizaje · {nombre}\nbanda = ± 1 desviación estándar entre 5 pliegues")
    ax.legend(loc="lower right", frameon=False)
    guardar(fig, "curva_aprendizaje.png")
    return tabla


def figura_comparacion(bitacora: pd.DataFrame) -> None:
    """Gráfica de puntos con F1-macro (media ± DE) de cada experimento en ambos esquemas."""
    import matplotlib.pyplot as plt

    from .plots import estilo, guardar

    estilo()
    ids = list(dict.fromkeys(bitacora["id"]))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(ids) + 1.4))
    for desplaz, (esq, color, etiqueta) in zip(
        (-0.15, 0.15),
        [("estratificada", "#2a78d6", "CV estratificada"), ("agrupada", "#eb6834", "CV agrupada por fuente")],
    ):
        sub = bitacora[bitacora["esquema"] == esq].set_index("id").loc[ids]
        pos = np.arange(len(ids)) + desplaz
        ax.errorbar(sub["f1_macro_media"], pos, xerr=sub["f1_macro_de"], fmt="o", ms=7, color=color,
                    elinewidth=2, capsize=0, label=etiqueta)
    ax.set_yticks(range(len(ids)), ids)
    ax.invert_yaxis()
    ax.set_xlabel("F1-macro en validación cruzada (0 a 1), media ± 1 DE de 15 pliegues")
    ax.set_title("Comparación de experimentos (solo partición de entrenamiento)")
    ax.axvline(0.5, color="#a3a29d", lw=1, ls=":")
    ax.legend(loc="lower left", frameon=False)
    guardar(fig, "comparacion_experimentos.png")


def ejecutar(ids: list[str] | None = None) -> pd.DataFrame:
    """Ejecuta los experimentos indicados (todos por defecto) y actualiza la bitácora."""
    exps = config.experimentos()
    ids = ids or list(exps)
    df = cargar_entrenamiento()
    ruta_bit = config.ruta("results/bitacora.csv")
    ruta_pl = config.ruta("results/tablas/pliegues.csv")
    previa = pd.read_csv(ruta_bit) if ruta_bit.exists() else pd.DataFrame()
    previa_pl = pd.read_csv(ruta_pl) if ruta_pl.exists() else pd.DataFrame()

    filas_bit, filas_pl = [], []
    for eid in ids:
        exp = exps[eid]
        for esquema in ESQUEMAS:
            t0 = time.perf_counter()
            filas = validar(exp["modelo"], df, esquema)
            seg = time.perf_counter() - t0
            r = resumir(eid, exp, filas, esquema, seg)
            r.update(fecha=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), version_codigo=version_codigo(),
                     version_datos=huella_datos(), huella_config=config.huella(exp["modelo"]))
            filas_bit.append(r)
            filas_pl += [{"id": eid, **f} for f in filas]
            registrar_corrida({"tipo": "validacion_cruzada", "experimento": eid, "esquema": esquema,
                               "pregunta": exp["pregunta"], "parametros": exp["modelo"],
                               "version_datos": huella_datos(), "particion": "train (v1)",
                               "metricas": {k: v for k, v in r.items() if k.endswith(("_media", "_de"))},
                               "segundos": round(seg, 1)})
            print(f"{eid:6s} {esquema:13s} F1-macro {r['f1_macro_media']:.3f} ± {r['f1_macro_de']:.3f}"
                  f"  (entren. {r['f1_macro_entrenamiento_media']:.3f})  {seg:6.1f} s")

    nueva = pd.DataFrame(filas_bit)
    if not previa.empty:
        previa = previa[~previa["id"].isin(ids)]
        nueva = pd.concat([previa, nueva], ignore_index=True)
    orden = {e: i for i, e in enumerate(exps)}
    nueva = nueva.sort_values(["id", "esquema"], key=lambda s: s.map(orden) if s.name == "id" else s)
    nueva.to_csv(ruta_bit, index=False)
    pl = pd.DataFrame(filas_pl)
    if not previa_pl.empty:
        pl = pd.concat([previa_pl[~previa_pl["id"].isin(ids)], pl], ignore_index=True)
    pl.to_csv(ruta_pl, index=False)

    if set(exps) <= set(nueva["id"]):
        sel = seleccionar(nueva, exps)
        with open(config.ruta("results/seleccion.json"), "w", encoding="utf-8") as f:
            json.dump(sel, f, ensure_ascii=False, indent=2)
        print("Modelo elegido con la regla a priori:", sel["elegido"])
        figura_comparacion(nueva)
        curva_aprendizaje(sel["modelo"], df, sel["elegido"])
    return nueva


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la línea de comandos."""
    p = argparse.ArgumentParser(description="Bitácora de experimentos (validación cruzada)")
    p.add_argument("ids", nargs="*", help="IDs de experimentos (por defecto, todos)")
    a = p.parse_args(argv)
    ejecutar(a.ids or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
