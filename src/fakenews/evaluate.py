"""Evaluación final ÚNICA en las particiones de prueba (DD y FD).

    python -m fakenews.evaluate --confirmo-uso-unico

Protocolo (docs/plan_evaluacion.md):
1. Configuración elegida por la regla a priori (``results/seleccion.json``). Si tiene búsqueda de
   hiperparámetros, se resuelve con CV interna SOLO en entrenamiento.
2. Prueba DD: modelo final, línea base NB (E-01) y sistema de reglas (E-00b), entrenados con entrenamiento.
3. Prueba FD: los mismos, reentrenados con entrenamiento + DD (como en FakeDeS 2021). También se reporta el
   modelo entrenado solo con entrenamiento.
4. Intervalos bootstrap, subgrupos, abstención, evaluación de reglas, latencia, fidelidad de explicaciones,
   tablas de errores y figuras.
5. Se guarda el modelo de producción (entrenamiento + DD) con versión.

Deja constancia en ``results/uso_prueba.log``. Si ya existe, se niega a repetir sin ``--forzar``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from . import config
from .data import cargar_entrenamiento, cargar_prueba, resumen
from .metrics import bootstrap_ic, calcular
from .models import construir_modelo
from .predict import Detector, EntradaInvalida, evidencia_fiel, ruta_modelo_por_defecto
from .rules import SistemaExperto
from .tracking import registrar_corrida
from .train import empaquetar, fijar_hiperparametros

LOG = "results/uso_prueba.log"
PAISES = {"co": "Colombia", "mx": "México", "es": "España", "ar": "Argentina", "cl": "Chile", "pe": "Perú",
          "ve": "Venezuela", "ec": "Ecuador", "bo": "Bolivia", "uy": "Uruguay", "cr": "Costa Rica"}


# ----------------------------------------------------------------------------------------- utilidades
def _ajustar(spec, df, semilla=42):
    modelo = construir_modelo(spec, semilla)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(df[["titular", "cuerpo"]], df["etiqueta"].to_numpy())
    return modelo


def _predecir(modelo, df):
    X = df[["titular", "cuerpo"]]  # noqa: N806
    return modelo.predict(X), modelo.predict_proba(X)[:, 1]


def pais_por_enlace(enlace: str) -> str:
    """Aproxima el país del medio por el dominio de nivel superior del enlace (.co -> Colombia)."""
    host = urlparse(enlace).netloc.lower() if enlace else ""
    if not host:
        return "Sin enlace"
    tld = host.rsplit(".", 1)[-1]
    return PAISES.get(tld, "Dominio genérico (.com, .org…)")


def tercil_longitud(df: pd.DataFrame, cortes: list[float] | None = None) -> tuple[pd.Series, list[float]]:
    """Tercil de longitud en palabras (cortes calculados en entrenamiento)."""
    n = df["texto"].str.split().str.len()
    if cortes is None:
        raise ValueError("cortes requeridos")
    etiquetas = [f"corta (< {cortes[0]:.0f})", f"media ({cortes[0]:.0f}–{cortes[1]:.0f})",
                 f"larga (> {cortes[1]:.0f})"]
    return pd.cut(n, [-1, cortes[0], cortes[1], np.inf], labels=etiquetas).astype(str), cortes


def por_subgrupo(df: pd.DataFrame, columna: str, pred, proba, minimo: int) -> pd.DataFrame:
    """Métricas por subgrupo; marca los subgrupos con pocos casos como no interpretables."""
    filas = []
    for valor, idx in df.groupby(columna).groups.items():
        pos = df.index.get_indexer(idx)
        y = df["etiqueta"].to_numpy()[pos]
        m = calcular(y, pred[pos], proba[pos] if len(np.unique(y)) == 2 else None)
        filas.append({"variable": columna, "subgrupo": valor, "n": len(pos), "falsas": int(y.sum()),
                      "f1_macro": m["f1_macro"], "exactitud": m["exactitud"],
                      "recall_falsa": m["recall_falsa"], "especificidad": m["especificidad"],
                      "interpretable": len(pos) >= minimo and 0 < y.sum() < len(y)})
    return pd.DataFrame(filas).sort_values("n", ascending=False)


def abstencion(y, proba, inf, sup) -> dict:
    """Cobertura y desempeño cuando el sistema se abstiene en la zona [inf, sup]."""
    cubierto = (proba < inf) | (proba > sup)
    pred = (proba >= 0.5).astype(int)
    m = calcular(y[cubierto], pred[cubierto]) if cubierto.sum() else {}
    return {"zona": [inf, sup], "cobertura": float(cubierto.mean()), "n_indeterminadas": int((~cubierto).sum()),
            "f1_macro_cubiertos": m.get("f1_macro"), "exactitud_cubiertos": m.get("exactitud"),
            "exactitud_indeterminadas": float((pred[~cubierto] == y[~cubierto]).mean())
            if (~cubierto).sum() else None}


def evaluar_reglas(sistema: SistemaExperto, df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """Soporte y precisión de cada regla respecto a la clase «falsa»."""
    S = sistema.matriz_senales(df)  # noqa: N806
    y = df["etiqueta"].to_numpy()
    filas = []
    for j, r in enumerate(sistema.reglas):
        act = S[:, j] == 1
        filas.append({"conjunto": nombre, "regla": r["id"], "nombre": r["nombre"], "peso": r["peso"],
                      "soporte": float(act.mean()), "n_activa": int(act.sum()),
                      "prop_falsas_si_activa": float(y[act].mean()) if act.any() else np.nan,
                      "prop_falsas_si_no_activa": float(y[~act].mean()) if (~act).any() else np.nan})
    return pd.DataFrame(filas)


# -------------------------------------------------------------------------------------------- figuras
def figuras(res: dict) -> None:
    """Matrices de confusión, curvas ROC/PR y calibración del modelo final en ambas pruebas."""
    import matplotlib.pyplot as plt

    from .plots import AZUL, NARANJA, TEXTO_2, estilo, guardar

    estilo()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (clave, titulo) in zip(axes, [("dd", "Prueba DD (v1, 2018)"), ("fd", "Prueba FD (v2, 2020–21)")]):
        y, pred = res[clave]["y"], res[clave]["pred"]
        cm = confusion_matrix(y, pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.0%} de la fila)", ha="center",
                        va="center", color="white" if cm[i, j] > cm.max() * 0.6 else "#0b0b0b", fontsize=10)
        ax.set_xticks([0, 1], ["Verdadera", "Falsa"])
        ax.set_yticks([0, 1], ["Verdadera", "Falsa"])
        ax.set_xlabel("Predicción del modelo")
        ax.set_ylabel("Etiqueta real")
        ax.grid(False)
        ax.set_title(f"{titulo}\nn = {len(y)} noticias")
    fig.suptitle("Matrices de confusión del modelo final (umbral 0,5)", x=0.02, ha="left", fontweight="bold")
    guardar(fig, "matrices_confusion.png")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for clave, color, nombre in [("dd", AZUL, "Prueba DD"), ("fd", NARANJA, "Prueba FD")]:
        y, p = res[clave]["y"], res[clave]["proba"]
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        a1.plot(fpr, tpr, lw=2, color=color, label=f"{nombre} (AUC {res[clave]['metricas']['roc_auc']:.2f})")
        a2.plot(rc, pr, lw=2, color=color, label=f"{nombre} (AP {res[clave]['metricas']['pr_auc']:.2f})")
    a1.plot([0, 1], [0, 1], ls=":", color=TEXTO_2, lw=1, label="Azar")
    a1.set_xlabel("Tasa de falsas acusaciones (1 − especificidad)")
    a1.set_ylabel("Recall de «falsa» (sensibilidad)")
    a1.set_title("Curva ROC")
    a2.axhline(0.5, ls=":", color=TEXTO_2, lw=1, label="Azar (clases balanceadas)")
    a2.set_xlabel("Recall de «falsa»")
    a2.set_ylabel("Precisión de «falsa»")
    a2.set_title("Curva precisión–recall")
    for a in (a1, a2):
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.02)
        a.legend(loc="lower right" if a is a1 else "lower left", frameon=False)
    guardar(fig, "curvas_roc_pr.png")

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    for clave, color, nombre in [("dd", AZUL, "Prueba DD"), ("fd", NARANJA, "Prueba FD")]:
        frac, media = calibration_curve(res[clave]["y"], res[clave]["proba"], n_bins=8, strategy="quantile")
        ax.plot(media, frac, marker="o", ms=6, lw=2, color=color,
                label=f"{nombre} (Brier {res[clave]['metricas']['brier']:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color=TEXTO_2, lw=1, label="Calibración perfecta")
    ax.set_xlabel("Probabilidad de «falsa» predicha (promedio por grupo)")
    ax.set_ylabel("Proporción real de falsas en el grupo")
    ax.set_title("Calibración de las probabilidades\n(8 grupos de igual tamaño)")
    ax.legend(loc="upper left", frameon=False)
    guardar(fig, "calibracion.png")


def figura_subgrupos(tabla: pd.DataFrame, nombre_archivo: str, titulo: str) -> None:
    """Puntos de F1-macro por subgrupo (solo subgrupos interpretables), con n en la etiqueta."""
    import matplotlib.pyplot as plt

    from .plots import AZUL, TEXTO_2, estilo, guardar

    t = tabla[tabla["interpretable"]].sort_values("f1_macro")
    if t.empty:
        return
    estilo()
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(t) + 1.3))
    etiquetas = [f"{v} · {s}  (n={n})" for v, s, n in zip(t["variable"], t["subgrupo"], t["n"])]
    ax.hlines(range(len(t)), 0, t["f1_macro"], color="#d9d8d3", lw=2)
    ax.plot(t["f1_macro"], range(len(t)), "o", ms=7, color=AZUL)
    for i, v in enumerate(t["f1_macro"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", color=TEXTO_2, fontsize=9)
    ax.set_yticks(range(len(t)), etiquetas)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1-macro (0 a 1)")
    ax.set_title(titulo)
    guardar(fig, nombre_archivo)


# ------------------------------------------------------------------------------------------ principal
def evaluar(forzar: bool = False) -> dict:
    """Ejecuta la evaluación final única. Ver docstring del módulo."""
    cfg = config.base()
    log = config.ruta(LOG)
    if log.exists() and not forzar:
        raise SystemExit(
            f"La prueba ya se usó (ver {LOG}). Repetirla invalidaría el protocolo. "
            "Si es imprescindible, use --forzar y declárelo en el informe."
        )
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  evaluación final"
                f"{' (FORZADA: repetición)' if forzar else ''}\n")

    t_inicio = time.perf_counter()
    exps = config.experimentos()
    sel = json.loads(config.ruta("results/seleccion.json").read_text(encoding="utf-8"))
    spec = sel["modelo"]
    tr = cargar_entrenamiento()

    hiper = {}
    if spec.get("busqueda"):  # resolver la búsqueda solo con entrenamiento
        gs = _ajustar(spec, tr)
        hiper = gs.best_params_
        spec = fijar_hiperparametros(spec, hiper)
    spec_nb = exps["E-01"]["modelo"]

    dd = cargar_prueba("prueba_dd", confirmo_uso_unico=True)
    fd = cargar_prueba("prueba_fd", confirmo_uso_unico=True)
    trdd = pd.concat([tr, dd], ignore_index=True)
    bcfg, inf_sup = cfg["evaluacion"], cfg["decision"]["abstencion"]
    B, S = bcfg["bootstrap_remuestreos"], bcfg["bootstrap_semilla"]  # noqa: N806

    sistema = SistemaExperto()
    reglas_clf = construir_modelo({"tipo": "reglas"}).fit(tr[["titular", "cuerpo"]], tr["etiqueta"])

    salida = {"seleccion": sel["elegido"], "spec_final": spec, "hiperparametros_fijados": hiper,
              "resumen_particiones": {"entrenamiento": resumen(tr), "prueba_dd": resumen(dd),
                                      "prueba_fd": resumen(fd)}}
    figs = {}
    modelos = {}
    for clave, df_ev, df_fit, desc in [("prueba_dd", dd, tr, "entrenamiento (676)"),
                                       ("prueba_fd", fd, trdd, "entrenamiento + DD (966)")]:
        y = df_ev["etiqueta"].to_numpy()
        final = _ajustar(spec, df_fit)
        nb = _ajustar(spec_nb, df_fit)
        modelos[clave] = final
        p_fin, q_fin = _predecir(final, df_ev)
        p_nb, q_nb = _predecir(nb, df_ev)
        p_rg, q_rg = _predecir(reglas_clf, df_ev)
        bloque = {
            "entrenado_con": desc,
            "modelo_final": {
                "metricas": calcular(y, p_fin, q_fin),
                "ic95": {m: bootstrap_ic(y, p_fin, m, B, S) for m in ("f1_macro", "especificidad",
                                                                       "recall_falsa", "exactitud")},
            },
            "linea_base_nb": {"metricas": calcular(y, p_nb, q_nb)},
            "sistema_reglas": {"metricas": calcular(y, p_rg, q_rg)},
            "diferencia_vs_nb_f1_macro": bootstrap_ic(y, p_fin, "f1_macro", B, S, pred_base=p_nb),
            "diferencia_vs_reglas_f1_macro": bootstrap_ic(y, p_fin, "f1_macro", B, S, pred_base=p_rg),
            "abstencion": abstencion(y, q_fin, *inf_sup),
        }
        if clave == "prueba_fd":  # también el modelo entrenado solo con entrenamiento
            solo_tr = modelos["prueba_dd"]
            p2, q2 = _predecir(solo_tr, df_ev)
            bloque["modelo_final_solo_entrenamiento"] = {
                "metricas": calcular(y, p2, q2), "ic95": {"f1_macro": bootstrap_ic(y, p2, "f1_macro", B, S)}}
        salida[clave] = bloque
        figs[clave[-2:]] = {"y": y, "pred": p_fin, "proba": q_fin, "metricas": bloque["modelo_final"]["metricas"]}

        # Subgrupos
        ev = df_ev.copy()
        cortes = list(np.percentile(tr["texto"].str.split().str.len(), [33.33, 66.67]))
        ev["longitud"], _ = tercil_longitud(ev, cortes)
        variables = ["tema", "longitud"]
        if clave == "prueba_fd":
            ev["titular_presente"] = np.where(ev["titular"].str.strip() == "", "sin titular", "con titular")
            ev["pais_dominio"] = ev["enlace"].map(pais_por_enlace)
            variables += ["titular_presente", "pais_dominio"]
        sub = pd.concat([por_subgrupo(ev, v, p_fin, q_fin, bcfg["min_casos_subgrupo"]) for v in variables])
        sub.to_csv(config.ruta(f"results/tablas/subgrupos_{clave}.csv"), index=False)
        figura_subgrupos(sub, f"subgrupos_{clave}.png",
                         f"F1-macro por subgrupo · {'Prueba DD (v1)' if clave == 'prueba_dd' else 'Prueba FD (v2)'}"
                         f"\nsolo subgrupos con ≥ {bcfg['min_casos_subgrupo']} casos y ambas clases")

        # Tabla de predicciones para análisis de errores
        det_expl = []
        from .predict import explicar_terminos
        for t, c in zip(ev["titular"], ev["cuerpo"]):
            ex = explicar_terminos(final, f"{t}\n{c}".strip())
            r = sistema.evaluar(t, c)
            det_expl.append({"terminos_hacia_falsa": ", ".join(x["termino"] for x in ex["hacia_falsa"][:6]),
                             "terminos_hacia_verdadera": ", ".join(x["termino"] for x in ex["hacia_verdadera"][:4]),
                             "reglas_activadas": ", ".join(a.id for a in r.activadas),
                             "puntaje_reglas": r.puntaje})
        tabla = pd.concat([ev[["id", "etiqueta", "tema", "fuente", "enlace", "longitud"]].reset_index(drop=True),
                           pd.DataFrame({"prob_falsa": q_fin.round(4), "pred": p_fin}),
                           pd.DataFrame(det_expl),
                           pd.DataFrame({"titular": ev["titular"].str[:160].values,
                                         "inicio_cuerpo": ev["cuerpo"].str[:400].str.replace("\n", " ").values})],
                          axis=1)
        tabla["resultado"] = np.select(
            [(tabla.etiqueta == 1) & (tabla.pred == 1), (tabla.etiqueta == 0) & (tabla.pred == 0),
             (tabla.etiqueta == 0) & (tabla.pred == 1)], ["VP", "VN", "FP"], "FN")
        tabla.to_csv(config.ruta(f"results/tablas/predicciones_{clave}.csv"), index=False)

    # Base de conocimiento: evaluación de cada regla (entrenamiento, DD y FD)
    pd.concat([evaluar_reglas(sistema, tr, "entrenamiento"), evaluar_reglas(sistema, dd, "prueba_dd"),
               evaluar_reglas(sistema, fd, "prueba_fd")]).to_csv(
        config.ruta("results/tablas/evaluacion_reglas.csv"), index=False)

    # Modelo de producción (entrenamiento + DD) = el mismo objeto evaluado en FD
    ruta = ruta_modelo_por_defecto()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(empaquetar(modelos["prueba_fd"], spec, "v1 train + development sin duplicados (966 noticias)",
                           {"metricas_prueba_fd": salida["prueba_fd"]["modelo_final"]["metricas"],
                            "seleccion": sel["elegido"]}), ruta, compress=3)

    # Latencia y fidelidad de explicaciones con el Detector (como en la app)
    det = Detector(ruta)
    lat, fieles, total, rechazadas = [], 0, 0, 0
    for t, c in zip(pd.concat([dd, fd])["titular"], pd.concat([dd, fd])["cuerpo"]):
        try:
            r = det.predecir(t, c)
        except EntradaInvalida:
            rechazadas += 1
            continue
        lat.append(r["latencia_ms"])
        total += 1
        fieles += evidencia_fiel(r, t, c)
    salida["app"] = {"latencia_ms_mediana": float(np.median(lat)), "latencia_ms_p95": float(np.percentile(lat, 95)),
                     "explicaciones_fieles": fieles, "explicaciones_total": total,
                     "entradas_rechazadas_por_validacion": rechazadas}

    figuras(figs)

    # Criterios de éxito fijados a priori
    dd_m, fd_m = salida["prueba_dd"], salida["prueba_fd"]
    bit = pd.read_csv(config.ruta("results/bitacora.csv"))
    ag = bit[bit.esquema == "agrupada"].set_index("id")
    sel_ag, reg_ag = ag.loc[sel["elegido"]], ag.loc["E-00b"]
    criterios = {
        "CE1_f1_dd_>=_0.77": dd_m["modelo_final"]["metricas"]["f1_macro"] >= 0.77,
        "CE2_mejora_vs_nb_ic_sin_0": dd_m["diferencia_vs_nb_f1_macro"]["ic95_inf"] > 0,
        "CE3_f1_fd_>=_0.70": fd_m["modelo_final"]["metricas"]["f1_macro"] >= 0.70,
        "CE4_especificidad_dd_>=_0.80": dd_m["modelo_final"]["metricas"]["especificidad"] >= 0.80,
        "CE5_supera_reglas_cv_agrupada": bool(sel_ag.f1_macro_media - reg_ag.f1_macro_media > sel_ag.f1_macro_de),
        "CE6_explicaciones_fieles_100": fieles == total,
        "CE7_latencia_mediana_<_1000ms": salida["app"]["latencia_ms_mediana"] < 1000,
    }
    salida["criterios_exito"] = {k: bool(v) for k, v in criterios.items()}
    salida["segundos"] = round(time.perf_counter() - t_inicio, 1)
    with open(config.ruta("results/metricas_finales.json"), "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2, default=float)
    registrar_corrida({"tipo": "evaluacion_final", "experimento": sel["elegido"], "spec": spec,
                       "metricas_dd": dd_m["modelo_final"]["metricas"], "metricas_fd": fd_m["modelo_final"]["metricas"],
                       "criterios": salida["criterios_exito"]})
    print(json.dumps({"dd": dd_m["modelo_final"]["metricas"], "fd": fd_m["modelo_final"]["metricas"],
                      "criterios": salida["criterios_exito"]}, ensure_ascii=False, indent=2))
    return salida


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la línea de comandos."""
    p = argparse.ArgumentParser(description="Evaluación final única en las particiones de prueba")
    p.add_argument("--confirmo-uso-unico", action="store_true", required=True,
                   help="confirma que es la evaluación final y que la prueba no se usó antes")
    p.add_argument("--forzar", action="store_true", help="repetir aunque exista el registro (se anota)")
    a = p.parse_args(argv)
    evaluar(forzar=a.forzar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
