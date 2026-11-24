"""Aplicación de demostración (Streamlit) del detector de noticias falsas.

Ejecutar desde la raíz del repositorio::

    streamlit run app/app.py
"""

from __future__ import annotations

import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from fakenews.predict import ADVERTENCIA_USO, Detector, EntradaInvalida  # noqa: E402

EJEMPLOS = {
    "Ejemplo ficticio 1 (estilo alarmista)": (
        "¡¡URGENTE!! LO QUE NO QUIEREN QUE SEPAS SOBRE EL AGUA DEL GRIFO",
        "Científicos descubrieron que el agua del grifo contiene una sustancia que nos ocultan desde hace "
        "años. Un estudio revela que tomar tres vasos de agua con limón cura definitivamente todas las "
        "enfermedades y no tiene efectos secundarios. Los gobiernos no quieren que esto se sepa porque "
        "perderían millones. Comparte este mensaje con todos tus contactos antes de que lo borren, es muy "
        "importante que todos lo sepan y que nadie se quede sin conocer este remedio milagroso!!!",
    ),
    "Ejemplo ficticio 2 (estilo informativo)": (
        "Alcaldía anuncia cierre parcial de una vía principal por obras durante tres semanas",
        "La Secretaría de Movilidad informó el lunes que la vía principal del sector norte tendrá un cierre "
        "parcial a partir del 2 de marzo por obras de mantenimiento. Según el comunicado, el cierre durará "
        "tres semanas y habrá desvíos señalizados por las calles aledañas. \"Pedimos a los conductores "
        "planear sus viajes con anticipación\", dijo la secretaria de Movilidad en rueda de prensa. La "
        "entidad explicó que los trabajos se realizarán en horario nocturno para reducir el impacto en el "
        "tráfico y que el transporte público mantendrá sus rutas habituales con ajustes menores.",
    ),
}
ESTADOS = {
    "sospechosa": ("⚠️", "Prioridad ALTA de revisión: sospechosa", "error"),
    "indeterminada": ("❔", "Indeterminada: el modelo duda, revisar con criterio humano", "warning"),
    "probablemente confiable": ("✅", "Prioridad BAJA: sin señales fuertes de sospecha", "success"),
}
ARCHIVO_MONITOREO = RAIZ / "data" / "monitoreo" / "predicciones.csv"


@st.cache_resource
def cargar_detector() -> Detector:
    """Carga el modelo una sola vez por sesión del servidor."""
    return Detector()


def registrar(resultado: dict, texto: str) -> None:
    """Registro de monitoreo SIN guardar el texto: hash, probabilidad, etiqueta, reglas y versión."""
    ARCHIVO_MONITOREO.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not ARCHIVO_MONITOREO.exists()
    with open(ARCHIVO_MONITOREO, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha_utc", "hash_texto", "n_palabras", "prob_falsa", "etiqueta", "reglas",
                        "version_modelo", "latencia_ms"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    hashlib.sha256(texto.encode("utf-8")).hexdigest()[:16], len(texto.split()),
                    resultado["prob_falsa"], resultado["etiqueta"],
                    "|".join(r["id"] for r in resultado["reglas"]["activadas"]),
                    resultado["version_modelo"], resultado["latencia_ms"]])


st.set_page_config(page_title="Detector de noticias falsas · Proyecto 22", page_icon="📰", layout="wide")
st.title("📰 Detector de noticias falsas en español")
st.caption("Proyecto 22 · Prototipo educativo para priorizar la verificación humana")
st.info(ADVERTENCIA_USO, icon="ℹ️")

try:
    detector = cargar_detector()
except FileNotFoundError as e:
    st.error(f"No se pudo cargar el modelo. {e}")
    st.stop()

with st.sidebar:
    md = detector.metadatos
    st.header("Ficha rápida")
    st.markdown(
        f"- **Modelo:** v{md.get('version')} ({md.get('seleccion', '')})\n"
        f"- **Reglas:** v{detector.reglas.version}\n"
        f"- **Entrenado con:** {md.get('datos_entrenamiento')}\n"
        f"- **scikit-learn:** {md.get('sklearn')}"
    )
    m = md.get("metricas_prueba_fd") or {}
    if m:
        st.markdown(
            "**Desempeño en noticias nuevas (v2, 2020–21):**\n"
            f"- F1-macro: {m['f1_macro']:.2f}\n- Recall de falsas: {m['recall_falsa']:.2f}\n"
            f"- Especificidad: {m['especificidad']:.2f}"
        )
    st.markdown(
        f"**Límites de uso**\n- Entre {detector.lim['min_palabras']} y {detector.lim['max_palabras']} "
        "palabras.\n- Solo español.\n- No usar para censurar ni sancionar.\n"
        f"- Zona «indeterminada»: probabilidad entre {detector.abst_inf:.2f} y {detector.abst_sup:.2f}."
    )
    for aviso in detector.avisos_carga:
        st.warning(aviso)
    monitorear = st.checkbox("Registrar la predicción para monitoreo (no se guarda el texto)", value=True)

ejemplo = st.selectbox("Cargar un ejemplo (textos inventados para la demostración)",
                       ["(ninguno)", *EJEMPLOS])
tit_def, cue_def = EJEMPLOS.get(ejemplo, ("", ""))
titular = st.text_input("Titular (opcional)", value=tit_def, max_chars=400)
cuerpo = st.text_area("Cuerpo de la noticia", value=cue_def, height=220)

if st.button("Analizar", type="primary"):
    try:
        r = detector.predecir(titular, cuerpo)
    except EntradaInvalida as e:
        st.error(f"Entrada no válida: {e}")
        st.stop()
    except Exception as e:  # noqa: BLE001 - mensaje claro ante cualquier fallo inesperado
        st.error(f"Ocurrió un error inesperado al analizar el texto ({type(e).__name__}). "
                 "Revise el texto e inténtelo de nuevo.")
        st.stop()

    icono, texto_estado, tipo = ESTADOS[r["etiqueta"]]
    getattr(st, tipo)(f"{icono} **{texto_estado}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad de «falsa» según el modelo", f"{r['prob_falsa']:.0%}")
    c2.metric("Puntaje del sistema de reglas", f"{r['reglas']['puntaje']:g} (umbral {r['reglas']['umbral']:g})")
    c3.metric("Tiempo de respuesta", f"{r['latencia_ms']:.0f} ms")
    st.progress(min(max(r["prob_falsa"], 0.0), 1.0))

    izq, der = st.columns(2)
    with izq:
        st.subheader("¿Qué palabras pesaron? (modelo)")
        st.caption("Contribución = peso TF-IDF del término en este texto × peso aprendido por el modelo.")
        filas = [{"término": t["termino"], "empuja hacia": "falsa", "contribución": t["contribucion"]}
                 for t in r["terminos"]["hacia_falsa"]]
        filas += [{"término": t["termino"], "empuja hacia": "verdadera", "contribución": t["contribucion"]}
                  for t in r["terminos"]["hacia_verdadera"]]
        st.dataframe(filas, hide_index=True)
    with der:
        st.subheader("¿Qué reglas de verificación se activaron? (sistema experto)")
        st.write(r["reglas"]["explicacion"])
        for a in r["reglas"]["activadas"]:
            signo = "➕" if a["peso"] > 0 else "➖"
            st.markdown(f"{signo} **{a['id']} · {a['descripcion']}** (peso {a['peso']:g})  \n"
                        f"Evidencia: {a['evidencia']}  \n_Fundamento: {a['fundamento']}_")
    for adv in r["advertencias"]:
        st.warning(adv)
    st.caption(f"Modelo v{r['version_modelo']} · reglas v{r['version_reglas']} · "
               "La decisión final corresponde a una persona.")
    if monitorear:
        registrar(r, f"{titular}\n{cuerpo}")
