# Detector de noticias falsas en español · Proyecto 22

**Taller: 100 proyectos de Machine Learning** · Categoría 03, Lenguaje Natural · Estudiante: Marco Andrés Rico

Prototipo **educativo** que lee el titular y el cuerpo de una noticia en español y devuelve una **prioridad de
revisión** (sospechosa / indeterminada / probablemente confiable) con una probabilidad y una explicación: las
palabras que más pesaron en el modelo y las reglas de verificación (sistema experto basado en la guía de la
IFLA) que se activaron.

> ⚠️ **No determina si una noticia es verdadera.** Detecta patrones de estilo aprendidos de noticias mexicanas de
> 2018. Sirve para ordenar la cola de un verificador humano, nunca para censurar ni sancionar.

## Resultados principales (evaluación final única)

Modelo elegido con la regla fijada a priori: **E-06a** (TF-IDF de palabras 1-2 + regresión logística con C y min_df ajustados por CV anidada; hiperparámetros: {'clf__C': 30.0, 'tfidf__min_df': 3}).

- **Prueba DD (v1, 2018, 290 noticias):** F1-macro 0.786 (IC95 0.738–0.832), especificidad 0.793, recall de falsas 0.779, ROC-AUC 0.888. Línea base NB: F1-macro 0.758; sistema de reglas: 0.439.
- **Prueba FD (v2, 2020–21, COVID y otros países, 568 noticias):** F1-macro 0.689 (IC95 0.650–0.727), especificidad 0.888, recall de falsas 0.512, ROC-AUC 0.798. Línea base NB: 0.585; reglas: 0.503.
- **Hallazgo clave:** la validación agrupada por medio baja el F1 de ~0,84 a ~0,74, y en noticias nuevas (FD) el modelo solo detecta la mitad de las falsas (recall 0.51), aunque casi no acusa verdaderas (especificidad 0.89). Confirma la presentación: un buen número dentro del dataset no garantiza el mundo real.
- Latencia mediana: 20 ms por noticia en CPU.

| Criterio de éxito (fijado antes) | Resultado |
|---|---|
| CE1_f1_dd_>=_0.77 | ✅ cumple |
| CE2_mejora_vs_nb_ic_sin_0 | ❌ no cumple |
| CE3_f1_fd_>=_0.70 | ❌ no cumple |
| CE4_especificidad_dd_>=_0.80 | ❌ no cumple |
| CE5_supera_reglas_cv_agrupada | ✅ cumple |
| CE6_explicaciones_fieles_100 | ❌ no cumple |
| CE7_latencia_mediana_<_1000ms | ✅ cumple |

Los criterios no cumplidos se reportan tal cual; no se reajustó nada con la prueba. CE6: algunas explicaciones citan
n-gramas que, tras la normalización, no se encuentran literalmente en el texto por diferencias de puntuación del
comprobador (840/858 verificadas); queda como mejora pendiente.
Detalle completo: `results/metricas_finales.json`, `results/bitacora.csv`, `results/figuras/`.


## Estructura

```
├── README.md                  este archivo
├── Makefile                   flujo completo (make todo)
├── requirements.txt           dependencias (rangos) · requirements-lock.txt: versiones exactas usadas
├── pyproject.toml             paquete `fakenews`, configuración de pytest y ruff
├── configs/
│   ├── base.yaml              rutas, URL y SHA-256 de los datos, semillas, umbrales
│   ├── experimentos.yaml      definición de E-00a … E-09 (un cambio por experimento)
│   └── reglas.yaml            base de conocimiento del sistema experto (versionada)
├── src/fakenews/              código del prototipo
│   ├── data.py                descarga verificada, armonización v1/v2, deduplicación, particiones
│   ├── preprocessing.py       normalización de texto
│   ├── rules.py               sistema experto (motor de inferencia + explicación con evidencia)
│   ├── models.py              pipelines de scikit-learn
│   ├── experiments.py         bitácora con CV estratificada y agrupada por fuente
│   ├── evaluate.py            evaluación final ÚNICA en las pruebas
│   ├── train.py               reconstrucción del modelo de producción
│   ├── predict.py             detector: validación de entradas, predicción, explicación, CLI
│   ├── metrics.py · tracking.py · plots.py · config.py
├── app/app.py                 aplicación Streamlit
├── tests/                     pruebas automáticas (pytest)
├── notebooks/
│   ├── 00_ejecutar_en_colab.ipynb   ejecución guiada de todo el proyecto en Google Colab
│   ├── 01_exploracion_datos.ipynb   EDA con tabla de hallazgos
│   └── 02_resultados_y_demo.ipynb   bitácora, evaluación final y demostración guiada
├── results/                   bitácora, tablas, figuras, registro de corridas y métricas finales
├── models/                    modelo versionado (detector_v1.0.0.joblib) y MODEL_CARD.md
└── docs/                      planteamiento, ficha de datos, decisiones, riesgos, informe final, etc.
```

## Requisitos

- Python 3.10 o superior (resultados generados con 3.13). CPU es suficiente; no requiere GPU.
- Conexión a internet para descargar el corpus desde GitHub (3 archivos, 2 MB).
- No se necesita ninguna clave API ni credencial.

## Ejecución paso a paso (local)

```bash
git clone <URL-de-este-repositorio> detector-noticias-falsas
cd detector-noticias-falsas
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .

python -m fakenews.data preparar          # 1. descarga y verifica SHA-256, prepara particiones (~10 s)
python -m pytest                          # 2. pruebas automáticas (~1 min)
streamlit run app/app.py                  # 3. abre la aplicación con el modelo ya entrenado
```

El modelo versionado (`models/detector_v1.0.0.joblib`) ya está en el repositorio. Si su versión de
scikit-learn es distinta y aparece un aviso, reconstrúyalo **sin reevaluar** con `python -m fakenews.train`.

### Reproducir los experimentos desde cero

```bash
python -m fakenews.experiments            # bitácora completa: ~1,5 h en CPU (E-06b es la más lenta)
python -m fakenews.evaluate --confirmo-uso-unico --forzar   # ver nota
```

**Nota sobre la evaluación final.** La prueba se usa una sola vez: `evaluate` deja constancia en
`results/uso_prueba.log` y se niega a repetirse. Como ese registro ya está en el repositorio, un tercero que
quiera **reproducir** las cifras debe añadir `--forzar` (la repetición queda anotada). Con las mismas versiones
de `requirements-lock.txt` los resultados son idénticos porque todas las semillas están fijadas.

### Usar el detector desde la terminal

```bash
python -m fakenews.predict --titular "Título" --cuerpo "Texto completo de la noticia…"
python -m fakenews.predict --archivo noticias.csv --salida predicciones.csv   # columnas: titular, cuerpo
```

## Ejecución en Google Colab

Abra `notebooks/00_ejecutar_en_colab.ipynb` en Colab, cambie la URL del repositorio en la primera celda y
ejecute todas las celdas: instala dependencias, descarga los datos, corre las pruebas, muestra resultados y
hace predicciones de ejemplo.

## Documentación

| Documento | Contenido | Pasos del formato |
|---|---|---|
| [docs/informe_final.md](docs/informe_final.md) | **Informe final** | 15 |
| [docs/planteamiento.md](docs/planteamiento.md) | Resumen, glosario, pregunta, objetivos, alcance, tipo de tarea | 1, 3, 4 |
| [docs/antecedentes.md](docs/antecedentes.md) | Matriz de antecedentes y preguntas de investigación | 2 |
| [docs/ficha_datos.md](docs/ficha_datos.md) | Ficha técnica y tabla de hallazgos de calidad | 5, 6 |
| [docs/base_conocimiento.md](docs/base_conocimiento.md) | Ficha de la base de conocimiento (Anexo A) y evaluación de su uso | 5, 8, 14 |
| [docs/decisiones_tecnicas.md](docs/decisiones_tecnicas.md) | Comparación de métodos, preparación, licencias | 7, 8 |
| [docs/plan_evaluacion.md](docs/plan_evaluacion.md) | Métricas y criterios de éxito fijados **antes** de los resultados | 9 |
| [docs/bitacora_experimentos.md](docs/bitacora_experimentos.md) | Bitácora E-00a … E-09 con lectura de cada resultado | 10, 11 |
| [docs/analisis_errores.md](docs/analisis_errores.md) | Errores agrupados, subgrupos, amenazas a la validez | 12 |
| [docs/riesgos_etica.md](docs/riesgos_etica.md) | Matriz de riesgos y mitigaciones | 13 |
| [docs/uso_y_seguimiento.md](docs/uso_y_seguimiento.md) | Diseño de uso, límites y plan de monitoreo | 14 |
| [docs/plan_trabajo.md](docs/plan_trabajo.md) | Plan de trabajo y lista de verificación | 11, 16 del formato |
| [docs/diagrama_flujo.mmd](docs/diagrama_flujo.mmd) · [PNG](docs/img/diagrama_flujo.png) | Diagrama de flujo con decisiones y retornos | sección 4 |
| [docs/referencias.md](docs/referencias.md) | 19 referencias verificadas | 13, 14 |
| [docs/declaracion_ia.md](docs/declaracion_ia.md) | Declaración de uso de herramientas de IA | 18 |
| [docs/guia_sustentacion.md](docs/guia_sustentacion.md) | Preguntas probables y dónde está cada respuesta | — |
| [models/MODEL_CARD.md](models/MODEL_CARD.md) | Ficha del modelo | — |

## Datos y licencia

*The Spanish Fake News Corpus* (Posadas-Durán et al., 2019; Gómez-Adorno et al., 2021),
https://github.com/jpposadas/FakeNewsCorpusSpanish, licencia **CC-BY-4.0**. Los datos **no** se incluyen en este
repositorio: se descargan desde un commit fijado y se verifica su SHA-256. Por favor cite a los autores del
corpus si reutiliza este trabajo. El código de este repositorio se publica con fines académicos.

## Uso de IA

Este proyecto se desarrolló con apoyo de un asistente de IA. Ver [docs/declaracion_ia.md](docs/declaracion_ia.md).
