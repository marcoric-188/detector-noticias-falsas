# Plan de trabajo, control de avance y lista de verificación

> Secciones 11 y 16 del formato. Las evidencias se pueden rastrear en el historial de commits (`git log`).

## 1. Plan de trabajo

| Fase | Actividad | Tiempo estimado | Evidencia / entregable | Estado |
|---|---|---|---|---|
| 1 | Comprensión e investigación | 3 h | Presentación inicial; `docs/antecedentes.md`, `docs/referencias.md` | Listo |
| 2 | Definición del problema y alcance | 1 h | `docs/planteamiento.md` | Listo |
| 3 | Datos y exploración | 2 h | `docs/ficha_datos.md`, `notebooks/01_exploracion_datos.ipynb`, `results/tablas/eda_hallazgos.csv` | Listo |
| 4 | Diseño del método y evaluación (antes de resultados) | 2 h | `docs/plan_evaluacion.md`, `docs/decisiones_tecnicas.md`, `configs/*.yaml` (commit `4453fe1`) | Listo |
| 5 | Prototipo y experimentos | 4 h + 1,5 h de cómputo | `src/`, `results/bitacora.csv`, `docs/bitacora_experimentos.md`, `results/corridas.jsonl` | Listo |
| 6 | Evaluación final, análisis de errores y riesgos | 3 h | `results/metricas_finales.json`, `docs/analisis_errores.md`, `docs/riesgos_etica.md` | Listo |
| 7 | Aplicación, documentación y socialización | 3 h | `app/app.py`, `docs/informe_final.md`, `notebooks/02_resultados_y_demo.ipynb`, sustentación | Listo (sustentación pendiente) |

Dependencias: 4 depende de 3 (los hallazgos definen la preparación); 5 no empieza hasta que 4 está confirmado en
git; 6 usa la prueba una sola vez después de cerrar 5.

## 2. Lista de verificación antes de entregar (sección 16 del formato)

- [x] Entendí el problema y puedo explicarlo sin leer el enunciado (`planteamiento.md` §2).
- [x] Definí a quién sirve la solución y qué queda fuera de alcance (`planteamiento.md` §7–8).
- [x] Investigué fuentes confiables y registré las referencias (19 en `referencias.md`).
- [x] Verifiqué la procedencia, licencia y limitaciones de los datos (`ficha_datos.md`, SHA‑256).
- [x] Justifiqué el tipo de tarea y el método propuesto (`planteamiento.md` §9, `decisiones_tecnicas.md`).
- [x] Planifiqué la preparación y evité la fuga de información (deduplicación, `Pipeline`, CV agrupada, prueba bloqueada, `tests/test_particiones.py`).
- [x] Seleccioné métricas que corresponden al problema (`plan_evaluacion.md` §2).
- [x] Definí una línea base y un criterio de éxito antes de ver resultados (commit `4453fe1`).
- [x] El diagrama de flujo incluye decisiones y retornos (`docs/img/diagrama_flujo.png`).
- [x] Documenté riesgos, sesgos, privacidad y supervisión humana (`riesgos_etica.md`).
- [x] Mis afirmaciones se apoyan en evidencia y no exageran el alcance (`informe_final.md` §Conclusiones).
- [x] Reconocí las herramientas de IA utilizadas y verifiqué sus resultados (`declaracion_ia.md`).
- [x] El trabajo se puede comprender y reproducir con las instrucciones entregadas (README, `00_ejecutar_en_colab.ipynb`).
- [ ] **Pendiente del estudiante:** subir el repositorio a GitHub, poner la URL en `planteamiento.md` y en el cuaderno de Colab, ejecutar el cuaderno en Colab y preparar la sustentación (`guia_sustentacion.md`).

## 3. Lista técnica de la rúbrica (Anexo B)

| Elemento | Dónde | Criterio |
|---|---|---|
| README con objetivo, alcance, requisitos, instalación y ejecución | `README.md` | 10 |
| Estructura docs/, notebooks/, src/, tests/, results/, configs/, requirements.txt | raíz | 10 |
| Dependencias con versiones fijadas; semillas y configuración separadas | `requirements-lock.txt`, `configs/base.yaml` | 10 |
| Pruebas automáticas que pasan | `tests/` (`python -m pytest`) | 10 |
| Sin credenciales ni datos personales; `.gitignore`; commits con sentido | `.gitignore`, `git log` | 10, 12 |
| Ficha de datos y evidencia de acceso; licencia verificada | `docs/ficha_datos.md`, `fakenews.data descargar` | 3 |
| Notebook de exploración con hallazgos | `notebooks/01_exploracion_datos.ipynb` | 3 |
| Ficha de la base de conocimiento y script que la reconstruye | `docs/base_conocimiento.md`, `fakenews.data preparar`, `configs/reglas.yaml` | 4 |
| Evaluación del uso de la base | `results/tablas/evaluacion_reglas.csv`, E‑00b, E‑09 | 4 |
| Particiones antes del modelado; transformaciones solo con entrenamiento; verificación de no‑fuga | `data.py`, `models.py`, `experiments.validar`, `tests/test_particiones.py` | 5 |
| Dos o más enfoques y línea base primero | E‑00a, E‑00b, E‑01 → E‑09 | 6 |
| Bitácora con ≥ 5 experimentos y resultados desfavorables | `docs/bitacora_experimentos.md` (11 experimentos) | 7 |
| Curvas de aprendizaje; diagnóstico; tiempo | `results/figuras/curva_aprendizaje.png`, columna `brecha_sobreajuste`, `segundos_total` | 7 |
| Modelo final exportado con versión | `models/detector_v1.0.0.joblib`, `models/MODEL_CARD.md` | 7 |
| Plan de evaluación previo; prueba usada una vez | `plan_evaluacion.md`, `results/uso_prueba.log` | 8 |
| Métricas por subgrupo, variabilidad y análisis de errores | `results/tablas/subgrupos_*.csv`, IC bootstrap, `analisis_errores.md` | 8, 9 |
| App que carga modelo y base versionados | `app/app.py` | 11 |
| Entradas validadas, errores claros, límites informados | `predict.Detector.validar`, `tests/test_predict.py` | 11 |
| Latencia, costo y plan de seguimiento | `metricas_finales.json` (`app`), `uso_y_seguimiento.md` | 11 |
| Matriz de riesgos y declaración de IA | `riesgos_etica.md`, `declaracion_ia.md` | 12 |
| Informe, presentación, gráficas, referencias | `informe_final.md`, presentación del estudiante, `results/figuras/` | 13 |
