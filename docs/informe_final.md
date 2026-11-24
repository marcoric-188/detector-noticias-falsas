# Informe final — Proyecto 22: Detector de noticias falsas

Estudiante: Marco Andrés Rico · 23-11-2026

Antecedentes y planteamiento: `planteamiento.md`, `antecedentes.md`. Datos: `ficha_datos.md`. Decisiones: `decisiones_tecnicas.md`. Plan a priori: `plan_evaluacion.md`. Riesgos: `riesgos_etica.md`. Uso y seguimiento: `uso_y_seguimiento.md`.

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

## Conclusión

El prototipo sirve para **priorizar**, no para decidir: dentro del corpus de 2018 llega al nivel del artículo original, pero con noticias de otro periodo, tema y país pierde la mitad de las falsas. Lo que el modelo aprende es en buena parte el estilo de los medios (sobre todo sátira mexicana), no la falsedad. Verificar hechos requiere evidencia externa; clasificar estilo no.
