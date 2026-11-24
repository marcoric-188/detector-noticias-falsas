# Ética, privacidad, seguridad y uso responsable

> Paso 13 del formato, secciones 10 y 18. Criterio 12 de la rúbrica. Organizado con las funciones del NIST
> AI RMF (gobernar, mapear, medir, gestionar) [R18]. Las cifras citadas provienen de `results/` y se detallan en
> `analisis_errores.md`.

## 1. Contexto de impacto

Clasificar una noticia como «sospechosa» puede afectar la reputación de un medio o de un periodista y la
libertad de expresión. Por eso el prototipo se limita a **priorizar** la revisión humana, con fines
educativos; nunca decide publicar, bloquear, etiquetar públicamente ni sancionar.

## 2. Matriz de riesgos y mitigación (sección 10 del formato)

| Riesgo | ¿Cómo podría presentarse? | Prob. / impacto | Medida de mitigación | Evidencia |
|---|---|---|---|---|
| **Datos sesgados o poco representativos** | El entrenamiento es casi solo México 2018 y sus falsas son sobre todo sátira; el modelo aprende «estilo de sitio satírico» | Alta / alto | Prueba FD en otro periodo, tema y país; CV agrupada por fuente; advertencia visible en la app; documentación del sesgo | `ficha_datos.md` H7–H8; bitácora (columna agrupada); `analisis_errores.md` |
| **Privacidad o uso sin permiso** | Corpus con nombres de figuras públicas; usuarios que pegan textos privados en la app | Baja / medio | Corpus CC‑BY‑4.0 citado y no redistribuido; la app **no guarda el texto** (solo hash, probabilidad, etiqueta y reglas) y el registro se puede desactivar; sin datos de usuarios de redes | `ficha_datos.md`; `app/app.py::registrar` |
| **Fuga de información** | Duplicados entre particiones; vocabulario ajustado con prueba; selección del modelo mirando la prueba | Media / alto | Deduplicación; TF‑IDF en `Pipeline`; CV agrupada; plan y regla de selección confirmados en git antes; prueba bloqueada con registro | `tests/test_particiones.py`; `results/uso_prueba.log`; commit `4453fe1` |
| **Error desigual entre grupos** | Peor desempeño con publicaciones de redes, noticias sin titular, temas nuevos (COVID‑19) o países distintos de México; más falsas acusaciones a medios con estilo coloquial | Alta / alto | Métricas por subgrupo publicadas; abstención en la zona de duda; prohibición de uso automático | `results/tablas/subgrupos_*.csv`; `analisis_errores.md` |
| **Salida engañosa** | El usuario interpreta «probablemente confiable» como «verdadera», o la probabilidad como certeza | Alta / alto | Etiquetas de prioridad (no de verdad); aviso fijo en cada salida; zona «indeterminada»; calibración medida; explicación con términos y reglas cuya fidelidad se prueba (CE6) | `predict.ADVERTENCIA_USO`; `results/figuras/calibracion.png` |
| **Uso indebido / decisión de alto impacto** | Usarlo para censurar, desmonetizar o acusar públicamente a un medio | Media / muy alto | Límites de uso en README, app y ficha del modelo; `requiere_revision_humana = True` siempre; no hay modo de acción automática | `models/MODEL_CARD.md`; `uso_y_seguimiento.md` |
| **Evasión adversaria** | Quien desinforma copia el estilo de un medio serio y el modelo lo deja pasar | Alta / alto | Se declara como limitación; reglas explicables como segunda señal; plan de reentrenamiento y monitoreo de deriva | `uso_y_seguimiento.md` §4 |
| **Dependencia, costo o seguridad de herramienta** | Cambio de versión de scikit‑learn rompe el modelo serializado; archivos manipulados | Media / medio | Versiones fijadas; aviso de versión al cargar; `fakenews.train` reconstruye; descarga con SHA‑256; sin claves API ni servicios de pago | `requirements-lock.txt`; `predict.Detector.__init__` |
| **Entradas hostiles** | HTML, scripts o instrucciones incrustadas en el texto | Media / bajo | Se eliminan etiquetas HTML; no hay modelo generativo que pueda obedecer instrucciones; validación de longitud e idioma | `tests/test_predict.py` (HTML, instrucciones incrustadas, vacíos, otro idioma) |

## 3. Minimización de datos y retención

- Entrada mínima: titular y cuerpo. No se piden autor, URL, usuario ni metadatos.
- El registro de monitoreo guarda: fecha, hash SHA‑256 truncado del texto, número de palabras, probabilidad,
  etiqueta, reglas activadas, versión del modelo y latencia. **No guarda el texto.** Retención sugerida: 90 días.
- Los datos del corpus permanecen en `data/` (ignorado por git).

## 4. Supervisión humana y corrección

- Toda salida lleva `requiere_revision_humana = True` y el aviso de uso.
- La zona «indeterminada» (probabilidad entre 0,40 y 0,60) se envía explícitamente a criterio humano.
- Apelación: un medio que considere que su contenido se marca injustamente puede pedir la revisión; el
  verificador registra el caso como «falso positivo confirmado», y esos casos alimentan el análisis de errores y
  el reentrenamiento (ver `uso_y_seguimiento.md`).

## 5. Accesibilidad y explicabilidad

- La app no comunica el estado solo con color: cada estado tiene icono y texto («Prioridad ALTA…»).
- Cada predicción muestra los términos que la empujaron y las reglas activadas con su evidencia textual y su
  fundamento en la guía de la IFLA.

## 6. Alto impacto y fines educativos

El dominio (medios, libertad de expresión) es sensible. Este prototipo se entrega **solo con fines educativos**.
Un uso real exigiría: datos colombianos actuales y anotados por verificadores, auditoría externa de sesgos,
evaluación con usuarios y una política de apelación pública.
