# Decisiones técnicas

> Pasos 4, 7 y 8 del formato, secciones 7 y 12. Criterios 5 y 6 de la rúbrica.
> Las secciones 1–4 se escribieron antes de los experimentos. La sección 5 se completa con los resultados.

## 1. Comparación de enfoques (sección 7 del formato)

| Aspecto | A. TF‑IDF + clasificador lineal (NB, regresión logística, SVM) | B. Sistema experto de reglas | C. *Transformer* en español (BETO) ajustado | 
|---|---|---|---|
| Cómo funciona | Convierte el texto en un vector de pesos de palabras o n‑gramas y aprende un peso por término | Suma puntos por señales de verificación escritas a mano (sin aprendizaje) y explica qué reglas se activaron | Modelo de lenguaje preentrenado que se ajusta con las 676 noticias |
| Datos y recursos | Cientos de ejemplos; CPU; segundos de entrenamiento | Ningún dato para construirlo; datos solo para evaluarlo | GPU recomendable; minutos u horas; 440 MB de pesos |
| Ventajas | Rápido, reproducible, interpretable (peso por término); robusto con pocos datos [R05]; generaliza mejor que modelos grandes fuera de distribución según [R06] | Explicación transparente; no depende del corpus; incorpora conocimiento experto [R09] | Mejor comprensión del contexto; resultado competitivo en FakeDeS (0,7321) [R02] |
| Limitaciones y riesgos | No entiende significado; aprende atajos (medio, formato) | Reglas rígidas; quien desinforma las evade; cobertura limitada | Sobreajuste con 676 ejemplos; caja negra; costo; en FakeDeS solo superó a BoW+SVM por 0,01 [R02] |
| Métricas pertinentes | F1‑macro, especificidad, ROC‑AUC, calibración | Soporte y precisión por regla; F1‑macro como clasificador | Mismas que A |
| Fuente técnica | R13 (scikit‑learn), R05, R06 | R09, R10 | R02 |
| Decisión | **Se usa como modelo principal** (pide la ficha del taller y la evidencia lo respalda) | **Se usa como línea base no‑ML y como capa de explicación** | **No se implementa.** La mejora esperada (≈ +0,01 de F1 en FakeDeS) no compensa el costo, la opacidad y el riesgo de sobreajuste en un prototipo educativo. Se deja como trabajo futuro. |

## 2. Línea base

Tres niveles, ejecutados antes de cualquier modelo complejo (E‑00a, E‑00b, E‑01): clase mayoritaria, sistema
de reglas y TF‑IDF + Naive Bayes. Ver `plan_evaluacion.md`.

## 3. Preparación de datos (paso 7)

| Paso | Decisión | Justificación |
|---|---|---|
| Armonizar columnas | v2 renombrada a los nombres de v1; etiqueta booleana → «Fake»/«True» | Mismo esquema para todo el flujo |
| Normalizar tokens | Cifras y `*NUMBER*` → `NUM`; URLs y `*URL*` → `URL`; correos → `EMAIL`; `*PHONE*` → `NUM` | Hallazgo H3: v1 y v2 se normalizaron distinto |
| Titular duplicado | Quitar el titular del inicio del cuerpo si lo repite | Hallazgo H4 |
| Texto de entrada | `titular + salto de línea + cuerpo` (o solo titular en E‑07) | El usuario puede pegar ambos |
| Limpieza HTML | Quitar etiquetas `<...>` y espacios repetidos | Entradas de la app pueden venir de páginas web |
| Minúsculas | Sí por defecto (dentro del vectorizador); E‑08 mide el efecto de conservarlas | Hallazgo H5 |
| Palabras vacías | No se eliminan | En detección de estilo, las palabras funcionales son informativas; TF‑IDF ya reduce su peso |
| Faltantes | Titular vacío → cadena vacía | No hay faltantes en el cuerpo |
| Duplicados | Se eliminan de las pruebas los textos con similitud coseno ≥ 0,9 con cualquier texto de entrenamiento | Hallazgos H9 y H10; prevención de fuga [R07] |
| Ajuste de transformaciones | TF‑IDF (vocabulario e idf) se ajusta dentro de un `Pipeline` en cada pliegue | Evita que el vocabulario de validación o prueba se filtre al entrenamiento [R13] |
| Particiones | Partición oficial del corpus + CV estratificada + CV agrupada por fuente (`StratifiedGroupKFold`) | Hallazgos H7 y H8: medir memorización del medio [R11] |
| Columnas excluidas de la entrada | `Source`, `Link`, `Topic`, `Id` | Revelan la etiqueta o no están disponibles en el uso |

## 4. Bibliotecas, modelos y licencias

| Componente | Versión usada | Licencia | Uso |
|---|---|---|---|
| Python | 3.13 (probado también en 3.11) | PSF | Lenguaje |
| scikit‑learn | 1.9 | BSD‑3‑Clause | TF‑IDF, modelos, validación |
| pandas / NumPy / SciPy | 3.0 / 2.5 / 1.18 | BSD‑3‑Clause | Datos |
| openpyxl | 3.1 | MIT | Lectura de .xlsx |
| matplotlib | 3.11 | Licencia PSF‑compatible (matplotlib) | Gráficas |
| PyYAML | 6.0 | MIT | Configuración |
| Streamlit | 1.67 | Apache‑2.0 | Aplicación |
| pytest / ruff | 9.1 / 0.16 | MIT | Pruebas y estilo |
| Spanish Fake News Corpus | v1 y v2, commit 732f487 | CC‑BY‑4.0 | Datos |
| Infografía IFLA | 2017 / 2020 | CC‑BY‑4.0 | Conocimiento para las reglas |

No se usa ningún modelo preentrenado ni API externa: no hay costos por consulta ni claves que proteger.

## 5. Resultados que sustentan la elección final

_(Se completa después de ejecutar la bitácora; ver `bitacora_experimentos.md`.)_
