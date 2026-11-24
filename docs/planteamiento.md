# Planteamiento del problema — Proyecto 22: Detector de noticias falsas

> Pasos 1, 3 y 4 del formato estándar. Criterio 1 de la rúbrica.

## 1. Identificación

| Campo | Información |
|---|---|
| Número y nombre del ejercicio | Ejercicio 22 · Detector de noticias falsas |
| Categoría / nivel | Categoría 03 · Lenguaje Natural (NLP) · Nivel principiante–intermedio |
| Estudiante(s) | Marco Andrés Rico |
| Fecha / versión | 23 de noviembre de 2026 · versión del documento 1.0 |
| Modalidad | Individual |
| Repositorio y cuaderno | _[enlace al repositorio en GitHub]_ · cuaderno guiado: `notebooks/00_ejecutar_en_colab.ipynb` |

## 2. Resumen del problema (paso 1 — 153 palabras)

Las noticias falsas circulan más rápido de lo que un equipo de verificación puede revisarlas. Un verificador
de un medio o de una organización como ColombiaCheck recibe cada día más piezas de las que puede leer, y
necesita decidir cuáles mirar primero. Este proyecto construye un prototipo que lee el titular y el cuerpo
de una noticia en español y devuelve una señal de prioridad: «sospechosa», «confiable» o «indeterminada»,
con una probabilidad y una explicación (palabras que más pesaron y reglas de verificación que se activaron).
El sistema no decide si algo es verdad: aprende patrones de estilo a partir de noticias ya etiquetadas por
personas, y por eso se evalúa no solo en noticias parecidas a las de entrenamiento (México, 2018), sino
también en noticias de otro periodo, tema y país (COVID-19, 2020–2021, incluida Colombia). La pregunta
central es cuánto de lo aprendido se sostiene cuando el contexto cambia.

## 3. Glosario inicial

| Término | Definición breve (elaboración propia a partir de las fuentes de `referencias.md`) |
|---|---|
| Noticia falsa | Pieza con forma de noticia cuyo contenido es falso según un verificador o sin respaldo en medios confiables. En este corpus incluye sátira presentada como noticia. |
| Verificación de hechos (*fact-checking*) | Contrastar una afirmación concreta con evidencia externa. Es una tarea distinta de clasificar el estilo de un texto [R08]. |
| Clasificación de estilo | Predecir la etiqueta a partir de rasgos del propio texto (palabras, puntuación, longitud) sin consultar evidencia externa. Es lo que hace este prototipo. |
| TF‑IDF | Representación que pondera cada palabra (o n‑grama) por su frecuencia en el documento y su rareza en el corpus. |
| n‑grama | Secuencia de *n* palabras (n‑gramas de palabras) o de *n* caracteres (n‑gramas de caracteres). |
| Regresión logística | Clasificador lineal que estima la probabilidad de la clase «falsa» y permite leer el peso de cada término. |
| Naive Bayes multinomial | Clasificador probabilístico simple para conteos de palabras; se usa como línea base. |
| Sistema experto | Programa que aplica reglas «SI … ENTONCES …» escritas a partir del conocimiento de un experto y explica qué reglas usó. |
| Fuga de datos | Información de evaluación que se filtra al entrenamiento o a la selección del modelo y produce resultados optimistas [R07]. |
| Validación cruzada agrupada | Validación en la que todos los textos de una misma fuente quedan del mismo lado de la partición. Mide si el modelo memoriza la fuente. |
| Cambio de distribución (*distribution shift*) | Diferencia entre los datos de entrenamiento y los del uso real (otro año, tema o país). |
| F1‑macro | Promedio simple del F1 de cada clase; trata igual a «falsa» y «verdadera». |
| Especificidad | Proporción de noticias verdaderas que el sistema deja pasar como confiables. Su complemento es la tasa de falsas acusaciones. |
| Abstención («no sé») | Respuesta «indeterminada» cuando la probabilidad cae en una zona de duda; se envía a revisión humana. |

## 4. Interpretación del enunciado

| Lo que el ejercicio exige (ficha 22 del taller) | Lo que es opcional y se añade con justificación |
|---|---|
| Usar un dataset de noticias etiquetadas | Usar un corpus **en español** con una segunda versión de otro periodo para medir generalización |
| Preprocesar con TF‑IDF | Comparar n‑gramas de palabras y de caracteres |
| Entrenar un clasificador lineal | Comparar Naive Bayes, regresión logística y SVM lineal |
| Analizar errores | Análisis por tema, longitud, país y fuente; sistema experto de reglas como explicación; aplicación de demostración |

Pregunta de investigación del taller: *¿Por qué verificar hechos es distinto de clasificar estilo?* Se responde
en el informe final con la evidencia del experimento fuera de distribución y del análisis de errores.

## 5. Pregunta de proyecto (paso 3)

> ¿Un clasificador lineal sobre TF‑IDF, entrenado con el Spanish Fake News Corpus v1 (México, 2018), alcanza
> un F1‑macro de al menos 0,77 en la partición de prueba del mismo corpus y de al menos 0,70 en el corpus v2
> (2020–2021, COVID‑19 y otros países), manteniendo una especificidad de al menos 0,80 (como máximo 20 % de
> noticias verdaderas acusadas por error)?

La pregunta es verificable: cada umbral está fijado en `docs/plan_evaluacion.md` antes de ver resultados y
se compara con valores publicados [R01, R02].

## 6. Objetivos

**Objetivo general.** Construir y evaluar de forma reproducible un prototipo que priorice noticias en español
para revisión humana a partir de su texto, cuantificando cuánto se degrada cuando cambian el periodo, el tema
y el país de las noticias.

**Objetivos específicos (medibles).**

| # | Objetivo | Indicador y meta |
|---|---|---|
| OE1 | Documentar y auditar el corpus (duplicados, fuentes, formato, sesgos). | Ficha de datos completa y tabla de hallazgos con cada problema cuantificado. |
| OE2 | Comparar al menos tres representaciones/clasificadores contra dos líneas base (clase mayoritaria y sistema de reglas). | ≥ 5 experimentos controlados en la bitácora, con media ± desviación en validación cruzada. |
| OE3 | Medir el desempeño en prueba dentro de distribución y fuera de distribución. | F1‑macro ≥ 0,77 (v1) y ≥ 0,70 (v2), con intervalos de confianza al 95 %. |
| OE4 | Controlar las falsas acusaciones. | Especificidad ≥ 0,80 en la prueba v1. |
| OE5 | Entregar una aplicación que explique cada predicción y se abstenga cuando dude. | App funcional con validación de entradas, explicación por palabras y reglas, latencia medida (< 1 s por noticia). |

## 7. Usuarios, contexto, entradas y salidas

| Elemento | Definición |
|---|---|
| Usuario principal | Verificador o editor de un medio o de una organización de verificación que necesita ordenar su cola de trabajo. |
| Usuario secundario | Docente o estudiante que estudia los límites de la detección automática. |
| Contexto de uso | Apoyo a la priorización. Una persona siempre toma la decisión final. |
| Entrada | Titular (opcional) y cuerpo de la noticia en español; entre 30 y 5.000 palabras. |
| Salida | Etiqueta de prioridad (sospechosa / indeterminada / confiable), probabilidad de «falsa», 8 términos que más empujaron la decisión, reglas del sistema experto activadas y advertencia de uso. |
| Escenario | El verificador pega la noticia en la app o procesa un lote de textos con la línea de comandos y revisa primero las marcadas como sospechosas. |

## 8. Qué NO resuelve el prototipo

- **No determina si una noticia es verdadera.** No consulta fuentes externas ni contrasta hechos.
- No analiza imágenes, videos, audios ni la propagación en redes sociales.
- No está pensado para mensajes muy cortos (trinos, cadenas de WhatsApp de una línea) ni para textos en otro idioma.
- No distingue sátira declarada de desinformación: en el corpus ambas se etiquetan como «falsa».
- No debe usarse para censurar, bloquear o sancionar contenido ni personas.

## 9. Tipo de tarea de ML (paso 4)

**Clasificación binaria supervisada de texto (NLP).**

Evidencia: (a) existe una variable objetivo discreta con dos valores (*True* / *Fake*) asignada por anotadores
humanos con una guía publicada [R01]; (b) la entrada es texto libre; (c) el uso pedido es ordenar y marcar
piezas, lo que corresponde a estimar una probabilidad de pertenencia a la clase «falsa». No es regresión (no
hay grado continuo de falsedad en este corpus), ni agrupamiento (las etiquetas existen y se quieren usar), ni
verificación de hechos (no hay base de evidencia para contrastar afirmaciones) [R08].

**Contraste con una alternativa no‑ML.** Un sistema experto de reglas, construido a partir de guías de
verificación como la infografía de la IFLA [R09] (¿cita fuentes?, ¿el titular es alarmista?, ¿pide que se
comparta?), resolvería la tarea sin entrenamiento y explicaría cada decisión. Se implementa y se evalúa como
**línea base no‑ML** (experimento E‑00b) y luego se usa como capa de explicación sobre el modelo. Si las reglas
solas igualaran al modelo, el modelo no estaría justificado; los resultados de la bitácora responden esa
pregunta.

## 10. Justificación

El español es la segunda lengua materna del mundo y la mayoría de los recursos de detección están en inglés
[R03, R04]. Un antecedente colombiano (Universidad de Antioquia) señala que la falta de datos en español es el
principal cuello de botella [R05]. Al mismo tiempo, la literatura muestra que los detectores que obtienen
cifras casi perfectas dentro de su dataset caen cerca del azar al cambiar de dataset [R06]. Por eso el valor de
este proyecto no está en una cifra alta, sino en medir con honestidad dónde funciona y dónde no.
