# Plan de evaluación (definido ANTES de ver resultados)

> Paso 9 del formato, sección 9. Criterio 8 de la rúbrica.
> Este archivo se confirmó en el repositorio en un commit anterior a cualquier experimento
> (ver `git log --follow docs/plan_evaluacion.md`). Cualquier cambio posterior se registra al final, en
> «Registro de cambios», con su motivo; los umbrales de éxito no se modifican después de ver resultados.

## 1. Conjuntos y uso de cada uno

| Conjunto | Origen | Uso | ¿Cuántas veces? |
|---|---|---|---|
| **Entrenamiento** | `train.xlsx` (v1, 676 noticias) | Validación cruzada para comparar y ajustar todos los modelos | Todas las que haga falta |
| **Prueba DD** (dentro de distribución) | `development.xlsx` (v1, 295 noticias), sin los textos casi duplicados de entrenamiento | Estimar el desempeño final del modelo elegido | **Una sola vez**, al final |
| **Prueba FD** (fuera de distribución) | `test.xlsx` (v2, 572 noticias, 2020–2021) sin casi duplicados de v1 | Estimar la generalización a otro periodo, tema y país | **Una sola vez**, al final |

El script `src/fakenews/evaluate.py` es el único que lee las particiones de prueba. Antes de leerlas exige la
opción `--confirmo-uso-unico` y deja un registro en `results/uso_prueba.log`. Si el registro ya existe, se
niega a repetir la evaluación salvo que se use `--forzar`, lo que queda anotado y debe declararse en el informe.

Para la prueba FD se reentrena la configuración elegida con entrenamiento + prueba DD (971 noticias), como
hicieron los participantes de FakeDeS 2021 [R02]; así la comparación con su tabla de resultados es justa.
Se reporta también el resultado FD del modelo entrenado solo con entrenamiento.

## 2. Métricas y por qué

El costo de los errores no es simétrico en el discurso, pero ambos importan en el uso:

- **Falso positivo** (verdadera marcada como sospechosa): acusa a un medio legítimo y le hace perder tiempo al
  verificador. Repetido, es una forma de censura.
- **Falso negativo** (falsa marcada como confiable): la pieza baja en la cola y puede circular sin revisión.

| Métrica | Papel | Lectura |
|---|---|---|
| **F1‑macro** | **Principal.** Selección de modelos y criterio de éxito. | Las clases están casi balanceadas (338/338 en entrenamiento); F1‑macro pesa igual ambas clases y es la métrica de ranking de FakeDeS 2021 [R02]. |
| Recall de «falsa» (sensibilidad) | Operativa | Proporción de falsas que el sistema pone arriba en la cola. |
| Especificidad | Operativa, con umbral | Proporción de verdaderas que no se acusan. 1 − especificidad = tasa de falsas acusaciones. |
| Precisión de «falsa» | Operativa | De lo que se marca como sospechoso, cuánto lo es realmente. |
| Exactitud | Solo comparación con literatura | El artículo original reporta exactitud [R01, R05]; con clases balanceadas es interpretable, pero no se usa para decidir. |
| ROC‑AUC y PR‑AUC | Diagnóstico | Calidad del ordenamiento, independiente del umbral (relevante porque el uso es priorizar). |
| Brier y curva de calibración | Diagnóstico | La app muestra una probabilidad; debe significar algo. |
| Matriz de confusión | Diagnóstico | Siempre junto a las cifras. |

## 3. Estrategia de validación

1. **Validación cruzada estratificada repetida**: 5 pliegues × 3 repeticiones (semillas 13, 42, 2026) sobre
   entrenamiento. Se reporta media ± desviación estándar de los 15 pliegues.
2. **Validación cruzada agrupada por fuente** (`StratifiedGroupKFold`, 5 pliegues × 3 semillas): todas las
   noticias de un mismo medio quedan en el mismo pliegue. La diferencia entre (1) y (2) estima cuánto del
   desempeño viene de reconocer el medio y no el contenido.
3. Toda transformación (TF‑IDF, vocabulario, idf) se ajusta **dentro** de cada pliegue mediante `Pipeline` de
   scikit‑learn [R13]. Las reglas del sistema experto no se ajustan con datos: sus pesos se fijan a mano en
   `configs/reglas.yaml` antes de ejecutar.
4. Los hiperparámetros se eligen con validación cruzada **anidada** (búsqueda interna de 3 pliegues dentro de
   cada pliegue externo) para que la estimación no sea optimista.

## 4. Líneas base

| ID | Línea base | Para qué |
|---|---|---|
| E‑00a | Clase mayoritaria (`DummyClassifier`) | Piso: cualquier modelo debe superarla con amplitud. |
| E‑00b | Sistema experto de reglas, sin aprendizaje (umbral de puntaje fijado a priori = 2) | Alternativa no‑ML. Si el modelo no la supera, el ML no se justifica. |
| E‑01 | TF‑IDF de palabras + Naive Bayes multinomial | Línea base de ML sencilla y estándar en clasificación de texto. |

## 5. Regla de selección del modelo final (fijada a priori)

1. Se elige la configuración con mayor F1‑macro medio en la **validación agrupada por fuente** (es la que mejor
   se parece al uso real, donde llegan medios no vistos).
2. Si dos configuraciones difieren en menos de una desviación estándar, se elige la más simple e interpretable
   (en este orden: regresión logística de palabras > regresión logística de caracteres > SVM).
3. El umbral de decisión es 0,5 sobre la probabilidad de «falsa». La zona de abstención («indeterminada») es
   [0,40; 0,60]. Ambos valores se fijan aquí y no se ajustan con las pruebas.

## 6. Criterios de éxito (fijados antes de los resultados)

| ID | Criterio | Umbral | Justificación del umbral |
|---|---|---|---|
| CE1 | F1‑macro en prueba DD | ≥ 0,77 | Igualar el orden de magnitud del artículo original del corpus (77,28 % de exactitud) [R01, R05]. |
| CE2 | Mejora sobre la línea base NB en prueba DD | Diferencia de F1‑macro > 0 con IC bootstrap 95 % que no incluya 0 | Que la complejidad añadida aporte valor demostrable. |
| CE3 | F1‑macro en prueba FD (v2) | ≥ 0,70 | Cerca de la línea base BoW+SVM de FakeDeS 2021 (0,7217) [R02]; se acepta un margen por ser un prototipo con un solo modelo lineal. |
| CE4 | Especificidad en prueba DD | ≥ 0,80 | No acusar por error a más de 1 de cada 5 noticias verdaderas. |
| CE5 | Mejora sobre el sistema de reglas (E‑00b) en CV agrupada | F1‑macro mayor por más de una desviación estándar | Justificar el ML frente a la alternativa no‑ML. |
| CE6 | Fidelidad de las explicaciones | 100 % de las reglas citadas se activan realmente en el texto y 100 % de los términos citados aparecen en el texto | Evitar «explicaciones decorativas» (Anexo A.2 de la rúbrica). |
| CE7 | Latencia de la app | < 1 s por noticia (mediana) en CPU | Uso interactivo. |

Si un criterio no se cumple, se reporta como no cumplido y se analiza; no se cambia el umbral.

## 7. Variabilidad, subgrupos e incertidumbre

- CV: media ± desviación de 15 pliegues para cada experimento.
- Pruebas: intervalos de confianza al 95 % por bootstrap (2.000 remuestreos, semilla 42) para F1‑macro,
  especificidad y recall de «falsa»; bootstrap pareado para la diferencia contra la línea base.
- Subgrupos en prueba DD: tema (9), tercil de longitud. Subgrupos en prueba FD: tema (7), presencia de titular
  (aproximación a «publicación de redes sociales»), dominio del enlace por país (.co, .mx, .es, .ar, otros).
  Subgrupos con menos de 20 casos se reportan, pero no se interpretan.

## 8. Análisis de errores planificado

1. Leer todos los falsos positivos y una muestra de falsos negativos de la prueba DD.
2. Agruparlos por tipo: sátira, noticia legítima con estilo coloquial, opinión, texto corto, publicación de
   redes, tema sin representación en entrenamiento.
3. Revisar qué términos y reglas pesaron en cada error.
4. Contrastar los errores FD con el cambio de tema (COVID‑19 no existe en entrenamiento).

## 9. Uso de la base de conocimiento (sistema experto) — evaluación planificada

- Para cada regla: soporte (porcentaje de noticias en que se activa) y precisión respecto a «falsa» en
  entrenamiento; luego, una sola vez, en pruebas DD y FD.
- Sistema de reglas como clasificador (E‑00b) comparado con el modelo.
- Modelo híbrido (TF‑IDF + reglas como rasgos, E‑09) para saber si las reglas aportan información nueva.
- Pruebas automáticas de fidelidad (CE6) y de entradas hostiles (vacías, otro idioma, HTML, texto con
  instrucciones incrustadas).

## Registro de cambios

| Fecha | Cambio | Motivo |
|---|---|---|
| 2026‑11‑23 | Versión 1.0 | Plan inicial, antes de los experimentos. |
