# Diseño de uso, despliegue y seguimiento

> Paso 14 del formato. Criterio 11 de la rúbrica. Las cifras de latencia provienen de la evaluación final
> (`results/metricas_finales.json`, bloque `app`).

## 1. Cómo accede una persona

| Canal | Para quién | Cómo |
|---|---|---|
| Aplicación web (Streamlit) | Verificador / docente | `streamlit run app/app.py` → pega titular y cuerpo → «Analizar» |
| Línea de comandos | Usuario técnico, lotes | `python -m fakenews.predict --titular … --cuerpo …` o `--archivo lote.csv --salida resultado.csv` |
| Notebook guiado | Evaluador | `notebooks/02_resultados_y_demo.ipynb` y `00_ejecutar_en_colab.ipynb` |
| Biblioteca | Integración | `from fakenews.predict import Detector; Detector().predecir(titular, cuerpo)` |

Esquema de componentes:

```
 Usuario ──► app/app.py (Streamlit) ─┐
 Lote CSV ─► fakenews.predict (CLI) ─┼─► Detector.validar ──► normalizar ──► modelo TF-IDF + clasificador (models/*.joblib)
                                     │                                   └─► SistemaExperto (configs/reglas.yaml)
                                     └─◄ etiqueta de prioridad + probabilidad + términos + reglas con evidencia + avisos
                                                     │
                                                     └─► registro de monitoreo SIN texto (data/monitoreo/predicciones.csv)
```

## 2. Entradas válidas, salidas y errores

| Elemento | Especificación |
|---|---|
| Entrada | `titular` (texto, opcional, ≤ 400 caracteres en la app) y `cuerpo` (texto obligatorio) |
| Validaciones | cuerpo no vacío; titular textual; entre 30 y 5.000 palabras; proporción de palabras funcionales del español ≥ 0,08; se eliminan etiquetas HTML con aviso |
| Salida | `etiqueta` ∈ {sospechosa, indeterminada, probablemente confiable}; `prob_falsa` ∈ [0, 1]; `decision_binaria`; `terminos` (hasta 8 hacia «falsa» y 4 hacia «verdadera» con su contribución); `reglas` (puntaje, explicación y cada regla activada con evidencia y fundamento); `advertencias`; `aviso_de_uso`; versiones de modelo y reglas; `latencia_ms`; `requiere_revision_humana = True` |
| Umbrales | `prob ≥ 0,60` → sospechosa; `prob ≤ 0,40` → probablemente confiable; en medio → indeterminada (fijados a priori en `configs/base.yaml`) |
| Mensajes de error | En español y accionables: «El cuerpo de la noticia está vacío…», «El texto tiene N palabras; se necesitan al menos 30…», «El texto no parece estar en español…», «…el máximo es 5000. Pegue solo una noticia a la vez». Errores inesperados muestran un mensaje genérico sin traza técnica. |
| Pruebas | `tests/test_predict.py` (14 casos) y `tests/test_app.py` |

## 3. Tiempo de respuesta, costo y escalabilidad

<!-- LATENCIA -->

- **Costo**: cero por consulta. No usa APIs externas ni GPU. El modelo pesa unos pocos MB.
- **Escalabilidad**: un proceso atiende cientos de noticias por minuto en CPU; para lotes grandes se usa la
  CLI con `--archivo`. Una API (por ejemplo, FastAPI) sería un envoltorio delgado sobre `Detector.predecir`;
  no se implementó porque el alcance del prototipo es demostrativo.
- **Accesibilidad**: estados con icono + texto; funciona con teclado; textos en español.
- **Protección de datos**: ver `riesgos_etica.md` §3 (no se guarda el texto).

## 4. Plan de seguimiento

| Qué vigilar | Señal | Cómo se mide | Umbral de alerta | Acción |
|---|---|---|---|---|
| Deriva de los datos de entrada | Cambia el tipo de texto que llega (más cortos, otros temas) | Distribución de `n_palabras` y proporción de rechazos por validación en el registro de monitoreo, semanal | Cambio de la mediana > 30 % o rechazos > 20 % | Revisar el uso; ajustar límites o comunicar alcance |
| Deriva de la predicción | La proporción de «sospechosas» o «indeterminadas» cambia | Proporción semanal por etiqueta y media de `prob_falsa` | Diferencia > 15 puntos frente a las 4 semanas previas | Muestrear 30 casos y etiquetarlos a mano |
| Degradación del desempeño | Los verificadores corrigen al modelo | Muestra mensual de 50 casos etiquetados por verificadores: F1‑macro y especificidad | F1‑macro < 0,65 o especificidad < 0,75 | Reentrenar con los casos nuevos |
| Errores desiguales | Un medio o tema concentra falsos positivos | Falsos positivos confirmados por fuente/tema | Un medio con ≥ 3 falsos positivos confirmados en un mes | Revisión de sesgo; documentar; considerar excluir el rasgo |
| Reglas obsoletas | Cambian las tácticas de desinformación | Soporte y precisión de cada regla en la muestra mensual | Regla con soporte < 1 % o precisión < 50 % durante 3 meses | Nueva versión de `reglas.yaml` con registro de cambios |
| Incidentes | Uso indebido, quejas de medios | Canal de reporte | Cualquier queja | Revisión humana, respuesta y registro |

**Reentrenamiento.** Cada vez que se acumulen ≥ 200 noticias nuevas etiquetadas por verificadores (idealmente
colombianas y recientes): se agregan a los datos con su fecha, se repite la bitácora con validación agrupada
por fuente **y temporal** (entrenar con lo antiguo, validar con lo reciente), se compara con el modelo vigente
en una prueba nueva no usada, y solo se publica una nueva versión (`1.1.0`, …) si mejora sin empeorar la
especificidad. La versión del modelo, de las reglas y de los datos queda en los metadatos del `.joblib` y se
muestra en la app.

## 5. Por qué no se despliega en producción

El modelo se entrenó con noticias mexicanas de 2018 y su desempeño fuera de distribución (ver informe) no es
suficiente para decisiones reales. Se entrega como aplicación local reproducible y con un diseño de despliegue
y seguimiento completo, apto para una prueba piloto con verificadores bajo supervisión.
