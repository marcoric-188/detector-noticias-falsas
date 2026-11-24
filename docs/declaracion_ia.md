# Declaración de uso responsable de herramientas de IA

> Sección 18 del formato. Condición excluyente de la rúbrica: sin esta declaración el trabajo no se califica.

| Pregunta | Respuesta |
|---|---|
| **Herramientas utilizadas** | Claude (Anthropic), modelo Claude Opus 5.5, usado como asistente de programación y redacción en una sesión del 23‑11‑2026. No se usó ninguna IA dentro del sistema: el detector es un modelo lineal de scikit‑learn y un sistema de reglas. |
| **¿Para qué se utilizaron?** | (1) Leer la presentación del estudiante, la rúbrica, el taller y el formato, y proponer la estructura del repositorio. (2) Buscar y verificar fuentes (artículos de FakeDeS 2021, Universidad de Antioquia, Iceland 2023, Kapoor y Narayanan 2023, IFLA). (3) Escribir el código de `src/`, `app/` y `tests/`. (4) Redactar los borradores de los documentos de `docs/`. (5) Ejecutar los experimentos y la evaluación en un entorno Linux y redactar la lectura de los resultados. |
| **¿Qué se verificó de forma independiente?** | Las referencias de `referencias.md` se abrieron en línea el 23‑11‑2026 (DOI, arXiv, repositorio del corpus, IFLA). Las cifras de terceros (77,28 %, 0,7217, 0,7666) se tomaron de las fuentes citadas, no de memoria. La descarga del corpus se valida con SHA‑256. Todos los resultados propios provienen de ejecuciones registradas en `results/corridas.jsonl` y en el historial de commits, y se pueden reproducir con `make todo`. Las pruebas automáticas (`pytest`) comprueban preprocesamiento, reglas, no‑fuga, modelos, detector, app y un simulacro de la evaluación. **Pendiente del estudiante:** reejecutar el proyecto en Colab siguiendo el README, revisar el código línea a línea y leer los ejemplos de error de `results/tablas/predicciones_*.csv`. |
| **¿Qué partes son elaboración propia?** | La elección del proyecto 22, el enfoque (TF‑IDF + clasificador lineal, sistema experto combinado, foco en generalización, riesgos de falsos positivos) y la presentación inicial *Detector de noticias falsas* son del estudiante. La decisión de usar el corpus en español, el diseño de los experimentos y la redacción final se revisaron y aprobaron por el estudiante. |
| **Datos personales o confidenciales compartidos con la IA** | Ninguno. Solo documentos del curso y datos públicos con licencia CC‑BY‑4.0. |

El uso de IA no elimina la responsabilidad del estudiante sobre la exactitud, la originalidad, la privacidad, la
citación y la comprensión del trabajo. Ningún experimento se presenta como ejecutado sin haberse ejecutado:
cada cifra de la bitácora y del informe se genera con el código del repositorio. Los commits creados con ayuda
de la IA lo indican con la línea `Co-Authored-By: Claude` en el historial de git.
