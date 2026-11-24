# Ficha técnica de los datos

> Pasos 5 y 6 del formato, sección 6. Criterio 3 de la rúbrica. Estructura inspirada en *Datasheets for
> Datasets* [R16]. Las cifras de la tabla de hallazgos se reproducen en `notebooks/01_exploracion_datos.ipynb`
> y en `results/tablas/eda_*.csv`.

## 1. Ficha

| Elemento | Registro |
|---|---|
| Nombre y fuente | *The Spanish Fake News Corpus*, versión 1.0 (MEX‑A3T 2020) y versión 2.0 (FakeDeS @ IberLEF 2021) [R03] |
| URL, autor y fecha de consulta | https://github.com/jpposadas/FakeNewsCorpusSpanish · Posadas‑Durán, Gómez‑Adorno, Sidorov, Bel Enguix, Porto Capetillo y colaboradores (IPN, UNAM) · consultado el 23‑11‑2026 · commit fijado `732f4870abd8d3f5ab4956fd92b040a7913614e3` |
| Evidencia de acceso | `python -m fakenews.data descargar` descarga los tres archivos desde el commit fijado y verifica su SHA‑256 (valores en `configs/base.yaml`). Si el hash no coincide, el script se detiene. |
| Licencia y condiciones | CC‑BY‑4.0 (insignia y sección *License* del README del repositorio). Permite usar y redistribuir citando a los autores. Aun así, **el repositorio de este proyecto no incluye los datos**: se descargan con el script y se cita a los autores en el README. |
| Formato y tamaño | 3 archivos Excel (.xlsx): `train.xlsx` 748 KB, `development.xlsx` 362 KB, `test.xlsx` 923 KB |
| Número de registros | v1: 676 (entrenamiento) + 295 (desarrollo, usada como prueba DD) = 971. v2: 572 (prueba FD) |
| Periodo y cobertura | v1: enero–julio 2018, mayoritariamente medios mexicanos; 9 temas (Política, Entretenimiento, Sociedad, Deporte, Ciencia, Salud, Economía, Seguridad, Educación). v2: noviembre 2020–marzo 2021; 7 temas (COVID‑19, Sociedad, Política, Internacional, Ciencia, Ambiental, Deporte); verificadores de Argentina, Bolivia, Chile, **Colombia**, Costa Rica, Ecuador, España, EE. UU., Francia, Perú, Uruguay, Inglaterra y Venezuela |
| Unidad de análisis | Una noticia (titular + cuerpo) publicada en la web o, en v2, una publicación de redes sociales con forma de noticia |
| Variables | `Id`, `Category` (etiqueta), `Topic`, `Source` (medio), `Headline` (titular), `Text` (cuerpo), `Link` (URL). En v2 los nombres están en mayúsculas y la etiqueta es booleana (`True` = verdadera). |
| Variable objetivo | `Category`: *True* / *Fake* → se codifica 1 = falsa, 0 = verdadera |
| Variables de entrada usadas | Titular y cuerpo (texto). `Topic`, `Source` y `Link` **no** se usan como entrada (no estarían disponibles o revelarían la etiqueta); solo se usan para análisis por subgrupo y para la validación agrupada. |
| Guía de etiquetado | v1: verdadera si hay evidencia de publicación en un sitio confiable; falsa si un sitio confiable o un verificador la contradice, o si no hay otra evidencia que la fuente original; se recolectaron pares verdadera–falsa del mismo evento. v2: falsa si un verificador (mayoría firmantes de la IFCN) lo determina; se agrega su par verdadero publicado en un medio confiable [R03]. No se reporta acuerdo entre anotadores. |
| Datos sensibles | Nombres de figuras públicas (políticos, artistas). No hay datos de personas privadas identificadas a propósito. No se usan perfiles de redes ni datos de usuarios. |
| Partición prevista | Oficial: v1 `train` para entrenamiento y validación cruzada; v1 `development` como prueba DD (una vez); v2 `test` como prueba FD (una vez). Ver `plan_evaluacion.md`. |

## 2. Tabla de hallazgos de calidad (EDA)

| # | Hallazgo | Cuantificación | Riesgo | Tratamiento |
|---|---|---|---|---|
| H1 | Clases balanceadas | Entrenamiento 338 / 338; DD 142 falsas / 153 verdaderas; FD 286 / 286 | Bajo | F1‑macro como métrica principal; no se requiere remuestreo |
| H2 | Sin valores faltantes en v1; en v2 faltan titulares | v1: 0 nulos. v2: 72 titulares vacíos (12,6 %), 7 fuentes y 3 enlaces vacíos | El modelo podría usar la ausencia de titular como señal | Titular vacío → cadena vacía; subgrupo «sin titular» en la evaluación FD |
| H3 | **Normalización distinta entre versiones** | v1 reemplazó cifras por `*NUMBER*` (5.076 apariciones), URLs por `*URL*` y teléfonos por `*PHONE*`; v2 tiene los dígitos originales | Cambio de distribución artificial: el modelo aprendería un token que no existe en v2 | Se aplica la **misma normalización a todo**: cifras → `NUM`, URLs → `URL`, correos → `EMAIL` (`src/fakenews/preprocessing.py`) |
| H4 | En v1 el cuerpo empieza repitiendo el titular | 95 % de las falsas y 98 % de las verdaderas en entrenamiento; en v2 solo 1 % | Doble conteo del titular solo en v1 | Se elimina el titular del inicio del cuerpo antes de unirlos |
| H5 | Titulares en mayúsculas casi solo en falsas | Primera línea en mayúsculas: 15,4 % de las falsas vs 1,5 % de las verdaderas (entrenamiento) | Señal de formato del sitio de origen, no de falsedad | Se mantiene en minúsculas por defecto (E‑03) y se mide su efecto (E‑08). La regla R02 la usa a sabiendas. |
| H6 | Diferencia de longitud entre clases | Mediana de palabras: falsas 269 vs verdaderas 382 (entrenamiento) | El modelo podría usar la longitud como atajo | Análisis de error por tercil de longitud |
| H7 | **Las falsas de v1 vienen sobre todo de sitios satíricos** | Entrenamiento: El Dizque 99, El Ruinaversal 55, Hay noticia 31, Censura 0 25 (62 % de las falsas en 4 medios) | El modelo aprende «estilo de sátira mexicana», no desinformación general [R15] | Validación agrupada por fuente; prueba FD (v2 trae desinformación real de redes y verificadores) |
| H8 | **Fuentes compartidas entre particiones** | 43 de las 72 fuentes de DD (60 %) también están en entrenamiento | Prueba DD optimista: mide reconocimiento de medio | Reportar CV agrupada por fuente como estimación principal para selección |
| H9 | **Duplicados entre entrenamiento y prueba DD** | 5 textos de DD con similitud coseno ≥ 0,9 con un texto de entrenamiento (4 idénticos); algunos con titulares distintos | Fuga directa | Se eliminan de la prueba DD (quedan 290). Prueba automática `tests/test_particiones.py` |
| H10 | Duplicados entre v1 y v2 | 4 textos de v2 con similitud ≥ 0,9 con v1 | Fuga hacia la prueba FD | Se eliminan de la prueba FD (quedan 568) |
| H11 | Errores de etiquetado/alineación posibles | Algunos pares DD–entrenamiento comparten el mismo cuerpo con titulares distintos | Ruido de etiquetas | Se documentan; se revisan en el análisis de errores |
| H12 | En v2 hay publicaciones de redes sociales solo en la clase falsa | 90 publicaciones (15,7 % del total), todas falsas [R03] | El formato de redes delata la clase | Subgrupo «sin titular» como aproximación; se reporta aparte |
| H13 | Variantes del español | v1 casi solo México; v2 incluye 13 países | Error desigual por país | Métricas por dominio del enlace (.co, .mx, .es, .ar, otros) en FD |

## 3. Representatividad y posibles sesgos

- **Temporal**: 2018 vs 2020–21. Las noticias de COVID‑19 no existen en v1.
- **Geográfico**: el entrenamiento representa casi solo a México. Colombia aparece únicamente en v2.
- **Tipo de falsedad**: v1 está dominada por sátira; v2 por desinformación verificada por *fact‑checkers*.
- **Medios**: pocas fuentes producen la mayoría de las falsas. Un modelo puede castigar el estilo de un medio
  y no su contenido.
- **Tamaño**: 676 ejemplos de entrenamiento es poco para modelos grandes; respalda modelos lineales [R05].

## 4. Limitaciones conocidas

- No hay acuerdo entre anotadores publicado.
- «Falsa» mezcla sátira, bulos y desinformación: es una etiqueta heterogénea.
- La etiqueta depende de lo que existía en la web en 2018 / 2021; algunas «verdaderas» pueden haberse
  corregido después.
- Los textos pertenecen a sus medios; el uso aquí es académico y sin redistribución.
