# Matriz de antecedentes y preguntas de investigación

> Paso 2 del formato. Criterio 2 de la rúbrica. Las cifras se toman de las fuentes indicadas; no son
> resultados propios.

## 1. Matriz de antecedentes

| Trabajo | Objetivo | Datos | Método | Métrica y resultado reportado | Limitación | Qué tomamos para este proyecto |
|---|---|---|---|---|---|---|
| Posadas‑Durán et al. 2019 [R01] | Crear el primer corpus en español para detección de noticias falsas por estilo | Spanish Fake News Corpus v1: 971 noticias, 9 temas, México, ene–jul 2018 | Rasgos lingüísticos (n‑gramas, POS) + aprendizaje supervisado | Exactitud 77,28 % (cifra citada por R05) | Corpus pequeño; muchas falsas provienen de sitios satíricos | Mismo corpus y partición oficial; su cifra como referencia de CE1 |
| Gómez‑Adorno et al. 2021, FakeDeS [R02] | Evaluar variación temática y de país | Entrenamiento: v1 (971). Prueba: v2, 572 noticias de 2020–21, COVID‑19, varios países incluida Colombia | Equipos con BERT/BETO, XLM‑R, TF‑IDF + ensambles | Mejor F1 0,7666; BoW+SVM 0,7217; BETO 0,7321. Segundo lugar con TF‑IDF + *stacking* (0,7548) | No reporta resultados por país | Usar v2 como prueba FD y comparar contra sus líneas base |
| Martínez‑Gallego et al. 2021 [R05] | Detección en español con aprendizaje profundo | v1 + corpus de Kaggle (2.571 noticias) | TF‑IDF + SVM/RF; BETO + LSTM | ≈ 80 % de exactitud tanto clásicos como BETO+LSTM | Mezcla corpus; solo exactitud | Con pocos datos, un modelo lineal es una elección razonable y barata |
| Iceland 2023 [R06] | Medir si los detectores generalizan fuera de su dataset | ISOT, LIAR y otros (inglés) | NB, SVM, árboles, BERT, CT‑BERT | Casi 99 % dentro de ISOT; F1 cercano a 50 fuera de distribución para la mayoría de *transformers* | Inglés; un solo autor | Evaluar siempre fuera de distribución |
| Oshikawa et al. 2018 [R11] | Revisar NLP para noticias falsas | Varios | Revisión | — | — | Riesgo de etiquetar por fuente → validación agrupada por medio |
| Wang 2017, LIAR [R12] | Benchmark de frases políticas | 12,8 mil frases de PolitiFact, 6 niveles | CNN híbrida | Exactitud ≈ 27 % en 6 clases | Frases cortas, inglés | Descartado: inglés y frases cortas; confirma que texto corto es difícil (motiva E‑07) |
| Rashkin et al. 2017 [R15] | Analizar lenguaje de sátira, engaño y propaganda | Noticias en inglés | Análisis léxico + modelos | Diferencias de estilo entre tipos de desinformación | Inglés | Explicar por qué la sátira de v1 es fácil y la desinformación de v2 no |

## 2. Datasets considerados

| Dataset | Idioma | Tamaño | Acceso desde el entorno de trabajo | Decisión |
|---|---|---|---|---|
| ISOT (Univ. Victoria) | Inglés | ≈ 45 mil | Sitio oficial no accesible desde el entorno; solo copias no oficiales | Descartado: sin procedencia verificable y con fuga conocida (prefijo «(Reuters)» en las verdaderas) |
| LIAR [R12] | Inglés | 12,8 mil frases | Solo copias no oficiales | Descartado: inglés, frases cortas, 6 niveles |
| FakeNewsNet [R14] | Inglés | ≈ 23 mil titulares | Repositorio oficial accesible (solo titulares + URL) | Descartado: el texto completo exige descargar artículos protegidos por derechos de autor |
| **Spanish Fake News Corpus v1 + v2** [R03] | **Español** | **971 + 572** | **Repositorio oficial, CC‑BY‑4.0** | **Elegido**: idioma del usuario, licencia abierta, dos periodos para medir generalización, incluye Colombia en v2 |

## 3. Preguntas de investigación

1. ¿Por qué verificar hechos es distinto de clasificar estilo? (pregunta del taller)
2. ¿Cuánto del desempeño viene de reconocer el medio y no el contenido? (validación agrupada, E‑00 a E‑09)
3. ¿Cuánto cae el desempeño al pasar de 2018 a 2020–21, de política mexicana a COVID‑19 y a otros países?
4. ¿Superan las reglas de un verificador humano (IFLA) a un modelo aprendido? ¿Se complementan?
5. ¿Qué tipos de noticias confunden al modelo (sátira, opinión, textos cortos, publicaciones de redes)?
