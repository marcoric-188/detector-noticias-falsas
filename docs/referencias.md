# Referencias

> Secciones 13 y 14 del formato. Criterio 2 de la rúbrica. Formato uniforme: autor u organización (año).
> Título. Publicación o repositorio. URL. Fecha de consulta. Cada referencia indica qué decisión del proyecto
> respalda. Todas las fuentes fueron abiertas y verificadas el 23‑11‑2026; las ideas se sintetizan con
> palabras propias, sin copiar párrafos.

| ID | Referencia | URL | Consulta | ¿Qué decisión respalda? |
|---|---|---|---|---|
| R01 | Posadas‑Durán, J. P., Gómez‑Adorno, H., Sidorov, G. y Moreno Escobar, J. J. (2019). Detection of fake news in a new corpus for the Spanish language. *Journal of Intelligent & Fuzzy Systems*, 36(5), 4869–4876. DOI 10.3233/JIFS‑179034 | https://doi.org/10.3233/JIFS-179034 | 2026‑11‑23 | Origen y guía de anotación del corpus v1; referencia de desempeño (77,28 % de exactitud, cifra citada por R05) para el criterio CE1. |
| R02 | Gómez‑Adorno, H., Posadas‑Durán, J. P., Bel Enguix, G. y Porto Capetillo, C. (2021). Overview of FakeDeS at IberLEF 2021: Fake News Detection in Spanish Shared Task. *Procesamiento del Lenguaje Natural*, 67, 223–231 | https://core.ac.uk/download/479277994.pdf | 2026‑11‑23 | Corpus v2 como prueba fuera de distribución; líneas base publicadas (BoW+SVM F1‑macro 0,7217; BETO 0,7321; mejor sistema 0,7666) para CE3; protocolo de entrenar con las 971 noticias de v1. |
| R03 | Posadas‑Durán, J. P. et al. (2021). *The Spanish Fake News Corpus* (repositorio, versiones 1.0 y 2.0, licencia CC‑BY‑4.0). GitHub, commit 732f487 | https://github.com/jpposadas/FakeNewsCorpusSpanish | 2026‑11‑23 | Fuente de datos, licencia, descripción de columnas, criterio de etiquetado y temas de cada versión. |
| R04 | Murayama, T. (2021). Dataset of Fake News Detection and Fact Verification: A Survey. arXiv:2111.03299 | https://arxiv.org/abs/2111.03299 | 2026‑11‑23 | Panorama de datasets (118 relacionados con noticias falsas); justifica revisar ruido de etiquetas y cambio en el tiempo. |
| R05 | Martínez‑Gallego, K., Álvarez‑Ortiz, A. M. y Arias‑Londoño, J. D. (2021). Fake News Detection in Spanish Using Deep Learning Techniques. Universidad de Antioquia. arXiv:2110.06461 | https://arxiv.org/abs/2110.06461 | 2026‑11‑23 | Antecedente colombiano: con poco dato en español, TF‑IDF + modelos clásicos (≈ 80 %) igualan a BETO+LSTM; respalda empezar con modelos lineales. |
| R06 | Iceland, M. (2023). How Good Are SOTA Fake News Detectors? arXiv:2308.02727 | https://arxiv.org/abs/2308.02727 | 2026‑11‑23 | Hallazgo de que los modelos tradicionales generalizan mejor fuera de su dataset; motiva la prueba FD y no confiar en la cifra dentro del dataset. |
| R07 | Kapoor, S. y Narayanan, A. (2023). Leakage and the reproducibility crisis in machine‑learning‑based science. *Patterns*, 4(9), 100804 | https://doi.org/10.1016/j.patter.2023.100804 | 2026‑11‑23 | Taxonomía de fuga de datos; respalda la deduplicación entre particiones, el ajuste dentro de `Pipeline` y el uso único de la prueba. |
| R08 | Thorne, J. y Vlachos, A. (2018). Automated Fact Checking: Task formulations, methods and future directions. COLING 2018. arXiv:1806.07687 | https://arxiv.org/abs/1806.07687 | 2026‑11‑23 | Distinción entre verificar hechos (con evidencia externa) y clasificar texto; delimita el alcance («no determina la verdad»). |
| R09 | IFLA – International Federation of Library Associations and Institutions (2017). *How To Spot Fake News* (infografía, CC‑BY‑4.0) | https://repository.ifla.org/handle/123456789/167 | 2026‑11‑23 | Base de conocimiento del sistema experto: cada regla de `configs/reglas.yaml` se deriva de uno de sus pasos. |
| R10 | IFLA (2020). *How to Spot Fake News – COVID‑19 Edition* (infografía) | https://repository.ifla.org/handle/20.500.14598/1289 | 2026‑11‑23 | Reglas R04 (llamado a difundir) y R09 (afirmaciones milagrosas de salud). |
| R11 | Oshikawa, R., Qian, J. y Wang, W. Y. (2018). A Survey on Natural Language Processing for Fake News Detection. arXiv:1811.00770 | https://arxiv.org/abs/1811.00770 | 2026‑11‑23 | Riesgo de que el modelo aprenda el estilo del medio y no la falsedad; motiva la validación agrupada por fuente. |
| R12 | Wang, W. Y. (2017). "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection. arXiv:1705.00648 | https://arxiv.org/abs/1705.00648 | 2026‑11‑23 | Dataset LIAR considerado y descartado (inglés, frases cortas); muestra que con textos cortos el desempeño baja mucho. |
| R13 | Pedregosa, F. et al. (2011). Scikit‑learn: Machine Learning in Python. *JMLR*, 12, 2825–2830; y documentación oficial de scikit‑learn (guía de validación cruzada y `Pipeline`) | https://scikit-learn.org/stable/modules/cross_validation.html | 2026‑11‑23 | Implementación de TF‑IDF, modelos lineales, `StratifiedGroupKFold`, `GridSearchCV` y calibración. Licencia BSD‑3. |
| R14 | Shu, K., Sliva, A., Wang, S., Tang, J. y Liu, H. (2017). Fake News Detection on Social Media: A Data Mining Perspective. arXiv:1708.01967 | https://arxiv.org/abs/1708.01967 | 2026‑11‑23 | Definiciones y distinción entre enfoques por contenido y por contexto social; este prototipo solo usa contenido. |
| R15 | Rashkin, H., Choi, E., Jang, J. Y., Volkova, S. y Choi, Y. (2017). Truth of Varying Shades: Analyzing Language in Fake News and Political Fact‑Checking. EMNLP 2017 | https://aclanthology.org/D17-1317/ | 2026‑11‑23 | La sátira, el engaño y la propaganda tienen estilos distintos; explica por qué el corpus v1 (con mucha sátira) facilita la clasificación. |
| R16 | Gebru, T. et al. (2021). Datasheets for Datasets. *Communications of the ACM*, 64(12). arXiv:1803.09010 | https://arxiv.org/abs/1803.09010 | 2026‑11‑23 | Estructura de `ficha_datos.md`. |
| R17 | Mitchell, M. et al. (2019). Model Cards for Model Reporting. FAT* 2019. arXiv:1810.03993 | https://arxiv.org/abs/1810.03993 | 2026‑11‑23 | Estructura de `models/MODEL_CARD.md` (uso previsto, métricas por subgrupo, límites). |
| R18 | NIST (2023). *Artificial Intelligence Risk Management Framework (AI RMF 1.0)* | https://www.nist.gov/itl/ai-risk-management-framework | 2026‑11‑23 | Organización de la matriz de riesgos y del plan de seguimiento. |
| R19 | Ochoa, J. (s. f.). *Taller: 100 proyectos de Machine Learning*, ficha 22, y *Formato estándar para el desarrollo de proyectos de ML* | Material del curso | 2026‑11‑23 | Enunciado, ruta de trabajo y entregables. |

## Cómo se evaluaron las fuentes (sección 13 del formato)

| Fuente | Autoría | Actualidad | Verificabilidad | Intereses / límites | Correspondencia |
|---|---|---|---|---|---|
| R01–R03 | Grupos académicos UNAM / IPN; revista indexada | 2019–2021, igual que los datos | Corpus y etiquetas públicos | Los autores del corpus evalúan sobre su propio corpus | Se cita solo para datos, guía y cifras |
| R05 | Universidad de Antioquia | 2021 | Preprint con tablas | Preprint sin revisión por pares; mezcla dos corpus | Se cita como antecedente, no como cifra comparable directa |
| R06 | Universidad de Rochester | 2023 | Preprint con tablas | Un solo autor, sin revisión por pares | Se cita para el hallazgo cualitativo de generalización |
| R07 | Princeton, revista *Patterns* | 2023 | Revisión por pares, código abierto | Ninguno identificado | Taxonomía de fuga |
| R09–R10 | Organización internacional de bibliotecas | 2017 / 2020 | Infografía pública con licencia abierta | Material divulgativo, no empírico | Se usa como conocimiento experto, no como evidencia de desempeño |
