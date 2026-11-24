# Flujo completo reproducible. En Windows sin make, ejecute los comandos de cada regla a mano.
PY ?= python

.PHONY: instalar datos eda experimentos evaluar modelo pruebas estilo app todo

instalar:            ## Instala dependencias y el paquete en modo editable
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

datos:               ## Descarga (con SHA-256) y prepara las particiones
	$(PY) -m fakenews.data preparar

eda:                 ## Reejecuta el notebook de exploración
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploracion_datos.ipynb

experimentos:        ## Bitácora completa con validación cruzada (solo entrenamiento, ~1,5 h en CPU)
	$(PY) -m fakenews.experiments

evaluar:             ## Evaluación final ÚNICA en las pruebas (se niega a repetirse)
	$(PY) -m fakenews.evaluate --confirmo-uso-unico

modelo:              ## Reconstruye el modelo de producción sin evaluar (p. ej., otra versión de scikit-learn)
	$(PY) -m fakenews.train

pruebas:             ## Pruebas automáticas
	$(PY) -m pytest

estilo:              ## Linter
	ruff check src tests app

app:                 ## Aplicación web local
	streamlit run app/app.py

todo: datos experimentos evaluar pruebas
