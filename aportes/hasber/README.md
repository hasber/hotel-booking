# Aporte de Hasber — Proyecto I

Primera versión completa del caso (7 de noviembre), construida por etapas: entendimiento del problema, perfilado, limpieza, EDA por dimensiones de negocio, modelado, verificación de fuga y drift, interpretación y traspaso. Sirvió de base para el notebook principal del repositorio y se conserva aquí como aporte, con su propio protocolo y sus propias cifras.

## Contenido
- `Trabajo.ipynb`: notebook completo y ejecutado (147 celdas, ~6 min).
- `tools/build_trabajo.py`: genera `Trabajo.ipynb` de forma reproducible (nbformat); `tools/psi_drift.py`: PSI por variable entre dos periodos; `tools/build_despliegue.py`: genera el plan de despliegue.
- `Despliegue_FastAPI.ipynb`: plan de despliegue del modelo, no ejecutado.
- `data/hotel_bookings_limpio.csv`: dataset preparado (87.230 filas) que escribe el notebook.
- `models/`: pipeline final entrenado (joblib) y metadata (features, umbrales, métricas).
- `docs/`: `contexto_dataset.md` (qué es una fila, target, variables prohibidas), `contexto_proyecto.md` (traspaso con todas las cifras), `registro_feedback.md`, `avance_11_noviembre.md`, `base_conocimiento_materia.md`, `perfil_hotel_bookings.md`.
- `requirements.txt`: librerías de esta versión (incluye shap y optuna).

## Protocolo (distinto al del notebook principal)
- Se eliminan los 31.994 duplicados exactos y 166 reservas sin huéspedes → 87.230 filas, 27,5 % de cancelación.
- 26 variables (incluye país, agencia, depósito, tarifa, peticiones especiales y tipo de cliente); se excluyen `reservation_status*` (fuga) y `assigned_room_type`, `required_car_parking_spaces` (datos de check-in).
- Partición aleatoria estratificada 75/25 con validación cruzada de 5 folds sobre train; test una sola vez. Además, partición temporal (entrenar 2015–2016, probar 2017) como prueba de drift.
- Modelo final: Gradient Boosting afinado con Optuna.

## Resultados
- CV 5 folds: Random Forest 0,890, Gradient Boosting 0,888, árbol 0,841, logística 0,829, KNN 0,826.
- Test aleatorio: ROC-AUC 0,898, PR-AUC 0,777; umbral 0,37 → F1 0,710, recall 75 %, precisión 67 %; el decil de mayor riesgo se cancela en 87,2 % de los casos.
- Partición temporal (2017): ROC-AUC 0,837.
- Sin fuga: ninguna variable sola supera 0,66 de AUC; con duplicados el AUC se inflaba a 0,936 por memorización.

## Cómo reejecutar
Desde esta carpeta, con `hotel_bookings.csv` copiado o enlazado aquí (el notebook lo busca en la carpeta de trabajo):

```bash
pip install -r requirements.txt
python tools/build_trabajo.py
python -m jupyter nbconvert --execute --to notebook --inplace Trabajo.ipynb --ExecutePreprocessor.timeout=1800
```

## Comparación con el notebook principal
No son comparables de forma directa: el principal conserva los duplicados, usa 15 variables y una partición temporal train/validación/test; esta versión deduplica, usa 26 variables y partición aleatoria. La diferencia de AUC (0,70 frente a 0,90) viene sobre todo de las variables excluidas (país, agencia, depósito, peticiones) y, en menor medida, de la partición; con el protocolo temporal y las 26 variables el AUC fue 0,837.
