# Cancelación de reservas de hotel — proyecto de ciencia de datos

Caso resuelto de principio a fin sobre el dataset público *Hotel Booking Demand* (Antonio, Almeida & Nunes, 2019), por etapas: entendimiento del problema, perfilado, limpieza, EDA, modelado, verificación y traspaso. Pregunta: **¿qué características de la reserva, del cliente, del canal y de las condiciones están asociadas con la cancelación y cuáles la predicen?** Objetivo operativo: identificar las reservas/clientes con más posibilidades de cancelar.

## Estructura del repositorio

```
Trabajo.ipynb                      notebook principal, ejecutado: entendimiento del problema → perfilado →
                                   limpieza → EDA → modelo → verificación → qué reportar
Despliegue_FastAPI.ipynb           plan del despliegue del modelo (pendiente de aplicar; no ejecutado)
hotel_bookings.csv                 dato crudo (119.390 reservas)
data/hotel_bookings_limpio.csv     dataset preparado (87.230 filas), fuente del modelado
models/                            modelo final entrenado (joblib) y metadata (features, umbrales, métricas)
docs/avance_11_noviembre.md        estado del proyecto para la sesión de avances
docs/contexto_dataset.md           qué es una fila, target, variables prohibidas, calidad (skill 01)
docs/contexto_proyecto.md          documento de traspaso con todas las cifras (skill 07)
docs/registro_feedback.md          retroalimentación recibida y cambios aplicados (skill 07)
docs/base_conocimiento_materia.md  conceptos del curso → dónde se aplican en el proyecto
docs/perfil_hotel_bookings.md      informe automático de perfilado
tools/build_trabajo.py             generador reproducible de Trabajo.ipynb
tools/build_despliegue.py          generador de Despliegue_FastAPI.ipynb
tools/psi_drift.py                 PSI (Population Stability Index) por variable entre dos periodos; lo usa el notebook en 7.4
requirements.txt                   librerías necesarias
```

## Reproducir

```bash
pip install -r requirements.txt
```

```bash
python tools/build_trabajo.py
```

```bash
python -m jupyter nbconvert --execute --to notebook --inplace Trabajo.ipynb --ExecutePreprocessor.timeout=1800
```

Ejecución completa ≈ 6 minutos (Python 3.11+; resultados deterministas con `RANDOM_STATE = 42`). El notebook escribe `data/hotel_bookings_limpio.csv` y `models/`.

## Resultado en una línea
Gradient Boosting afinado: ROC-AUC 0,898 en test; el 10 % de reservas con mayor riesgo predicho se cancela en el 87 % de los casos; variables clave: país de origen, agencia, anticipación, segmento y peticiones especiales. Sin fuga (verificado); drift temporal real (AUC 0,837 al predecir 2017 con 2015–2016). Lenguaje de asociación, no causalidad.
