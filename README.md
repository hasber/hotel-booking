# Proyecto 1 · Cancelaciones hoteleras

¿Qué tienen en común las reservas que más se cancelan? ¿Podemos anticiparlas? Proyecto de ciencia de datos sobre el dataset público *Hotel Booking Demand* (119.390 reservas de dos hoteles, 2015–2017), siguiendo la metodología CRISP-DM: comprensión del negocio, comprensión de los datos, preparación, modelado y evaluación.

Variable objetivo: `is_canceled` (1 = cancelada, 0 = no cancelada). Problema de clasificación binaria.

## Entrega principal

| Archivo | Qué es |
|---|---|
| `Notebook_Analisis_CRISP_DM_Hotel.ipynb` | Notebook principal, ejecutado: exploración, preparación, comparación de tres modelos, evaluación en un periodo posterior, escenarios de uso y conclusiones |
| `docs/Documentación_notebook_CRISP_DM_Hotel.md` | Informe de apoyo con la explicación ampliada de cada decisión y resultado |
| `resultados_crispdm/` | Tablas que exporta el notebook: particiones, comparación de modelos, evaluación en prueba, regla básica, escenarios de seguimiento, perfiles e importancias |
| `presentacion/` | Presentación de la sustentación (pptx y pdf, 16 diapositivas, 20 minutos) |
| `hotel_bookings.csv` | Datos originales |

## Resultado en breve
- La tasa de cancelación crece con la anticipación (de 9,6 % con 0–7 días a 67,7 % con más de 365) y es mucho mayor en reservas con cancelaciones previas (91,6 %), en el segmento Groups (61,1 %) y en el City Hotel (41,7 %). Son asociaciones, no causas.
- Con partición temporal (entrenamiento hasta junio de 2016, validación segundo semestre de 2016, prueba 2017), Random Forest obtuvo el mejor F1 de validación (0,611) con umbral 0,25. En prueba: F1 0,624, recall 89,25 %, precisión 47,99 %, ROC-AUC 0,699.
- Una regla básica que siempre predice "no se cancela" tiene más exactitud (61,3 % frente a 58,4 %) pero no detecta ninguna cancelación: por eso no se elige por exactitud.
- Si el hotel solo revisara el 10 % de reservas de mayor riesgo, encontraría el 17 % de las cancelaciones con una tasa de acierto del 67 %.

## Reproducir

```bash
pip install -r requirements.txt
```

Abrir `Notebook_Analisis_CRISP_DM_Hotel.ipynb` y ejecutarlo desde la primera celda con `hotel_bookings.csv` en la misma carpeta (o por consola: `python -m jupyter nbconvert --execute --to notebook --inplace Notebook_Analisis_CRISP_DM_Hotel.ipynb`). Semilla 42; escribe `resultados_crispdm/`.

## Aportes individuales
Versiones previas y análisis complementarios de cada integrante, con sus propios protocolos (no comparables directamente con el principal):

| Carpeta | Autor | Enfoque |
|---|---|---|
| `aportes/hasber/` | Hasber Malpica | Flujo por etapas con deduplicación, 26 variables, Gradient Boosting, verificación de fuga y drift |
| `aportes/juan/` | Juan Monterrosa | Partición temporal train/validación/test con deduplicación, Random Forest elegido por Average Precision |

Cada carpeta tiene su README con instrucciones.
