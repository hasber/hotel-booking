# Resultados del aporte de Juan

## Pregunta
¿Con qué desempeño se pueden ordenar reservas por riesgo de cancelación
al evaluar en un periodo posterior al entrenamiento?

## Preparación
- Registros originales: 119,390.
- Registros después de limpieza: 87,230.
- ADR convertidos a nulo: 2.
- Huella SHA-256 del CSV: `d10cdeb07de9309007f528d6499a2c0467ae7a6b287b80ffd759e11e61801639`.

## Protocolo
- Entrenamiento: llegadas anteriores a julio de 2016.
- Validación: llegadas de julio a diciembre de 2016.
- Test: llegadas desde enero de 2017.
- Imputación y codificación ajustadas solamente en entrenamiento.
- Modelo elegido por Average Precision en validación.
- Umbral elegido por máximo F1 en validación.
- Modelo y umbral congelados antes de evaluar test.
- No se reentrena el modelo después de seleccionar el umbral.

## Resultado en test
- Modelo: Random_Forest.
- Umbral: 0.21.
- ROC-AUC: 0.7455.
- Average Precision: 0.5697.
- Precision: 46.57%.
- Recall: 77.89%.
- F1: 0.5829.
- Prevalencia del test: 31.95%.
- Verdaderos positivos: 7,872.
- Falsos positivos: 9,030.
- Falsos negativos: 2,234.
- Verdaderos negativos: 12,497.

## Interpretación
Al umbral seleccionado, el modelo identifica 7,872 cancelaciones
y genera 9,030 alertas en reservas que no cancelaron.
Quedan 2,234 cancelaciones sin detectar.

El umbral maximiza F1 en validación; no representa un óptimo económico.
Para decidir su uso se necesitan los costos de intervenir y de no detectar
una cancelación.

Average Precision resume el desempeño precision-recall;
no se presenta como área trapezoidal bajo esa curva.

## Limitaciones
- Los duplicados exactos podrían representar reservas legítimas idénticas.
- La evaluación se hace sobre datos deduplicados y no representa
  necesariamente la prevalencia operacional.
- La partición usa fecha de llegada, no fecha de emisión de la predicción.
- No se reconstruye cuándo se conoció el resultado de cada reserva.
- Faltan versiones históricas de las variables al momento de reservar.
- Las exclusiones reducen riesgos, pero no prueban ausencia total de fuga.
- Las asociaciones encontradas no demuestran causalidad.
- Se utiliza una sola ventana de validación y una sola ventana de test.
- No se midieron intervalos de confianza.
- El rendimiento corresponde a dos hoteles y al periodo del dataset.

## Comparación con el trabajo del equipo
No comparar directamente este resultado con el AUC de un test aleatorio.
Para una comparación controlada, entrenar ambos enfoques con las mismas
filas y el mismo protocolo, usando asignacion_particiones.csv.

El modelo guardado originalmente por el equipo no debe evaluarse como
independiente sobre estas filas si pudo entrenarse con ellas.

Acordar también el tratamiento de duplicados, el momento de predicción
y las métricas antes de concluir que un enfoque es superior.
