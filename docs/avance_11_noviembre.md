# Avance del proyecto — sesión del 11 de noviembre

Proyecto: **Cancelación de reservas de hotel** · Dataset: *Hotel Booking Demand* (Antonio, Almeida & Nunes, 2019), 119.390 reservas de dos hoteles de Portugal, llegadas jul-2015 a ago-2017 · Autor: Hasber Malpica.

## 1. Entendimiento del problema
- **Qué se resuelve:** anticipar, al momento de la reserva, qué clientes/reservas tienen más posibilidades de cancelar, para que revenue management actúe (confirmación activa, depósito, overbooking calibrado).
- **Tipo de problema:** clasificación binaria supervisada sobre `is_canceled`, con salida en probabilidad para ordenar por riesgo. Tiene componente temporal (tres años).
- **Alcance:** EDA interpretado por dimensiones de negocio, preparación documentada, modelo con validación cruzada y test único, verificación de fuga y drift, interpretación. Fuera: causalidad, datos externos, despliegue completo (queda planteado).
- **Pregunta de negocio:** ¿Qué características de la reserva, del cliente, del canal y de las condiciones están asociadas con la cancelación y cuáles tienen mayor capacidad de predecirla?

## 2. Estado de cada fase (todo en `Trabajo.ipynb`, ejecutado completo)
| Fase | Estado | Resultado resumido |
|---|---|---|
| Exploración de datos (perfilado + EDA) | ✅ | 32 variables clasificadas en 5 dimensiones; tablas con tres métricas (reservas, canceladas, tasa) y gráficos interpretados; hallazgo principal: la tasa sube con la anticipación (8,4 % → 39,7 %) y el mercado doméstico la amplifica (68,8 % vs 29,6 % con > 180 días) |
| Preparación de datos | ✅ | 119.390 → 87.230 filas (31.994 duplicados exactos, 166 reservas sin huéspedes); nulos tratados sin regla única; `adr` −6,38 y 5.400 a nulo; atípicos conservados; variables derivadas; tabla de decisiones con alternativa descartada |
| Modelo | ✅ | 6 modelos comparados con CV 5 folds (Dummy 0,50 → logística 0,829 → Gradient Boosting 0,888 / Random Forest 0,890); afinado con Optuna 0,891 → **modelo final: Gradient Boosting afinado, ROC-AUC 0,898 en test**; umbral de negocio 0,37 (F1 0,710, recall 75 %); el 10 % de reservas de mayor riesgo predicho se cancela en 87,2 % de los casos |
| Verificación | ✅ | Sin fuga (ninguna variable sola > 0,66; variables post-evento excluidas); duplicados inflaban el AUC a 0,936; drift temporal real: AUC 0,837 al predecir 2017 con 2015–2016 |
| Interpretación | ✅ | Importancia por permutación, impureza, coeficientes y SHAP coinciden: país, agencia, anticipación, segmento, peticiones especiales |
| Presentación (20 min + 5) | ⏳ | Pendiente: diapositivas, guion y banco de preguntas a partir de `docs/contexto_proyecto.md` |
| Despliegue (FastAPI) | ⏳ | Planteado en `Despliegue_FastAPI.ipynb`, se aplicará tras la clase de despliegue; el artefacto `models/modelo_cancelacion.joblib` ya existe |

## 3. Decisiones que vale la pena discutir en la asesoría
1. **Duplicados exactos (26,8 % de las filas):** se eliminaron porque no hay ID de reserva y conservarlos pone copias idénticas en train y test (memorización). Parecen bloques de reservas de grupos. ¿Es aceptable para el curso o se prefiere conservarlos con partición por grupo?
2. **`required_car_parking_spaces` y `assigned_room_type`** se trataron como post-evento (0 cancelaciones en 7.306 reservas con parqueo). Incluirlas sube el AUC +0,024, pero no se conocen al reservar.
3. **Modelo final afinado vs. por defecto:** el afinado gana 0,003 en CV (fuera del margen de empate de ±0,002); test se reporta para ambos, la selección se hizo solo en CV.
4. **Drift temporal:** la métrica honesta para producción sería 0,84, no 0,89. ¿Conviene presentar la partición temporal como evaluación principal?
5. **Umbral:** 0,37 (máx F1) o 0,31 (recall 80 %); depende del costo de falso negativo vs. falso positivo que defina revenue.

## 4. Próximos pasos hasta el 25 de noviembre
- Presentación de 20 minutos con storytelling (problema → hallazgo → modelo → verificación → recomendación) y banco de preguntas.
- Incorporar la retroalimentación de la asesoría del 11 en `docs/registro_feedback.md` y regenerar el notebook si cambia alguna cifra.
- Aplicar el despliegue con FastAPI según lo visto en clase.
- Mantener el repositorio actualizado (fecha máxima 25 de noviembre).

## 5. Dónde está cada cosa en el repositorio
`Trabajo.ipynb` (notebook completo) · `tools/build_trabajo.py` (lo genera) · `data/hotel_bookings_limpio.csv` · `models/` (modelo y metadata) · `docs/contexto_dataset.md`, `docs/contexto_proyecto.md` (todas las cifras), `docs/registro_feedback.md` · `Despliegue_FastAPI.ipynb` (plan) · `docs/base_conocimiento_materia.md` (conceptos del curso aplicados) · `docs/avance_11_noviembre.md` (este documento).
