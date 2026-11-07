# Registro de retroalimentación — Cancelación de reservas de hotel

| # | Fecha | Fuente | Comentario (textual) | Tipo | Etapa | Cambio realizado (dónde) | Estado |
|---|---|---|---|---|---|---|---|
| 1 | 2026-11-07 | usuario | "la idea para el data set es el target para predecir los clientes que tienen más posibilidades de cancelar" | alcance | 04 | Target confirmado (`is_canceled`); se añadió la sección 6.8.1 (deciles de riesgo y lift) y el ranking se repite para el modelo final en 7.5; recomendación operativa orientada a priorizar reservas por probabilidad | hecho |
| 2 | 2026-11-07 | autorrevisión | El afinado con Optuna supera al por defecto por 0,003 en CV completo (fuera del margen ±0,002 fijado) | corrección | 05 | Paso 5 reescrito: el afinado es el modelo final y se evalúa en test una sola vez; 6.7 queda como reporte del base; sección 8 y traspaso actualizados | hecho |
| 3 | 2026-11-07 | autorrevisión | `required_car_parking_spaces` tiene 0 cancelaciones en 7.306 reservas: se comporta como dato de check-in | corrección | 01–04 | Movida a `post_event_columns`; demostración en 7.3 (incluirla sube el AUC +0,024) | hecho |
| 4 | 2026-11-07 | autorrevisión | 31.994 duplicados exactos concentrados en Groups/Offline TA + Non Refund: ¿bloques reales o repetidos? | profundización | 02, 05 | Decisión documentada (eliminar) + cuantificación de la inflación de métricas en 7.3 (0,936 vs 0,894) + supuesto a confirmar con el dueño del dato | hecho; supuesto pendiente |
| 5 | 2026-11-07 | autorrevisión | La partición temporal cae a 0,837: drift real | profundización | 05 | Sección 7.4 con PSI, cambio de mezcla de agencias y variantes sin agent/country; limitación y recomendación de reentrenar | hecho |
| 6 | 2026-11-07 | profesor | "El proyecto debe comenzar con un entendimiento claro del problema… qué se busca resolver, qué tipo de problema, cuál será el alcance y qué resultados o entregables se esperan" | comunicación | 00 | Sección 0 nueva en `Trabajo.ipynb` (qué se resuelve, tipo de problema, alcance dentro/fuera, entregables, mapa fases pedidas → secciones) | hecho |
| 7 | 2026-11-07 | profesor | "pongan dentro del repositorio de GitHub todo lo que hicieron… en caso contrario no habría visibilidad de mi lado" | alcance | 07 | Repositorio git inicializado con notebook, generadores, datos, modelo, docs, skills y material del curso; `.gitignore` y `requirements.txt`; README con estructura | hecho (falta crear el remoto en GitHub y hacer push) |
| 8 | 2026-11-07 | profesor | "El miércoles 11 de noviembre tendremos los primeros avances del proyecto" | alcance | 07 | `docs/avance_11_noviembre.md` con estado por fase y decisiones a discutir en la asesoría | hecho |
| 9 | 2026-11-07 | profesor | "tendrán un espacio de 20 minutos de exposición + 5 minutos de preguntas… ejercicio para practicar storytelling" | alcance | 06 | El guion pasa de 5 a 20 minutos (sección 9 del traspaso); la presentación sigue pendiente | pendiente |
| 10 | 2026-11-07 | profesor + usuario | Clase sobre despliegue con FastAPI; el usuario pide "lo de las fast api no lo hagas aún, déjalo en otro libro como algo a aplicar" | alcance | — | `Despliegue_FastAPI.ipynb` con el plan y el borrador del servicio, sin ejecutar; el notebook principal guarda el artefacto `models/modelo_cancelacion.joblib` | pendiente (aplicar tras la clase) |
| 11 | 2026-11-07 | material del curso | Sesiones 2–6 del curso (CRISP-DM, preparación de datos, modelo, pipelines) | profundización | 00 | `docs/base_conocimiento_materia.md`: concepto de la materia → dónde se aplica en el proyecto; mapa CRISP-DM en la sección 0 del notebook | hecho |
| 12 | 2026-11-07 | usuario | "creemos el repositorio llamado hotel-booking, lo vamos a dejar público… elimina todo aquello que no haga parte de la entrega" | alcance | 07 | Repositorio público `hotel-booking`; fuera del repo: material del curso, paquete de procedimiento y archivos de Claude Code; `psi_drift.py` movido a `tools/` | hecho |

Tipos: corrección (algo está mal) · profundización (falta evidencia) · comunicación (no se entiende) · alcance (pide algo nuevo).

Pendientes para la siguiente iteración:
- [ ] Crear el repositorio remoto en GitHub y subir el contenido (fecha máxima 25 de noviembre).
- [ ] Presentación (skill 06): diapositivas, guion de 20 minutos con storytelling y banco de preguntas a partir de las secciones 9–10 del traspaso.
- [ ] Aplicar el despliegue con FastAPI (`Despliegue_FastAPI.ipynb`) tras la clase del 11 de noviembre.
- [ ] Llevar a la asesoría del 11 las decisiones de `docs/avance_11_noviembre.md` §3 (duplicados, post-evento, umbral, drift).
- [ ] Confirmar con el dueño del dato el supuesto de los duplicados y la disponibilidad de `booking_changes` al momento de predecir.
