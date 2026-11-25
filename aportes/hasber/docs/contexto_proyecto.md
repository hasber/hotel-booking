# Documento de traspaso — Proyecto "Cancelación de reservas de hotel"

> **Para quien reciba este documento (persona o agente):** aquí está el contexto esencial del proyecto: dataset, decisiones de limpieza, variables creadas, hallazgos con cifras exactas, resultados del modelo, verificación de fuga y drift, respuesta a la pregunta de negocio y preguntas probables. Con esto puedes retomar el trabajo sin releer los notebooks.

Fecha de exportación: 2026-11-07 (actualizado con las etapas 1–5 y verificación) · Autor: Hasber Malpica, con Claude Code · Curso: Especialización (Universidad de Medellín / Comfama).

## 1. Archivos del proyecto
| Archivo | Tipo | Qué es | Lee / escribe |
|---|---|---|---|
| `hotel_bookings.csv` | Datos | 119.390 reservas × 32 columnas, llegadas 2015-07-01 → 2017-08-31, dos hoteles de Portugal | — |
| `Trabajo.ipynb` | Notebook | Todo el flujo en 9 partes (0–8): entendimiento del problema, perfilado, limpieza, variables, EDA, conclusiones, modelado, verificación, qué reportar (ejecutado completo, ~6 min) | lee crudo; escribe `data/hotel_bookings_limpio.csv` y `models/` |
| `tools/build_trabajo.py` | Script | Generador reproducible del notebook (nbformat). `python tools/build_trabajo.py` y luego nbconvert `--execute` | escribe `Trabajo.ipynb` |
| `data/hotel_bookings_limpio.csv` | Datos | 87.230 filas × 40 columnas; fuente única del modelado | — |
| `models/modelo_cancelacion.joblib`, `models/metadata.json` | Modelo | Pipeline final (preprocesador + HGB afinado) y metadata (features, umbrales, métricas en test) | escritos por el notebook (7.5) |
| `Despliegue_FastAPI.ipynb` | Notebook | Plan y borrador del servicio FastAPI; **no ejecutado**, se aplica tras la clase de despliegue | generado por `tools/build_despliegue.py` |
| `docs/avance_11_noviembre.md` | Doc | Estado por fase y decisiones a discutir en la asesoría | — |
| `docs/base_conocimiento_materia.md` | Doc | Conceptos del curso (sesiones 2–6) → dónde se aplican en el proyecto | — |
| `tools/psi_drift.py` | Script | PSI por variable entre dos periodos (sección 7.4) | — |
| `docs/contexto_dataset.md` | Doc | Skill 01: grano, diccionario, target, prohibidas, calidad | — |
| `docs/perfil_hotel_bookings.md` | Doc | Informe automático de `perfilar_dataset.py` | — |
| `docs/registro_feedback.md` | Doc | Skill 07: retroalimentación recibida y cambios | — |

## 2. Diccionario de datos (por bloques)
**Hotel y estancia:** hotel (City/Resort), stays_in_weekend_nights, stays_in_week_nights, reserved_room_type (A–L), meal (BB/HB/FB/SC). · **Cliente:** adults, children, babies, is_repeated_guest, previous_cancellations, previous_bookings_not_canceled, customer_type (Transient/Transient-Party/Contract/Group), country (ISO3, 177). · **Canal:** market_segment (8), distribution_channel (5), agent (333 códigos; nulo = sin agencia), company (352; nulo = sin empresa). · **Tiempo:** lead_time (días entre reserva y llegada), arrival_date_* (año/mes/semana/día), days_in_waiting_list. · **Condiciones:** deposit_type (No Deposit/Non Refund/Refundable), adr (tarifa diaria €), booking_changes, total_of_special_requests, required_car_parking_spaces. · **Resultado:** is_canceled (target), reservation_status, reservation_status_date, assigned_room_type.

## 2b. Entendimiento del problema (sección 0 del notebook)
Qué se resuelve: anticipar al reservar qué clientes/reservas cancelarán, para confirmar, pedir depósito o calibrar overbooking. Tipo: clasificación binaria supervisada con salida en probabilidad (ranking de riesgo), con componente temporal. Alcance: EDA interpretado, preparación documentada, modelo con CV y test único, verificación de fuga/drift, interpretación; fuera: causalidad, datos externos, despliegue completo (planteado en `Despliegue_FastAPI.ipynb`). Entregables: notebook ejecutado, CSV limpio, contexto_dataset, contexto_proyecto, modelo guardado, presentación de 20 min.

## 3. Pregunta de negocio y target
> ¿Qué características de la reserva (hotel, estancia, anticipación), del cliente (composición, historial, origen), del canal (segmento, distribución, agencia) y de las condiciones (depósito, tarifa, peticiones) están asociadas con la cancelación de una reserva y cuáles tienen mayor capacidad de predecirla?

Objetivo operativo declarado por el usuario: **predecir qué clientes/reservas tienen más posibilidades de cancelar** (ranking de riesgo). Target: `is_canceled` (1 = cancelada o no-show), clasificación binaria. Prevalencia 37,0 % en el crudo, **27,5 % tras limpieza**. Tres métricas del EDA: Reservas, Canceladas, Tasa (ponderada); impacto: ingreso en riesgo = Σ adr × noches de las canceladas. **Nota de fuga:** `reservation_status`/`_date` son el target; `assigned_room_type` y `required_car_parking_spaces` son datos de check-in (post-evento).

## 4. Hallazgos del perfilado
- Sin ID de reserva ni de cliente. 31.994 filas duplicadas exactas (26,8 %), concentradas en Groups (14.869), Offline TA/TO (10.330) y depósito Non Refund (13.549 de 14.587); las duplicadas se cancelan 58,4 % vs 26,2 %.
- Nulos: company 94,3 %, agent 13,7 % (esperables), country 488, children 4.
- `Undefined` en meal (1.169 = SC), market_segment (2), distribution_channel (5). Sin variantes de escritura.
- adr −6,38 y 5.400 (errores), adr = 0 en 1.959 (680 Complementary), 180 reservas con 0 huéspedes, 715 estancias de 0 noches (4,9 % cancelación).
- `is_canceled` ⇔ `reservation_status` ≠ Check-Out en el 100 % de las filas.
- Años incompletos: 2015 (jul–dic), 2017 (ene–ago).

## 5. Decisiones de limpieza (df → df_clean)
| Variable | Acción | Valor usado | Justificación |
|---|---|---|---|
| todas | drop_duplicates | 119.390 → 87.396 | sin ID no se valida que sean reservas distintas; evita fuga train/test (verificado en 7.3: inflaban el AUC a 0,936) |
| huéspedes = 0 | eliminar | 166 filas → 87.230 | reserva inutilizable |
| adr −6,38 / 5.400 | → NaN, conservar | 2 filas | imputación dentro del pipeline |
| children | moda | 0 | 4 nulos |
| country | constante | "Unknown" | 488 |
| agent / company | constante + bandera | "SinAgencia"/"SinEmpresa", has_agent, has_company | nulo con significado |
| meal | reemplazo | Undefined → SC | diccionario oficial |
| fechas | to_datetime | arrival_date, reservation_status_date | 0 NaT |

Filas: **119.390 → 87.230**. Outliers de adr (hasta 510) conservados; p95 solo para visualizar.

## 6. Variables creadas
| Variable | Cálculo |
|---|---|
| total_nights | weekend + week nights |
| total_guests | adults + children + babies |
| ingreso_estimado | adr × total_nights |
| room_changed | reserved ≠ assigned (solo verificación) |
| arrival_date, arrival_month_num, arrival_quarter, arrival_dow, year_month | de la fecha de llegada |
| lead_bin | 0–7 / 8–30 / 31–90 / 91–180 / >180 días |
| es_PRT | country == PRT |

Correlaciones ≥ 0,9: ninguna. Más alta con el target: lead_time +0,185; negativas: parqueo −0,184, peticiones −0,121.

## 7. Resultados clave con cifras exactas
### Indicadores globales
87.230 reservas · 24.009 canceladas (27,52 %) · ingreso estimado 34,45 M€, en riesgo 11,48 M€ (33 %) · lead_time mediana 49 días (canceladas 80 vs 38) · adr mediana 98,2 € (canceladas 109,8 vs 94,9).

### Por hotel
City 53.274 reservas / 30,1 % · Resort 33.956 / 23,5 %.

### Por canal
Online TA 51.553 (59 %) / 35,4 % / 18.242 cancelaciones (76 %) / 9,29 M€ · Offline TA/TO 13.855 / 14,8 % · Direct 11.780 / 14,7 % · Groups 4.922 / 27,1 % · Corporate 4.200 / 12,1 %. Sin agencia 12,8 %; con empresa 10,9 %.

### Por cliente
Portugal 27.355 (31,4 %) / 35,8 % / 40,7 % de cancelaciones / 36,8 % del ingreso en riesgo; GBR 19,0 %, FRA 19,6 %, ESP 25,7 %, DEU 19,6 %. Repetido 7,7 % vs 28,3 %. Cancelación previa = 1: 76,2 %. Transient 30,1 %, Transient-Party 15,2 %, Contract 16,3 %, Group 9,8 %. ≥ 4 huéspedes 41,1 %.

### Por tiempo
Anticipación: 0–7 d 8,4 % · 8–30 d 25,4 % · 31–90 d 32,0 % · 91–180 d 35,0 % · >180 d 39,7 %. Tendencia: mismo mes 2015 < 2016 < 2017 (agosto 23,3 → 32,7 → 36,7 %). Trimestres 2016: 20,2 / 25,9 / 29,8 / 28,3 %.

### Por condiciones
Non Refund 1.038 reservas / 94,7 % (983 de Portugal) · adr ≤ 50 € 15,0 % → > 200 € 37,6 % · 0 peticiones 33,3 %, ≥ 3 16,2 % · 0 cambios 30,2 %, 1 cambio 13,9 % · parqueo ≥ 1: 0 de 7.306 canceladas (post-evento) · cambio de habitación: 4,7 % vs 31,5 % (post-evento).

### Hallazgo principal: anticipación × origen
| Anticipación | Otros países | Portugal |
|---|---|---|
| 0–7 d | 5,6 % (8.759) | 11,1 % (9.444) |
| 8–30 d | 21,8 % | 33,4 % |
| 31–90 d | 27,2 % | 46,8 % |
| 91–180 d | 28,8 % | 55,1 % |
| >180 d | 29,6 % (8.710) | 68,8 % (3.047) |

Se repite en 3 de 3 años y en los 2 hoteles (City 10,5 → 44,9 %; Resort 6,0 → 33,9 %). Portugal cancela más en los 7 segmentos con reservas de ambos orígenes (Groups 58,3 vs 1,6 %; Offline TA/TO 43,4 vs 0,5 %; Online TA 43,5 vs 33,2 %). No es Non Refund (3,6 % de Portugal). Acompañantes: 5,8 % con cancelación previa (vs 0,2 %), 57 % sin peticiones (vs 47 %), lead mediana 25 vs 60 días.

### Patrón transversal
Cliente (origen, historial) y Canal (agencia, segmento) explican la eficiencia; Hotel y estancia explica volumen pero aporta poco controlando lo demás.

## 8. Conclusiones del EDA y pregunta final
1. Volumen ≠ eficiencia: City/Online TA dominan y cancelan más. 2. Directo/corporativo/offline estables (12–15 %). 3. Anticipación 8 → 40 %. 4. Portugal amplifica en todos los canales. 5. Señales de compromiso reducen; cancelación previa dispara. 6. Asociación, no causalidad; duplicados eliminados son una limitación.
> ¿Con qué precisión puede un modelo anticipar, reserva a reserva, cuáles se cancelarán usando solo lo que se conoce al reservar, y qué variables pesan más controlando todas a la vez?

## 9. Guion de exposición (propuesta, **20 min + 5 de preguntas**; pendiente de pptx — skill 06)
Formato definido por el profesor: 20 minutos de exposición, 5 de preguntas, énfasis en storytelling (mostrar todo lo elaborado).

| Bloque | Tiempo | Contenido y cifras |
|---|---|---|
| 1. El problema | 0:00–2:30 | Qué pierde el hotel con una cancelación; qué se busca (anticipar al reservar, priorizar clientes de mayor riesgo); tipo de problema (clasificación con probabilidad); alcance y entregables |
| 2. Los datos | 2:30–5:00 | 119.390 reservas, 2 hoteles, 2015–2017; sin ID; 31.994 duplicados exactos y por qué se eliminaron (memorización 0,936 vs 0,894); 87.230 filas, 27,5 % de cancelación; 5 dimensiones de negocio |
| 3. Preparación | 5:00–7:00 | Nulos sin regla única (agent/company con significado, country Unknown), adr inválidos, 166 sin huéspedes, atípicos conservados; parqueo y habitación asignada como post-evento (0 de 7.306) |
| 4. Hallazgos del EDA | 7:00–11:00 | Gráfico estrella anticipación × origen (8,4 → 39,7 %; Portugal 68,8 % vs 29,6 %); Online TA 59 % volumen / 76 % cancelaciones / 9,3 M€; señales de compromiso (repetido 7,7 %, ≥3 peticiones 16,2 %); cancelación previa 76,2 %; no es mezcla (7/7 segmentos) |
| 5. El modelo | 11:00–15:00 | Pipeline y 4 etapas de preprocesamiento; Dummy → logística 0,829 → HGB 0,888 / RF 0,890 → afinado 0,891; test: AUC 0,898, PR-AUC 0,777; umbral 0,37 (recall 75 %, precision 67 %) vs 0,31 (recall 80 %); deciles: D1 87,2 % (lift 3,2), D1+D2 = 56 % de las cancelaciones |
| 6. ¿Es confiable? | 15:00–18:00 | Sin fuga (variable sola máx 0,66; status daría 1,0; parqueo +0,024); importancias por 3 métodos + SHAP coinciden (país, agencia, anticipación, segmento, peticiones); drift temporal: 0,837 en 2017 → reentrenar |
| 7. Recomendación y cierre | 18:00–20:00 | Puntuar al reservar, revisar > 0,37 con > 30 días, empezar por Online TA y mercado doméstico; limitaciones (asociación, umbral por validar, duplicados); siguiente paso: despliegue FastAPI |
| Preguntas | 20:00–25:00 | Banco de la sección 10 |

## 10. Banco de preguntas y respuestas
- *¿Por qué eliminaron 32.000 filas?* Sin ID de reserva no se puede saber si son bloques de grupos o registros repetidos; conservarlas pone copias idénticas en train y test: el AUC "subía" a 0,936 y en las filas con copia en train era 0,983 (memorización). Se documenta como limitación y supuesto a confirmar.
- *¿Por qué la tasa es 27,5 % y no 37 %?* Porque los duplicados se concentraban en reservas canceladas (58 %). La tasa reportada es la del dataset deduplicado.
- *¿El parqueo no es una buena variable?* 0 cancelaciones en 7.306 reservas: se registra al llegar. Incluirla sube el AUC +0,024 pero es fuga.
- *¿Por qué Gradient Boosting y no la logística?* La logística da 0,829 vs 0,888: no es empate, la relación no es aditiva (anticipación × origen × canal). Entre RF (0,890) y HGB (0,888) sí hay empate y gana HGB por parsimonia; el afinado (0,891) es el final.
- *¿Por qué evaluaron test dos veces?* Test nunca eligió nada: la selección se hizo en CV. 6.7 reporta el base; 7.5 reporta el final afinado. Ambas cifras se publican.
- *¿Qué umbral usarían?* 0,37 (máx F1 0,710: recall 75 %, precision 67 %); con 0,31 recall 80 %. Revenue decide por costo de FN (habitación vacía) vs FP (contacto/overbooking innecesario).
- *¿AUC 0,90 no es sospechoso?* No: ninguna variable sola supera 0,66; la curva acumulada sube por escalones explicables; `reservation_status` sola daría 1,0.
- *¿Generaliza en el tiempo?* Peor: entrenando 2015–2016 y probando 2017, AUC 0,837 (drift en agent PSI 0,58, adr 0,14; tasa base creciente). Reentrenar con ventana móvil.
- *¿Portugal es causa?* No. Es asociación robusta (7/7 segmentos, 5/5 rangos, brecha ajustada 16,2 pp). Accionable vía política de depósito/confirmación para ese mercado.
- *¿Hicieron tuning?* Optuna 20 trials: +0,001 en submuestra, +0,003 en CV completo (max_iter es el que importa).
- *¿Por qué One-Hot con min_frequency?* País (177) y agencia (333) explotarían en columnas; con mínimo 100 filas el AUC baja solo 0,002 (0,8095 → 0,8075). "Todo ordinal" cae a 0,753.
- *¿Y las imputaciones?* Solo 2 nulos (adr): simple/KNN/indicador empatan exactamente; se usa la mediana dentro del pipeline.

## 11. Etapa de modelado y verificación (2026-11-07)
**Notebook:** `Trabajo.ipynb` partes 6–7 (ejecutado completo; todas las celdas con salida).
### Planteamiento
26 features (15 numéricas + 11 nominales). Excluidas: fuga (reservation_status, _date), post-evento (assigned_room_type, room_changed, required_car_parking_spaces), redundantes (company → has_company; año/semana/día/trimestre/mes numérico; totales derivados). `assert` de fuga. Split 75/25 estratificado (65.422 / 21.808). Preprocesador: mediana + StandardScaler; One-Hot min_frequency 100 → 156 columnas.
### Experimentación (CV 5 folds, train)
| Modelo | AUC | PR-AUC | s |
|---|---|---|---|
| Random Forest 300 | 0,8899 | 0,763 | 19 |
| Gradient Boosting (HGB) | 0,8879 | 0,759 | 8 |
| Árbol prof. 6 | 0,8406 | 0,638 | 1 |
| Logística | 0,8292 | 0,649 | 3 |
| KNN 25 | 0,8259 | 0,657 | 7 |
| Dummy | 0,5 | 0,275 | — |
| **HGB afinado (Optuna)** | **0,8911** | 0,765 | 19 |
Ganador por parsimonia entre empatados: HGB; afinado supera por 0,003 → modelo final. Parámetros: learning_rate 0,048, max_iter 501, max_leaf_nodes 48, l2 0,054, min_samples_leaf 11.
### Métricas en test
Base: AUC 0,894, PR-AUC 0,769; 0,5 → P 0,734 / R 0,611 / F1 0,667; 0,36 → P 0,663 / R 0,751 / F1 0,704 (TP 4.505, FN 1.497, FP 2.292, TN 13.514). **Final:** AUC 0,8976, PR-AUC 0,777; 0,37 → P 0,674 / R 0,749 / F1 0,710 (TP 4.495, FN 1.507, FP 2.173, TN 13.633); recall ≥ 80 % en 0,31. Deciles (final): D1 87,2 % (lift 3,17), D2 66,8 %, D3 49,6 %; D1+D2 = 56,0 % de las cancelaciones, D1–D3 = 74,0 %.
### Verificación
Variable sola: agent 0,658, lead_time 0,638, market_segment 0,619, adr 0,611, country 0,608. Ablación (de 0,882): country −0,061, peticiones −0,023, lead_time −0,022, agent −0,020. Curva acumulada: agent 0,658 → +lead_time 0,729 → +country 0,827 → +peticiones 0,857 → meseta 0,86–0,88. Prohibidas: status 1,0; parqueo + cambio de habitación llevan 0,882 → 0,905. Duplicados: crudo 0,936 (copias en train 0,983; resto 0,884). PSI 2015–16 vs 2017: arrival_date_month 5,2 (cobertura), agent 0,58, adr 0,14, country 0,11, target 0,02. Adversarial 0,5005. Temporal 2017: 0,837 (sin agent 0,803; sin agent ni country 0,781).
### Interpretación
Permutación (base): country 0,119, agent 0,099, lead_time 0,083, market_segment 0,064, peticiones 0,045, customer_type 0,033. Por dimensión: Cliente 35,2 %, Canal 32,9 %, Tiempo 17,0 %, Condiciones 13,3 %, Hotel 1,6 %. Subconjuntos (30k, 3 folds): todas 0,882, top-6 0,863, top-3 0,819, sin country 0,821, sin country ni agent 0,808, solo lead_time 0,638. Logística: lead_time +0,59, previous_cancellations +0,45, peticiones −0,63, Non Refund +1,60, PRT +1,42. Brecha Portugal: cruda 12,2 pp, ajustada 16,2 pp. SHAP (final): country 1,14, agent 0,79, market_segment 0,70, lead_time 0,52, peticiones 0,44. Tres caras: country en el top-3 de los tres métodos.
### Respuesta
La cancelación está asociada sobre todo con *quién* reserva (origen, historial), *por dónde* (agencia online y agencia concreta) y *con cuánta anticipación*; las señales de compromiso la reducen. El modelo ordena el riesgo con AUC 0,898 y el decil más riesgoso cancela en 87 % de los casos.
### Recomendación · Limitaciones
Puntuar al crear la reserva; revisar las que superen 0,37 con > 30 días; empezar por Online TA y mercado doméstico; calibrar overbooking con la probabilidad; reentrenar trimestralmente. Limitaciones: asociación; umbral por validar; duplicados; booking_changes/days_in_waiting_list asumidas disponibles; drift temporal (métrica honesta 0,84).

## 12. Qué celdas tienen salida
Todas (ejecución completa con nbconvert, Python 3.15, pandas 3.0.6, scikit-learn 1.9.1, shap 0.54.1, optuna 5.0.0). Tiempo ≈ 6 min.

## 13. Cómo usar este documento en otro chat
Sube este documento (y el CSV si hace falta) y pide, por ejemplo:
- "Con base en este contexto, redacta las 6 diapositivas y el guion siguiendo la skill 06."
- "Diseña una prueba de hipótesis para la brecha Portugal vs. resto dentro de Online TA."
- "Añade al notebook una partición temporal con ventana móvil trimestral y compara con 0,837."
- "El profesor pidió X": regístralo en `registro_feedback.md` y aplica la skill 07.
