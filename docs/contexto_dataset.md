# Contexto del dataset — Hotel Booking Demand (`hotel_bookings.csv`)

Fecha: 2026-11-07 · Autor: Hasber Malpica (con Claude Code) · Fuente del dato: `hotel_bookings.csv` (Antonio, Almeida & Nunes, 2019, *Hotel booking demand datasets*, dos hoteles de Portugal) · Periodo cubierto: llegadas del 2015-07-01 al 2017-08-31.

## 1. Archivo y grano
- Archivo: `hotel_bookings.csv` · 119.390 filas × 32 columnas.
- **Una fila representa:** una reserva de hotel (una habitación), cancelada o no, con llegada prevista entre julio de 2015 y agosto de 2017, en uno de dos hoteles (City Hotel: 79.330 filas; Resort Hotel: 40.060).
- Identificadores: **no hay identificador de reserva ni de cliente.** `agent` y `company` son códigos de intermediario/empresa, no de reserva.
- Duplicados exactos: **31.994** (26,8 % de las filas; quedan 87.396 únicas). Se concentran en `market_segment` Groups (14.869) y Offline TA/TO (10.330), en City Hotel (25.902) y en `deposit_type` Non Refund (13.549 de las 14.587 filas Non Refund). La tasa de cancelación de las filas duplicadas es 58,4 % frente a 26,2 % en las no duplicadas.
  - Lectura: parecen **bloques de reservas** (un operador reserva n habitaciones idénticas el mismo día). Sin identificador de reserva no es posible distinguir un bloque legítimo de un registro repetido.
  - Decisión (limpieza): **eliminar duplicados exactos** → 87.396 filas. Motivos: (a) sin ID no hay forma de validar que sean reservas distintas; (b) conservarlos pondría filas idénticas en train y test (fuga por duplicados, causa 3 del checklist de la skill 05) y el modelo mediría memorización. Costo: los bloques de grupos quedan subrepresentados y la prevalencia baja de 37,0 % a 27,5 %. Se documenta como limitación y en la etapa de verificación se cuantifica el efecto (modelo con duplicados y split aleatorio vs. deduplicado).

## 2. Diccionario de datos
Descripciones tomadas del artículo original (Antonio et al., 2019); las marcadas ⚠️ son inferidas.

| Columna | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| hotel | categórica nominal | Resort Hotel o City Hotel | City Hotel |
| is_canceled | binaria (**target**) | 1 si la reserva fue cancelada o no-show, 0 si se completó (check-out) | 0 |
| lead_time | numérica discreta | Días entre la fecha de la reserva y la de llegada | 342 |
| arrival_date_year / _month / _week_number / _day_of_month | temporal | Fecha de llegada descompuesta | 2015 / July / 27 / 1 |
| stays_in_weekend_nights / stays_in_week_nights | numérica discreta | Noches de fin de semana / entre semana reservadas | 0 / 2 |
| adults / children / babies | numérica discreta | Composición de los huéspedes | 2 / 0 / 0 |
| meal | categórica nominal | Régimen de comidas: BB, HB, FB, SC/Undefined (sin paquete) | BB |
| country | categórica nominal (177) | País de origen (ISO 3166-3) | PRT |
| market_segment | categórica nominal (8) | Segmento comercial: Online TA, Offline TA/TO, Groups, Direct, Corporate, Complementary, Aviation, Undefined | Online TA |
| distribution_channel | categórica nominal (5) | Canal de distribución: TA/TO, Direct, Corporate, GDS, Undefined | TA/TO |
| is_repeated_guest | binaria | 1 si el cliente ya se había hospedado | 0 |
| previous_cancellations / previous_bookings_not_canceled | numérica discreta | Historial del cliente antes de esta reserva | 0 / 0 |
| reserved_room_type | categórica nominal (10) | Tipo de habitación reservada (código anonimizado) | A |
| assigned_room_type | categórica nominal (12) | Tipo de habitación asignada (puede diferir por overbooking o petición) — **post-evento** | A |
| booking_changes | numérica discreta | Número de modificaciones a la reserva hasta el check-in o la cancelación ⚠️ parcialmente post-evento | 0 |
| deposit_type | categórica nominal (3) | No Deposit, Non Refund (depósito por el total), Refundable | No Deposit |
| agent | categórica nominal (333) | ID de la agencia de viajes que hizo la reserva; nulo = sin agencia | 240 |
| company | categórica nominal (352) | ID de la empresa que paga; nulo = sin empresa | — |
| days_in_waiting_list | numérica discreta | Días en lista de espera antes de confirmarse | 0 |
| customer_type | categórica nominal (4) | Transient, Transient-Party, Contract, Group | Transient |
| adr | numérica continua | Average Daily Rate: ingreso por noche de habitación (€) | 94,58 |
| required_car_parking_spaces | numérica discreta | Plazas de parqueo solicitadas | 0 |
| total_of_special_requests | numérica discreta | Peticiones especiales (cama doble, piso alto…) | 1 |
| reservation_status | categórica (3) | Canceled / Check-Out / No-Show — **es el target** | Check-Out |
| reservation_status_date | fecha | Fecha del último estado — **posterior al evento** | 2015-07-01 |

## 3. Pregunta de negocio y target
- **Pregunta:** *¿Qué características de la reserva (hotel, estancia, anticipación), del cliente (composición, historial, origen), del canal (segmento, distribución, agencia) y de las condiciones (depósito, tarifa, peticiones) están asociadas con la cancelación de una reserva y cuáles tienen mayor capacidad de predecirla?*
- **Target:** `is_canceled` (1 = cancelada o no-show) · tipo: binario → **clasificación**. Prevalencia: 37,0 % en el crudo, **27,5 % tras deduplicar**.
- Por qué este target y no otro: la decisión operativa del hotel es binaria (anticipar una cancelación para hacer overbooking controlado, pedir depósito o contactar al cliente). No existe una magnitud continua equivalente en el dataset; `adr` mide precio, no resultado.
- Tres métricas para el EDA (análogas a volumen / resultado absoluto / eficiencia): **Reservas** (n), **Canceladas** (n) y **Tasa de cancelación** (canceladas/reservas, ponderada por grupo, nunca promedio de tasas). Como métrica de impacto: **ingreso en riesgo** = Σ `adr` × noches de las reservas canceladas.
- Target derivado para decisión operativa: la probabilidad predicha con un umbral de negocio (máximo F1 / recall ≥ 80 % / 0,5) ⚠️ por validar con el área de revenue.

## 4. Clasificación de variables y dimensiones de negocio

| Columna | Tipo estadístico | Rol | Dimensión |
|---|---|---|---|
| hotel | nominal | predictor | Hotel y estancia |
| stays_in_weekend_nights, stays_in_week_nights (+ `total_nights`) | discreta | predictor | Hotel y estancia |
| reserved_room_type, meal | nominal | predictor | Hotel y estancia |
| adults, children, babies (+ `total_guests`) | discreta | predictor | Cliente |
| is_repeated_guest, previous_cancellations, previous_bookings_not_canceled | binaria / discreta | predictor | Cliente |
| customer_type, country | nominal | predictor | Cliente |
| market_segment, distribution_channel, agent, company | nominal | predictor (agent/company como categoría con mínimo de frecuencia o bandera) | Canal |
| lead_time, days_in_waiting_list | discreta | predictor | Tiempo y anticipación |
| arrival_date_year, arrival_date_month, arrival_date_week_number, arrival_date_day_of_month (+ `arrival_date`, `arrival_quarter`, `arrival_dow`) | temporal | predictor (mes/trimestre como categóricos) | Tiempo y anticipación |
| deposit_type, adr | nominal / continua | predictor | Condiciones de la reserva |
| booking_changes, total_of_special_requests | discreta | predictor (señal de compromiso del cliente) ⚠️ | Condiciones de la reserva |
| required_car_parking_spaces | discreta | **prohibido (post-evento)**: 0 cancelaciones en 7.306 reservas con plaza → se registra al check-in | Condiciones de la reserva |
| is_canceled | binaria | **target** | Resultado |
| reservation_status, reservation_status_date | nominal / fecha | **prohibido (fuga)** | Resultado |
| assigned_room_type | nominal | **prohibido (post-evento)** | Resultado |

Dimensiones definidas (5): **Hotel y estancia**, **Cliente**, **Canal**, **Tiempo y anticipación**, **Condiciones de la reserva**.

## 5. Variables prohibidas como predictores
```python
leakage_columns    = ["reservation_status", "reservation_status_date"]  # el target se deriva de reservation_status
post_event_columns = ["assigned_room_type", "room_changed", "required_car_parking_spaces"]
# assigned_room_type se asigna al check-in: las reservas que cambiaron de habitación se cancelan 4,7 % vs 31,5 % (dedup; 5,4 % vs 41,6 % en crudo).
# required_car_parking_spaces: 0 cancelaciones en 7.306 reservas con plaza; incluirla sube el AUC +0,024 pero es dato de llegada.
id_columns         = []                        # no hay IDs; agent/company son categorías de negocio, no identificadores de fila
redundant_columns  = ["arrival_date_week_number", "arrival_date_day_of_month"]  # redundantes con mes/trimestre; se conservan mes y año
```
Nota sobre `booking_changes`: se acumula hasta el check-in o la cancelación, así que una parte se conoce después del momento de reserva. Se conserva como predictor (señal de compromiso), pero se vigila en la verificación de fuga (AUC univariado y ablación); si el modelo se usara en el momento de crear la reserva habría que excluirla.

## 6. Calidad del dato (cifras del crudo)
- Nulos: `company` 94,3 % (esperable: reserva sin empresa), `agent` 13,7 % (esperable: sin intermediario), `country` 0,41 % (488, error/desconocido), `children` 4 (error).
- Variantes de escritura: ninguna. Sí hay categoría `Undefined` en `meal` (1.169; equivale a SC según el diccionario), `market_segment` (2) y `distribution_channel` (5).
- Valores sospechosos: `adr` = −6,38 (1 fila) y `adr` = 5.400 (1 fila; el siguiente máximo es 510) → errores; `adr` = 0 en 1.959 filas (680 son Complementary: válido, estancia sin cobro); reservas con **0 huéspedes** (adults+children+babies = 0): 180 → inválidas; estancias de 0 noches: 715 (válidas: uso de día; su tasa de cancelación es 4,9 %); `adults` hasta 55 y `babies` hasta 10 en filas aisladas (se conservan).
- Validación de fórmulas: `is_canceled` = 1 ⇔ `reservation_status` ∈ {Canceled (43.017), No-Show (1.207)}; 0 ⇔ Check-Out (75.166). Coincide en el 100 % de las filas → confirma que `reservation_status` es fuga total.
- Rango temporal: llegadas 2015-07-01 → 2017-08-31 (2015: 21.996; 2016: 56.707; 2017: 40.687). 2015 y 2017 son años incompletos (julio–diciembre y enero–agosto): comparar por mes dentro de cada año, no totales anuales.
- Correlaciones ≥ 0,9 entre numéricas: ninguna. Correlaciones más altas con el target (dedup): `lead_time` +0,185, `adr` +0,128, `required_car_parking_spaces` −0,184, `total_of_special_requests` −0,121.
- Alta cardinalidad: `country` 177 valores (139 con < 100 filas), `agent` 333, `company` 352 → One-Hot con `min_frequency` o bandera binaria.

## 7. Decisiones pendientes para la limpieza
- [x] Eliminar duplicados exactos (119.390 → 87.396) y documentar la limitación.
- [ ] Eliminar las reservas con 0 huéspedes (180 en crudo; recontar tras deduplicar).
- [ ] `adr` −6,38 y 5.400 → NaN (se conservan como nulo; el pipeline de modelado imputa con mediana de train).
- [ ] `children` nulo → 0 (moda); `country` nulo → "Unknown"; `agent`/`company` nulo → "None" (constante con significado) + banderas `has_agent`, `has_company`.
- [ ] `meal` Undefined → SC (diccionario oficial). `market_segment`/`distribution_channel` Undefined se conservan (7 filas).
- [ ] Construir `arrival_date` (fecha), `arrival_month_num`, `arrival_quarter`, `arrival_dow`, `total_nights`, `total_guests`.
- [ ] `reservation_status_date` a fecha solo para la verificación de fuga; no entra al modelo.

## 8. Supuestos a confirmar con el dueño del dato
- ⚠️ Que las filas idénticas son bloques de reservas legítimos y no registros repetidos. Si fueran legítimas, el análisis subestima el peso de Groups/Offline TA con Non Refund.
- ⚠️ Que `booking_changes` y `days_in_waiting_list` están disponibles en el momento en que el hotel usaría la predicción.
- ⚠️ Que `adr` = 0 corresponde a estancias de cortesía y no a tarifas no registradas.
- ⚠️ Que `Undefined` en `meal` equivale a SC (lo dice el artículo original, no el archivo).
