# Perfil de `hotel_bookings.csv`

- Dimensiones: **119,390 filas × 32 columnas**
- Duplicados exactos: **31,994**

## Columnas

| Columna | dtype | Nulos | % nulos | Únicos | Candidato ID | Ejemplo |
|---|---|---|---|---|---|---|
| hotel | str | 0 | 0.00 | 2 |  | Resort Hotel |
| is_canceled | int64 | 0 | 0.00 | 2 |  | 0 |
| lead_time | int64 | 0 | 0.00 | 479 |  | 342 |
| arrival_date_year | int64 | 0 | 0.00 | 3 |  | 2015 |
| arrival_date_month | str | 0 | 0.00 | 12 |  | July |
| arrival_date_week_number | int64 | 0 | 0.00 | 53 |  | 27 |
| arrival_date_day_of_month | int64 | 0 | 0.00 | 31 |  | 1 |
| stays_in_weekend_nights | int64 | 0 | 0.00 | 17 |  | 0 |
| stays_in_week_nights | int64 | 0 | 0.00 | 35 |  | 0 |
| adults | int64 | 0 | 0.00 | 14 |  | 2 |
| children | float64 | 4 | 0.00 | 5 |  | 0.0 |
| babies | int64 | 0 | 0.00 | 5 |  | 0 |
| meal | str | 0 | 0.00 | 5 |  | BB |
| country | str | 488 | 0.41 | 177 |  | PRT |
| market_segment | str | 0 | 0.00 | 8 |  | Direct |
| distribution_channel | str | 0 | 0.00 | 5 |  | Direct |
| is_repeated_guest | int64 | 0 | 0.00 | 2 |  | 0 |
| previous_cancellations | int64 | 0 | 0.00 | 15 |  | 0 |
| previous_bookings_not_canceled | int64 | 0 | 0.00 | 73 |  | 0 |
| reserved_room_type | str | 0 | 0.00 | 10 |  | C |
| assigned_room_type | str | 0 | 0.00 | 12 |  | C |
| booking_changes | int64 | 0 | 0.00 | 21 |  | 3 |
| deposit_type | str | 0 | 0.00 | 3 |  | No Deposit |
| agent | float64 | 16,340 | 13.69 | 333 |  | 304.0 |
| company | float64 | 112,593 | 94.31 | 352 |  | 110.0 |
| days_in_waiting_list | int64 | 0 | 0.00 | 128 |  | 0 |
| customer_type | str | 0 | 0.00 | 4 |  | Transient |
| adr | float64 | 0 | 0.00 | 8,879 |  | 0.0 |
| required_car_parking_spaces | int64 | 0 | 0.00 | 5 |  | 0 |
| total_of_special_requests | int64 | 0 | 0.00 | 6 |  | 0 |
| reservation_status | str | 0 | 0.00 | 3 |  | Check-Out |
| reservation_status_date | str | 0 | 0.00 | 926 |  | 2015-07-01 |

## Columnas que parecen fechas

- `arrival_date_month`: 0001-01-01 00:00:00 → 0001-12-01 00:00:00 (0 no convertibles)
- `reservation_status_date`: 2014-10-17 00:00:00 → 2017-09-14 00:00:00 (0 no convertibles)

## Categóricas con variantes de escritura (unificar en limpieza)

- (ninguna detectada)

## Categóricas (≤ max-cat valores): frecuencias

- `hotel` (2 valores): City Hotel=79,330, Resort Hotel=40,060
- `arrival_date_month` (12 valores): August=13,877, July=12,661, May=11,791, October=11,160, April=11,089, June=10,939, September=10,508, March=9,794, February=8,068, November=6,794
- `meal` (5 valores): BB=92,310, HB=14,463, SC=10,650, Undefined=1,169, FB=798
- `market_segment` (8 valores): Online TA=56,477, Offline TA/TO=24,219, Groups=19,811, Direct=12,606, Corporate=5,295, Complementary=743, Aviation=237, Undefined=2
- `distribution_channel` (5 valores): TA/TO=97,870, Direct=14,645, Corporate=6,677, GDS=193, Undefined=5
- `reserved_room_type` (10 valores): A=85,994, D=19,201, E=6,535, F=2,897, G=2,094, B=1,118, C=932, H=601, P=12, L=6
- `assigned_room_type` (12 valores): A=74,053, D=25,322, E=7,806, F=3,751, G=2,553, C=2,375, B=2,163, H=712, I=363, K=279
- `deposit_type` (3 valores): No Deposit=104,641, Non Refund=14,587, Refundable=162
- `customer_type` (4 valores): Transient=89,613, Transient-Party=25,124, Contract=4,076, Group=577
- `reservation_status` (3 valores): Check-Out=75,166, Canceled=43,017, No-Show=1,207

## Numéricas: resumen y valores sospechosos

| Columna | min | p1 | mediana | media | p99 | max | ≤0 | negativos | asimetría |
|---|---|---|---|---|---|---|---|---|---|
| is_canceled | 0 | 0 | 0 | 0.3704 | 1 | 1 | 75,166 | 0 | 0.54 |
| lead_time | 0 | 0 | 69 | 104 | 444 | 737 | 6,345 | 0 | 1.35 |
| arrival_date_year | 2015 | 2015 | 2016 | 2016 | 2017 | 2017 | 0 | 0 | -0.23 |
| arrival_date_week_number | 1 | 2 | 28 | 27.17 | 53 | 53 | 0 | 0 | -0.01 |
| arrival_date_day_of_month | 1 | 1 | 16 | 15.8 | 31 | 31 | 0 | 0 | -0.00 |
| stays_in_weekend_nights | 0 | 0 | 1 | 0.9276 | 4 | 19 | 51,998 | 0 | 1.38 |
| stays_in_week_nights | 0 | 0 | 2 | 2.5 | 10 | 50 | 7,645 | 0 | 2.86 |
| adults | 0 | 1 | 2 | 1.856 | 3 | 55 | 403 | 0 | 18.32 |
| children | 0 | 0 | 0 | 0.1039 | 2 | 10 | 110,796 | 0 | 4.11 |
| babies | 0 | 0 | 0 | 0.007949 | 0 | 10 | 118,473 | 0 | 24.65 |
| is_repeated_guest | 0 | 0 | 0 | 0.03191 | 1 | 1 | 115,580 | 0 | 5.33 |
| previous_cancellations | 0 | 0 | 0 | 0.08712 | 1 | 26 | 112,906 | 0 | 24.46 |
| previous_bookings_not_canceled | 0 | 0 | 0 | 0.1371 | 3 | 72 | 115,770 | 0 | 23.54 |
| booking_changes | 0 | 0 | 0 | 0.2211 | 3 | 21 | 101,314 | 0 | 6.00 |
| agent | 1 | 1 | 14 | 86.69 | 390 | 535 | 0 | 0 | 1.09 |
| company | 6 | 16 | 179 | 189.3 | 513 | 543 | 0 | 0 | 0.60 |
| days_in_waiting_list | 0 | 0 | 0 | 2.321 | 75 | 391 | 115,692 | 0 | 11.94 |
| adr | -6.38 | 0 | 94.58 | 101.8 | 252 | 5400 | 1,960 | 1 | 10.53 |
| required_car_parking_spaces | 0 | 0 | 0 | 0.06252 | 1 | 8 | 111,974 | 0 | 4.16 |
| total_of_special_requests | 0 | 0 | 0 | 0.5714 | 3 | 5 | 70,318 | 0 | 1.35 |

Revisa manualmente: edades fuera de 18–100, cantidades ≤ 0, porcentajes fuera de 0–100, negativos que no caben, y columnas que deberían ser fórmula de otras (validar con tolerancia).

## Correlaciones |r| ≥ 0,90 (anotar; no eliminar en EDA)

- (ninguna)
