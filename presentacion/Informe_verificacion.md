# Informe breve de verificación — Presentación Proyecto 1 (CRISP-DM Hotel)

**Archivos entregados (carpeta `presentacion/`)**
- `Presentacion_CRISP_DM_Hotel.pptx` — 16 diapositivas, 16:9, editable (texto, tarjetas, gráficos nativos de PowerPoint y tablas). Notas del presentador en cada diapositiva, con expositor y tiempo al inicio.
- `Presentacion_CRISP_DM_Hotel.pdf` — solo para revisión visual (exportado desde PowerPoint).
- Este informe.

**Secuencia, tiempos y responsables:** se respetaron exactamente los 16 títulos, el orden y los tiempos del guion. Diapositivas 1–5 Hasber Gonzalo Malpica (6:40), 6–10 Carlos Andrés Restrepo (6:40), 11–16 Juan Carlos Monterrosa (6:40). El expositor y el tiempo están en las notas, no en pantalla.

## Fuentes utilizadas
1. `Notebook_Analisis_CRISP_DM_Hotel.ipynb` (V5.1): salidas ejecutadas de todas las celdas.
2. `docs/Documentación_notebook_CRISP_DM_Hotel.md`: tablas de validación, prueba, escenarios y perfiles.
3. `hotel_bookings.csv`: para reconstruir las series completas de los gráficos (tasas por rango de anticipación, por segmento, por hotel, por cancelaciones previas, matriz de correlaciones y nulos por columna) ejecutando el mismo código del notebook.

Además, se re-ejecutó el código del notebook (semilla 42) en un entorno aparte, sin escribir en el repositorio, para confirmar los valores con cuatro decimales que el notebook muestra truncados en pantalla (F1 de validación de los tres modelos, F1/recall/precisión/ROC-AUC de prueba, matriz de confusión, escenarios de seguimiento e importancias). Todos coincidieron con la documentación y con el guion.

## Cifras verificadas (coinciden con las fuentes)
- 119.390 reservas, 32 columnas, 31.994 filas idénticas, 44.224 canceladas (37,04 %). Nulos: company 112.593, agent 16.340, country 488, children 4.
- Rangos de anticipación: 0–7 d 9,6 % (19.746); 8–30 d 27,9 % (18.960); 31–90 d 37,7 % (29.553); 91–180 d 44,7 % (26.439); 181–365 d 55,5 % (21.544); 366+ d 67,7 % (3.148).
- Cancelaciones previas Sí 91,6 % (6.484); Groups 61,1 % (19.811, entre segmentos ≥ 100 reservas); City Hotel 41,7 % (79.330). Correlación lead_time–is_canceled 0,293.
- 15 variables → 51 columnas; particiones 49.157 / 29.546 / 40.687 por fecha prevista de llegada.
- F1 validación: Random Forest 0,6111, árbol 0,6013, logística 0,5755; umbral 0,25.
- Prueba: F1 0,6242, recall 89,25 %, precisión 47,99 %, ROC-AUC 0,6991, exactitud 58,41 %; VP 14.052, FN 1.693, FP 15.229, VN 9.713; 29.281 alertas (71,97 %).
- Regla básica: exactitud 61,30 %, recall 0 %.
- Importancias: previous_cancellations 0,381; lead_time 0,224; market_segment 0,147 (y las siguientes cuatro del notebook).
- Perfiles en entrenamiento: 96,66 % (5.809); 64,59 % (7.930); 61,85 % (10.389).
- Escenarios: 4.069 / 2.712 / 1.357; 8.138 / 4.679 / 3.459; 29.281 / 14.052 / 15.229.

## Cifras o elementos que no se pudieron validar u observaciones
- **Ninguna cifra del guion quedó sin validar.**
- La carpeta `resultados_crispdm/` que menciona la documentación **no existe en el repositorio** (el notebook la genera al ejecutarse). No se usó la carpeta `resultados_integracion/`, que corresponde a la versión V2 (umbral 0,225, F1 de prueba 0,6220) y no coincide con la V5.1.
- Dos cifras del guion se mostraron con los decimales que entrega el notebook: hotel City 41,7 % y Groups 61,1 % (el guion las cita igual).
- No se incluyó ningún dato externo, API, ahorro ni efecto de contactar clientes.

## Notas de diseño
- Paleta azul marino / turquesa / blanco / grises; Calibri; una idea por diapositiva; títulos que comunican el hallazgo.
- Gráficos nativos editables en las diapositivas 5, 6, 7, 11, 13, 14 y 15; mapa de correlaciones como tabla coloreada editable (diapositiva 8); matriz de confusión con formas editables (diapositiva 12). Cada gráfico tiene título, unidad, interpretación de una frase y la fuente en el pie.
- Palabras visibles por diapositiva (incluyendo título y etiquetas de tarjetas): la mayoría entre 45 y 80; las diapositivas 10, 12 y 16 superan ese rango por las etiquetas de la línea de tiempo, de la matriz y del cierre.
- Validación del archivo (esquema OOXML y gráficos) sin errores; renderizado completo en PowerPoint revisado diapositiva por diapositiva.
