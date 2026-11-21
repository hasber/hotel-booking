# Proyecto 1: ¿podemos anticipar las cancelaciones hoteleras?

**Informe de apoyo del notebook `Notebook_Analisis_CRISP_DM_Hotel.ipynb` — V5.1 (mejoras de negocio)**  
**Datos:** `hotel_bookings.csv`, 119.390 reservas  
**Enfoque:** CRISP-DM, con código sencillo y resultados ejecutados  
**No incluye:** API ni despliegue.

## Nuestra conclusión, antes de entrar en el detalle

Queríamos saber qué características aparecen con más frecuencia entre las reservas canceladas y si un modelo podía ayudarnos a detectar reservas de riesgo. Encontramos que la **anticipación** y el **historial de cancelaciones** destacan en el análisis; también hay diferencias por tipo de hotel y segmento comercial. No estamos diciendo que esas variables *causen* la cancelación.

Probamos regresión logística, árbol de decisión y Random Forest, tres opciones presentes en el material visible de la clase 2. Elegimos **Random Forest** porque tuvo el mayor **F1 de validación (0,6111)**. Su umbral seleccionado fue **0,25**. En el conjunto final obtuvo **F1 de 0.624**, **recall de 89,25 %** y **precisión de 47,99 %**.

Eso tiene una consecuencia práctica: detectó 14.052 cancelaciones, pero generó 15.229 alertas incorrectas y marcó cerca del 71,97 % de las reservas. Además, comparamos el modelo con una regla muy sencilla: aunque la regla obtiene más exactitud, no detecta ninguna cancelación. Por eso no evaluamos solamente el porcentaje de aciertos.

También probamos qué pasaría si el hotel pudiera revisar únicamente el 10 % o el 20 % de las reservas con mayor riesgo estimado. Esta comparación nos ayuda a pensar en el trabajo que generan las alertas. **Todavía no sabemos si contactar a un cliente reduce las cancelaciones ni cuánto costaría hacerlo.**

## 1. Entendimiento del problema

Una cancelación puede obligar al hotel a ajustar ocupación y planificación. La idea del proyecto es reconocer cuáles reservas muestran más probabilidad de terminar canceladas. La columna que queremos predecir es `is_canceled`: **0 = no se canceló; 1 = se canceló**.

Se trata de **clasificación binaria**. Por eso no usamos MAE o MAPE como indicadores principales: estas métricas se explicaron en clase para problemas de regresión donde se pronostican cantidades. Para clasificación trabajamos precisión, recall, F1, matriz de confusión y ROC-AUC, vistas en las clases 2 y 5.

**Alcance:** un análisis reproducible que permita entender relaciones en los datos, comparar modelos y exponer sus limitaciones. No construimos un sistema del hotel, ni una API, ni comprobamos el efecto de contactar clientes.

## 2. Comprensión de los datos

### 2.1. ¿Con qué empezamos?

Cargamos la base directamente con `pd.read_csv("hotel_bookings.csv")` y la revisamos con `head()`, `info()`, `describe()`, `isna()` y `duplicated()`. Tenemos **119.390 registros y 32 columnas**. Encontramos **44.224 cancelaciones**, equivalentes al **37,04 %** de las reservas.

Detectamos columnas con datos faltantes y varias filas idénticas. Decidimos **no borrar automáticamente las filas idénticas** porque la base no tiene un identificador único que demuestre que son la misma reserva. Esa decisión influye en las tasas y hay que explicarla.

### 2.2. ¿Qué nos dijeron los gráficos?

- **Anticipación:** la mediana es de aproximadamente **69 días**. No todas las reservas se hacen cerca de la fecha de llegada, y las tasas de cancelación cambian entre rangos de anticipación.
- **Duración y tarifa:** predominan estadías cortas; la tarifa diaria tiene valores muy dispersos. Acotamos lo que se dibuja para leer mejor el gráfico, pero no quitamos esas observaciones de la base.
- **Cancelaciones:** no basta mirar el número total de cancelaciones. Comparamos la **tasa** de cada grupo, es decir, las canceladas divididas entre todas las reservas de ese grupo.
- **Tipo de hotel y segmento comercial:** hay grupos que se cancelan más que otros. Es una asociación observada, no una prueba de que pertenecer a un grupo provoque una cancelación.
- **Mes previsto de llegada:** las tasas varían durante el año, pero el dataset no abarca años completos de forma uniforme. No podemos llamar a eso una regla fija de temporada.
- **Correlaciones:** el mapa de calor permite ver relaciones lineales entre variables numéricas. Una correlación alta no significa causalidad ni demuestra por sí sola qué variable será mejor para predecir.

En el notebook cada gráfico va acompañado de una lectura corta. Aquí dejamos las razones y las limitaciones para la sustentación.

## 3. Preparación de los datos

### 3.1. ¿Con qué variables entrenamos?

Usamos 15 variables: anticipación, noches, número de personas, historial de reservas, tipo de hotel, mes de llegada, comida contratada, segmento, canal de distribución y tipo de habitación reservada. No incluimos `reservation_status` ni otras columnas que revelan el resultado final de la reserva.

La elección tiene un supuesto importante: para predecir al reservar, las variables deberían conocerse **en ese momento**. El dataset no permite demostrarlo para todas; es una limitación que debemos reconocer.

### 3.2. ¿Por qué hicimos tres grupos?

Separar entrenamiento, validación y prueba permite enseñar el modelo con unas reservas, tomar decisiones con otras y tener un periodo final para evaluar sin ajustar nada después de conocer el resultado.

| Grupo | Reservas | Cancelación |
|---|---:|---:|
| Entrenamiento | 49.157 | 35,77 % |
| Validación | 29.546 | 36,87 % |
| Prueba | 40.687 | 38,70 % |

Usamos como referencia la **fecha prevista de llegada**, no la fecha real de creación de la reserva. Es mejor que mezclar aleatoriamente las fechas para este ejercicio, pero **no equivale a una simulación perfecta de información histórica disponible**. Una reserva que llega en 2017 pudo haberse realizado en 2016.

### 3.3. ¿Qué transformaciones hicimos?

Usamos `fillna()` para completar faltantes numéricos con la mediana **del entrenamiento** y las categorías ausentes con `"Sin dato"`. Después, `get_dummies()` transforma categorías de texto en columnas de ceros y unos. Finalmente, la media y la desviación de entrenamiento ayudan a poner números en escalas comparables, especialmente para regresión logística.

Aplicamos esas mismas reglas a validación y prueba. Así no usamos sus estadísticas para preparar los datos de entrenamiento.

## 4. Modelado y elección

Entrenamos los **mismos tres modelos** con los mismos grupos y variables, para que la comparación sea justa. Para cada uno miramos varios valores posibles del umbral de alerta y nos quedamos con el que daba mejor **F1 sobre validación**. Luego comparamos el F1 de los tres modelos.

| Modelo | Umbral | F1 entrenamiento | F1 validación | Precisión | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.25 | 0.657 | 0.611 | 47,36 % | 86,12 % | 0.710 |
| Árbol de decisión | 0.30 | 0.644 | 0.601 | 45,40 % | 88,99 % | 0.683 |
| Regresión logística | 0.15 | 0.574 | 0.576 | 41,82 % | 92,27 % | 0.681 |

**Por qué F1:** es una forma de equilibrar dos necesidades: detectar cancelaciones reales (*recall*) sin producir demasiadas alertas equivocadas (*precisión*). No conocemos todavía los costos reales de los errores, de modo que este es un criterio técnico razonable, pero no una decisión comercial definitiva.

**Sobreajuste:** comparamos F1 de entrenamiento con F1 de validación, porque en la clase 6 se trabajó la diferencia entre aprender demasiado los ejemplos y funcionar bien con datos nuevos. En Random Forest obtuvimos **0,657 en entrenamiento y 0,611 en validación**. La diferencia merece observarse; por sí sola no demuestra un problema grave.

Seleccionamos **Random Forest**, con umbral **0,25**. No revisamos el test para cambiarlo.

## 5. Evaluación final: del resultado técnico a la decisión del hotel

Usamos el periodo de prueba una sola vez para evaluar las decisiones tomadas con entrenamiento y validación. Las mejoras de esta versión **no cambiaron el modelo, sus variables ni el umbral de 0,25**: sirven para interpretar mejor su utilidad.

### 5.1. ¿Qué tan bien quedó Random Forest?

| Métrica en prueba | Resultado |
|---|---:|
| Reservas del periodo de prueba | 40.687 |
| Cancelaciones reales | 15.745 |
| Exactitud | 58,41 % |
| Precisión | 47,99 % |
| Recall | 89,25 % |
| F1 | 0,6242 |
| ROC-AUC | 0,6991 |
| Reservas marcadas como posibles cancelaciones | 29.281 (71,97 %) |

Detectamos **14.052 cancelaciones**, pero se nos escaparon **1.693**. Además, señalamos como riesgosas **15.229 reservas que finalmente no se cancelaron**. Por eso, aunque el recall sea alto, no recomendaríamos revisar todas las alertas sin pensar antes en la capacidad del hotel.

### 5.2. ¿Era realmente necesario entrenar un modelo?

Probamos una regla básica que no usa inteligencia artificial ni variables predictoras: **predecir siempre la clase mayoritaria del entrenamiento**, que en este caso es «no se cancela».

| Métrica | Regla básica | Random Forest |
|---|---:|---:|
| Exactitud | 61,30 % | 58,41 % |
| Recall | 0,00 % | 89,25 % |
| F1 | 0,0000 | 0,6242 |

La regla básica tiene más **exactitud**, porque la mayoría de las reservas no se cancela. Pero no detecta **ninguna** cancelación. Random Forest se equivoca más en el total, aunque sí identifica gran parte de los casos que queríamos detectar.

**Nuestra interpretación:** escoger un modelo solo porque tiene más exactitud sería un error para este problema. F1 y recall nos permiten evaluar algo que la exactitud esconde: si estamos encontrando las cancelaciones.

No estamos diciendo que Random Forest sea rentable, solo que responde mejor al objetivo predictivo que esta regla elemental.

### 5.3. ¿Qué pasa si el hotel solo puede revisar el 10 % o el 20 %?

Ordenamos las reservas del periodo de prueba de mayor a menor puntuación de riesgo. Después simulamos tres alternativas de seguimiento. **No volvimos a entrenar ni a escoger umbrales con estos resultados**.

| Escenario | Reservas por revisar | Cancelaciones detectadas | Falsas alertas | % de cancelaciones detectadas | % que sí se canceló entre las revisadas |
|---|---:|---:|---:|---:|---:|
| Revisar el 10 % más riesgoso | 4.069 | 2.712 | 1.357 | 17,22 % | 66,65 % |
| Revisar el 20 % más riesgoso | 8.138 | 4.679 | 3.459 | 29,72 % | 57,50 % |
| Revisar todas las alertas del umbral 0,25 | 29.281 | 14.052 | 15.229 | 89,25 % | 47,99 % |

Para ponerlo en contexto, en todo el periodo de prueba se canceló el **38,70 %** de las reservas. En el grupo del 10 % más riesgoso, la tasa fue **66,65 %**, por encima de ese promedio. Esto sugiere que el ordenamiento del modelo sí concentra reservas más propensas a cancelarse.

Pero aparece el intercambio que tendría que evaluar el hotel:

- Si solo puede revisar el **10 %**, el volumen de trabajo es manejable, pero deja fuera aproximadamente el **82,78 %** de las cancelaciones.
- Si puede revisar el **20 %**, encuentra más casos, aunque también aumenta el número de alertas equivocadas.
- Si usa todas las alertas del **umbral 0,25**, detecta muchas cancelaciones, pero tendría que revisar casi tres cuartas partes de las reservas.

**Nuestra conclusión:** estas cifras ayudan a discutir prioridades. No bastan para decidir cuántos clientes contactar; faltan costos, capacidad de los empleados y una prueba de que el seguimiento tenga algún efecto.

### 5.4. ¿Qué características tienen las reservas que más se cancelan?

Esta es la parte que responde directamente a nuestra pregunta de negocio. Tomamos el conjunto de **entrenamiento** y revisamos cinco características. Para cada una mostramos el grupo con la tasa más alta **entre los que tienen al menos 1.000 reservas**, evitando destacar grupos mínimos que pueden presentar resultados inestables.

| Característica | Grupo destacado | Reservas del grupo | Tasa de cancelación |
|---|---|---:|---:|
| Antecedentes de cancelación | Sí | 5.809 | 96,66 % |
| Anticipación | 181–365 días | 7.930 | 64,59 % |
| Segmento comercial | Groups | 10.389 | 61,85 % |
| Tipo de hotel | City Hotel | 31.623 | 41,66 % |
| Canal de distribución | TA/TO | 39.824 | 39,63 % |

La tasa de cancelación promedio de entrenamiento fue **35,77 %**. Algunos grupos están muy por encima, sobre todo el de clientes con cancelaciones previas.

**¿Qué podría hacer el hotel con esta información?** Usarla para revisar si conviene hacer seguimiento diferenciado a ciertas reservas, por ejemplo las de mucha anticipación o con antecedentes de cancelación. Eso es una propuesta de uso, **no una acción que hayamos probado en este proyecto**.

Una misma reserva puede pertenecer simultáneamente a varios grupos. **Sus porcentajes no se suman**, y las relaciones no demuestran que el segmento o el tipo de hotel sean la causa de la cancelación. Además, el filtro de 1.000 reservas deja fuera grupos más pequeños, incluso si su tasa es muy alta; se hizo para privilegiar comparaciones con un número razonable de observaciones.

### 5.5. ¿Qué variables utilizó más Random Forest?

En el modelo final, las importancias calculadas a partir de los árboles fueron:

| Variable | Importancia dentro del modelo |
|---|---:|
| `previous_cancellations` | 0,381 |
| `lead_time` | 0,224 |
| `market_segment` | 0,147 |
| `arrival_date_month` | 0,054 |
| `hotel` | 0,051 |
| `distribution_channel` | 0,041 |
| `stays_in_week_nights` | 0,027 |

**La diferencia importante:** una tabla de tasas nos dice *en qué grupos se observó más cancelación*; la importancia del modelo nos dice *qué columnas utilizó más para clasificar*. No son dos formas de medir exactamente lo mismo, ni representan causalidad.

## 6. ¿Con qué nos quedamos?

Respondimos a las dos preguntas iniciales: encontramos perfiles con distintas tasas de cancelación y entrenamos un modelo que ayuda a ordenar reservas por riesgo. También vimos que la utilidad depende de la tarea: **tener más exactitud no garantiza detectar cancelaciones**, y **detectar casi todas obliga a generar muchas alertas**.

Lo más sensato sería que un hotel comenzara por definir a cuántas reservas realmente puede hacer seguimiento, cuánto cuestan las alertas equivocadas y qué se espera lograr con ese seguimiento. Sin esos datos, una recomendación comercial definitiva sería inventar una certeza que todavía no tenemos.

Hay límites que no debemos pasar por alto: usamos la **fecha prevista de llegada** para separar los periodos (no la fecha de creación), el archivo no demuestra que todas las variables estuvieran disponibles desde el momento de reservar, y los registros idénticos podrían corresponder a reservas distintas. **No desarrollamos una API ni medimos el impacto de acciones sobre clientes.**

## Relación del proyecto con las siete clases

La siguiente matriz se basa en lo **visible en las grabaciones**. Los minutos son aproximados y corresponden a muestras revisadas de cada video. Como no hay transcripción completa del audio, no afirmamos haber verificado cada explicación verbal del profesor.

| Tema o técnica | Evidencia visible de clase | Estado | Aplicación / decisión |
|---|---|---|---|
| GitHub y entorno de trabajo | Clase 1, aprox. 00:41–01:29 | Visto | Documentación y repositorio |
| CRISP-DM, fases y modelado | Clase 2, aprox. 01:29 y 02:29–03:05 | Visto | Estructura de seis etapas |
| Regresión logística, árbol y Random Forest; F1 | Clase 2, aprox. 02:53, diapositiva de modelado | Visto como ejemplo de clase | Comparación de tres modelos con F1 |
| Manipulación básica de datos con pandas y Colab | Clase 3, aprox. 00:17–01:29, 02:05–03:17 | Visto en pantalla | Selección y revisión de columnas |
| Tablas, correlaciones, distribución y gráficos | Clase 4, aprox. 02:29–03:17 | Visto | EDA e interpretaciones |
| MAE, RMSE, MAPE | Clase 5, aprox. 00:53–01:17 | Visto para regresión | Se aclara por qué no son centrales para clasificación |
| Matriz de confusión, precisión, recall, F1 y ROC | Clase 5, aprox. 01:53–02:53 | Visto | Evaluación de cancelaciones |
| Sobreajuste y subajuste | Clase 6, aprox. 00:05–01:29 | Visto | Comparar F1 entrenamiento y validación |
| Separación entrenamiento/prueba en ejercicios | Clase 6, aprox. 02:17–02:41 | Visto | Se adapta a particiones temporales propias |
| FastAPI y endpoints | Clase 7, aprox. 00:29–01:41 | Visto | **Fuera del alcance por decisión del equipo** |
| `get_dummies()`, partición por fecha prevista, búsqueda de umbral, importancia del árbol | Su uso exacto no está verificado en las muestras de grabación | No evidenciado completamente | Se mantienen como decisiones del proyecto; no se presentan como recetas dictadas en clase |

**Importante:** «no evidenciado» no quiere decir «prohibido». Para acreditar al detalle todas las funciones harían falta los notebooks originales del profesor o una transcripción completa de los videos.

## Preguntas que conviene poder responder en la sustentación

**¿Por qué es clasificación?** Porque queremos decidir entre dos resultados: se cancela o no se cancela.

**¿Por qué F1 y no MAPE?** Porque estamos clasificando, y F1 combina precisión y recall. MAPE se utiliza para medir error porcentual en predicciones numéricas.

**¿Por qué usamos tres conjuntos?** Uno para aprender, otro para escoger y uno final que no usamos en las decisiones anteriores.

**¿Qué descubrimos sobre las variables?** Las cancelaciones previas y la anticipación destacan en el modelo. Las tasas descriptivas muestran qué perfiles tienen mayor proporción de cancelaciones.

**¿Por qué no debemos decir que el modelo es perfecto?** Porque produce numerosas falsas alertas y aún no demostramos su impacto en el hotel.

## Archivos del repositorio

- `Notebook_Analisis_CRISP_DM_Hotel.ipynb`: análisis, gráficos, comentarios cortos y resultados ejecutados.
- `hotel_bookings.csv`: archivo original que utiliza el notebook (ya está en el repositorio).
- `resultados_crispdm/`: tablas de comparación, evaluación, grupos y variables.
- `docs/Documentación_notebook_CRISP_DM_Hotel.md`: esta explicación ampliada.
- `README.md`: puerta de entrada del repositorio, que debe mantener los aportes del equipo.

*Las cifras provienen de la ejecución del notebook V5.1 con `hotel_bookings.csv`. Los nuevos cuadros son análisis de la prueba y del entrenamiento, no una nueva elección del modelo. Si cambiamos el código o el archivo de datos, tenemos que volver a ejecutar y actualizar estas cifras.*
