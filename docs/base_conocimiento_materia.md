# Base de conocimiento de la materia → dónde se aplica en el proyecto

Resumen del material del curso (sesiones 2 a 6: CRISP-DM, preparación de datos, ejemplo de preparación, modelo, pipelines; no se incluye en el repositorio por ser material del profesor) y, para cada concepto, la sección de `Trabajo.ipynb` donde se aplica. Sirve como checklist frente a la aclaración del profesor (entendimiento del problema → exploración → preparación → solución final) y para la sustentación ("apliquen lo visto en clase").

## 1. Qué enseña cada sesión

| Sesión | Contenido | Dataset / tipo |
|---|---|---|
| 2 — CRISP-DM (pdf) | Metodología y sus 6 fases; traducir objetivo de negocio → objetivo de ciencia de datos → criterio de éxito; comprensión de datos (histogramas, boxplots, correlaciones, nulos, cardinalidad, outliers); preparación (imputación, codificación, escalado, feature engineering, train/test); modelado (algoritmo según tipo de problema, CV k=5, comparar logística/árbol/RF); evaluación (accuracy, recall, precision, F1, AUC, matriz de confusión, ROC, lift chart); comunicación (storytelling, público de negocio vs. técnico); despliegue (tiempo real vs. lotes) y mantenimiento (data drift, reentrenamiento); KPIs; entregables por fase | Ejemplos narrados: churn, recompra (clasificación) |
| 3 — Preparación de datos | Limpieza / integración / transformación: `SimpleImputer`, `fillna` (media, moda, constante con significado, media por grupo), `drop_duplicates`, `pd.to_datetime(errors="coerce")`, `str.lower().str.contains`, `pd.merge`, z-score / min-max, `StandardScaler`, `LabelEncoder`, `OneHotEncoder`, máscaras booleanas y `df.loc` | DataFrames de juguete |
| 4 — Ejemplo de preparación | Diccionario de datos, `df.info()`, `isna().sum()`, histograma antes de imputar, imputación con constante con significado (0 / "No Alley" / "None"), consistencia cruzada entre columnas y eliminación de filas inconsistentes | House Prices (regresión) |
| 5 — Modelo | Supervisado vs. no supervisado; métricas de regresión (MAE, MSE, RMSE, R², MAPE) y de clasificación (matriz de confusión, accuracy, precision, recall, F1, ROC/AUC, `classification_report`); clustering (K-Means, codo, silhouette); under/overfitting, bias-variance, regularización | California Housing, MNIST, blobs |
| 6 — Pipelines | `Pipeline` (scaler → PCA → clasificador), `train_test_split`, `cross_validate`, comparación de clasificadores con `set_params` (logística, SVC, KNN, árbol, RF, GradientBoosting), `GridSearchCV` (`best_params_`, `best_score_`), evaluación única en test | Iris |

## 2. Concepto de la materia → dónde se aplica en el proyecto

| # | Concepto | Dónde se aplica (sección de `Trabajo.ipynb`) | Estado |
|---|---|---|---|
| 1 | CRISP-DM: comprensión del negocio (problema, tipo, alcance, entregables, criterio de éxito) | 0.1–0.6 | ✅ |
| 2 | Objetivo de negocio → objetivo de ciencia de datos → métrica de éxito | 0.6 (tabla de traducción y KPIs) | ✅ |
| 3 | KPIs y preguntas de validación de la métrica | 0.6 (tasa de cancelación, ingreso en riesgo, recall/precision al umbral, lift del decil 1) | ✅ |
| 4 | Comprensión de los datos: `df.info()`, `describe`, nulos, cardinalidad, outliers | 1.3–1.7 | ✅ |
| 5 | Diccionario de datos | `docs/contexto_dataset.md` §2; 1.3 | ✅ |
| 6 | Histogramas, correlaciones, boxplots | 3.2–3.3, 4.2–4.3 (histogramas con p95), heatmap y barras de correlación | ✅ (boxplots sustituidos por tablas por rangos) |
| 7 | Hipótesis iniciales | 3.3 (anticipación como hipótesis) y 4.10 (verificación) | ✅ |
| 8 | Duplicados (`drop_duplicates`) | 1.2 y 2.0, con análisis de dónde se concentran | ✅ |
| 9 | Imputación: media/moda, constante con significado, media por grupo | 2.2 (moda en children; "Unknown", "SinAgencia", "SinEmpresa"); mediana dentro del pipeline (6.5 etapa 1) | ✅ |
| 10 | Consistencia cruzada entre columnas y filas inválidas | 1.7–1.8 (0 huéspedes, adr, target vs. status) y 2.3 | ✅ |
| 11 | Formatos de fecha y texto (`to_datetime`, `str.strip`, reemplazos explícitos) | 2.1, 2.4 | ✅ |
| 12 | Integración con `pd.merge` | No aplica: una sola tabla | — |
| 13 | Codificación `OneHotEncoder` / `LabelEncoder` / ordinal | 6.5 etapa 2 (One-Hot con `min_frequency`; contraste "todo ordinal") | ✅ |
| 14 | Escalado z-score / min-max (`StandardScaler`, `MinMaxScaler`) | 6.5 etapa 3 (Standard vs. MinMax vs. Robust vs. ninguno) | ✅ |
| 15 | Feature engineering y selección de features | 3.1 (variables derivadas), 6.3 (exclusiones justificadas), 6.10.4 (subconjuntos) | ✅ |
| 16 | Train/test con `train_test_split` | 6.4 (75/25 estratificado) | ✅ |
| 17 | `Pipeline` + `ColumnTransformer` | 6.5 (todo lo que aprende de datos vive en el pipeline) | ✅ |
| 18 | Validación cruzada (`cross_validate`, k = 5) | 6.5–6.6 (`StratifiedKFold(5)`) | ✅ |
| 19 | Comparar clasificadores: logística, KNN, árbol, RF, Gradient Boosting | 6.6 (más Dummy como piso) | ✅ (SVC omitido por costo en 65k filas) |
| 20 | Búsqueda de hiperparámetros (`GridSearchCV`) | 7 Paso 4 con Optuna (misma idea, búsqueda bayesiana) | ✅ equivalente |
| 21 | Evaluación única en test | 6.7 (base) y 7 Paso 5 (final) | ✅ |
| 22 | Matriz de confusión, accuracy, precision, recall, F1 | 6.7–6.8 | ✅ |
| 23 | F1 depende del umbral; accuracy engañosa con desbalance | 6.8 (barrido de umbrales, Dummy) | ✅ |
| 24 | Curva ROC y AUC; `classification_report` | 6.7 (ROC y PR) | ✅ |
| 25 | Lift chart | 6.8.1 y 7 Paso 5 (deciles de riesgo con lift) | ✅ |
| 26 | Under/overfitting, bias-variance, regularización | 6.6 (árbol limitado a profundidad 6), 7 Paso 4 (`l2_regularization`, `min_samples_leaf`) | ✅ parcial |
| 27 | Importancia de variables (gráficos) | 6.10, 7 Paso 2 (tres métodos) y Paso 6 (SHAP) | ✅ |
| 28 | Despliegue: tiempo real vs. lotes | 0.3 y `Despliegue_FastAPI.ipynb` (pendiente) | ⏳ |
| 29 | Mantenimiento: data drift y reentrenamiento | 7.4 (PSI, adversarial, partición temporal) | ✅ |
| 30 | Comunicación: storytelling, visuales, público de negocio | Presentación de 20 min (pendiente, skill 06); interpretaciones en cada sección | ⏳ |
| 31 | Métricas de regresión (MAE, RMSE, R², MAPE) | No aplica: target binario | — |
| 32 | Clustering, PCA, CNN | No aplica al problema | — |

## 3. Reglas del material que el proyecto respeta
- "Garbage in, garbage out" → perfilado y limpieza documentados antes de cualquier modelo.
- "Accuracy puede ser engañosa con data desbalanceada" → se reportan ROC-AUC, PR-AUC, F1 al umbral y el Dummy.
- "Un buen modelo debe funcionar sobre data que no ha visto" → CV sobre train, test una vez, y además partición temporal.
- "Verificar si el modelo cumple los objetivos de negocio, no solo métricas técnicas" → deciles de riesgo y umbral ligados al uso operativo.
- "A negocio no le importan los algoritmos sino cómo los resultados ayudan a sus objetivos" → respuesta por dimensiones y recomendación operativa.
- "La iteración es crucial" → `docs/registro_feedback.md` registra cada ajuste.
