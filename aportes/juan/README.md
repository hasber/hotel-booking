# Aporte de Juan — Proyecto I

## Contenido
- Proyecto_Juan.ipynb: auditoría, limpieza, EDA y modelado.
- requirements_juan.txt: versiones utilizadas en esta ejecución.
- resultados/: tablas, gráficos, predicciones e informe generado.

## Ejecución
1. Abrir el repositorio hotel-booking en VS Code.
2. Seleccionar un entorno Python con las dependencias instaladas.
3. Abrir aportes/juan/Proyecto_Juan.ipynb.
4. Ejecutar las celdas en orden.
5. Leer resultados/informe_resultados.md.

Se utiliza hotel_bookings.csv de la raíz del repositorio.
Las salidas se escriben únicamente en aportes/juan.
Cada ejecución reemplaza los resultados anteriores de este aporte.

## Aporte metodológico
Se separan entrenamiento, validación y test por fecha de llegada.
La selección del modelo y del umbral se realiza con validación.
El test se usa después de congelar esas decisiones.

## Revisión del equipo
Revisar las reglas de limpieza, exclusiones, particiones y conclusiones.
Para comparar modelos, usar las mismas filas de evaluación y evitar
evaluar como independiente un modelo que ya entrenó con esas filas.

## Autoría y herramientas
Autor: Juan.
Código preparado con asistencia de ChatGPT.
Juan y el equipo deben revisar la ejecución e interpretar sus resultados.
