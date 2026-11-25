# -*- coding: utf-8 -*-
"""Genera Despliegue_FastAPI.ipynb: notebook de PLANEACIÓN del despliegue del modelo (no ejecutado).
Se aplicará después de la clase sobre despliegues con FastAPI. Uso: python tools/build_despliegue.py
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Despliegue del modelo con FastAPI — plan (pendiente de aplicar)

> **Estado: no ejecutado.** Este notebook deja planteado cómo se expondría el modelo final de `Trabajo.ipynb` como un servicio web. Se aplicará después de la sesión de clase sobre despliegues con FastAPI (11 de noviembre). Nada de lo que hay aquí forma parte de la entrega actual; es la guía para el siguiente paso.

## 1. Qué se quiere lograr
Que revenue management pueda **puntuar una reserva nueva** (o un lote) y recibir su probabilidad de cancelación, una alerta según el umbral de negocio y un ranking de riesgo, sin abrir el notebook. El artefacto ya existe: `models/modelo_cancelacion.joblib` (pipeline completo: preprocesador + Gradient Boosting afinado) y `models/metadata.json` (features, umbrales, métricas en test).

## 2. Decisiones que ya están tomadas (vienen del proyecto)
- La entrada solo puede contener columnas conocidas **al reservar**: nada de `reservation_status`, `assigned_room_type` ni `required_car_parking_spaces` (fuga / post-evento).
- El umbral por defecto es el de máximo F1 del notebook (0,37); el cliente de la API puede pasar otro (p. ej. 0,31 para recall ≥ 80 %).
- `agent` viaja como texto (código o `"SinAgencia"`) y la API deriva `has_agent`; `company` se reduce a `has_company`.
- El modelo no se reentrena en la API: si cambia el notebook, se regenera el artefacto y se reinicia el servicio.

## 3. Pasos a aplicar
1. Instalar dependencias: `pip install fastapi uvicorn httpx`.
2. Crear `api/main.py` con el esquema de entrada (pydantic), la carga del artefacto y tres endpoints: `GET /salud`, `POST /predecir`, `POST /predecir_lote`.
3. Probar con `TestClient` (una reserva de riesgo alto debe puntuar más que una de riesgo bajo; una entrada inválida debe dar 422).
4. Ejecutar `python -m uvicorn api.main:app --reload` y probar en `/docs`.
5. Documentar en el README y registrar el cierre en `docs/registro_feedback.md`.
6. Opcional: contenedor Docker, registro de predicciones y monitoreo de drift (PSI mensual sobre las entradas, como en la sección 7.4 del notebook).

## 4. Borrador del servicio (no ejecutar todavía)
La celda siguiente está marcada con `%%writefile` para que, cuando se aplique, escriba `api/main.py` directamente. Hasta entonces no se ejecuta.
""")
code(r'''
# %%writefile api/main.py
"""API mínima (FastAPI) para puntuar reservas con el modelo final del proyecto. BORRADOR: no aplicado aún."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Literal
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO = RAIZ / "models" / "modelo_cancelacion.joblib"
RUTA_META = RAIZ / "models" / "metadata.json"
app = FastAPI(title="Cancelación de reservas de hotel", version="0.1-borrador")
_modelo, _meta = None, {}

def cargar():
    global _modelo, _meta
    if _modelo is None:
        if not RUTA_MODELO.exists():
            raise HTTPException(503, "No existe el artefacto; ejecuta Trabajo.ipynb")
        _modelo = joblib.load(RUTA_MODELO); _meta = json.loads(RUTA_META.read_text(encoding="utf-8"))
    return _modelo, _meta

class Reserva(BaseModel):
    hotel: Literal["City Hotel", "Resort Hotel"]
    lead_time: int = Field(ge=0)
    arrival_date_month: str
    arrival_dow: str = "Friday"
    stays_in_weekend_nights: int = 0
    stays_in_week_nights: int = 1
    adults: int = 2
    children: int = 0
    babies: int = 0
    meal: str = "BB"
    country: str = "PRT"
    market_segment: str = "Online TA"
    distribution_channel: str = "TA/TO"
    is_repeated_guest: int = 0
    previous_cancellations: int = 0
    previous_bookings_not_canceled: int = 0
    reserved_room_type: str = "A"
    booking_changes: int = 0
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"] = "No Deposit"
    agent: str = "SinAgencia"
    has_company: int = 0
    days_in_waiting_list: int = 0
    customer_type: str = "Transient"
    adr: float = 100.0
    total_of_special_requests: int = 0

def _df(reservas, features):
    df = pd.DataFrame([r.model_dump() for r in reservas]); df["has_agent"] = (df["agent"] != "SinAgencia").astype(int)
    return df[features]

@app.get("/salud")
def salud():
    _, meta = cargar(); return {"estado": "ok", "modelo": meta.get("modelo"), "metricas_test": meta.get("metricas_test")}

@app.post("/predecir")
def predecir(reserva: Reserva, umbral: float | None = None):
    modelo, meta = cargar(); umbral = umbral or meta["metricas_test"]["umbral_max_f1"]
    p = float(modelo.predict_proba(_df([reserva], meta["features"]))[0, 1])
    return {"probabilidad_cancelacion": round(p, 4), "alerta": p >= umbral, "umbral": umbral}

@app.post("/predecir_lote")
def predecir_lote(reservas: list[Reserva], umbral: float | None = None):
    modelo, meta = cargar(); umbral = umbral or meta["metricas_test"]["umbral_max_f1"]
    probas = modelo.predict_proba(_df(reservas, meta["features"]))[:, 1]
    orden = sorted(range(len(reservas)), key=lambda i: -probas[i])
    return {"umbral": umbral, "ranking": [{"posicion": k + 1, "indice_entrada": i, "probabilidad_cancelacion": round(float(probas[i]), 4),
                                           "alerta": bool(probas[i] >= umbral)} for k, i in enumerate(orden)]}
''')
md(r"""
## 5. Borrador de pruebas (no ejecutar todavía)
""")
code(r'''
# from fastapi.testclient import TestClient
# from api.main import app
# cliente = TestClient(app)
# alto = {"hotel": "City Hotel", "lead_time": 300, "arrival_date_month": "August", "country": "PRT", "market_segment": "Groups",
#         "deposit_type": "Non Refund", "agent": "1", "previous_cancellations": 1, "adr": 120}
# bajo = {"hotel": "Resort Hotel", "lead_time": 3, "arrival_date_month": "February", "country": "GBR", "market_segment": "Direct",
#         "distribution_channel": "Direct", "is_repeated_guest": 1, "total_of_special_requests": 2, "booking_changes": 1, "adr": 60}
# assert cliente.post("/predecir", json=alto).json()["probabilidad_cancelacion"] > cliente.post("/predecir", json=bajo).json()["probabilidad_cancelacion"]
# assert cliente.post("/predecir", json={**alto, "hotel": "Otro"}).status_code == 422
# print(cliente.post("/predecir_lote", json=[bajo, alto]).json())
''')
md(r"""
## 6. Preguntas que resolver en la clase de despliegue
- ¿Cómo versionar el artefacto (fecha de entrenamiento, hash de los datos) y cómo reentrenar con ventana móvil dado el drift temporal (AUC 0,837 en 2017)?
- ¿Dónde registrar las predicciones para medir después la tasa real de cancelación por decil (monitoreo del lift)?
- ¿Conviene devolver también las contribuciones SHAP de cada reserva para explicar la alerta a revenue?
""")

nb["cells"] = cells
out = Path(__file__).resolve().parent.parent / "Despliegue_FastAPI.ipynb"
nbf.write(nb, out)
print(f"Escrito {out} con {len(cells)} celdas (sin ejecutar)")
