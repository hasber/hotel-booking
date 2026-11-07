#!/usr/bin/env python3
"""PSI (Population Stability Index) entre dos periodos para todas las columnas.

Uso como módulo:
    from psi_drift import psi_tabla
    tabla = psi_tabla(df, col_periodo="Order_Year", antiguo=[2022, 2023, 2024], reciente=[2025],
                      columnas=features + [target])

Uso como script:
    python psi_drift.py datos_limpio.csv --periodo Order_Year --antiguo 2022,2023,2024 --reciente 2025

Interpretación: PSI < 0,10 estable · 0,10–0,25 cambio moderado · > 0,25 cambio fuerte.
Para una categórica con PSI alto, mira qué categoría aparece/desaparece (cambio de esquema).
"""
import argparse

import numpy as np
import pandas as pd


def psi(ref, act, bins=10, eps=1e-6):
    """PSI entre dos Series. Numéricas: cuantiles de ref; categóricas: por categoría."""
    ref = ref.dropna()
    act = act.dropna()
    if ref.empty or act.empty:
        return np.nan
    if pd.api.types.is_numeric_dtype(ref) and ref.nunique() > bins:
        cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        cortes[0], cortes[-1] = -np.inf, np.inf
        r = pd.cut(ref, cortes).value_counts(normalize=True, sort=False)
        a = pd.cut(act, cortes).value_counts(normalize=True, sort=False)
    else:
        cats = sorted(set(ref.astype(str)) | set(act.astype(str)))
        r = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
        a = act.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    r = r.values + eps
    a = a.values + eps
    return float(np.sum((a - r) * np.log(a / r)))


def psi_tabla(df, col_periodo, antiguo, reciente, columnas=None):
    ref = df[df[col_periodo].isin(antiguo)]
    act = df[df[col_periodo].isin(reciente)]
    columnas = columnas or [c for c in df.columns if c != col_periodo]
    filas = []
    for c in columnas:
        v = psi(ref[c], act[c])
        nivel = "estable" if v < 0.10 else ("moderado" if v < 0.25 else "FUERTE")
        filas.append({"Variable": c, "PSI": round(v, 4), "Nivel": nivel})
    return pd.DataFrame(filas).sort_values("PSI", ascending=False).reset_index(drop=True)


def categorias_que_cambian(df, col, col_periodo):
    """Tabla de participación de cada categoría por periodo (para explicar un PSI alto)."""
    return (pd.crosstab(df[col_periodo], df[col], normalize="index") * 100).round(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ruta")
    ap.add_argument("--periodo", required=True)
    ap.add_argument("--antiguo", required=True, help="valores separados por coma")
    ap.add_argument("--reciente", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.ruta, low_memory=False)

    def conv(s):
        out = []
        for x in s.split(","):
            try:
                out.append(int(x))
            except ValueError:
                out.append(x)
        return out

    t = psi_tabla(df, a.periodo, conv(a.antiguo), conv(a.reciente))
    print(t.to_string(index=False))
    for c in t[t["Nivel"] != "estable"]["Variable"]:
        if df[c].nunique() <= 30:
            print(f"\n== {c}: participación (%) por periodo ==")
            print(categorias_que_cambian(df, c, a.periodo).to_string())
