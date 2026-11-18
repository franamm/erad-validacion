"""
helpers_granularidad.py — Agregación temporal: horaria / diaria / mensual.
"""
import numpy as np
import pandas as pd
from data_loader import N_HORAS, horas


# BLOQUE 3 — HELPERS GRANULARIDAD (8760 h)
# ─────────────────────────────────────────────────────────────

MESES_H   = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
MESES_NOM = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

def _agg_serie(serie, gran):
    if gran == "Horaria" or N_HORAS <= 168:
        return list(horas), list(serie)
    elif gran == "Mensual" and N_HORAS >= 720:
        vals, labels, h = [], [], 0
        for i, mh in enumerate(MESES_H):
            end = min(h + mh, N_HORAS)
            if h >= N_HORAS:
                break
            vals.append(sum(serie[h:end]))
            labels.append(MESES_NOM[i])
            h = end
        return labels, vals
    else:
        n = N_HORAS // 24
        return list(range(1, n + 1)), [sum(serie[d*24:(d+1)*24]) for d in range(n)]

def _agg_df_energia(df_e, gran):
    if gran == "Horaria" or N_HORAS <= 168:
        return df_e
    elif gran == "Mensual" and N_HORAS >= 720:
        grupos, h = [], 0
        for mh in MESES_H:
            end = min(h + mh, len(df_e))
            if h >= len(df_e):
                break
            grupos.append(df_e.iloc[h:end].sum())
            h = end
        return pd.DataFrame(grupos, index=MESES_NOM[:len(grupos)])
    else:
        n = len(df_e) // 24
        return pd.DataFrame(
            [df_e.iloc[d*24:(d+1)*24].sum() for d in range(n)],
            index=range(1, n + 1)
        )

def _x_label(gran):
    return "Día" if gran == "Diaria" else ("Mes" if gran == "Mensual" else "Hora")

def _y_lbl(base, gran):
    return base if gran == "Horaria" else base.replace("[MW]", "[MWh]")

def _apply_gran(fig, gran):
    if gran == "Horaria" or N_HORAS <= 168:
        return fig
    for tr in fig.data:
        if tr.y is not None and len(tr.y) == N_HORAS:
            x_new, y_new = _agg_serie(list(tr.y), gran)
            tr.x = x_new
            tr.y = y_new
    return fig

# ─────────────────────────────────────────────────────────────