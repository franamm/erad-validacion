"""
helpers_estadisticos.py — Funciones de agregación estadística:
promedios por hora y selección del peor caso entre escenarios.
"""
import numpy as np
import pandas as pd
from data_loader import BUSES, horas, N_HORAS, demand_factor
from helpers_operativos import _df_energia, _results_to_list


# BLOQUE 5 — HELPERS PROMEDIO / PEOR CASO
# ─────────────────────────────────────────────────────────────

def _esc_peor_caso(SAIDI_dict):
    if not SAIDI_dict:
        return 0
    return max(SAIDI_dict, key=SAIDI_dict.get) - 1

def _promedio_por_hora(lista_series):
    return np.mean(lista_series, axis=0).tolist()

def _peor_caso_por_hora(lista_series):
    return np.max(lista_series, axis=0).tolist()

def _lineas_danadas_todos(results_dict):
    return [_lineas_danadas(v) for v in _results_to_list(results_dict) if v]

def _df_energia_promedio(results_dict):
    idx   = [f"H{h}" for h in horas]
    valid = [v for v in _results_to_list(results_dict) if v]
    if not valid:
        return pd.DataFrame(), pd.DataFrame()
    if len(valid) == 1:
        return _df_energia(valid[0], esc_idx=0)
    acum_s  = {z: np.zeros(N_HORAS) for z in [1,2,3]}
    acum_ns = {z: np.zeros(N_HORAS) for z in [1,2,3]}
    for ei, v in enumerate(valid):
        ds, dns = _df_energia(v, esc_idx=min(ei, demand_factor.shape[1]-1))
        for z in [1,2,3]:
            acum_s[z]  += ds[z].values
            acum_ns[z] += dns[z].values
    n = len(valid)
    return (pd.DataFrame({z: acum_s[z]/n  for z in [1,2,3]}, index=idx),
            pd.DataFrame({z: acum_ns[z]/n for z in [1,2,3]}, index=idx))

def _df_energia_peorcase(results_dict, SAIDI_dict):
    valid = [v for v in _results_to_list(results_dict) if v]
    if not valid:
        return _df_energia([])
    peor = _esc_peor_caso(SAIDI_dict)
    idx  = min(peor, len(valid) - 1)
    return _df_energia(valid[idx], esc_idx=min(idx, demand_factor.shape[1]-1))

# ─────────────────────────────────────────────────────────────