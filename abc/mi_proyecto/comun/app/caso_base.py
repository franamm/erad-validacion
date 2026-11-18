# ============================================================
# LEGADO — reemplazado por pipeline.py (etapa E5)
# ============================================================
# Este script calculaba el caso base con fallas para UN escenario y lo
# cacheaba dentro del propio app.pkl. Hoy esa función la cumple:
#     python pipeline.py --etapas E5
# que lo hace para todos los escenarios, en archivos separados y sin
# escribir jamás sobre app.pkl.
#
# Se conserva solo como registro de cómo se generó el estudio congelado.
# NO ejecutar: su rama de recálculo escribe sobre app.pkl y lo dejaría
# con solo dos claves, destruyendo el artefacto histórico.
# ============================================================
import pickle
import numpy as np

from data_loader import (
    net, BUSES, LINES, GENS, ESS, S,
    faultlinMC, repairMC, demand_factor,
    alphaDGsolar, alphaDGeolica, bus_zone,
    P_D_base, results_sin, SAIDI_sin,
    N_eventos, sw_cand_list, C_incentivo_edrp,
    N_HORAS, horas, numesc, esc_esp_ui,
    r_econ, n_econ, frp, VoLL, data, _norm, buses_criticos
)

from optimization_base import resolver_optimizacion_base
from helpers_operativos import _results_to_list
config_base = {
        "dg_sol":          False,
        "dg_eol":          False,
        "dg_fos":          False,
        "ess_2h":          False,
        "ess_4h":          False,
        "ess_6h":          False,
        "reconfig_topo":   False,
        "prior_criticos":  False,
        "buses_criticos":  buses_criticos,
        "meg":             False,
        "n_meg":           0,
        "P_meg_max":       0,
        "T_tra_meg":       {b: 0 for b in BUSES},
        "mess":            False,
        "n_mess":          0,
        "h_mess":          0,
        "E_cap_mess":      0,
        "E_cap_mess_total": 0,
        "P_mess_max":      0,
        "T_tra_mess":      0,
        "inv_switch_rcs":  False,
        "inv_switch_ms":   False,
        "sw_candidatas":   sw_cand_list,
        "T_maniobra_MS":   0,
        "edrp":            False,
        "numesc":          numesc,
        "esc_debug":       esc_esp_ui - 1,
        "horas":           N_HORAS,
    }

# Resolver caso base sin estrategias y guardar resultados en cache, como es un escenario solo se hace una vez previo a la ejecución de los escenarios con estrategias. 
# data["results_sin"] = results_sin

# valid_sin   = [v for v in _results_to_list(results_sin) if v]
# esc_idx_kpi = min(esc_esp_ui - 1, len(valid_sin) - 1) if valid_sin else 0
# res_kpi     = valid_sin[esc_idx_kpi] if valid_sin else []
# P_sub_total = sum(sum(res_kpi[h-1].get("P_SUB", {}).get(s, 0) for s in S) for h in horas if h - 1 < len(res_kpi)) if res_kpi else 0.0

# data["P_SUB_tot"] = P_sub_total

# with open(r"C:\Users\usuario\erad\abc\mi_proyecto\app\app.pkl", "wb") as f:
#     pickle.dump(data, f)

import pickle
from pathlib import Path
import numpy as np

APP_DIR = Path(__file__).resolve().parent
from rutas import DATOS_DIR  # datos del caso activo (ver rutas.py)
CACHE_BASE = DATOS_DIR / "app.pkl"
    
if CACHE_BASE.exists():
    print("Cargando caso base desde cache...")

    with open(CACHE_BASE, "rb") as f:
        cache = pickle.load(f)

    results_sin = cache["results_sin"]
    P_sub_total = cache["P_SUB_tot"]

else:
    print("Calculando caso base sin estrategias...")

    results_sin = resolver_optimizacion_base(
        data=data,
        config=config_base
    )

    results_sin = {
        esc: res
        for esc, res in results_sin.items()
        if res
    }

    valid_sin = [
        v for v in _results_to_list(results_sin)
        if v
    ]

    esc_idx_kpi = min(
        esc_esp_ui - 1,
        len(valid_sin) - 1
    ) if valid_sin else 0

    res_kpi = valid_sin[esc_idx_kpi] if valid_sin else []

    P_sub_total = sum(
        sum(
            res_kpi[h - 1].get("P_SUB", {}).get(s, 0)
            for s in S
        )
        for h in horas
        if h - 1 < len(res_kpi)
    ) if res_kpi else 0.0

    cache = {
        "results_sin": results_sin,
        "P_SUB_tot": P_sub_total,
    }

    with open(CACHE_BASE, "wb") as f:
        pickle.dump(cache, f)

data["results_sin"] = results_sin
data["P_SUB_tot"] = P_sub_total