# ============================================================
# LEGADO — reemplazado por pipeline.py (etapa E4)
# ============================================================
# Calculaba el caso sin fallas (faultlinMC = 0) con y sin GD, para
# obtener P_SUB_tot_sin_fallas y los CAPEX/OPEX de GD que descuenta el
# objetivo de optimization_3. Hoy:
#     python pipeline.py --etapas E4
# reproduce esos valores EXACTO (verificado al centavo) sin tocar app.pkl.
#
# NO ejecutar: la línea `pickle.dump(data, f)` de más abajo corre de forma
# incondicional y persiste sobre app.pkl el faultlinMC en CEROS que fija
# la línea 25 — que es justamente cómo el app.pkl actual quedó con su
# matriz de fallas anulada.
# ============================================================
import pickle
import numpy as np
from pathlib import Path

# from abc.mi_proyecto.app.caso_base import P_sub_total
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
from helpers_inversion import _extraer_costos_objetivo_DG


faultlinMC   = data["faultlinMC"]
Horastot = faultlinMC.shape[0]
num_esc = faultlinMC.shape[1]
n_lin = faultlinMC.shape[2]
data["faultlinMC"] = np.zeros((Horastot, num_esc, n_lin), dtype=np.int8)
faultlinMC   = data["faultlinMC"]
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

config_base_DG_sol = {
        "dg_sol":          True,
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

config_base_DG_eol = {
        "dg_sol":          False,
        "dg_eol":          True,
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

# results_sin_fallas   = resolver_optimizacion_base(data=data, config=config_base)
# results_sin_fallas = {esc: res for esc, res in results_sin_fallas.items() if res}
# data["results_sin_fallas"] = results_sin_fallas

# valid_sin_fallas   = [v for v in _results_to_list(results_sin_fallas) if v]
# esc_idx_kpi = min(esc_esp_ui - 1, len(valid_sin_fallas) - 1) if valid_sin_fallas else 0
# res_kpi     = valid_sin_fallas[esc_idx_kpi] if valid_sin_fallas else []
# P_sub_total_sin_fallas = sum(sum(res_kpi[h-1].get("P_SUB", {}).get(s, 0) for s in S) for h in horas if h - 1 < len(res_kpi)) if res_kpi else 0.0
# data["P_SUB_tot_sin_fallas"] = P_sub_total_sin_fallas
# data_DG_sol = data
# data_DG_sol["P_sub_total"] = P_sub_total_sin_fallas
# results_sin_fallas_DG_sol   = resolver_optimizacion_base(data=data_DG_sol, config=config_base_DG_sol)
# results_sin_fallas_DG_sol = {esc: res for esc, res in results_sin_fallas_DG_sol.items() if res}
# data["results_sin_fallas_DG_sol"] = results_sin_fallas_DG_sol
# valid_sin_fallas_DG_sol   = [v for v in _results_to_list(results_sin_fallas_DG_sol) if v]
# esc_idx_kpi_DG_sol = min(esc_esp_ui - 1, len(valid_sin_fallas_DG_sol) - 1) if valid_sin_fallas_DG_sol else 0
# res_kpi_DG_sol     = valid_sin_fallas_DG_sol[esc_idx_kpi_DG_sol] if valid_sin_fallas_DG_sol else []
# data_DG_eol = data
# data_DG_eol["P_sub_total"] = P_sub_total_sin_fallas
# results_sin_fallas_DG_eol   = resolver_optimizacion_base(data=data_DG_eol, config=config_base_DG_eol)
# results_sin_fallas_DG_eol = {esc: res for esc, res in results_sin_fallas_DG_eol.items() if res}
# data["results_sin_fallas_DG_eol"] = results_sin_fallas_DG_eol
# valid_sin_fallas_DG_eol   = [v for v in _results_to_list(results_sin_fallas_DG_eol) if v]
# esc_idx_kpi_DG_eol = min(esc_esp_ui - 1, len(valid_sin_fallas_DG_eol) - 1) if valid_sin_fallas_DG_eol else 0
# res_kpi_DG_eol     = valid_sin_fallas_DG_eol[esc_idx_kpi_DG_eol] if valid_sin_fallas_DG_eol else []


# ── Normalizar resultados: results_sin → 0-indexed ───────────
def _norm(d):
    """Re-indexa cualquier dict a 0-indexed según orden de claves."""
    if not d:
        return {}
    return {i: d[k] for i, k in enumerate(sorted(d.keys()))}


APP_DIR = Path(__file__).resolve().parent
from rutas import DATOS_DIR  # datos del caso activo (ver rutas.py)
CACHE_BASE = DATOS_DIR / "app.pkl"

with open(CACHE_BASE, "wb") as f:
    pickle.dump(data, f)

import pickle
from pathlib import Path
import numpy as np

    
if CACHE_BASE.exists():
    print("Cargando caso base desde cache...")

    with open(CACHE_BASE, "rb") as f:
        cache = pickle.load(f)

    results_sin_fallas = cache["results_sin_fallas"]
    P_sub_total_sin_fallas = cache["P_SUB_tot_sin_fallas"]
    results_sin_fallas_DG_sol = cache["results_sin_fallas_DG_sol"]
    results_sin_fallas_DG_eol = cache["results_sin_fallas_DG_eol"]
    data_DG_sol = data
    data_DG_eol = data

    res_kpi = _norm(results_sin_fallas)
    res_kpi_DG_sol =_norm(results_sin_fallas_DG_sol)
    costos_DG_sol = _extraer_costos_objetivo_DG(res_kpi_DG_sol, res_kpi, config_base_DG_sol, data_DG_sol)
    CAPEX_DG_sol = costos_DG_sol["capex_anual"]
    cache["CAPEX_DG_sol_sin_fallas"] = CAPEX_DG_sol
    OPEX_DG_sol = costos_DG_sol["opex_anual"]
    cache["OPEX_DG_sol_sin_fallas"] = OPEX_DG_sol
    print("DG SOLAR", CAPEX_DG_sol, OPEX_DG_sol)
    res_kpi_DG_eol =_norm(results_sin_fallas_DG_eol)
    costos_DG_eol = _extraer_costos_objetivo_DG(res_kpi_DG_eol, res_kpi, config_base_DG_eol, data_DG_eol)
    CAPEX_DG_eol = costos_DG_eol["capex_anual"]
    cache["CAPEX_DG_eol_sin_fallas"] = CAPEX_DG_eol
    OPEX_DG_eol = costos_DG_eol["opex_anual"]
    cache["OPEX_DG_eol_sin_fallas"] = OPEX_DG_eol
    print("DG EÓLICA", CAPEX_DG_eol, OPEX_DG_eol)

    with open(CACHE_BASE, "wb") as f:
        pickle.dump(cache, f)

else:
    print("Calculando caso base sin estrategias...")

    results_sin_fallas = resolver_optimizacion_base(data=data, config=config_base)
    results_sin_fallas = {esc: res for esc, res in results_sin_fallas.items() if res}
    valid_sin_fallas = [v for v in _results_to_list(results_sin_fallas) if v]

    esc_idx_kpi = min(esc_esp_ui - 1, len(valid_sin_fallas) - 1) if valid_sin_fallas else 0
    res_kpi = valid_sin_fallas[esc_idx_kpi] if valid_sin_fallas else []

    P_sub_total_sin_fallas = sum(sum(res_kpi[h - 1].get("P_SUB", {}).get(s, 0) for s in S) for h in horas if h - 1 < len(res_kpi)) if res_kpi else 0.0
    cache = {"results_sin_fallas": results_sin_fallas, "P_SUB_tot_sin_fallas": P_sub_total_sin_fallas,}

    with open(CACHE_BASE, "wb") as f:
        pickle.dump(cache, f)

    

data["results_sin_fallas"] = results_sin_fallas
data["P_SUB_tot_sin_fallas"] = P_sub_total_sin_fallas




