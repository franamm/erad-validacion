"""
helpers_operativos.py — Funciones de extracción de series temporales
por hora: ENS, carga servida, DG, ESS, MESS, EDRP.
"""
import numpy as np
import pandas as pd
from data_loader import BUSES, LINES, horas, N_HORAS, bus_zone, P_D_base, demand_factor, VoLL


# BLOQUE 4 — HELPERS OPERATIVOS
# ─────────────────────────────────────────────────────────────

PALETA = [
    "#d62728", "#1f77b4", "#2ca02c", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf"
]


PALETA = [
    "#d62728",  # rojo
    "#1f77b4",  # azul
    "#2ca02c",  # verde
    "#9467bd",  # morado
    # "#8c564b",  # café
    "#e377c2",  # rosado
    "#bcbd22",  # oliva
    "#17becf",  # celeste
    "#ff7f0e",  # naranjo
    "#7f7f7f",  # gris
    "#8c564b",  # café
    "#393b79",  # azul oscuro
    "#637939",  # verde oscuro
]
NODEFILL    = {1: "#BFDBFE", 2: "#FDE68A",  3: "#A7F3D0"}

def _nombre_estrategia_corto(label, cfg=None):
    """Acorta labels muy largos para tablas y gráficos."""
    if not label:
        return label
    txt = str(label).strip()
    txt_low = txt.lower()

    if cfg is not None:
        keys_all = [
            "dg_sol", "dg_eol", "dg_fos",
            "ess_5h", "ess_4h", "ess_6h",
            "reconfig_topo", "inv_tie", "prior_criticos", "meg",
            "inv_switch_rcs", "inv_switch_ms", "edrp", "veg_mgmt",
            "refuerzo", "soterramiento",
        ]
        n_on = sum(1 for k in keys_all if cfg.get(k, False))
        if n_on >= 10:
            return "Estrategia completa"

    if len(txt) > 70:
        if all(s in txt_low for s in [
            "dg sol", "dg eól", "dg fos", "ess 5h", "ess 4h", "ess 6h",
            "reconfig", "priorización", "switch rcs", "switch ms", "edrp"
        ]):
            return "Estrategia completa"
        return txt[:67] + "..."
    return txt

def _color_zona(z):
    return {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}.get(z, "#888")

def _results_to_list(results_dict):
    return [results_dict[k] for k in sorted(results_dict.keys())]

def _ens_por_hora(results_esc):
    if not results_esc or len(results_esc) < len(horas):
        return [0.0] * N_HORAS
    return [sum(results_esc[h-1].get("P_LC", {}).get(b, 0) for b in BUSES) for h in horas]

def _carga_servida_por_hora(results_esc):
    if not results_esc or len(results_esc) < len(horas):
        return [0.0] * N_HORAS
    return [results_esc[h-1].get("carga_servida_MW", 0) for h in horas]

def _lineas_danadas(results_esc):
    if not results_esc or len(results_esc) < len(horas):
        return [0] * N_HORAS
    vals, ultimo = [], 0
    for h in horas:
        d = results_esc[h - 1]
        if d:
            ultimo = d.get("n°lineas_danadas", 0)
        vals.append(ultimo)
    return vals

def _tasa_interconexion_por_hora(results_esc):
    if not results_esc or len(results_esc) < len(horas):
        return [100.0] * N_HORAS
    total = len(LINES)
    return [(total - results_esc[h-1].get("n°lineas_danadas", 0)) / total * 100 if h-1 < len(results_esc) else 0.0
            for h in horas]

def _ens_por_zona_total(results_esc):
    ens_zona = {1: 0.0, 2: 0.0, 3: 0.0}
    if not results_esc or len(results_esc) < len(horas):
        return ens_zona
    for h in horas:
        P_LC = results_esc[h-1].get("P_LC", {})
        for b in BUSES:
            ens_zona[bus_zone.get(b, 1)] += P_LC.get(b, 0)
    return ens_zona

def _energia_dg_por_hora(results_esc):
    if not results_esc or len(results_esc) < len(horas):
        return [0.0]*N_HORAS, [0.0]*N_HORAS
    sol, eol = [], []
    for h in horas:
        d = results_esc[h-1]
        sol.append(sum(d.get("P_DG_sol", {}).values()))
        eol.append(sum(d.get("P_DG_eol", {}).values()))
    return sol, eol

def _energia_fos_por_hora(results_esc):
    if not results_esc or len(results_esc) < len(horas):
        return [0.0] * N_HORAS
    def __gfh_safe(r,h): return sum(r[h-1].get("P_DG_fos", {}).values()) if h-1 < len(r) else 0.0
    return [__gfh_safe(results_esc, h) for h in horas]

def _energia_meg_por_hora(results_esc):
    if not results_esc or len(results_esc) < len(horas):
        return [0.0] * N_HORAS
    def __gmh_safe(r,h): return sum(r[h-1].get("P_MEG", {}).values()) if h-1 < len(r) else 0.0
    return [__gmh_safe(results_esc, h) for h in horas]

def _energia_mess_por_hora(results_esc, modo="descarga"):
    """Serie horaria agregada de MESS usando la clave real P_mess del optimizador.
    modo = 'neta' suma algebraica; 'descarga' solo potencias positivas; 'movilizada' suma abs()."""
    if not results_esc or len(results_esc) < len(horas):
        return [0.0] * N_HORAS

    vals = []
    for h in horas:
        d = results_esc[h - 1]
        p_mess = d.get("P_mess", {}) or d.get("P_MESS", {})
        if not isinstance(p_mess, dict):
            vals.append(0.0)
            continue
        p_vals = [float(v) for v in p_mess.values()]
        if modo == "neta":
            vals.append(sum(p_vals))
        elif modo == "movilizada":
            vals.append(sum(abs(v) for v in p_vals))
        else:
            vals.append(sum(max(0.0, v) for v in p_vals))
    return vals

def _soc_mess_por_hora(results_esc, modo="promedio"):
    if not results_esc or len(results_esc) < len(horas):
        return [0.0] * N_HORAS

    vals = []
    for h in horas:
        d = results_esc[h - 1]
        soc_mess = d.get("SOC_mess", {}) or d.get("SOC_MESS_MWh", {})
        if not isinstance(soc_mess, dict) or not soc_mess:
            vals.append(0.0)
            continue
        soc_vals = [float(v) for v in soc_mess.values()]
        if modo == "total":
            vals.append(sum(soc_vals))
        else:
            vals.append(sum(soc_vals) / len(soc_vals))
    return vals

def _mess_por_bus(results_esc, modo="descarga"):
    if not results_esc or len(results_esc) < len(horas):
        return pd.DataFrame(0.0, index=[f"H{h}" for h in horas], columns=BUSES)

    rows = []
    idx = []
    for h in horas:
        d = results_esc[h - 1]
        p_mess = d.get("P_mess", {}) or d.get("P_MESS", {})
        row = {b: 0.0 for b in BUSES}
        if isinstance(p_mess, dict):
            for key, val in p_mess.items():
                if isinstance(key, tuple) and len(key) == 2:
                    _, b = key
                    if b in row:
                        if modo == "neta":
                            row[b] += float(val)
                        elif modo == "movilizada":
                            row[b] += abs(float(val))
                        else:
                            row[b] += max(0.0, float(val))
        rows.append(row)
        idx.append(f"H{h}")
    return pd.DataFrame(rows, index=idx)

def _extraer_cap_mess(results_label):
    K_vals = []
    for res_lista in _results_to_list(results_label):
        if not res_lista:
            continue
        caps = res_lista[0].get("CAP_MESS_MWh", {})
        if caps:
            K_vals.append(caps)
    if not K_vals:
        return {}
    all_k = set().union(*[d.keys() for d in K_vals])
    return {k: np.mean([d.get(k, 0) for d in K_vals]) for k in sorted(all_k)}

def _mess_resumen_unidades(results_esc):
    if not results_esc:
        return pd.DataFrame()
    km_ids = sorted({k for d in results_esc for k in d.get("CAP_MESS_MWh", {}).keys()})
    rows = []
    for k in km_ids:
        cap = results_esc[0].get("CAP_MESS_MWh", {}).get(k, 0.0)
        # "P_MESS_MW" es la clave vigente; "P_MESS_MAX_MW" la antigua (respaldo
        # para resultados ya guardados).
        _pmax_dic = results_esc[0].get("P_MESS_MW") or results_esc[0].get("P_MESS_MAX_MW", {})
        pmax = _pmax_dic.get(k, 0.0)
        h_eq = results_esc[0].get("H_MESS_h", {}).get(k, 0.0)
        soc_fin = results_esc[-1].get("SOC_mess", {}).get(k, 0.0)
        energia_desc = 0.0
        energia_carg = 0.0
        horas_act = 0
        buses_visitados = set()
        for d in results_esc:
            p_m = d.get("P_mess", {})
            u_m = d.get("u_mess", {})
            p_net_k = sum(v for (kk, _), v in p_m.items() if kk == k)
            energia_desc += max(0.0, p_net_k)
            energia_carg += max(0.0, -p_net_k)
            if abs(p_net_k) > 1e-9:
                horas_act += 1
            for (kk, b), u in u_m.items():
                if kk == k and u > 0.5:
                    buses_visitados.add(b)
        rows.append({
            "MESS": k,
            "Capacidad [kWh]": round(cap, 3),
            "Potencia máx [kW]": round(pmax, 3),
            "Duración eq. [h]": round(h_eq, 3),
            "Descarga acumulada [kWh]": round(energia_desc, 3),
            "Carga acumulada [kWh]": round(energia_carg, 3),
            "Horas activo": int(horas_act),
            "Buses visitados": len(buses_visitados),
            "SOC final [kWh]": round(soc_fin, 3),
        })
    return pd.DataFrame(rows)

def _mess_localizacion_horaria(results_esc):
    if not results_esc:
        return pd.DataFrame()
    km_ids = sorted({k for d in results_esc for k in d.get("CAP_MESS_MWh", {}).keys()})
    rows = []
    for d in results_esc:
        h = d.get("hora", None)
        p_m = d.get("P_mess", {})
        u_m = d.get("u_mess", {})
        soc_m = d.get("SOC_mess", {})
        for k in km_ids:
            buses_k = [b for (kk, b), u in u_m.items() if kk == k and u > 0.5]
            p_net = sum(v for (kk, _), v in p_m.items() if kk == k)
            rows.append({
                "Hora": h,
                "MESS": k,
                "Bus": buses_k[0] if buses_k else "—",
                "Potencia neta [kW]": round(p_net, 5),
                "Descarga [kW]": round(max(0.0, p_net), 5),
                "Carga [kW]": round(max(0.0, -p_net), 5),
                "SOC [kWh]": round(soc_m.get(k, 0.0), 5),
            })
    return pd.DataFrame(rows)

def _tie_cerrados_por_hora(results_esc):
    if not results_esc or len(results_esc) < len(horas):
        return [0.0] * N_HORAS
    def __zth_safe(r,h): return sum(r[h-1].get("z_tie", {}).values()) if h-1 < len(r) else 0.0
    return [__zth_safe(results_esc, h) for h in horas]

def _soc_promedio_por_hora(results_esc):
    if not results_esc or len(results_esc) < len(horas):
        return [0.0] * N_HORAS
    vals = []
    for h in horas:
        d = results_esc[h-1]
        soc_combined = {}
        soc_combined.update(d.get("SOC_MWh_5h", {}))
        soc_combined.update(d.get("SOC_MWh_4h", {}))
        soc_combined.update(d.get("SOC_MWh_6h", {}))
        vals.append(np.mean(list(soc_combined.values())) if soc_combined else 0)
    return vals
def _df_energia(results_esc, esc_idx=0):
    """esc_idx: índice de escenario para demand_factor (0-indexed)."""
    idx   = [f"H{h}" for h in horas]
    df_ns = pd.DataFrame(index=idx, columns=[1,2,3], dtype=float).fillna(0.0)
    df_s  = df_ns.copy()
    if not results_esc or len(results_esc) < len(horas):
        return df_s, df_ns
    for h in horas:
        label   = f"H{h}"
        P_LC    = results_esc[h-1].get("P_LC", {})
        # Demanda real = P_D_base[b] * demand_factor[h-1, esc, b]
        # demand_factor shape: (Horastot, numesc, n_buses)
        h_idx   = h - 1
        pd_zona = {1: 0.0, 2: 0.0, 3: 0.0}
        ns_zona = {1: 0.0, 2: 0.0, 3: 0.0}
        for bi, b in enumerate(BUSES):
            try:
                df_h = P_D_base[b] * demand_factor[h_idx, esc_idx, bi]
            except (IndexError, KeyError):
                df_h = P_D_base[b]
            pd_zona[bus_zone.get(b, 1)] += df_h
            ns_zona[bus_zone.get(b, 1)] += P_LC.get(b, 0)
        for z in [1, 2, 3]:
            df_ns.at[label, z] = ns_zona[z]
            df_s.at[label, z]  = pd_zona[z] - ns_zona[z]
    return df_s, df_ns

def _edrp_por_hora(results_esc):
    if not results_esc or len(results_esc) < len(horas):
        return [0.0] * N_HORAS
    vals = []
    for h in horas:
        d = results_esc[h-1]
        edrp_total = sum(
            v for (b, c), v in d.get("P_EDRP", {}).items()
        )
        vals.append(edrp_total)
    return vals


def calcular_edrp_por_nodo(results_esc, BUSES):
    edrp_por_nodo = {b: 0.0 for b in BUSES}

    for r in results_esc:
        for b in BUSES:
            edrp_por_nodo[b] += r["EDRP_total"].get(b, 0.0)

    return edrp_por_nodo

import pandas as pd
import pandas as pd

def _flatten_resultados_horarios(obj):
    """
    Convierte results_activos / results_por_esc / listas anidadas
    en una lista plana de diccionarios horarios.
    """
    planos = []

    if obj is None:
        return planos

    # Caso: diccionario
    if isinstance(obj, dict):
        # Si ya es un resultado horario
        if "P_EDRP" in obj or "EDRP_total" in obj:
            return [obj]

        # Si es dict de escenarios o estrategias
        for v in obj.values():
            planos.extend(_flatten_resultados_horarios(v))

        return planos

    # Caso: lista
    if isinstance(obj, list):
        for item in obj:
            planos.extend(_flatten_resultados_horarios(item))

        return planos

    return planos


def calcular_edrp_nodo_bloque(results, BUSES, C_bloques_EDRP=4):
    """
    Calcula EDRP acumulado por nodo y por bloque.
    Acepta resultados planos o anidados.
    """

    results_planos = _flatten_resultados_horarios(results)

    rows = []

    for b in BUSES:
        fila = {"bus": b}
        total_bus = 0.0

        for c in range(C_bloques_EDRP):
            total_bloque = 0.0

            for r in results_planos:
                p_edrp = r.get("P_EDRP", {})

                # Caso normal: llave tipo tupla
                val = p_edrp.get((b, c), 0.0)

                # Respaldo por si la llave quedó como string
                if val == 0.0:
                    val = p_edrp.get(str((b, c)), 0.0)

                total_bloque += val

            fila[f"Bloque_{c+1}"] = total_bloque
            total_bus += total_bloque

        fila["EDRP_total"] = total_bus
        rows.append(fila)

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values("EDRP_total", ascending=False)

    return df