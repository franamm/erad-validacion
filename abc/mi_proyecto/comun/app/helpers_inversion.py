"""
helpers_inversion.py — Extracción de capacidades instaladas y cálculo
de costos (CAPEX / OPEX / ENS) para cada estrategia de resiliencia.
"""
import numpy as np
import re
import unicodedata
from data_loader import (
    BUSES, LINES, GENS, ESS, S, horas, N_HORAS, data,
    VoLL, N_eventos, frp, frp_movil, bus_zone,
    C_inv_DG_sol, C_op_DG_sol, C_inv_DG_eol, C_op_DG_eol,
    C_inv_DG_fos, C_op_DG_fos, C_comb_DG_fos,
    C_inv_ESS_5h, C_op_ESS_5h, C_inv_ESS_4h, C_op_ESS_4h,
    C_inv_ESS_6h, C_op_ESS_6h, h_ess_4h, h_ess_5h, h_ess_6h,
    C_inv_MEG, C_op_MEG, C_comb_MEG,
    C_inv_mess, C_op_mess, C_tra_mess,
    C_inv_SWITCH, C_op_SWITCH, C_inv_RCS, C_op_RCS,
    C_inv_MS, C_op_MS, C_incentivo_edrp, VoLL_local, C_sub,
    C_inv_veg, C_op_veg, C_inv_ref, C_op_ref, C_inv_ug, C_op_ug,
    C_inv_TIE, C_op_TIE,
)
from helpers_operativos import _results_to_list, PALETA


def _baseline_de(base_list, i):
    """Baseline correspondiente al i-ésimo escenario de la estrategia.

    Antes esto caía en silencio a `base_list[-1]` cuando el índice no
    existía, así que al analizar un escenario sin baseline propio la
    comparación se hacía contra el baseline de OTRO escenario sin ningún
    aviso. Ahora se devuelve vacío, que los llamadores ya manejan como
    "sin comparación disponible".
    """
    if i < len(base_list):
        return base_list[i]
    print(f"[!] No hay baseline para el escenario en posición {i} "
          f"({len(base_list)} disponible/s). Correr la etapa E5 del pipeline "
          f"para ese escenario.")
    return []

h_ess_4h = 4
h_ess_5h = 5
h_ess_6h = 6

MAPA_COLORES_ESTRATEGIAS = {
    "DG Sol": "#f1c40f",
    "DG Eól": "#3498db",
    "DG Fos": "#8c564b",  # café
    "ESS 4h": "#8e44ad",
    "ESS 5h": "#bcbd22",  # oliva
    "ESS 6h": "#e377c2",  # rosado
    "Reconfig": "#2ecc71",
    "Enlaces nuevos": "#16a085",
    "Priorización cargas críticas": "#e67e22",
    "MEG": "#e67e22",
    "MESS": "#17becf",
    "Switch RCS": "#1f77b4",
    "Switch MS": "#2ca02c",
    "EDRP": "#d62728",
    "Gestión de vegetación": "#556b2f",  # verde oliva oscuro
    "Refuerzo estructural": "#7f8c8d",  # gris acero
    "Soterramiento": "#4b3621",  # café tierra
}


def _normalizar_nombre_estrategia(nombre):
    """
    Normaliza nombres para reconocer diferencias como:

        DG Eól / DG Eol
        MEG×2 / MEGx2
        MESS×3 (5h)
        espacios adicionales
    """
    texto = str(nombre).strip()

    texto = unicodedata.normalize(
        "NFKD",
        texto,
    )

    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )

    texto = texto.replace("×", "x")
    texto = texto.casefold()

    return " ".join(texto.split())


def obtener_color_estrategia(nombre):
    """
    Devuelve un color estable según el tipo de estrategia.

    También reconoce nombres dinámicos, como:

        MEG×2
        MESS×1 (5h)
        Switch RCS (4 líneas)
        Switch MS (4 líneas)
    """
    nombre_normalizado = _normalizar_nombre_estrategia(
        nombre
    )

    # Conviene comprobar MESS antes que MEG.
    reglas = [
        (
            "priorizacion cargas criticas",
            MAPA_COLORES_ESTRATEGIAS[
                "Priorización cargas críticas"
            ],
        ),
        (
            "switch rcs",
            MAPA_COLORES_ESTRATEGIAS["Switch RCS"],
        ),
        (
            "switch ms",
            MAPA_COLORES_ESTRATEGIAS["Switch MS"],
        ),
        (
            "ess 4h",
            MAPA_COLORES_ESTRATEGIAS["ESS 4h"],
        ),
        (
            "ess 5h",
            MAPA_COLORES_ESTRATEGIAS["ESS 5h"],
        ),
        (
            "ess 6h",
            MAPA_COLORES_ESTRATEGIAS["ESS 6h"],
        ),
        (
            "dg sol",
            MAPA_COLORES_ESTRATEGIAS["DG Sol"],
        ),
        (
            "dg eol",
            MAPA_COLORES_ESTRATEGIAS["DG Eól"],
        ),
        (
            "dg fos",
            MAPA_COLORES_ESTRATEGIAS["DG Fos"],
        ),
        (
            "reconfig",
            MAPA_COLORES_ESTRATEGIAS["Reconfig"],
        ),
        (
            "mess",
            MAPA_COLORES_ESTRATEGIAS["MESS"],
        ),
        (
            "meg",
            MAPA_COLORES_ESTRATEGIAS["MEG"],
        ),
        (
            "edrp",
            MAPA_COLORES_ESTRATEGIAS["EDRP"],
        ),
        (
            "gestion vegetacion",
            MAPA_COLORES_ESTRATEGIAS["Gestión de vegetación"],
        ),
        (
            "refuerzo estructural",
            MAPA_COLORES_ESTRATEGIAS["Refuerzo estructural"],
        ),
        (
            "soterramiento",
            MAPA_COLORES_ESTRATEGIAS["Soterramiento"],
        ),
    ]

    for texto_busqueda, color in reglas:
        if texto_busqueda in nombre_normalizado:
            return color

    # Color para nombres no reconocidos
    return "#95a5a6"


def _traducir_nombre_estrategia_display(nombre):
    """
    Traduce siglas al español solo para mostrar en gráficos
    (leyenda, hover, etiquetas de barra), sin tocar el nombre
    original que se usa en el resto de la app (matching de
    colores, tablas, claves de datos, etc.).

    "ESS" (Energy Storage System) -> "SAE" (Sistema de
    Almacenamiento de Energía). Usa \\b para no afectar "MESS"
    (Almacenamiento móvil), que es una estrategia distinta.

    "DG" (Distributed Generation) -> "GD" (Generación
    Distribuida), término regulatorio usado en Chile (Ley 20.571).

    "MESS" (Mobile Emergency Storage System) -> "SAE móvil" y
    "MEG" (Mobile Emergency Generator) -> "GD móvil": versión
    desplegable/temporal de SAE y GD, respectivamente. Se
    reemplazan antes que ESS/DG para no dejar residuos como
    "SAE" o "GD" sueltos dentro del nombre.

    "Switch RCS"/"Switch MS" y "EDRP" se dejan tal cual, a
    pedido explícito.
    """
    texto = str(nombre)
    texto = re.sub(r"\bMESS\b", "SAE móvil", texto)
    texto = re.sub(r"\bMEG\b", "GD móvil", texto)
    texto = re.sub(r"\bESS\b", "SAE", texto)
    texto = re.sub(r"\bDG\b", "GD", texto)
    return texto

# BLOQUE 6 — HELPERS DE INVERSIÓN
# ─────────────────────────────────────────────────────────────

def _extraer_cap_dg(results_label):
    acum_sol = {g: [] for g in GENS}
    acum_eol = {g: [] for g in GENS}
    acum_fos = {g: [] for g in GENS}

    for res_lista in _results_to_list(results_label):
        if not res_lista:
            continue
        hora0 = res_lista[0]
        for g in GENS:
            acum_sol[g].append(hora0.get("CAP_DG_sol_kW", {}).get(g, 0))
            acum_eol[g].append(hora0.get("CAP_DG_eol_kW", {}).get(g, 0))
            acum_fos[g].append(hora0.get("CAP_DG_fos_kW", {}).get(g, 0))

    return {
        g: {
            "sol": acum_sol[g] if acum_sol[g] else 0,
            "eol": acum_eol[g] if acum_eol[g] else 0,
            "fos": acum_fos[g] if acum_fos[g] else 0,
            "bus":  g,                      # índice g == bus
            "zona": bus_zone.get(g, "—"),   # zona desde el dict global
        }
        for g in GENS
    }
#         }
#         for g in GENS
#     }

def _extraer_cap_meg(results_label):
    K_vals = []
    for res_lista in _results_to_list(results_label):
        if not res_lista:
            continue
        caps = res_lista[0].get("CAP_MEG_MW", {})
        if caps:
            K_vals.append(caps)
    if not K_vals:
        return {}
    all_k = set().union(*[d.keys() for d in K_vals])
    return {k: np.mean([d.get(k, 0) for d in K_vals]) for k in sorted(all_k)}


def _extraer_cap_mess(results_label):
    K_vals = []
    for res_lista in _results_to_list(results_label):
        if not res_lista:
            continue
        caps = res_lista[0].get("CAP_MESS_MW", {})
        if caps:
            K_vals.append(caps)
    if not K_vals:
        return {}
    all_k = set().union(*[d.keys() for d in K_vals])
    return {k: np.mean([d.get(k, 0) for d in K_vals]) for k in sorted(all_k)}



def _extraer_switches_inv(results_label, sw_cands):
    """Retorna y_rcs, y_ms (decisión inversión) y horas activas promedio por línea."""
    if not sw_cands:
        return {}
    valid = [v for v in _results_to_list(results_label) if v]
    if not valid:
        return {}
    out = {}
    for l in sw_cands:
        y_rcs_vals = [res_lista[0].get("y_rcs", {}).get(l, 0) for res_lista in valid]
        y_ms_vals  = [res_lista[0].get("y_ms",  {}).get(l, 0) for res_lista in valid]
        horas_act  = sum(1 for res_lista in valid for hd in res_lista if hd.get("z_sw", {}).get(l, 0) < 0.5)
        p_sw_prom = np.mean([
            sum(hd.get("P_sw", {}).get(l, 0) for hd in res_lista)
            for res_lista in valid if res_lista
        ]) if valid else 0
        out[l] = {
            "y_rcs": np.mean(y_rcs_vals),
            "y_ms":  np.mean(y_ms_vals),
            "horas_activa_prom": round(horas_act / len(valid), 1),
            "P_sw_total_prom":   round(p_sw_prom, 3),
        }
    return out

def _extraer_costos_objetivo_DG(results_estrategia, results_base, cfg, data):
    costos = []
    base_list = _results_to_list(results_base)

    for i, horas_data in enumerate(_results_to_list(results_estrategia)):
        if not horas_data:
            continue

        base_esc = _baseline_de(base_list, i)

        _n_h = len(horas_data)
        _h_safe = [h for h in horas if h - 1 < _n_h]
        _n_hb = len(base_esc) if base_esc else 0
        _hb_safe = [h for h in horas if h - 1 < _n_hb]

        def _hd(h, key, sub=None):
            d = horas_data[h - 1].get(key, {})
            return d.get(sub, 0) if sub is not None else d

        hora0 = horas_data[0]
        funcion_objetivo = hora0.get("fn_objetivo", 0)

        # CAPEX DG
        capex_sol = sum(hora0.get("CAP_DG_sol_kW", {}).get(g, 0) * C_inv_DG_sol[g] * frp for g in GENS) 
        capex_eol = sum(hora0.get("CAP_DG_eol_kW", {}).get(g, 0) * C_inv_DG_eol[g] * frp for g in GENS) 
        capex_fos = sum(hora0.get("CAP_DG_fos_kW", {}).get(g, 0) * C_inv_DG_fos[g] * frp for g in GENS)

        # OPEX DG
        opex_sol = sum(hora0.get("CAP_DG_sol_kW", {}).get(g, 0) * C_op_DG_sol[g]  for g in GENS) 
        opex_eol = sum(hora0.get("CAP_DG_eol_kW", {}).get(g, 0) * C_op_DG_eol[g]  for g in GENS)
        opex_fos = sum(sum(_hd(h, "P_DG_fos", g) * (C_comb_DG_fos[g]*0.000242 + C_op_DG_fos[g]) for h in _h_safe) for g in GENS) + sum(hora0.get("CAP_DG_fos_kW", {}).get(g, 0) * C_inv_DG_fos[g] * 0.01 for g in GENS)
        # comb_fos = sum(C_comb_DG_fos[g] * _hd(h, "P_DG_fos", g) * frp for h in _h_safe for g in GENS)
        # OPEX MESS
        # opex_mess = sum(C_tra_mess * cfg.get("T_tra_mess",{}).get(b,2) for k in cap_mess for b in BUSES)
        faultlinMC = data["faultlinMC"]
        Horastot, num_esc, lineas = faultlinMC.shape
        esc_idx = data.get("esc_esp", 0)  # índice del escenario que estás procesando

        horas_evento = [t for t in range(Horastot) if np.any(faultlinMC[t, esc_idx, :] == 1)]
        bloques_evento = []
        if horas_evento:
            inicio = horas_evento[0]
            fin = horas_evento[0]
            for t in horas_evento[1:]:
                if t > fin + 1:
                    bloques_evento.append((inicio, fin))
                    inicio = fin = t
                else:
                    fin = t
            bloques_evento.append((inicio, fin))


        # ENS
        ens_cost_base = sum(
            VoLL.get(b, 1.0) * base_esc[h - 1].get("P_LC", {}).get(b, 0)
            for h in _hb_safe for b in BUSES
        ) * N_eventos

        ens_cost_estr = sum(VoLL_local.get(b, 1.0) * _hd(h, "P_LC", b) for h in _h_safe for b in BUSES) * N_eventos if cfg.get("prior_criticos", False) else sum(VoLL.get(b, 1.0) * _hd(h, "P_LC", b) for h in _h_safe for b in BUSES) * N_eventos 

                        
        cost_ahorro_sub = - (data.get("P_SUB_tot_sin_fallas") - sum(_hd(h, "P_SUB", s) for h in _h_safe for s in S)) * C_sub.get(0, 1.0)

        capex_total = (
            capex_sol + capex_eol + capex_fos 
        )

        opex_total = (opex_sol + opex_eol + opex_fos )

        ahorro_ens = ens_cost_base - ens_cost_estr
        costos.append({
            "capex_anual": capex_total,
            "opex_anual": opex_total + cost_ahorro_sub,
            "costo_total": capex_total + opex_total + cost_ahorro_sub,
            "fn_objetivo": funcion_objetivo,
            "costo_ens_base": ens_cost_base,
            "costo_ens_estr": ens_cost_estr,
            "ahorro_ens": ahorro_ens,
            "beneficio": funcion_objetivo,
        })

    if not costos:
        return {
            "capex_anual": 0,
            "opex_anual": 0,
            "costo_total": 0,
            "fn_objetivo": 0,
            "costo_ens_base": 0,
            "costo_ens_estr": 0,
            "ahorro_ens": 0,
            "beneficio": 0,
        }

    return {k: np.mean([c[k] for c in costos]) for k in costos[0]}


def _extraer_costos_objetivo(results_estrategia, results_base, cfg, data):
    costos = []
    base_list = _results_to_list(results_base)

    sw_cands_local = data.get("sw_cand_list", [])
    # Las mismas tie-lines con que resolvió el optimizador (su config), no
    # "las tres últimas": en Oberrhein los enlaces son [8, 20, 28, 62, 82, 175]
    # y el OPEX de maniobra se sumaba sobre líneas sin w_tie (= 0).
    _ties_cfg = (cfg or {}).get("tie_indices")
    tie_indices_local = set(_ties_cfg if _ties_cfg is not None else LINES[-3:])
    if (cfg or {}).get("inv_tie"):
        tie_indices_local |= set((cfg or {}).get("tie_candidatas") or [])

    

    for i, horas_data in enumerate(_results_to_list(results_estrategia)):
        if not horas_data:
            continue

        base_esc = _baseline_de(base_list, i)

        _n_h = len(horas_data)
        _h_safe = [h for h in horas if h - 1 < _n_h]
        _n_hb = len(base_esc) if base_esc else 0
        _hb_safe = [h for h in horas if h - 1 < _n_hb]

        def _hd(h, key, sub=None):
            d = horas_data[h - 1].get(key, {})
            return d.get(sub, 0) if sub is not None else d

        hora0 = horas_data[0]
        funcion_objetivo = hora0.get("fn_objetivo", 0)
        x_re = hora0.get("x_re", {})

        def _get_x_re(k, b):
            """
            Permite leer x_re aunque esté guardado como:
            - x_re[(k,b)]
            - x_re[k][b]
            - x_re["k,b"]
            - x_re[str((k,b))]
            """
            if not isinstance(x_re, dict):
                return 0

            if (k, b) in x_re:
                return x_re.get((k, b), 0)

            if k in x_re and isinstance(x_re[k], dict):
                return x_re[k].get(b, 0)

            return x_re.get(f"{k},{b}", x_re.get(str((k, b)), 0))

        # `OPEX_DG_<tec>_sin_fallas` NO es solo OPEX: pipeline._capex_opex_dg lo
        # guarda como  O_sf + (-(P_SUB_tot_sin_fallas - P_SUB_sf_<tec>) * C_sub),
        # es decir OPEX puro MÁS el crédito de subestación que esa tecnología
        # lograría sin fallas. Sumados dan el término correcto, pero mezclados en
        # una sola cifra no se puede auditar ninguna de las dos mitades. Acá se
        # separan para reportarlas aparte; el total no cambia.
        _csub = C_sub.get(0, 1.0)
        _p_sf0 = data.get("P_SUB_tot_sin_fallas", 0.0) or 0.0

        def _split_sf(tec):
            """(OPEX puro sin fallas, crédito de subestación embebido) de `tec`."""
            guardado = data.get(f"OPEX_DG_{tec}_sin_fallas") or 0.0
            p_tec = data.get(f"P_SUB_sf_{tec}")
            if p_tec is None:
                # Sin el dato por tecnología no se puede separar; se atribuye
                # todo a OPEX, que es el comportamiento histórico.
                return guardado, 0.0
            credito = -(_p_sf0 - p_tec) * _csub
            return guardado - credito, credito

        # CAPEX DG
        sub_sf_sol = sub_sf_eol = 0.0
        if cfg.get("dg_sol", False):
            _o_sf, _cred = _split_sf("sol")
            opex_sol = sum(hora0.get("CAP_DG_sol_kW", {}).get(g, 0) * C_op_DG_sol[g]  for g in GENS) - _o_sf
            sub_sf_sol = -_cred
            capex_sol = (sum(hora0.get("CAP_DG_sol_kW", {}).get(g, 0)* C_inv_DG_sol[g]* frp for g in GENS) - data.get("CAPEX_DG_sol_sin_fallas", 0))
        else:
            capex_sol = 0.0
            opex_sol = 0.0

        if cfg.get("dg_eol", False):
            _o_sf, _cred = _split_sf("eol")
            capex_eol = (sum(hora0.get("CAP_DG_eol_kW", {}).get(g, 0)* C_inv_DG_eol[g]* frp for g in GENS) - data.get("CAPEX_DG_eol_sin_fallas", 0))
            opex_eol = sum(hora0.get("CAP_DG_eol_kW", {}).get(g, 0) * C_op_DG_eol[g]  for g in GENS) - _o_sf
            sub_sf_eol = -_cred
        else:
            capex_eol = 0.0
            opex_eol = 0.0
        capex_fos = sum(hora0.get("CAP_DG_fos_kW", {}).get(g, 0) * C_inv_DG_fos[g] * frp for g in GENS)

        # CAPEX ESS
        capex_ess_5h = sum(hora0.get("CAP_ESS_kWh_5h", {}).get(e, 0)/h_ess_5h * C_inv_ESS_5h[e] * frp for e in ESS)
        capex_ess_4h = sum(hora0.get("CAP_ESS_kWh_4h", {}).get(e, 0)/h_ess_4h * C_inv_ESS_4h[e] * frp for e in ESS)
        capex_ess_6h = sum(hora0.get("CAP_ESS_kWh_6h", {}).get(e, 0)/h_ess_6h * C_inv_ESS_6h[e] * frp for e in ESS)

        # CAPEX MEG
        cap_meg = hora0.get("CAP_MEG_MW", {})
        # Igual que el objetivo: anualidad `frp_meg` si el caso la define, y
        # las primeras `meg_existentes` unidades (flota de la empresa) sin CAPEX.
        _frp_meg = data.get("frp_meg", frp)
        _n_exist = int(cfg.get("meg_existentes", 0) or 0)
        capex_meg = sum(C_inv_MEG * _frp_meg * cap_meg.get(k, 0) for k in cap_meg
                        if int(k) >= _n_exist)
        # MEG arrendado: pago por evento y unidad (incluye combustible).
        arriendo_meg = (data.get("MEG_arriendo", [0.0])[0] * len(hora0.get("arriendos_MEG", []) or [])
                        if cfg.get("meg_arriendo", False) else 0.0)

        # CAPEX MESS
        cap_mess = hora0.get("CAP_MESS_MWh", {})
        # Los optimizadores vigentes exportan "P_MESS_MW"; "P_MESS_MAX_MW" es el
        # nombre antiguo (optimization_base / optimization_3), se mantiene como
        # respaldo para no romper resultados ya guardados.
        P_max_mess = hora0.get("P_MESS_MW") or hora0.get("P_MESS_MAX_MW", {})
        capex_mess = sum(C_inv_mess * frp_movil * cap_mess.get(k, 0) + C_op_mess * frp_movil * P_max_mess.get(k, 0) for k in cap_mess)

        # CAPEX switches
        capex_rcs = (
            sum(hora0.get("y_rcs", {}).get(l, 0) * C_inv_RCS * frp for l in sw_cands_local)
            if cfg.get("inv_switch_rcs", False) else 0
        )
        capex_ms = (
            sum(hora0.get("y_ms", {}).get(l, 0) * C_inv_MS * frp for l in sw_cands_local)
            if cfg.get("inv_switch_ms", False) else 0
        )

        # CAPEX / OPEX enlaces nuevos (medida inv_tie): y_tie es decisión del
        # MILP por enlace candidato, y el costo va por km construido.
        if cfg.get("inv_tie", False):
            _long_km_tie = data.get("longitud_km_linea", {})
            _tie_cands = cfg.get("tie_candidatas", sorted(tie_indices_local))
            capex_tie = sum(hora0.get("y_tie", {}).get(l, 0) * C_inv_TIE * _long_km_tie.get(l, 0) * frp for l in _tie_cands)
            opex_tie  = sum(hora0.get("y_tie", {}).get(l, 0) * C_op_TIE  * _long_km_tie.get(l, 0)       for l in _tie_cands)
        else:
            capex_tie = 0.0
            opex_tie = 0.0

        # CAPEX / OPEX gestión de vegetación / refuerzo / soterramiento —
        # ahora SÍ son salida del MILP: x_veg/x_ref/x_ug es una decisión
        # por línea candidata (sw_cand_list), igual que y_rcs/y_ms. El
        # costo se calcula solo sobre las líneas que el solver eligió
        # tratar, no sobre todo el universo de candidatas.
        longitud_km_linea = data.get("longitud_km_linea", {})
        if cfg.get("veg_mgmt", False):
            capex_veg = sum(hora0.get("x_veg", {}).get(l, 0) * C_inv_veg * longitud_km_linea.get(l, 0) * frp for l in sw_cands_local)
            opex_veg  = sum(hora0.get("x_veg", {}).get(l, 0) * C_op_veg  * longitud_km_linea.get(l, 0)        for l in sw_cands_local)
        else:
            capex_veg = 0.0
            opex_veg = 0.0

        if cfg.get("refuerzo", False):
            capex_ref = sum(hora0.get("x_ref", {}).get(l, 0) * C_inv_ref * longitud_km_linea.get(l, 0) * frp for l in sw_cands_local)
            opex_ref  = sum(hora0.get("x_ref", {}).get(l, 0) * C_op_ref  * longitud_km_linea.get(l, 0)        for l in sw_cands_local)
        else:
            capex_ref = 0.0
            opex_ref = 0.0

        if cfg.get("soterramiento", False):
            capex_ug = sum(hora0.get("x_ug", {}).get(l, 0) * C_inv_ug * longitud_km_linea.get(l, 0) * frp for l in sw_cands_local)
            opex_ug  = sum(hora0.get("x_ug", {}).get(l, 0) * C_op_ug  * longitud_km_linea.get(l, 0)        for l in sw_cands_local)
        else:
            capex_ug = 0.0
            opex_ug = 0.0

        # OPEX DG
        
        opex_fos = sum(sum(_hd(h, "P_DG_fos", g) * (C_comb_DG_fos[g]*0.000242 + C_op_DG_fos[g]) for h in _h_safe) for g in GENS) + sum(hora0.get("CAP_DG_fos_kW", {}).get(g, 0) * C_inv_DG_fos[g] * 0.01 for g in GENS)
        # comb_fos = sum(C_comb_DG_fos[g] * _hd(h, "P_DG_fos", g) * frp for h in _h_safe for g in GENS)

        # OPEX ESS
        opex_ess_5h = sum(hora0.get("CAP_ESS_kWh_5h", {}).get(e, 0)/h_ess_5h * C_op_ESS_5h[e] for e in ESS)
        opex_ess_4h = sum(hora0.get("CAP_ESS_kWh_4h", {}).get(e, 0)/h_ess_4h * C_op_ESS_4h[e] for e in ESS)
        opex_ess_6h = sum(hora0.get("CAP_ESS_kWh_6h", {}).get(e, 0)/h_ess_6h * C_op_ESS_6h[e] for e in ESS)
        # OPEX MEG
        opex_meg = sum(C_op_MEG *  cfg.get("T_tra_meg",{}).get(b,2) * _get_x_re(k, b) for k in cap_meg for b in BUSES)
        comb_meg = sum(C_comb_MEG *0.000242 * sum(_hd(h, "P_MEG").values()) for h in _h_safe)
        # OPEX MESS
        # opex_mess = sum(C_tra_mess * cfg.get("T_tra_mess",{}).get(b,2) for k in cap_mess for b in BUSES)
        faultlinMC = data["faultlinMC"]
        Horastot, num_esc, lineas = faultlinMC.shape
        esc_idx = data.get("esc_esp", 0)  # índice del escenario que estás procesando

        horas_evento = [t for t in range(Horastot) if np.any(faultlinMC[t, esc_idx, :] == 1)]
        bloques_evento = []
        if horas_evento:
            inicio = horas_evento[0]
            fin = horas_evento[0]
            for t in horas_evento[1:]:
                if t > fin + 1:
                    bloques_evento.append((inicio, fin))
                    inicio = fin = t
                else:
                    fin = t
            bloques_evento.append((inicio, fin))

        def _get_u_mess(t0, k, b):
            """
            Lee u_mess[k,b,t] directamente desde horas_data.

            t0 está indexado desde 0, igual que faultlinMC.
            Admite los formatos:
                {(k, b): valor}
                {k: {b: valor}}
                {"k,b": valor}
                {"(k, b)": valor}
            """
            if not 0 <= t0 < len(horas_data):
                return 0.0

            u_data = horas_data[t0].get("u_mess", {})

            if not isinstance(u_data, dict):
                return 0.0

            if (k, b) in u_data:
                return float(u_data.get((k, b), 0.0))

            if k in u_data and isinstance(u_data[k], dict):
                return float(u_data[k].get(b, 0.0))

            return float(
                u_data.get(
                    f"{k},{b}",
                    u_data.get(str((k, b)), 0.0)
                )
            )


        def _get_bus_base_mess(k):
            """
            Permite definir bus_base_mess como:
                {k: bus}
                un bus común
                un set de un único elemento
            """
            base_cfg = cfg.get(
                "bus_base_mess",
                data.get("bus_base_mess", 0)
            )

            if isinstance(base_cfg, dict):
                return base_cfg.get(k, 0)

            if isinstance(base_cfg, set):
                if len(base_cfg) != 1:
                    raise ValueError(
                        "bus_base_mess debe contener una sola base "
                        "cuando está definido como set."
                    )
                return next(iter(base_cfg))

            return base_cfg

        opex_mess=0
        for h_ini, h_fin in bloques_evento:
            for k in cap_mess:

                b_base = _get_bus_base_mess(k)

                for b in BUSES:

                    if b == b_base:
                        continue

                    # Se verifica si el MESS estuvo asignado a este bus
                    # en algún momento del evento.
                    mess_usado_en_evento = max(
                        _get_u_mess(t0, k, b)
                        for t0 in range(h_ini, h_fin + 1)
                        if t0 < len(horas_data)
                    ) if h_ini < len(horas_data) else 0.0

                    if mess_usado_en_evento <= 0.5:
                        continue

                    tiempo_traslado =  cfg.get(
    "T_tra_mess",
    data.get("T_tra_mess", 2.0)
)

                    # Factor 2: viaje desde la base y posterior retorno.
                    opex_mess += (
                        C_tra_mess
                        * 2.0
                        * tiempo_traslado
                        * mess_usado_en_evento
                    )
        # OPEX de maniobra de enlaces — se paga tanto con reconfiguración
        # (enlace existente) como con inversión en enlaces nuevos.
        opex_switch = (
            sum(C_op_SWITCH * _hd(h, "w_tie", l) for h in _h_safe for l in tie_indices_local)
            if (cfg.get("reconfig_topo", False) or cfg.get("inv_tie", False)) else 0
        )

        # EDRP
        costo_edrp = (
            sum(
                C_incentivo_edrp[c] * sum(horas_data[h-1].get("P_EDRP", {}).get((b, c), 0) for b in BUSES)
                for h in _h_safe for c in range(4)
            ) * N_eventos
            if cfg.get("edrp", False) else 0
        )

        # ENS
        ens_cost_base = sum(
            VoLL.get(b, 1.0) * base_esc[h - 1].get("P_LC", {}).get(b, 0)
            for h in _hb_safe for b in BUSES
        ) * N_eventos

        ens_cost_estr = sum(VoLL_local.get(b, 1.0) * _hd(h, "P_LC", b) for h in _h_safe for b in BUSES) * N_eventos if cfg.get("prior_criticos", False) else sum(VoLL.get(b, 1.0) * _hd(h, "P_LC", b) for h in _h_safe for b in BUSES) * N_eventos 

                        
        cost_ahorro_sub = - (data.get("P_SUB_tot", 0) - sum(_hd(h, "P_SUB", s) for h in _h_safe for s in S)) * C_sub.get(0, 1.0)

        # Término NETO de subestación: el crédito bruto respecto del caso base
        # con fallas, menos el que las renovables lograrían sin fallas (las dos
        # mitades que `_split_sf` acaba de separar del OPEX).
        termino_sub = cost_ahorro_sub + sub_sf_sol + sub_sf_eol

        # El MISMO término restringido a las horas con falla. `termino_sub` se
        # calcula sobre las 8.760 h y compara contra un caso sin fallas evaluado
        # a la capacidad óptima SIN fallas, no a la instalada; cuando la
        # estrategia instala más capacidad que esa referencia, el excedente
        # genera también en las horas sanas y su desplazamiento de compra queda
        # acreditado como resiliencia sin serlo. Restringir a las horas con
        # falla elimina esa fuga: en las horas sanas la contribución debería ser
        # nula por construcción, y todo lo que aparezca ahí es operación normal.
        _hf = [h for h in _h_safe
               if h - 1 < _n_hb
               and base_esc[h - 1].get("n°lineas_fuera_servicio", 0) > 0]
        _sf_h = data.get("P_SUB_sf_h")
        _bruto_f = sum(
            sum(base_esc[h - 1].get("P_SUB", {}).get(s, 0) for s in S)
            - sum(_hd(h, "P_SUB", s) for s in S)
            for h in _hf
        )
        _sf_f = 0.0
        for _t, _on in (("sol", cfg.get("dg_sol", False)),
                        ("eol", cfg.get("dg_eol", False))):
            _st = data.get(f"P_SUB_sf_{_t}_h")
            if _on and _sf_h is not None and _st is not None:
                _sf_f += sum(float(_sf_h[h - 1]) - float(_st[h - 1]) for h in _hf
                             if h - 1 < len(_sf_h) and h - 1 < len(_st))
        termino_sub_falla = -(_bruto_f - _sf_f) * _csub

        capex_total = (
            capex_sol + capex_eol + capex_fos +
            capex_ess_5h + capex_ess_4h + capex_ess_6h +
            capex_meg + capex_mess + capex_rcs + capex_ms +
            capex_veg + capex_ref + capex_ug + capex_tie
        )

        opex_total = (opex_sol + opex_eol + opex_fos + opex_ess_5h + opex_ess_4h + opex_ess_6h + opex_meg + comb_meg + arriendo_meg + opex_mess + opex_switch + costo_edrp + opex_veg + opex_ref + opex_ug + opex_tie)

        ahorro_ens = ens_cost_base - ens_cost_estr


        costos.append({
            "capex_anual": capex_total,
            # `opex_anual` se conserva como el TOTAL (OPEX de las medidas mas el
            # termino neto de subestacion), que es lo que entra al costo y lo que
            # consumen la app y el barrido. Las dos mitades van aparte para poder
            # auditarlas: opex_medidas + termino_subestacion == opex_anual.
            "opex_anual": opex_total + termino_sub,
            "opex_medidas": opex_total,
            "termino_subestacion": termino_sub,
            # Versión atribuida a resiliencia: mismo OPEX incremental, pero con
            # el término de subestación contado solo en las horas con falla.
            "termino_subestacion_falla": termino_sub_falla,
            "opex_resiliencia": opex_total + termino_sub_falla,
            "costo_total": capex_total + opex_total + termino_sub,
            "fn_objetivo": funcion_objetivo,
            "costo_ens_base": ens_cost_base,
            "costo_ens_estr": ens_cost_estr,
            "ahorro_ens": ahorro_ens,
            "beneficio": funcion_objetivo,
        })

    if not costos:
        return {
            "capex_anual": 0,
            "opex_anual": 0,
            "opex_medidas": 0,
            "termino_subestacion": 0,
            "termino_subestacion_falla": 0,
            "opex_resiliencia": 0,
            "costo_total": 0,
            "fn_objetivo": 0,
            "costo_ens_base": 0,
            "costo_ens_estr": 0,
            "ahorro_ens": 0,
            "beneficio": 0,
        }

    return {k: np.mean([c[k] for c in costos]) for k in costos[0]}


import numpy as np
import pandas as pd
import plotly.graph_objects as go

""""""
def _transformacion_symlog(valor, umbral_lineal=100_000):
    """
    Transformación simétrica logarítmica.

    Mantiene el signo del valor y permite representar:
    - costos negativos;
    - costo cero;
    - costos positivos.
    """
    valor = np.asarray(valor, dtype=float)

    return (
        np.sign(valor)
        * np.log10(1 + np.abs(valor) / umbral_lineal)
    )


def _formato_usd(valor):
    """Formato compacto para etiquetas y ejes."""
    if pd.isna(valor):
        return "—"

    signo = "-" if valor < 0 else ""
    valor_abs = abs(valor)

    if valor_abs >= 1_000_000:
        return f"{signo}US$ {valor_abs / 1_000_000:,.1f} M"

    if valor_abs >= 1_000:
        return f"{signo}US$ {valor_abs / 1_000:,.0f} mil"

    return f"{signo}US$ {valor_abs:,.0f}"


def _ticks_symlog(minimo, maximo, umbral_lineal):
    """
    Genera marcas del eje Y usando los valores monetarios reales.
    """
    max_abs = max(abs(minimo), abs(maximo))

    niveles = [umbral_lineal]

    while niveles[-1] < max_abs:
        niveles.append(niveles[-1] * 10)

    ticks_reales = []

    # Valores negativos
    for valor in reversed(niveles):
        if minimo <= -valor <= maximo:
            ticks_reales.append(-valor)

    ticks_reales.append(0)

    # Valores positivos
    for valor in niveles:
        if minimo <= valor <= maximo:
            ticks_reales.append(valor)

    ticks_transformados = [
        float(_transformacion_symlog(v, umbral_lineal))
        for v in ticks_reales
    ]

    textos = [
        _formato_usd(v)
        for v in ticks_reales
    ]

    return ticks_transformados, textos
""""""


def construir_curva_abatimiento(df,columna_reduccion,columna_costo_unitario,titulo,titulo_x,titulo_y,columna_medida="Medida",referencia=None,nombre_referencia=None,):
    datos = df.copy()

    # Asegurar que las columnas sean numéricas
    datos[columna_reduccion] = pd.to_numeric(datos[columna_reduccion], errors="coerce")
    datos[columna_costo_unitario] = pd.to_numeric(datos[columna_costo_unitario], errors="coerce")

    # Eliminar divisiones inválidas, infinitos y medidas sin mejora
    datos = datos.replace([np.inf, -np.inf], np.nan)

    datos = datos.dropna(subset=[columna_medida, columna_reduccion, columna_costo_unitario,])

    datos = datos[datos[columna_reduccion] > 0].copy()

    if datos.empty:
        return None, datos

    # Orden típico de una curva MAC:
    # desde el menor costo unitario al mayor
    datos = datos.sort_values(columna_costo_unitario, ascending=True).reset_index(drop=True)

    # Construcción del eje acumulado
    datos["Reducción acumulada final"] = (datos[columna_reduccion].cumsum())
    datos["Reducción acumulada inicial"] = (datos["Reducción acumulada final"] - datos[columna_reduccion])
    datos["Centro barra"] = (datos["Reducción acumulada inicial"] + datos[columna_reduccion] / 2)
    

    # Información mostrada al pasar el cursor
    customdata = np.column_stack([
        datos[columna_medida],
        datos[columna_reduccion],
        datos["Reducción acumulada inicial"],
        datos["Reducción acumulada final"],
        datos[columna_costo_unitario],
    ])

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=datos["Centro barra"],
            y=datos[columna_costo_unitario],
            width=datos[columna_reduccion],
            customdata=customdata,
            name="Medidas",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Reducción individual: %{customdata[1]:,.4f}<br>"
                "Tramo acumulado: "
                "%{customdata[2]:,.4f} – %{customdata[3]:,.4f}<br>"
                "Costo unitario: %{customdata[4]:,.2f}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Línea de costo cero
    fig.add_hline(
        y=0,
        line_width=1
    )

    # Referencia opcional, por ejemplo VoLL en la curva ENS
    if referencia is not None:
        etiqueta = (
            nombre_referencia
            if nombre_referencia is not None
            else "Referencia"
        )

        fig.add_hline(
            y=referencia,
            line_dash="dash",
            annotation_text=f"{etiqueta}: {referencia:,.2f}",
            annotation_position="top right"
        )

    fig.update_layout(
        title=titulo,
        xaxis_title=titulo_x,
        yaxis_title=titulo_y,
        bargap=0,
        hovermode="closest",
        showlegend=False,
        # template="plotly_white",
        height=550,
        margin=dict(l=70, r=40, t=80, b=70),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    fig.update_xaxes(
        rangemode="tozero",
        showgrid=True
    )

    fig.update_yaxes(
        zeroline=True,
        showgrid=True
    )

    return fig, datos


import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _transformacion_symlog(valor, umbral_lineal=100_000):
    """
    Transformación simétrica logarítmica.

    Permite representar simultáneamente:
    - valores negativos;
    - cero;
    - valores positivos.

    El parámetro umbral_lineal controla la compresión visual.
    """
    valor = np.asarray(valor, dtype=float)

    return (
        np.sign(valor)
        * np.log10(1 + np.abs(valor) / umbral_lineal)
    )

def _formato_numero_es(valor, decimales=1):
    """
    Formato español/chileno:
    1.234.567,8
    """
    texto = f"{valor:,.{decimales}f}"

    return (
        texto
        .replace(",", "TEMP")
        .replace(".", ",")
        .replace("TEMP", ".")
    )

def _formato_compacto_usd(valor, decimales=2):
    """
    Formatea valores monetarios usando punto para miles
    y coma para decimales.

    Antepone solo "$" (sin "US"): el "USD"/"US$" completo ya queda
    establecido por el título del eje (p. ej. "[USD/hora]"), así
    que repetirlo en cada etiqueta solo agrega ruido visual.
    """
    if pd.isna(valor):
        return "—"

    signo = "-" if valor < 0 else ""
    valor_abs = abs(valor)

    if valor_abs >= 1_000_000_000:
        numero = _formato_numero_es(
            valor_abs / 1_000_000_000,
            decimales,
        )
        return f"{signo}$ {numero} mil MM"

    if valor_abs >= 1_000_000:
        numero = _formato_numero_es(
            valor_abs / 1_000_000,
            decimales,
        )
        return f"{signo}$ {numero} MM"

    if valor_abs >= 1_000:
        numero = _formato_numero_es(
            valor_abs / 1_000,
            decimales,
        )
        return f"{signo}$ {numero} mil"

    numero = _formato_numero_es(valor_abs, decimales)
    return f"{signo}$ {numero}"

def _generar_ticks_symlog(
    minimo_real,
    maximo_real,
    umbral_lineal=100_000,
):
    """
    Genera marcas legibles para un eje simétrico logarítmico,
    manteniendo etiquetas con los valores monetarios reales.
    """
    max_abs = max(
        abs(float(minimo_real)),
        abs(float(maximo_real)),
        umbral_lineal,
    )

    niveles = []

    # Agrega valores menores al umbral para mejorar la lectura
    candidatos_iniciales = [
        umbral_lineal / 10,
        umbral_lineal,
    ]

    for valor in candidatos_iniciales:
        if valor > 0:
            niveles.append(valor)

    valor_actual = umbral_lineal * 10

    while valor_actual <= max_abs * 10:
        niveles.append(valor_actual)
        valor_actual *= 10

    niveles = sorted(set(niveles))

    ticks_reales = []

    # Valores negativos, desde el más grande al más cercano a cero
    for valor in reversed(niveles):
        if minimo_real <= -valor <= maximo_real:
            ticks_reales.append(-valor)

    # Siempre incluir cero
    ticks_reales.append(0.0)

    # Valores positivos
    for valor in niveles:
        if minimo_real <= valor <= maximo_real:
            ticks_reales.append(valor)

    # Incluir extremos cuando no coinciden con los ticks estándar
    if minimo_real < 0:
        ticks_reales.append(float(minimo_real))

    if maximo_real > 0:
        ticks_reales.append(float(maximo_real))

    ticks_reales = sorted(set(ticks_reales))

    ticks_transformados = [
        float(
            _transformacion_symlog(
                valor,
                umbral_lineal=umbral_lineal,
            )
        )
        for valor in ticks_reales
    ]

    etiquetas = [
        _formato_compacto_usd(valor)
        for valor in ticks_reales
    ]

    return ticks_transformados, etiquetas


def construir_curva_abatimiento_3(
    df,
    columna_reduccion,
    columna_costo_unitario,
    titulo,
    titulo_x,
    titulo_y,
    columna_medida="Estrategia",
    columna_costo_total=None,
    referencia=None,
    nombre_referencia=None,
    escala_y="symlog",
    umbral_lineal=100_000,
    mostrar_etiquetas=True,
    altura=650,
):
    """
    Construye una curva de costos de abatimiento.

    Parámetros
    ----------
    df : pandas.DataFrame
        Tabla con las estrategias y sus indicadores.

    columna_reduccion : str
        Columna que contiene la mejora física.
        Ejemplo: "Δ ENS [MWh]".

    columna_costo_unitario : str
        Columna que contiene el costo por unidad de mejora.
        Ejemplo: "Costo por ENS evitada [USD/MWh]".

    titulo : str
        Título del gráfico.

    titulo_x : str
        Título del eje X.

    titulo_y : str
        Título del eje Y.

    columna_medida : str
        Nombre de la columna con las estrategias.
        Si está como índice, se convierte automáticamente en columna.

    columna_costo_total : str o None
        Columna opcional con el costo total de la medida para mostrar
        en el cursor.

    referencia : float o None
        Línea horizontal opcional, por ejemplo VoLL.

    nombre_referencia : str o None
        Nombre de la línea de referencia.

    escala_y : str
        "symlog" para escala simétrica logarítmica.
        "linear" para escala lineal normal.

    umbral_lineal : float
        Parámetro de compresión de la escala symlog.

    mostrar_etiquetas : bool
        Muestra el costo unitario sobre las barras.

    altura : int
        Altura del gráfico en píxeles.

    Retorna
    -------
    fig : plotly.graph_objects.Figure o None
    datos : pandas.DataFrame
        Tabla ordenada utilizada para construir la curva.
    """

    datos = df.copy()

    # -------------------------------------------------
    # Recuperar Estrategia si está almacenada como índice
    # -------------------------------------------------
    if (
        columna_medida not in datos.columns
        and datos.index.name == columna_medida
    ):
        datos = datos.reset_index()

    # -------------------------------------------------
    # Validar columnas requeridas
    # -------------------------------------------------
    columnas_requeridas = [
        columna_medida,
        columna_reduccion,
        columna_costo_unitario,
    ]

    if columna_costo_total is not None:
        columnas_requeridas.append(columna_costo_total)

    columnas_faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in datos.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas para construir la curva MAC: "
            f"{columnas_faltantes}. "
            f"Columnas disponibles: {datos.columns.tolist()}. "
            f"Nombre del índice: {datos.index.name}"
        )

    # -------------------------------------------------
    # Convertir columnas numéricas
    # -------------------------------------------------
    datos[columna_reduccion] = pd.to_numeric(
        datos[columna_reduccion],
        errors="coerce",
    )

    datos[columna_costo_unitario] = pd.to_numeric(
        datos[columna_costo_unitario],
        errors="coerce",
    )

    if columna_costo_total is not None:
        datos[columna_costo_total] = pd.to_numeric(
            datos[columna_costo_total],
            errors="coerce",
        )

    datos = datos.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # -------------------------------------------------
    # Eliminar filas inválidas
    # -------------------------------------------------
    datos = datos.dropna(
        subset=[
            columna_medida,
            columna_reduccion,
            columna_costo_unitario,
        ]
    )

    # Solo medidas que mejoran efectivamente el indicador
    datos = datos[
        datos[columna_reduccion] > 0
    ].copy()

    if datos.empty:
        return None, datos

    # -------------------------------------------------
    # Ordenar desde menor a mayor costo unitario
    # -------------------------------------------------
    datos = datos.sort_values(
        columna_costo_unitario,
        ascending=True,
    ).reset_index(drop=True)

    # -------------------------------------------------
    # Construcción del eje acumulado indicativo
    # -------------------------------------------------
    datos["Reducción acumulada final"] = (
        datos[columna_reduccion].cumsum()
    )

    datos["Reducción acumulada inicial"] = (
        datos["Reducción acumulada final"]
        - datos[columna_reduccion]
    )

    datos["Centro barra"] = (
        datos["Reducción acumulada inicial"]
        + datos[columna_reduccion] / 2
    )
    datos["Color"] = datos[columna_medida].apply(
    obtener_color_estrategia
)

    # Mantener costo original
    datos["Costo unitario real"] = (
        datos[columna_costo_unitario]
    )

    # -------------------------------------------------
    # Transformación del eje vertical
    # -------------------------------------------------
    if escala_y == "symlog":
        datos["Costo graficado"] = _transformacion_symlog(
            datos["Costo unitario real"],
            umbral_lineal=umbral_lineal,
        )
    elif escala_y == "linear":
        datos["Costo graficado"] = (
            datos["Costo unitario real"]
        )
    else:
        raise ValueError(
            "escala_y debe ser 'symlog' o 'linear'."
        )

    datos["Etiqueta costo"] = datos[
        "Costo unitario real"
    ].apply(_formato_compacto_usd)

    # -------------------------------------------------
    # Información del cursor
    # -------------------------------------------------
    if columna_costo_total is not None:
        customdata = np.column_stack([
            datos[columna_medida].astype(str),
            datos[columna_reduccion],
            datos["Reducción acumulada inicial"],
            datos["Reducción acumulada final"],
            datos["Costo unitario real"],
            datos[columna_costo_total],
        ])

        hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "Reducción individual: %{customdata[1]:,.4f}<br>"
            "Tramo indicativo: "
            "%{customdata[2]:,.4f} – %{customdata[3]:,.4f}<br>"
            "Costo unitario real: US$ %{customdata[4]:,.2f}<br>"
            "Costo neto medida: US$ %{customdata[5]:,.0f}<br>"
            "<extra></extra>"
        )

    else:
        customdata = np.column_stack([
            datos[columna_medida].astype(str),
            datos[columna_reduccion],
            datos["Reducción acumulada inicial"],
            datos["Reducción acumulada final"],
            datos["Costo unitario real"],
        ])

        hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "Reducción individual: %{customdata[1]:,.4f}<br>"
            "Tramo indicativo: "
            "%{customdata[2]:,.4f} – %{customdata[3]:,.4f}<br>"
            "Costo unitario real: US$ %{customdata[4]:,.2f}<br>"
            "<extra></extra>"
        )

    # -------------------------------------------------
    # Construcción del gráfico
    # -------------------------------------------------
    fig = go.Figure()
   

    for _, fila in datos.iterrows():

        if columna_costo_total is not None:
            customdata_fila = [[
                str(fila[columna_medida]),
                fila[columna_reduccion],
                fila["Reducción acumulada inicial"],
                fila["Reducción acumulada final"],
                fila["Costo unitario real"],
                fila[columna_costo_total],
            ]]

            hovertemplate_fila = (
                "<b>%{customdata[0]}</b><br>"
                "Reducción individual: %{customdata[1]:,.4f}<br>"
                "Tramo indicativo: "
                "%{customdata[2]:,.4f} – %{customdata[3]:,.4f}<br>"
                "Costo unitario real: US$ %{customdata[4]:,.2f}<br>"
                "Costo neto medida: US$ %{customdata[5]:,.0f}<br>"
                "<extra></extra>"
            )

        else:
            customdata_fila = [[
                str(fila[columna_medida]),
                fila[columna_reduccion],
                fila["Reducción acumulada inicial"],
                fila["Reducción acumulada final"],
                fila["Costo unitario real"],
            ]]

            hovertemplate_fila = (
                "<b>%{customdata[0]}</b><br>"
                "Reducción individual: %{customdata[1]:,.4f}<br>"
                "Tramo indicativo: "
                "%{customdata[2]:,.4f} – %{customdata[3]:,.4f}<br>"
                "Costo unitario real: US$ %{customdata[4]:,.2f}<br>"
                "<extra></extra>"
            )

    fig.add_trace(
        go.Bar(
            x=[fila["Centro barra"]],
            y=[fila["Costo graficado"]],
            width=[fila[columna_reduccion]],
            name=str(fila[columna_medida]),
            marker_color=fila["Color"],
            customdata=customdata_fila,
            hovertemplate=hovertemplate_fila,
            text=(
                [fila["Etiqueta costo"]]
                if mostrar_etiquetas
                else None
            ),
            textposition=(
                "outside"
                if mostrar_etiquetas
                else None
            ),
            cliponaxis=False,
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Bar(
            x=datos["Centro barra"],
            y=datos["Costo graficado"],
            width=datos[columna_reduccion],
            customdata=customdata,
            text=(
                datos["Etiqueta costo"]
                if mostrar_etiquetas
                else None
            ),
            textposition=(
                "outside"
                if mostrar_etiquetas
                else None
            ),
            cliponaxis=False,
            hovertemplate=hovertemplate,
            name="Estrategias",
            marker_color=datos["Color"],
        )
    )

    # Línea de costo cero
    fig.add_hline(
        y=0,
        line_width=1.5,
        annotation_text="Costo cero",
        annotation_position="bottom right",
    )
    fig.update_layout(
    separators=",.",
)

    # -------------------------------------------------
    # Línea de referencia opcional
    # -------------------------------------------------
    if referencia is not None:
        referencia = float(referencia)

        if escala_y == "symlog":
            referencia_graficada = float(
                _transformacion_symlog(
                    referencia,
                    umbral_lineal=umbral_lineal,
                )
            )
        else:
            referencia_graficada = referencia

        etiqueta_referencia = (
            nombre_referencia
            if nombre_referencia is not None
            else "Referencia"
        )

        fig.add_hline(
            y=referencia_graficada,
            line_dash="dash",
            line_width=1.5,
            annotation_text=(
                f"{etiqueta_referencia}: "
                f"{_formato_compacto_usd(referencia)}"
            ),
            annotation_position="top right",
        )

    # -------------------------------------------------
    # Configuración del eje Y
    # -------------------------------------------------
    if escala_y == "symlog":
        minimo_real = datos["Costo unitario real"].min()
        maximo_real = datos["Costo unitario real"].max()

        # Considerar referencia en los límites
        if referencia is not None:
            minimo_real = min(minimo_real, referencia)
            maximo_real = max(maximo_real, referencia)

        tickvals, ticktext = _generar_ticks_symlog(
            minimo_real=minimo_real,
            maximo_real=maximo_real,
            umbral_lineal=umbral_lineal,
        )

        fig.update_yaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            title_text=(
                f"{titulo_y}<br>"
                "<sup>Escala simétrica logarítmica</sup>"
            ),
            zeroline=False,
            showgrid=True,
        )

    else:
        fig.update_yaxes(
            title_text=titulo_y,
            tickformat=",.0f",
            zeroline=False,
            showgrid=True,
        )

    # -------------------------------------------------
    # Diseño general
    # -------------------------------------------------
    fig.update_layout(
        title={
            "text": titulo,
            "x": 0.02,
            "xanchor": "left",
        },
        xaxis_title=titulo_x,
        bargap=0,
        hovermode="closest",
        showlegend=False,
        # template="plotly_white",
        height=altura,
        margin=dict(
            l=125,
            r=70,
            t=100,
            b=90,
        ),
        uniformtext_minsize=9,
        uniformtext_mode="hide",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"

    )

    fig.update_xaxes(
        rangemode="tozero",
        showgrid=True,
        tickformat=",.2f",
    )

    return fig, datos



def construir_curva_abatimiento_2(
    df,
    columna_reduccion,
    columna_costo_unitario,
    titulo,
    titulo_x,
    titulo_y,
    columna_medida="Estrategia",
    columna_costo_total=None,
    referencia=None,
    nombre_referencia=None,
    escala_y="symlog",
    umbral_lineal=100_000,
    mostrar_etiquetas=True,
    incluir_variaciones_negativas=False,

    # Filtra variaciones cuyo valor absoluto no supera este umbral:
    # ruido numérico de la optimización (p. ej. 0,001 en vez de 0
    # exacto) que de otro modo aparecería como una barra falsa.
    umbral_ruido=0.0,

    # Lista en la leyenda (sin dibujar barra) las estrategias que
    # quedaron fuera de la curva por no mejorar el indicador.
    mostrar_excluidas_en_leyenda=True,

    # Dimensiones del gráfico
    ancho=1500,
    altura=850,

    # Tamaños de fuente
    fuente_general=18,
    fuente_titulo=26,
    fuente_ejes=20,
    fuente_ticks=17,
    fuente_leyenda=16,
    fuente_etiquetas=15,
    fuente_anotaciones=15,

    # Exportación opcional
    ruta_exportacion=None,
    escala_exportacion=2,
):
    """
    Construye una curva de costos marginales de abatimiento.

    El ancho de cada barra representa la magnitud de la variación
    del indicador, mientras que la altura representa el costo unitario.

    Por defecto, solo se incluyen reducciones positivas. Cuando
    incluir_variaciones_negativas=True:

        - las mejoras se ubican a la derecha de cero;
        - los deterioros se ubican a la izquierda de cero;
        - el ancho de cada barra utiliza el valor absoluto de la variación.

    Los colores se asignan mediante obtener_color_estrategia(), por lo
    que cada tipo de estrategia conserva su color aunque cambie el orden.

    Las estrategias que no mejoran el indicador (variación nula o
    negativa) no pueden dibujarse como barra, pero sí se listan en la
    leyenda cuando mostrar_excluidas_en_leyenda=True, para que su
    ausencia sea explícita y no se confunda con no haber sido evaluadas.

    Parámetros adicionales
    ----------------------
    ancho, altura : int
        Dimensiones base del gráfico en píxeles.

    mostrar_excluidas_en_leyenda : bool
        Agrega entradas de leyenda (sin barra asociada) para las
        estrategias sin efecto sobre el indicador o que lo empeoran.

    ruta_exportacion : str o None
        Ruta opcional para exportar la figura, por ejemplo:
        "curva_mac_ens.png".

    escala_exportacion : int o float
        Multiplicador de resolución. Por ejemplo, ancho=1500,
        altura=850 y escala_exportacion=2 genera aproximadamente
        3000 x 1700 píxeles.

    Retorna
    -------
    fig : plotly.graph_objects.Figure o None
    datos : pandas.DataFrame
        Datos procesados utilizados para construir el gráfico.
    """

    datos = df.copy()

    # =========================================================
    # 1. RECUPERAR LA COLUMNA DE ESTRATEGIA DESDE EL ÍNDICE
    # =========================================================

    if columna_medida not in datos.columns:

        if datos.index.name == columna_medida:
            datos = datos.reset_index()

        elif datos.index.name is not None:
            datos = datos.reset_index()

            if columna_medida not in datos.columns:
                posible_indice = datos.columns[0]

                datos = datos.rename(
                    columns={
                        posible_indice: columna_medida
                    }
                )

    # =========================================================
    # 2. VALIDAR COLUMNAS REQUERIDAS
    # =========================================================

    columnas_requeridas = [
        columna_medida,
        columna_reduccion,
        columna_costo_unitario,
    ]

    if columna_costo_total is not None:
        columnas_requeridas.append(
            columna_costo_total
        )

    columnas_faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in datos.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas para construir la curva MAC: "
            f"{columnas_faltantes}. "
            f"Columnas disponibles: {datos.columns.tolist()}. "
            f"Nombre del índice: {datos.index.name}"
        )

    # =========================================================
    # 3. CONVERTIR COLUMNAS NUMÉRICAS
    # =========================================================

    datos[columna_reduccion] = pd.to_numeric(
        datos[columna_reduccion],
        errors="coerce",
    )

    datos[columna_costo_unitario] = pd.to_numeric(
        datos[columna_costo_unitario],
        errors="coerce",
    )

    if columna_costo_total is not None:
        datos[columna_costo_total] = pd.to_numeric(
            datos[columna_costo_total],
            errors="coerce",
        )

    datos = datos.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Primero se eliminan solo filas sin estrategia
    # o sin variación física.
    datos = datos.dropna(
        subset=[
            columna_medida,
            columna_reduccion,
        ]
    )

    # =========================================================
    # 4. SELECCIONAR VARIACIONES
    # =========================================================

    # Copia previa al filtro. Las filas que se descarten aquí siguen
    # siendo estrategias evaluadas: se recuperan más abajo para
    # listarlas en la leyenda.
    datos_evaluados = datos.copy()

    if incluir_variaciones_negativas:

        # Se mantienen mejoras y deterioros.
        # Se excluye cero (y ruido numérico cercano a cero, según
        # umbral_ruido) porque produciría una barra sin ancho real.
        datos = datos[
            datos[columna_reduccion].abs() > umbral_ruido
        ].copy()

    else:

        # Curva MAC convencional: solo mejoras reales (por encima
        # del umbral de ruido numérico).
        datos = datos[
            datos[columna_reduccion] > umbral_ruido
        ].copy()

    if datos.empty:
        return None, datos

    # =========================================================
    # 5. RECUPERAR COSTOS UNITARIOS FALTANTES
    # =========================================================

    if columna_costo_total is not None:

        mascara_recalculable = (
            datos[columna_costo_unitario].isna()
            & datos[columna_costo_total].notna()
            & datos[columna_reduccion].ne(0)
        )

        datos.loc[
            mascara_recalculable,
            columna_costo_unitario,
        ] = (
            datos.loc[
                mascara_recalculable,
                columna_costo_total,
            ]
            / datos.loc[
                mascara_recalculable,
                columna_reduccion,
            ].abs()
        )

    # Eliminar únicamente las filas cuyo costo unitario
    # sigue sin poder calcularse.
    datos = datos.dropna(
        subset=[
            columna_costo_unitario,
        ]
    )

    if datos.empty:
        return None, datos

    # =========================================================
    # 6. ORDENAR POR COSTO UNITARIO
    # =========================================================

    datos = datos.sort_values(
        columna_costo_unitario,
        ascending=True,
    ).reset_index(drop=True)

    # =========================================================
    # 7. IDENTIFICAR MEJORAS Y DETERIOROS
    # =========================================================

    datos["Tipo variación"] = np.where(
        datos[columna_reduccion] > 0,
        "Mejora",
        "Deterioro",
    )

    # Plotly requiere anchos positivos.
    datos["Ancho barra"] = (
        datos[columna_reduccion].abs()
    )

    # =========================================================
    # 8. CONSTRUIR EL EJE X ACUMULADO
    # =========================================================

    datos["Reducción acumulada inicial"] = np.nan
    datos["Reducción acumulada final"] = np.nan

    mascara_positiva = (
        datos[columna_reduccion] > 0
    )

    if mascara_positiva.any():

        reducciones_positivas = datos.loc[
            mascara_positiva,
            columna_reduccion,
        ]

        acumulado_positivo = (
            reducciones_positivas.cumsum()
        )

        datos.loc[
            mascara_positiva,
            "Reducción acumulada final",
        ] = acumulado_positivo

        datos.loc[
            mascara_positiva,
            "Reducción acumulada inicial",
        ] = (
            acumulado_positivo
            - reducciones_positivas
        )

    mascara_negativa = (
        datos[columna_reduccion] < 0
    )

    if mascara_negativa.any():

        magnitudes_negativas = datos.loc[
            mascara_negativa,
            columna_reduccion,
        ].abs()

        acumulado_negativo = (
            magnitudes_negativas.cumsum()
        )

        # La primera barra negativa ocupa, por ejemplo,
        # el tramo -5 a 0.
        datos.loc[
            mascara_negativa,
            "Reducción acumulada inicial",
        ] = -acumulado_negativo

        datos.loc[
            mascara_negativa,
            "Reducción acumulada final",
        ] = -(
            acumulado_negativo
            - magnitudes_negativas
        )

    datos["Centro barra"] = (
        datos["Reducción acumulada inicial"]
        + datos["Reducción acumulada final"]
    ) / 2

    # =========================================================
    # 9. ASIGNAR COLORES POR TIPO DE ESTRATEGIA
    # =========================================================

    datos["Color"] = (
        datos[columna_medida]
        .apply(obtener_color_estrategia)
    )

    # =========================================================
    # 10. PREPARAR COSTOS Y ETIQUETAS
    # =========================================================

    datos["Costo unitario real"] = (
        datos[columna_costo_unitario]
    )

    if escala_y == "symlog":

        datos["Costo graficado"] = (
            _transformacion_symlog(
                datos["Costo unitario real"],
                umbral_lineal=umbral_lineal,
            )
        )

    elif escala_y == "linear":

        datos["Costo graficado"] = (
            datos["Costo unitario real"]
        )

    else:
        raise ValueError(
            "escala_y debe ser 'symlog' o 'linear'."
        )

    datos["Etiqueta costo"] = (
        datos["Costo unitario real"]
        .apply(_formato_compacto_usd)
    )

    # =========================================================
    # 11. CONSTRUIR EL GRÁFICO
    # =========================================================

    fig = go.Figure()

    # Evita entradas repetidas en la leyenda si eventualmente
    # existe más de una fila con el mismo nombre.
    nombres_leyenda = set()

    for _, fila in datos.iterrows():

        # Traducido solo para mostrar (leyenda/hover/nombre de
        # barra); el "Estrategia" original de fila[columna_medida]
        # sigue intacto para el resto de la app.
        nombre_estrategia = _traducir_nombre_estrategia_display(
            fila[columna_medida]
        )

        mostrar_en_leyenda = (
            nombre_estrategia
            not in nombres_leyenda
        )

        nombres_leyenda.add(
            nombre_estrategia
        )

        if columna_costo_total is not None:

            customdata_fila = [[
                nombre_estrategia,
                float(fila[columna_reduccion]),
                str(fila["Tipo variación"]),
                float(
                    fila[
                        "Reducción acumulada inicial"
                    ]
                ),
                float(
                    fila[
                        "Reducción acumulada final"
                    ]
                ),
                float(
                    fila["Costo unitario real"]
                ),
                float(
                    fila[columna_costo_total]
                ),
            ]]

            hovertemplate = (
                "<b>%{customdata[0]}</b><br>"
                "Variación individual: "
                "%{customdata[1]:,.4f}<br>"
                "Resultado: %{customdata[2]}<br>"
                "Tramo indicativo: "
                "%{customdata[3]:,.4f} – "
                "%{customdata[4]:,.4f}<br>"
                "Costo unitario real: "
                "US$ %{customdata[5]:,.2f}<br>"
                "Costo neto medida: "
                "US$ %{customdata[6]:,.0f}<br>"
                "<extra></extra>"
            )

        else:

            customdata_fila = [[
                nombre_estrategia,
                float(fila[columna_reduccion]),
                str(fila["Tipo variación"]),
                float(
                    fila[
                        "Reducción acumulada inicial"
                    ]
                ),
                float(
                    fila[
                        "Reducción acumulada final"
                    ]
                ),
                float(
                    fila["Costo unitario real"]
                ),
            ]]

            hovertemplate = (
                "<b>%{customdata[0]}</b><br>"
                "Variación individual: "
                "%{customdata[1]:,.4f}<br>"
                "Resultado: %{customdata[2]}<br>"
                "Tramo indicativo: "
                "%{customdata[3]:,.4f} – "
                "%{customdata[4]:,.4f}<br>"
                "Costo unitario real: "
                "US$ %{customdata[5]:,.2f}<br>"
                "<extra></extra>"
            )

        fig.add_trace(
            go.Bar(
                x=[
                    fila["Centro barra"]
                ],

                y=[
                    fila["Costo graficado"]
                ],

                width=[
                    fila["Ancho barra"]
                ],

                name=nombre_estrategia,

                legendgroup=nombre_estrategia,

                showlegend=mostrar_en_leyenda,

                marker=dict(
                    color=fila["Color"],
                    line=dict(
                        color="white",
                        width=0.8,
                    ),
                ),

                customdata=customdata_fila,

                hovertemplate=hovertemplate,

                text=(
                    [fila["Etiqueta costo"]]
                    if mostrar_etiquetas
                    else None
                ),

                textposition=(
                    "outside"
                    if mostrar_etiquetas
                    else None
                ),

                textfont=dict(
                    size=fuente_etiquetas,
                ),

                # Por defecto Plotly encoge/oculta el texto "outside"
                # para que quepa dentro del ancho de la barra (aunque
                # esté afuera de ella). Eso era lo que hacía
                # desaparecer el costo en las barras angostas.
                # "none" lo deja siempre a su tamaño normal.
                constraintext="none",

                cliponaxis=False,
            )
        )

    # =========================================================
    # 11b. ESTRATEGIAS SIN BARRA EN LA CURVA
    # =========================================================

    # Una estrategia que no mejora el indicador no tiene ancho de
    # barra que dibujar, pero omitirla del todo hace parecer que no
    # se evaluó. Se agrega como entrada de leyenda sin datos.

    if mostrar_excluidas_en_leyenda:

        # "Δ SAIFI" -> "SAIFI", para redactar la etiqueta.
        nombre_indicador = (
            columna_reduccion
            .replace("Δ", "")
            .strip()
        )

        variacion_evaluada = (
            datos_evaluados[columna_reduccion]
        )

        # Sin efecto: dentro del umbral de ruido numérico.
        excluidas = [
            (fila[columna_medida], "sin efecto sobre")
            for _, fila in datos_evaluados[
                variacion_evaluada.abs() <= umbral_ruido
            ].iterrows()
        ]

        # Deterioro: solo quedan fuera de la curva cuando no se
        # grafican las variaciones negativas.
        if not incluir_variaciones_negativas:

            excluidas += [
                (fila[columna_medida], "aumenta")
                for _, fila in datos_evaluados[
                    variacion_evaluada < -umbral_ruido
                ].iterrows()
            ]

        for medida, efecto in excluidas:

            nombre_estrategia = (
                _traducir_nombre_estrategia_display(
                    medida
                )
            )

            if nombre_estrategia in nombres_leyenda:
                continue

            nombres_leyenda.add(
                nombre_estrategia
            )

            fig.add_trace(
                go.Bar(
                    x=[None],
                    y=[None],

                    name=(
                        f"{nombre_estrategia} "
                        f"({efecto} {nombre_indicador})"
                    ),

                    legendgroup=nombre_estrategia,

                    showlegend=True,

                    marker=dict(
                        color=obtener_color_estrategia(
                            medida
                        ),

                        opacity=0.35,

                        line=dict(
                            color="grey",
                            width=1.2,
                        ),
                    ),

                    hoverinfo="skip",
                )
            )

    # =========================================================
    # 12. LÍNEAS DE REFERENCIA
    # =========================================================

    fig.add_hline(
        y=0,
        line_width=1.5,
        line_color="black",
        annotation_text="Costo cero",
        annotation_position="bottom right",
    )

    if (
        incluir_variaciones_negativas
        and mascara_negativa.any()
    ):
        fig.add_vline(
            x=0,
            line_width=1.3,
            line_dash="dot",
            line_color="black",
        )

    if referencia is not None:

        referencia = float(referencia)

        if escala_y == "symlog":

            referencia_graficada = float(
                _transformacion_symlog(
                    referencia,
                    umbral_lineal=umbral_lineal,
                )
            )

        else:

            referencia_graficada = referencia

        etiqueta_referencia = (
            nombre_referencia
            if nombre_referencia is not None
            else "Referencia"
        )

        fig.add_hline(
            y=referencia_graficada,
            line_dash="dash",
            line_width=1.7,
            annotation_text=(
                f"{etiqueta_referencia}: "
                f"{_formato_compacto_usd(referencia)}"
            ),
            annotation_position="top right",
        )

    # =========================================================
    # 13. CONFIGURAR EJE Y
    # =========================================================

    if escala_y == "symlog":

        minimo_real = float(
            datos["Costo unitario real"].min()
        )

        maximo_real = float(
            datos["Costo unitario real"].max()
        )

        if referencia is not None:

            minimo_real = min(
                minimo_real,
                referencia,
            )

            maximo_real = max(
                maximo_real,
                referencia,
            )

        tickvals, ticktext = (
            _generar_ticks_symlog(
                minimo_real=minimo_real,
                maximo_real=maximo_real,
                umbral_lineal=umbral_lineal,
            )
        )

        fig.update_yaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,

            title_text=(
                f"{titulo_y}<br>"
                "<sup>"
                "Escala simétrica logarítmica"
                "</sup>"
            ),

            title_font=dict(
                size=fuente_ejes,
            ),

            tickfont=dict(
                size=fuente_ticks,
            ),

            zeroline=False,
            showgrid=True,
        )

    else:

        fig.update_yaxes(
            title_text=titulo_y,
            tickformat=",.0f",

            title_font=dict(
                size=fuente_ejes,
            ),

            tickfont=dict(
                size=fuente_ticks,
            ),

            zeroline=False,
            showgrid=True,
        )

    # =========================================================
    # 14. TÍTULO DEL EJE X
    # =========================================================

    titulo_x_final = titulo_x

    if (
        incluir_variaciones_negativas
        and mascara_negativa.any()
    ):
        titulo_x_final = (
            f"{titulo_x}<br>"
            "<sup>"
            "Izquierda: deterioro del indicador | "
            "Derecha: mejora del indicador"
            "</sup>"
        )

    # =========================================================
    # 15. DISEÑO GENERAL
    # =========================================================

    fig.update_layout(
        title=dict(
            text=titulo,
            x=0.02,
            xanchor="left",
            font=dict(
                size=fuente_titulo,
            ),
        ),

        width=ancho,
        height=altura,

        font=dict(
            family="Arial",
            size=fuente_general,
        ),

        xaxis_title=titulo_x_final,

        barmode="overlay",
        bargap=0,

        hovermode="closest",

        showlegend=True,

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        margin=dict(
            l=165,
            r=80,
            # Un poco más que el original (125) para que la etiqueta
            # de costo de la barra más alta no quede pegada/cortada
            # contra el borde superior.
            t=140,
            b=220,
        ),

        legend=dict(
            title_text=None,

            orientation="h",

            yanchor="top",
            y=-0.23,

            xanchor="center",
            x=0.5,

            traceorder="normal",

            font=dict(
                size=fuente_leyenda,
            ),

            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),

        hoverlabel=dict(
            font=dict(
                family="Arial",
                size=fuente_general,
            ),
        ),

        separators=",.",

        # "hide" ocultaba por completo la etiqueta de costo en las barras
        # angostas (no les entraba el texto horizontal). Con las etiquetas
        # rotadas (textangle=-90) el ancho requerido es mínimo, así que
        # desactivamos el autoescalado/ocultamiento uniforme y dejamos
        # que todas las barras muestren su costo.
        uniformtext_mode=False,
    )

    fig.update_xaxes(
        showgrid=True,
        tickformat=",.2f",

        title_font=dict(
            size=fuente_ejes,
        ),

        tickfont=dict(
            size=fuente_ticks,
        ),
    )

    fig.update_annotations(
        font=dict(
            size=fuente_anotaciones,
        )
    )

    # Ajustar explícitamente el rango para incluir
    # correctamente valores negativos y positivos.
    minimo_x = float(
        datos["Reducción acumulada inicial"].min()
    )

    maximo_x = float(
        datos["Reducción acumulada final"].max()
    )

    rango_total = maximo_x - minimo_x

    margen_x = (
        rango_total * 0.03
        if rango_total > 0
        else 1.0
    )

    fig.update_xaxes(
        range=[
            minimo_x - margen_x,
            maximo_x + margen_x,
        ]
    )

    # # =========================================================
    # # 16. EXPORTACIÓN OPCIONAL EN ALTA RESOLUCIÓN
    # # =========================================================

    # if ruta_exportacion is not None:

    #     try:
    #         fig.write_image(
    #             ruta_exportacion,
    #             width=ancho,
    #             height=altura,
    #             scale=escala_exportacion,
    #         )

    #     except Exception as error:
    #         raise RuntimeError(
    #             "No fue posible exportar la figura. "
    #             "Verifica que Kaleido esté instalado mediante: "
    #             "'pip install -U kaleido'. "
    #             f"Error original: {error}"
    #         ) from error

    return fig, datos