# optimizador.py
import numpy as np
# from abc.mi_proyecto.app.caso_base import P_sub_total
import pyomo.environ as pyo
import pickle
import os

from aislacion import (
    adyacencia_lineas, expandir_dano, leer_config as leer_config_aislacion,
)
from limites_dg import (
    leer_config as leer_config_limites_dg,
    aplicar_tope as aplicar_tope_dg,
    aplicar_tope_conjunto as aplicar_tope_dg_conjunto,
)
from ramales import piso_corte as piso_corte_ramales


# Lo único que este módulo resuelve: el caso base, con DG como referencia de
# inversión para E4. Las demás medidas se evalúan con `optimization_6_oficial`.
MEDIDAS_PERMITIDAS = ("dg_sol", "dg_eol", "dg_fos")

# Medidas que el archivo todavía sabe armar, pero que nadie usa por acá. El
# código se conserva a propósito: `aislacion.py` y `limites_dg.py` dependen de
# que los dos modelos se puedan leer lado a lado, porque el ahorro de cada
# medida se mide contra la física del caso base y las dos tienen que coincidir.
MEDIDAS_NO_SOPORTADAS = (
    "reconfig_topo", "inv_tie", "inv_switch_rcs", "inv_switch_ms",
    "ess_4h", "ess_5h", "ess_6h", "meg", "mess", "edrp",
    "refuerzo", "soterramiento",
)


def _exigir_alcance(config):
    """Rechaza medidas fuera del alcance del caso base, en vez de armar vacío.

    Sin esto, pedir `ess_4h=True` acá construía el modelo igual y devolvía un
    caso base SIN ESS, sin una sola advertencia: un resultado creíble y
    equivocado, que es la peor forma de fallar. Las familias de variables de
    esas medidas se crean vacías (ver más abajo), así que el silencio era total.
    """
    pedidas = [m for m in MEDIDAS_NO_SOPORTADAS if config.get(m)]
    if pedidas:
        raise ValueError(
            f"optimization_base resuelve el caso base y solo admite "
            f"{', '.join(MEDIDAS_PERMITIDAS)}; se pidió {', '.join(pedidas)}. "
            f"Para evaluar esas medidas va optimization_6_oficial "
            f"(run_barrido_medidas.py).")


def resolver_optimizacion_base(data, config):
    _exigir_alcance(config)

    _cache_file = config.get("cache_file", None)
    if _cache_file and os.path.exists(_cache_file):
        with open(_cache_file, "rb") as _f:
            _cached = pickle.load(_f)
        return _cached["results_por_esc"]

    net           = data["net"]
    # snet          = data["snet"]
    faultlinMC    = data["faultlinMC"]
    repairMC      = data["repairMC"]
    alphaDGsolar  = data["alphaDGsolar"]
    alphaDGeolica = data["alphaDGeolica"]
    demand_factor = data["demand_factor"]
    N_eventos     = data["N_eventos"]
    _n_horas = config.get("horas", 24)
    horas = range(_n_horas) if isinstance(_n_horas, int) else range(len(_n_horas))
    Horastot = faultlinMC.shape[0]
    num_esc  = config.get("numesc", 1)
    esc_sel = config.get("esc_debug", None)
    VoLL      = data["VoLL"]
    VoLL_local = dict(VoLL)
    sw_cand_list = data.get("sw_cand_list", [])

    # Tope de capacidad de DG por barra, como fracción de su demanda base.
    # Configurable para poder calcular E4 con distintos supuestos sin editar
    # el código: `pipeline.py --etapas E4 --lim-dg 1.0 --etiqueta LIM-1.00`.
    # El default 0.3 conserva el comportamiento histórico de este módulo.
    # OJO: tiene que coincidir con el de optimization_6_oficial, o la
    # estrategia se compara contra un caso sin fallas de otra física.
    Lim = float(config.get("lim_dg", 0.3))


     # Subestación
    P_sub_total_sin_fallas = data.get("P_SUB_tot_sin_fallas",0)
    C_sub     = data["C_sub"]

    print(P_sub_total_sin_fallas)

    #Conjuntos
    BUSES = list(net.bus.index)
    LINES = list(net.line.index)
    GENS  = BUSES
    ESS   = BUSES
    S     = [0, 1]
    
    #Capacidad líneas
    S_base = 25; V_base = 20.0
    Z_base = V_base**2*1000 / S_base
    x_ij = {l: net.line.x_ohm_per_km.iloc[i] * net.line.length_km.iloc[i] / Z_base
             for i, l in enumerate(LINES)}
    # Capacidad [kW] = √3 · tensión nominal de la línea [kV] · I_max [kA] · 1000.
    # La tensión es la de su barra de origen (dato de la red, p. ej. 12 kV en El
    # Melón). Antes era 20 kV fijo, el de la red de validación, que es de 20 kV,
    # así que ahí el resultado no cambia.
    cap  = {l: (3**0.5) * float(net.bus.vn_kv[net.line.from_bus[l]]) * net.line.max_i_ka[i]*1000
             for i, l in enumerate(LINES)}
    
    #Demanda
    P_D_base = data["P_D_base"]

    #Costos caso base
    # C_sub     = {s: 0.097 for s in S}
    # C_sub     = {s: 0.00001 for s in S}


    # Parámetros económicos
    r = data["r"]
    n = data["años"]
    frp = r * (1+r)**n / ((1+r)**n - 1)

    # PRIORIZACIÓN DE CARGA
    buses_criticos = config.get("buses_criticos",[])
    usar_prior_criticos = config.get("prior_criticos", False)

    # EDRP
    usar_edrp = config.get("edrp", False)
    BUSES_EDRP = [b for b in BUSES if b not in buses_criticos] if usar_edrp else []
    C_bloques_EDRP = 4
    k_EDRP = [0.25, 0.25, 0.25, 0.25]          # fracción de demanda base por bloque
    C_incentivo_EDRP = data.get("EDRP_incentivo", [1.0, 2.0, 3.0, 4.0])  # USD/kW por bloque
    alpha_edrp = data.get("alpha_edrp", 0.20)  # justificado en literatura

    # DG SOLAR
    C_inv_DG_sol  = data["DG_sol"][0]
    C_op_DG_sol   = data["DG_sol"][1]
    usar_dg_sol = config.get("dg_sol", False)

    # DG EÓLICA
    C_inv_DG_eol  = data["DG_eol"][0]
    C_op_DG_eol   = data["DG_eol"][1]
    usar_dg_eol = config.get("dg_eol", False)

    # DG FÓSIL
    C_inv_DG_fos  = data["DG_fos"][0]   # USD/MW — CAPEX
    C_op_DG_fos   = data["DG_fos"][1]   # USD/MW·año — OPEX fijo
    C_comb_DG_fos = data["DG_fos"][2]   # USD/MWh — costo variable diésel
    usar_dg_fos   = config.get("dg_fos", False)

    # ESS 2 HORAS
    eta_charge = 0.9; eta_discharge = 0.9
    SOC_min = 0.1;    SOC_max = 0.9
    C_inv_ESS_5h = data["ESS 5h"][0]
    C_op_ESS_5h = data["ESS 5h"][1]
    usar_ess_5h    = config.get("ess_5h",    False)
    h_ess_5h   = 5

    # ESS 4 HORAS
    eta_charge = 0.9; eta_discharge = 0.9
    SOC_min = 0.1;    SOC_max = 0.9
    C_inv_ESS_4h = data["ESS 4h"][0]
    C_op_ESS_4h = data["ESS 4h"][1]
    usar_ess_4h    = config.get("ess_4h",    False)
    h_ess_4h   = 4

    # ESS 6 HORAS
    eta_charge = 0.9; eta_discharge = 0.9
    SOC_min = 0.1;    SOC_max = 0.9
    C_inv_ESS_6h = data["ESS 6h"][0]
    C_op_ESS_6h = data["ESS 6h"][1]
    usar_ess_6h    = config.get("ess_6h", False)
    h_ess_6h   = 6

    # AISLACIÓN DE LA FALLA POR SECCIONADORES PREEXISTENTES
    # Es un supuesto de red, no una medida: tiene que estar activo en el
    # caso base exactamente igual que en las estrategias, o el "ahorro de
    # ENS" de cada medida se calcula contra una física distinta y sale
    # negativo aunque la medida sirva. Ver aislacion.py.
    usar_aislacion_fija, sec_existentes = leer_config_aislacion(config, data, LINES)

    # RECONFIGURACIÓN TOPOLÓGICA
    # Mismas tie-lines que la estrategia (ver optimization_6_oficial): son
    # las líneas normalmente abiertas de la red, no "las últimas tres". Si
    # el caso base usara otro conjunto, compararía contra una topología
    # distinta y el ahorro de cada medida quedaría mal medido.
    # Respaldo histórico solo si la clave FALTA (ver optimization_6_oficial).
    tie_indices = set(config["tie_indices"] if config.get("tie_indices") is not None
                      else LINES[-3:])
    tie_list = list(tie_indices)
    # M_tie = max(cap.values()) * 10
    # Cota de los ángulos (rad). Por defecto π/2, como siempre; se puede ampliar
    # para comprobar que no limita flujos. El big-M sigue a la cota, porque
    # |θi-θj| puede llegar a 2·THETA_MAX.
    THETA_MAX = float(config.get("theta_max", np.pi / 2))
    M_tie = {l: 2 * THETA_MAX / x_ij[l] for l in tie_list}
    usar_reconfig_topo = config.get("reconfig_topo", False)
    C_inv_SWITCH = data["SWITCH"][0]
    C_op_SWITCH = data["SWITCH"][1]

    # INVERSIÓN RCS O MS
    usar_inv_switch_rcs = config.get("inv_switch_rcs", False)
    usar_inv_switch_ms  = config.get("inv_switch_ms",  False)
    T_maniobra_MS       = config.get("T_maniobra_MS",  1)    # horas
    C_inv_RCS    = data["RCS"][0]
    C_opfijo_RCS = data["RCS"][1]
    C_inv_MS     = data["MS"][0]
    C_opfijo_MS  = data["MS"][1]
    M_sw = {l: 2 * THETA_MAX / x_ij[l] for l in sw_cand_list}
    bus_zone = data.get("bus_zone", {})  # ej: {0:1, 1:1, 2:1, 3:2, 4:2, ...}

    # MEG
    usar_meg   = config.get("meg", False)
    n_meg      = config.get("n_meg", 2)
    K          = list(range(n_meg))
    P_meg_max  = config.get("P_meg_max", 1.0)
    # P_meg_tot  = config.get("P_meg_tot", 2.0)   
    C_inv_MEG = data["MEG"][0]                  # USD/MW — CAPEX
    C_op_MEG   = data["MEG"][1]   # USD/MW·año — OPEX fijo
    C_comb_MEG = data["MEG"][2]   # USD/MWh    — diésel variable

    # tiempo de traslado en horas; puede ser escalar o diccionario por bus
    T_tra_meg_cfg = config.get("T_tra_meg", 2.0)
    if isinstance(T_tra_meg_cfg, (int, float)): T_tra_meg = {b: int(T_tra_meg_cfg) for b in BUSES}
    else:
        T_tra_meg = {b: int(T_tra_meg_cfg.get(b, 2)) for b in BUSES}
    
    # ══════════════════════════════════════════════════════
    # MESS — Mobile Emergency Storage System
    # ══════════════════════════════════════════════════════

    # --- Parámetros MESS (agregar en data/config) ---
    usar_mess       = config.get("mess", False)
    n_mess          = config.get("n_mess", 1)
    KM              = list(range(n_mess))          # índices de unidades MESS
    E_cap_mess      = config.get("E_cap_mess", 50.0)   # MWh — capacidad fija por unidad
    P_mess_max      = config.get("P_mess_max", 1.0)   # MW — potencia máx carga/descarga
    h_mess          = config.get("h_mess", None)
    if h_mess is None:
        E_cap_mess = config.get("E_cap_mess", 4.0)  # MWh
        h_mess = E_cap_mess / P_mess_max if P_mess_max > 0 else 0.0
    else:
        E_cap_mess = P_mess_max * h_mess
    eta_ch_mess     = config.get("eta_ch_mess", 0.9)
    eta_dch_mess    = config.get("eta_dch_mess", 0.9)
    SOC_min_mess    = config.get("SOC_min_mess", 0.1)
    SOC_max_mess    = config.get("SOC_max_mess", 0.9)
    C_inv_mess      = data.get("MESS", [537.5, 5.0])[0]  # USD/kWh CAPEX
    C_op_mess       = data.get("MESS", [537.5, 5.0])[1]  # USD/kWh OPEX variable

    # Tiempo de tránsito hacia cada bus (horas) — reutiliza T_tra_meg o define propio
    T_tra_mess_cfg  = config.get("T_tra_mess", 1.0)
    if isinstance(T_tra_mess_cfg, (int, float)):
        T_tra_mess = {b: int(T_tra_mess_cfg) for b in BUSES}
    else:
        T_tra_mess = {b: int(T_tra_mess_cfg.get(b, 1)) for b in BUSES}

    g_to_idx    = {g: i for i, g in enumerate(GENS)}

    # Lista única de índices MC reales a procesar
    # Con esc_sel=N → solo ese escenario; sin esc_sel → 0..num_esc-1
    escenarios = [esc_sel] if esc_sel is not None else list(range(num_esc))

    results_por_esc    = {esc: [] for esc in escenarios}
    estado_lineas_estr = np.zeros((len(escenarios), Horastot, len(LINES)))
    

    def resolver_esc(pos, esc):
        # ── 1. Calcular estado de daño por hora ──────────────────────────────
        Dam_tl_acum = np.zeros(len(LINES))
        Dam_tl_por_hora = {}

        for h in horas:
            Dam_nuevas = faultlinMC[h, esc, :]
            Rep_nuevas = repairMC[h, esc, :]
            for l in range(len(LINES)):
                if Rep_nuevas[l] == 1:
                    Dam_tl_acum[l] = 0
            Dam_tl_acum = np.maximum(Dam_tl_acum, Dam_nuevas)
            estado_lineas_estr[pos, h, :] = Dam_tl_acum
            Dam_tl_por_hora[h] = Dam_tl_acum.copy()

        # ── 2. Pre-cómputos (fuera del modelo) ───────────────────────────────
        bus_pos     = {b: i for i, b in enumerate(BUSES)}
        from_bus    = {l: int(net.line.from_bus[l]) for l in LINES}
        to_bus      = {l: int(net.line.to_bus[l])   for l in LINES}

        # ── 2b. Zona de aislación de la falla ────────────────────────────────
        # Daño FÍSICO (líneas averiadas, las que requieren cuadrilla). Se
        # guarda aparte porque `Dam_tl_por_hora` pasa a representar las
        # líneas FUERA DE SERVICIO, que es lo que ve el modelo.
        Dam_fis_por_hora = {t: Dam_tl_por_hora[t].copy() for t in horas}

        if usar_aislacion_fija:
            lineas_en_bus = adyacencia_lineas(LINES, from_bus, to_bus)
            Dam_tl_por_hora = expandir_dano(
                Dam_tl_por_hora, horas, LINES, from_bus, to_bus,
                lineas_en_bus, sec_existentes, tie_indices)
            _n_fis = sum(int(Dam_fis_por_hora[t].sum()) for t in horas)
            _n_fs  = sum(int(Dam_tl_por_hora[t].sum())  for t in horas)
            print(f"[AISLACIÓN] Esc {esc}: seccionadores en {sorted(sec_existentes)} "
                  f"→ {_n_fis} línea·hora averiadas se traducen en {_n_fs} "
                  f"línea·hora fuera de servicio.")
            if not sec_existentes:
                print("[AISLACIÓN] ⚠️ Sin seccionadores declarados: cualquier "
                      "falla deja fuera al alimentador completo hasta su "
                      "reparación (solo despeja el interruptor de cabecera).")

        # Líneas activas por hora (solo depende de t, no de b) — 15x menos trabajo
        activas_por_hora = {t: [l for l in LINES if l not in tie_indices and Dam_tl_por_hora[t][l] == 0] for t in horas}
        # activas_por_hora = {t: [ l for l in LINES if l not in tie_indices and l not in sw_cand_list and Dam_tl_por_hora[t][l] == 0] for t in horas}
        # Inflow/outflow por (t, b): cada línea se recorre UNA vez por hora y se
        # deposita en su barra, en vez de barrer la lista completa de líneas por
        # cada barra. Es O(T·L) en lugar de O(T·B·L): con 179 barras y 8760 h
        # baja de 567 a 3,2 millones de operaciones, y era el costo dominante de
        # armar el modelo (el solver ni había empezado). El orden dentro de cada
        # lista se conserva —las líneas entran en el orden de
        # `activas_por_hora`—, así que las expresiones del balance salen término
        # por término idénticas a las de antes.
        buses_validos = set(BUSES)
        inflow_act = {(t, b): [] for t in horas for b in BUSES}
        outflow_act = {(t, b): [] for t in horas for b in BUSES}
        for t in horas:
            for l in activas_por_hora[t]:
                if to_bus[l] in buses_validos:
                    inflow_act[t, to_bus[l]].append(l)
                if from_bus[l] in buses_validos:
                    outflow_act[t, from_bus[l]].append(l)

        # Demanda pre-computada
        P_D_hora_arr = {(t, b): P_D_base[b] * float(demand_factor[t, esc, bus_pos[b]]) for t in horas for b in BUSES}

        # Ramales con fusible fundidos (ver ramales.py). Misma lógica que en
        # optimization_6_oficial: si difieren, el ahorro de una medida se mide
        # contra un caso base con otra física.
        piso_lc = piso_corte_ramales(data, esc, horas)
        if piso_lc:
            print(f"[RAMALES] Esc {esc}: {len(piso_lc)} barra·hora con ramal fundido · "
                  f"{sum(P_D_hora_arr[k] * v for k, v in piso_lc.items()):.1f} kWh de corte forzado")

        # Sets activos según medidas habilitadas — evita loops de .fix()
        ESS_activo_5h  = ESS  if usar_ess_5h   else []
        ESS_activo_4h  = ESS  if usar_ess_4h   else []
        ESS_activo_6h  = ESS  if usar_ess_6h   else []
        GENS_sol    = GENS if usar_dg_sol else []
        GENS_eol    = GENS if usar_dg_eol else []
        GENS_fos    = GENS if usar_dg_fos else []
        K_meg       = K    if usar_meg    else []
        # Mismo patrón que `K_meg`, extendido a reconfiguración y switches. Sus
        # variables se creaban siempre y después se fijaban en 0 una por una:
        # para Oberrhein a 8760 h eran 3,2 millones de binarias inútiles (más el
        # doble loop de `.fix(0)` en Python) en un modelo que el caso base usa
        # SOLO con DG solar y eólica. Con las listas vacías el E4/E5 queda como
        # lo que es, un LP.
        #
        # OJO: `tie_indices` NO se toca. Es la topología de la red —las líneas
        # normalmente abiertas— y la usan la zona de aislación, el límite de
        # línea y el tope de DG. Lo que se apaga acá son las VARIABLES de
        # maniobra sobre esas líneas.
        tie_var_list = tie_list if usar_reconfig_topo else []
        sw_var_list  = sw_cand_list if (usar_inv_switch_rcs or usar_inv_switch_ms) else []
        sw_activas_por_hora = {t: [l for l in sw_var_list if Dam_tl_por_hora[t][l] == 0]
                               for t in horas}

        # ess_en_bus y gens_en_bus filtrados por sets activos
        # ess_en_bus_act_5h  = {b: [e for e in ESS_activo_5h  if snet.sgen.bus[e] == b] for b in BUSES}
        # ess_en_bus_act_4h  = {b: [e for e in ESS_activo_4h  if snet.sgen.bus[e] == b] for b in BUSES}
        # ess_en_bus_act_6h  = {b: [e for e in ESS_activo_6h  if snet.sgen.bus[e] == b] for b in BUSES}
        # gens_en_bus_sol = {b: [g for g in GENS_sol    if snet.sgen.bus[g] == b] for b in BUSES}
        # gens_en_bus_eol = {b: [g for g in GENS_eol    if snet.sgen.bus[g] == b] for b in BUSES}
        # gens_en_bus_fos = {b: [g for g in GENS_fos    if snet.sgen.bus[g] == b] for b in BUSES}
        gens_en_bus_sol   = {b: [b] if b in GENS_sol      else [] for b in BUSES}
        gens_en_bus_eol   = {b: [b] if b in GENS_eol      else [] for b in BUSES}
        gens_en_bus_fos   = {b: [b] if b in GENS_fos      else [] for b in BUSES}
        ess_en_bus_act_5h = {b: [b] if b in ESS_activo_5h else [] for b in BUSES}
        ess_en_bus_act_4h = {b: [b] if b in ESS_activo_4h else [] for b in BUSES}
        ess_en_bus_act_6h = {b: [b] if b in ESS_activo_6h else [] for b in BUSES}

        horas_evento = [t for t in horas if np.any(Dam_tl_por_hora[t] == 1)]
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

        hora_inicio_evento = bloques_evento[0][0] if bloques_evento else None
        hora_fin_evento = bloques_evento[-1][1] if bloques_evento else None

        # ── 3. Construcción del modelo ───────────────────────────────────────
        model   = pyo.ConcreteModel()
        model.T = pyo.Set(initialize=horas)

        # Variables siempre activas
        model.P_LC  = pyo.Var(model.T, BUSES, within=pyo.NonNegativeReals)
        model.P_SUB = pyo.Var(model.T, S,     within=pyo.NonNegativeReals)
        model.theta = pyo.Var(model.T, BUSES,  within=pyo.Reals)
        for t in horas: model.theta[t, 0].fix(0)

        # Variables condicionales — solo se crean si la medida está activa
        model.CAP_DG_sol  = pyo.Var(GENS_sol,  within=pyo.NonNegativeReals)
        model.P_DG_sol    = pyo.Var(model.T, GENS_sol,  within=pyo.NonNegativeReals)

        model.CAP_DG_eol  = pyo.Var(GENS_eol,  within=pyo.NonNegativeReals)
        model.P_DG_eol    = pyo.Var(model.T, GENS_eol,  within=pyo.NonNegativeReals)

        model.CAP_DG_fos  = pyo.Var(GENS_fos,  within=pyo.NonNegativeReals)
        model.P_DG_fos    = pyo.Var(model.T, GENS_fos,  within=pyo.NonNegativeReals)

        model.CAP_ESS_5h     = pyo.Var(ESS_activo_5h, within=pyo.NonNegativeReals)
        model.P_charge_5h    = pyo.Var(model.T, ESS_activo_5h, within=pyo.NonNegativeReals)
        model.P_discharge_5h = pyo.Var(model.T, ESS_activo_5h, within=pyo.NonNegativeReals)
        model.SOC_5h         = pyo.Var(model.T, ESS_activo_5h, within=pyo.NonNegativeReals)

        model.CAP_ESS_4h     = pyo.Var(ESS_activo_4h, within=pyo.NonNegativeReals)
        model.P_charge_4h    = pyo.Var(model.T, ESS_activo_4h, within=pyo.NonNegativeReals)
        model.P_discharge_4h = pyo.Var(model.T, ESS_activo_4h, within=pyo.NonNegativeReals)
        model.SOC_4h         = pyo.Var(model.T, ESS_activo_4h, within=pyo.NonNegativeReals)
        # model.x_ess_4h       = pyo.Var(model.T, ESS_activo_4h, within=pyo.Binary)

        model.CAP_ESS_6h     = pyo.Var(ESS_activo_6h, within=pyo.NonNegativeReals)
        model.P_charge_6h    = pyo.Var(model.T, ESS_activo_6h, within=pyo.NonNegativeReals)
        model.P_discharge_6h = pyo.Var(model.T, ESS_activo_6h, within=pyo.NonNegativeReals)
        model.SOC_6h         = pyo.Var(model.T, ESS_activo_6h, within=pyo.NonNegativeReals)
        
        model.z_tie = pyo.Var(model.T, tie_var_list, within=pyo.Binary)
        model.w_tie = pyo.Var(model.T, tie_var_list, within=pyo.Binary)
        model.P_tie = pyo.Var(model.T, tie_var_list, within=pyo.Reals)

        #INVERSIÓN RCS-MS
        model.y_rcs = pyo.Var(sw_var_list, within=pyo.Binary)
        model.y_ms  = pyo.Var(sw_var_list, within=pyo.Binary)
        model.z_sw  = pyo.Var(model.T, sw_var_list, within=pyo.Binary)
        model.P_sw  = pyo.Var(model.T, sw_var_list, within=pyo.Reals)

        #MEG
        model.CAP_MEG = pyo.Var(K_meg,  within=pyo.NonNegativeReals)
        model.x_re    = pyo.Var(K_meg,  BUSES, within=pyo.Binary)
        model.P_MEG   = pyo.Var(model.T, K_meg, BUSES, within=pyo.NonNegativeReals)
        model.psi_meg = pyo.Var(K_meg,  BUSES, within=pyo.NonNegativeReals)
        # Solo con MEG: es un Param mutable por (hora, barra), o sea 1,6 millones
        # de objetos Pyomo para Oberrhein a 8760 h, y nadie más lo lee.
        if usar_meg:
            model.a_tb = pyo.Param(model.T, BUSES, initialize=0, mutable=True, within=pyo.Binary)

        #MESS
        KM_mess = KM if usar_mess else []

        model.u_mess   = pyo.Var(KM_mess, BUSES, model.T, within=pyo.Binary)
        model.v_mess   = pyo.Var(KM_mess, BUSES, model.T, within=pyo.Binary)  # arribo
        model.Pdch_mess = pyo.Var(model.T, KM_mess, within=pyo.NonNegativeReals)
        model.Pch_mess  = pyo.Var(model.T, KM_mess, within=pyo.NonNegativeReals)
        model.SOC_mess  = pyo.Var(model.T, KM_mess, within=pyo.NonNegativeReals)

        # Variable auxiliar linealización bilineal u * (Pdch - Pch)
        # psi_mess[t,k,b] ≈ (Pdch_mess[t,k] - Pch_mess[t,k]) * u_mess[k,b,t]
        model.psi_mess_dch = pyo.Var(model.T, KM_mess, BUSES, within=pyo.NonNegativeReals)
        model.psi_mess_ch  = pyo.Var(model.T, KM_mess, BUSES, within=pyo.NonNegativeReals)

        #EDRP
        BLOQUES_EDRP = list(range(C_bloques_EDRP)) if usar_edrp else []
        model.P_EDRP = pyo.Var(model.T, BUSES, BLOQUES_EDRP, within=pyo.NonNegativeReals) 
                      
        # Sin reconfiguración, `tie_var_list` está vacía: las variables no
        # existen y no hay nada que fijar (antes se creaban las 3·T·|tie| y se
        # fijaban en 0 una por una). Con reconfiguración, se fija lo mismo que
        # siempre.
        if usar_reconfig_topo:
            for t in horas:
                if np.sum(Dam_tl_por_hora[t]) == 0:
                    for l in tie_var_list:
                        model.z_tie[t, l].fix(0)
                        model.P_tie[t, l].fix(0)

            # Una enlace DAÑADA no puede cerrarse. Es conductor aéreo expuesto:
            # estar normalmente abierta no la hace inmune al viento, y de hecho
            # entra al sorteo de `construir_faultlin.py` como cualquier otra
            # línea. Faltaba el equivalente de lo que sí se hace más abajo con
            # los switches, así que una enlace rota seguía disponible para
            # reconfigurar y sobrevaloraba `reconfig_topo` / `inv_tie`.
            for t in horas:
                for l in tie_var_list:
                    if Dam_tl_por_hora[t][l] == 1:
                        model.z_tie[t, l].fix(0)
                        model.w_tie[t, l].fix(0)
                        model.P_tie[t, l].fix(0)

        if sw_var_list:
            # y_rcs / y_ms: fijar a 0 el tipo de switch en que no se invierte.
            if not usar_inv_switch_rcs:
                for l in sw_var_list:
                    model.y_rcs[l].fix(0)

            if not usar_inv_switch_ms:
                for l in sw_var_list:
                    model.y_ms[l].fix(0)

            # Fijar z_sw y P_sw a 0 si la línea está dañada en esa hora
            for t in horas:
                for l in sw_var_list:
                    if Dam_tl_por_hora[t][l] == 1:
                        model.z_sw[t, l].fix(0)
                        model.P_sw[t, l].fix(0)

        # ── 4. Restricciones ────────────────────────────────────────────────

        def power_balance_rule(m, t, b):
            inf  = sum((m.theta[t, from_bus[l]] - m.theta[t, to_bus[l]]) / x_ij[l] for l in inflow_act[t, b])
            outf = sum((m.theta[t, from_bus[l]] - m.theta[t, to_bus[l]]) / x_ij[l] for l in outflow_act[t, b])
            g_sol = sum(m.P_DG_sol[t, g]  for g in gens_en_bus_sol[b])
            g_eol = sum(m.P_DG_eol[t, g]  for g in gens_en_bus_eol[b])
            g_fos = sum(m.P_DG_fos[t, g]  for g in gens_en_bus_fos[b])
            ess_p_5h = sum(m.P_discharge_5h[t, e] - m.P_charge_5h[t, e] for e in ess_en_bus_act_5h[b])
            ess_p_4h = sum(m.P_discharge_4h[t, e] - m.P_charge_4h[t, e] for e in ess_en_bus_act_4h[b])
            ess_p_6h = sum(m.P_discharge_6h[t, e] - m.P_charge_6h[t, e] for e in ess_en_bus_act_6h[b])
            g_meg = sum(m.P_MEG[t, k, b]  for k in K_meg)
            g_mess = sum(m.psi_mess_dch[t, k, b] - m.psi_mess_ch[t, k, b] for k in KM_mess) if usar_mess else 0
            inf_tie  = sum(m.P_tie[t, l] for l in tie_var_list if to_bus[l]   == b)
            outf_tie = sum(m.P_tie[t, l] for l in tie_var_list if from_bus[l] == b)
            inf_sw  = sum(m.P_sw[t, l] for l in sw_activas_por_hora[t] if to_bus[l]   == b)
            outf_sw = sum(m.P_sw[t, l] for l in sw_activas_por_hora[t] if from_bus[l] == b)
            sub_p = m.P_SUB[t, 0] if b == 1 else (m.P_SUB[t, 1] if b == 12 else 0)
            edrp_red = sum(m.P_EDRP[t, b, c] for c in BLOQUES_EDRP) if usar_edrp else 0
            return (inf - outf + (inf_tie - outf_tie) + (inf_sw - outf_sw) + sub_p + g_sol + g_eol + g_fos + ess_p_5h + ess_p_4h + ess_p_6h + g_meg + g_mess == P_D_hora_arr[t, b] - m.P_LC[t, b] - edrp_red)
            # return (inf - outf + (inf_tie - outf_tie) + sub_p + g_sol + g_eol + g_fos + ess_p_5h + ess_p_4h + ess_p_6h + g_meg + g_mess == P_D_hora_arr[t, b] - m.P_LC[t, b] - edrp_red)
        
        def theta_rule(m, t, b):
            return (-THETA_MAX, m.theta[t, b], THETA_MAX)

        def curtail_rule(m, t, b):
            # Piso = carga detrás de fusibles fundidos; 0 sin ramales (modelo idéntico).
            piso = P_D_hora_arr[t, b] * piso_lc[t, b] if (t, b) in piso_lc else 0
            return (piso, m.P_LC[t, b], P_D_hora_arr[t, b])

        def sub_rule(m, t, s):
            return (0, m.P_SUB[t, s], net.trafo.sn_mva[0] * 1000)

        def line_limit_rule(m, t, l):
            if Dam_tl_por_hora[t][l] == 1 or l in tie_indices:
                return pyo.Constraint.Skip
            i, j = from_bus[l], to_bus[l]
            return pyo.inequality(-cap[l], (m.theta[t, i] - m.theta[t, j]) / x_ij[l], cap[l])

        model.power_balance     = pyo.Constraint(model.T, BUSES, rule=power_balance_rule)
        model.curtailment_limits = pyo.Constraint(model.T, BUSES, rule=curtail_rule)
        model.sub_limits         = pyo.Constraint(model.T, S,     rule=sub_rule)
        model.line_limits        = pyo.Constraint(model.T, LINES, rule=line_limit_rule)
        model.theta_limits       = pyo.Constraint(model.T, BUSES, rule=theta_rule)

        # Mismo tope que `optimization_6_oficial` (ver `limites_dg.py`): la
        # capacidad que E4 deja instalar es la que define el CAPEX/OPEX sin
        # fallas contra el que se descuenta la estrategia, así que los dos
        # módulos TIENEN que agrupar y acotar igual.
        grupos_dg, Lim, lim_dg_bus = leer_config_limites_dg(
            config, data, net, BUSES, LINES, from_bus, to_bus,
            tie_indices=tie_indices, bus_zone=bus_zone,
            sec_existentes=sec_existentes, lim_default=0.3,
            verbose=bool(config.get("verbose")))

        # `lim_dg_conjunto`: solar + eólica comparten UN tope (suma ≤ lim_dg); sin él,
        # cada tecnología tiene el suyo (histórico). Ver `limites_dg.aplicar_tope_conjunto`.
        conjunto_dg = bool(config.get("lim_dg_conjunto", False))

        if usar_dg_sol:
            model.dg_sol_limits = pyo.Constraint(model.T, GENS_sol, rule=lambda m, t, g: m.P_DG_sol[t, g] <= alphaDGsolar[t, esc, g_to_idx[g]] * m.CAP_DG_sol[g])
            if not conjunto_dg:
                aplicar_tope_dg(model, "dg_sol", model.CAP_DG_sol, GENS_sol,
                                grupos_dg, Lim, P_D_base, lim_dg_bus)

        if usar_dg_eol:
            # La disponibilidad eólica sale del recurso (curva de potencia, ya
            # dentro de `alphaDGeolica`) y de la topología. Misma forma que la
            # solar, para que el ranking entre ambas medidas sea comparable.
            model.dg_eol_limits = pyo.Constraint(
                model.T, GENS_eol,
                rule=lambda m, t, g: m.P_DG_eol[t, g]
                <= alphaDGeolica[t, esc, g_to_idx[g]] * m.CAP_DG_eol[g])

            # ── `a_eol_wind`, DESACTIVADO el 2026-09-08 ──────────────────
            # Anulaba la generación de toda zona con alguna línea dañada.
            # Se deja comentado, no borrado, por si se quiere reactivar.
            #
            # Por qué salió:
            #
            #   - NO era un apagado por viento. Se disparaba por DAÑO DE
            #     LÍNEA. Una zona con una línea rota por la ráfaga de las 15 h
            #     quedaba sin eólica hasta las 18 h, con viento de 8 m/s,
            #     porque `Dam_tl_por_hora` es acumulativo hasta la reparación.
            #   - El apagado por viento fuerte YA EXISTE y es el estándar
            #     IEC 61400: `V_CUT_OUT = 25 m/s` en `construir_perfiles.py`,
            #     sobre media de 10 min. En Arauco no se activa nunca porque
            #     el sostenido máximo a 30 m de buje es 22,3 m/s — es el
            #     sitio, no un mecanismo faltante.
            #   - Rompía la simetría con la solar, que no tiene ningún
            #     término equivalente. Un ranking sobre reglas distintas para
            #     cada tecnología no compara nada.
            #   - Efecto medido: apagaba la eólica ~9 h/año, y esas 9 h eran
            #     el 100 % de las horas con ENS. `dg_eol` generaba 31,5 GWh y
            #     aportaba EXACTAMENTE CERO a la resiliencia, con ahorro neto
            #     negativo; la solar, con la misma capacidad instalada, bajaba
            #     la ENS un 74 %.
            #
            # LIMITACIÓN A DECLARAR: sin esto la turbina genera con ráfagas de
            # 39 m/s, y una real dispararía por protección. Es una cota
            # superior de disponibilidad eólica. Se acepta porque la solar
            # tampoco tiene desconexión por tormenta: el sesgo es simétrico.
            # Un trip por ráfaga es específico de cada fabricante y el umbral
            # elegido decide por sí solo si `dg_eol` sirve, así que va como
            # sensibilidad declarada, nunca como supuesto base.
            #
            # a_eol_wind = {}
            # for t in horas:
            #     zonas_afectadas_viento = set()
            #     for l in LINES:
            #         if Dam_tl_por_hora[t][l] == 1:
            #             bi = from_bus[l]
            #             bj = to_bus[l]
            #             zi = bus_zone.get(bi, None)
            #             zj = bus_zone.get(bj, None)
            #
            #             if zi is not None:
            #                 zonas_afectadas_viento.add(zi)
            #             if zj is not None:
            #                 zonas_afectadas_viento.add(zj)
            #     for g in GENS:
            #         zona_g = bus_zone.get(g, None)
            #
            #         if zona_g in zonas_afectadas_viento:
            #             a_eol_wind[t, g] = 0
            #         else:
            #             a_eol_wind[t, g] = 1
            # model.dg_eol_limits = pyo.Constraint(model.T, GENS_eol, rule=lambda m, t, g: m.P_DG_eol[t, g] <= alphaDGeolica[t, esc, g_to_idx[g]] * m.CAP_DG_eol[g] * a_eol_wind[t, g])

            if not conjunto_dg:
                aplicar_tope_dg(model, "dg_eol", model.CAP_DG_eol, GENS_eol,
                                grupos_dg, Lim, P_D_base, lim_dg_bus)

        if conjunto_dg:
            _techs_dg = [(v, gs) for usar, v, gs in ((usar_dg_sol, model.CAP_DG_sol if usar_dg_sol else None, GENS_sol),
                                                    (usar_dg_eol, model.CAP_DG_eol if usar_dg_eol else None, GENS_eol)) if usar]
            if _techs_dg:
                aplicar_tope_dg_conjunto(model, "dg_ambas", _techs_dg,
                                         grupos_dg, Lim, P_D_base, lim_dg_bus)

        # Capacidad DG impuesta desde afuera: {"sol": {barra: kW}, "eol": {...}}.
        # La usa run_barrido_medidas para evaluar SIN fallas el plan de DG que
        # eligió una corrida CON fallas: con la misma DG en los mismos nodos,
        # la diferencia entre las dos corridas es el costo de las fallas y el
        # CAPEX se cancela. Sin la clave, la capacidad queda libre (E4).
        _cap_fija = config.get("cap_dg_fija") or {}
        for _tec, _usar, _var, _gens in (("sol", usar_dg_sol, model.CAP_DG_sol, GENS_sol),
                                         ("eol", usar_dg_eol, model.CAP_DG_eol, GENS_eol)):
            if _usar and _cap_fija.get(_tec) is not None:
                for g in _gens:
                    _var[g].fix(float(_cap_fija[_tec].get(g, 0.0)))

        if usar_dg_fos:
            model.dg_fos_limits = pyo.Constraint(model.T, GENS_fos, rule=lambda m, t, g: m.P_DG_fos[t, g] <= m.CAP_DG_fos[g])
            # No estaba acotada: E4 solo corre sol/eol (`pipeline.py:715`),
            # así que este bloque nunca se ejecutaba y la falta de tope no
            # se notaba. Se cierra para que base y 6 no diverjan.
            aplicar_tope_dg(model, "dg_fos", model.CAP_DG_fos, GENS_fos,
                            grupos_dg, Lim, P_D_base, lim_dg_bus)

        if usar_ess_5h:
            model.charge_limits_5h    = pyo.Constraint(model.T, ESS_activo_5h, rule=lambda m, t, e: m.P_charge_5h[t, e] <= m.CAP_ESS_5h[e] / h_ess_5h)
            model.discharge_limits_5h = pyo.Constraint(model.T, ESS_activo_5h, rule=lambda m, t, e: m.P_discharge_5h[t, e] <= m.CAP_ESS_5h[e] / h_ess_5h)
            model.soc_min_5h          = pyo.Constraint(model.T, ESS_activo_5h, rule=lambda m, t, e: m.SOC_5h[t, e] >= SOC_min * m.CAP_ESS_5h[e])
            model.soc_max_5h          = pyo.Constraint(model.T, ESS_activo_5h, rule=lambda m, t, e: m.SOC_5h[t, e] <= SOC_max * m.CAP_ESS_5h[e])
        
            def soc_evolution_rule_5h(m, t, e):
                if t == 0:
                    # return m.SOC_5h[t, e] == m.SOC_5h[horas[-1], e]
                    return m.SOC_5h[t, e] == m.CAP_ESS_5h[e]/2

                return m.SOC_5h[t, e] == (m.SOC_5h[t-1, e] + eta_charge * m.P_charge_5h[t, e] - (1 / eta_discharge) * m.P_discharge_5h[t, e])
            
            model.soc_evolution_5h = pyo.Constraint(model.T, ESS_activo_5h, rule=soc_evolution_rule_5h)

        if usar_ess_4h:
            model.charge_limits_4h    = pyo.Constraint(model.T, ESS_activo_4h, rule=lambda m, t, e: m.P_charge_4h[t, e] <= m.CAP_ESS_4h[e] / h_ess_4h)
            model.discharge_limits_4h = pyo.Constraint(model.T, ESS_activo_4h, rule=lambda m, t, e: m.P_discharge_4h[t, e] <= m.CAP_ESS_4h[e] / h_ess_4h)
            def soc_evolution_rule_4h(m, t, e):
                if t == 0:
                    # return m.SOC_4h[t, e] == m.SOC_4h[horas[-1], e]
                    return m.SOC_4h[t, e] == m.CAP_ESS_4h[e]/2
                return m.SOC_4h[t, e] == (m.SOC_4h[t-1, e] + eta_charge * m.P_charge_4h[t, e] - (1 / eta_discharge) * m.P_discharge_4h[t, e])
            
            model.soc_evolution_4h = pyo.Constraint(model.T, ESS_activo_4h, rule=soc_evolution_rule_4h)
            model.soc_min_4h          = pyo.Constraint(model.T, ESS_activo_4h, rule=lambda m, t, e: m.SOC_4h[t, e] >= SOC_min * m.CAP_ESS_4h[e])
            model.soc_max_4h          = pyo.Constraint(model.T, ESS_activo_4h, rule=lambda m, t, e: m.SOC_4h[t, e] <= SOC_max * m.CAP_ESS_4h[e])
            # model.charge_discharge_4h = pyo.Constraint(model.T, ESS_activo_4h, rule=lambda m, t, e: m.x_ess[t,e])


        if usar_ess_6h:
            model.charge_limits_6h    = pyo.Constraint(model.T, ESS_activo_6h, rule=lambda m, t, e: m.P_charge_6h[t, e] <= m.CAP_ESS_6h[e] / h_ess_6h)
            model.discharge_limits_6h = pyo.Constraint(model.T, ESS_activo_6h, rule=lambda m, t, e: m.P_discharge_6h[t, e] <= m.CAP_ESS_6h[e] / h_ess_6h)
            model.soc_min_6h          = pyo.Constraint(model.T, ESS_activo_6h, rule=lambda m, t, e: m.SOC_6h[t, e] >= SOC_min * m.CAP_ESS_6h[e])
            model.soc_max_6h          = pyo.Constraint(model.T, ESS_activo_6h, rule=lambda m, t, e: m.SOC_6h[t, e] <= SOC_max * m.CAP_ESS_6h[e])

            def soc_evolution_rule_6h(m, t, e):
                if t == 0:
                    # return m.SOC_6h[t, e] == m.SOC_6h[horas[-1], e]
                    return m.SOC_6h[t, e] ==  m.CAP_ESS_6h[e]/2
                return m.SOC_6h[t, e] == (m.SOC_6h[t-1, e] + eta_charge * m.P_charge_6h[t, e] - (1 / eta_discharge) * m.P_discharge_6h[t, e])
            
            model.soc_evolution_6h = pyo.Constraint(model.T, ESS_activo_6h, rule=soc_evolution_rule_6h)

        if usar_reconfig_topo:
            model.tie_link_upper = pyo.Constraint(model.T, tie_list, rule=lambda m, t, l: (m.theta[t, from_bus[l]] - m.theta[t, to_bus[l]]) / x_ij[l] - m.P_tie[t, l] <= M_tie[l] * (1 - m.z_tie[t, l]))
            model.tie_link_lower = pyo.Constraint(model.T, tie_list, rule=lambda m, t, l: (m.theta[t, from_bus[l]] - m.theta[t, to_bus[l]]) / x_ij[l] - m.P_tie[t, l] >= -M_tie[l] * (1 - m.z_tie[t, l]))
            model.tie_flow_upper = pyo.Constraint(model.T, tie_list, rule=lambda m, t, l: m.P_tie[t, l] <= cap[l] * m.z_tie[t, l])
            model.tie_flow_lower = pyo.Constraint(model.T, tie_list, rule=lambda m, t, l: m.P_tie[t, l] >= -cap[l] * m.z_tie[t, l])
            model.switching_limit = pyo.Constraint(model.T, tie_list, rule=lambda m, t, l: pyo.Constraint.Skip if t == 0 else m.z_tie[t, l] - m.z_tie[t-1, l] <= 1)
            # Variable de maniobra de tie-lines

            def tie_switch_change_pos_rule(m, t, l):
                if t == horas[0]:
                    # tie-lines normalmente abiertas: z_ref = 0
                    return m.w_tie[t, l] >= m.z_tie[t, l]
                return m.w_tie[t, l] >= m.z_tie[t, l] - m.z_tie[t-1, l]

            def tie_switch_change_neg_rule(m, t, l):
                if t == horas[0]:
                    # tie-lines normalmente abiertas: z_ref = 0
                    return m.w_tie[t, l] >= -m.z_tie[t, l]
                return m.w_tie[t, l] >= m.z_tie[t-1, l] - m.z_tie[t, l]

            model.tie_switch_change_pos = pyo.Constraint(
                model.T, tie_list, rule=tie_switch_change_pos_rule
            )

            model.tie_switch_change_neg = pyo.Constraint(
                model.T, tie_list, rule=tie_switch_change_neg_rule
            )

        if sw_cand_list and (usar_inv_switch_rcs or usar_inv_switch_ms):
            a_ms = {t: 0 for t in horas}
            for h_ini, h_fin in bloques_evento:
                t_disp = h_ini + T_maniobra_MS
                for t in horas:
                    if t_disp <= t <= h_fin:
                        a_ms[t] = 1
            def maniobra_ms_rule(m, t, l):
                if a_ms[t] == 0:
                    return m.z_sw[t, l] <= m.y_rcs[l]
                return pyo.Constraint.Skip
            model.un_tipo_sw = pyo.Constraint(sw_cand_list, rule=lambda m, l: m.y_rcs[l] + m.y_ms[l] <= 1)
            model.sw_existe = pyo.Constraint(model.T, sw_cand_list,rule=lambda m, t, l: m.z_sw[t, l] <= m.y_rcs[l] + m.y_ms[l])

            model.maniobra_ms = pyo.Constraint(model.T, sw_cand_list, rule=maniobra_ms_rule)
            model.sw_link_upper = pyo.Constraint(model.T, sw_cand_list,rule=lambda m, t, l: (m.theta[t, from_bus[l]] - m.theta[t, to_bus[l]]) / x_ij[l] - m.P_sw[t, l] <= M_sw[l] * (1 - m.z_sw[t, l]))
            model.sw_link_lower = pyo.Constraint(model.T, sw_cand_list,rule=lambda m, t, l: (m.theta[t, from_bus[l]] - m.theta[t, to_bus[l]]) / x_ij[l] - m.P_sw[t, l] >= -M_sw[l] * (1 - m.z_sw[t, l]))
            model.sw_flow_upper = pyo.Constraint(model.T, sw_cand_list,rule=lambda m, t, l: m.P_sw[t, l] <=  cap[l] * m.z_sw[t, l])
            model.sw_flow_lower = pyo.Constraint(model.T, sw_cand_list,rule=lambda m, t, l: m.P_sw[t, l] >= -cap[l] * m.z_sw[t, l])
            # Si la línea candidata está dañada, queda abierta
            # model.sw_no_danada = pyo.Constraint(
            #     model.T, sw_cand_list,
            #     rule=lambda m, t, l:
            #         m.z_sw[t, l] <= 1 - Dam_tl_por_hora[t][l]
            # )

            # # Si está sana y no hay capacidad de maniobra disponible, debe quedar cerrada
            # model.sw_cerrada_si_no_maniobra = pyo.Constraint(
            #     model.T, sw_cand_list,
            #     rule=lambda m, t, l:
            #         m.z_sw[t, l] >= 1 - Dam_tl_por_hora[t][l] - m.y_rcs[l] - a_ms[t] * m.y_ms[l]
# )


        if usar_meg:
            duracion_servicio = 5
            for h_ini, h_fin in bloques_evento:
                for b in BUSES:
                    llegada = h_ini + T_tra_meg.get(b, 2)
                    fin_op = min(llegada + duracion_servicio - 1, Horastot - 1)
                    for t in range(int(llegada), int(fin_op) + 1):
                        if t in horas:
                            model.a_tb[t, b] = 1
                        # model.cap_meg_ub    = pyo.Constraint(K_meg, rule=lambda m, k: m.CAP_MEG[k] <= P_meg_max)
            model.meg_un_bus    = pyo.Constraint(K_meg, rule=lambda m, k: sum(m.x_re[k, b] for b in BUSES) <= 1)
            model.bus_un_meg    = pyo.Constraint(BUSES, rule=lambda m, b: sum(m.x_re[k, b] for k in K_meg) <= 1)
            # model.meg_si_cap    = pyo.Constraint(K_meg, BUSES, rule=lambda m, k, b: m.x_re[k, b] <= m.CAP_MEG[k] / P_meg_max)
            model.cap_meg = pyo.Constraint(K_meg, rule=lambda m, k: m.CAP_MEG[k] <= P_meg_max * sum(m.x_re[k, b] for b in BUSES))
            model.psi_meg_ub_cap = pyo.Constraint(K_meg, BUSES, rule=lambda m, k, b: m.psi_meg[k, b] <= m.CAP_MEG[k])
            model.psi_meg_ub_bin = pyo.Constraint(K_meg, BUSES, rule=lambda m, k, b: m.psi_meg[k, b] <= P_meg_max * m.x_re[k, b])
            model.psi_meg_lb    = pyo.Constraint(K_meg, BUSES, rule=lambda m, k, b: m.psi_meg[k, b] >= m.CAP_MEG[k] - P_meg_max * (1 - m.x_re[k, b]))
            model.p_meg_ub      = pyo.Constraint(model.T, K_meg, BUSES, rule=lambda m, t, k, b: m.P_MEG[t, k, b] <= m.psi_meg[k, b] * m.a_tb[t, b])

        # ══════════════════════════════════════════════════════
        # MESS — Restricciones (insertar al final de la sección restricciones, ~línea 310)
        # ══════════════════════════════════════════════════════

        if usar_mess:
            # --- Disponibilidad operativa por hora-bus (análogo a a_tb del MEG) ---
            mess_op = {(k, b, t): 0 for k in KM_mess for b in BUSES for t in horas}
            for h_ini, h_fin in bloques_evento:
                for b in BUSES:
                    llegada = h_ini + T_tra_mess.get(b, 1)
                    for t in horas:
                        if t >= llegada:
                            for k in KM_mess:
                                mess_op[k, b, t] = 1

            # (1) Un MESS en máximo un bus por período
            model.mess_un_bus = pyo.Constraint(KM_mess, model.T,rule=lambda m, k, t: sum(m.u_mess[k, b, t] for b in BUSES) <= 1)
            # (2) Máximo un MESS por bus por período
            model.bus_un_mess = pyo.Constraint(BUSES, model.T,rule=lambda m, b, t: sum(m.u_mess[k, b, t] for k in KM_mess) <= 1)
            # (3) MESS solo opera si disponible (tiempo tránsito + evento activo)
            model.mess_disponible = pyo.Constraint(KM_mess, BUSES, model.T,rule=lambda m, k, b, t: m.u_mess[k, b, t] <= mess_op[k, b, t])
            # (4) Linealización bilineal descarga: psi_dch[t,k,b] ≈ Pdch[t,k]*u[k,b,t]
            model.mess_dch_ub_P = pyo.Constraint(model.T, KM_mess, BUSES,rule=lambda m, t, k, b: m.psi_mess_dch[t, k, b] <= m.Pdch_mess[t, k])
            model.mess_dch_ub_u = pyo.Constraint(model.T, KM_mess, BUSES,rule=lambda m, t, k, b: m.psi_mess_dch[t, k, b] <= P_mess_max * m.u_mess[k, b, t])
            model.mess_dch_lb = pyo.Constraint(model.T, KM_mess, BUSES,rule=lambda m, t, k, b: m.psi_mess_dch[t, k, b] >= m.Pdch_mess[t, k] - P_mess_max * (1 - m.u_mess[k, b, t]))
            # (5) Linealización bilineal carga: psi_ch[t,k,b] ≈ Pch[t,k]*u[k,b,t]
            model.mess_ch_ub_P = pyo.Constraint(model.T, KM_mess, BUSES,rule=lambda m, t, k, b: m.psi_mess_ch[t, k, b] <= m.Pch_mess[t, k])
            model.mess_ch_ub_u = pyo.Constraint(model.T, KM_mess, BUSES,rule=lambda m, t, k, b: m.psi_mess_ch[t, k, b] <= P_mess_max * m.u_mess[k, b, t])
            model.mess_ch_lb = pyo.Constraint(model.T, KM_mess, BUSES,rule=lambda m, t, k, b: m.psi_mess_ch[t, k, b] >= m.Pch_mess[t, k] - P_mess_max * (1 - m.u_mess[k, b, t]) )
            # (6) Límites de potencia totales
            model.mess_dch_ub = pyo.Constraint(model.T, KM_mess,rule=lambda m, t, k: m.Pdch_mess[t, k] <= P_mess_max * sum(m.u_mess[k, b, t] for b in BUSES))
            model.mess_ch_ub = pyo.Constraint(model.T, KM_mess,rule=lambda m, t, k: m.Pch_mess[t, k] <= P_mess_max * sum(m.u_mess[k, b, t] for b in BUSES))
            # (7) Dinámica SoC
            def soc_mess_rule(m, t, k):
                if t == 0:
                    return m.SOC_mess[t, k] == E_cap_mess * 0.5  # SoC inicial al 50%
                return (m.SOC_mess[t, k] ==
                        m.SOC_mess[t-1, k]
                        + eta_ch_mess * m.Pch_mess[t, k]
                        - (1 / eta_dch_mess) * m.Pdch_mess[t, k])

            model.mess_soc_ev    = pyo.Constraint(model.T, KM_mess, rule=soc_mess_rule)
            model.mess_soc_min   = pyo.Constraint(model.T, KM_mess, rule=lambda m, t, k: m.SOC_mess[t, k] >= SOC_min_mess * E_cap_mess)
            model.mess_soc_max   = pyo.Constraint(model.T, KM_mess, rule=lambda m, t, k: m.SOC_mess[t, k] <= SOC_max_mess * E_cap_mess)


        # Al inicio de resolver_esc, antes de construir el modelo:
        if usar_prior_criticos:
            for b in buses_criticos:
                if b in VoLL_local:
                    VoLL_local[b] = VoLL_local[b] * 100
                else:
                    print(f"[PRIOR] ⚠️ Bus crítico {b} no encontrado en VoLL")

        # Activación temporal del EDRP: solo durante horas con evento activo
        a_edrp = {t: 0 for t in horas}
        if usar_edrp:
            for h_ini, h_fin in bloques_evento:
                for t in range(h_ini, h_fin + 1):
                    if t in horas:
                        a_edrp[t] = 1
            buses_no_edrp = [b for b in BUSES if b not in BUSES_EDRP]
            for t in horas:
                for b in buses_no_edrp:
                    for c in BLOQUES_EDRP:
                        model.P_EDRP[t, b, c].fix(0)
            model.edrp_ub = pyo.Constraint(model.T, BUSES_EDRP, BLOQUES_EDRP, rule=lambda m, t, b, c: m.P_EDRP[t, b, c] <= k_EDRP[c] * alpha_edrp * P_D_hora_arr[t, b] * a_edrp[t])
        else:
            for t in horas:
                for b in BUSES:
                    for c in BLOQUES_EDRP:
                        model.P_EDRP[t, b, c].fix(0)

        # ── 5. Función objetivo ──────────────────────────────────────────────
        def obj_rule(m):
            total = 0
            if usar_dg_sol:
                total += sum(C_inv_DG_sol[g] * m.CAP_DG_sol[g] * frp for g in GENS_sol)
                total += sum(m.CAP_DG_sol[g] * C_op_DG_sol[g] for g in GENS_sol)
            if usar_dg_eol:
                total += sum(C_inv_DG_eol[g] * m.CAP_DG_eol[g] * frp for g in GENS_eol)
                total += sum(m.CAP_DG_eol[g] * C_op_DG_eol[g]  for g in GENS_eol)
            # Crédito de subestación UNA sola vez por corrida, fuera de los `if`
            # de cada tecnología (como en optimization_6). Antes iba dentro de
            # los dos y con sol+eol se duplicaba: el E4/4 (ambas) valía cada kWh
            # de DG al doble y llenaba el tope eólico (13.423 kW contra 13.029
            # del óptimo real), lo que dejaba `cst_sobrecosto_dg` en ~ -680.
            # Ver ECUACIONES_OBJETIVO_DG.md §8.3.
            if usar_dg_sol or usar_dg_eol:
                total += -((P_sub_total_sin_fallas - sum(m.P_SUB[t, s] for t in horas for s in S))*C_sub[0])

            if usar_dg_fos:
                total += sum(C_inv_DG_fos[g] * m.CAP_DG_fos[g] * frp for g in GENS_fos)
                total += sum(m.P_DG_fos[t, g] * (C_comb_DG_fos[g]*0.000242 + C_op_DG_fos[g]) for t in horas for g in GENS_fos)
            # Costeo de ESS IGUAL que en optimization_6_oficial (línea 1422):
            #     (C_inv * frp + C_op) * (CAP / h_ess)
            # Tres cosas que las versiones anteriores de este bloque hacían
            # distinto, y que impedían comparar E4 con el barrido:
            #   1) el CAPEX se anualiza con `frp`; la rama de 5h (heredada de
            #      la de 2h) lo cobraba entero, ~9,4x más caro, así que la
            #      batería nunca entraba;
            #   2) el OPEX NO se multiplica por `frp`: ya es un costo anual;
            #   3) se divide por la duración, porque `CAP_ESS` está en kWh y
            #      los costos son por kW de potencia.
            # Cambiar esto no altera ningún artefacto: `config_medida` deja
            # todas las banderas de ESS apagadas y ninguna etapa las enciende,
            # así que este bloque no se había ejecutado nunca.
            if usar_ess_5h:
                total += sum((C_inv_ESS_5h[e] * frp + C_op_ESS_5h[e])
                             * (m.CAP_ESS_5h[e] / h_ess_5h) for e in ESS_activo_5h)
            if usar_ess_4h:
                total += sum((C_inv_ESS_4h[e] * frp + C_op_ESS_4h[e])
                             * (m.CAP_ESS_4h[e] / h_ess_4h) for e in ESS_activo_4h)
            if usar_ess_6h:
                total += sum((C_inv_ESS_6h[e] * frp + C_op_ESS_6h[e])
                             * (m.CAP_ESS_6h[e] / h_ess_6h) for e in ESS_activo_6h)
            if usar_meg:
                total += sum((C_inv_MEG) * frp * m.CAP_MEG[k] for k in K_meg)
                total += sum((C_op_MEG* T_tra_meg.get(b, 2) * m.x_re[k,b]) for k in K_meg for b in BUSES)
                total += sum(C_comb_MEG*0.000242 * m.P_MEG[t, k, b] for t in horas for k in K_meg for b in BUSES)
                # total += sum((C_inv_MEG + C_op_MEG) * frp * m.CAP_MEG[k] for k in K_meg)
                # total += sum(C_comb_MEG * m.P_MEG[t, k, b] for t in horas for k in K_meg for b in BUSES)
            if usar_mess:
                total += sum(C_inv_mess * E_cap_mess for k in KM_mess)
                total += sum(C_op_mess  * model.Pdch_mess[t, k] for t in horas for k in KM_mess)
            if usar_reconfig_topo:
                total += sum(C_op_SWITCH[l] * m.w_tie[t, l] for t in horas for l in tie_list)
            if usar_inv_switch_rcs and sw_cand_list:
                total += sum((C_inv_RCS) * frp * m.y_rcs[l] for l in sw_cand_list)
                # total += sum((C_inv_RCS + C_opfijo_RCS) * frp * m.y_rcs[l] for l in sw_cand_list)
            if usar_inv_switch_ms and sw_cand_list:
                total += sum((C_inv_MS) * frp * m.y_ms[l] for l in sw_cand_list)
                # total += sum((C_inv_MS + C_opfijo_MS) * frp * m.y_ms[l] for l in sw_cand_list)
            # NUEVO: OPEX variable de switches candidatos (solo si alguna inversión activa)
            if sw_cand_list and (usar_inv_switch_rcs or usar_inv_switch_ms):
                total += sum(C_op_SWITCH[12] * m.z_sw[t, l] for t in horas for l in sw_cand_list)
            if usar_edrp:
                total += sum(C_incentivo_EDRP[c] * m.P_EDRP[t, b, c] for t in horas for b in BUSES for c in BLOQUES_EDRP) * N_eventos
            # total += sum(VoLL[b] * m.P_LC[t, b] for t in horas for b in BUSES) * N_eventos
            total += sum(VoLL_local[b] * m.P_LC[t, b] for t in horas for b in BUSES) * N_eventos
            # total += sum(C_sub[s] * m.P_SUB[t, s] for t in horas for s in S)
            return total
        # def obj_rule(m):
        #     total = 0
        #     if usar_dg_sol:
        #         total += sum(C_inv_DG_sol[g] * frp * m.CAP_DG_sol[g] for g in GENS_sol)
        #         total += sum(m.P_DG_sol[t, g] * C_op_DG_sol[g] * frp for t in horas for g in GENS_sol)
        #     if usar_dg_eol:
        #         total += sum(C_inv_DG_eol[g] * frp * m.CAP_DG_eol[g] for g in GENS_eol)
        #         total += sum(m.P_DG_eol[t, g] * C_op_DG_eol[g] * frp for t in horas for g in GENS_eol)
        #     if usar_dg_fos:
        #         total += sum(C_inv_DG_fos[g] * frp * m.CAP_DG_fos[g] for g in GENS_fos)
        #         total += sum(m.P_DG_fos[t, g] * (C_op_DG_fos[g] + C_comb_DG_fos[g]) * frp for t in horas for g in GENS_fos)
        #     if usar_ess_5h:
        #         total += sum(C_inv_ESS_5h[e] * frp * m.CAP_ESS_5h[e] for e in ESS_activo_5h)
        #         total += sum(m.P_discharge_5h[t, e] * C_op_ESS_5h[e] * frp for t in horas for e in ESS_activo_5h)
        #     if usar_ess_4h:
        #         total += sum(C_inv_ESS_4h[e] * frp * m.CAP_ESS_4h[e] for e in ESS_activo_4h)
        #         total += sum(m.P_discharge_4h[t, e] * C_op_ESS_4h[e] * frp for t in horas for e in ESS_activo_4h)
        #     if usar_ess_6h:
        #         total += sum(C_inv_ESS_6h[e] * frp * m.CAP_ESS_6h[e] for e in ESS_activo_6h)
        #         total += sum(m.P_discharge_6h[t, e] * C_op_ESS_6h[e] * frp for t in horas for e in ESS_activo_6h)
        #     if usar_meg:
        #         total += sum((C_inv_MEG + C_op_MEG) * frp * m.CAP_MEG[k] for k in K_meg)
        #         total += sum(C_comb_MEG * frp * m.P_MEG[t, k, b] for t in horas for k in K_meg for b in BUSES)
        #     if usar_mess:
        #         total += sum(C_inv_mess * frp * E_cap_mess for k in KM_mess)
        #         total += sum(C_op_mess * frp * model.Pdch_mess[t, k] for t in horas for k in KM_mess)
        #     if usar_reconfig_topo:
        #         total += sum(C_op_SWITCH[l] * m.z_tie[t, l] for t in horas for l in tie_list)
        #     if usar_inv_switch_rcs and sw_cand_list:
        #         total += sum((C_inv_RCS) * frp * m.y_rcs[l] for l in sw_cand_list)
        #         # total += sum((C_inv_RCS + C_opfijo_RCS) * frp * m.y_rcs[l] for l in sw_cand_list)
        #     if usar_inv_switch_ms and sw_cand_list:
        #         total += sum((C_inv_MS) * frp * m.y_ms[l] for l in sw_cand_list)
        #         # total += sum((C_inv_MS + C_opfijo_MS) * frp * m.y_ms[l] for l in sw_cand_list)
        #     # NUEVO: OPEX variable de switches candidatos (solo si alguna inversión activa)
        #     if sw_cand_list and (usar_inv_switch_rcs or usar_inv_switch_ms):
        #         total += sum(C_op_SWITCH[12] * m.z_sw[t, l] for t in horas for l in sw_cand_list)
        #     if usar_edrp:
        #         total += sum(C_incentivo_EDRP[c] * m.P_EDRP[t, b, c] for t in horas for b in BUSES for c in BLOQUES_EDRP) * N_eventos
        #     # total += sum(VoLL[b] * m.P_LC[t, b] for t in horas for b in BUSES) * N_eventos
        #     total += sum(VoLL_local[b] * m.P_LC[t, b] for t in horas for b in BUSES) * N_eventos
        #     total += sum(C_sub[s] * m.P_SUB[t, s] for t in horas for s in S)
        #     return total

        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # ── 6. Solver ────────────────────────────────────────────────────────
        # 8 hilos es el valor histórico y se conserva en cualquier equipo con
        # 8 núcleos o más. Solo baja si el proceso tiene menos ASIGNADOS (un
        # trabajo de SLURM chico): pedir 8 sobre 4 núcleos solo multiplica la
        # memoria de los nodos sin acelerar nada.
        import recursos
        import tempfile
        from optimization_6_oficial import _resolver_con_reintento
        _threads = int(config.get("gurobi_threads")
                       or os.environ.get("ERAD_GUROBI_THREADS")
                       or min(8, recursos.cpus_disponibles()))
        # Interfaz: "lp" (default, la histórica) escribe el modelo a un .lp que
        # Gurobi relee; "direct" (gurobi_direct) y "appsi" (appsi_gurobi) lo
        # pasan en memoria. Medido en El Melón (esc 57, 720 h, 198 barras, 80
        # seccionadores): lp 16,5 s, direct 25,1 s, appsi 25,2 s, con el mismo
        # resultado (|ΔP_LC| = 0, |ΔP_SUB| < 5e-8). Se deja lp.
        _interfaz = config.get("gurobi_interfaz", "lp")
        # Resguardos de optimization_6: SoftMemLimit para que Gurobi termine
        # ordenadamente con su mejor solución en vez de morir por memoria (el
        # 40 % restante es para el modelo de Pyomo), árbol B&B a disco desde la
        # mitad de ese presupuesto, y un TimeLimit alto (24 h, el del servidor).
        _mem_gb = config.get("gurobi_mem_gb") or max(2.0, 0.6 * recursos.ram_disponible_gb())
        _nodefile_dir = (config.get("gurobi_nodefile_dir") or os.environ.get("ERAD_GUROBI_NODEFILE_DIR")
                         or os.path.join(tempfile.gettempdir(), "gurobi_nodefiles"))
        os.makedirs(_nodefile_dir, exist_ok=True)
        _opts = {
            'Method': 2, 'Threads': _threads, 'NumericFocus': 1,
            'MIPGap': 0.02, 'MIPFocus': 1, 'DegenMoves': 0, 'Cuts': 2,
            'SoftMemLimit': float(_mem_gb), 'NodefileStart': float(max(0.5, _mem_gb / 2)),
            'NodefileDir': _nodefile_dir,
            'TimeLimit': float(config.get("gurobi_time_limit", 86400)),
        }

        def _crear_solver():
            s = pyo.SolverFactory({"direct": "gurobi_direct", "appsi": "appsi_gurobi"}.get(_interfaz, "gurobi"))
            s.options.update(_opts)
            return s

        print(f"[RECURSOS] {recursos.resumen()} -> interfaz={_interfaz}, Threads={_threads}, "
              f"SoftMemLimit={_mem_gb:.1f} GB, TimeLimit={_opts['TimeLimit']:.0f} s")
        print(f"[Esc {esc+1}] Resolviendo {len(horas)} h × {len(BUSES)} buses...")

        solver, result = _resolver_con_reintento(
            _crear_solver, model, tee=True,
            intentos=int(config.get("gurobi_reintentos_licencia", 3)),
            espera_s=float(config.get("gurobi_espera_reintento_s", 30)))

        # Un corte por TimeLimit/SoftMemLimit deja un incumbente utilizable: se
        # acepta si Gurobi reporta al menos una solución, y su gap queda en el
        # resultado (solo con la interfaz directa se puede leer).
        _term = str(result.solver.termination_condition)
        try:
            _gap_final = float(solver._solver_model.MIPGap)
            _hay_incumbente = solver._solver_model.SolCount > 0
        except Exception:                       # noqa: BLE001 — interfaz lp
            # Con el .lp no hay modelo de gurobipy que consultar: hay solución
            # si Pyomo le cargó valor a las variables.
            _gap_final = None
            try:
                _hay_incumbente = next(iter(model.P_LC.values())).value is not None
            except Exception:
                _hay_incumbente = False
        if _term not in ("optimal", "feasible") and _hay_incumbente:
            print(f"Escenario {esc+1} - Terminó por '{_term}'; se usa la mejor solución encontrada"
                  + (f" (gap {_gap_final:.2%})." if _gap_final is not None else "."))

        if _term in ("optimal", "feasible") or _hay_incumbente:
            print(f"Escenario {esc+1} - Obj Total: {pyo.value(model.obj):.4f}")

            # ── 7. Extracción de resultados ──────────────────────────────────────
            def _val(v):
                try:
                    r = pyo.value(v)
                    return 0.0 if r is None else r
                except Exception:
                    return 0.0

            theta_vals    = np.array([[_val(model.theta[t, b])    for b in BUSES] for t in horas])
            P_LC_vals     = np.array([[_val(model.P_LC[t, b])     for b in BUSES] for t in horas])
            P_SUB_vals    = np.array([[_val(model.P_SUB[t, s])    for s in S]     for t in horas])
            P_DG_sol_vals = np.array([[_val(model.P_DG_sol[t, g]) for g in GENS]  for t in horas]) if usar_dg_sol else np.zeros((len(horas), len(GENS)))
            CAP_sol_vals  = np.array([_val(model.CAP_DG_sol[g])   for g in GENS]) if usar_dg_sol else np.zeros(len(GENS))
            P_DG_eol_vals = np.array([[_val(model.P_DG_eol[t, g]) for g in GENS]  for t in horas]) if usar_dg_eol else np.zeros((len(horas), len(GENS)))
            CAP_eol_vals  = np.array([_val(model.CAP_DG_eol[g])   for g in GENS]) if usar_dg_eol else np.zeros(len(GENS))
            P_DG_fos_vals = np.array([[_val(model.P_DG_fos[t, g]) for g in GENS]  for t in horas]) if usar_dg_fos else np.zeros((len(horas), len(GENS)))
            CAP_fos_vals  = np.array([_val(model.CAP_DG_fos[g])   for g in GENS]) if usar_dg_fos else np.zeros(len(GENS))
            P_ESS_vals_5h    = np.array([[_val(model.P_discharge_5h[t,e]) - _val(model.P_charge_5h[t,e]) for e in ESS] for t in horas]) if usar_ess_5h else np.zeros((len(horas), len(ESS)))
            SOC_vals_5h      = np.array([[_val(model.SOC_5h[t, e])       for e in ESS]  for t in horas]) if usar_ess_5h else np.zeros((len(horas), len(ESS)))
            P_ESS_vals_4h    = np.array([[_val(model.P_discharge_4h[t,e]) - _val(model.P_charge_4h[t,e]) for e in ESS] for t in horas]) if usar_ess_4h else np.zeros((len(horas), len(ESS)))
            SOC_vals_4h      = np.array([[_val(model.SOC_4h[t, e])       for e in ESS]  for t in horas]) if usar_ess_4h else np.zeros((len(horas), len(ESS)))
            P_ESS_vals_6h    = np.array([[_val(model.P_discharge_6h[t,e]) - _val(model.P_charge_6h[t,e]) for e in ESS] for t in horas]) if usar_ess_6h else np.zeros((len(horas), len(ESS)))
            SOC_vals_6h      = np.array([[_val(model.SOC_6h[t, e])       for e in ESS]  for t in horas]) if usar_ess_6h else np.zeros((len(horas), len(ESS)))
            CAP_ESS_vals_5h = np.array([_val(model.CAP_ESS_5h[e]) for e in ESS]) if usar_ess_5h else np.zeros(len(ESS))
            CAP_ESS_vals_4h = np.array([_val(model.CAP_ESS_4h[e]) for e in ESS]) if usar_ess_4h else np.zeros(len(ESS))
            CAP_ESS_vals_6h = np.array([_val(model.CAP_ESS_6h[e]) for e in ESS]) if usar_ess_6h else np.zeros(len(ESS))
            # Sin reconfiguración las variables no existen, pero la forma de la
            # salida no cambia: ceros sobre `tie_list`. Es lo mismo que salía
            # antes (estaban fijas en 0) y lo que esperan `Pf_calc` y el
            # diccionario de resultados.
            if usar_reconfig_topo:
                z_tie_vals = np.array([[_val(model.z_tie[t, l]) for l in tie_list] for t in horas])
                w_tie_vals = np.array([[_val(model.w_tie[t, l]) for l in tie_list] for t in horas])
                P_tie_vals = np.array([[_val(model.P_tie[t, l]) for l in tie_list] for t in horas])
            else:
                z_tie_vals = w_tie_vals = P_tie_vals = np.zeros((len(horas), len(tie_list)))
            CAP_meg_vals  = np.array([_val(model.CAP_MEG[k])       for k in K])    if usar_meg else np.zeros(len(K))
            x_re_vals     = np.array([[_val(model.x_re[k, b])      for b in BUSES] for k in K]) if usar_meg else np.zeros((len(K), len(BUSES)))
            P_MEG_vals    = np.array([[[_val(model.P_MEG[t, k, b]) for b in BUSES] for k in K] for t in horas]) if usar_meg else np.zeros((len(horas), len(K), len(BUSES)))
            u_mess_vals = np.array([[[ _val(model.u_mess[k, b, t]) for b in BUSES] for k in KM_mess] for t in horas]) if usar_mess else np.zeros((len(horas), n_mess, len(BUSES)))
            P_mess_vals = np.array([[[ _val(model.psi_mess_dch[t, k, b]) - _val(model.psi_mess_ch[t, k, b]) for b in BUSES] for k in KM_mess] for t in horas]) if usar_mess else np.zeros((len(horas), n_mess, len(BUSES)))
            SOC_mess_vals = np.array([[ _val(model.SOC_mess[t, k]) for k in KM_mess] for t in horas]) if usar_mess else np.zeros((len(horas), n_mess))
            P_EDRP_vals = np.array([[[pyo.value(model.P_EDRP[t, b, c]) or 0.0 for c in BLOQUES_EDRP] for b in BUSES] for t in horas]) if usar_edrp else np.zeros((len(horas), len(BUSES), C_bloques_EDRP))
            if sw_var_list:
                z_sw_vals  = np.array([[_val(model.z_sw[t, l])  for l in sw_var_list] for t in horas])
                P_sw_vals  = np.array([[_val(model.P_sw[t, l])  for l in sw_var_list] for t in horas])
                y_rcs_vals = np.array([round(_val(model.y_rcs[l])) for l in sw_var_list])
                y_ms_vals  = np.array([round(_val(model.y_ms[l]))  for l in sw_var_list])
            elif sw_cand_list:
                # Hay switches candidatos pero no se invierte en ellos: ceros con
                # la forma de siempre, que es lo que indexan los diccionarios de
                # abajo sobre `sw_cand_list`.
                z_sw_vals = P_sw_vals = np.zeros((len(horas), len(sw_cand_list)))
                y_rcs_vals = y_ms_vals = np.zeros(len(sw_cand_list))
            else:
                z_sw_vals = P_sw_vals = y_rcs_vals = y_ms_vals = np.array([])
            

            for i, h in enumerate(horas):
                Dam_tl = Dam_tl_por_hora[h]
                Pf_calc = {}
                for l in LINES:
                    fi, tj = from_bus[l], to_bus[l]
                    if Dam_tl[l] == 1:
                        Pf_calc[(fi, tj)] = 0.0
                    elif l in tie_indices:
                        Pf_calc[(fi, tj)] = P_tie_vals[i, tie_list.index(l)]
                    else:
                        Pf_calc[(fi, tj)] = (theta_vals[i, bus_pos[fi]] - theta_vals[i, bus_pos[tj]]) / x_ij[l]

                P_D_hora = {b: P_D_hora_arr[h, b] for b in BUSES}
                ENS_h    = float(P_LC_vals[i].sum())
                carga_servida = sum(P_D_hora[b] - P_LC_vals[i, bi] for bi, b in enumerate(BUSES))
                # `lineas_danadas_*` mantiene su significado histórico: las
                # líneas físicamente averiadas. `lineas_fuera_servicio_*`
                # agrega las que quedan sin servicio solo por la zona de
                # aislación (se reponen cerrando el seccionador).
                lineas_danadas_indices = np.where(Dam_fis_por_hora[h] == 1)[0].tolist()
                lineas_fuera_indices   = np.where(Dam_tl == 1)[0].tolist()
                # seccionadas = frontera de la zona, se abrieron por maniobra.
                # aisladas    = sin seccionador, perdieron suministro por
                #               arrastre. Ver optimization_6_oficial.
                _dan = set(lineas_danadas_indices)
                lineas_seccionadas_indices = [x for x in lineas_fuera_indices
                                              if x not in _dan and x in sec_existentes]
                lineas_aisladas_indices = [x for x in lineas_fuera_indices
                                           if x not in _dan and x not in sec_existentes]

                results_por_esc[esc].append({
                    'escenario': esc, 'hora': h + 1,
                    'ENS_MWh': ENS_h, 'carga_servida_MW': carga_servida,
                    'n°lineas_danadas': len(lineas_danadas_indices),
                    'lineas_danadas_indices': lineas_danadas_indices,
                    'lineas_danadas_nombres': [net.line.at[x,'name'] if x in net.line.index else f"L{x}" for x in lineas_danadas_indices],
                    'n°lineas_fuera_servicio': len(lineas_fuera_indices),
                    'lineas_fuera_servicio_indices': lineas_fuera_indices,
                    'lineas_seccionadas_indices': lineas_seccionadas_indices,
                    'lineas_aisladas_indices': lineas_aisladas_indices,
                    'seccionadores': sorted(sec_existentes) if usar_aislacion_fija else [],
                    'status': 'optimal',
                    'solver_term': _term,
                    'solver_gap': _gap_final,
                    'theta':       {b: theta_vals[i, bi] * 180/np.pi     for bi, b in enumerate(BUSES)},
                    'Pb':          {b: P_D_hora[b] - P_LC_vals[i, bi]    for bi, b in enumerate(BUSES)},
                    'P_LC':        {b: round(P_LC_vals[i, bi], 5)        for bi, b in enumerate(BUSES)},
                    'Pf':          Pf_calc,
                    'P_SUB':       {s: P_SUB_vals[i, si]                 for si, s in enumerate(S)},
                    'P_DG_sol':    {g: P_DG_sol_vals[i, gi]              for gi, g in enumerate(GENS)},
                    'CAP_DG_sol_kW': {g: CAP_sol_vals[gi]                for gi, g in enumerate(GENS)},
                    'P_DG_eol':    {g: P_DG_eol_vals[i, gi]              for gi, g in enumerate(GENS)},
                    'CAP_DG_eol_kW': {g: CAP_eol_vals[gi]                for gi, g in enumerate(GENS)},
                    'P_DG_fos':    {g: P_DG_fos_vals[i, gi]              for gi, g in enumerate(GENS)},
                    'CAP_DG_fos_kW': {g: CAP_fos_vals[gi]                for gi, g in enumerate(GENS)},
                    'P_ESS_5h':       {e: P_ESS_vals_5h[i, ei]           for ei, e in enumerate(ESS)},
                    'SOC_MWh_5h':     {e: SOC_vals_5h[i, ei]             for ei, e in enumerate(ESS)},
                    'P_ESS_4h':       {e: P_ESS_vals_4h[i, ei]           for ei, e in enumerate(ESS)},
                    'SOC_MWh_4h':     {e: SOC_vals_4h[i, ei]             for ei, e in enumerate(ESS)},
                    'P_ESS_6h':       {e: P_ESS_vals_6h[i, ei]           for ei, e in enumerate(ESS)},
                    'SOC_MWh_6h':     {e: SOC_vals_6h[i, ei]             for ei, e in enumerate(ESS)},    
                    'CAP_ESS_kWh_5h': {e: CAP_ESS_vals_5h[ei]            for ei, e in enumerate(ESS)},
                    'CAP_ESS_kWh_4h': {e: CAP_ESS_vals_4h[ei]            for ei, e in enumerate(ESS)},
                    'CAP_ESS_kWh_6h': {e: CAP_ESS_vals_6h[ei]            for ei, e in enumerate(ESS)},
                    'z_tie':       {l: round(z_tie_vals[i, li])          for li, l in enumerate(tie_list)},
                    'w_tie':       {l: round(w_tie_vals[i, li])          for li, l in enumerate(tie_list)},
                    'P_tie':       {l: P_tie_vals[i, li]                 for li, l in enumerate(tie_list)},
                    'CAP_MEG_MW':  {k: CAP_meg_vals[ki]                  for ki, k in enumerate(K)},
                    'x_re':        {(k,b): round(x_re_vals[ki,bi])       for ki,k in enumerate(K) for bi,b in enumerate(BUSES)},
                    'P_MEG':       {(k,b): P_MEG_vals[i,ki,bi]           for ki,k in enumerate(K) for bi,b in enumerate(BUSES)},
                    'u_mess':   {(k,b): round(u_mess_vals[i,ki,bi]) for ki,k in enumerate(KM) for bi,b in enumerate(BUSES)},
                    'P_mess':   {(k,b): P_mess_vals[i,ki,bi]        for ki,k in enumerate(KM) for bi,b in enumerate(BUSES)},
                    'SOC_mess': {k: SOC_mess_vals[i,ki]              for ki,k in enumerate(KM)},
                    'H_MESS_h': {k: h_mess for k in KM_mess},
                    'CAP_MESS_MWh': {k: E_cap_mess for k in KM_mess},
                    'P_MESS_MAX_MW': {k: P_mess_max for k in KM_mess},
                    'P_EDRP': {(b, c): P_EDRP_vals[i, bi, c]             for bi, b in enumerate(BUSES) for c in range(C_bloques_EDRP)},
                    'EDRP_total': {b: sum(P_EDRP_vals[i, bi, c]          for c in range(C_bloques_EDRP)) for bi, b in enumerate(BUSES)},
                    'z_sw':    {l: round(z_sw_vals[i, li])  for li, l in enumerate(sw_cand_list)},
                    'P_sw':    {l: P_sw_vals[i, li]         for li, l in enumerate(sw_cand_list)},
                    'y_rcs':   {l: y_rcs_vals[li]           for li, l in enumerate(sw_cand_list)},
                    'y_ms':    {l: y_ms_vals[li]            for li, l in enumerate(sw_cand_list)},
                    'VoLL':    VoLL_local,
                })
                        # print(f"  [Hora {h+1:>4}/{len(horas)}] Servida: {carga_servida:.3f} MW | "
                        #   f"LC: {P_LC_vals[i].sum():.3f} MW | Líneas dañadas: {num_danadas}")
        else:
            print(f"Escenario {esc+1} - Error: {result.solver.termination_condition}")

    for pos, esc in enumerate(escenarios):
        resolver_esc(pos, esc)
        
    # ── CACHÉ: guardar resultados para reutilizarlos ─────────────────────────
    _results_finales = {esc: res for esc, res in results_por_esc.items() if res}
    if _cache_file:
        with open(_cache_file, "wb") as _f:
            pickle.dump({"results_por_esc": _results_finales}, _f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[CACHÉ] Resultados guardados en '{_cache_file}'.")

    return  _results_finales

