# optimization_6_plan_fijo.py
# ============================================================
# COPIA de optimization_6_oficial.py dedicada al POSPROCESAMIENTO.
# Única diferencia: acepta config["plan_fijo"] (buscar "PLAN DE INVERSIÓN
# IMPUESTO"), que fija la inversión de un plan ya resuelto y deja libre la
# operación. La usa run_plan_fijo.py para evaluar el plan del escenario i en
# el escenario j. La app NO la importa.
#
# ⚠️ Si se corrige optimization_6_oficial.py, hay que replicar el cambio acá,
# o la matriz plan × escenario deja de ser comparable con el barrido.
# Copiada el 2026-09-15; resincronizada con el oficial el 2026-10-25
# (flota MEG existente `meg_existentes`, MEG arrendado y frp_meg).
# ============================================================
import math
import numpy as np
import pyomo.environ as pyo
import pickle
import os
import tempfile
import time

import recursos

# Re-exportados a propósito: la app los importa desde acá desde antes de
# que existiera `aislacion.py`.
from radialidad import restricciones_radiales
from aislacion import (  # noqa: F401
    adyacencia_lineas, zona_aislada, expandir_dano,
    leer_config as leer_config_aislacion,
)
from limites_dg import (
    leer_config as leer_config_limites_dg,
    aplicar_tope as aplicar_tope_dg,
    aplicar_tope_conjunto as aplicar_tope_dg_conjunto,
)
from ramales import piso_corte as piso_corte_ramales


def _ram_disponible_gb(default=8.0):
    """RAM que este proceso puede usar, en GB.

    En Windows es la RAM física, como siempre. En un servidor Linux es el
    mínimo con el límite del cgroup / trabajo SLURM: ver `recursos.py`.
    """
    return recursos.ram_disponible_gb(default)


def _es_error_de_licencia(e):
    """True si la excepción del solver viene de la licencia WLS, no del modelo."""
    msg = str(e).lower()
    return ("token" in msg or "license" in msg or "licencia" in msg
            or "wls" in msg)


def _resolver_y_cargar(solver, model, tee):
    """`solver.solve` que NO pierde el incumbente cuando Gurobi corta.

    Pyomo traduce los estados de Gurobi que no conoce (MEM_LIMIT por
    SoftMemLimit, INTERRUPTED, NUMERIC) a `TerminationCondition.error`, y con la
    carga automática de la solución `solve` levanta `ValueError: Cannot load a
    SolverResults object with bad status: error` aunque Gurobi tenga un
    incumbente. Eso pasó en una combinación de 11 medidas (9 h de cómputo y 8
    soluciones, perdidas): el `except` del optimizador lo tomó por un fallo del
    solver y el chequeo de `SolCount` nunca llegó a ejecutarse.

    Con `gurobi_direct` se resuelve sin cargar y las variables se cargan a mano
    si Gurobi reporta al menos una solución, sea cual sea el estado con que
    terminó. Quien llama decide, mirando `termination_condition` y el gap, si
    esa solución sirve. Las otras interfaces (.lp, appsi) siguen igual.
    """
    if type(solver).__name__ != "GurobiDirect":
        return solver.solve(model, tee=tee)
    result = solver.solve(model, tee=tee, load_solutions=False)
    try:
        n_sol = int(solver._solver_model.SolCount)
    except Exception:                               # noqa: BLE001
        n_sol = 0
    if n_sol > 0:
        solver.load_vars()
    return result


def _resolver_con_reintento(crear_solver, model, tee=True, intentos=3, espera_s=30):
    """Resuelve reintentando cuando Gurobi falla por token WLS vencido.

    Con licencia WLS el token se renueva contra el servidor de Gurobi mientras
    el entorno por defecto está vivo. En un barrido largo (varios escenarios de
    8760 h, hasta 1800 s cada uno) basta un corte de red en el equipo remoto
    para que la renovación falle y el escenario siguiente muera con
    "Token expired" al crear el modelo. Descartar el entorno por defecto obliga
    a gurobipy a pedir un token nuevo en el reintento.

    Devuelve (solver, result) del intento que sí resolvió; cualquier error que
    no sea de licencia se propaga tal cual.
    """
    ultimo = None
    for intento in range(1, intentos + 1):
        solver = crear_solver()
        try:
            return solver, _resolver_y_cargar(solver, model, tee)
        except Exception as e:
            if not _es_error_de_licencia(e):
                raise
            ultimo = e
            print(f"[licencia] Intento {intento}/{intentos} falló ({e}). "
                  "Descartando el entorno de Gurobi para pedir un token nuevo...")
            try:
                import gurobipy
                gurobipy.disposeDefaultEnv()
            except Exception:
                pass
            if intento < intentos:
                time.sleep(espera_s)
    raise ultimo


def resolver_optimizacion(data, config):

    _cache_file = config.get("cache_file", None)
    if _cache_file and os.path.exists(_cache_file):
        with open(_cache_file, "rb") as _f:
            _cached = pickle.load(_f)
        return _cached["results_por_esc"]

    net           = data["net"]
    # snet          = data["snet"]
    faultlinMC    = data["faultlinMC"]
    faultlinMC_veg = data.get("faultlinMC_veg")
    faultlinMC_ref = data.get("faultlinMC_ref")
    faultlinMC_ug  = data.get("faultlinMC_ug")
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
    # Tope de capacidad de DG, como fracción de la demanda base. El grupo
    # sobre el que se agrega lo decide `lim_dg_modo` (ver `limites_dg.py`);
    # acá solo se deja leído el valor para lo que lo consulte antes de
    # construir las restricciones. Tiene que coincidir con el de
    # optimization_base (que calcula E4/E5), o la estrategia se compara
    # contra un caso base construido con otra física.
    Lim = float(config.get("lim_dg", 1))

    #Subestación
    # Baseline de compra a la subestación, POR ESCENARIO: el crédito del
    # objetivo compara contra el caso base DEL MISMO escenario que se está
    # resolviendo. Antes era un único escalar (el del escenario congelado)
    # que se aplicaba a cualquier escenario, así que al mover el selector
    # de escenario el crédito quedaba referido al baseline equivocado.
    P_sub_total_por_esc = data.get("P_SUB_tot_por_esc", {})
    P_sub_total_global = data.get("P_SUB_tot", 0.0) 

    #Generación distribuida
    CAPEX_DG_sol_sin_fallas = data.get("CAPEX_DG_sol_sin_fallas")
    OPEX_DG_sol_sin_fallas = data.get("OPEX_DG_sol_sin_fallas")
    CAPEX_DG_eol_sin_fallas = data.get("CAPEX_DG_eol_sin_fallas")
    OPEX_DG_eol_sin_fallas = data.get("OPEX_DG_eol_sin_fallas")
    # Referencia conjunta (E4/4). None si el artefacto de E4 es anterior a la
    # corrección de §8.2; en ese caso el objetivo usa las dos individuales,
    # que es el comportamiento histórico.
    CAPEX_DG_ambas_sin_fallas = data.get("CAPEX_DG_ambas_sin_fallas")
    OPEX_DG_ambas_sin_fallas = data.get("OPEX_DG_ambas_sin_fallas")
    print("DG SOLAR", CAPEX_DG_sol_sin_fallas,OPEX_DG_sol_sin_fallas)




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
    C_sub     = data["C_sub"]
    # P_sub_tot = data["results_sin"][]



    # Parámetros económicos
    r = data["r"]
    n = data["años"]
    frp = r * (1+r)**n / ((1+r)**n - 1)
    frp_movil = data["frp_movil"]

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

    # RECONFIGURACIÓN TOPOLÓGICA
    # Las tie-lines son las líneas NORMALMENTE ABIERTAS de la red: en
    # pandapower, las que llevan un switch abierto. No son "las últimas
    # tres" — eso era cierto solo por convención de la red del estudio, y
    # en una red arbitraria deja como enlace a la línea que quedó al final
    # de la tabla. Por eso ahora vienen en la config (E0 las guarda en
    # red.pkl y `pipeline.config_medida` las pasa).
    # El respaldo `LINES[-3:]` mantiene el comportamiento histórico para
    # configs antiguas: en la red de 15 barras `tie_indices` ya vale
    # [12, 13, 14], que son exactamente las últimas tres. Solo si la clave
    # FALTA: una lista vacía es una red sin enlaces de respaldo, y con `or`
    # dejaba abiertas tres líneas cualesquiera (carga desconectada sin falla).
    tie_indices = set(config["tie_indices"] if config.get("tie_indices") is not None
                      else LINES[-3:])
    usar_reconfig_topo = config.get("reconfig_topo", False)
    C_inv_SWITCH = data["SWITCH"][0]
    C_op_SWITCH = data["SWITCH"][1]

    # INVERSIÓN EN LÍNEAS DE ENLACE (tie-lines NUEVAS)
    # — medida independiente de `reconfig_topo`. La diferencia conceptual:
    #   * reconfig_topo : las tie-lines YA EXISTEN (normalmente abiertas);
    #     el modelo solo decide cuándo cerrarlas y paga la maniobra.
    #   * inv_tie       : tie_cand_list es la lista de enlaces CANDIDATOS a
    #     construir. y_tie[l] ∈ {0,1} decide si se invierte en ese enlace, y
    #     solo un enlace construido puede cerrarse (z_tie <= y_tie). Se paga
    #     CAPEX + OPEX fijo del enlace por km, y además la maniobra (igual
    #     que reconfig).
    # Ambas comparten las variables z_tie/w_tie/P_tie y el flujo DC: la
    # inversión solo agrega la capa de decisión y el costo de capital.
    usar_inv_tie   = config.get("inv_tie", False)
    tie_cand_list  = [l for l in config.get("tie_candidatas", sorted(tie_indices))
                      if l in set(LINES)]
    # Un enlace candidato NO puede formar parte del flujo "siempre cerrado":
    # mientras no se construya no existe. Por eso las candidatas se suman al
    # universo de tie-lines (se sacan de activas_por_hora y de line_limits, y
    # su flujo pasa por P_tie, gobernado por z_tie <= y_tie).
    if usar_inv_tie:
        tie_indices = tie_indices | set(tie_cand_list)
    tie_list = sorted(tie_indices)
    # Una línea no puede ser a la vez enlace candidato (flujo por P_tie) y
    # candidata de switch/hardening (flujo por P_sw): se contaría dos veces
    # en el balance. El enlace manda.
    _sw_solapadas = [l for l in sw_cand_list if l in tie_indices]
    if _sw_solapadas:
        print(f"[!] Líneas {_sw_solapadas} son enlaces candidatos y además "
              f"candidatas de switch/hardening; se excluyen de sw_cand_list.")
        sw_cand_list = [l for l in sw_cand_list if l not in tie_indices]
    # M_tie = max(cap.values()) * 10
    # Cota de los ángulos (rad); por defecto π/2. El big-M sigue a la cota porque
    # |θi-θj| puede llegar a 2·THETA_MAX. Igual que optimization_base.
    THETA_MAX = float(config.get("theta_max", np.pi / 2))
    M_tie = {l: 2 * THETA_MAX / x_ij[l] for l in tie_list}
    # Un enlace nuevo queda expuesto a las mismas fallas que el resto de la
    # red (faultlinMC ya cubre todas las líneas). Ponerlo en False recupera
    # el supuesto de reconfig_topo (tie-lines inmunes a la falla).
    tie_expuesta_falla = config.get("tie_expuesta_a_falla", True)
    C_inv_TIE = data.get("TIE", [0.0, 0.0])[0]   # USD/km — CAPEX línea nueva
    C_op_TIE  = data.get("TIE", [0.0, 0.0])[1]   # USD/km·año — OPEX fijo

    # INVERSIÓN RCS O MS
    usar_inv_switch_rcs = config.get("inv_switch_rcs", False)
    usar_inv_switch_ms  = config.get("inv_switch_ms",  False)
    T_maniobra_MS       = config.get("T_maniobra_MS",  1)    # horas
    T_maniobra_tie      = int(config.get("T_maniobra_tie", 0))   # horas; 0 = instantánea
    radial_tie          = bool(config.get("radial_tie", False))  # cierre de enlaces sin mallar la red
    Conjunto_ij = [0,1]
    C_inv_RCS    = data["RCS"][0]
    C_opfijo_RCS = data["RCS"][1]
    C_inv_MS     = data["MS"][0]
    C_opfijo_MS  = data["MS"][1]
    M_sw = {l: 2 * THETA_MAX / x_ij[l] for l in sw_cand_list}
    bus_zone = data.get("bus_zone", {})  # ej: {0:1, 1:1, 2:1, 3:2, 4:2, ...}

    # ── AISLACIÓN DE LA FALLA POR SECCIONADORES PREEXISTENTES ───────────
    # Sin esta opción el modelo saca de servicio ÚNICAMENTE la línea
    # averiada, lo que equivale a suponer un seccionador en cada extremo
    # de cada línea: la falla se aísla siempre, gratis y al instante. Es
    # un supuesto optimista que subestima ENS/SAIDI/SAIFI y, sobre todo,
    # deja sin beneficio a la inversión en switches (y_rcs / y_ms), que
    # en ese mundo solo sirven para abrir líneas sanas.
    #
    # Con `aislacion_fallas_fija = True` se modela el comportamiento real:
    # los seccionadores particionan el alimentador en secciones y una
    # falla dentro de una sección deja FUERA A TODA LA SECCIÓN, porque la
    # protección aguas arriba despeja y la reposición solo alcanza hasta
    # el seccionador más cercano (zona de aislación / protection zone).
    #
    # `seccionadores_existentes` es la lista de líneas que llevan
    # seccionador (índices de LINES). Es un DATO, no una decisión: por eso
    # la zona se calcula en Python antes de armar el MILP y no agrega ni
    # una variable binaria. La versión endógena (elegir dónde invertir en
    # seccionadores nuevos) requiere la variable de zona zeta[t,l] y las
    # restricciones de propagación, y no está incluida acá.
    usar_aislacion_fija, sec_existentes = leer_config_aislacion(config, data, LINES)

    # GESTIÓN DE VEGETACIÓN / REFUERZO ESTRUCTURAL / SOTERRAMIENTO
    # — decisión por línea dentro del MILP (igual que RCS/MS): el solver
    # elige a qué líneas candidatas (sw_cand_list) conviene aplicar cada
    # medida, sopesando su CAPEX/OPEX por km contra el ENS evitado.
    usar_veg_mgmt      = config.get("veg_mgmt", False)
    usar_refuerzo      = config.get("refuerzo", False)
    usar_soterramiento = config.get("soterramiento", False)
    longitud_km_linea  = data.get("longitud_km_linea", {})
    C_inv_veg = data.get("VEG", [0, 0])[0]
    C_op_veg  = data.get("VEG", [0, 0])[1]
    C_inv_ref = data.get("REF", [0, 0])[0]
    C_op_ref  = data.get("REF", [0, 0])[1]
    C_inv_ug  = data.get("UG",  [0, 0])[0]
    C_op_ug   = data.get("UG",  [0, 0])[1]

    # MEG
    usar_meg   = config.get("meg", False)
    n_meg      = config.get("n_meg", 2)
    K          = list(range(n_meg))
    P_meg_max  = config.get("P_meg_max", 1.0)
    # P_meg_tot  = config.get("P_meg_tot", 2.0)   
    C_inv_MEG = data["MEG"][0]                  # USD/MW — CAPEX
    C_op_MEG   = data["MEG"][1]   # USD/MW·año — OPEX fijo
    C_comb_MEG = data["MEG"][2]   # USD/MWh    — diésel variable
    # Anualidad del CAPEX del MEG comprado. Sin la clave `frp_meg` en los
    # costos del caso se usa `frp`, como siempre (validación idéntica).
    frp_meg = data.get("frp_meg", frp)
    # Las primeras `meg_existentes` unidades de K ya son de la empresa: sin
    # CAPEX, y una unidad comprada solo se ubica si las anteriores ya se
    # ubicaron (restricción de orden más abajo). En 0, el modelo de siempre.
    n_meg_exist = min(int(config.get("meg_existentes", 0) or 0), n_meg)

    # tiempo de traslado en horas; puede ser escalar o diccionario por bus
    T_tra_meg_cfg = config.get("T_tra_meg", 2.0)
    if isinstance(T_tra_meg_cfg, (int, float)): T_tra_meg = {b: int(T_tra_meg_cfg) for b in BUSES}
    else:
        T_tra_meg = {b: int(T_tra_meg_cfg.get(b, 2)) for b in BUSES}

    # MEG ARRENDADO a un tercero: se paga por evento y por unidad
    # (`MEG_arriendo` = [USD/evento]) y el pago incluye el combustible, así
    # que no hay CAPEX ni costo variable. Cada unidad sirve a lo más
    # `P_meg_arr` kW durante `h_meg_arr` horas desde su llegada, y puede ir
    # a un bus distinto en cada evento (se arrienda cuando ocurre).
    usar_meg_arr = config.get("meg_arriendo", False)
    n_meg_arr    = int(config.get("n_meg_arr", 1) or 0)
    KA           = list(range(n_meg_arr))
    P_meg_arr    = float(config.get("P_meg_arr", 200.0))
    h_meg_arr    = int(config.get("h_meg_arr", 6))
    C_evento_meg_arr = data.get("MEG_arriendo", [0.0])[0]
    T_tra_arr_cfg = config.get("T_tra_meg_arr", T_tra_meg_cfg)
    if isinstance(T_tra_arr_cfg, (int, float)): T_tra_meg_arr = {b: int(T_tra_arr_cfg) for b in BUSES}
    else:
        T_tra_meg_arr = {b: int(T_tra_arr_cfg.get(b, 2)) for b in BUSES}

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
    SOC_min_mess    = config.get("SOC_min_mess", 0)
    SOC_max_mess    = config.get("SOC_max_mess", 0.9)
    # C_inv_mess      = data.get("MESS", [537.5, 5.0])[0]  # USD/kWh CAPEX
    # C_op_mess       = data.get("MESS", [537.5, 5.0])[1]  # USD/kWh OPEX variable
    C_inv_mess      = data.get("MESS", [])[0]  # USD/kWh CAPEX
    C_op_mess       = data.get("MESS", [])[1]  # USD/kWh OPEX variable
    C_tra_mess       = data.get("MESS", [])[2]  # USD/kWh OPEX variable

    # Tiempo de tránsito hacia cada bus (horas) — reutiliza T_tra_meg o define propio
    T_tra_mess_cfg  = config.get("T_tra_mess", 2.0)
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
        # Baseline de subestación del escenario que se está resolviendo
        P_sub_total = P_sub_total_por_esc.get(esc, P_sub_total_global)
        if esc not in P_sub_total_por_esc:
            print(f"[!] Escenario {esc} sin baseline propio de P_SUB; se usa el "
                  f"del escenario congelado. Correr: python pipeline.py "
                  f"--etapas E5 --escenarios {esc}")

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

        # Mismo cálculo de daño acumulado, pero usando el faultlinMC ya
        # adelgazado de cada medida — permite que la decisión (x_veg/x_ref/
        # x_ug, por línea) elija, para cada línea candidata, entre el daño
        # original y el daño con la medida aplicada.
        def _acumular_dano(matriz):
            acum = np.zeros(len(LINES))
            por_hora = {}
            for h in horas:
                nuevas = matriz[h, esc, :]
                reps   = repairMC[h, esc, :]
                for l in range(len(LINES)):
                    if reps[l] == 1:
                        acum[l] = 0
                acum = np.maximum(acum, nuevas)
                por_hora[h] = acum.copy()
            return por_hora

        Dam_veg_por_hora = _acumular_dano(faultlinMC_veg) if (usar_veg_mgmt and faultlinMC_veg is not None) else None
        Dam_ref_por_hora = _acumular_dano(faultlinMC_ref) if (usar_refuerzo and faultlinMC_ref is not None) else None
        Dam_ug_por_hora  = _acumular_dano(faultlinMC_ug)  if (usar_soterramiento and faultlinMC_ug  is not None) else None

        def _dano_efectivo(m, t, l):
            d = Dam_tl_por_hora[t][l]
            if Dam_veg_por_hora is not None and l in sw_cand_list:
                d = d - m.x_veg[l] * (Dam_tl_por_hora[t][l] - Dam_veg_por_hora[t][l])
            if Dam_ref_por_hora is not None and l in sw_cand_list:
                d = d - m.x_ref[l] * (Dam_tl_por_hora[t][l] - Dam_ref_por_hora[t][l])
            if Dam_ug_por_hora is not None and l in sw_cand_list:
                d = d - m.x_ug[l]  * (Dam_tl_por_hora[t][l] - Dam_ug_por_hora[t][l])
            return d

        # ── 2. Pre-cómputos (fuera del modelo) ───────────────────────────────
        bus_pos     = {b: i for i, b in enumerate(BUSES)}
        from_bus    = {l: int(net.line.from_bus[l]) for l in LINES}
        to_bus      = {l: int(net.line.to_bus[l])   for l in LINES}

        # ── 2b. Zona de aislación de la falla ────────────────────────────────
        # Daño FÍSICO: las líneas realmente averiadas. Se guarda aparte
        # porque `Dam_tl_por_hora` pasa a representar, más abajo, las
        # líneas FUERA DE SERVICIO (averiadas + arrastradas por la
        # aislación), que es lo que ve el modelo. La distinción importa
        # para reportar: las cuadrillas reparan las averiadas, mientras
        # que las aisladas se reponen con solo cerrar el seccionador.
        Dam_fis_por_hora = {t: Dam_tl_por_hora[t].copy() for t in horas}

        lineas_en_bus = adyacencia_lineas(LINES, from_bus, to_bus)

        def _zona_aislada(dam_vec):
            return zona_aislada(dam_vec, LINES, from_bus, to_bus,
                                lineas_en_bus, sec_existentes, tie_indices)

        if usar_aislacion_fija:
            def _expandir_zonas(por_hora):
                if por_hora is None:
                    return None
                return {t: _zona_aislada(por_hora[t]) for t in horas}

            # La zona se recalcula para cada matriz de daño: si el
            # hardening evita la avería de una línea, la sección afectada
            # también se achica, y `_dano_efectivo` compara contra la zona
            # correcta. (La comparación sigue siendo línea a línea, así
            # que con varias medidas de hardening simultáneas el crédito
            # por zona es aproximado — igual que en el resto del módulo.)
            Dam_tl_por_hora  = _expandir_zonas(Dam_tl_por_hora)
            Dam_veg_por_hora = _expandir_zonas(Dam_veg_por_hora)
            Dam_ref_por_hora = _expandir_zonas(Dam_ref_por_hora)
            Dam_ug_por_hora  = _expandir_zonas(Dam_ug_por_hora)

            _n_fis = sum(int(Dam_fis_por_hora[t].sum()) for t in horas)
            _n_fs  = sum(int(Dam_tl_por_hora[t].sum())  for t in horas)
            print(f"[AISLACIÓN] Esc {esc}: seccionadores en {sorted(sec_existentes)} "
                  f"→ {_n_fis} línea·hora averiadas se traducen en {_n_fs} "
                  f"línea·hora fuera de servicio.")
            if not sec_existentes:
                print("[AISLACIÓN] ⚠️ Sin seccionadores declarados: cualquier "
                      "falla deja fuera al alimentador completo hasta su "
                      "reparación (solo despeja el interruptor de cabecera).")

        # Cualquiera de estas 5 medidas requiere que el daño de las líneas
        # candidatas (sw_cand_list) se resuelva DENTRO del MILP (vía P_sw/
        # z_sw), porque el daño puede depender de una decisión (x_veg/
        # x_ref/x_ug) y ya no es un dato fijo que se pueda filtrar en
        # Python antes de construir el modelo.
        usar_candidatas_dano = (
            usar_inv_switch_rcs or usar_inv_switch_ms
            or usar_veg_mgmt or usar_refuerzo or usar_soterramiento
        )

        # Líneas activas por hora (solo depende de t, no de b) — 15x menos trabajo
        activas_por_hora = {t: [l for l in LINES if l not in tie_indices and Dam_tl_por_hora[t][l] == 0] for t in horas}
        if usar_candidatas_dano:
            activas_por_hora = {t: [l for l in LINES if l not in tie_indices and l not in sw_cand_list and Dam_tl_por_hora[t][l] == 0] for t in horas}

        # Las candidatas solo entran por P_sw cuando alguna de las 5 medidas
        # está habilitada. Se incluyen TODAS (sin filtrar por el daño
        # original) porque el daño efectivo puede depender de x_veg/x_ref/
        # x_ug; las restricciones (P_sw <= cap*z_sw, z_sw <= 1-daño_efectivo)
        # ya fuerzan flujo cero cuando corresponde.
        if usar_candidatas_dano:
            sw_cand_activas_por_hora = {t: list(sw_cand_list) for t in horas}
        else:
            sw_cand_activas_por_hora = {t: [] for t in horas}

        # Inflow/outflow por (t, b): cada línea se recorre UNA vez por hora y se
        # deposita en su barra, en vez de barrer la lista completa de líneas por
        # cada barra. Es O(T·L) en lugar de O(T·B·L) — con 179 barras y 8760 h,
        # 3,2 millones de operaciones en vez de 567 millones. El orden dentro de
        # cada lista se conserva, así que las expresiones del balance salen
        # término por término idénticas a las de antes.
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

        # Ramales con fusible fundidos (ver ramales.py): fracción de la demanda
        # de la barra que se corta sí o sí. Vacío si el caso no tiene ramales.
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

        # MEG arrendado: ventana de servicio de cada evento y bus, y para cada
        # (hora, bus) los eventos cuya ventana la cubre. Solo existen variables
        # en esas horas, igual que el MESS.
        EV_arr = list(range(len(bloques_evento))) if usar_meg_arr else []
        KA_act = KA if usar_meg_arr else []
        arr_eventos_tb = {}
        for e in EV_arr:
            h_ini = bloques_evento[e][0]
            for b in BUSES:
                llegada = int(h_ini + T_tra_meg_arr.get(b, 2))
                for t in range(llegada, min(llegada + h_meg_arr, Horastot)):
                    if t in horas:
                        arr_eventos_tb.setdefault((t, b), []).append(e)
        idx_arr = [(t, k, b) for (t, b) in arr_eventos_tb for k in KA_act]
        arr_k_en_tb = {(t, b): KA_act for (t, b) in arr_eventos_tb} if KA_act else {}

        hora_inicio_evento = bloques_evento[0][0] if bloques_evento else None
        hora_fin_evento = bloques_evento[-1][1] if bloques_evento else None

        # ══════════════════════════════════════════════════════
        # MESS — Ventana temporal reducida por evento (+ pre-carga)
        # ══════════════════════════════════════════════════════
        # Fuera de un evento no hay P_LC que aliviar (no hay costo de arbitraje
        # en el objetivo), así que el óptimo trivialmente pone Pch=Pdch=0. Por
        # eso acotar las variables/restricciones del MESS a la ventana de
        # evento (+ una ventana previa de carga) es una reducción exacta del
        # modelo, no una aproximación: el resultado óptimo no cambia, solo se
        # evita crear/resolver variables que el solver fijaría en cero igual.
        #
        # La ventana previa de carga (T_carga_mess) le permite al modelo
        # decidir la trayectoria de carga antes de cada evento —conocido de
        # antemano dentro de este escenario— respetando la tasa real de carga,
        # en vez de asumir un SOC arbitrario al inicio del evento.
        KM_mess = KM if usar_mess else []
        T_tra_max = max(T_tra_mess.values()) if T_tra_mess else 0
        T_carga_mess = math.ceil(h_mess / eta_ch_mess) if (usar_mess and eta_ch_mess > 0) else 0
        # Horas de "recuperación" que necesita el MESS entre dos eventos para
        # volver a su base (T_tra_max) y recargarse por completo (T_carga_mess)
        # antes de la ventana de pre-carga del siguiente evento.
        recovery_mess = T_tra_max + T_carga_mess
        _bloques_norm = sorted((int(h0), int(h1)) for h0, h1 in bloques_evento)

        # eventos_mess: lista de (pre_ini, h_ini, h_fin).
        # pre_ini..h_ini = ventana de pre-carga (MESS fijo en su base, cargando).
        # h_ini..h_fin   = ventana del evento (puede desplazarse y descargar).
        # Eventos separados por menos de recovery_mess+1 horas se fusionan (no
        # alcanza a volver a base + recargarse antes del siguiente evento).
        eventos_mess = []
        for h0, h1 in _bloques_norm:
            if eventos_mess and h0 - eventos_mess[-1][2] < recovery_mess + 1:
                pre_ini_prev, h_ini_prev, _ = eventos_mess[-1]
                eventos_mess[-1] = (pre_ini_prev, h_ini_prev, h1)
            else:
                cota_inferior = (eventos_mess[-1][2] + 1) if eventos_mess else 0
                pre_ini = max(cota_inferior, h0 - T_carga_mess, 0)
                eventos_mess.append((pre_ini, h0, h1))

        horas_mess_activas = set()
        for pre_ini, h_ini, h_fin in eventos_mess:
            for t in range(pre_ini, min(h_fin, max(horas)) + 1):
                horas_mess_activas.add(t)
        horas_mess_activas = sorted(horas_mess_activas)
        horas_mess_set = set(horas_mess_activas)
        eventos_pre_ini_set = {pre_ini for pre_ini, h_ini, h_fin in eventos_mess}

        bus_base_mess = {k: 1 for k in KM_mess}
        N_mess_bus = {b: 1 for b in BUSES}
        for b in BUSES:
            cantidad_en_base = sum(1 for k in KM_mess if bus_base_mess[k] == b)
            N_mess_bus[b] = max(1, cantidad_en_base)

        # ── 3. Construcción del modelo ───────────────────────────────────────
        model   = pyo.ConcreteModel()
        model.T = pyo.Set(initialize=horas)
        model.T_mess = pyo.Set(initialize=horas_mess_activas if usar_mess else [])

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
        
        model.z_tie = pyo.Var(model.T, tie_list, within=pyo.Binary)
        # w_tie NO necesita ser binaria: solo aparece en w >= z[t] - z[t-1],
        # w >= z[t-1] - z[t] y con costo POSITIVO en el objetivo. Al
        # minimizar cae siempre a max(0, |Δz|), que ya es 0/1 porque z_tie
        # es binaria. Declararla continua en [0,1] da exactamente el mismo
        # óptimo con |T|·|tie| binarias menos.
        model.w_tie = pyo.Var(model.T, tie_list, within=pyo.NonNegativeReals, bounds=(0, 1))
        model.P_tie = pyo.Var(model.T, tie_list, within=pyo.Reals)

        # INVERSIÓN EN LÍNEAS DE ENLACE — y_tie[l] = 1 si se construye el
        # enlace candidato l. Con reconfig_topo (enlace preexistente) queda
        # fijo en 1; sin ninguna de las dos medidas, en 0.
        model.y_tie = pyo.Var(tie_list, within=pyo.Binary)

        #INVERSIÓN RCS-MS
        model.y_rcs = pyo.Var(sw_cand_list, within=pyo.Binary)
        model.y_ms  = pyo.Var(sw_cand_list, within=pyo.Binary)
        # z_sw solo es una DECISIÓN cuando hay switches que permitan maniobrar
        # (RCS o MS). Sin ellos, las restricciones sw_no_danada y
        # sw_cerrada_si_no_maniobra se cierran una contra otra y dejan
        # z_sw = 1 - daño_efectivo, o sea queda determinada por x_ref/x_ug
        # (binarias) y toma valores 0/1 sola. En ese caso declararla continua
        # en [0,1] es exacto y ahorra |T|·|sw_cand| binarias, que es la
        # familia más grande del modelo cuando se activan refuerzo y/o
        # soterramiento sobre el horizonte completo.
        z_sw_es_decision = usar_inv_switch_rcs or usar_inv_switch_ms
        if z_sw_es_decision:
            model.z_sw = pyo.Var(model.T, sw_cand_list, within=pyo.Binary)
        else:
            model.z_sw = pyo.Var(model.T, sw_cand_list, within=pyo.NonNegativeReals, bounds=(0, 1))
        model.P_sw  = pyo.Var(model.T, sw_cand_list, within=pyo.Reals)
        # --------------------------------------------------
            # Variable de maniobra del switch candidato
            #
            # w_sw[t,l] = 1 si el estado de la línea cambia entre t-1 y t.
            # Sirve solo si el costo operacional se cobra por apertura/cierre,
            # no por estar cerrada.
            #
            # Igual que w_tie: solo aparece acotada por abajo por ±Δz_sw y
            # con costo positivo en el objetivo, así que el óptimo la deja
            # en max(0, |Δz_sw|) sin necesidad de declararla entera.
            # --------------------------------------------------
        model.w_sw = pyo.Var(model.T, sw_cand_list, within=pyo.NonNegativeReals, bounds=(0, 1))

        # GESTIÓN DE VEGETACIÓN / REFUERZO / SOTERRAMIENTO — decisión por
        # línea candidata (igual patrón que y_rcs/y_ms): 1 si esa línea
        # recibe la medida.
        model.x_veg = pyo.Var(sw_cand_list, within=pyo.Binary)
        model.x_ref = pyo.Var(sw_cand_list, within=pyo.Binary)
        model.x_ug  = pyo.Var(sw_cand_list, within=pyo.Binary)

        #MEG
        model.CAP_MEG = pyo.Var(K_meg,  within=pyo.NonNegativeReals)
        model.x_re    = pyo.Var(K_meg,  BUSES, within=pyo.Binary)
        model.P_MEG   = pyo.Var(model.T, K_meg, BUSES, within=pyo.NonNegativeReals)
        model.psi_meg = pyo.Var(K_meg,  BUSES, within=pyo.NonNegativeReals)
        model.a_tb = pyo.Param(model.T, BUSES, initialize=0, mutable=True, within=pyo.Binary)

        # MEG arrendado: x_arr[e,k,b] = 1 si en el evento e se arrienda la
        # unidad k y se lleva al bus b. P_ARR solo en su ventana de servicio.
        model.x_arr = pyo.Var(EV_arr, KA_act, BUSES, within=pyo.Binary)
        model.P_ARR = pyo.Var(idx_arr, within=pyo.NonNegativeReals)

        #MESS — variables solo sobre la ventana reducida model.T_mess.
        # CAP_mess no es variable: E_cap_mess es un dato de entrada (parámetro).
        # El dimensionamiento se decide de forma discreta con y_mess (cuántas
        # unidades de tamaño E_cap_mess se despliegan), no de forma continua.
        model.u_mess    = pyo.Var(KM_mess, BUSES, model.T_mess, within=pyo.Binary)
        model.Pdch_mess = pyo.Var(model.T_mess, KM_mess, BUSES, within=pyo.NonNegativeReals)
        model.Pch_mess  = pyo.Var(model.T_mess, KM_mess, BUSES, within=pyo.NonNegativeReals)
        model.SOC_mess  = pyo.Var(model.T_mess, KM_mess, within=pyo.NonNegativeReals)
        # Despliegue de la unidad k: 1 si se invierte en ella. Es la decisión de
        # dimensionamiento, pero discreta (capacidad instalada = sum_k y_mess[k] *
        # E_cap_mess): si el MESS no aporta lo suficiente, y_mess = 0 y su CAPEX
        # desaparece del objetivo, igual que CAP_MEG/CAP_ESS pueden ir a cero.
        model.y_mess    = pyo.Var(KM_mess, within=pyo.Binary)

        #EDRP
        BLOQUES_EDRP = list(range(C_bloques_EDRP)) if usar_edrp else []
        model.P_EDRP = pyo.Var(model.T, BUSES, BLOQUES_EDRP, within=pyo.NonNegativeReals) 
                      
        # Las tie-lines se operan si hay reconfiguración (enlace existente)
        # o si hay inversión en enlaces nuevos (inv_tie).
        usar_tie_flujo = usar_reconfig_topo or usar_inv_tie

        # Fijar tie-lines si no se opera ninguna
        if not usar_tie_flujo:
            for t in horas:
                for l in tie_list:
                    model.z_tie[t, l].fix(0)
                    model.w_tie[t, l].fix(0)
                    model.P_tie[t, l].fix(0)
            for l in tie_list:
                model.y_tie[l].fix(0)
        else:
            for t in horas:
                if np.sum(Dam_tl_por_hora[t]) == 0:
                    for l in tie_list:
                        model.z_tie[t, l].fix(0)
                        model.P_tie[t, l].fix(0)
            if usar_inv_tie:
                # Solo son decisión de inversión las candidatas declaradas;
                # el resto de tie-lines existe si además hay reconfiguración.
                for l in tie_list:
                    if l not in tie_cand_list:
                        model.y_tie[l].fix(1 if usar_reconfig_topo else 0)
            else:
                # reconfig_topo puro: los enlaces ya están construidos.
                for l in tie_list:
                    model.y_tie[l].fix(1)

        # Una enlace DAÑADA no puede cerrarse. Es conductor aéreo expuesto:
        # estar normalmente abierta no la hace inmune al viento, y entra al
        # sorteo de `construir_faultlin.py` como cualquier otra línea. Sin
        # esto una enlace rota seguía disponible para reconfigurar, lo que
        # sobrevaloraba `reconfig_topo` / `inv_tie`.
        #
        # `y_tie` NO se toca: el enlace sigue existiendo como activo, sólo
        # está fuera de servicio esa hora. Es el mismo criterio que se aplica
        # a `sw_cand_list` más abajo.
        #
        # Tiene que estar también en optimization_base.py, o el caso base
        # castigaría la enlace dañada y el modelo con medidas no, inflando el
        # ahorro atribuido a la reconfiguración.
        for t in horas:
            for l in tie_list:
                if Dam_tl_por_hora[t][l] == 1:
                    model.z_tie[t, l].fix(0)
                    model.w_tie[t, l].fix(0)
                    model.P_tie[t, l].fix(0)

        # Retardo de maniobra de las ties: durante las primeras T_maniobra_tie
        # horas de cada evento no se puede cerrar ninguna (la cuadrilla o el
        # telemando aún no actúa). Mismo criterio que el MS (a_ms más abajo:
        # disponible desde h_ini + T). Con T = 0 (por defecto) no hace nada y el
        # modelo queda idéntico al anterior. Solo afecta a reconfig_topo e inv_tie.
        if usar_tie_flujo and T_maniobra_tie > 0:
            for h_ini, h_fin in bloques_evento:
                for t in range(h_ini, min(h_ini + T_maniobra_tie, h_fin + 1)):
                    for l in tie_list:
                        model.z_tie[t, l].fix(0)
                        model.w_tie[t, l].fix(0)
                        model.P_tie[t, l].fix(0)

        # Radialidad: cerrar un enlace no puede dejar la red mallada. Con la
        # aislación fija las líneas fuera de servicio de cada hora son un dato,
        # así que se resuelve con fijaciones y cotas lineales (ver radialidad.py).
        # Apagada por defecto: sin la clave el modelo queda idéntico al anterior.
        if usar_tie_flujo and radial_tie:
            if usar_candidatas_dano:
                print("[RADIAL] ⚠️ Con veg/refuerzo/soterramiento o inv_switch el daño o la "
                      "aislación cambian dentro del MILP; la radialidad usa el daño base "
                      "(aproximación).")
            _fuentes_sub = [b for b in (1, 12) if b in BUSES]
            model.radial_tie = pyo.ConstraintList()
            _n_fij = _n_grp = 0
            for t in horas:
                if np.sum(Dam_tl_por_hora[t]) == 0:
                    continue
                _fijar, _grupos = restricciones_radiales(
                    Dam_tl_por_hora[t], LINES, from_bus, to_bus, tie_list, _fuentes_sub)
                for l in _fijar:
                    model.z_tie[t, l].fix(0)
                    model.w_tie[t, l].fix(0)
                    model.P_tie[t, l].fix(0)
                _n_fij += len(_fijar)
                for ls, cota in _grupos:
                    model.radial_tie.add(sum(model.z_tie[t, l] for l in ls) <= cota)
                _n_grp += len(_grupos)
            print(f"[RADIAL] Esc {esc}: {_n_fij} enlace·hora fijados en 0 (cerrarían un "
                  f"ciclo) y {_n_grp} restricciones de ciclo/paralelo.")

        if sw_cand_list:
            # Si la medida RCS no está habilitada, no se puede invertir en RCS
            if not usar_inv_switch_rcs:
                for l in sw_cand_list:
                    model.y_rcs[l].fix(0)

            # Si la medida MS no está habilitada, no se puede invertir en MS
            if not usar_inv_switch_ms:
                for l in sw_cand_list:
                    model.y_ms[l].fix(0)

            # Si la medida correspondiente no está habilitada, no se puede
            # elegir esa línea como tratada.
            if Dam_veg_por_hora is None:
                for l in sw_cand_list:
                    model.x_veg[l].fix(0)
            if Dam_ref_por_hora is None:
                for l in sw_cand_list:
                    model.x_ref[l].fix(0)
            if Dam_ug_por_hora is None:
                for l in sw_cand_list:
                    model.x_ug[l].fix(0)

            # A lo más un tratamiento de "hardening" por línea (no tiene
            # sentido podar Y soterrar la misma línea a la vez).
            model.un_tratamiento = pyo.Constraint(
                sw_cand_list,
                rule=lambda m, l: m.x_veg[l] + m.x_ref[l] + m.x_ug[l] <= 1
            )

        # PLAN DE INVERSIÓN IMPUESTO — evaluación fuera de muestra.
        # {clave: {id: valor}} con las mismas claves que el `plan` del barrido
        # (run_barrido_medidas.CLAVES_PLAN). Fija SOLO la inversión: la
        # operación queda libre, porque es el recurso con que el plan elegido
        # en otro escenario enfrenta las fallas de este. Un id ausente vale 0
        # (el `plan` guardado trae solo los no nulos). Lo usa run_plan_fijo.py.
        # Las variables que ya quedaron fijas arriba (medida apagada, enlaces
        # preexistentes de reconfig_topo) no se tocan.
        plan_fijo = config.get("plan_fijo")
        if plan_fijo is not None:
            def _fijar(var, indices, clave, usar, entera=False):
                if not usar:
                    return
                vals = {str(k): float(v) for k, v in (plan_fijo.get(clave) or {}).items()}
                for i in indices:
                    if var[i].fixed:
                        continue
                    v = vals.get(str(i), 0.0)
                    var[i].fix(int(round(v)) if entera else v)

            _fijar(model.CAP_DG_sol, GENS_sol, "CAP_DG_sol_kW", usar_dg_sol)
            _fijar(model.CAP_DG_eol, GENS_eol, "CAP_DG_eol_kW", usar_dg_eol)
            _fijar(model.CAP_DG_fos, GENS_fos, "CAP_DG_fos_kW", usar_dg_fos)
            _fijar(model.CAP_ESS_4h, ESS_activo_4h, "CAP_ESS_kWh_4h", usar_ess_4h)
            _fijar(model.CAP_ESS_5h, ESS_activo_5h, "CAP_ESS_kWh_5h", usar_ess_5h)
            _fijar(model.CAP_ESS_6h, ESS_activo_6h, "CAP_ESS_kWh_6h", usar_ess_6h)
            _fijar(model.y_tie, tie_list, "y_tie", usar_inv_tie, entera=True)
            _fijar(model.y_rcs, sw_cand_list, "y_rcs", usar_inv_switch_rcs, entera=True)
            _fijar(model.y_ms,  sw_cand_list, "y_ms",  usar_inv_switch_ms,  entera=True)
            _fijar(model.x_veg, sw_cand_list, "x_veg", True, entera=True)
            _fijar(model.x_ref, sw_cand_list, "x_ref", True, entera=True)
            _fijar(model.x_ug,  sw_cand_list, "x_ug",  True, entera=True)
            # Flota existente (k < meg_existentes): ya es de la empresa, no es
            # inversión del plan; queda libre (sin CAPEX, como en el oficial).
            # El arriendo tampoco se fija: es operación por evento.
            _fijar(model.CAP_MEG, [k for k in K_meg if k >= n_meg_exist], "CAP_MEG_MW", usar_meg)
            _fijar(model.y_mess, KM_mess, "y_mess", usar_mess, entera=True)

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
            g_meg += sum(m.P_ARR[t, k, b] for k in arr_k_en_tb.get((t, b), ()))
            g_mess = sum(m.Pdch_mess[t, k, b] - m.Pch_mess[t, k, b] for k in KM_mess) if (usar_mess and t in horas_mess_set) else 0
            inf_tie  = sum(m.P_tie[t, l] for l in tie_list if to_bus[l]   == b)
            outf_tie = sum(m.P_tie[t, l] for l in tie_list if from_bus[l] == b)
            inf_sw  = sum(m.P_sw[t, l] for l in sw_cand_activas_por_hora[t] if to_bus[l]   == b)
            outf_sw = sum(m.P_sw[t, l] for l in sw_cand_activas_por_hora[t] if from_bus[l] == b)
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
            # Tie-lines tienen restricciones propias.
            # Las candidatas solo saltan este límite cuando alguna de las 5
            # medidas de daño-por-decisión está activa, porque en ese caso
            # su flujo se modela mediante P_sw.
            if (
                Dam_tl_por_hora[t][l] == 1
                or l in tie_indices
                or (usar_candidatas_dano and l in sw_cand_list)
            ):
                return pyo.Constraint.Skip
            i, j = from_bus[l], to_bus[l]
            return pyo.inequality(-cap[l], (m.theta[t, i] - m.theta[t, j]) / x_ij[l], cap[l])

        model.power_balance     = pyo.Constraint(model.T, BUSES, rule=power_balance_rule)
        model.curtailment_limits = pyo.Constraint(model.T, BUSES, rule=curtail_rule)
        model.sub_limits         = pyo.Constraint(model.T, S,     rule=sub_rule)
        model.line_limits        = pyo.Constraint(model.T, LINES, rule=line_limit_rule)
        model.theta_limits       = pyo.Constraint(model.T, BUSES, rule=theta_rule)

        # Tope de capacidad de DG. Con `lim_dg_modo="bus"` (default) es el
        # tope histórico por barra; con "alimentador"/"zona"/"seccion" es un
        # agregado por grupo que deja el emplazamiento libre dentro del
        # grupo. Tiene que coincidir con el de `optimization_base` (E4/E5).
        grupos_dg, Lim, lim_dg_bus = leer_config_limites_dg(
            config, data, net, BUSES, LINES, from_bus, to_bus,
            tie_indices=tie_indices, bus_zone=bus_zone,
            sec_existentes=sec_existentes, verbose=bool(config.get("verbose")))

        # `lim_dg_conjunto`: solar + eólica comparten UN tope (suma ≤ lim_dg); sin él,
        # cada tecnología tiene el suyo (histórico). Ver `limites_dg.aplicar_tope_conjunto`.
        conjunto_dg = bool(config.get("lim_dg_conjunto", False))

        if usar_dg_sol:
            model.dg_sol_limits = pyo.Constraint(model.T, GENS_sol, rule=lambda m, t, g: m.P_DG_sol[t, g] <= alphaDGsolar[t, esc, g_to_idx[g]] * m.CAP_DG_sol[g])
            # model.dg_sol_limits_2 = pyo.Constraint(model.T, GENS_sol, rule=lambda m, t, g: alphaDGsolar[t, esc, g_to_idx[g]] * m.CAP_DG_sol[g]  <= Lim * P_D_hora_arr[t,g])
            if not conjunto_dg:
                aplicar_tope_dg(model, "dg_sol", model.CAP_DG_sol, GENS_sol,
                                grupos_dg, Lim, P_D_base, lim_dg_bus)

        if usar_dg_eol:
            # Misma forma que la solar (línea 831): la disponibilidad sale del
            # recurso —curva de potencia, ya dentro de `alphaDGeolica`— y de la
            # topología. Ver la nota extensa en optimization_base.py, que tiene
            # que decir lo mismo que este archivo o E4/E5 y las estrategias
            # quedan con supuestos distintos.
            model.dg_eol_limits = pyo.Constraint(
                model.T, GENS_eol,
                rule=lambda m, t, g: m.P_DG_eol[t, g]
                <= alphaDGeolica[t, esc, g_to_idx[g]] * m.CAP_DG_eol[g])

            # ── `a_eol_wind`, DESACTIVADO el 2026-09-08 ──────────────────
            # Anulaba la generación de toda zona con alguna línea dañada. Se
            # deja comentado, no borrado, por si se quiere reactivar.
            #
            #   - NO era un apagado por viento: se disparaba por DAÑO DE LÍNEA,
            #     y `Dam_tl_por_hora` es acumulativo hasta la reparación.
            #   - El apagado por viento fuerte ya existe: `V_CUT_OUT = 25 m/s`
            #     (IEC 61400) en `construir_perfiles.py`. En Arauco no se
            #     activa porque el sostenido máximo a 30 m es 22,3 m/s.
            #   - Rompía la simetría con la solar, que no tiene equivalente.
            #   - Efecto: apagaba ~9 h/año, el 100 % de las horas con ENS.
            #     `dg_eol` generaba 31,5 GWh y aportaba CERO a la resiliencia.
            #
            # LIMITACIÓN: sin esto la turbina genera con ráfagas de 39 m/s. Es
            # cota superior de disponibilidad eólica, simétrica con la solar.
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


        if usar_dg_fos:
            model.dg_fos_limits = pyo.Constraint(model.T, GENS_fos, rule=lambda m, t, g: m.P_DG_fos[t, g] <= m.CAP_DG_fos[g])
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

        if usar_tie_flujo:
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

        # ── INVERSIÓN EN LÍNEAS DE ENLACE ────────────────────────────────
        if usar_inv_tie:
            # Horas en las que z_tie NO quedó fijado en 0 más arriba (las
            # horas sin daño en toda la red fijan z_tie = P_tie = 0). Solo
            # en esas horas las restricciones de abajo pueden ser activas;
            # en el resto serían 0 <= algo, redundantes. Con un horizonte
            # anual eso es la enorme mayoría de las 8760 filas por enlace,
            # y el costo de armarlas en Pyomo domina el tiempo de corrida.
            # Se indexan sobre esta lista y NO sobre model.T: con
            # Constraint.Skip Pyomo igual llama a la regla una vez por hora
            # y descarta el resultado, que es justo el costo que domina el
            # armado del modelo. Indexando el conjunto reducido, la regla
            # nunca se llama para las horas sin evento.
            horas_tie_libres = [t for t in horas if np.sum(Dam_tl_por_hora[t]) > 0]

            # Un enlace solo puede cerrarse si se construyó.
            model.tie_solo_si_construida = pyo.Constraint(
                horas_tie_libres, tie_list,
                rule=lambda m, t, l: m.z_tie[t, l] <= m.y_tie[l]
            )

            # El enlace nuevo también puede fallar durante el evento: si la
            # línea está dañada en t, no puede estar cerrada. Donde la línea
            # está sana la restricción es z_tie <= 1, redundante para una
            # binaria, así que se omite.
            if tie_expuesta_falla:
                def tie_no_danada_rule(m, t, l):
                    if Dam_tl_por_hora[t][l] == 0:
                        return pyo.Constraint.Skip
                    return m.z_tie[t, l] <= 1 - Dam_tl_por_hora[t][l]

                model.tie_no_danada = pyo.Constraint(
                    horas_tie_libres, tie_list, rule=tie_no_danada_rule
                )

        if sw_cand_list and usar_candidatas_dano:
            a_ms = {t: 0 for t in horas}
            for h_ini, h_fin in bloques_evento:
                t_disp = h_ini + T_maniobra_MS
                for t in horas:
                    if t_disp <= t <= h_fin:
                        a_ms[t] = 1


            # --------------------------------------------------
            # A lo más una tecnología por línea candidata
            # --------------------------------------------------
            model.un_tipo_sw = pyo.Constraint(sw_cand_list, rule=lambda m, l: m.y_rcs[l] + m.y_ms[l] <= 1)

            # --------------------------------------------------
            # Si la línea está dañada (daño efectivo, tras aplicar la
            # medida de hardening elegida para esa línea si corresponde),
            # no puede estar cerrada: d_efectivo[t,l] = 1  =>  z_sw[t,l] = 0
            # --------------------------------------------------
            model.sw_no_danada = pyo.Constraint(model.T, sw_cand_list, rule=lambda m, t, l: m.z_sw[t, l] <= 1 - _dano_efectivo(m, t, l))

            # --------------------------------------------------
            # Si la línea está sana y NO hay capacidad de maniobra,
            # debe permanecer cerrada (Shahbazian et al. 2020: un switch
            # RCS/MS habilita la capacidad de AISLAR activamente ese punto
            # de la red durante una falla; su ausencia no abre una línea
            # sana, solo impide usarla como punto de aislamiento).
            #
            # Es decir:
            # - si no hay RCS instalado,
            # - y no hay MS disponible (o está fuera de su ventana de
            #   maniobra a_ms),
            # - y la línea no está dañada,
            # entonces z_sw[t,l] = 1.
            #
            # Nota: la versión anterior de esta restricción (z_sw <= y_rcs
            # + a_ms*y_ms) forzaba el efecto contrario -- abría
            # permanentemente cualquier línea candidata sin inversión,
            # incluso sana y fuera de falla. Se reemplaza por la cota
            # inferior correcta:
            # --------------------------------------------------
            model.sw_cerrada_si_no_maniobra = pyo.Constraint(
                model.T, sw_cand_list,
                rule=lambda m, t, l: m.z_sw[t, l] >= (1 - _dano_efectivo(m, t, l)) - m.y_rcs[l] - a_ms[t] * m.y_ms[l]
            )

            # --------------------------------------------------
            # Relación flujo DC - estado de cierre de línea
            #
            # Si z_sw[t,l] = 1:
            # P_sw[t,l] = (theta_i - theta_j)/x_l
            #
            # Si z_sw[t,l] = 0:
            # P_sw[t,l] queda desacoplado por Big-M,
            # pero luego queda forzado a cero por límites de flujo.
            # --------------------------------------------------
            model.sw_link_upper = pyo.Constraint(
                model.T,
                sw_cand_list,
                rule=lambda m, t, l:
                    (m.theta[t, from_bus[l]] - m.theta[t, to_bus[l]]) / x_ij[l]
                    - m.P_sw[t, l]
                    <= M_sw[l] * (1 - m.z_sw[t, l])
            )

            model.sw_link_lower = pyo.Constraint(
                model.T,
                sw_cand_list,
                rule=lambda m, t, l:
                    (m.theta[t, from_bus[l]] - m.theta[t, to_bus[l]]) / x_ij[l]
                    - m.P_sw[t, l]
                    >= -M_sw[l] * (1 - m.z_sw[t, l])
            )

            # --------------------------------------------------
            # Límites de flujo por línea candidata
            #
            # Si z_sw[t,l] = 0:
            # P_sw[t,l] = 0
            #
            # Si z_sw[t,l] = 1:
            # -cap[l] <= P_sw[t,l] <= cap[l]
            # --------------------------------------------------
            model.sw_flow_upper = pyo.Constraint(
                model.T,
                sw_cand_list,
                rule=lambda m, t, l:
                    m.P_sw[t, l] <= cap[l] * m.z_sw[t, l]
            )

            model.sw_flow_lower = pyo.Constraint(
                model.T,
                sw_cand_list,
                rule=lambda m, t, l:
                    m.P_sw[t, l] >= -cap[l] * m.z_sw[t, l]
            )

            horas_list = list(horas)

            def sw_change_pos_rule(m, t, l):
                idx = horas_list.index(t)

                if idx == 0:
                    # Las líneas candidatas son normalmente cerradas antes del horizonte.
                    # Estado de referencia: z_ref = 1.
                    return m.w_sw[t, l] >= m.z_sw[t, l] - 1

                t_prev = horas_list[idx - 1]
                return m.w_sw[t, l] >= m.z_sw[t, l] - m.z_sw[t_prev, l]

            def sw_change_neg_rule(m, t, l):
                idx = horas_list.index(t)

                if idx == 0:
                    # Si en t=0 la línea pasa de cerrada a abierta,
                    # se cuenta una maniobra.
                    return m.w_sw[t, l] >= 1 - m.z_sw[t, l]

                t_prev = horas_list[idx - 1]
                return m.w_sw[t, l] >= m.z_sw[t_prev, l] - m.z_sw[t, l]

            model.sw_change_pos = pyo.Constraint(
                model.T,
                sw_cand_list,
                rule=sw_change_pos_rule
            )

            model.sw_change_neg = pyo.Constraint(
                model.T,
                sw_cand_list,
                rule=sw_change_neg_rule
            )


#             def maniobra_ms_rule(m, t, l):
#                 if a_ms[t] == 0:
#                     return m.z_sw[t, l] <= m.y_rcs[l]
#                 return pyo.Constraint.Skip
            
#             SWITCH_instalado = {(t, b): [l for l in sw_cand_activas_por_hora[t] if to_bus[l] == b] for t in horas for b in BUSES}
#             SWITCH_activado = {(t, b): [l for l in sw_cand_activas_por_hora[t] if from_bus[l] == b] for t in horas for b in BUSES}
#             sum(m.P_sw[t, l] for l in sw_cand_activas_por_hora[t] if to_bus[l]   == b)
#             def MS_RCS(m, t, b):
#                 for l in sw_cand_activas_por_hora:
#                     if to_bus[l] == b:
#                         return m.z_sw[t, l] <= m.y_rcs[l] + m.y_ms[l] 
#                     elif from_bus[l] == b:
#                         return m.z_sw[t, l] <= m.y_rcs[l] + m.y_ms[l] 


#             model.un_tipo_sw = pyo.Constraint(sw_cand_list, rule=lambda m, l: m.y_rcs[l] + m.y_ms[l] <= 1 )
#             model.sw_existe = pyo.Constraint(model.T, sw_cand_list,rule=lambda m, t, l: m.z_sw[t, l] <= m.y_rcs[l] + m.y_ms[l])
#             model.maniobra_ms = pyo.Constraint(model.T, sw_cand_list, rule=maniobra_ms_rule)
#             model.sw_link_upper = pyo.Constraint(model.T, sw_cand_list,rule=lambda m, t, l: (m.theta[t, from_bus[l]] - m.theta[t, to_bus[l]]) / x_ij[l] - m.P_sw[t, l] <= M_sw[l] * (1 - m.z_sw[t, l]))
#             model.sw_link_lower = pyo.Constraint(model.T, sw_cand_list,rule=lambda m, t, l: (m.theta[t, from_bus[l]] - m.theta[t, to_bus[l]]) / x_ij[l] - m.P_sw[t, l] >= -M_sw[l] * (1 - m.z_sw[t, l]))
#             model.sw_flow_upper = pyo.Constraint(model.T, sw_cand_list,rule=lambda m, t, l: m.P_sw[t, l] <=  cap[l] * m.z_sw[t, l])
#             model.sw_flow_lower = pyo.Constraint(model.T, sw_cand_list,rule=lambda m, t, l: m.P_sw[t, l] >= -cap[l] * m.z_sw[t, l])
#             # Si la línea candidata está dañada, queda abierta
#             # model.sw_no_danada = pyo.Constraint(
#             #     model.T, sw_cand_list,
#             #     rule=lambda m, t, l:
#             #         m.z_sw[t, l] <= 1 - Dam_tl_por_hora[t][l]
#             # )

#             # # Si está sana y no hay capacidad de maniobra disponible, debe quedar cerrada
#             # model.sw_cerrada_si_no_maniobra = pyo.Constraint(
#             #     model.T, sw_cand_list,
#             #     rule=lambda m, t, l:
#             #         m.z_sw[t, l] >= 1 - Dam_tl_por_hora[t][l] - m.y_rcs[l] - a_ms[t] * m.y_ms[l]
# # )


        # ══════════════════════════════════════════════════════
        # FORMACIÓN DE MICRORREDES (Kim y Dvorkin, 2019 — Ec. 8a-8b)
        # ══════════════════════════════════════════════════════
        # No es una medida con costo propio (no tiene CAPEX/OPEX): es una
        # restricción operativa que solo tiene efecto si además hay líneas
        # controlables (RCS y/o MS) activas. Acota el número de líneas
        # candidatas abiertas (más allá de las dañadas) por la cantidad de
        # buses con inyección local positiva en esa hora -- así el modelo
        # puede sectorizar deliberadamente la red en torno a fuentes
        # propias (DG, ESS, MEG, MESS), pero no más de lo que esas fuentes
        # pueden sostener. Fuera de eventos de falla esta restricción no
        # es vinculante (abrir líneas sin falla no reporta beneficio en
        # la función objetivo, así que el solver no lo hace aunque la
        # restricción lo permita).
        usar_microredes = config.get("microredes", False)

        if usar_microredes and sw_cand_list and (usar_inv_switch_rcs or usar_inv_switch_ms):
            M_micro = net.trafo.sn_mva[0] * 1000  # cota válida: ninguna inyección local supera la capacidad total del alimentador

            def _inyeccion_local(m, t, b):
                g_sol = sum(m.P_DG_sol[t, g] for g in gens_en_bus_sol[b])
                g_eol = sum(m.P_DG_eol[t, g] for g in gens_en_bus_eol[b])
                g_fos = sum(m.P_DG_fos[t, g] for g in gens_en_bus_fos[b])
                ess_p_5h = sum(m.P_discharge_5h[t, e] - m.P_charge_5h[t, e] for e in ess_en_bus_act_5h[b])
                ess_p_4h = sum(m.P_discharge_4h[t, e] - m.P_charge_4h[t, e] for e in ess_en_bus_act_4h[b])
                ess_p_6h = sum(m.P_discharge_6h[t, e] - m.P_charge_6h[t, e] for e in ess_en_bus_act_6h[b])
                g_meg = sum(m.P_MEG[t, k, b] for k in K_meg)
                g_meg += sum(m.P_ARR[t, k, b] for k in arr_k_en_tb.get((t, b), ()))
                g_mess = sum(m.Pdch_mess[t, k, b] - m.Pch_mess[t, k, b] for k in KM_mess) if (usar_mess and t in horas_mess_set) else 0
                return g_sol + g_eol + g_fos + ess_p_5h + ess_p_4h + ess_p_6h + g_meg + g_mess

            # z_fuente[t,b] = 1 si el bus b inyecta localmente algo > 0 en la hora t
            model.z_fuente = pyo.Var(model.T, BUSES, within=pyo.Binary)
            model.fuente_link = pyo.Constraint(
                model.T, BUSES,
                rule=lambda m, t, b: _inyeccion_local(m, t, b) <= M_micro * m.z_fuente[t, b]
            )

            # N.º de líneas candidatas dañadas por hora (constante, no variable)
            n_danadas_t = {t: sum(Dam_tl_por_hora[t][l] for l in sw_cand_list) for t in horas}

            # max(n_danadas_t, n_fuentes[t]) linealizado con selector binario
            BigM_micro_card = len(sw_cand_list) + len(BUSES)
            model.M_micro_max   = pyo.Var(model.T, within=pyo.NonNegativeReals)
            model.sel_micro_max = pyo.Var(model.T, within=pyo.Binary)

            model.micro_max_lb1 = pyo.Constraint(model.T, rule=lambda m, t: m.M_micro_max[t] >= n_danadas_t[t])
            model.micro_max_lb2 = pyo.Constraint(model.T, rule=lambda m, t: m.M_micro_max[t] >= sum(m.z_fuente[t, b] for b in BUSES))
            model.micro_max_ub1 = pyo.Constraint(model.T, rule=lambda m, t: m.M_micro_max[t] <= n_danadas_t[t] + BigM_micro_card * (1 - m.sel_micro_max[t]))
            model.micro_max_ub2 = pyo.Constraint(model.T, rule=lambda m, t: m.M_micro_max[t] <= sum(m.z_fuente[t, b] for b in BUSES) + BigM_micro_card * m.sel_micro_max[t])

            # Ec. (8a): n.º de líneas candidatas abiertas <= max(dañadas, fuentes disponibles)
            model.microredes_cardinalidad = pyo.Constraint(
                model.T,
                rule=lambda m, t: sum(1 - m.z_sw[t, l] for l in sw_cand_list) <= m.M_micro_max[t]
            )

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
            # Flota existente primero: la unidad k (comprada o no) solo se
            # ubica si la k-1 ya se ubicó. Solo con meg_existentes > 0.
            if n_meg_exist > 0 and len(K_meg) > 1:
                model.meg_orden = pyo.Constraint(
                    K_meg[1:], rule=lambda m, k: sum(m.x_re[k, b] for b in BUSES) <= sum(m.x_re[k - 1, b] for b in BUSES))

        if usar_meg_arr and EV_arr and KA_act:
            # Una unidad arrendada va a un solo bus por evento, y un bus recibe
            # a lo más una unidad por evento (como el MEG propio).
            model.arr_un_bus = pyo.Constraint(
                EV_arr, KA_act, rule=lambda m, e, k: sum(m.x_arr[e, k, b] for b in BUSES) <= 1)
            model.arr_bus_una = pyo.Constraint(
                EV_arr, BUSES, rule=lambda m, e, b: sum(m.x_arr[e, k, b] for k in KA_act) <= 1)
            # Hasta P_meg_arr kW mientras dura la ventana del evento que la trajo.
            model.arr_p_ub = pyo.Constraint(
                idx_arr, rule=lambda m, t, k, b: m.P_ARR[t, k, b] <= P_meg_arr * sum(m.x_arr[e, k, b] for e in arr_eventos_tb[(t, b)]))

        # ══════════════════════════════════════════════════════
        # MESS — Restricciones (solo sobre model.T_mess)
        # ══════════════════════════════════════════════════════

        if usar_mess:
            # Unidades idénticas: desplegar {0,1} equivale a desplegar {0,2}.
            # Forzar y_mess[k] >= y_mess[k+1] elimina esas soluciones simétricas
            # y acelera el branch-and-bound.
            if len(KM_mess) > 1:
                model.mess_simetria = pyo.Constraint(
                    KM_mess[:-1], rule=lambda m, k: m.y_mess[k] >= m.y_mess[k + 1]
                )
            # Un MESS en máximo un bus por período
            model.mess_un_bus = pyo.Constraint(KM_mess, model.T_mess, rule=lambda m, k, t: sum(m.u_mess[k, b, t] for b in BUSES) <= 1)
            # Máximo un MESS por bus por período (9e)
            def bus_un_mess(m, b, t):
                return sum(m.u_mess[k, b, t] for k in KM_mess) <= N_mess_bus[b]
            model.bus_un_mess = pyo.Constraint(BUSES, model.T_mess, rule=bus_un_mess)

            # SOC (9b): se resetea a 0 al inicio de cada ventana de pre-carga
            # (pre_ini) y desde ahí el modelo decide cuánto/cuándo cargar,
            # respetando la tasa real de carga, hasta llegar al evento.
            def soc_mess_rule(m, t, k):
                if t in eventos_pre_ini_set:
                    return m.SOC_mess[t, k] == 0.0
                return (m.SOC_mess[t, k] == m.SOC_mess[t - 1, k]
                        + sum(eta_ch_mess * m.Pch_mess[t, k, b] - (1 / eta_dch_mess) * m.Pdch_mess[t, k, b] for b in BUSES))
            model.mess_soc_ev  = pyo.Constraint(model.T_mess, KM_mess, rule=soc_mess_rule)
            # SOC (4b) Límites energía almacenada — E_cap_mess es parámetro fijo, no variable.
            model.mess_soc_max = pyo.Constraint(model.T_mess, KM_mess, rule=lambda m, t, k: m.SOC_mess[t, k] <= E_cap_mess * m.y_mess[k])

            # Límites de potencia totales (9c)-(9d)
            model.mess_ch_ub  = pyo.Constraint(model.T_mess, KM_mess, BUSES, rule=lambda m, t, k, b: m.Pch_mess[t, k, b] * eta_ch_mess <= (E_cap_mess / h_mess) * m.u_mess[k, b, t])
            model.mess_dch_ub = pyo.Constraint(model.T_mess, KM_mess, BUSES, rule=lambda m, t, k, b: m.Pdch_mess[t, k, b] / eta_dch_mess <= (E_cap_mess / h_mess) * m.u_mess[k, b, t])

            # ── Trayectoria por evento (pre-carga en base, ida, permanencia, exclusividad de destino) ──
            model.mess_mobility = pyo.ConstraintList()

            for pre_ini, h_ini, h_fin in eventos_mess:
                for k in KM_mess:
                    b_base = bus_base_mess[k]

                    # Pre-carga + inicio del evento: si la unidad se despliega,
                    # permanece en su base; si no (y_mess = 0), no está en ningún bus.
                    for t in range(pre_ini, h_ini + 1):
                        if t not in horas_mess_set:
                            continue
                        model.mess_mobility.add(
                            model.u_mess[k, b_base, t] == model.y_mess[k]
                        )
                        for b in BUSES:
                            if b != b_base:
                                model.u_mess[k, b, t].fix(0)

                    destinos_finales = sum(
                        model.u_mess[k, b, h_fin] for b in BUSES if b != b_base
                    )
                    model.mess_mobility.add(destinos_finales <= model.y_mess[k])

                    for b in BUSES:
                        if b == b_base:
                            continue
                        delay_ida = T_tra_mess.get(b, 2)
                        llegada = h_ini + delay_ida
                        destino_elegido = model.u_mess[k, b, h_fin]

                        if llegada > h_fin:
                            model.mess_mobility.add(destino_elegido == 0)
                            continue

                        for t in range(h_ini + 1, h_fin + 1):
                            if t not in horas_mess_set:
                                continue
                            if t < llegada:
                                model.mess_mobility.add(model.u_mess[k, b, t] == 0)
                            else:
                                model.mess_mobility.add(model.u_mess[k, b, t] == destino_elegido)

                    for t in range(h_ini + 1, h_fin + 1):
                        if t not in horas_mess_set:
                            continue
                        model.mess_mobility.add(
                            model.u_mess[k, b_base, t] == model.y_mess[k] - destinos_finales
                        )


        # # Al inicio de resolver_esc, antes de construir el modelo:
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
            # Referencia sin fallas de la DG. Con AMBAS tecnologías activas se
            # usa la constante conjunta (E4/4) y no la suma de las dos
            # individuales: solar y eólica compiten por el mismo tope de
            # `lim_dg` por alimentador, así que el óptimo sin fallas conjunto
            # instala menos de cada una que sus óptimos por separado, y sumar
            # las dos constantes infla el crédito. Ver ECUACIONES_OBJETIVO_DG.md
            # §8.2. Si el artefacto de E4 es anterior a esta corrección no trae
            # las claves `*_ambas_*`, y se cae al comportamiento histórico.
            _ambas = usar_dg_sol and usar_dg_eol and CAPEX_DG_ambas_sin_fallas is not None
            if _ambas:
                total += -CAPEX_DG_ambas_sin_fallas
                total += -OPEX_DG_ambas_sin_fallas

            if usar_dg_sol:
                if not _ambas:
                    total += -CAPEX_DG_sol_sin_fallas
                    total += -OPEX_DG_sol_sin_fallas
                total += sum(C_inv_DG_sol[g] * m.CAP_DG_sol[g] * frp for g in GENS_sol)
                total += sum(m.CAP_DG_sol[g] * C_op_DG_sol[g] for g in GENS_sol)
            if usar_dg_eol:
                if not _ambas:
                    total += -CAPEX_DG_eol_sin_fallas
                    total += -OPEX_DG_eol_sin_fallas
                total += sum(C_inv_DG_eol[g] * m.CAP_DG_eol[g] * frp for g in GENS_eol)
                total += sum(m.CAP_DG_eol[g] * C_op_DG_eol[g] for g in GENS_eol)

            if usar_dg_fos:
                total += sum(C_inv_DG_fos[g] * m.CAP_DG_fos[g] * frp for g in GENS_fos)
                total += sum(m.CAP_DG_fos[g] * C_inv_DG_fos[g]*0.01 for g in GENS_fos)
                total += sum(m.P_DG_fos[t, g] * (C_comb_DG_fos[g]*0.000242 + C_op_DG_fos[g]) for t in horas for g in GENS_fos)
            if usar_ess_5h: #cambió a 5 horas
                total += sum((C_inv_ESS_5h[e] * frp + C_op_ESS_5h[e]) * (m.CAP_ESS_5h[e]/h_ess_5h) for e in ESS_activo_5h)
            if usar_ess_4h:
                total += sum((C_inv_ESS_4h[e] * frp + C_op_ESS_4h[e]) * (m.CAP_ESS_4h[e]/h_ess_4h) for e in ESS_activo_4h)
            if usar_ess_6h:
                total += sum((C_inv_ESS_6h[e] * frp + C_op_ESS_6h[e]) * (m.CAP_ESS_6h[e]/h_ess_6h) for e in ESS_activo_6h)
            if usar_meg:
                total += sum((C_inv_MEG) * frp_meg * m.CAP_MEG[k] for k in K_meg if k >= n_meg_exist)
                total += sum((C_op_MEG* T_tra_meg.get(b, 2) * m.x_re[k,b]) for k in K_meg for b in BUSES)
                total += sum(C_comb_MEG*0.000242 * m.P_MEG[t, k, b] for t in horas for k in K_meg for b in BUSES)
            if usar_meg_arr and EV_arr and KA_act:
                total += C_evento_meg_arr * sum(m.x_arr[e, k, b] for e in EV_arr for k in KA_act for b in BUSES)
            if usar_mess:
                total += sum((C_inv_mess * E_cap_mess + C_op_mess * E_cap_mess/h_mess) * frp_movil * m.y_mess[k] for k in KM_mess)
                for pre_ini, h_ini, h_fin in eventos_mess:
                    for k in KM_mess:
                        b_base = bus_base_mess[k]
                        for b in BUSES:
                            if b == b_base:
                                continue
                            total += (C_tra_mess * (2*T_tra_mess.get(b,2))* m.u_mess[k, b, h_fin])
            if usar_tie_flujo:
                # Costo de maniobra de los enlaces (común a reconfig e inv_tie)
                total += sum(C_op_SWITCH * m.w_tie[t, l] for t in horas for l in tie_list)
            if usar_inv_tie:
                # CAPEX + OPEX fijo del enlace nuevo, por km de línea
                # construida (anualizado con frp). Solo la línea: la maniobra
                # del switch se cobra aparte, en el término de arriba.
                total += sum(
                    (C_inv_TIE * frp + C_op_TIE) * longitud_km_linea.get(l, 0.0) * m.y_tie[l]
                    for l in tie_cand_list
                )
            if usar_inv_switch_rcs and sw_cand_list:
                total += sum((C_inv_RCS) *frp * m.y_rcs[l] for l in sw_cand_list)
            if usar_inv_switch_ms and sw_cand_list:
                total += sum((C_inv_MS)* frp * m.y_ms[l] for l in sw_cand_list)
            if sw_cand_list and (usar_inv_switch_rcs or usar_inv_switch_ms):
                total += sum(C_op_SWITCH * m.w_sw[t, l] for t in horas for l in sw_cand_list)
            if usar_veg_mgmt and sw_cand_list:
                total += sum((C_inv_veg * frp + C_op_veg) * longitud_km_linea.get(l, 0) * m.x_veg[l] for l in sw_cand_list)
            if usar_refuerzo and sw_cand_list:
                total += sum((C_inv_ref * frp + C_op_ref) * longitud_km_linea.get(l, 0) * m.x_ref[l] for l in sw_cand_list)
            if usar_soterramiento and sw_cand_list:
                total += sum((C_inv_ug * frp + C_op_ug) * longitud_km_linea.get(l, 0) * m.x_ug[l] for l in sw_cand_list)
            if usar_edrp:
                total += sum(C_incentivo_EDRP[c] * m.P_EDRP[t, b, c] for t in horas for b in BUSES for c in BLOQUES_EDRP) * N_eventos

            total += sum(VoLL_local[b] * m.P_LC[t, b] for t in horas for b in BUSES) * N_eventos

            # Crédito por ahorro de compra a subestación respecto del caso base
            # con fallas. Se incorpora una única vez independientemente de qué
            # combinación de DG solar/eólica esté activa (antes se sumaba una
            # vez por cada tecnología renovable activa, duplicándose cuando
            # ambas se evaluaban en conjunto).
            total += -((P_sub_total - sum(m.P_SUB[t, s] for t in horas for s in S))*C_sub[0])
            return total
        
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # ── 6. Solver ────────────────────────────────────────────────────────
        # ── Control de memoria ───────────────────────────────────────────────
        # Con el horizonte completo (8760 h) el modelo tiene ~10^5–10^6 columnas
        # y el branch & bound agotaba la RAM ("GurobiError: Out of memory").
        # Los parámetros de abajo acotan el consumo sin cambiar el óptimo:
        #   Method=2      → barrier SOLO en la raíz. Por defecto Gurobi lanza el
        #                   optimizador concurrente, que mantiene una copia del
        #                   modelo por cada algoritmo que corre en paralelo.
        #   Threads       → cada hilo guarda su propia copia de la relajación
        #                   del nodo; con este tamaño, 8 hilos multiplican la
        #                   memoria mucho más de lo que aceleran.
        #   NodefileStart → vuelca el árbol B&B a disco al superar ese umbral
        #                   (GB) en vez de acumularlo en memoria.
        #   SoftMemLimit  → Gurobi termina ordenadamente devolviendo la mejor
        #                   solución encontrada, en vez de lanzar "Out of
        #                   memory" y matar la corrida completa.
        #   TimeLimit     → ídem por tiempo. El incumbente se acepta más abajo,
        #                   así que un corte por tiempo/memoria NO descarta el
        #                   resultado: devuelve la mejor solución con su gap.
        # En un servidor, /tmp suele ser chico (o tmpfs, o sea RAM): el árbol
        # B&B volcado ahí compite con el propio modelo. ERAD_GUROBI_NODEFILE_DIR
        # lo manda a un disco con espacio sin tocar el config.
        _nodefile_dir = (config.get("gurobi_nodefile_dir")
                         or os.environ.get("ERAD_GUROBI_NODEFILE_DIR"))
        if not _nodefile_dir:
            _nodefile_dir = os.path.join(tempfile.gettempdir(), "gurobi_nodefiles")
        os.makedirs(_nodefile_dir, exist_ok=True)

        # Los tres recursos se calculan según la MÁQUINA donde corre, no
        # fijos: así el mismo código aprovecha un servidor grande y sigue
        # siendo seguro en un notebook, sin editar nada. `config` los pisa.
        _mem_gb = config.get("gurobi_mem_gb")
        if _mem_gb is None:
            # 60% de la RAM: el otro 40% lo necesita el propio modelo de
            # Pyomo, que vive en la memoria del proceso Python.
            _mem_gb = max(2.0, 0.6 * _ram_disponible_gb())

        _threads = config.get("gurobi_threads")
        if _threads is None:
            # La mitad de los núcleos, con tope 16: el speedup del branch &
            # bound es muy sublineal y cada hilo guarda su propia copia de la
            # relajación del nodo. Con 8 núcleos da 4 (el valor histórico);
            # con 28, da 14. Son los núcleos ASIGNADOS, no los del nodo: en
            # un trabajo de SLURM con 8 núcleos sobre un nodo de 128,
            # `os.cpu_count()` decía 128 y daba 16 hilos.
            _threads = max(1, min(16, recursos.cpus_disponibles() // 2))

        _nodefile_start = config.get("gurobi_nodefile_start_gb")
        if _nodefile_start is None:
            # A partir de acá el árbol B&B se vuelca a disco, que es órdenes
            # de magnitud más lento que la RAM. El valor fijo de 0.5 GB hacía
            # que una máquina con 64 GB empezara a escribir al disco casi de
            # inmediato; se usa un tercio del presupuesto de memoria (con la
            # mitad, la corrida todas10 del 2026-10-23 murió por OOM en el
            # servidor de 251 GB con _mem_gb=200).
            _nodefile_start = max(0.5, _mem_gb / 3)

        _opts = {
        "Threads": int(_threads),
        "MIPGap": float(config.get("mip_gap", 0.02)),
        "MIPFocus": int(config.get("gurobi_mipfocus", 1)),
        "Presolve": int(config.get("gurobi_presolve", 2)),
        "Cuts": 1,
        "Heuristics": float(config.get("gurobi_heuristics", 0.05)),
        "Symmetry": 2,
        # Barrier (2) es rápido en la raíz pero su factorización es lo que más
        # memoria pide; en redes grandes el simplex dual (1) sí cabe.
        "Method": int(config.get("gurobi_method", 2)),
        "NodefileStart": float(_nodefile_start),
        "NodefileDir": _nodefile_dir,
        "TimeLimit": float(config.get("gurobi_time_limit", 1800)),
        "SoftMemLimit": float(_mem_gb),
    }
        print(f"[RECURSOS] {recursos.resumen()} -> Threads={_opts['Threads']}, "
              f"SoftMemLimit={_opts['SoftMemLimit']:.1f} GB, "
              f"NodefileStart={_opts['NodefileStart']:.1f} GB, "
              f"NodefileDir={_nodefile_dir}")

        # El solver se crea dentro de una fábrica para poder rehacerlo en el
        # reintento por token WLS vencido (ver _resolver_con_reintento).
        def _crear_solver():
            s = pyo.SolverFactory("gurobi_direct")
            s.options.update(_opts)
            return s

        print("Costo sub",C_sub)
        print("Sub_total",P_sub_total)

        print(f"[Esc {esc+1}] Resolviendo {len(horas)} h × {len(BUSES)} buses "
              f"(Threads={_opts['Threads']}, "
              f"SoftMemLimit={_opts['SoftMemLimit']:.1f} GB, "
              f"TimeLimit={_opts['TimeLimit']:.0f} s)...")

        try:
            solver, result = _resolver_con_reintento(
                _crear_solver, model, tee=True,
                intentos=int(config.get("gurobi_reintentos_licencia", 3)),
                espera_s=float(config.get("gurobi_espera_reintento_s", 30)),
            )
        except Exception as e:
            # Un fallo del solver en un escenario (memoria, licencia, etc.) no
            # debe abortar el barrido completo: se marca este escenario como
            # fallido y se sigue con el siguiente.
            print(f"Escenario {esc+1} - Solver falló: {type(e).__name__}: {e}")
            return

        # Un corte por TimeLimit/SoftMemLimit/interrupción deja un incumbente
        # utilizable: se acepta si Gurobi reporta al menos una solución.
        _term = str(result.solver.termination_condition)
        # El gap real con que quedó la solución, no el pedido: es lo único que
        # dice si un incumbente está cerca del óptimo o es basura. Se guarda
        # para que viaje con el resultado (ver 'solver_gap' más abajo).
        try:
            _gap_final = float(solver._solver_model.MIPGap)
        except Exception:                       # noqa: BLE001 — otro solver
            _gap_final = None
        _hay_incumbente = False
        if _term not in ("optimal", "feasible"):
            try:
                _hay_incumbente = solver._solver_model.SolCount > 0
            except Exception:
                _hay_incumbente = False
            if _hay_incumbente:
                if _gap_final is not None:
                    print(f"Escenario {esc+1} - Terminó por '{_term}'; se usa la "
                          f"mejor solución encontrada (gap {_gap_final:.2%}).")
                else:
                    print(f"Escenario {esc+1} - Terminó por '{_term}'; se usa la "
                          f"mejor solución encontrada.")

        if _term in ("optimal", "feasible") or _hay_incumbente:
            print(f"Escenario {esc+1} - Obj Total: {pyo.value(model.obj):.4f}")

            # ── 7. Extracción de resultados ──────────────────────────────────────
            def _val(v):
                try:
                    r = pyo.value(v)
                    return 0.0 if r is None else r
                except Exception:
                    return 0.0

            fn_objetivo   = pyo.value(model.obj)
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
            z_tie_vals    = np.array([[_val(model.z_tie[t, l])     for l in tie_list] for t in horas])
            w_tie_vals    = np.array([[_val(model.w_tie[t, l])     for l in tie_list] for t in horas])
            P_tie_vals    = np.array([[_val(model.P_tie[t, l])     for l in tie_list] for t in horas])
            y_tie_vals    = np.array([round(_val(model.y_tie[l]))  for l in tie_list])
            CAP_meg_vals  = np.array([_val(model.CAP_MEG[k])       for k in K])    if usar_meg else np.zeros(len(K))
            x_re_vals     = np.array([[_val(model.x_re[k, b])      for b in BUSES] for k in K]) if usar_meg else np.zeros((len(K), len(BUSES)))
            P_MEG_vals    = np.array([[[_val(model.P_MEG[t, k, b]) for b in BUSES] for k in K] for t in horas]) if usar_meg else np.zeros((len(horas), len(K), len(BUSES)))
            # MEG arrendado: potencia por (hora, bus) sumando unidades, y la
            # lista de arriendos (evento, unidad, bus) que pagó el objetivo.
            P_ARR_tb = {}
            for (t, k, b) in idx_arr:
                v = _val(model.P_ARR[t, k, b])
                if v > 1e-9:
                    P_ARR_tb[(t, b)] = P_ARR_tb.get((t, b), 0.0) + v
            arriendos_meg = [(e, k, b) for e in EV_arr for k in KA_act for b in BUSES
                             if _val(model.x_arr[e, k, b]) > 0.5]
            # Fuera de horas_mess_set las variables no existen en el modelo (no
            # se crearon sobre model.T_mess): el MESS se reporta en su bus base,
            # sin cargar/descargar, tal como asume el resto del horizonte.
            u_mess_vals    = np.zeros((len(horas), n_mess, len(BUSES)))
            Pdis_mess_vals = np.zeros((len(horas), n_mess, len(BUSES)))
            Pch_mess_vals  = np.zeros((len(horas), n_mess, len(BUSES)))
            SOC_mess_vals  = np.zeros((len(horas), n_mess))
            y_mess_vals    = np.zeros(len(KM_mess))
            if usar_mess:
                y_mess_vals = np.array([round(_val(model.y_mess[k])) for k in KM_mess])
                for ti, t in enumerate(horas):
                    if t in horas_mess_set:
                        for ki, k in enumerate(KM_mess):
                            SOC_mess_vals[ti, ki] = _val(model.SOC_mess[t, k])
                            for bi, b in enumerate(BUSES):
                                Pdis_mess_vals[ti, ki, bi] = _val(model.Pdch_mess[t, k, b])
                                Pch_mess_vals[ti, ki, bi]  = _val(model.Pch_mess[t, k, b])
                                u_mess_vals[ti, ki, bi]    = _val(model.u_mess[k, b, t])
                    else:
                        for ki, k in enumerate(KM_mess):
                            u_mess_vals[ti, ki, bus_pos[bus_base_mess[k]]] = y_mess_vals[ki]
            CAP_mess_vals = (np.array([E_cap_mess * y_mess_vals[ki] for ki in range(len(KM_mess))])
                             if usar_mess else np.zeros(len(KM_mess)))
            P_EDRP_vals = np.array([[[pyo.value(model.P_EDRP[t, b, c]) or 0.0 for c in BLOQUES_EDRP] for b in BUSES] for t in horas]) if usar_edrp else np.zeros((len(horas), len(BUSES), C_bloques_EDRP))
            if sw_cand_list and usar_candidatas_dano:
                z_sw_vals  = np.array([[_val(model.z_sw[t, l])  for l in sw_cand_list] for t in horas])
                P_sw_vals  = np.array([[_val(model.P_sw[t, l])  for l in sw_cand_list] for t in horas])
                w_sw_vals  = np.array([[_val(model.w_sw[t, l])  for l in sw_cand_list] for t in horas])
                y_rcs_vals = np.array([round(_val(model.y_rcs[l])) for l in sw_cand_list])
                y_ms_vals  = np.array([round(_val(model.y_ms[l]))  for l in sw_cand_list])
                x_veg_vals = np.array([round(_val(model.x_veg[l])) for l in sw_cand_list])
                x_ref_vals = np.array([round(_val(model.x_ref[l])) for l in sw_cand_list])
                x_ug_vals  = np.array([round(_val(model.x_ug[l]))  for l in sw_cand_list])
                sw_pos = {l: li for li, l in enumerate(sw_cand_list)}
            else:
                z_sw_vals = P_sw_vals = w_sw_vals = np.array([])
                y_rcs_vals = y_ms_vals = np.array([])
                x_veg_vals = x_ref_vals = x_ug_vals = np.array([])
                sw_pos = {}

            if usar_microredes and sw_cand_list and (usar_inv_switch_rcs or usar_inv_switch_ms):
                z_fuente_vals = np.array([[round(_val(model.z_fuente[t, b])) for b in BUSES] for t in horas])
            else:
                z_fuente_vals = np.array([])

            for i, h in enumerate(horas):
                Dam_tl = Dam_tl_por_hora[h]
                Pf_calc = {}
                for l in LINES:
                    fi, tj = from_bus[l], to_bus[l]
                    if l in tie_indices:
                        Pf_calc[(fi, tj)] = P_tie_vals[i, tie_list.index(l)]
                    elif usar_candidatas_dano and l in sw_pos:
                        # El daño efectivo (post-hardening) ya está incorporado
                        # en z_sw/P_sw — no usar el Dam original acá.
                        Pf_calc[(fi, tj)] = P_sw_vals[i, sw_pos[l]]
                    elif Dam_tl[l] == 1:
                        Pf_calc[(fi, tj)] = 0.0
                    else:
                        Pf_calc[(fi, tj)] = (theta_vals[i, bus_pos[fi]] - theta_vals[i, bus_pos[tj]]) / x_ij[l]

                P_D_hora = {b: P_D_hora_arr[h, b] for b in BUSES}
                ENS_h    = float(P_LC_vals[i].sum())
                carga_servida = sum(P_D_hora[b] - P_LC_vals[i, bi] for bi, b in enumerate(BUSES))
                # `lineas_danadas_*` mantiene su significado histórico: las
                # líneas físicamente averiadas (las que requieren cuadrilla).
                # `lineas_fuera_servicio_*` agrega las que quedan sin
                # servicio solo por la zona de aislación; se reponen al
                # cerrar el seccionador, sin reparación de por medio.
                lineas_danadas_indices = np.where(Dam_fis_por_hora[h] == 1)[0].tolist()
                lineas_fuera_indices   = np.where(Dam_tl == 1)[0].tolist()
                # Dentro de las que quedan fuera sin estar averiadas hay dos
                # situaciones distintas:
                #   seccionadas → tienen seccionador y son la FRONTERA de la
                #     zona: se abrieron por maniobra, es la acción de aislar;
                #   aisladas    → no tienen con qué maniobrar y perdieron
                #     suministro por arrastre. Es el daño colateral.
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
                    'y_tie':       {l: int(y_tie_vals[li])               for li, l in enumerate(tie_list)} if usar_inv_tie else {},
                    'CAP_MEG_MW':  {k: CAP_meg_vals[ki]                  for ki, k in enumerate(K)},
                    'x_re':        {(k,b): round(x_re_vals[ki,bi])       for ki,k in enumerate(K) for bi,b in enumerate(BUSES)},
                    'P_MEG':       {(k,b): P_MEG_vals[i,ki,bi]           for ki,k in enumerate(K) for bi,b in enumerate(BUSES)},
                    'P_MEG_arr':   {b: P_ARR_tb.get((h, b), 0.0)         for b in BUSES} if usar_meg_arr else {},
                    'arriendos_MEG': arriendos_meg if (usar_meg_arr and i == 0) else [],
                    'u_mess':   {(k,b): round(u_mess_vals[i,ki,bi]) for ki,k in enumerate(KM) for bi,b in enumerate(BUSES)},
                    'Pdis_mess':   {(k,b): Pdis_mess_vals[i,ki,bi]        for ki,k in enumerate(KM) for bi,b in enumerate(BUSES)},
                    'Pch_mess':   {(k,b): Pch_mess_vals[i,ki,bi]        for ki,k in enumerate(KM) for bi,b in enumerate(BUSES)},
                    'P_mess':   {(k,b): Pdis_mess_vals[i,ki,bi] - Pch_mess_vals[i,ki,bi]        for ki,k in enumerate(KM) for bi,b in enumerate(BUSES)},
                    'SOC_mess': {k: SOC_mess_vals[i,ki]              for ki,k in enumerate(KM)},
                    'H_MESS_h': {k: h_mess for k in KM_mess},
                    'y_mess': {k: int(y_mess_vals[ki])                    for ki, k in enumerate(KM_mess)},
                    'CAP_MESS_MWh': {k: CAP_mess_vals[ki]                  for ki, k in enumerate(KM_mess)},
                    'P_MESS_MW': {k: CAP_mess_vals[ki]/h_mess                  for ki, k in enumerate(KM_mess)},
                    'P_EDRP': {(b, c): P_EDRP_vals[i, bi, c]             for bi, b in enumerate(BUSES) for c in range(C_bloques_EDRP)},
                    'EDRP_total': {b: sum(P_EDRP_vals[i, bi, c]          for c in range(C_bloques_EDRP)) for bi, b in enumerate(BUSES)},
                    'z_sw':    {l: round(z_sw_vals[i, li])  for li, l in enumerate(sw_cand_list)} if sw_cand_list and usar_candidatas_dano else {},
                    'P_sw':    {l: P_sw_vals[i, li]         for li, l in enumerate(sw_cand_list)} if sw_cand_list and usar_candidatas_dano else {},
                    'w_sw':    {l: round(w_sw_vals[i, li])  for li, l in enumerate(sw_cand_list)} if sw_cand_list and usar_candidatas_dano else {},
                    'y_rcs':   {l: y_rcs_vals[li]           for li, l in enumerate(sw_cand_list)} if sw_cand_list and usar_candidatas_dano else {},
                    'y_ms':    {l: y_ms_vals[li]            for li, l in enumerate(sw_cand_list)} if sw_cand_list and usar_candidatas_dano else {},
                    'x_veg':   {l: int(x_veg_vals[li])      for li, l in enumerate(sw_cand_list)} if sw_cand_list and usar_candidatas_dano else {},
                    'x_ref':   {l: int(x_ref_vals[li])      for li, l in enumerate(sw_cand_list)} if sw_cand_list and usar_candidatas_dano else {},
                    'x_ug':    {l: int(x_ug_vals[li])       for li, l in enumerate(sw_cand_list)} if sw_cand_list and usar_candidatas_dano else {},
                    'z_fuente': {b: int(z_fuente_vals[i, bi]) for bi, b in enumerate(BUSES)} if usar_microredes and sw_cand_list and (usar_inv_switch_rcs or usar_inv_switch_ms) else {},
                    'VoLL':    VoLL_local,
                    'fn_objetivo': fn_objetivo,
                    # Cómo terminó el solver. Se propaga junto al resultado
                    # porque una corrida cortada por TimeLimit devuelve un
                    # incumbente que se ve igual que un óptimo: sin esto, quien
                    # lea el resultado después no tiene forma de distinguirlos
                    # y un incumbente malo entra a los promedios como si nada.
                    'solver_term': _term,
                    'solver_gap': _gap_final,
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

