# ============================================================
# app_v6.py — Dashboard de Resiliencia de Red de Distribución
# Estructura modular: cada módulo tiene responsabilidad única.
# ============================================================
import streamlit as st
import io
import pickle
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from data_loader import (
    net, BUSES, LINES, GENS, ESS,
    faultlinMC, repairMC, demand_factor,
    alphaDGsolar, alphaDGeolica, bus_zone,
    P_D_base,
    N_eventos, sw_cand_list, tie_cand_list, C_incentivo_edrp,
    N_HORAS, horas, numesc, esc_esp_ui,
    r_econ, n_econ, frp,frp_movil, VoLL, data, _norm, buses_criticos, h_ess_4h,h_ess_5h ,h_ess_6h,
    baseline_esc, ESCENARIOS_CON_BASELINE, escenarios_con_baseline,
    p_sub_total_esc, lim_dg_e4, escenarios_muestra, escenarios_barrido )

# from caso_base import results_sin, P_sub_total

from indicadores import fn_SAIDI, fn_SAIFI, fn_ENS_total, fn_ASAI, fn_ens_costo_base


from helpers_granularidad import MESES_H,MESES_NOM,_agg_serie, _agg_df_energia, _x_label, _y_lbl, _apply_gran
from helpers_operativos import (
    PALETA, NODEFILL, _nombre_estrategia_corto, _color_zona, _results_to_list,
    _ens_por_hora, _carga_servida_por_hora, _lineas_danadas,
    _tasa_interconexion_por_hora, _ens_por_zona_total,
    _energia_dg_por_hora, _energia_fos_por_hora,
    _energia_meg_por_hora, _energia_mess_por_hora,
    _soc_mess_por_hora, _mess_por_bus, _extraer_cap_mess,
    _mess_resumen_unidades, _mess_localizacion_horaria,
    _tie_cerrados_por_hora, _soc_promedio_por_hora,
    _df_energia, _edrp_por_hora, calcular_edrp_nodo_bloque
)
from helpers_estadisticos import (
    _esc_peor_caso, _promedio_por_hora, _peor_caso_por_hora,
    _lineas_danadas_todos, _df_energia_promedio, _df_energia_peorcase,
)
from helpers_inversion import (
    _extraer_cap_dg, _extraer_cap_meg, _extraer_switches_inv,
    _extraer_costos_objetivo, construir_curva_abatimiento_2
)
from helpers_sinergias import (
    INDICADORES as INDICADORES_SIN, describir_medidas, sugerir_combinada,
    validar_diseno_factorial, claves_escenarios_comunes, serie_indicador,
    resumen_interaccion, tabla_factorial, tabla_interaccion_ic,
    df_por_escenario, tabla_economica, figura_waterfall,
    figura_barras_factorial, figura_distribucion_interaccion,
)


from grafo import (
    build_fig_grafo, build_fig_grafo_resumen, resumir_simulacion,
    resumir_fallas_mc, tablas_resumen,
    RESUMEN_SCHEMA,
)
# from optimization_3 import resolver_optimizacion
# from optimization_3_corregido import resolver_optimizacion
# from optimization_5_estocastica import (
    # resolver_optimizacion as resolver_optimizacion_estocastica)
from optimization_6_oficial import resolver_optimizacion


def _resolver_con_veg(data, config):
    """
    Gestión de vegetación, refuerzo estructural y soterramiento son
    decisiones por línea candidata (x_veg/x_ref/x_ug) DENTRO del MILP
    (ver optimization_3_corregido.py) — el solver elige a qué líneas de
    sw_cand_list conviene aplicar cada medida, sopesando su CAPEX/OPEX
    por km contra el ENS evitado. Ya no se hace swap de faultlinMC acá;
    optimization_3_corregido.py toma data["faultlinMC_veg"/"_ref"/"_ug"]
    directamente para construir el daño efectivo por línea.

    Con config["estocastico"] = True se despacha al modelo de dos etapas
    (optimization_5_estocastica): un solo plan de inversión compartido por
    todos los escenarios de config["escenarios_estoc"], y la operación
    resuelta por escenario. El contrato de entrada/salida es idéntico, así
    que el resto de la app no cambia.
    """
    if config.get("veg_mgmt", False) and not data.get("faultlinMC_veg_disponible", False):
        st.error(
            "❌ No se encontró 'FaultlinMC_veg.npy'. Corre la celda "
            "'GESTIÓN DE VEGETACIÓN' en Prueba 12 oficial.ipynb antes "
            "de activar esta medida."
        )
        return None
    if config.get("estocastico", False):
        return resolver_optimizacion_estocastica(data=data, config=config)
    return resolver_optimizacion(data=data, config=config)

# Figuras en pantalla vs. exportación: los helpers viven en formato_figuras.py
# para que la app v8 use exactamente los mismos (ver el comentario allá).
from formato_figuras import _fig_pantalla, _descargas_figura
from curvas_mac import curvas_mac_estrategias, tabla_costos_unidad


# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Red de Distribución",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Normalizar results_sin (ya viene procesado desde data_loader)

# SIDEBAR — controles globales mínimos
# ─────────────────────────────────────────────────────────────

st.sidebar.title("⚡ Panel de Control")

gran = st.sidebar.radio(
    "📅 Granularidad de gráficos",
    ["Diaria", "Mensual", "Horaria"],
    index=2,
    help="Horaria puede ser lento con 8760 h"
) if N_HORAS > 168 else "Horaria"

if numesc > 1:
    vista_global = st.sidebar.radio(
        "🔍 Vista global",
        ["📊 Promedio", "⚠️ Peor caso", "📌 Escenario seleccionado"],
        index=0,
    )
else:
    vista_global = "📌 Escenario seleccionado"
    st.sidebar.caption("📌 Solo hay 1 escenario Monte Carlo — se muestra ese caso único (no aplica Promedio/Peor caso).")

zona_sel = st.sidebar.multiselect(
    "Zonas a mostrar", options=[1, 2, 3], default=[1, 2, 3],
    format_func=lambda z: f"Zona {z}"
)

# ── Escenario Monte Carlo — ÚNICO selector de toda la app ────
# Antes había dos sliders independientes: uno en la pestaña de optimización
# ("esc_tab2") y otro en los KPIs de resultados ("esc_t3_kpi"). Nada obligaba
# a que coincidieran, así que la tabla de indicadores podía estar mostrando el
# escenario 7 mientras el grafo de más abajo, en la misma pantalla, dibujaba
# el 3 — sin ningún aviso. Al vivir acá se elige una vez y rige para todo.
#
# Va en la sidebar (que se ejecuta antes que las pestañas) a propósito: el
# escenario decide CUÁL baseline se carga más abajo, así que tiene que estar
# resuelto antes de esa carga. Con el slider dentro de tab2 había que leerlo
# del session_state, y quedaba un rerun por detrás de lo que el usuario acababa
# de elegir.
st.sidebar.markdown("---")
if numesc > 1:
    esc_sel = st.sidebar.slider(
        "🎲 Escenario Monte Carlo",
        min_value=1, max_value=numesc, value=esc_esp_ui, step=1,
        key="esc_global",
        help=f"Rige para todas las pestañas. Escenario del caso base "
             f"(esc_esp): {esc_esp_ui}",
    )
    if esc_sel == esc_esp_ui:
        st.sidebar.caption(f"📌 Es el escenario del caso base (`esc_esp = {esc_esp_ui}`).")
else:
    esc_sel = esc_esp_ui
    st.sidebar.caption(f"🎲 Solo hay 1 escenario disponible (esc_esp = {esc_esp_ui}).")
esc_sel = max(1, min(int(esc_sel), numesc))

# ─────────────────────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────────────────────

# Orden = orden del flujo de trabajo: de dónde salen los datos, cómo se
# optimiza, qué resultó, y el barrido de medidas al final.
tab1, tab2, tab3, tab4 = st.tabs([
    "📥 Datos de entrada", "⚙️ Optimización", "📊 Resultados", "🧪 Barrido"])

# Dentro de "Datos de entrada" hay dos preguntas distintas y ninguna es la
# otra: qué red y qué parámetros tiene cargados AHORA el modelo (lo que se
# mira a diario), y de dónde salieron esos datos / cómo se cambian (lo que
# se hace cuando cambia la red o el escenario climático). Una pestaña por
# pregunta, para no volver a tener dos entradas sueltas compitiendo.
with tab1:
    sub_modelo, sub_origen = st.tabs([
        "🔎 Modelo cargado", "🗺️ Origen de los datos"])
    with sub_origen:
        import entrada_red_vista
        entrada_red_vista.render()

# El barrido (cada medida x N escenarios) se CALCULA por consola con
# run_barrido_medidas.py y se guarda en artefactos/barrido/. Esta pestaña
# solo lee esos artefactos: son cientos de MILP de 8760 h y resolverlos
# dentro de Streamlit significaría perderlos en el primer rerun.
with tab4:
    import barrido_vista
    barrido_vista.render()

# `esc_sel` ya quedó fijado por el slider de la sidebar, más arriba.

# ─────────────────────────────────────────────────────────────
# SUPUESTO DE AISLACIÓN — se lee del estado de los widgets del panel
# "🔒 Aislación de la falla" (más abajo). Igual que el modo estocástico,
# se lee ACÁ porque determina CUÁL caso base corresponde: cada supuesto
# de aislación tiene su propio juego de baselines, y comparar una
# estrategia con aislación contra un caso base sin ella da un "ahorro de
# ENS" negativo aunque la medida sirva.
# ─────────────────────────────────────────────────────────────
from aislacion import variante as _variante_aislacion

_AISL_ON = bool(st.session_state.get("t_aislacion", False))
_AISL_SEC = list(st.session_state.get("sec_existentes_ui", []) or [])
VARIANTE_AISL = _variante_aislacion(_AISL_ON, _AISL_SEC)
ESCENARIOS_CON_BASELINE = escenarios_con_baseline(VARIANTE_AISL)

# El crédito de subestación del objetivo compara ΣP_SUB contra el del caso
# base. `data` trae el de la variante "base" (se arma al importar
# data_loader), así que hay que reapuntarlo al supuesto activo.
data["P_SUB_tot_por_esc"] = {e: p_sub_total_esc(e, VARIANTE_AISL)
                             for e in sorted(ESCENARIOS_CON_BASELINE)}
data["P_SUB_tot"] = data["P_SUB_tot_por_esc"].get(esc_sel - 1, 0.0)
# Lo mismo para el optimizador: los seccionadores viajan en `config`, pero
# dejarlos también en `data` cubre a quien construya su propio config.
data["seccionadores_existentes"] = sorted(_AISL_SEC) if _AISL_ON else []

# ─────────────────────────────────────────────────────────────
# MODO ESTOCÁSTICO — se lee del estado del widget de la pestaña 2.
# Streamlit vuelve a ejecutar el script completo al tocar el toggle, así
# que para cuando el usuario aprieta "Calcular" estas variables ya reflejan
# su elección. Se leen ACÁ (antes de tab2) porque el caso base contra el
# que se compara depende de cuántos escenarios entran al modelo.
# ─────────────────────────────────────────────────────────────
ESTOC_ACTIVO = bool(st.session_state.get("estoc_activo", False))
_estoc_sel_ui = st.session_state.get("estoc_escenarios", []) or []
# El multiselect guarda números 1-based (igual que el slider de escenario);
# internamente todo el resto del código usa índices 0-based.
ESCENARIOS_ESTOC = sorted({
    int(n) - 1 for n in _estoc_sel_ui
    if 1 <= int(n) <= numesc and (int(n) - 1) in ESCENARIOS_CON_BASELINE
})
if ESTOC_ACTIVO and not ESCENARIOS_ESTOC:
    # Sin selección todavía: se arranca con el escenario del slider.
    ESCENARIOS_ESTOC = [esc_sel - 1] if (esc_sel - 1) in ESCENARIOS_CON_BASELINE else []

# Caso base DEL ESCENARIO SELECCIONADO. Antes `results_sin` era un único
# baseline (el del escenario congelado) que se comparaba contra cualquier
# escenario que eligiera el usuario, sin aviso.
# En modo estocástico se cargan los baselines de TODOS los escenarios del
# programa: la comparación estrategia-vs-base es por escenario, así que el
# conjunto de baselines tiene que ser el mismo conjunto que resuelve el
# modelo (helpers_inversion._baseline_de empareja por posición).
# El sufijo `--seccionadores` es obligatorio al regenerar: sin él el
# pipeline reconstruye el baseline del supuesto histórico (sin aislación),
# que no es el que corresponde a la selección activa.
_ARG_SEC = (f" --seccionadores {','.join(str(l) for l in sorted(_AISL_SEC))}"
            if VARIANTE_AISL != "base" else "")

if ESTOC_ACTIVO and ESCENARIOS_ESTOC:
    results_sin = {}
    for _e in ESCENARIOS_ESTOC:
        _b = baseline_esc(_e, VARIANTE_AISL)
        if _b.get(_e):
            results_sin[_e] = _b[_e]
    _faltan_base = [e + 1 for e in ESCENARIOS_ESTOC if e not in results_sin]
    if _faltan_base:
        st.error(
            f"Los escenarios {_faltan_base} no tienen caso base calculado "
            f"para el supuesto de aislación activo (`{VARIANTE_AISL}`). "
            f"Generarlos con: `python pipeline.py --etapas E5 --escenarios "
            f"{','.join(str(e-1) for e in _faltan_base)}{_ARG_SEC}`"
        )
else:
    results_sin = baseline_esc(esc_sel - 1, VARIANTE_AISL)
if not results_sin:
    st.error(
        f"El escenario {esc_sel} no tiene caso base calculado para el "
        f"supuesto de aislación activo (`{VARIANTE_AISL}`). "
        f"Generarlo con: `python pipeline.py --etapas E5 --escenarios "
        f"{esc_sel-1}{_ARG_SEC}`  \n"
        f"Escenarios disponibles en esta variante: "
        f"{len(ESCENARIOS_CON_BASELINE)} de {numesc}."
    )

# Índices REALES (0-based) de los escenarios presentes en los resultados, en
# el mismo orden en que `_results_to_list` los devuelve (claves ordenadas).
ESC_PRESENTES = sorted(results_sin.keys())


def _pos_esc(esc_0b, n_disp=None):
    """Posición de un escenario (índice 0-based) dentro de los resultados.

    Toda la app venía traduciendo "escenario N" a posición con
    `min(esc_sel - 1, len(valid) - 1)`, que solo acierta cuando los
    resultados traen los escenarios 0..k-1 correlativos. Con un conjunto
    estocástico como [0, 5, 9] eso apunta al escenario equivocado. Acá se
    busca la posición real y se cae en 0 si ese escenario no está.
    """
    try:
        pos = ESC_PRESENTES.index(int(esc_0b))
    except (ValueError, TypeError):
        pos = 0
    if n_disp is not None:
        pos = min(pos, max(n_disp - 1, 0))
    return pos


SAIDI_sin = {}
valid_sin   = [v for v in _results_to_list(results_sin) if v]
esc_idx_kpi = _pos_esc(esc_esp_ui - 1, len(valid_sin)) if valid_sin else 0
res_kpi     = valid_sin[esc_idx_kpi] if valid_sin else []

SAIDI_sin[esc_esp_ui+1] = fn_SAIDI(res_kpi, P_D_base, BUSES, horas)
SAIDI_sin   = {i + 1: v for i, (k, v) in enumerate(sorted(SAIDI_sin.items()))}

# ═════════════════════════════════════════════════════════════
# PESTAÑA 1 — DATOS DE ENTRADA · sub-pestaña "Modelo cargado"
# ═════════════════════════════════════════════════════════════
# `sub_modelo` se creó arriba, junto a `sub_origen`. Se reabre acá (y no
# se movió todo el bloque) porque un `with <contenedor>` de Streamlit se
# puede reabrir cuantas veces se quiera: es la misma técnica con la que
# tab3 se declara arriba y se llena mil líneas más abajo.

with sub_modelo:
    st.title("📥 Parámetros del Sistema")
    st.header("🕸️ B — Grafo de la Red")

    col_ctrl, col_info = st.columns([3, 1])
    with col_ctrl:
        # Slider 1-based, igual que el campo 'hora' de los resultados y que
        # el eje X de los gráficos (data_loader.horas = range(1, N+1)). Antes
        # iba de 0 a N-1, así que la misma hora física aparecía como "90" acá
        # y como "91" en el gráfico de ENS: una hora de desfase aparente.
        # `key` nuevo a propósito: el valor guardado del slider viejo (0)
        # queda fuera del rango nuevo.
        hora_ui = st.slider("Hora", 1, N_HORAS, 1, key="slider_grafo_inicio_h1")
    with col_info:
        st.caption(f"Hora seleccionada: **{hora_ui}**")

    hora_grafo = hora_ui - 1        # índice 0-based en listas y arrays

    # Estado de fallas para la hora seleccionada
    fmc_hora = {}
    try:
        eidx = min(max(esc_sel - 1, 0), faultlinMC.shape[1] - 1)
        hidx = min(hora_grafo, faultlinMC.shape[0] - 1)
        for li, l in enumerate(LINES):
            if li < faultlinMC.shape[2]:
                fmc_hora[l] = bool(faultlinMC[hidx, eidx, li] > 0)
    except Exception:
        pass

    # Resultados de la hora seleccionada (caso base o estrategia activa)
    
    res_hora = {}
    # results_sin ahora trae SOLO el escenario seleccionado, con su índice
    # 0-based como clave; el `min(..., len(results_sin)-1)` anterior asumía
    # un dict 0-indexado con todos los escenarios y fallaba salvo esc_sel=1.
    h_list = results_sin.get(esc_sel - 1, [])
    if h_list and hora_grafo < len(h_list):
        res_hora = h_list[hora_grafo]

    # Se pasa la hora 1-based: dentro de build_fig_grafo solo se usa para
    # mostrarla, no para indexar.
    fig_grafo = build_fig_grafo(hora_ui, esc_sel, res_hora, fmc_hora, {}, buses_criticos)
    st.plotly_chart(fig_grafo, use_container_width=True, key="grafo_red_principal")

    st.subheader("🔌 Red Eléctrica — Buses")
    df_bus_show = net.bus[["name", "vn_kv"]].copy()
    df_bus_show.index.name = "Bus"
    df_bus_show["zona"] = df_bus_show.index.map(lambda b: bus_zone.get(b, "—"))
    df_bus_show.columns = ["Nombre", "Tensión nominal [kV]", "Zona"]
    st.dataframe(df_bus_show, use_container_width=True)

    st.subheader("🔌 Red Eléctrica — Líneas")
    df_line_show = net.line[["name", "from_bus", "to_bus", "length_km",
                              "r_ohm_per_km", "x_ohm_per_km", "max_i_ka"]].copy()
    df_line_show.columns = ["Nombre", "Bus origen", "Bus destino",
                             "Long. [km]", "R [Ω/km]", "X [Ω/km]", "I_max [kA]"]
    df_line_show["Candidata switch"] = df_line_show.index.map(lambda l: "✅" if l in sw_cand_list else "—")
    st.dataframe(df_line_show, use_container_width=True)

    # ── Parámetros Monte Carlo ───────────────────────────────
    st.subheader("🎲 Parámetros Monte Carlo")
    col1, col2, col3 = st.columns(3)
    col1.metric("N° Escenarios",  numesc)
    col2.metric("N° Horas",       N_HORAS)
    col3.metric("Tiempo rep. [h]", 3)

    # ── Parámetros económicos ────────────────────────────────
    st.subheader("💰 Parámetros Económicos")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tasa de descuento r", f"{r_econ:.1%}")
    col2.metric("Vida útil n [años]",  n_econ)
    col3.metric("Factor recuperación frp", f"{frp:.4f}")
    voll_mean_display = np.mean(list(VoLL.values())) if VoLL else 0
    col4.metric("VoLL promedio [USD/kWh]", f"{voll_mean_display:.1f}")

    df_voll = pd.DataFrame({"Bus": list(VoLL.keys()), "VoLL [USD/kWh]": list(VoLL.values())})
    with st.expander("VoLL por bus"):
        st.dataframe(df_voll.set_index("Bus"), use_container_width=True)

    # ── Costos de medidas ────────────────────────────────────
    st.subheader("📋 Costos de Medidas de Resiliencia")
    def _scalar(x):
        if isinstance(x, dict):
            return np.mean(list(x.values()))
        return float(x)

    df_costos = pd.DataFrame([
        {"Tecnología": "DG Solar",   "CAPEX [USD/kW]": _scalar(data["DG_sol"][0]), "OPEX [USD/kW·año]": _scalar(data["DG_sol"][1])},
        {"Tecnología": "DG Eólica",  "CAPEX [USD/kW]": _scalar(data["DG_eol"][0]), "OPEX [USD/kW·año]": _scalar(data["DG_eol"][1])},
        {"Tecnología": "DG Fósil",   "CAPEX [USD/kW]": _scalar(data["DG_fos"][0]), "OPEX [USD/kW·año]": _scalar(data["DG_fos"][1])},
        {"Tecnología": "ESS 5h",     "CAPEX [USD/kWh]": _scalar(data["ESS 5h"][0]), "OPEX [USD/kWh·año]": _scalar(data["ESS 5h"][1])},
        {"Tecnología": "ESS 4h",     "CAPEX [USD/kWh]": _scalar(data["ESS 4h"][0]), "OPEX [USD/kWh·año]": _scalar(data["ESS 4h"][1])},
        {"Tecnología": "ESS 6h",     "CAPEX [USD/kWh]": _scalar(data["ESS 6h"][0]), "OPEX [USD/kWh·año]": _scalar(data["ESS 6h"][1])},
        {"Tecnología": "MEG",        "CAPEX [USD/kW]": _scalar(data["MEG"][0]),     "OPEX [USD/kW·año]": _scalar(data["MEG"][1])},
        {"Tecnología": "EDRP (bloque 1)", "CAPEX [USD/kW]": 0, "OPEX [USD/kW·año]": C_incentivo_edrp[0]},
        {"Tecnología": "EDRP (bloque 2)", "CAPEX [USD/kW]": 0, "OPEX [USD/kW·año]": C_incentivo_edrp[1]},
        {"Tecnología": "EDRP (bloque 3)", "CAPEX [USD/kW]": 0, "OPEX [USD/kW·año]": C_incentivo_edrp[2]},
        {"Tecnología": "EDRP (bloque 4)", "CAPEX [USD/kW]": 0, "OPEX [USD/kW·año]": C_incentivo_edrp[3]},
    ])
    st.dataframe(df_costos.set_index("Tecnología"), use_container_width=True)

    # ── Perfil de demanda base ───────────────────────────────
    st.subheader("📈 Perfil de Demanda Base por Bus")
    fig_dem = go.Figure()
    buses_sample = BUSES[:] if len(BUSES) > 8 else BUSES
    n_show = min(N_HORAS, 24)
    for idx_b, b in enumerate(buses_sample):
        color = PALETA[idx_b % len(PALETA)]
        # Perfil real: P_D_base[b] * demand_factor[h, esc=0, bus_idx]
        bi = BUSES.index(b)
        y_perfil = [
            P_D_base.get(b, 0) * demand_factor[h, 0, bi]
            for h in range(n_show)
        ]
        fig_dem.add_trace(go.Scatter(
            x=list(range(1, n_show + 1)),
            y=y_perfil,
            mode="lines", name=f"Bus {b}",
            line=dict(color=color, width=2)
        ))
    fig_dem.update_layout(
        title="Perfil de demanda real [kW] — primeras 24 h (esc. 1)",
        xaxis_title="Hora", yaxis_title="P_D [kW]",
        height=380
    )
    st.plotly_chart(fig_dem, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PESTAÑA 2 — OPTIMIZACIÓN
# ═════════════════════════════════════════════════════════════

with tab2:
    st.title("⚙️ Configuración y Ejecución de Optimización")

    st.subheader("🔧 Activar Medidas de Resiliencia")

    def _marcar_todas_medidas():
        for _k in ["t_sol", "t_eol", "t_fos", "t_meg", "t_mess", "t_ess_5h", "t_ess_4h", "t_ess_6h", "t_crit", "t_edrp", "t_rec", "t_rcs", "t_ms", "t_veg", "t_micro", "t_ref", "t_ug"]:
            st.session_state[_k] = True

    def _desmarcar_todas_medidas():
        for _k in ["t_sol", "t_eol", "t_fos", "t_meg", "t_mess", "t_ess_5h", "t_ess_4h", "t_ess_6h", "t_crit", "t_edrp", "t_rec", "t_rcs", "t_ms", "t_veg", "t_micro", "t_ref", "t_ug"]:
            st.session_state[_k] = False


    col_mark1, col_mark2, _ = st.columns([1, 1, 4])
    col_mark1.button("✅ Marcar todas", key="btn_marcar_todas", on_click=_marcar_todas_medidas)
    col_mark2.button("↩️ Desmarcar todas", key="btn_desmarcar_todas", on_click=_desmarcar_todas_medidas)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    usar_dg_sol = col_m1.toggle("☀️ DG Solar", value=False, key="t_sol")
    usar_dg_eol = col_m1.toggle("💨 DG Eólica", value=False, key="t_eol")
    usar_dg_fos = col_m1.toggle("⛽ DG Fósil", value=False, key="t_fos")
    usar_meg    = col_m1.toggle("🚛 MEG", value=False, key="t_meg")

    # usar_ess_5h = col_m2.toggle("🔋 ESS 5h", value=False, key="t_ess_5h")
    usar_ess_4h = col_m2.toggle("🔋 ESS 4h", value=False, key="t_ess_4h")
    usar_ess_5h = col_m2.toggle("🔋 ESS 5h", value=False, key="t_ess_5h")
    usar_ess_6h = col_m2.toggle("🔋 ESS 6h", value=False, key="t_ess_6h")
    usar_mess   = col_m2.toggle("🚚 MESS", value=False, key="t_mess")
    usar_prior_criticos = col_m3.toggle("Priorización de cargas críticas", value=False, key="t_crit")

    usar_edrp = col_m3.toggle("📉 EDRP (Cargas Interrumpibles)", value=False, key="t_edrp")
    usar_veg_mgmt = col_m3.toggle(
        "🌳 Gestión de vegetación", value=False, key="t_veg",
        help="Poda/despeje de vegetación: reduce la probabilidad de falla por "
             "viento en las líneas tratadas (requiere FaultlinMC_veg.npy, "
             "generado en Prueba 12 oficial.ipynb). No agrega variables al "
             "MILP: se compara como escenario aparte, igual que en la tesis "
             "de Vieyra et al. y Cerrai et al. (2019)."
    )
    if usar_veg_mgmt and not data.get("faultlinMC_veg_disponible", False):
        st.warning(
            "⚠️ No se encontró 'FaultlinMC_veg.npy' junto a FaultlinMC.npy. "
            "Corre la celda 'GESTIÓN DE VEGETACIÓN' en Prueba 12 oficial.ipynb "
            "para generarlo antes de calcular esta medida."
        )
    usar_refuerzo = col_m3.toggle(
        "🔩 Refuerzo estructural", value=False, key="t_ref",
        help="Reforzar postes (clase superior): reduce la probabilidad de "
             "falla por viento en las líneas tratadas. Reducción de 45% "
             "(Darestani y Shafieezadeh, 2020, Engineering Structures: "
             "P_falla 0.64→0.34 a 150 mph al subir de Clase 5 a Clase 3). "
             "No agrega variables al MILP: se compara como escenario aparte."
    )
    usar_soterramiento = col_m3.toggle(
        "🕳️ Soterramiento", value=False, key="t_ug",
        help="Convertir línea aérea a subterránea: reducción de 95% en la "
             "probabilidad de falla por viento/árboles en las líneas "
             "tratadas (Ma et al., 2018: conductor soterrado ~invulnerable "
             "al término de fragilidad eólica; Larsen et al., 2020, Energy: "
             "confirma empíricamente la reducción de SAIFI). No agrega "
             "variables al MILP: se compara como escenario aparte."
    )
    usar_reconfig  = col_m4.toggle("🔀 Reconfiguración topológica", value=False, key="t_rec")
    usar_inv_tie   = col_m4.toggle(
        "🔗 Inversión en líneas de enlace", value=False, key="t_tie",
        help="A diferencia de la reconfiguración topológica (que usa enlaces "
             "YA existentes y solo paga la maniobra), acá cada tie-line es una "
             "CANDIDATA a construir: el MILP decide y_tie[l] ∈ {0,1} y paga "
             "CAPEX+OPEX por km del enlace. Solo un enlace construido puede "
             "cerrarse durante la falla."
    )
    usar_inv_rcs   = col_m4.toggle("🔌 Inversión Switch RCS (automático)", value=False, key="t_rcs")
    usar_inv_ms    = col_m4.toggle("🔌 Inversión Switch MS (manual)", value=False, key="t_ms")
    usar_microredes = col_m4.toggle(
        "🏝️ Formación de microrredes", value=False, key="t_micro",
        help="Restricción de cardinalidad (Kim y Dvorkin, 2019, Ec. 8a-8b): "
             "acota el n.º de líneas candidatas abiertas por la cantidad de "
             "buses con inyección local (DG/ESS/MEG/MESS) disponible en esa "
             "hora. No tiene CAPEX/OPEX propio: requiere Switch RCS y/o MS "
             "activos para tener efecto."
    )
    if usar_microredes and not (usar_inv_rcs or usar_inv_ms):
        st.warning(
            "⚠️ 'Formación de microrredes' no tiene efecto sin 'Inversión "
            "Switch RCS' o 'Inversión Switch MS' activos (son las líneas "
            "que esta restricción puede sectorizar)."
        )

    # ─────────────────────────────
    # Aislación de la falla por seccionadores preexistentes
    #
    # OJO: esto NO es una medida de resiliencia, es un supuesto de la red.
    # Por eso se aplica por igual al caso base y a todas las estrategias
    # (entra en `config` y también en `_config_base_medida`), y no aparece
    # en la lista de medidas a comparar ni en los chequeos de inversión.
    # ─────────────────────────────
    with st.container(border=True):
        st.markdown("### 🔒 Aislación de la falla")
        usar_aislacion = st.toggle(
            "Modelar zona de aislación por seccionadores",
            value=False, key="t_aislacion",
            help="Apagado: al fallar una línea sale de servicio SOLO esa "
                 "línea, lo que equivale a suponer un seccionador en cada "
                 "extremo de cada línea (aislación perfecta y gratis). "
                 "Encendido: la falla deja fuera a toda la sección "
                 "delimitada por los seccionadores declarados abajo, que es "
                 "el comportamiento real de un alimentador (Ettehad et al., "
                 "2024; Shahbazian et al., 2020)."
        )

        _lbl_linea = {}
        for _l in LINES:
            try:
                _lbl_linea[_l] = (f"L{_l} ({int(net.line.from_bus[_l])}–"
                                  f"{int(net.line.to_bus[_l])})")
            except Exception:
                _lbl_linea[_l] = f"L{_l}"

        if usar_aislacion:
            sec_existentes_sel = st.multiselect(
                "Líneas con seccionador preexistente",
                options=list(sw_cand_list),
                format_func=lambda l: _lbl_linea.get(l, f"L{l}"),
                key="sec_existentes_ui",
                help="Índices de línea que llevan seccionador. Son un dato "
                     "de la red (inversión hundida), no una decisión del "
                     "modelo: no tienen CAPEX en la función objetivo.",
            )

            # Vista previa del alcance de la falla, línea por línea. Se
            # calcula con la MISMA función que usa el optimizador, así que
            # lo que se muestra acá es exactamente lo que se va a resolver:
            # es la validación rápida de si la selección de seccionadores
            # quedó como se esperaba, sin tener que correr el MILP.
            #
            # Se lista por línea fallada y NO por "sección" porque las
            # secciones se solapan: una línea que lleva seccionador se
            # aísla a sí misma (falla en L2 → sale solo L2), mientras que
            # esa misma L2 sí entra en la salida cuando la falla ocurre en
            # una vecina sin seccionador (falla en L3 → salen L2, L3, L4).
            try:
                from optimization_6_oficial import (
                    zona_aislada as _zona_aislada_fn,
                    adyacencia_lineas as _adyacencia_fn,
                )

                _fb = {l: int(net.line.from_bus[l]) for l in LINES}
                _tb = {l: int(net.line.to_bus[l]) for l in LINES}
                _adj = _adyacencia_fn(LINES, _fb, _tb)
                # Las tie-lines NO son seccionadores: son líneas de enlace
                # normalmente abiertas que sirven para REPONER el tramo sano
                # aguas abajo (variable z_tie), mientras que el seccionador
                # es el equipo que CORTA para acotar la sección que sale.
                # Entran acá solo como exclusión: al estar normalmente
                # abiertas no conducen corriente de falla, así que la salida
                # no se propaga a través de ellas ni se listan como
                # ubicación posible de la falla.
                _ties = set(tie_cand_list)

                _filas = []
                for _l in LINES:
                    if _l in _ties:
                        continue
                    _dam = {k: 0 for k in LINES}
                    _dam[_l] = 1
                    _out = _zona_aislada_fn(dict(_dam), LINES, _fb, _tb, _adj,
                                            sec_existentes_sel, _ties)
                    _fuera = [k for k in LINES if _out[k] == 1]
                    _filas.append({
                        "Línea fallada": _lbl_linea.get(_l, f"L{_l}"),
                        "Seccionador": "sí" if _l in sec_existentes_sel else "—",
                        "Nº fuera de servicio": len(_fuera),
                        "Arrastra además a": ", ".join(
                            _lbl_linea.get(k, f"L{k}") for k in _fuera if k != _l
                        ) or "—",
                    })

                _n_lin = len(_filas)
                _peor = max((f["Nº fuera de servicio"] for f in _filas), default=0)
                _prom = (sum(f["Nº fuera de servicio"] for f in _filas) / _n_lin
                         if _n_lin else 0)
                st.caption(
                    f"Sobre {_n_lin} líneas del alimentador: en el **peor caso "
                    f"una falla deja fuera {_peor} línea(s)**, en promedio "
                    f"{_prom:.1f}. Con aislación perfecta ese número sería 1."
                )
                st.dataframe(pd.DataFrame(_filas), hide_index=True,
                             use_container_width=True)
                if not sec_existentes_sel:
                    st.warning(
                        "⚠️ Sin seccionadores declarados: cualquier falla "
                        "deja fuera al alimentador completo hasta que se "
                        "repare (solo despeja el interruptor de cabecera)."
                    )
            except Exception as _e:
                st.caption(f"No se pudo previsualizar el alcance de la falla: {_e}")
        else:
            sec_existentes_sel = []
            st.caption(
                "Supuesto actual: aislación perfecta (equivale a un "
                "seccionador en cada línea). Subestima ENS/SAIDI/SAIFI y "
                "deja sin beneficio a la inversión en switches."
            )

    st.subheader("🔩 Parámetros Condicionales")

    # ─────────────────────────────
    # MEG
    # ─────────────────────────────
    if usar_meg:
        with st.container(border=True):
            st.markdown("### MEG - Generación móvil")

            col_meg1, col_meg2 = st.columns(2)

            n_meg = col_meg1.number_input(
                "Nº MEGs",
                min_value=1,
                max_value=15,
                value=3,
                step=1,
                key="n_meg"
            )

            P_meg_max = col_meg2.number_input(
                "Cap. máx. MEG [kW]",
                min_value=0.1,
                value=50.0,
                step=0.1,
                key="P_meg_max"
            )

            T_tra_meg_h = col_meg2.number_input(
                "Tiempo de traslado MEG [h]",
                min_value=0,
                max_value=N_HORAS,
                value=2,
                step=1,
                help="Hora a partir de la cual el MEG está disponible",
                key="T_tra_meg_h"
            )
    else:
        n_meg, P_meg_max, T_tra_meg_h = 2, 1.0, 2


    # ─────────────────────────────
    # MESS
    # ─────────────────────────────
    if usar_mess:
        with st.container(border=True):
            st.markdown("### MESS - Almacenamiento móvil")

            col_mess1, col_mess2 = st.columns(2)

            n_mess = col_mess1.number_input(
                "Nº MESS",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                key="n_mess"
            )

            h_mess = col_mess1.number_input(
                "Duración MESS [h]",
                min_value=1.0,
                max_value=24.0,
                value=3.0,
                step=1.0,
                help="Duración equivalente del MESS a potencia nominal, igual al concepto usado en ESS",
                key="h_mess"
            )

            P_mess_max = col_mess2.number_input(
                "Potencia máxima MESS [kW]",
                min_value=0.1,
                value=200.0,
                step=0.1,
                key="P_mess_max"
            )

            E_cap_mess = P_mess_max * h_mess


            T_tra_mess = col_mess2.number_input(
                "Tiempo de traslado MESS [h]",
                min_value=0,
                max_value=N_HORAS,
                value=2,
                step=1,
                help="Hora a partir de la cual el MESS puede quedar disponible en el bus de destino",
                key="T_tra_mess"
            )
            col_mess2.caption(
                f"Capacidad energética derivada: {E_cap_mess:.2f} kWh = "
                f"{P_mess_max:.2f} kW × {h_mess:.2f} h"
            )
    else:
        n_mess = 0
        h_mess = 4.0
        P_mess_max = 0.0
        E_cap_mess = 0.0
        T_tra_mess = 1
    # ─────────────────────────────
    # MS / RCS
    # ─────────────────────────────
    if usar_inv_ms or usar_inv_rcs:
        with st.container(border=True):
            st.markdown("### MS / RCS - Maniobras e inversión en switches")

            # st.caption(f"🔌 Líneas inter-zona candidatas detectadas: {sw_cand_list}")
            st.caption(f"🔌 Líneas candidatas detectadas: {sw_cand_list}")


            col_ms1, col_ms2 = st.columns(2)

            if usar_inv_ms:
                T_maniobra = col_ms1.number_input(
                    "Tiempo maniobra MS [h]",
                    min_value=0,
                    max_value=6,
                    value=1,
                    step=1,
                    key="T_maniobra_ms"
                )
            else:
                T_maniobra = 1
    else:
        T_maniobra = 1

    st.subheader("🎲 Escenario Monte Carlo a analizar")
    # El selector vive en la sidebar y rige para toda la app (ver el bloque
    # "Escenario Monte Carlo" de arriba). Acá solo se informa cuál está activo:
    # es el que va a `config["esc_debug"]`, o sea el que realmente resuelve el
    # MILP al apretar "Calcular".
    if numesc > 1:
        st.info(
            f"Se optimizará el **escenario {esc_sel}** de {numesc}. "
            "Para cambiarlo, usa el selector 🎲 de la barra lateral — es el "
            "mismo que gobierna los KPIs y los grafos de la pestaña de "
            "resultados."
            + (f"  \n📌 Es el escenario del caso base (`esc_esp = {esc_esp_ui}`)."
               if esc_sel == esc_esp_ui else "")
        )
    else:
        st.info(f"Solo hay 1 escenario disponible (esc_esp = {esc_esp_ui}).")

    # ── Modelo estocástico de dos etapas ─────────────────────────────
    st.markdown("---")
    usar_estocastico = st.toggle(
        "🎲 Modelo estocástico de dos etapas",
        key="estoc_activo",
        help="Resuelve UN plan de inversión común a todos los escenarios "
             "seleccionados (decisión here-and-now), con la operación "
             "optimizada por escenario. Sin esto, cada escenario se resuelve "
             "por separado y obtiene su propio plan, óptimo conociendo de "
             "antemano sus fallas (wait-and-see).",
    )

    if usar_estocastico:
        _opciones_esc = [e + 1 for e in sorted(ESCENARIOS_CON_BASELINE)]
        # El default se filtra contra las opciones: Streamlit lanza excepción
        # si trae un valor que no está en `options`.
        _default_esc = [n for n in (st.session_state.get("estoc_escenarios")
                                    or [esc_sel]) if n in _opciones_esc]
        escenarios_estoc_ui = st.multiselect(
            "Escenarios del programa estocástico",
            options=_opciones_esc,
            default=_default_esc,
            key="estoc_escenarios",
            help="Solo aparecen los escenarios que ya tienen caso base (etapa "
                 "E5 del pipeline). Todos entran con la misma probabilidad "
                 "1/|Omega|. Empezar con pocos: el modelo crece de forma "
                 "lineal en el número de escenarios.",
        )
        escenarios_estoc = sorted({int(n) - 1 for n in escenarios_estoc_ui})
        if not escenarios_estoc:
            st.warning("Selecciona al menos un escenario.")
        else:
            st.caption(
                f"|Omega| = {len(escenarios_estoc)} escenario(s), "
                f"p = {1/len(escenarios_estoc):.4f} cada uno. "
                f"El horizonte es de {N_HORAS} h por escenario."
            )
            if N_HORAS < 8760:
                st.caption(
                    "⚠️ Con horizonte parcial el CAPEX se anualiza sobre un "
                    "año completo pero el beneficio solo se cuenta en las "
                    f"{N_HORAS} h modeladas, así que el modelo tiende a no "
                    "invertir."
                )
    else:
        escenarios_estoc = []

    config = {
        "dg_sol":          usar_dg_sol,
        "dg_eol":          usar_dg_eol,
        "dg_fos":          usar_dg_fos,
        "ess_5h":          usar_ess_5h,
        "ess_4h":          usar_ess_4h,
        "ess_6h":          usar_ess_6h,
        "reconfig_topo":   usar_reconfig,
        "inv_tie":         usar_inv_tie,
        "tie_candidatas":  tie_cand_list,
        "prior_criticos":  usar_prior_criticos,
        "buses_criticos":  buses_criticos,
        "meg":             usar_meg,
        "n_meg":           n_meg,
        "P_meg_max":       P_meg_max,
        "T_tra_meg":       {b: T_tra_meg_h for b in BUSES},
        "mess":            usar_mess,
        "n_mess":          n_mess,
        "h_mess":          h_mess,
        "E_cap_mess":      E_cap_mess,
        "E_cap_mess_total": n_mess * E_cap_mess,
        "P_mess_max":      P_mess_max,
        "T_tra_mess":      T_tra_mess,
        "inv_switch_rcs":  usar_inv_rcs,
        "inv_switch_ms":   usar_inv_ms,
        "sw_candidatas":   sw_cand_list,
        "T_maniobra_MS":   T_maniobra,
        "microredes":      usar_microredes,
        # Supuesto de red (no una medida): se aplica igual al caso base y a
        # todas las estrategias, para que la comparación sea contra el mismo
        # comportamiento físico de aislación.
        "aislacion_fallas_fija":    usar_aislacion,
        "seccionadores_existentes": sec_existentes_sel,
        "edrp":            usar_edrp,
        "veg_mgmt":        usar_veg_mgmt,
        "refuerzo":        usar_refuerzo,
        "soterramiento":   usar_soterramiento,
        # Tope de DG del artefacto E4 que se cargó: el crédito de DG del
        # objetivo sale de ahí, así que resolver con otro tope compararía
        # contra una capacidad máxima distinta (ver data_loader.lim_dg_e4).
        **({"lim_dg": lim_dg_e4} if lim_dg_e4 is not None else {}),
        "numesc":          numesc,
        # En modo estocástico esc_debug tiene que ir en None: si viene, el
        # modelo colapsa a ese único escenario e ignora escenarios_estoc.
        "esc_debug":       None if usar_estocastico else esc_sel - 1,
        "estocastico":     usar_estocastico,
        "escenarios_estoc": escenarios_estoc,
        "prob_escenarios": None,      # uniforme
        "horas":           N_HORAS,
    }

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    recalcular  = col_btn1.button("▶️ Calcular",  key="btn_recalcular")
    limpiar_hist = col_btn2.button("🗑️ Limpiar",  key="btn_limpiar")

    col_bt_todas, _ = st.columns([1, 2])
    recalcular_todas = col_bt_todas.button("▶️ Calcular cada medida", key="btn_recalcular_cada_una")

    # El crédito de DG del objetivo sale del artefacto de E4. Si ese artefacto
    # no dice con qué tope de capacidad se calculó, no hay forma de garantizar
    # que la optimización resuelva con el mismo, y la diferencia no produce
    # ningún error: solo deja el resultado mal en silencio.
    if lim_dg_e4 is None:
        st.caption(
            "ℹ️ El artefacto de caso sin fallas (E4) no registra con qué tope de "
            "capacidad de DG se calculó, así que rige el default del modelo. "
            "Para dejarlo explícito: "
            "`python pipeline.py --etapas E4 --lim-dg 1.0 --etiqueta LIM-1.00`."
        )
    else:
        st.caption(f"ℹ️ Tope de capacidad de DG: **{lim_dg_e4} × demanda de cada "
                   f"barra**, tomado del artefacto de caso sin fallas (E4).")

    def _config_base_medida():
        return {
            "dg_sol": False,
            "dg_eol": False,
            "dg_fos": False,
            "ess_5h": False,
            "ess_4h": False,
            "ess_6h": False,
            "reconfig_topo": False,
            "inv_tie": False,
            "tie_candidatas": tie_cand_list,
            "prior_criticos": False,
            "buses_criticos": buses_criticos,
            "meg": False,
            "n_meg": n_meg,
            "P_meg_max": P_meg_max,
            "T_tra_meg": {b: T_tra_meg_h for b in BUSES},
            "mess": False,
            "n_mess": n_mess,
            "h_mess": h_mess,
            "E_cap_mess": E_cap_mess,
            "E_cap_mess_total": n_mess * E_cap_mess,
            "P_mess_max": P_mess_max,
            "T_tra_mess": T_tra_mess,
            "inv_switch_rcs": False,
            "inv_switch_ms": False,
            "sw_candidatas": sw_cand_list,
            "T_maniobra_MS": T_maniobra,
            "microredes": False,
            # No se apaga con las medidas: es el supuesto de red común a
            # todas las corridas individuales y al caso base.
            "aislacion_fallas_fija": usar_aislacion,
            "seccionadores_existentes": sec_existentes_sel,
            "edrp": False,
            "veg_mgmt": False,
            "refuerzo": False,
            "soterramiento": False,
            **({"lim_dg": lim_dg_e4} if lim_dg_e4 is not None else {}),
            "numesc": numesc,
            "esc_debug": None if usar_estocastico else esc_sel - 1,
            "estocastico": usar_estocastico,
            "escenarios_estoc": escenarios_estoc,
            "prob_escenarios": None,
            "horas": N_HORAS,
        }

    if recalcular_todas:
        medidas_activas = []
        if usar_dg_sol: medidas_activas.append(("DG Sol", {"dg_sol": True}))
        if usar_dg_eol: medidas_activas.append(("DG Eól", {"dg_eol": True}))
        if usar_dg_fos: medidas_activas.append(("DG Fos", {"dg_fos": True}))
        if usar_ess_4h: medidas_activas.append(("ESS 4h", {"ess_4h": True}))
        if usar_ess_5h: medidas_activas.append(("ESS 5h", {"ess_5h": True}))
        if usar_ess_6h: medidas_activas.append(("ESS 6h", {"ess_6h": True}))
        if usar_reconfig: medidas_activas.append(("Reconfig", {"reconfig_topo": True}))
        if usar_inv_tie: medidas_activas.append((f"Enlaces nuevos ({len(tie_cand_list)} cand.)", {"inv_tie": True}))
        if usar_prior_criticos: medidas_activas.append(("Priorización cargas críticas", {"prior_criticos": True}))
        if usar_meg: medidas_activas.append((f"MEG×{n_meg}", {"meg": True}))
        if usar_mess: medidas_activas.append((f"MESS×{n_mess} ({h_mess:.0f}h)", {"mess": True}))
        if usar_inv_rcs: medidas_activas.append((f"Switch RCS ({len(sw_cand_list)} líneas)", {"inv_switch_rcs": True, "microredes": usar_microredes}))
        if usar_inv_ms: medidas_activas.append((f"Switch MS ({len(sw_cand_list)} líneas)", {"inv_switch_ms": True, "microredes": usar_microredes}))
        if usar_edrp: medidas_activas.append(("EDRP", {"edrp": True}))
        if usar_veg_mgmt: medidas_activas.append(("Gestión vegetación", {"veg_mgmt": True}))
        if usar_refuerzo: medidas_activas.append(("Refuerzo estructural", {"refuerzo": True}))
        if usar_soterramiento: medidas_activas.append(("Soterramiento", {"soterramiento": True}))

        if not medidas_activas:
            st.warning("⚠️ Activa al menos una medida (toggle) para poder calcularlas de forma individual.")
        else:
            import time

            historial = st.session_state.get("historial", {})
            total_medidas = len(medidas_activas)
            progreso_medidas = st.progress(0.0)
            estado_medidas = st.empty()
            resumen_calculo = []

            for i_med, (label_med, flags_med) in enumerate(medidas_activas, start=1):
                estado_medidas.info(f"⏳ Calculando medida {i_med}/{total_medidas}: **{label_med}**...")

                config_med = _config_base_medida()
                config_med.update(flags_med)

                t0_med = time.time()
                results_med = _resolver_con_veg(data, config_med)
                t_elapsed_med = time.time() - t0_med

                if results_med:
                    # En modo estocástico NO se renumeran los escenarios: las
                    # claves son los índices MC reales y tienen que seguir
                    # coincidiendo con las de `results_sin` para que la
                    # comparación contra el caso base sea la correcta.
                    if not usar_estocastico:
                        results_med = _norm(results_med)
                    saidi_med = {
                        esc + 1: round(fn_SAIDI(res_lista, P_D_base, BUSES, horas), 2)
                        for esc, res_lista in results_med.items()
                    }

                    nombre_med = label_med
                    base_nombre_med = nombre_med
                    contador_med = 2
                    while nombre_med in historial:
                        nombre_med = f"{base_nombre_med} ({contador_med})"
                        contador_med += 1

                    historial[nombre_med] = {
                        "results": results_med,
                        "saidi": saidi_med,
                        "config": config_med.copy(),
                        "tiempo": round(t_elapsed_med, 1),
                    }
                    resumen_calculo.append({"Medida": nombre_med, "Estado": "✅ OK", "Tiempo [s]": round(t_elapsed_med, 1)})
                else:
                    resumen_calculo.append({"Medida": label_med, "Estado": "❌ Error", "Tiempo [s]": round(t_elapsed_med, 1)})

                progreso_medidas.progress(i_med / total_medidas)

            st.session_state["historial"] = historial
            estado_medidas.success(f"✅ Se calcularon {total_medidas} medida(s) de forma individual y se guardaron en el historial.")
            st.dataframe(pd.DataFrame(resumen_calculo).set_index("Medida"), use_container_width=True)

    if limpiar_hist:
        st.session_state["historial"] = {}
        st.success("🗑️ Historial limpiado")

    if recalcular:
        partes = []
        if usar_dg_sol:  partes.append("DG Sol")
        if usar_dg_eol:  partes.append("DG Eól")
        if usar_dg_fos:  partes.append("DG Fos")
        # if usar_ess_5h:  partes.append(f"ESS 5h")
        if usar_ess_4h:  partes.append(f"ESS 4h")
        if usar_ess_5h:  partes.append(f"ESS 5h")
        if usar_ess_6h:  partes.append(f"ESS 6h")
        if usar_reconfig: partes.append("Reconfig")
        if usar_inv_tie: partes.append(f"Enlaces nuevos ({len(tie_cand_list)} cand.)")
        if usar_prior_criticos: partes.append("Priorización cargas críticas")
        if usar_meg:     partes.append(f"MEG×{n_meg}")
        if usar_mess:    partes.append(f"MESS×{n_mess} ({h_mess:.0f}h)")
        if usar_inv_rcs:          partes.append(f"Switch RCS ({len(sw_cand_list)} líneas)")
        if usar_inv_ms:           partes.append(f"Switch MS ({len(sw_cand_list)} líneas)")
        if usar_microredes and (usar_inv_rcs or usar_inv_ms): partes.append("Microrredes")
        if usar_edrp: partes.append("EDRP")
        if usar_veg_mgmt: partes.append("Gestión vegetación")
        if usar_refuerzo: partes.append("Refuerzo estructural")
        if usar_soterramiento: partes.append("Soterramiento")

        label = " + ".join(partes) if partes else "Sin estrategia"

        log_placeholder = st.empty()
        with st.spinner(f"⏳ Optimizando '{label}'..."):
            import time
            t0 = time.time()
            results_nuevo = _resolver_con_veg(data, config)
            t_elapsed = time.time() - t0

        if results_nuevo:
            if not usar_estocastico:
                results_nuevo = _norm(results_nuevo)
            saidi_nuevo   = {
                esc + 1: round(fn_SAIDI(res_lista, P_D_base, BUSES, horas), 2)
                for esc, res_lista in results_nuevo.items()
            }
            historial = st.session_state.get("historial", {})
            _base_label = label
            _counter = 2
            while label in historial:
                label = f"{_base_label} ({_counter})"
                _counter += 1
            historial[label] = {
                "results": results_nuevo,
                "saidi":   saidi_nuevo,
                "config":  config.copy(),
                "tiempo":  round(t_elapsed, 1),
            }
            st.session_state["historial"] = historial
            st.success(f"✅ '{label}' guardado")

            # Métricas intas
            st.subheader("📌 Resultados Post-Optimización")
            _ens_series = [_ens_por_hora(v) for v in _results_to_list(results_nuevo) if v]
            ens_total   = sum(sum(s) for s in _ens_series) if _ens_series else 0.0
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric("⏱️ Tiempo de cómputo", f"{t_elapsed:.1f} s")
            cm2.metric("⚡ ENS total [kWh]", f"{ens_total:.2f}")
            cm3.metric("📊 SAIDI promedio", f"{np.mean(list(saidi_nuevo.values())):.2f} h/año")
        else:
            st.error("❌ Error en la optimización")

    with st.expander("📋 Ver log del solver"):
        st.info("Los mensajes del solver se muestran en la consola del servidor.")

    # ── Historial de estrategias ─────────────────────────────
    historial = st.session_state.get("historial", {})
    if historial:
        st.subheader("📚 Historial de Estrategias Calculadas")
        rows = []
        for lbl, val in historial.items():
            saidi_vals_h = list(val["saidi"].values())
            rows.append({
                "Estrategia":       lbl,
                "SAIDI prom.":      f"{np.mean(saidi_vals_h):.2f}",
                "SAIDI peor caso":  f"{max(saidi_vals_h):.2f}",
                "Tiempo [s]":       val.get("tiempo", "—"),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Estrategia"), use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Guardar / Cargar resultados")

    # Se guardan los resultados de optimización, no los gráficos.
    # Las figuras y tablas se reconstruyen automáticamente al cargar el historial.
    VERSION_ARCHIVO_RESULTADOS = 2
    LIMITE_RECOMENDADO_MB = 190.0

    def _firma_modelo_actual():
        """Datos mínimos para evitar cargar resultados incompatibles.

        Delega en pipeline.firma_modelo() para que la app y el barrido de
        consola produzcan exactamente la misma firma: si divergieran, la app
        rechazaría archivos que en realidad sí puede leer.
        """
        import pipeline as _P
        return _P.firma_modelo()

    def _serializar_resultados(historial_exportar):
        """
        Serializa directamente dentro de GZIP.

        Evita crear primero un pickle completo y luego otra copia comprimida,
        por lo que usa menos memoria. El nivel 1 es mucho más rápido y suele
        comprimir suficientemente bien los diccionarios horarios repetitivos.
        """
        payload = {
            "version_archivo": VERSION_ARCHIVO_RESULTADOS,
            "fecha_exportacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "firma_modelo": _firma_modelo_actual(),
            "historial": historial_exportar,
        }

        buffer = io.BytesIO()
        with gzip.GzipFile(
            fileobj=buffer,
            mode="wb",
            compresslevel=1,
            mtime=0,
        ) as archivo_gzip:
            pickle.dump(
                payload,
                archivo_gzip,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return buffer.getvalue()

    def _deserializar_resultados(datos_comprimidos):
        """Abre únicamente archivos generados por esta misma aplicación."""
        with gzip.GzipFile(fileobj=io.BytesIO(datos_comprimidos), mode="rb") as archivo_gzip:
            payload = pickle.load(archivo_gzip)

        if not isinstance(payload, dict) or "historial" not in payload:
            raise ValueError("El archivo no contiene un historial válido.")

        version = payload.get("version_archivo")
        if version not in (1, VERSION_ARCHIVO_RESULTADOS):
            raise ValueError(
                f"Versión de archivo incompatible: {version}. "
                f"La aplicación espera la versión {VERSION_ARCHIVO_RESULTADOS}."
            )

        firma_guardada = payload.get("firma_modelo", {})
        firma_actual = _firma_modelo_actual()
        campos_criticos = [
            "N_HORAS", "numesc", "BUSES", "LINES",
            "shape_faultlinMC", "shape_demand_factor", "P_D_base",
        ]
        diferencias = [
            campo for campo in campos_criticos
            if firma_guardada.get(campo) != firma_actual.get(campo)
        ]
        if diferencias:
            raise ValueError(
                "Los resultados corresponden a otra configuración del modelo. "
                "Difieren: " + ", ".join(diferencias)
            )

        historial_cargado = payload["historial"]
        if not isinstance(historial_cargado, dict):
            raise ValueError("El historial del archivo no es un diccionario válido.")

        return historial_cargado

    def _agregar_historial_sin_sobrescribir(destino, nuevo):
        """Permite cargar uno o varios archivos y conserva estrategias repetidas."""
        for nombre, contenido in nuevo.items():
            nombre_final = nombre
            contador = 2
            while nombre_final in destino:
                nombre_final = f"{nombre} ({contador})"
                contador += 1
            destino[nombre_final] = contenido
        return destino

    historial_actual = st.session_state.get("historial", {})

    if historial_actual:
        nombres_estrategias = list(historial_actual.keys())
        estrategia_exportar = st.sidebar.selectbox(
            "Resultado a preparar",
            options=nombres_estrategias,
            key="estrategia_exportar_resultados",
            help=(
                "Se recomienda descargar una estrategia por archivo. "
                "Después puedes cargar varios archivos simultáneamente."
            ),
        )

        col_preparar_1, col_preparar_2 = st.sidebar.columns(2)

        preparar_una = col_preparar_1.button(
            "Preparar una",
            key="preparar_una_estrategia_gz",
            use_container_width=True,
        )
        preparar_todas = col_preparar_2.button(
            "Preparar todas",
            key="preparar_historial_completo_gz",
            use_container_width=True,
            help="Puede tardar y ocupar más memoria cuando el historial es grande.",
        )

        if preparar_una:
            try:
                with st.sidebar.spinner("Comprimiendo estrategia..."):
                    archivo = _serializar_resultados({
                        estrategia_exportar: historial_actual[estrategia_exportar]
                    })
                indice = nombres_estrategias.index(estrategia_exportar) + 1
                st.session_state["resultados_gz_bytes"] = archivo
                st.session_state["resultados_gz_nombre"] = (
                    f"resultado_estrategia_{indice:02d}.pkl.gz"
                )
                st.session_state["resultados_gz_descripcion"] = estrategia_exportar
            except Exception as e:
                st.sidebar.error(f"No se pudo preparar la estrategia: {e}")

        if preparar_todas:
            try:
                with st.sidebar.spinner("Comprimiendo historial completo..."):
                    archivo = _serializar_resultados(historial_actual)
                st.session_state["resultados_gz_bytes"] = archivo
                st.session_state["resultados_gz_nombre"] = "resultados_optimizacion.pkl.gz"
                st.session_state["resultados_gz_descripcion"] = "Historial completo"
            except Exception as e:
                st.sidebar.error(f"No se pudo preparar el historial: {e}")

        if "resultados_gz_bytes" in st.session_state:
            datos_descarga = st.session_state["resultados_gz_bytes"]
            size_mb = len(datos_descarga) / (1024 ** 2)
            descripcion = st.session_state.get(
                "resultados_gz_descripcion", "Resultados preparados"
            )

            st.sidebar.caption(f"Preparado: **{descripcion}**")
            st.sidebar.write(f"Tamaño comprimido: **{size_mb:.2f} MB**")

            if size_mb <= LIMITE_RECOMENDADO_MB:
                st.sidebar.download_button(
                    label="⬇️ Descargar archivo preparado",
                    data=datos_descarga,
                    file_name=st.session_state.get(
                        "resultados_gz_nombre",
                        "resultados_optimizacion.pkl.gz",
                    ),
                    mime="application/gzip",
                    key="descargar_resultados_gz",
                    on_click="ignore",
                    use_container_width=True,
                    help=(
                        "La descarga no vuelve a ejecutar la aplicación. "
                        "Incluye resultados, configuración, SAIDI y tiempo de cálculo."
                    ),
                )
            else:
                st.sidebar.error(
                    "El archivo preparado supera 190 MB. "
                    "Selecciona una sola estrategia y presiona 'Preparar una'."
                )

            if st.sidebar.button(
                "Liberar archivo preparado",
                key="liberar_resultados_preparados",
                use_container_width=True,
            ):
                st.session_state.pop("resultados_gz_bytes", None)
                st.session_state.pop("resultados_gz_nombre", None)
                st.session_state.pop("resultados_gz_descripcion", None)
                st.rerun()
    else:
        st.sidebar.info("Ejecuta o carga al menos una estrategia para habilitar la descarga.")

    # Se pueden seleccionar varios archivos cuando las estrategias se guardaron por separado.
    # La carga es explícita mediante botón para evitar que Streamlit vuelva a
    # descomprimir archivos grandes en cada rerun de la interfaz.
    archivos_subidos = st.sidebar.file_uploader(
        "📂 Seleccionar resultados guardados",
        type=["gz", "pkl"],
        accept_multiple_files=True,
        help=(
            "Admite archivos .pkl.gz creados por esta aplicación. "
            "Selecciona uno o varios archivos y luego presiona 'Cargar seleccionados'."
        ),
        key="seleccionar_resultados_guardados",
    )

    def _identificador_archivo_subido(archivo):
        """Identificador liviano: no copia ni descomprime el archivo."""
        file_id = getattr(archivo, "file_id", "")
        size = getattr(archivo, "size", None)
        if size is None:
            posicion = archivo.tell()
            archivo.seek(0, io.SEEK_END)
            size = archivo.tell()
            archivo.seek(posicion)
        return f"{file_id}|{archivo.name}|{int(size)}"

    def _deserializar_archivo_subido(archivo):
        """Deserializa desde el UploadedFile, evitando archivo.getvalue()."""
        archivo.seek(0)
        nombre = archivo.name.lower()

        if nombre.endswith(".pkl.gz") or nombre.endswith(".gz"):
            with gzip.GzipFile(fileobj=archivo, mode="rb") as archivo_gzip:
                payload = pickle.load(archivo_gzip)
        elif nombre.endswith(".pkl"):
            # Compatibilidad opcional con archivos pickle sin comprimir.
            payload = pickle.load(archivo)
        else:
            raise ValueError(
                f"Formato no reconocido para '{archivo.name}'. "
                "Usa un archivo .pkl.gz generado por esta aplicación."
            )

        if not isinstance(payload, dict) or "historial" not in payload:
            raise ValueError(
                f"'{archivo.name}' no contiene un historial de resultados válido."
            )

        version = payload.get("version_archivo")
        if version not in (1, VERSION_ARCHIVO_RESULTADOS):
            raise ValueError(
                f"'{archivo.name}' usa la versión {version}; "
                f"esta aplicación espera {VERSION_ARCHIVO_RESULTADOS}."
            )

        firma_guardada = payload.get("firma_modelo", {})
        firma_actual = _firma_modelo_actual()
        campos_criticos = [
            "N_HORAS", "numesc", "BUSES", "LINES",
            "shape_faultlinMC", "shape_demand_factor", "P_D_base",
        ]
        diferencias = [
            campo for campo in campos_criticos
            if firma_guardada.get(campo) != firma_actual.get(campo)
        ]
        if diferencias:
            raise ValueError(
                f"'{archivo.name}' corresponde a otra configuración del modelo. "
                "Difieren: " + ", ".join(diferencias)
            )

        historial_cargado = payload["historial"]
        if not isinstance(historial_cargado, dict):
            raise ValueError(
                f"El historial de '{archivo.name}' no es un diccionario válido."
            )

        return historial_cargado

    cargar_seleccionados = st.sidebar.button(
        "⬆️ Cargar seleccionados",
        key="boton_cargar_resultados_seleccionados",
        disabled=not bool(archivos_subidos),
        use_container_width=True,
    )

    if cargar_seleccionados and archivos_subidos:
        historial_destino = dict(st.session_state.get("historial", {}))
        archivos_ok = 0
        estrategias_nuevas = 0
        errores_carga = []

        barra_carga = st.sidebar.progress(0.0)
        estado_carga = st.sidebar.empty()
        total_archivos = len(archivos_subidos)

        for indice_archivo, archivo in enumerate(archivos_subidos, start=1):
            identificador = _identificador_archivo_subido(archivo)
            estado_carga.info(
                f"Cargando {indice_archivo}/{total_archivos}: {archivo.name}"
            )

            try:
                historial_paquete = _deserializar_archivo_subido(archivo)
                cantidad_antes = len(historial_destino)
                historial_destino = _agregar_historial_sin_sobrescribir(
                    historial_destino,
                    historial_paquete,
                )
                estrategias_nuevas += len(historial_destino) - cantidad_antes
                archivos_ok += 1

                # Se conserva solo como registro informativo; la carga depende
                # del botón, no de reruns automáticos.
                cargados_previos = set(
                    st.session_state.get("_archivos_resultados_cargados", [])
                )
                cargados_previos.add(identificador)
                st.session_state["_archivos_resultados_cargados"] = list(
                    cargados_previos
                )
            except Exception as exc:
                errores_carga.append(f"{archivo.name}: {exc}")
            finally:
                barra_carga.progress(indice_archivo / total_archivos)

        if archivos_ok:
            st.session_state["historial"] = historial_destino
            st.session_state.pop("resultados_gz_bytes", None)
            st.session_state.pop("resultados_gz_nombre", None)
            st.session_state.pop("resultados_gz_descripcion", None)
            estado_carga.success(
                f"Carga terminada: {archivos_ok} archivo(s), "
                f"{estrategias_nuevas} estrategia(s) incorporada(s)."
            )

        if errores_carga:
            st.sidebar.error("No se pudieron cargar algunos archivos:")
            for detalle_error in errores_carga:
                st.sidebar.caption(f"• {detalle_error}")

        # No se usa st.rerun(): la pestaña Resultados está después de este
        # bloque y leerá el historial actualizado en esta misma ejecución.


# ═════════════════════════════════════════════════════════════
# PESTAÑA 3 — RESULTADOS
# ═════════════════════════════════════════════════════════════

with tab3:
    st.title("📊 Resultados y Análisis de Resiliencia")

    historial = st.session_state.get("historial", {})

    # Determinar resultados activos para comparación
    results_activos  = None
    saidi_activos    = SAIDI_sin
    label_activos    = "Caso base"
    config_activo = {
        "seccionadores_existentes": sorted(_AISL_SEC) if _AISL_ON else [],
    }
    if historial:
        ultimo_label    = list(historial.keys())[-1]
        results_activos = historial[ultimo_label]["results"]
        saidi_activos   = historial[ultimo_label]["saidi"]
        label_activos   = ultimo_label
        config_activo   = historial[ultimo_label].get("config", {})

    # Helper vista para tab3
    def _get_esc_idx_t3():
        valid = [v for v in _results_to_list(results_sin) if v]
        if vista_global == "⚠️ Peor caso":
            return min(_esc_peor_caso(SAIDI_sin), len(valid) - 1)
        elif vista_global == "📌 Escenario seleccionado":
            return _pos_esc(esc_sel - 1, len(valid)) if numesc > 1 else 0
        else:
            return None  # promedio

    def _get_serie_t3(results_dict, fn, saidi_ref=None):
        valid = [v for v in _results_to_list(results_dict) if v]
        if not valid:
            return [0.0] * N_HORAS
        saidi_ref = saidi_ref or SAIDI_sin
        if vista_global == "📊 Promedio":
            return _promedio_por_hora([fn(v) for v in valid])
        elif vista_global == "⚠️ Peor caso":
            idx = min(_esc_peor_caso(saidi_ref), len(valid) - 1)
            return fn(valid[idx])
        else:
            idx = _pos_esc(esc_sel - 1, len(valid))
            return fn(valid[idx])

    # ── Sub-sección A: Indicadores de confiabilidad ──────────
    st.header("🔢 A — Indicadores de Confiabilidad")

    gran_kpi = st.radio(
        "Granularidad KPIs",
        ["Red completa", "Por zona", "Por barra"],
        horizontal=True, key="gran_kpi"
    )
    # Estos KPIs tenían su propio slider ("esc_t3_kpi"), independiente del que
    # gobernaba el grafo y las series de más abajo: la misma pantalla podía
    # mostrar dos escenarios distintos. Ahora todo lee el selector de la
    # sidebar.
    if numesc > 1:
        st.caption(
            f"🎲 Escenario **{esc_sel}** de {numesc} (se cambia en la barra lateral)."
        )

    valid_sin   = [v for v in _results_to_list(results_sin) if v]
    esc_idx_kpi = _pos_esc(esc_sel - 1, len(valid_sin)) if valid_sin else 0
    res_kpi     = valid_sin[esc_idx_kpi] if valid_sin else []

    # SAIDI
    saidi_kpi = fn_SAIDI(res_kpi, P_D_base, BUSES, horas) if res_kpi else 0.0
    saifi_kpi = fn_SAIFI(res_kpi, P_D_base, BUSES, horas) if res_kpi else 0.0

    # ENS total
    ens_kpi = sum(
        sum(res_kpi[h-1].get("P_LC", {}).get(b, 0) for b in BUSES)
        for h in horas if h - 1 < len(res_kpi)
    ) if res_kpi else 0.0

    # % Energía distribuida
    # Energía total demandada con demand_factor
    # Índice sobre el array crudo `demand_factor`, no sobre los resultados:
    # acá el `min` sí corresponde (es un clamp de rango del array).
    _esc_idx_kpi = min(esc_sel - 1, demand_factor.shape[1] - 1)
    pd_total = sum(
        P_D_base.get(b, 0) * demand_factor[h-1, _esc_idx_kpi, bi]
        for h in horas for bi, b in enumerate(BUSES)
    )
    pct_energia = (1 - ens_kpi / pd_total) * 100 if pd_total > 0 else 0.0

    # Tasa de interconexión promedio
    tasa_intercon = np.mean(_tasa_interconexion_por_hora(res_kpi)) if res_kpi else 100.0

    if gran_kpi == "Red completa":
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("SAIDI [h/año]",            f"{saidi_kpi:.2f}")
        col2.metric("SAIFI [int./año]",          f"{saifi_kpi:.2f}")
        col3.metric("ENS total [kWh]",               f"{ens_kpi:.2f}")
        col4.metric("% Energía distribuida",         f"{pct_energia:.1f}%")
        col5.metric("Tasa interconexión [%]",        f"{tasa_intercon:.1f}%")

    elif gran_kpi == "Por zona":
        zonas_list = [1, 2, 3]
        cols = st.columns(3)
        for idx_z, z in enumerate(zonas_list):
            buses_z = [b for b in BUSES if bus_zone.get(b, 0) == z]
            saidi_z = fn_SAIDI(res_kpi, P_D_base, buses_z, horas) if res_kpi else 0.0
            saifi_z = fn_SAIFI(res_kpi, P_D_base, buses_z, horas) if res_kpi else 0.0
            ens_z   = sum(
                sum(res_kpi[h-1].get("P_LC", {}).get(b, 0) for b in buses_z) if h - 1 < len(res_kpi) else 0.0
                for h in horas
            ) if res_kpi else 0.0
            cols[idx_z].markdown(f"**Zona {z}**")
            cols[idx_z].metric(f"SAIDI Z{z}",  f"{saidi_z:.2f}")
            cols[idx_z].metric(f"SAIFI Z{z}",  f"{saifi_z:.2f}")
            cols[idx_z].metric(f"ENS Z{z} [kWh]", f"{ens_z:.2f}")

    else:  # Por barra
        rows_kpi = []
        for b in BUSES:
            ens_b  = sum(res_kpi[h-1].get("P_LC", {}).get(b, 0) for h in horas if h - 1 < len(res_kpi)) if res_kpi else 0.0
            saidi_b = fn_SAIDI(res_kpi, P_D_base, [b], horas) if res_kpi else 0.0
            saifi_b = fn_SAIFI(res_kpi, P_D_base, [b], horas) if res_kpi else 0.0
            rows_kpi.append({"Bus": b, "Zona": bus_zone.get(b, "—"),
                              "SAIDI": round(saidi_b, 3), "SAIFI": round(saifi_b, 3), "ENS [kWh]": round(ens_b, 3)})
        st.dataframe(pd.DataFrame(rows_kpi).set_index("Bus"), use_container_width=True)

    # ── SAIDI/SAIFI mensual acumulado ─────────────────────────
    if N_HORAS >= 720:
        st.subheader("📅 SAIDI y SAIFI Acumulados Mensualmente")

        def _saidi_mensual(results_esc):
            vals, h_start = [], 0
            for mh in MESES_H:
                h_end  = min(h_start + mh, N_HORAS)
                subset = {i: results_esc[h_start + i] for i in range(h_end - h_start)
                          if h_start + i < len(results_esc)}
                vals.append(fn_SAIDI(subset, P_D_base, BUSES, range(1, len(subset)+1)))
                h_start = h_end
                if h_start >= N_HORAS:
                    break
            return vals

        saidi_base_mensual  = _saidi_mensual(res_kpi)
        meses_labels        = MESES_NOM[:len(saidi_base_mensual)]

        fig_mens = go.Figure()
        fig_mens.add_trace(go.Scatter(
            x=meses_labels, y=saidi_base_mensual,
            mode="lines+markers", name="Caso base",
            line=dict(color="#d62728", width=2)
        ))
        if results_activos:
            valid_act = [v for v in _results_to_list(results_activos) if v]
            if valid_act:
                idx_a = _pos_esc(esc_sel - 1, len(valid_act))
                saidi_act_mensual = _saidi_mensual(valid_act[idx_a])
                fig_mens.add_trace(go.Scatter(
                    x=meses_labels, y=saidi_act_mensual[:len(meses_labels)],
                    mode="lines+markers", name=label_activos,
                    line=dict(color="#2ca02c", width=2)
                ))
        fig_mens.update_layout(
            title="SAIDI mensual acumulado [h/año]",
            xaxis_title="Mes", yaxis_title="SAIDI [h/año]", height=380
        )
        st.plotly_chart(_fig_pantalla(fig_mens, height=380),
                        use_container_width=True, key="saidi_mensual")
        _descargas_figura(fig_mens, "saidi_mensual_acumulado",
                          key="saidi_mens",
                          etiqueta="el gráfico de SAIDI mensual")


    st.header("🕸️ Grafo de la Red")

    col_ctrl, col_info = st.columns([3, 1])
    with col_ctrl:
        # 1-based, igual que el eje X del gráfico de ENS (ver la nota del
        # slider de la pestaña de entradas).
        hora_ui = st.slider("Hora", 1, N_HORAS, 1, key="slider_grafo_h1")
    with col_info:
        st.caption(f"Hora seleccionada: **{hora_ui}**")

    hora_grafo = hora_ui - 1        # índice 0-based en listas y arrays

    # Estado de fallas
    fmc_hora = {}
    try:
        eidx = min(max(esc_sel - 1, 0), faultlinMC.shape[1] - 1)
        hidx = min(hora_grafo, faultlinMC.shape[0] - 1)
        for li, l in enumerate(LINES):
            if li < faultlinMC.shape[2]:
                fmc_hora[l] = bool(faultlinMC[hidx, eidx, li] > 0)
    except Exception:
        pass

    if results_activos:
        _valid = [v for v in _results_to_list(results_activos) if v]
        _idx   = _pos_esc(esc_sel - 1, len(_valid)) if _valid else 0
        _lista = _valid[_idx] if _valid else []
        res_hora = _lista[hora_grafo] if hora_grafo < len(_lista) else {}
    else:
        res_hora = {}

    fig_grafo = build_fig_grafo(hora_ui, esc_sel, res_hora, fmc_hora, config_activo, buses_criticos)
    # Tamaño de EXPORTACIÓN (el que se descarga). A la pantalla va la copia
    # reescalada de más abajo: con 1200 px de alto y fuente 20 el grafo salía
    # larguísimo y con las etiquetas encima unas de otras.
    fig_grafo.update_layout(
        width=1800,
        height=1200,
        font=dict(size=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(_fig_pantalla(fig_grafo, height=680, font=12),
                    use_container_width=True, key="grafo_red_hora_t3")
    _descargas_figura(fig_grafo, f"grafo_red_esc{esc_sel}_h{hora_ui}",
                      key="grafo_hora_t3", etiqueta="este grafo")

    # ── Grafo RESUMEN de toda la simulación (descargable) ─────
    # El grafo de arriba muestra una hora; este agrega el horizonte completo
    # del escenario: qué líneas fallaron (y cuántas horas), dónde se instaló
    # GD/ESS/MEG/MESS y qué líneas recibieron medidas fijas.
    st.subheader("🧾 Resumen de la simulación — grafo descargable")

    # El agregado no es un escenario: funde varios en un mapa de exposición.
    # Va acá y no como una figura aparte porque es la misma red, la misma
    # leyenda y el mismo CSV — lo único que cambia es sobre qué se agregó.
    #
    # POR DEFECTO agrega los escenarios que resolvió la TANDA DEL BARRIDO
    # que se está estudiando, no los 100 del Monte Carlo: el mapa acompaña a
    # esos resultados, y agregarlo sobre otra población lo vuelve un dibujo
    # de otra cosa. La tanda es la del selector de la pestaña 🧪 Barrido, así
    # que las dos vistas hablan siempre del mismo conjunto.
    _TANDA_BARR = st.session_state.get("barr_nombre")

    def _sello_barrido(tanda):
        """mtime del resumen de la tanda: invalida la caché si se recorre el
        barrido y aparecen escenarios nuevos. Mismo criterio que
        `barrido_vista.sello`, replicado acá para no importar la vista
        entera solo por esto."""
        if not tanda:
            return 0.0
        import pipeline as _Pb
        _r = _Pb.ART_DIR / tanda / "resumen_individual.csv"
        return _r.stat().st_mtime if _r.exists() else 0.0

    _ESC_TANDA = (escenarios_barrido(_TANDA_BARR, _sello_barrido(_TANDA_BARR))
                  if _TANDA_BARR else [])
    _ESC_MUESTRA, _META_MUESTRA = escenarios_muestra()

    _SETS_ESC = {}
    if _ESC_TANDA:
        _SETS_ESC[f"Barrido en estudio · {_TANDA_BARR} "
                  f"({len(_ESC_TANDA)} esc)"] = _ESC_TANDA
    if _ESC_MUESTRA and _ESC_MUESTRA != _ESC_TANDA:
        _SETS_ESC[f"Muestra estratificada propuesta "
                  f"({len(_ESC_MUESTRA)} esc)"] = _ESC_MUESTRA
    if ESCENARIOS_CON_BASELINE:
        _SETS_ESC[f"Con caso base en `{VARIANTE_AISL}` "
                  f"({len(ESCENARIOS_CON_BASELINE)} esc)"] = sorted(ESCENARIOS_CON_BASELINE)
    _SETS_ESC[f"Todos ({numesc} esc)"] = list(range(numesc))
    _OPC_TODAS = "Fallas agregadas"
    _opciones_resumen = ["Caso base"] + (["Estrategia activa"] if historial else []) \
                        + [_OPC_TODAS]
    _caso_resumen = st.radio(
        "Caso del grafo resumen", _opciones_resumen,
        horizontal=True, key="caso_grafo_resumen",
        help="«Caso base» y «Estrategia activa» son UN escenario (un estado "
             "realizable de la red). «Fallas agregadas» funde varios "
             "escenarios: es el valor esperado del proceso de fallas, no una "
             "situación que ocurra tal cual.",
    )
    _res_agregado = _caso_resumen == _OPC_TODAS

    _esc_agg = []
    if _res_agregado:
        _nom_set = st.selectbox(
            "Escenarios a agregar", list(_SETS_ESC), index=0,
            key="set_esc_grafo_resumen",
            help="Por defecto, los escenarios que resolvió la tanda del "
                 "barrido en estudio: es el único conjunto que hace este mapa "
                 "comparable con los resultados por medida, porque son las "
                 "mismas fallas que vieron esas corridas.",
        )
        _esc_agg = _SETS_ESC[_nom_set]
        if not _ESC_TANDA:
            st.caption(
                "⚠️ No hay tanda de barrido seleccionada (pestaña 🧪 Barrido), "
                "así que no se puede ofrecer su conjunto de escenarios: el que "
                "elijas acá puede no coincidir con el de los resultados por medida."
            )
    _res_es_base = _caso_resumen == "Caso base"
    _res_datos = results_sin if _res_es_base else results_activos
    _res_label = "Caso base" if _res_es_base else label_activos
    # El caso base NO hereda los seccionadores del cfg de la estrategia: son un
    # supuesto de red, y solo valen si la aislación está encendida (con el
    # toggle apagado el modelo supone un seccionador en cada línea, que es lo
    # mismo que no dibujar ninguno).
    # El agregado es del proceso de fallas, no de una estrategia: lleva el
    # mismo supuesto de red que el caso base.
    _res_cfg = ({"seccionadores_existentes": sorted(_AISL_SEC) if _AISL_ON else []}
                if (_res_es_base or _res_agregado) else config_activo)
    if _res_agregado:
        # Comparable = mismos escenarios que corrió el barrido. Cualquier otro
        # conjunto sigue siendo un mapa válido de la red, pero ya no explica
        # los resultados por medida que lo acompañan.
        _es_tanda = bool(_ESC_TANDA) and _esc_agg == _ESC_TANDA
        st.caption(
            f"Mapa de exposición sobre {len(_esc_agg)} escenarios: "
            f"{', '.join(str(e + 1) for e in _esc_agg)}. El grosor de cada "
            f"línea es su frecuencia de falla y el hover dice en cuántos "
            f"escenarios falló. **No es un estado realizable de la red** (una "
            f"línea puede figurar con horas de falla y de aislación porque "
            f"pasaron en escenarios distintos); para narrar la operación usa "
            f"un escenario concreto."
            + ("" if _AISL_ON else
               "  \n⚪ Con la aislación apagada solo se dibuja la línea "
               "averiada. Enciéndela para ver también el arrastre.")
        )
        if _es_tanda:
            _txt_sesgo = ""
            if _ESC_TANDA == _ESC_MUESTRA and _META_MUESTRA.get("sesgo_pct") is not None:
                _txt_sesgo = (
                    f" Coincide con la muestra estratificada por "
                    f"{_META_MUESTRA.get('criterio', 'ens')} en "
                    f"{_META_MUESTRA.get('n_estratos', '?')} estratos "
                    f"(sesgo de la media: {_META_MUESTRA['sesgo_pct']:.2f} %).")
            st.caption(
                f"✅ Son los escenarios de `{_TANDA_BARR}`, los mismos que "
                f"resolvieron las medidas de esa tanda: este mapa y los "
                f"resultados por medida hablan de las mismas fallas.{_txt_sesgo}"
            )
        elif _ESC_TANDA:
            st.warning(
                f"Este conjunto **no** es el de la tanda en estudio "
                f"(`{_TANDA_BARR}`, {len(_ESC_TANDA)} escenarios): el mapa "
                f"deja de ser comparable con sus resultados por medida, "
                f"porque no son las mismas fallas. Úsalo para caracterizar la "
                f"red, no para contrastar contra una medida."
            )
    elif _res_es_base:
        # Sin marcadores azules el grafo se ve idéntico al de siempre, así que
        # el aviso tiene que distinguir las dos causas posibles: el supuesto
        # apagado y el baseline de esa variante que no está calculado.
        _sec_texto = ", ".join(f"L{l}" for l in _res_cfg["seccionadores_existentes"])
        if not _AISL_ON:
            st.caption(
                "⚪ Aislación apagada: el caso base supone un seccionador en cada "
                "línea (aislación perfecta), así que no hay seccionadores que "
                "marcar. Enciende «🔒 Aislación de la falla» en la pestaña "
                "Optimización y elige las líneas con seccionador."
            )
        elif not _sec_texto:
            st.caption(
                "⚠️ Aislación encendida pero sin líneas seleccionadas: al fallar "
                "una línea cae el alimentador completo. Indica las líneas en "
                "Aislación de la falla → Líneas con seccionador preexistente."
            )
        else:
            st.caption(f"Seccionadores preexistentes: {_sec_texto} "
                       f"(variante `{VARIANTE_AISL}`).")
            if not results_sin:
                st.warning(
                    f"El escenario {esc_sel} no tiene caso base calculado para "
                    f"`{VARIANTE_AISL}`, así que el grafo muestra la red sin "
                    f"fallas. Escenarios disponibles en esta variante (numeración "
                    f"de la app): "
                    f"{', '.join(str(e + 1) for e in sorted(ESCENARIOS_CON_BASELINE))}."
                )

    if _res_datos and not _res_agregado:
        _valid_r = [v for v in _results_to_list(_res_datos) if v]
        _idx_r   = _pos_esc(esc_sel - 1, len(_valid_r)) if _valid_r else 0
        _lista_r = _valid_r[_idx_r] if _valid_r else []
    else:
        _lista_r = []

    try:
        _fmc_esc = faultlinMC[:, min(max(esc_sel - 1, 0), faultlinMC.shape[1] - 1), :]
    except Exception:
        _fmc_esc = None

    if not _lista_r and not _res_agregado:
        st.info(
            "Aún no hay resultados de optimización cargados para este escenario. "
            "El resumen se arma con las fallas del Monte Carlo, pero sin ubicación "
            "de medidas: ejecuta una estrategia en la pestaña de optimización."
        )

    # Agregar 8760 h toma ~1.5 s: se cachea en sesión y solo se recalcula si
    # cambia la estrategia activa, el escenario o el largo del horizonte.
    # RESUMEN_SCHEMA en la clave: sin él, un resumen cacheado por una versión
    # anterior del código se reusa aunque le falten llaves nuevas.
    _sec_cfg = list(_res_cfg.get("seccionadores_existentes", []) or [])
    _clave_res = (_res_es_base, _res_agregado, str(_res_label), int(esc_sel),
                  len(_lista_r), RESUMEN_SCHEMA, tuple(sorted(map(str, _sec_cfg))),
                  tuple(_esc_agg))
    if st.session_state.get("_resumen_sim_key") != _clave_res:
        if _res_agregado:
            # Se arma directo de faultlinMC (~0.1 s): no necesita baselines
            # resueltos, así que el conjunto de escenarios lo decide la tanda
            # en estudio y no lo que haya quedado en disco.
            with st.spinner(f"Agregando las fallas de {len(_esc_agg)} escenarios…"):
                st.session_state["_resumen_sim"] = resumir_fallas_mc(
                    esc_lista=_esc_agg, seccionadores=_sec_cfg)
        else:
            with st.spinner("Agregando todas las horas del escenario…"):
                st.session_state["_resumen_sim"] = resumir_simulacion(
                    _lista_r, faultlin_esc=_fmc_esc, seccionadores=_sec_cfg)
        st.session_state["_resumen_sim_key"] = _clave_res
    resumen_sim = st.session_state["_resumen_sim"]

    _tot = resumen_sim["totales"]
    _n_esc_res = int(_tot.get("n_esc", 0) or 0)
    if _res_agregado:
        # En el agregado los conteos de medidas son cero (no hay optimización
        # detrás), así que las métricas se cambian por las que sí tienen
        # sentido: exposición por línea y horas de arrastre.
        _h_arrastre = sum(d.get("horas_aislada", 0) for d in resumen_sim["lineas"].values())
        _h_secc = sum(d.get("horas_seccionada", 0) for d in resumen_sim["lineas"].values())
        cr1, cr2, cr3, cr4, cr5 = st.columns(5)
        cr1.metric("Escenarios agregados", f"{_n_esc_res}")
        cr2.metric("Líneas que fallan", f"{_tot['lineas_con_falla']} de "
                                        f"{len(resumen_sim['lineas'])}")
        cr3.metric("Horas-línea en falla", f"{_tot['horas_falla_totales']:,}")
        cr4.metric("Horas de seccionamiento", f"{_h_secc:,}",
                   help="Horas-línea que se abrieron por maniobra para aislar.")
        cr5.metric("Horas de arrastre", f"{_h_arrastre:,}",
                   help="Horas-línea sanas que quedaron sin servicio por no "
                        "tener seccionador con qué cortar. Es la parte del "
                        "problema que ataca invertir en seccionadores.")
    else:
        cr1, cr2, cr3, cr4, cr5 = st.columns(5)
        cr1.metric("Líneas con falla", f"{_tot['lineas_con_falla']}")
        cr2.metric("Horas-línea en falla", f"{_tot['horas_falla_totales']}")
        cr3.metric("Buses con GD", f"{_tot['buses_con_dg']}")
        cr4.metric("Buses con ESS", f"{_tot['buses_con_ess']}")
        cr5.metric("Líneas con medida fija", f"{_tot['lineas_con_medida']}")

    if _res_agregado:
        _titulo_res = f"Mapa de exposición — {_n_esc_res} escenarios Monte Carlo"
        _sub_res = (
            f"{_tot['n_horas']:,} h-escenario · "
            f"{_tot['lineas_con_falla']} líneas con falla · "
            f"{_tot['horas_falla_totales']:,} horas-línea en falla · "
            f"supuesto de aislación `{VARIANTE_AISL}` · "
            f"valor esperado, no un escenario realizable"
        )
    else:
        _titulo_res = f"Resumen de la simulación — {_res_label}"
        _sub_res = (
            f"Escenario {esc_sel} · {_tot['n_horas']} h simuladas · "
            f"{_tot['lineas_con_falla']} líneas con falla · "
            f"{_tot['buses_con_dg']} buses con GD · "
            f"{_tot['buses_con_ess']} buses con ESS · "
            f"{_tot['lineas_con_medida']} líneas con medida fija · "
            f"ENS {_tot['ens_total_kwh']:,.0f} kWh"
        )

    fig_resumen = build_fig_grafo_resumen(
        resumen_sim, _res_cfg, buses_criticos,
        titulo=_titulo_res, subtitulo=_sub_res,
        mostrar_seccionadores=_res_es_base or _res_agregado,
    )
    st.plotly_chart(fig_resumen, use_container_width=True, key="grafo_resumen_simulacion")

    df_lin_res, df_bus_res = tablas_resumen(resumen_sim)

    # Renderizar el PNG (~3 s) y el HTML (~2 s) en cada rerun haría lentísima
    # la pestaña, así que se generan solo cuando se piden.
    _prep = st.checkbox(
        "🖼️ Preparar la imagen y el HTML del grafo resumen para descargar "
        "(tarda unos segundos)",
        key="chk_preparar_descarga_resumen",
    )

    # El nombre del archivo tiene que decir sobre qué se agregó: un
    # "grafo_resumen_esc7.png" que en realidad son los 100 escenarios es
    # justo el tipo de figura que después nadie sabe reproducir.
    _sufijo_res = (f"agregado_{_n_esc_res}esc_{VARIANTE_AISL}" if _res_agregado
                   else f"esc{esc_sel}")

    col_r1, col_r2, col_r3 = st.columns(3)

    if _prep:
        with st.spinner("Generando archivos del grafo resumen…"):
            try:
                # Solo `scale`: pasar width/height distintos al layout deja los
                # marcadores (px) fuera de sitio respecto de la geometría
                # (unidades de dato) y el grafo sale desarmado.
                png_resumen = fig_resumen.to_image(format="png", scale=3)
            except Exception:
                col_r1.caption(
                    "📷 PNG no disponible: falta el paquete `kaleido` en este entorno "
                    "(`pip install --upgrade kaleido`). Usa el HTML interactivo."
                )
            else:
                col_r1.download_button(
                    "📷 Descargar grafo resumen (PNG)",
                    data=png_resumen,
                    file_name=f"grafo_resumen_{_sufijo_res}.png",
                    mime="image/png",
                    key="download_grafo_resumen_png",
                    help="Imagen de alta resolución, lista para la tesis o la presentación.",
                )

            col_r2.download_button(
                "🌐 Descargar grafo resumen (HTML interactivo)",
                data=fig_resumen.to_html(include_plotlyjs="cdn"),
                file_name=f"grafo_resumen_{_sufijo_res}.html",
                mime="text/html",
                key="download_grafo_resumen_html",
                help="Conserva el hover con el detalle de cada línea y cada bus.",
            )
    else:
        col_r1.caption("Marca la casilla para habilitar PNG y HTML del grafo.")

    _buf_res = io.BytesIO()
    with pd.ExcelWriter(_buf_res, engine="openpyxl") as _wr_res:
        df_lin_res.to_excel(_wr_res, sheet_name="Lineas", index=False)
        df_bus_res.to_excel(_wr_res, sheet_name="Buses", index=False)
        pd.DataFrame([_tot]).to_excel(_wr_res, sheet_name="Totales", index=False)
    _buf_res.seek(0)
    col_r3.download_button(
        "📥 Descargar resumen en tablas (Excel)",
        data=_buf_res,
        file_name=f"resumen_simulacion_{_sufijo_res}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_resumen_simulacion_xlsx",
        help="Mismos datos del grafo: fallas por línea, medidas fijas y capacidades por bus.",
    )

    with st.expander("📋 Ver tablas del resumen"):
        st.markdown("**Líneas — fallas y medidas fijas**")
        st.dataframe(df_lin_res, use_container_width=True, hide_index=True)
        st.markdown("**Buses — capacidades instaladas y carga no servida**")
        st.dataframe(df_bus_res, use_container_width=True, hide_index=True)


    # ── Sub-sección C: Capacidades instaladas ────────────────
    st.header("🏗️ C — Capacidades Instaladas de Medidas")

    if results_activos:
        cap_dg = _extraer_cap_dg(results_activos)
        rows_dg = [{
            "Bus":              cap_dg[g]["bus"],
            "Zona":             cap_dg[g]["zona"],
            "CAP_DG_sol [kW]": round(cap_dg[g]["sol"][0], 2),
            "CAP_DG_eol [kW]": round(cap_dg[g]["eol"][0], 2),
            "CAP_DG_fos [kW]": round(cap_dg[g]["fos"][0], 2),
        } for g in GENS]
        st.subheader("Generadores Distribuidos")
        st.dataframe(pd.DataFrame(rows_dg), use_container_width=True)

        # MEG
        cap_meg = _extraer_cap_meg(results_activos)
        if cap_meg:
            st.subheader("MEG — Capacidades y buses asignados")
            valid_act2 = [v for v in _results_to_list(results_activos) if v]
            rows_meg = []
            for k, cap_k in cap_meg.items():
                x_re_avg = {}
                for res_lista in valid_act2:
                    if res_lista:
                        for (kk, bb), val in res_lista[0].get("x_re", {}).items():
                            if kk == k:
                                x_re_avg[bb] = x_re_avg.get(bb, 0) + val
                n_sc = len(valid_act2) or 1
                bus_asig = max(x_re_avg, key=x_re_avg.get) if x_re_avg else "—"
                horas_act = sum(
                    1 for res_lista in valid_act2 if res_lista
                    for hd in res_lista
                    if sum(hd.get("P_MEG", {}).get((k, bb), 0) for bb in BUSES) > 0.01
                )
                rows_meg.append({"Unidad k": k, "Cap. [kW]": round(cap_k, 1),
                                  "Bus asignado": bus_asig, "Horas activas": horas_act})
            st.dataframe(pd.DataFrame(rows_meg).set_index("Unidad k"), use_container_width=True)

        # MESS
        cap_mess = _extraer_cap_mess(results_activos)

        if cap_mess:
            st.subheader("MESS — Capacidades, ubicación y operación")

            # Cada elemento de valid_act2 corresponde a un escenario
            # y contiene una lista de resultados horarios.
            valid_act2 = [
                escenario
                for escenario in _results_to_list(results_activos)
                if escenario
            ]

            # Obtener potencia máxima guardada en los resultados.
            pmax_mess = {}

            for escenario in valid_act2:
                if escenario:
                    # "P_MESS_MW" es la clave vigente; "P_MESS_MAX_MW" la
                    # antigua (respaldo para resultados ya guardados).
                    pmax_mess = (escenario[0].get("P_MESS_MW")
                                 or escenario[0].get("P_MESS_MAX_MW", {}))
                    if pmax_mess:
                        break

            rows_mess = []

            for k, cap_k in cap_mess.items():

                # ----------------------------------------------------
                # Contadores de ubicación del MESS
                # ----------------------------------------------------
                ocupacion_bus = {b: 0 for b in BUSES}

                horas_operando_total = 0
                horas_cargando_total = 0
                horas_descargando_total = 0

                for escenario in valid_act2:

                    for hd in escenario:

                        # u_mess[(k, b)] = 1 cuando el MESS k está en el bus b
                        ubicacion_hora = hd.get("u_mess", {})

                        for b in BUSES:
                            if ubicacion_hora.get((k, b), 0) > 0.5:
                                ocupacion_bus[b] += 1

                        # P_MESS puede representar potencia neta:
                        # positiva = descarga
                        # negativa = carga
                        potencia_hora = sum(
                            hd.get("P_mess", {}).get((k, b), 0.0)
                            for b in BUSES
                        )

                        if abs(potencia_hora) > 0.01:
                            horas_operando_total += 1

                        if potencia_hora > 0.01:
                            horas_descargando_total += 1

                        elif potencia_hora < -0.01:
                            horas_cargando_total += 1

                # ----------------------------------------------------
                # Bus en el que permaneció más horas
                # ----------------------------------------------------
                buses_ocupados = {
                    b: n_horas
                    for b, n_horas in ocupacion_bus.items()
                    if n_horas > 0
                }

                if buses_ocupados:
                    bus_principal = max(
                        buses_ocupados,
                        key=buses_ocupados.get
                    )

                    buses_visitados = ", ".join(
                        str(b)
                        for b in sorted(
                            buses_ocupados,
                            key=buses_ocupados.get,
                            reverse=True
                        )
                    )
                else:
                    bus_principal = "—"
                    buses_visitados = "—"

                # Promedio por escenario para evitar que el indicador
                # dependa directamente de la cantidad de escenarios.
                n_escenarios = len(valid_act2) or 1

                horas_operando_prom = horas_operando_total / n_escenarios
                horas_cargando_prom = horas_cargando_total / n_escenarios
                horas_descargando_prom = horas_descargando_total / n_escenarios

                # La potencia máxima normalmente es un diccionario por unidad.
                if isinstance(pmax_mess, dict):
                    pmax_k = pmax_mess.get(k, 0.0)
                else:
                    pmax_k = pmax_mess

                rows_mess.append({
                    "Unidad k": k,
                    "Capacidad [MWh]": round(float(cap_k), 3),
                    "P máx. [MW]": round(float(pmax_k), 3),
                    "Bus principal": bus_principal,
                    "Buses visitados": buses_visitados,
                    "Horas operando": round(horas_operando_prom, 1),
                    "Horas cargando": round(horas_cargando_prom, 1),
                    "Horas descargando": round(horas_descargando_prom, 1),
                })

            df_mess = pd.DataFrame(rows_mess).set_index("Unidad k")

            st.dataframe(
                df_mess,
                use_container_width=True
            )
        # # ESS
        # st.subheader("ESS — Capacidad y ciclos")
        # valid_act3 = [v for v in _results_to_list(results_activos) if v]
        # rows_ess = []
        #     })
        # st.dataframe(pd.DataFrame(rows_ess).set_index("ESS"), use_container_width=True)

        # ESS
        st.subheader("ESS — Capacidad y ciclos")
        valid_act3 = [v for v in _results_to_list(results_activos) if v]
        durations = ["5h", "4h", "6h"]
        rows_ess = []

        for ess in ESS:
            row = {"ESS": ess}
            for dur in durations:
                caps_kwh, ciclos = [], 0
                for res_lista in valid_act3:
                    if res_lista:
                        cap_e = (res_lista[0].get(f"CAP_ESS_kWh_{dur}", {}).get(ess, 0))/int(dur[0])
                        caps_kwh.append(cap_e)
                        ciclos += sum(
                            1 for hd in res_lista
                            if hd.get(f"P_ESS_{dur}", {}).get(ess, 0) > 0.01
                        )
                row[f"Cap. {dur} [kW]"] = round(np.mean(caps_kwh) if caps_kwh else 0, 2)
                row[f"Ciclos {dur}"]     = ciclos
            rows_ess.append(row)

        st.dataframe(pd.DataFrame(rows_ess).set_index("ESS"), use_container_width=True)



        config_activo = historial.get(list(historial.keys())[-1], {}).get("config", {})
        if config_activo.get("reconfig_topo", False) or config_activo.get("inv_tie", False):
            st.subheader(
                "🔗 Inversión en líneas de enlace — Tie-lines"
                if config_activo.get("inv_tie", False)
                else "🔀 Reconfiguración Topológica — Tie-lines"
            )
            tie_indices_list = list(set(LINES[-3:]))  # igual que en optimizer
            valid_tie = [v for v in _results_to_list(results_activos) if v]
            n_esc = max(len(valid_tie), 1)
            rows_tie = []
            for l in tie_indices_list:
                horas_cerrada = sum(
                    1 for res_lista in valid_tie if res_lista
                    for hd in res_lista
                    if hd.get("z_tie", {}).get(l, 0) > 0.5
                )
                operaciones_cerrada = sum(
                    1 for res_lista in valid_tie if res_lista
                    for hd in res_lista
                    if hd.get("w_tie", {}).get(l, 0) > 0.5
                )
                construida = max(
                    (hd.get("y_tie", {}).get(l, 0) for res_lista in valid_tie if res_lista for hd in res_lista),
                    default=0,
                )
                rows_tie.append({
                    "Tie-line": l,
                    "Nombre": net.line.at[l, "name"] if l in net.line.index else f"L{l}",
                    "Bus origen": net.line.at[l, "from_bus"] if l in net.line.index else "—",
                    "Bus destino": net.line.at[l, "to_bus"] if l in net.line.index else "—",
                    **({"Enlace construido": "✅" if construida > 0.5 else "—",
                        "Largo [km]": round(data.get("longitud_km_linea", {}).get(l, 0), 2)}
                       if config_activo.get("inv_tie", False) else {}),
                    "N° operaciones": round(operaciones_cerrada / n_esc, 1),
                    "Horas cerrada": round(horas_cerrada / n_esc, 1),
                    "% tiempo activa": round(horas_cerrada / n_esc / N_HORAS * 100, 1),
                })
            st.dataframe(pd.DataFrame(rows_tie).set_index("Tie-line"), use_container_width=True)


        config_activo = historial[ultimo_label]["config"] if historial else {}

        if config_activo.get("veg_mgmt") or config_activo.get("refuerzo") or config_activo.get("soterramiento"):
            st.subheader("🌳🔩🕳️ Líneas tratadas — Vegetación / Refuerzo / Soterramiento")

            sw_cands_tratadas = data.get("sw_cand_list", [])
            valid_tratadas = [v for v in _results_to_list(results_activos) if v]
            longitud_km_linea = data.get("longitud_km_linea", {})

            def _prom_decision(clave, l):
                vals = [res_lista[0].get(clave, {}).get(l, 0) for res_lista in valid_tratadas if res_lista]
                return float(np.mean(vals)) if vals else 0.0

            rows_tratadas = []
            for l in sw_cands_tratadas:
                p_veg = _prom_decision("x_veg", l) if config_activo.get("veg_mgmt") else 0.0
                p_ref = _prom_decision("x_ref", l) if config_activo.get("refuerzo") else 0.0
                p_ug  = _prom_decision("x_ug",  l) if config_activo.get("soterramiento") else 0.0
                if p_veg < 0.5 and p_ref < 0.5 and p_ug < 0.5:
                    continue
                fb = int(net.line.at[l, "from_bus"]) if l in net.line.index else "—"
                tb = int(net.line.at[l, "to_bus"])   if l in net.line.index else "—"
                rows_tratadas.append({
                    "Línea":          l,
                    "Nombre":         net.line.at[l, "name"] if l in net.line.index else f"L{l}",
                    "Bus origen":     fb,
                    "Bus destino":    tb,
                    "Longitud [km]":  round(longitud_km_linea.get(l, 0), 3),
                    "Vegetación":     "✅" if p_veg >= 0.5 else "—",
                    "Refuerzo":       "✅" if p_ref >= 0.5 else "—",
                    "Soterramiento":  "✅" if p_ug  >= 0.5 else "—",
                })

            if rows_tratadas:
                st.dataframe(pd.DataFrame(rows_tratadas).set_index("Línea"), use_container_width=True)
            else:
                st.caption("El solver no eligió tratar ninguna línea con estas medidas en esta corrida.")
            st.caption(
                "El solver elige por línea candidata (`sw_cand_list`) si conviene "
                "aplicar cada medida, sopesando su CAPEX/OPEX por km contra el ENS "
                "evitado — no se aplica a toda la red de forma fija."
            )

        if config_activo.get("inv_switch_rcs") or config_activo.get("inv_switch_ms"):
            st.subheader("🔌 Inversión en Switches — Líneas Inter-zona")

            sw_cands = config_activo.get("sw_candidatas", [])
            sw_data  = _extraer_switches_inv(results_activos, sw_cands)

            if sw_data:
                rows_sw = []
                for l, vals in sw_data.items():
                    fb = int(net.line.at[l, "from_bus"]) if l in net.line.index else "—"
                    tb = int(net.line.at[l, "to_bus"])   if l in net.line.index else "—"
                    rows_sw.append({
                        "Línea":              l,
                        "Nombre":             net.line.at[l, "name"] if l in net.line.index else f"L{l}",
                        "Bus origen (zona)":  f"{fb} (Z{bus_zone.get(fb,'?')})",
                        "Bus destino (zona)": f"{tb} (Z{bus_zone.get(tb,'?')})",
                        "RCS instalado":      "✅" if vals["y_rcs"] > 0.5 else "—",
                        "MS instalado":       "✅" if vals["y_ms"]  > 0.5 else "—",
                        "Horas activa (prom)": vals["horas_activa_prom"],
                        "Energía transferida [kWh]": vals["P_sw_total_prom"],
                    })

                df_sw = pd.DataFrame(rows_sw).set_index("Línea")
                st.dataframe(df_sw, use_container_width=True)

                # Métricas resumen
                n_rcs = sum(1 for v in sw_data.values() if v["y_rcs"] > 0.5)
                n_ms  = sum(1 for v in sw_data.values() if v["y_ms"]  > 0.5)
                n_ninguno = len(sw_cands) - n_rcs - n_ms
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Líneas candidatas",   len(sw_cands))
                c2.metric("RCS instalados",       n_rcs)
                c3.metric("MS instalados",        n_ms)
                c4.metric("Sin switch (no rentable)", n_ninguno)
            else:
                st.info("No hay resultados de switches para este escenario.")

    else:
        st.info("⚙️ Ejecuta al menos una optimización en la pestaña **⚙️ Optimización** para ver capacidades instaladas.")

    # ── Sub-sección D: Costos y beneficio neto ───────────────
    st.header("💵 D — Costos y Beneficio Neto")

    if results_activos:
        costos = _extraer_costos_objetivo(results_activos, results_sin, config_activo, data)

        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("CAPEX anualizado [USD]",  f"{costos['capex_anual']:,.0f}")
        col_c2.metric("OPEX anual [USD]",         f"{costos['opex_anual']:,.0f}")
        col_c3.metric("Ahorro ENS [USD]",         f"{costos['ahorro_ens']:,.0f}")

        beneficio = costos["beneficio"]
        payback   = (costos["capex_anual"] / (costos["ahorro_ens"] - costos["opex_anual"])
                     if (costos["ahorro_ens"] - costos["opex_anual"]) > 0 else float("inf"))

        col_b1, col_b2 = st.columns(2)
        col_b1.metric("💰 Beneficio neto [USD]",       f"{beneficio:,.0f}",
                      delta="Positivo" if beneficio > 0 else "Negativo")
        col_b2.metric("📅 Período de recuperación",
                      f"{payback:.1f} años" if payback != float("inf") else "No recuperable")

        # Gráfico barras apiladas
        fig_costos = go.Figure()
        categorias = ["Caso base", label_activos]
        capex_vals = [0, costos["capex_anual"]]
        opex_vals  = [0, costos["opex_anual"]]
        fig_costos.add_trace(go.Bar(name="CAPEX anualizado", x=categorias, y=capex_vals,
                                    marker_color="#1f77b4"))
        fig_costos.add_trace(go.Bar(name="OPEX anual",       x=categorias, y=opex_vals,
                                    marker_color="#ff7f0e"))
        fig_costos.update_layout(
            barmode="stack", title="CAPEX + OPEX por escenario [USD]",
            yaxis_title="Costo [USD]", height=380
        )
        st.plotly_chart(_fig_pantalla(fig_costos, height=380),
                        use_container_width=True, key="capex_opex")
        _descargas_figura(fig_costos, "capex_opex", key="costos_cmp",
                          etiqueta="el gráfico de CAPEX+OPEX")
    else:
        st.info("⚙️ Ejecuta al menos una optimización para ver el análisis de costos.")

    # ── Sub-sección E: Comparación de estrategias ────────────
    st.header("⚖️ E — Comparación de Estrategias")

    if not historial:
        st.info("⚙️ Configura estrategias en la pestaña **⚙️ Optimización** y presiona **▶️ Calcular**.")
    else:
        esc_labels      = list(SAIDI_sin.keys())
        periodo_h       = N_HORAS

        # ══════════════════════════════════════════════════════
        # CASO BASE POR ESTRATEGIA
        #
        # Cada estrategia se compara contra el caso base DEL ESCENARIO Y DEL
        # SUPUESTO DE AISLACIÓN CON QUE SE RESOLVIÓ, no contra el del
        # escenario que esté seleccionado ahora en la barra lateral.
        #
        # Por qué importa: `_norm()` re-indexa los resultados de la estrategia
        # a {0: [...]} y borra el índice Monte Carlo, mientras que
        # `baseline_esc()` devuelve {esc_seleccionado: [...]}. Los dos quedan
        # como listas de un elemento, así que `_results_to_list` emparejaba la
        # posición 0 de una con la posición 0 de la otra aunque fueran
        # escenarios distintos. Y los escenarios NO son intercambiables: la
        # ENS del caso base va de 4.469 kWh (esc. 34) a 47.045 kWh (esc. 21),
        # un factor de 10. Comparar contra el escenario equivocado mostraba
        # aumentos de ENS enormes para medidas que en realidad funcionaban.
        # ══════════════════════════════════════════════════════
        def _ind_base(base_dict):
            """SAIDI/SAIFI/ENS promedio de un caso base. None si viene vacío."""
            vs = [v for v in _results_to_list(base_dict or {}) if v]
            if not vs:
                return None
            return {
                "saidi": float(np.mean([fn_SAIDI(v, P_D_base, BUSES, horas) for v in vs])),
                "saifi": float(np.mean([fn_SAIFI(v, P_D_base, BUSES, horas) for v in vs])),
                "ens":   float(np.mean([fn_ENS_total(v, BUSES, horas) for v in vs])),
            }

        def _escenarios_de(cfg):
            """Escenarios MC (0-based) sobre los que se resolvió la estrategia."""
            if cfg.get("estocastico"):
                return sorted(int(e) for e in (cfg.get("escenarios_estoc") or []))
            e = cfg.get("esc_debug")
            return [int(e)] if e is not None else []

        def _variante_de(cfg):
            """Supuesto de aislación con que se resolvió la estrategia."""
            return _variante_aislacion(
                bool(cfg.get("aislacion_fallas_fija", False)),
                list(cfg.get("seccionadores_existentes", []) or []),
            )

        _memo_base = {}

        def _base_de(cfg):
            """(baseline, indicadores, aviso) del caso base propio de `cfg`.

            `aviso` queda en None cuando la comparación es legítima; si no,
            trae el texto que explica por qué no lo es.
            """
            escs, var = _escenarios_de(cfg), _variante_de(cfg)
            clave = (tuple(escs), var)
            if clave in _memo_base:
                return _memo_base[clave]

            if not escs:
                # Estrategia vieja (guardada antes de que se registrara el
                # escenario) o corrida sobre todos: se cae al base mostrado.
                res = (results_sin, _ind_base(results_sin),
                       "no registra con qué escenario se resolvió, así que se "
                       "compara contra el caso base del escenario "
                       f"{esc_sel} que está seleccionado")
                _memo_base[clave] = res
                return res

            base, faltan = {}, []
            for e in escs:
                b = baseline_esc(e, var)
                if b.get(e):
                    base[e] = b[e]
                else:
                    faltan.append(e + 1)

            if not base:
                res = (None, None,
                       f"no tiene caso base calculado para el escenario "
                       f"{', '.join(str(f) for f in faltan)} en el supuesto "
                       f"`{var}` (correr la etapa E5 del pipeline)")
            elif faltan:
                res = (base, _ind_base(base),
                       f"le faltan los baselines de los escenarios "
                       f"{', '.join(str(f) for f in faltan)}: el promedio del "
                       f"caso base se calcula solo con los que sí están")
            else:
                res = (base, _ind_base(base), None)
            _memo_base[clave] = res
            return res

        def _etiqueta_esc(cfg):
            """Texto corto del escenario de una estrategia, para las tablas."""
            escs = _escenarios_de(cfg)
            if not escs:
                return "—"
            if len(escs) == 1:
                return str(escs[0] + 1)
            return f"{len(escs)} esc. (estocástico)"

        # Base del escenario seleccionado: sigue siendo la fila "Caso base" de
        # la tabla y la referencia de los gráficos por escenario.
        valid_sin_e     = [v for v in _results_to_list(results_sin) if v]
        _ind_sel        = _ind_base(results_sin) or {"saidi": 0.0, "saifi": 0.0, "ens": 0.0}
        saidi_base_prom = _ind_sel["saidi"]
        saifi_base_prom = _ind_sel["saifi"]
        ens_base_prom   = _ind_sel["ens"]
        # Costo de ENS caso base = función objetivo sin estrategias (VoLL × ENS × N_eventos)
        ens_costo_base  = fn_ens_costo_base(results_sin, horas, BUSES, VoLL, N_eventos)

        # Aviso global: si el historial mezcla escenarios, las filas de la
        # tabla NO son comparables entre sí aunque cada una esté bien
        # calculada contra su propio base.
        _escs_hist = {tuple(_escenarios_de(v.get("config", {})))
                      for v in historial.values()}
        _vars_hist = {_variante_de(v.get("config", {})) for v in historial.values()}
        if len(_escs_hist) > 1:
            st.warning(
                "⚠️ El historial mezcla estrategias resueltas sobre escenarios "
                "Monte Carlo distintos. Cada una se compara contra el caso base "
                "de SU escenario (la columna «Escenario» lo indica), así que "
                "sus Δ son correctos, pero **no son comparables entre sí**: la "
                "ENS del caso base varía hasta 10× de un escenario a otro. Para "
                "un ranking de medidas, córrelas todas sobre el mismo escenario "
                "o usa la pestaña 🧪 Barrido."
            )
        if len(_vars_hist) > 1:
            st.error(
                "❌ El historial mezcla supuestos de aislación distintos "
                f"({', '.join(sorted(_vars_hist))}). Esas corridas no son "
                "comparables entre sí bajo ningún criterio: el supuesto cambia "
                "el caso base y el alcance físico de cada falla."
            )

        def _get_serie_comp(results_dict, fn, saidi_ref=None):
            valid = [v for v in _results_to_list(results_dict) if v]
            if not valid:
                return [0.0] * N_HORAS
            saidi_ref = saidi_ref or SAIDI_sin
            if vista_global == "📊 Promedio":
                return _promedio_por_hora([fn(v) for v in valid])
            elif vista_global == "⚠️ Peor caso":
                idx = min(_esc_peor_caso(saidi_ref), len(valid) - 1)
                return fn(valid[idx])
            else:
                idx = _pos_esc(esc_sel - 1, len(valid))
                return fn(valid[idx])

        # ══════════════════════════════════════════════════════
        # PANEL 0 — TARJETAS POR ESTRATEGIA
        # ══════════════════════════════════════════════════════
        st.markdown("### 🗂️ Resumen por Estrategia")

        for i_s, (lbl_s, datos_s) in enumerate(historial.items()):
            cfg_s  = datos_s["config"]
            # Caso base DE ESTA estrategia (su escenario, su supuesto de
            # aislación), no el del escenario que esté seleccionado.
            base_s, ind_base_s, aviso_s = _base_de(cfg_s)
            sv_s   = list(datos_s["saidi"].values())
            prom_s = np.mean(sv_s)

            valid_s  = [v for v in _results_to_list(datos_s["results"]) if v]
            saifi_s  = np.mean([fn_SAIFI(v, P_D_base, BUSES, horas) for v in valid_s]) if valid_s else 0
            ens_s    = np.mean([fn_ENS_total(v, BUSES, horas)        for v in valid_s]) if valid_s else 0

            if ind_base_s:
                saidi_b_s, saifi_b_s, ens_b_s = (ind_base_s["saidi"],
                                                 ind_base_s["saifi"],
                                                 ind_base_s["ens"])
            else:
                saidi_b_s = saifi_b_s = ens_b_s = float("nan")
            diff_s = saidi_b_s - prom_s
            pct_s  = diff_s / saidi_b_s * 100 if saidi_b_s else 0

            has_invest = any(cfg_s.get(k) for k in (
                "dg_sol","dg_eol","dg_fos","ess_5h","ess_4h","ess_6h","meg","mess",
                "inv_switch_rcs","inv_switch_ms","reconfig_topo","inv_tie","veg_mgmt",
                "refuerzo","soterramiento"))
            eco_s = (_extraer_costos_objetivo(datos_s["results"], base_s, cfg_s, data)
                     if has_invest and base_s else None)

            if eco_s is None:
                semaforo = "🟡 Sin inversión" if base_s else "⚪ Sin caso base"
            elif eco_s["beneficio"] > 0:
                semaforo = "🟢 Rentable"
            elif eco_s["beneficio"] > -eco_s["costo_total"] * 0.2:
                semaforo = "🟡 Marginalmente rentable"
            else:
                semaforo = "🔴 No rentable"

            _esc_s = _etiqueta_esc(cfg_s)
            with st.expander(f"**{lbl_s}** — escenario {_esc_s} — {semaforo}",
                             expanded=True):
                if aviso_s:
                    st.warning(f"⚠️ Esta estrategia {aviso_s}.")
                if ind_base_s is None:
                    st.error(
                        "❌ Sin caso base propio no se puede calcular ninguna "
                        "mejora: los Δ de abajo quedan sin valor."
                    )

                # ── Medidas: Activadas vs. Realmente Usadas ──
                st.markdown("**🔧 Medidas Activadas vs. Realmente Usadas por el Optimizador:**")

                cap_dg_s  = _extraer_cap_dg(datos_s["results"])
                cap_meg_s = _extraer_cap_meg(datos_s["results"])

                # ESS: capacidad instalada real por duración
                ess_cap = {}
                # for dur in ["5h", "4h", "6h"]:
                for dur in ["4h","5h", "6h"]:
                    cap_dur = np.mean([
                        sum(v[0].get(f"CAP_ESS_kWh_{dur}", {}).values())
                        for v in valid_s if v
                    ]) if valid_s else 0
                    ess_cap[dur] = cap_dur/int(dur[0])

                # Reconfig: horas promedio con al menos una tie-line cerrada
                # tie_indices = list(set(sw_cand_list)) if sw_cand_list else (LINES[-3:] if len(LINES) >= 3 else LINES)
                tie_indices = LINES[-3:] if len(LINES) >= 3 else LINES

                tie_horas = 0.0
                if valid_s and (cfg_s.get("reconfig_topo") or cfg_s.get("inv_tie")):
                    tie_horas = np.mean([
                        sum(1 for hd in v
                            if any(hd.get("z_tie", {}).get(l, 0) > 0.5
                                   for l in tie_indices))
                        for v in valid_s
                    ])

                # Enlaces nuevos: cuántos decidió construir el solver (y_tie)
                tie_construidos = 0
                if valid_s and cfg_s.get("inv_tie"):
                    for l in cfg_s.get("tie_candidatas", tie_indices):
                        vals_y = [v[0].get("y_tie", {}).get(l, 0) for v in valid_s if v]
                        if vals_y and float(np.mean(vals_y)) >= 0.5:
                            tie_construidos += 1

                # RCS / MS: cantidad de líneas donde se invirtió realmente
                rcs_inst, ms_inst = 0, 0
                if valid_s and (cfg_s.get("inv_switch_rcs") or cfg_s.get("inv_switch_ms")):
                    sw_data_s = _extraer_switches_inv(datos_s["results"], sw_cand_list)
                    rcs_inst  = sum(1 for v in sw_data_s.values() if v["y_rcs"] > 0.5)
                    ms_inst   = sum(1 for v in sw_data_s.values() if v["y_ms"]  > 0.5)

                # EDRP: kWh promedio reducidos por escenario
                edrp_mwh = 0.0
                if valid_s and cfg_s.get("edrp"):
                    edrp_mwh = np.mean([
                        sum(
                            val for hd in v
                            for (b, c), val in hd.get("P_EDRP", {}).items()
                        )
                        for v in valid_s
                    ])

                # Prior. críticos: no tiene capacidad numérica, solo on/off
                prior_act = cfg_s.get("prior_criticos", False)

                # Vegetación / refuerzo / soterramiento: km efectivamente
                # tratados según lo que decidió el solver (x_veg/x_ref/x_ug
                # por línea candidata), no el universo completo de candidatas.
                def _km_tratados(clave, activo):
                    if not (valid_s and activo):
                        return 0.0
                    total_km = 0.0
                    for l in sw_cand_list:
                        vals = [v[0].get(clave, {}).get(l, 0) for v in valid_s if v]
                        p = float(np.mean(vals)) if vals else 0.0
                        if p >= 0.5:
                            total_km += data.get("longitud_km_linea", {}).get(l, 0)
                    return total_km

                km_veg = _km_tratados("x_veg", cfg_s.get("veg_mgmt", False))
                km_ref = _km_tratados("x_ref", cfg_s.get("refuerzo", False))
                km_ug  = _km_tratados("x_ug",  cfg_s.get("soterramiento", False))

                # ── Lista unificada de medidas ─────────────────
                cap_mess_res = _extraer_cap_mess(datos_s["results"]) if cfg_s.get("mess", False) else {}
                cap_mess_total_res = float(sum(cap_mess_res.values())) if cap_mess_res else 0.0
                mess_resumen_df = pd.DataFrame()
                mess_loc_df = pd.DataFrame()
                mess_bus_df = pd.DataFrame()
                if cfg_s.get("mess", False) and valid_s:
                    res_ref_mess = valid_s[0]
                    mess_resumen_df = _mess_resumen_unidades(res_ref_mess)
                    mess_loc_df = _mess_localizacion_horaria(res_ref_mess)
                    mess_bus_aux = _mess_por_bus(res_ref_mess, modo="descarga")
                    mess_bus_acum = mess_bus_aux.sum(axis=0).sort_values(ascending=False)
                    mess_bus_df = pd.DataFrame({
                        "Bus": mess_bus_acum.index,
                        "Descarga acumulada MESS [kW-h eq]": mess_bus_acum.values,
                    })
                medidas_info = [
                    ("☀️ DG Solar",   "dg_sol",
                     sum(cap_dg_s[g]["sol"][0] for g in GENS) if cap_dg_s else 0,
                     "kW"),
                    ("💨 DG Eólica",  "dg_eol",
                     sum(cap_dg_s[g]["eol"][0] for g in GENS) if cap_dg_s else 0,
                     "kW"),
                    ("⛽ DG Fósil",   "dg_fos",
                     sum(cap_dg_s[g]["fos"][0] for g in GENS) if cap_dg_s else 0,
                     "kW"),
                    #  ("🔋 ESS 5h",      "ess_5h",         ess_cap["5h"],  "kWh"),
                    ("🔋 ESS 4h",      "ess_4h",         ess_cap["4h"],  "kWh"),
                    ("🔋 ESS 5h",      "ess_5h",         ess_cap["5h"],  "kWh"),
                    ("🔋 ESS 6h",      "ess_6h",         ess_cap["6h"],  "kWh"),
                    ("🔀 Reconfig.",   "reconfig_topo",  tie_horas,      "h-activa"),
                    ("🔗 Enlaces nuevos", "inv_tie",     tie_construidos, f"/{len(cfg_s.get('tie_candidatas', tie_indices))} enlaces"),
                    ("🔌 Switch RCS",  "inv_switch_rcs", rcs_inst,       f"/{len(sw_cand_list)} lín."),
                    ("🔌 Switch MS",   "inv_switch_ms",  ms_inst,        f"/{len(sw_cand_list)} lín."),
                    ("🚛 MEG",         "meg",
                     sum(cap_meg_s.values()) if cap_meg_s else 0, "kW"),
                    ("🚚 MESS",        "mess",           cap_mess_total_res, "kWh"),
                    ("📉 EDRP",        "edrp",           edrp_mwh,       "kWh"),
                    ("🔺 Prior. críticos", "prior_criticos", 1.0 if prior_act else 0.0, "activa"),
                    ("🌳 Gestión vegetación", "veg_mgmt", km_veg, "km podados"),
                    ("🔩 Refuerzo estructural", "refuerzo", km_ref, "km reforzados"),
                    ("🕳️ Soterramiento", "soterramiento", km_ug, "km soterrados"),
                ]

                # ── Tabla lista de medidas ────────────────────
                rows_med = []
                for (mnom, mkey, cap_val, unidad) in medidas_info:
                    activada = cfg_s.get(mkey, False)
                    if not activada:
                        estado = "❌ No activada"
                        uso    = "—"
                    elif cap_val < 0.001:
                        estado = "⚠️ Activada"
                        uso    = "Óptimo = 0"
                    else:
                        estado = "✅ Usada"
                        uso    = "Sí" if unidad == "activa" else f"{cap_val:.1f} {unidad}"
                    rows_med.append({"Medida": mnom, "Estado": estado, "Uso real": uso})

                df_med = pd.DataFrame(rows_med).set_index("Medida")

                def _color_estado(val):
                    if val.startswith("✅"): return "color: #155724; font-weight: bold"
                    if val.startswith("⚠️"): return "color: #856404; font-weight: bold"
                    if val.startswith("❌"): return "color: #721c24"
                    return ""

                st.dataframe(
                    df_med.style.applymap(_color_estado, subset=["Estado"]),
                    use_container_width=True,
                    height=35 * (len(rows_med) + 1) + 10,
                )

                st.markdown("---")

                # ── KPIs confiabilidad ────────────────────────
                # Los Δ van contra el base DE ESTA estrategia (`*_b_s`), que es
                # el del escenario con que se resolvió.
                st.markdown("**🔢 Indicadores de Confiabilidad:**")
                st.caption(
                    f"Δ contra el caso base del escenario {_esc_s} "
                    f"(SAIDI {saidi_b_s:.3f} · SAIFI {saifi_b_s:.3f} · "
                    f"ENS {ens_b_s:,.0f} kWh)."
                    if ind_base_s else "Sin caso base disponible."
                )
                k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
                k1.metric("SAIDI [h/año]",    f"{prom_s:.3f}",
                          delta=f"{-(saidi_b_s-prom_s):.3f}", delta_color="inverse")
                k2.metric("Δ SAIDI %",        f"{pct_s:.1f}%",
                          delta=f"{pct_s:.1f}%", delta_color="normal")
                k3.metric("SAIFI [int./año]",  f"{saifi_s:.3f}",
                          delta=f"{-(saifi_b_s-saifi_s):.3f}", delta_color="inverse")
                k4.metric("ENS total [kWh]",  f"{ens_s:.2f}",
                          delta=f"{-(ens_b_s-ens_s):.2f}", delta_color="inverse")

                # ── KPIs económicos ───────────────────────────
                if eco_s:
                    st.markdown("**💰 Evaluación Económica:**")
                    e1,e2,e3,e4,e5 = st.columns(5)
                    roi_s = eco_s["beneficio"] / eco_s["costo_total"] * 100 if eco_s["costo_total"] > 0 else 0
                    pb_s  = eco_s["costo_total"] / eco_s["ahorro_ens"] if eco_s["ahorro_ens"] > 0 else float("inf")
                    e1.metric("CAPEX anual [USD]",    f"${eco_s['capex_anual']:,.0f}")
                    e2.metric("OPEX anual [USD]",     f"${eco_s['opex_anual']:,.0f}")
                    e3.metric("Costo total [USD]",    f"${eco_s['costo_total']:,.0f}")
                    e4.metric("Función objetivo [USD]",    f"${eco_s['fn_objetivo']:,.0f}")
                    e5.metric("Ahorro ENS [USD]",     f"${eco_s['ahorro_ens']:,.0f}")
                    # e6.metric("Beneficio neto [USD]", f"${eco_s['beneficio']:,.0f}",
                            #   delta="Positivo" if eco_s["beneficio"] > 0 else "Negativo")
                    
                    st.markdown("**💰 Costo unitario de mejora:**")

                    def _unitario(costo, mejora):
                        """USD por unidad de indicador evitada.

                        Solo tiene sentido con mejora > 0: si la medida empeora
                        el indicador (o lo deja igual) el cociente sería
                        negativo o una división por cero, no un costo unitario.
                        """
                        if not np.isfinite(mejora) or mejora <= 0:
                            return "— (no mejora)"
                        return f"${costo/mejora:,.2f}"

                    ee1,ee2,ee3 = st.columns(3)
                    ee1.metric("ENS [USD/kWh]",
                               _unitario(eco_s["costo_total"], ens_b_s - ens_s))
                    ee2.metric("SAIDI [USD/(h/año)]",
                               _unitario(eco_s["costo_total"], saidi_b_s - prom_s))
                    ee3.metric("SAIFI [USD/(int./año)]",
                               _unitario(eco_s["costo_total"], saifi_b_s - saifi_s))
                    # ee1.metric("ENS [USD/MWh]",    f"${(eco_s['capex_anual']+eco_s['opex_anual'])/(ens_base_prom-ens_s):,.2f}")
                    # ee2.metric("SAIDI [USD/MWh]",    f"${(eco_s['capex_anual']+eco_s['opex_anual'])/(saidi_base_prom-prom_s):,.2f}")
                    # ee3.metric("SAIFI [USD/MWh]",    f"${(eco_s['capex_anual']+eco_s['opex_anual'])/(saifi_base_prom-saifi_s):,.2f}")
                    # e7.metric("ROI / Payback", f"{roi_s:.1f}% / " +
                    #           (f"{pb_s:.1f}a" if pb_s != float("inf") else "∞"))

        st.markdown("---")

        # ══════════════════════════════════════════════════════
        # PANEL 1 — TABLA COMPARATIVA COMPLETA
        # ══════════════════════════════════════════════════════

        st.markdown("### 📊 Tabla comparativa de indicadores")

        st.caption(
            "Cada estrategia se compara contra el caso base **de su propio "
            "escenario** (columna «Escenario»). La fila *Caso base* es la del "
            f"escenario {esc_sel}, el seleccionado en la barra lateral."
        )

        rows_comp = [{
            "Estrategia": "Caso base",
            "Escenario": str(esc_sel),
            "SAIDI [h/año]": saidi_base_prom,
            "SAIFI [int./año]": saifi_base_prom,
            "ENS total [kWh]": ens_base_prom,
            # "ASAI [p.u.]": asai_base,
            "Δ SAIDI": np.nan,
            "Δ SAIDI %": np.nan,
            "Δ SAIFI": np.nan,
            "Δ SAIFI %": np.nan,
            "Costo ENS estrategia [USD]": ens_costo_base,
            "CAPEX+OPEX [USD]": 0.0,
            "Ahorro ENS [USD]": 0.0,
            "Beneficio neto [USD]": 0.0,
        }]

        for lbl_h, val_h in historial.items():
            cfg_h = val_h.get("config", {})
            lbl_h_corto = _nombre_estrategia_corto(lbl_h, cfg_h)
            # Base propio de esta estrategia: su escenario, su supuesto de
            # aislación. Restar contra el base del escenario seleccionado daba
            # Δ sin sentido (hasta 10× de diferencia entre escenarios).
            base_h, ind_base_h, _aviso_h = _base_de(cfg_h)
            if ind_base_h:
                saidi_b_h, saifi_b_h, ens_b_h = (ind_base_h["saidi"],
                                                 ind_base_h["saifi"],
                                                 ind_base_h["ens"])
            else:
                saidi_b_h = saifi_b_h = ens_b_h = float("nan")

            sv_h    = list(val_h["saidi"].values())
            prom_h  = np.mean(sv_h)
            diff_h  = saidi_b_h - prom_h
            pct_h   = diff_h / saidi_b_h * 100 if saidi_b_h else np.nan
            vr_h    = [v for v in _results_to_list(val_h["results"]) if v]
            saifi_h = np.mean([fn_SAIFI(v, P_D_base, BUSES, horas) for v in vr_h]) if vr_h else 0
            d_saifi_h = saifi_b_h - saifi_h
            pct_saifi_h = d_saifi_h / saifi_b_h * 100 if saifi_b_h > 0 else np.nan
            ens_h   = np.mean([fn_ENS_total(v, BUSES, horas) for v in vr_h]) if vr_h else 0

            has_inv = any(cfg_h.get(k) for k in (
                "dg_sol","dg_eol","dg_fos","ess_5h","ess_4h","ess_6h","meg","mess",
                "inv_switch_rcs","inv_switch_ms","reconfig_topo","inv_tie","edrp","veg_mgmt",
                "refuerzo","soterramiento"
            ))
            eco_h = (_extraer_costos_objetivo(val_h["results"], base_h, cfg_h, data)
                     if base_h else {})
            if has_inv and eco_h:
                costo_ens_estr = eco_h["costo_ens_estr"]
                costo_total = eco_h["costo_total"]
                ahorro_ens  = eco_h["ahorro_ens"]
                beneficio   = eco_h["beneficio"]
            else:
                costo_ens_estr = eco_h.get("costo_ens_estr", np.nan)
                costo_total = 0.0
                ahorro_ens  = 0.0
                beneficio   = 0.0

            delta_ENS = ens_b_h - ens_h
            rows_comp.append({
                "Estrategia": lbl_h_corto,
                "Escenario": _etiqueta_esc(cfg_h),
                "SAIDI [h/año]": prom_h,
                "SAIFI [int./año]": saifi_h,
                "ENS total [kWh]": ens_h,
                # "ASAI [p.u.]": asai_h,
                "Δ SAIDI": diff_h,
                "Δ SAIDI %": pct_h,
                "Δ SAIFI": d_saifi_h,
                "Δ SAIFI %": pct_saifi_h,
                "Δ ENS": delta_ENS,
                "Δ ENS %": delta_ENS / ens_b_h * 100 if ens_b_h else np.nan,
                "Costo ENS estrategia [USD]": costo_ens_estr,
                "CAPEX+OPEX [USD]": costo_total,
                "Ahorro ENS [USD]": ahorro_ens,
                "Beneficio neto [USD]": beneficio,
                # El costo por unidad de mejora solo existe si HUBO mejora: con
                # Δ < 0 (la medida empeora el indicador) el cociente sale
                # negativo y se lee como si costara plata negativa. Queda en
                # blanco; el Δ de más arriba es el que muestra el empeoramiento.
                "Reducción ENS [USD/kWh]":
                    costo_total/delta_ENS if delta_ENS > 0 else np.nan,
                "Reducción SAIDI [USD/(h/año)]":
                    costo_total/diff_h if diff_h > 0 else np.nan,
                "Reducción SAIFI [USD/(int./año)]":
                    costo_total/d_saifi_h if d_saifi_h > 0 else np.nan,
            })

        df_comp_raw = pd.DataFrame(rows_comp)
        _seen = {}
        _unique_names = []
        for _name in df_comp_raw["Estrategia"]:
            if _name in _seen:
                _seen[_name] += 1
                _unique_names.append(f"{_name} ({_seen[_name]})")
            else:
                _seen[_name] = 0
                _unique_names.append(_name)
        df_comp_raw["Estrategia"] = _unique_names
        df_comp = df_comp_raw.set_index("Estrategia")

        def _color_pos_neg(v):
            if pd.isna(v):
                return ""
            if v > 0:
                return "background-color:#d4edda;color:#155724;font-weight:bold"
            if v < 0:
                return "background-color:#f8d7da;color:#721c24;font-weight:bold"
            return ""
        
        def _color_neg_pos(v):
            if pd.isna(v):
                return ""
            if v < 0:
                return "background-color:#d4edda;color:#155724;font-weight:bold"
            if v > 0:
                return "background-color:#f8d7da;color:#721c24;font-weight:bold"
            return ""


        cmet1, cmet2 = st.columns([1, 2])
        with cmet1:
            st.metric("Costo ENS caso base [USD]", f"{ens_costo_base:,.0f}")
        with cmet2:
            st.caption("Referencia económica sin estrategias; la tabla muestra el costo ENS resultante de cada estrategia.")

        st.markdown("## 🔢 Indicadores de confiabilidad")
        st.caption(
            "Δ positivo = mejora respecto del caso base del escenario indicado "
            "en la columna «Escenario»."
        )
        cols_conf = [
            "Escenario",
            "SAIDI [h/año]", "SAIFI [int./año]", "ENS total [kWh]",
            "Δ SAIDI", "Δ SAIDI %", "Δ SAIFI", "Δ SAIFI %", "Δ ENS","Δ ENS %",
        ]
        df_conf = df_comp[cols_conf]
        st.dataframe(
            df_conf.style
                .format({
                    "SAIDI [h/año]": "{:.3f}",
                    "SAIFI [int./año]": "{:.3f}",
                    "ENS total [kWh]": "{:,.2f}",
                    # "ASAI [p.u.]": "{:.6f}",
                    "Δ SAIDI": "{:.3f}",
                    "Δ SAIDI %": "{:.1f}%",
                    "Δ SAIFI": "{:.3f}",
                    "Δ SAIFI %": "{:.1f}%",
                    "Δ ENS": "{:.1f}",
                    "Δ ENS %": "{:.1f}%",
                }, na_rep="—")
                .applymap(_color_pos_neg, subset=["Δ SAIDI %", "Δ SAIFI %", "Δ ENS %"]),
            use_container_width=True,
            height=220
        )

        st.markdown("## 💰 Indicadores económicos")
        cols_eco = [
            "Costo ENS estrategia [USD]", "CAPEX+OPEX [USD]", "Ahorro ENS [USD]", "Beneficio neto [USD]"
        ]
        df_eco = df_comp[cols_eco]
        st.dataframe(
            df_eco.style
                .format({
                    "Costo ENS estrategia [USD]": "{:,.0f}",
                    "CAPEX+OPEX [USD]": "{:,.0f}",
                    "Ahorro ENS [USD]": "{:,.0f}",
                    "Beneficio neto [USD]": "{:,.0f}",
                }, na_rep="—"),
                # .applymap( _color_pos_neg, subset=["Ahorro ENS [USD]"]),
                # .applymap(_color_neg_pos, subset=["Beneficio neto [USD]"]),
            use_container_width=True,
            height=180
        )

        st.caption(
            "**Costo ENS estrategia** = VoLL × ENS resultante de cada estrategia × N_eventos | "
            "**CAPEX+OPEX** = costo anualizado de la estrategia | "
            "**Ahorro ENS** = reducción del costo ENS respecto al caso base | "
            "**SAIFI** = interrupciones sostenidas por cliente | "
            # "**ASAI** = fracción del tiempo con suministro (1.0 = 100%)"
        )

        df_mac = df_comp.reset_index()
        # Mismos parámetros que la app v8: viven en curvas_mac.py.
        _curvas_mac = curvas_mac_estrategias(df_mac)
        fig_mac_ens, tabla_mac_ens = _curvas_mac["ens"]
        fig_mac_saidi, tabla_mac_saidi = _curvas_mac["saidi"]
        fig_mac_saifi, tabla_mac_saifi = _curvas_mac["saifi"]

        st.markdown("## Costos por unidad de mejora")
        df_unidad = tabla_costos_unidad(df_mac, _curvas_mac)
        df_unidad.index = df_comp.index
        st.dataframe(
            df_unidad.style
                .format({
                    "Reducción ENS [USD/MWh]": "{:,.1f}",
                    "Reducción SAIDI [USD/(h/año)]": "{:,.0f}",
                    "Reducción SAIFI [USD/(int./año)]": "{:,.0f}",
                }, na_rep="—"),
                # .applymap( _color_pos_neg, subset=["Reducción ENS [USD/kWh]","Reducción SAIDI [USD/(h/año)]", "Reducción SAIFI [USD/(int./año)]"]),
                # .applymap(_color_neg_pos, subset=["Beneficio neto [USD]"]),
            use_container_width=True,
            height=180
        )


        tab_ens, tab_saidi, tab_saifi = st.tabs([
            "ENS",
            "SAIDI",
            "SAIFI"
        ])

        with tab_ens:
            if fig_mac_ens is not None:
                # `construir_curva_abatimiento_2` ya devuelve la figura en
                # tamaño de tesis (1500×850, fuentes 15-26): eso es lo que se
                # descarga más abajo, y a la pantalla va la copia reescalada.
                st.plotly_chart(
                    _fig_pantalla(fig_mac_ens, height=460),
                    use_container_width=True
                )
                col_png_ens, col_html_ens = st.columns(2)
                try:
                    png_ens = fig_mac_ens.to_image(format="png", scale=3)
                except Exception:
                    col_png_ens.caption(
                        "📷 PNG no disponible: falta el paquete "
                        "`kaleido` en este entorno (`pip install --upgrade kaleido`)."
                    )
                else:
                    col_png_ens.download_button(
                        "📷 Descargar PNG (alta resolución)",
                        data=png_ens,
                        file_name="curva_mac_ens.png",
                        mime="image/png",
                        key="download_mac_ens_png",
                        help="Lista para pegar en PPT: no es interactiva, pero cada barra ya muestra su costo.",
                    )
                col_html_ens.download_button(
                    "🌐 Descargar HTML interactivo",
                    data=fig_mac_ens.to_html(include_plotlyjs="cdn"),
                    file_name="curva_mac_ens.html",
                    mime="text/html",
                    key="download_mac_ens_html",
                    help="Ábrelo en un navegador para hover, zoom y leyenda clickeable durante la defensa.",
                )

        with tab_saidi:
            if fig_mac_saidi is not None:
                st.plotly_chart(
                    _fig_pantalla(fig_mac_saidi, height=460),
                    use_container_width=True
                )
                col_png_saidi, col_html_saidi = st.columns(2)
                try:
                    png_saidi = fig_mac_saidi.to_image(format="png", scale=3)
                except Exception:
                    col_png_saidi.caption(
                        "📷 PNG no disponible: falta el paquete "
                        "`kaleido` en este entorno (`pip install --upgrade kaleido`)."
                    )
                else:
                    col_png_saidi.download_button(
                        "📷 Descargar PNG (alta resolución)",
                        data=png_saidi,
                        file_name="curva_mac_saidi.png",
                        mime="image/png",
                        key="download_mac_saidi_png",
                        help="Lista para pegar en PPT: no es interactiva, pero cada barra ya muestra su costo.",
                    )
                col_html_saidi.download_button(
                    "🌐 Descargar HTML interactivo",
                    data=fig_mac_saidi.to_html(include_plotlyjs="cdn"),
                    file_name="curva_mac_saidi.html",
                    mime="text/html",
                    key="download_mac_saidi_html",
                    help="Ábrelo en un navegador para hover, zoom y leyenda clickeable durante la defensa.",
                )

        with tab_saifi:
            if fig_mac_saifi is not None:
                st.plotly_chart(
                    _fig_pantalla(fig_mac_saifi, height=460),
                    use_container_width=True
                )
                col_png_saifi, col_html_saifi = st.columns(2)
                try:
                    png_saifi = fig_mac_saifi.to_image(format="png", scale=3)
                except Exception:
                    col_png_saifi.caption(
                        "📷 PNG no disponible: falta el paquete "
                        "`kaleido` en este entorno (`pip install --upgrade kaleido`)."
                    )
                else:
                    col_png_saifi.download_button(
                        "📷 Descargar PNG (alta resolución)",
                        data=png_saifi,
                        file_name="curva_mac_saifi.png",
                        mime="image/png",
                        key="download_mac_saifi_png",
                        help="Lista para pegar en PPT: no es interactiva, pero cada barra ya muestra su costo.",
                    )
                col_html_saifi.download_button(
                    "🌐 Descargar HTML interactivo",
                    data=fig_mac_saifi.to_html(include_plotlyjs="cdn"),
                    file_name="curva_mac_saifi.html",
                    mime="text/html",
                    key="download_mac_saifi_html",
                    help="Ábrelo en un navegador para hover, zoom y leyenda clickeable durante la defensa.",
                )

        


        # ══════════════════════════════════════════════════════
        # PANEL 1.5 — GUARDAR TODOS LOS RESULTADOS CALCULADOS
        # ══════════════════════════════════════════════════════
        st.markdown("### 💾 Guardar Todos los Resultados Calculados")
        st.caption(
            f"Hay **{len(historial)}** estrategia(s) calculada(s) y guardada(s) en el historial de esta sesión "
            "(reutiliza la misma tabla comparativa mostrada arriba, sin recalcular nada)."
        )

        col_exp1, col_exp2, col_exp3 = st.columns(3)

        csv_bytes_comp = df_comp.to_csv().encode("utf-8")
        col_exp1.download_button(
            "📥 Descargar tabla comparativa (CSV)",
            data=csv_bytes_comp,
            file_name="indicadores_todas_medidas.csv",
            mime="text/csv",
            key="download_tabla_comp_csv",
            help="Incluye SAIDI, SAIFI, ENS, costos y beneficios de todas las estrategias calculadas.",
        )

        buf_xlsx_comp = io.BytesIO()
        with pd.ExcelWriter(buf_xlsx_comp, engine="openpyxl") as writer_comp:
            df_comp.to_excel(writer_comp, sheet_name="Resumen_Medidas")
        buf_xlsx_comp.seek(0)
        col_exp2.download_button(
            "📥 Descargar tabla comparativa (Excel)",
            data=buf_xlsx_comp,
            file_name="indicadores_todas_medidas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_tabla_comp_xlsx",
            help="Misma tabla en formato Excel, lista para análisis adicional.",
        )

        if col_exp3.button("🗑️ Limpiar historial de medidas", key="btn_limpiar_desde_resultados"):
            st.session_state["historial"] = {}
            st.success("🗑️ Historial limpiado. Vuelve a la pestaña ⚙️ Optimización para recalcular.")
            st.rerun()

        st.markdown("---")

        # ══════════════════════════════════════════════════════
        # PANEL 2 — GRÁFICO SAIDI POR ESCENARIO MC
        # ══════════════════════════════════════════════════════
        st.markdown("### 📉 SAIDI por Escenario Monte Carlo")

        fig_comp_saidi = go.Figure()
        fig_comp_saidi.add_trace(go.Scatter(
            x=esc_labels, y=list(SAIDI_sin.values()),
            mode="lines+markers", name="Caso base",
            line=dict(color="#d62728", width=2, dash="dash"), marker=dict(size=4)
        ))
        fig_comp_saidi.add_hline(
            y=saidi_base_prom, line_dash="dot", line_color="#d62728",
            annotation_text=f"Prom base: {saidi_base_prom:.2f}",
            annotation_position="top left"
        )
        for i_c, (lbl_h, val_h) in enumerate(historial.items()):
            s_vals  = [val_h["saidi"].get(e, 0) for e in esc_labels]
            color_c = PALETA[i_c % len(PALETA)]
            fig_comp_saidi.add_trace(go.Scatter(
                x=esc_labels, y=s_vals, mode="lines+markers", name=_nombre_estrategia_corto(lbl_h, val_h.get("config", {})),
                line=dict(color=color_c, width=2), marker=dict(size=4)
            ))
            fig_comp_saidi.add_hline(
                y=np.mean(s_vals), line_dash="dot", line_color=color_c,
                annotation_text=f"Prom {lbl_h}: {np.mean(s_vals):.2f}",
                annotation_position="bottom right"
            )
        # Layout de exportación; a la pantalla va la copia reescalada.
        fig_comp_saidi.update_layout(
            xaxis_title="Escenario", yaxis_title="SAIDI [h/año]",
            width=1600, height=700, font=dict(size=26),
            xaxis=dict(title_font=dict(size=26), tickfont=dict(size=22)),
            yaxis=dict(title_font=dict(size=26), tickfont=dict(size=22)),
            legend=dict(orientation="h", y=-0.25, font=dict(size=22)),
            margin=dict(l=90, r=40, t=40, b=90),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(
            _fig_pantalla(fig_comp_saidi, height=380, font=13,
                          margin=dict(l=60, r=30, t=30, b=60)),
            use_container_width=True, key="saidi_por_escenario",
        )
        _descargas_figura(fig_comp_saidi, "saidi_por_escenario",
                          key="saidi_esc",
                          etiqueta="el gráfico de SAIDI por escenario")

        # Métricas delta debajo del gráfico. El delta va contra el base del
        # escenario de CADA estrategia, igual que en la tabla comparativa.
        delta_cols = st.columns(max(len(historial), 1))
        for dc, (lbl_h, val_h) in zip(delta_cols, historial.items()):
            sv_dc = list(val_h["saidi"].values())
            _ind_dc = _base_de(val_h.get("config", {}))[1]
            _saidi_b_dc = _ind_dc["saidi"] if _ind_dc else float("nan")
            dc.metric(label=lbl_h, value=f"{np.mean(sv_dc):.2f} h/año",
                      delta=f"{-(_saidi_b_dc - np.mean(sv_dc)):.2f} h/año",
                      delta_color="inverse",
                      help=f"Contra el caso base del escenario "
                           f"{_etiqueta_esc(val_h.get('config', {}))}.")

        st.markdown("---")

        # ── SAIFI y ENS por zona en tabs ──────────────────────
        tab_sfi, tab_ens_z = st.tabs(["📊 SAIFI por estrategia", "⚡ ENS por zona"])

        with tab_sfi:
            # Ojo: estas son barras de NIVEL, no de mejora. La barra del caso
            # base es la del escenario seleccionado, así que solo es la
            # referencia correcta para las estrategias de ESE escenario.
            st.caption(
                f"La barra «Caso base» corresponde al escenario {esc_sel}. "
                "Las estrategias resueltas sobre otro escenario no se miden "
                "contra ella — para eso está la columna Δ de la tabla."
            )
            fig_sfi = go.Figure()
            lbl_all   = [f"Caso base (esc {esc_sel})"] + list(historial.keys())
            saifi_all = [saifi_base_prom] + [
                np.mean([fn_SAIFI(v, P_D_base, BUSES, horas)
                         for v in _results_to_list(val_h["results"]) if v]) or 0
                for val_h in historial.values()
            ]
            col_all = ["#888888"] + [PALETA[j % len(PALETA)] for j in range(len(historial))]
            fig_sfi.add_trace(go.Bar(name="SAIFI", x=lbl_all, y=saifi_all,
                                     marker_color=col_all))
            fig_sfi.update_layout(
                title="SAIFI promedio por estrategia [int./año]",
                yaxis_title="SAIFI [int./año]",
                width=1600, height=700, font=dict(size=26),
                xaxis=dict(tickfont=dict(size=20)),
                yaxis=dict(title_font=dict(size=26), tickfont=dict(size=22)),
                margin=dict(l=90, r=40, t=80, b=120),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(
                _fig_pantalla(fig_sfi, height=340, font=13,
                              margin=dict(l=60, r=30, t=50, b=80)),
                use_container_width=True, key="saifi_por_estrategia",
            )
            _descargas_figura(fig_sfi, "saifi_por_estrategia",
                              key="saifi_estr",
                              etiqueta="el gráfico de SAIFI por estrategia")

        # ZONE_FILL   = {1: "rgba(59,130,246,0.12)",  2: "rgba(245,158,11,0.12)",   3: "rgba(16,185,129,0.12)"}
        # ZONE_BORDER = {1: "rgba(29,78,216,0.55)",   2: "rgba(180,83,9,0.55)",     3: "rgba(4,120,87,0.55)"}
        NODEFILL = {
                    1: "#93C5FD",  # azul medio pastel
                    2: "#FCD34D",  # amarillo medio pastel
                    3: "#6EE7B7"   # verde medio pastel
                }
        NODEFILL = {1: "#BFDBFE", 2: "#FDE68A", 3: "#A7F3D0"}
        with tab_ens_z:
            zonas = [1, 2, 3]
            zonas_lbl = [f"Zona {z}" for z in zonas]
            colores_zona = [NODEFILL[z] for z in zonas]

            fig_zona = go.Figure()

            # =========================
            # Caso base
            # =========================
            ensz_base = {z: 0.0 for z in zonas}

            if valid_sin_e:
                for v in valid_sin_e:
                    ez = _ens_por_zona_total(v)
                    for z in zonas:
                        ensz_base[z] += ez[z]

                for z in zonas:
                    ensz_base[z] /= len(valid_sin_e)

            fig_zona.add_trace(go.Bar(
                name="Caso base",
                x=zonas_lbl,
                y=[ensz_base[z] for z in zonas],
                marker=dict(
                    # color="#888888",
                    color=colores_zona,
                    opacity=1,
                    # line=dict(color=colores_zona, width=1.5),
                    pattern=dict(shape="/")   # patrón para distinguir base
                ),
                text=[
                f"{ensz_base[z]:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
                for z in zonas],
                textposition="outside",
                textfont=dict(size=26)
            ))

            # =========================
            # Estrategias
            # =========================
            for i_z, (lbl_h, val_h) in enumerate(historial.items()):

                vr_z = [v for v in _results_to_list(val_h["results"]) if v]
                ensz_h = {z: 0.0 for z in zonas}

                if vr_z:
                    for v in vr_z:
                        ez = _ens_por_zona_total(v)
                        for z in zonas:
                            ensz_h[z] += ez[z]

                    for z in zonas:
                        ensz_h[z] /= len(vr_z)

                nombre_estrategia = _nombre_estrategia_corto(
                    lbl_h,
                    val_h.get("config", {})
                )

                fig_zona.add_trace(go.Bar(
                    name=nombre_estrategia,
                    x=zonas_lbl,
                    y=[ensz_h[z] for z in zonas],
                    marker=dict(
                        color=colores_zona,
                        opacity=0.85,
                        line=dict(color=colores_zona, width=1.5)
                    ),
                   text=[
                    f"{ensz_h[z]:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    for z in zonas
                ],
                    textposition="outside",
                    textfont=dict(size=26)
                ))

            # =========================
            # Layout
            # =========================
            fig_zona.update_layout(
                barmode="group",
                xaxis_title="Zona",
                yaxis_title="ENS [kWh]",

                width=1600,
                height=900,

                font=dict(size=40),
            #     title=dict(
            #     text="ENS por zona",
            #     font=dict(size=30),
            #     x=0.02
            # ),

                xaxis=dict(
                    title_font=dict(size=30),
                    tickfont=dict(size=30)
                ),
                yaxis=dict(
                    title_font=dict(size=30),
                    tickfont=dict(size=30)
                ),

                legend=dict(
                    orientation="h",
                    y=-0.25,
                    font=dict(size=25)
                ),

                margin=dict(l=90, r=40, t=110, b=8),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            # El layout de arriba es el de EXPORTACIÓN (1600×900, fuente 40,
            # etiquetas de barra en 26). En el navegador el ancho lo manda el
            # contenedor, así que esas fuentes quedaban gigantes sobre un
            # gráfico angosto: a la pantalla va la copia reescalada.
            st.plotly_chart(
                _fig_pantalla(fig_zona, height=900, font=13, text_size=11,
                              margin=dict(l=60, r=30, t=40, b=60)),
                use_container_width=True, key="ens_por_zona",
            )
            _descargas_figura(fig_zona, "ens_por_zona", key="ens_zona",
                              etiqueta="el gráfico de ENS por zona")

        # ── ENS horaria / diaria / mensual ───────────────────
        st.markdown("### ⚡ ENS por período — comparación")
        fig_comp_ens = go.Figure()
        ens_sin_serie = _get_serie_comp(results_sin, _ens_por_hora)
        xp, yp = _agg_serie(ens_sin_serie, gran)
        fig_comp_ens.add_trace(go.Scatter(x=xp, y=yp, mode="lines",
                                           name="Caso base",
                                           line=dict(color="#d62728", width=2)))
        for lbl_h, val_h in historial.items():
            ens_h  = _get_serie_comp(val_h["results"], _ens_por_hora, val_h["saidi"])
            xh, yh = _agg_serie(ens_h, gran)
            fig_comp_ens.add_trace(go.Scatter(x=xh, y=yh, mode="lines", name=_nombre_estrategia_corto(lbl_h, val_h.get("config", {}))))
        # Un solo layout, el de EXPORTACIÓN. Antes eran dos `update_layout`
        # encadenados y el segundo pisaba la altura del primero (250 → 700),
        # así que el primero solo servía para los títulos de los ejes.
        fig_comp_ens.update_layout(
            xaxis_title=_x_label(gran),
            yaxis_title="ENS [kWh]",
            width=1600,
            height=700,
            font=dict(size=30),
            xaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
            yaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
            legend=dict(orientation="h", y=-0.25, font=dict(size=25)),
            margin=dict(l=90, r=40, t=110, b=80),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(
            _fig_pantalla(fig_comp_ens, height=380, font=13,
                          margin=dict(l=60, r=30, t=30, b=60)),
            use_container_width=True, key="ens_por_periodo",
        )
        _descargas_figura(fig_comp_ens, f"ens_por_{_x_label(gran).lower()}",
                          key="ens_periodo",
                          etiqueta="el gráfico de ENS por período")

        # EDRP
        if any(val_h["config"].get("edrp", False) for val_h in historial.values()):
            st.subheader("📉 EDRP — Reducción de carga por bloque")
            valid_edrp = [v for v in _results_to_list(results_activos) if v]
            if valid_edrp:
                edrp_serie = _get_serie_comp(results_activos, _edrp_por_hora, saidi_activos)
                xp_e, yp_e = _agg_serie(edrp_serie, gran)
                fig_edrp = go.Figure()
                fig_edrp.add_trace(go.Bar(x=xp_e, y=yp_e, name="EDRP activado [kW]",
                                           marker_color="#9467bd"))
                fig_edrp.update_layout(
                    title=f"Reducción EDRP por {_x_label(gran).lower()}",
                    xaxis_title=_x_label(gran),
                    yaxis_title=_y_lbl("EDRP [kW]", gran), height=320
                )
                st.plotly_chart(fig_edrp, use_container_width=True)

                df_edrp_nb = calcular_edrp_nodo_bloque(results_activos,BUSES,C_bloques_EDRP=4)

                st.markdown("#### EDRP acumulado por nodo y bloque")
                st.dataframe(df_edrp_nb, width="stretch")

                # Gráfico total por nodo
                fig_edrp_nodo = go.Figure()

                fig_edrp_nodo.add_trace(go.Bar(
                    x=df_edrp_nb["bus"].astype(str),
                    y=df_edrp_nb["EDRP_total"],
                    name="EDRP total por nodo"
                ))

                fig_edrp_nodo.update_layout(
                    title="EDRP acumulado por nodo",
                    xaxis_title="Bus",
                    yaxis_title="EDRP acumulado",
                    height=360
                )

                st.plotly_chart(fig_edrp_nodo, width="stretch")

                # Gráfico apilado por bloque
                fig_edrp_bloques = go.Figure()

                for c in range(4):
                    fig_edrp_bloques.add_trace(go.Bar(
                        x=df_edrp_nb["bus"].astype(str),
                        y=df_edrp_nb[f"Bloque_{c+1}"],
                        name=f"Bloque {c+1}"
                    ))

                fig_edrp_bloques.update_layout(
                    title="EDRP acumulado por nodo y bloque",
                    xaxis_title="Bus",
                    yaxis_title="EDRP acumulado",
                    barmode="stack",
                    height=380
                )

                st.plotly_chart(fig_edrp_bloques, width="stretch")

    # ══════════════════════════════════════════════════════════
    # Sub-sección E2 — VALOR ESPERADO SOBRE LOS ESCENARIOS
    #
    # El panel E de arriba compara estrategias resueltas en la sesión, y cada
    # una vive en UN escenario Monte Carlo: es la foto de una realización, no
    # el valor esperado. Acá se lee el barrido (medida × N escenarios, que se
    # calcula por consola con run_barrido_medidas.py) y se promedia sobre los
    # escenarios estudiados.
    #
    # No recalcula nada: son los mismos artefactos que consume la pestaña
    # 🧪 Barrido, reusando sus cargadores para que las dos vistas no se
    # desincronicen.
    # ══════════════════════════════════════════════════════════
    st.header("📈 E2 — Valor Esperado sobre los Escenarios")

    import barrido_vista as BV

    _barridos = BV.listar_barridos()
    if not _barridos:
        st.info(
            "No hay barridos calculados. El valor esperado sale de resolver "
            "cada medida sobre varios escenarios, que es un trabajo de consola "
            "(`run_barrido_medidas.py`). Configúralo en la pestaña 🧪 Barrido."
        )
    else:
        _cb1, _cb2 = st.columns([2, 1])
        # Barrido se renderiza antes que E2: adoptar su selección también al
        # actualizar una sesión que antes tenía dos selectores independientes.
        if st.session_state.get("barr_nombre") in _barridos:
            st.session_state["ve_barrido"] = st.session_state["barr_nombre"]
        _nom_b = _cb1.selectbox(
            "Barrido (tanda · variante de aislación)", _barridos,
            format_func=lambda n: n.replace("/", "  ·  ", 1),
            key="ve_barrido",
            on_change=BV.sincronizar_barrido, args=("ve_barrido",),
            help="Cada supuesto de aislación tiene su propio barrido; los de "
                 "supuestos distintos no son comparables entre sí.",
        )
        _incluir_comb = _cb2.checkbox(
            "Incluir combinadas", value=True, key="ve_combinadas",
            help="Agrega las corridas de la fase 3 (varias medidas juntas).",
        )

        _B = BV.cargar_barrido(_nom_b, BV.sello(_nom_b))
        _df_ind = _B["individual"]
        _escs_b = sorted(_df_ind["escenario"].unique()) if not _df_ind.empty else []
        _nh_b = int(_df_ind["horas"].iloc[0]) if "horas" in _df_ind and not _df_ind.empty else 0
        _base_b = BV.cargar_base(_B["variante"], tuple(int(e) for e in _escs_b), _nh_b)

        if _base_b.empty:
            st.warning(
                f"⚠️ Faltan los baselines (etapa E5) de la variante "
                f"`{_B['variante']}` para estos escenarios, así que no se "
                f"puede calcular cuánto evitó cada medida."
            )
        else:
            # `enriquecer` agrega el Δ de cada corrida contra el caso base del
            # MISMO escenario, que es la única resta que significa algo.
            _dfe = BV.enriquecer(_df_ind, _base_b, _B["mip_gap"])
            _piezas = [_dfe.assign(_caso=_dfe["medida"], _tipo="Individual")]

            if _incluir_comb and not _B["combinadas"].empty:
                _dce = BV.enriquecer(_B["combinadas"], _base_b, _B["mip_gap"])
                if "combinacion" in _dce.columns:
                    _piezas.append(_dce.assign(_caso=_dce["combinacion"],
                                               _tipo="Combinada"))
            _todo = pd.concat(_piezas, ignore_index=True)

            # Las corridas "sospechosas" (peor que no hacer nada por más de lo
            # que el MIPGap explica: típicamente cortadas por --time_limit en
            # un incumbente malo) contaminan cualquier promedio en el que
            # entren, así que se excluyen y se informa cuántas fueron.
            _n_sosp = int(_todo.get("sospechosa", pd.Series(dtype=bool)).sum())
            if "sospechosa" in _todo:
                _todo = _todo[~_todo["sospechosa"]]

            from graficos_valor_esperado import seleccionar_corridas, resumir_indicadores
            _todo = seleccionar_corridas(_todo)
            _escs_b = sorted(_todo['escenario'].unique())

            if _todo.empty:
                st.warning("⚠️ No quedó ninguna corrida válida para promediar.")
            else:
                _ve = resumir_indicadores(_todo)

                _ve["Medida"] = [BV._nombre(m) if t == "Individual" else BV._nombre_combo(m)
                                 for t, m in zip(_ve["_tipo"], _ve["_caso"])]
                _ve = _ve.sort_values("d_ens_pct", ascending=False)

                st.caption(
                    f"Promedio sobre **{len(_escs_b)} escenarios** "
                    f"({', '.join(str(int(e) + 1) for e in _escs_b[:12])}"
                    f"{'…' if len(_escs_b) > 12 else ''}) · variante "
                    f"`{_B['variante']}` · horizonte {_nh_b} h. Cada corrida se "
                    f"compara contra el caso base de SU escenario y después se "
                    f"promedian los Δ."
                    + (f"  \n⚠️ Se excluyeron **{_n_sosp}** corrida(s) sospechosa(s) "
                       f"(peor que el caso base por más de lo que explica el "
                       f"MIPGap)." if _n_sosp else "")
                )

                # Valores esperados del CASO BASE sobre los mismos escenarios
                # que quedaron seleccionados. Se toman de `_base_b`, que es la
                # fuente de la que `enriquecer` saca los Δ, y no del promedio
                # de las columnas `*_base` de `_ve`: eso último promedia una
                # vez por medida y, si dos medidas no corrieron exactamente los
                # mismos escenarios, da un caso base que no es el de nadie.
                _base_sel = _base_b[_base_b["escenario"].isin(_escs_b)]
                _base_esp = {c: float(_base_sel[c].mean())
                             for c in ("ens", "saidi", "saifi", "costo_ens")
                             if c in _base_sel.columns}

                _c1, _c2, _c3, _c4 = st.columns(4)
                _c1.metric("Casos promediados", len(_ve))
                _c2.metric("Escenarios", len(_escs_b))
                _c3.metric("Corridas usadas", len(_todo))
                _c4.metric("ENS esperada del caso base",
                           f"{_base_esp.get('ens', float('nan')):,.0f} kWh",
                           help="Sin ninguna medida, promediando los mismos "
                                "escenarios. Es la referencia contra la que se "
                                "miden todos los Δ de las tablas.")

                # ── Confiabilidad ────────────────────────────
                st.markdown("#### 🔢 Indicadores de confiabilidad (valor esperado)")
                _t_conf = _ve.rename(columns={
                    "n_esc": "N° esc.",
                    "saidi": "SAIDI [h/año]",
                    "saifi": "SAIFI [int./año]",
                    "ens": "ENS total [kWh]",
                    "d_saidi": "Δ SAIDI",
                    "d_saidi_pct": "Δ SAIDI %",
                    "d_saifi": "Δ SAIFI",
                    "d_saifi_pct": "Δ SAIFI %",
                    "d_ens": "Δ ENS",
                    "d_ens_pct": "Δ ENS %",
                    "d_ens_pct_std": "σ Δ ENS %",
                })
                _cols_c = ["Medida", "_tipo", "N° esc.", "SAIDI [h/año]",
                           "SAIFI [int./año]", "ENS total [kWh]", "Δ SAIDI",
                           "Δ SAIDI %", "Δ SAIFI", "Δ SAIFI %", "Δ ENS",
                           "Δ ENS %", "σ Δ ENS %"]
                _cols_c = [c for c in _cols_c if c in _t_conf.columns]
                _t_conf = _t_conf[_cols_c].rename(columns={"_tipo": "Tipo"})
                # El caso base como primera fila: la tabla daba los NIVELES de
                # cada medida (SAIDI, SAIFI, ENS) sin decir de qué nivel se
                # parte, así que no se podía leer si 13.150 kWh es mucho o poco
                # sin ir a buscar el dato a otra vista. Los Δ quedan vacíos a
                # propósito: el caso base no mejora nada respecto de sí mismo.
                _fila_base = {"Medida": "Caso base (sin medidas)",
                              "Tipo": "Referencia",
                              "N° esc.": len(_escs_b),
                              "SAIDI [h/año]": _base_esp.get("saidi"),
                              "SAIFI [int./año]": _base_esp.get("saifi"),
                              "ENS total [kWh]": _base_esp.get("ens")}
                _t_conf = pd.concat(
                    [pd.DataFrame([_fila_base]).reindex(columns=_t_conf.columns),
                     _t_conf], ignore_index=True)
                st.dataframe(
                    _t_conf.style
                        .format({"SAIDI [h/año]": "{:.3f}",
                                 "SAIFI [int./año]": "{:.3f}",
                                 "ENS total [kWh]": "{:,.0f}",
                                 "Δ SAIDI": "{:.3f}", "Δ SAIDI %": "{:.1%}",
                                 "Δ SAIFI": "{:.3f}", "Δ SAIFI %": "{:.1%}",
                                 "Δ ENS": "{:,.0f}", "Δ ENS %": "{:.1%}",
                                 "σ Δ ENS %": "{:.1%}"}, na_rep="—"),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "La primera fila es el **caso base** (no hacer nada) sobre "
                    "los mismos escenarios: es el nivel del que parten todas "
                    "las medidas, y por eso sus Δ van vacíos. "
                    "Δ positivo = mejora. **σ Δ ENS %** es la dispersión entre "
                    "escenarios: un promedio alto con σ alta significa que la "
                    "medida sirve mucho en unos escenarios y poco en otros."
                )

                # ── Económicos ───────────────────────────────
                if "cst_costo_total" in _ve.columns:
                    st.markdown("#### 💰 Indicadores económicos (valor esperado)")
                    _t_eco = _ve.rename(columns={
                        "cst_capex_anual": "CAPEX anual [USD]",
                        "cst_opex_anual": "OPEX anual [USD]",
                        "cst_costo_total": "CAPEX+OPEX [USD]",
                        "cst_costo_ens_estr": "Costo de la ENS [USD]",
                        "cst_ahorro_ens": "Ahorro ENS [USD]",
                        "cst_ahorro_neto": "Beneficio neto [USD]",
                        "usd_kwh": "Reducción ENS [USD/MWh]",
                        "usd_saidi": "Reducción SAIDI [USD/(h/año)]",
                        "usd_saifi": "Reducción SAIFI [USD/(int./año)]",
                    }).copy()
                    # `usd_kwh` viene en USD/kWh; se tabula en USD/MWh porque en
                    # kWh los valores quedan bajo 1 y no se distinguen entre sí.
                    if "Reducción ENS [USD/MWh]" in _t_eco.columns:
                        _t_eco["Reducción ENS [USD/MWh]"] *= 1000.0
                    _cols_e = ["Medida", "CAPEX anual [USD]", "OPEX anual [USD]",
                               "CAPEX+OPEX [USD]", "Costo de la ENS [USD]",
                               "Ahorro ENS [USD]",
                               "Beneficio neto [USD]",
                               "Reducción ENS [USD/MWh]",
                               "Reducción SAIDI [USD/(h/año)]",
                               "Reducción SAIFI [USD/(int./año)]"]
                    _cols_e = [c for c in _cols_e if c in _t_eco.columns]
                    _t_eco = _t_eco[_cols_e]
                    # Misma referencia que en la tabla de confiabilidad. Acá el
                    # dato que importa es el costo de la ENS de no hacer nada:
                    # el 'Ahorro ENS' de cada medida es exactamente la resta
                    # contra esta fila. Sus costos son cero porque no se invierte,
                    # y los costos unitarios quedan vacíos (no hay mejora que
                    # dividir), en vez de un 0 que se leería como "gratis".
                    _fila_base_eco = {"Medida": "Caso base (sin medidas)",
                                      "CAPEX anual [USD]": 0.0,
                                      "OPEX anual [USD]": 0.0,
                                      "CAPEX+OPEX [USD]": 0.0,
                                      "Costo de la ENS [USD]": _base_esp.get("costo_ens"),
                                      "Ahorro ENS [USD]": 0.0,
                                      "Beneficio neto [USD]": 0.0}
                    _t_eco = pd.concat(
                        [pd.DataFrame([_fila_base_eco]).reindex(columns=_t_eco.columns),
                         _t_eco], ignore_index=True)
                    st.dataframe(
                        _t_eco.style.format({
                            "CAPEX anual [USD]": "{:,.0f}",
                            "OPEX anual [USD]": "{:,.0f}",
                            "CAPEX+OPEX [USD]": "{:,.0f}",
                            "Costo de la ENS [USD]": "{:,.0f}",
                            "Ahorro ENS [USD]": "{:,.0f}",
                            "Beneficio neto [USD]": "{:,.0f}",
                            "Reducción ENS [USD/MWh]": "{:,.1f}",
                            "Reducción SAIDI [USD/(h/año)]": "{:,.0f}",
                            "Reducción SAIFI [USD/(int./año)]": "{:,.0f}",
                        }, na_rep="—"),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption(
                        "La primera fila es el **caso base**: no invierte nada, "
                        "así que todo su costo es la ENS que deja sin servir, y "
                        "el «Ahorro ENS» de cada medida es la resta contra ese "
                        "número. "
                        "Los costos por unidad de mejora se calculan como "
                        "**E[costo] / E[Δ]**, no como el promedio de los "
                        "cocientes por escenario: promediar un cociente pondera "
                        "de más los escenarios donde la mejora fue chica y da un "
                        "número que no corresponde a ninguna decisión real. "
                        "Quedan en blanco si la mejora no supera el umbral de la curva. "
                        "Un costo unitario negativo indica ahorro neto de costos."
                    )

                from graficos_valor_esperado import render as render_graficos_ve
                render_graficos_ve(_todo, _B, BUSES, bus_zone,
                                   _fig_pantalla, _descargas_figura)

                # ── Gráfico de valor esperado ────────────────
                if "d_ens_pct" in _ve.columns:
                    _ord = _ve.dropna(subset=["d_ens_pct"]).sort_values("d_ens_pct")
                    fig_ve = go.Figure()
                    fig_ve.add_trace(go.Bar(
                        y=_ord["Medida"], x=_ord["d_ens_pct"] * 100,
                        orientation="h",
                        error_x=dict(type="data",
                                     array=(_ord["d_ens_pct_std"] * 100).fillna(0),
                                     visible=True),
                        marker_color=[PALETA[i % len(PALETA)]
                                      for i in range(len(_ord))],
                        text=[f"{v:.1%}" for v in _ord["d_ens_pct"]],
                        textposition="outside",
                    ))
                    fig_ve.update_layout(
                        width=1600, height=max(500, 46 * len(_ord)),
                        xaxis_title="Reducción esperada de ENS [% del caso base]",
                        yaxis_title="", font=dict(size=26),
                        xaxis=dict(title_font=dict(size=26), tickfont=dict(size=22)),
                        yaxis=dict(tickfont=dict(size=22)),
                        margin=dict(l=90, r=60, t=40, b=70),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                    )
                    st.plotly_chart(
                        _fig_pantalla(fig_ve,
                                      height=max(320, 30 * len(_ord)),
                                      font=13, text_size=11,
                                      margin=dict(l=60, r=40, t=30, b=50)),
                        use_container_width=True, key="ve_barras_ens",
                    )
                    st.caption(
                        "Barras de error = ±1σ entre escenarios."
                    )
                    _descargas_figura(fig_ve, "valor_esperado_ens",
                                      key="ve_ens",
                                      etiqueta="el gráfico de valor esperado")

                # ── Descarga de la tabla ─────────────────────
                _buf_ve = io.BytesIO()
                with pd.ExcelWriter(_buf_ve, engine="openpyxl") as _wr_ve:
                    _ve.drop(columns=["_caso"]).to_excel(
                        _wr_ve, sheet_name="Valor_esperado", index=False)
                _buf_ve.seek(0)
                _dv1, _dv2 = st.columns(2)
                _dv1.download_button(
                    "📥 Descargar valor esperado (Excel)", data=_buf_ve,
                    file_name=f"valor_esperado_{_B['variante']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="ve_download_xlsx",
                )
                _dv2.download_button(
                    "📥 Descargar valor esperado (CSV)",
                    data=_ve.drop(columns=["_caso"]).to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"valor_esperado_{_B['variante']}.csv",
                    mime="text/csv", key="ve_download_csv",
                )

    # ── Sub-sección F: Análisis de sinergias ─────────────────
    st.header("🔗 F — Análisis de Sinergias entre Medidas")

    st.caption(
        "Diseño factorial 2×2: compara cada medida por separado contra la "
        "combinación de ambas, sobre los mismos escenarios Monte Carlo."
    )

    def _fmt_num(v):
        """Formato compacto e independiente de la escala del indicador."""
        if isinstance(v, str):
            return v
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        try:
            return f"{float(v):,.4g}"
        except (TypeError, ValueError):
            return str(v)

    def _tabla_fmt(df):
        return df.copy().map(_fmt_num) if hasattr(df, "map") else df.copy().applymap(_fmt_num)

    if len(historial) < 3:
        st.info(
            "Se necesitan al menos **3 estrategias** en el historial para el "
            "análisis: la medida A sola, la medida B sola y ambas en conjunto "
            "(el caso base ya está disponible). Ejemplo: *DG eólico*, "
            "*ESS 5h* y *DG eólico + ESS 5h*."
        )
    else:
        labels_hist = list(historial.keys())

        col_a, col_b, col_ab = st.columns(3)
        with col_a:
            lbl_A = st.selectbox("🅰️ Medida A (sola)", labels_hist,
                                 index=0, key="sin_lbl_a")
        with col_b:
            idx_b = 1 if len(labels_hist) > 1 else 0
            lbl_B = st.selectbox("🅱️ Medida B (sola)", labels_hist,
                                 index=idx_b, key="sin_lbl_b")
        with col_ab:
            sugerida = sugerir_combinada(historial, lbl_A, lbl_B)
            idx_ab = labels_hist.index(sugerida) if sugerida in labels_hist else (
                2 if len(labels_hist) > 2 else 0)
            lbl_AB = st.selectbox("🅰️➕🅱️ Combinada (A + B)", labels_hist,
                                  index=idx_ab, key="sin_lbl_ab")
            if sugerida:
                st.caption(f"Sugerida automáticamente: **{sugerida}**")

        cfg_A = historial[lbl_A].get("config", {})
        cfg_B = historial[lbl_B].get("config", {})
        cfg_AB = historial[lbl_AB].get("config", {})

        st.markdown(
            f"- **A** = {lbl_A} → _{describir_medidas(cfg_A)}_\n"
            f"- **B** = {lbl_B} → _{describir_medidas(cfg_B)}_\n"
            f"- **AB** = {lbl_AB} → _{describir_medidas(cfg_AB)}_"
        )

        ok_diseno, avisos = validar_diseno_factorial(cfg_A, cfg_B, cfg_AB)
        for aviso in avisos:
            st.warning(f"⚠️ {aviso}")

        if len({lbl_A, lbl_B, lbl_AB}) < 3:
            st.error("❌ Selecciona tres estrategias distintas.")
        else:
            res_A = historial[lbl_A]["results"]
            res_B = historial[lbl_B]["results"]
            res_AB = historial[lbl_AB]["results"]

            claves_comunes = claves_escenarios_comunes(
                results_sin, res_A, res_B, res_AB
            )

            if not claves_comunes:
                st.error(
                    "❌ No hay escenarios con resultados válidos en las cuatro "
                    "corridas. El análisis pareado exige los mismos escenarios "
                    "Monte Carlo en base, A, B y AB."
                )
            else:
                if len(claves_comunes) < numesc:
                    st.info(
                        f"ℹ️ Se usan {len(claves_comunes)} de {numesc} escenarios "
                        "(los presentes y válidos en las cuatro corridas)."
                    )

                # ── Series por escenario e interacción ────────
                series_ind = {}
                resumenes = {}
                for nombre_ind, fn_ind in INDICADORES_SIN.items():
                    x_base = serie_indicador(results_sin, fn_ind, claves_comunes)
                    x_a = serie_indicador(res_A, fn_ind, claves_comunes)
                    x_b = serie_indicador(res_B, fn_ind, claves_comunes)
                    x_ab = serie_indicador(res_AB, fn_ind, claves_comunes)
                    series_ind[nombre_ind] = (x_base, x_a, x_b, x_ab)
                    resumenes[nombre_ind] = resumen_interaccion(
                        x_base, x_a, x_b, x_ab
                    )

                # ── Tabla factorial ──────────────────────────
                st.markdown("### 📐 Tabla factorial 2×2")
                st.caption(
                    "Δ = mejora respecto del caso base (menos es mejor). "
                    "I = Δ_AB − (Δ_A + Δ_B). S = Δ_AB / (Δ_A + Δ_B)."
                )
                df_fact = tabla_factorial(resumenes, lbl_A, lbl_B, lbl_AB)
                st.dataframe(_tabla_fmt(df_fact), use_container_width=True)

                # ── Interacción con IC bootstrap ─────────────
                st.markdown("### 🎯 Interacción y significancia estadística")
                st.caption(
                    "IC bootstrap percentil (5.000 remuestreos) sobre la "
                    "interacción calculada escenario a escenario. Al usar los "
                    "mismos eventos de falla en las cuatro corridas, el "
                    "contraste es pareado y de baja varianza. Si el IC 95% "
                    "incluye el cero, no hay evidencia de sinergia."
                )
                df_ic = tabla_interaccion_ic(resumenes)
                st.dataframe(_tabla_fmt(df_ic), use_container_width=True)

                for nombre_ind, r in resumenes.items():
                    st.markdown(
                        f"- **{nombre_ind}** — {r['etiqueta']}: {r['lectura']}"
                    )

                st.caption(
                    "El contraste multiplicativo (columna *I multiplicativo*) "
                    "usa la referencia 1 − (1−r_A)(1−r_B) y es el indicado "
                    "cuando las reducciones son grandes: el benchmark aditivo "
                    "satura al acercarse el indicador a cero."
                )

                # ── Figuras ──────────────────────────────────
                st.markdown("### 📊 Descomposición del efecto conjunto")
                ind_sel = st.selectbox(
                    "Indicador a graficar",
                    list(INDICADORES_SIN.keys()),
                    key="sin_ind_graf",
                )
                r_sel = resumenes[ind_sel]

                fig_wf = figura_waterfall(r_sel, ind_sel, lbl_A, lbl_B)
                if fig_wf is not None:
                    st.plotly_chart(fig_wf, use_container_width=True)
                    st.caption(
                        "Cascada: caso base → efecto de A → efecto de B → "
                        "interacción → combinada. Una barra de interacción "
                        "hacia abajo indica sinergia."
                    )

                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    fig_bar = figura_barras_factorial(
                        r_sel, ind_sel, lbl_A, lbl_B, lbl_AB
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                with col_f2:
                    fig_hist = figura_distribucion_interaccion(r_sel, ind_sel)
                    if fig_hist is not None:
                        st.plotly_chart(fig_hist, use_container_width=True)

                # ── Atribución económica ─────────────────────
                st.markdown("### 💰 Atribución económica del beneficio conjunto")

                def _eco_seguro(res, cfg):
                    """El análisis físico no debe caerse si falla el costeo."""
                    try:
                        return _extraer_costos_objetivo(res, results_sin, cfg, data)
                    except Exception as err:
                        st.warning(f"⚠️ No se pudo calcular el costeo: {err}")
                        return {}

                eco_A = _eco_seguro(res_A, cfg_A)
                eco_B = _eco_seguro(res_B, cfg_B)
                eco_AB = _eco_seguro(res_AB, cfg_AB)

                df_eco, inter_benef = tabla_economica(
                    eco_A, eco_B, eco_AB, lbl_A, lbl_B, lbl_AB
                )
                st.dataframe(_tabla_fmt(df_eco.set_index("Medida")),
                             use_container_width=True)

                col_e1, col_e2 = st.columns(2)
                col_e1.metric(
                    "Interacción en beneficio",
                    _fmt_num(inter_benef),
                    help="B_AB − (B_A + B_B). Positivo: la cartera conjunta "
                         "rinde más que la suma de las medidas por separado.",
                )
                costo_extra = (
                    float(eco_AB.get("costo_total", 0.0))
                    - float(eco_A.get("costo_total", 0.0))
                    - float(eco_B.get("costo_total", 0.0))
                )
                col_e2.metric(
                    "Diferencia de costo AB vs. A+B",
                    _fmt_num(costo_extra),
                    help="Costo total de la cartera conjunta menos la suma de "
                         "los costos individuales.",
                )

                st.caption(
                    "El reparto de Shapley divide el beneficio conjunto de "
                    "forma consistente (φ_A + φ_B = B_AB), a diferencia de las "
                    "contribuciones marginales, que no suman el total."
                )
                st.warning(
                    "⚠️ **Comparación justa:** la cartera A+B siempre invierte "
                    "más que A o B por separado, por lo que superarlas en los "
                    "indicadores es casi trivial. Para un argumento de "
                    "planificación, contrasta a **igual presupuesto** o "
                    "compara directamente el **beneficio neto**."
                )

                # ── Exportar detalle por escenario ───────────
                df_esc = df_por_escenario(claves_comunes, series_ind)
                with st.expander("🔍 Detalle escenario por escenario"):
                    st.dataframe(df_esc, use_container_width=True)
                st.download_button(
                    "⬇️ Descargar análisis de sinergias (CSV)",
                    data=df_esc.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"sinergias_{lbl_A}_{lbl_B}.csv".replace(" ", "_"),
                    mime="text/csv",
                    key="sin_download_csv",
                )

    # ── Exportación Excel ─────────────────────────────────────
    st.header("📥 Exportar Reporte")

    def _generar_excel():
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment
            from openpyxl.utils import get_column_letter
        except ImportError:
            st.error("openpyxl no está instalado. Instálalo con: pip install openpyxl")
            return None

        wb = openpyxl.Workbook()

        HDR_FONT = Font(name="Calibri", bold=True, color="FFFFFF")
        HDR_FILL = PatternFill("solid", fgColor="1F3864")
        CENTER    = Alignment(horizontal="center", vertical="center")
        LEFT      = Alignment(horizontal="left",   vertical="center", indent=1)

        def _estilo_hdr(ws, row, col_start, col_end):
            for c in range(col_start, col_end + 1):
                cell = ws.cell(row=row, column=c)
                cell.font      = HDR_FONT
                cell.fill      = HDR_FILL
                cell.alignment = CENTER

        def _autowidth(ws):
            for col in ws.columns:
                max_len = 0
                col_letter = get_column_letter(col[0].column)
                for cell in col:
                    if cell.value:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 40)

        # ── Hoja KPIs ──────────────────────────────────────────
        ws_kpi = wb.active
        ws_kpi.title = "KPIs"
        headers_kpi = ["Métrica", "Red completa"] + [f"Zona {z}" for z in [1,2,3]]
        for ci, h in enumerate(headers_kpi, 1):
            ws_kpi.cell(row=1, column=ci, value=h)
        _estilo_hdr(ws_kpi, 1, 1, len(headers_kpi))

        metricas_kpi = []
        for metric_name, fn_kpi in [
            ("SAIDI [h/año]",      lambda buses: fn_SAIDI(res_kpi, P_D_base, buses, horas)),
            ("SAIFI [int./año]",   lambda buses: fn_SAIFI(res_kpi, P_D_base, buses, horas)),
            ("ENS total [kWh]",        lambda buses: sum(sum(res_kpi[h-1].get("P_LC",{}).get(b,0)
                                                    for b in buses) for h in horas if h-1 < len(res_kpi)) if res_kpi else 0),
        ]:
            row_data = [metric_name,
                        round(fn_kpi(BUSES), 3)]
            for z in [1, 2, 3]:
                buses_z = [b for b in BUSES if bus_zone.get(b, 0) == z]
                row_data.append(round(fn_kpi(buses_z), 3))
            metricas_kpi.append(row_data)

        ens_total_kpi = sum(sum(res_kpi[h-1].get("P_LC",{}).get(b,0) for b in BUSES) for h in horas if h-1 < len(res_kpi)) if res_kpi else 0
        _esc_xls = min(esc_sel - 1, demand_factor.shape[1] - 1)
        pd_total_kpi = sum(
            P_D_base.get(b, 0) * demand_factor[h-1, _esc_xls, bi]
            for h in horas for bi, b in enumerate(BUSES)
        )
        pct_e_kpi = round((1 - ens_total_kpi / pd_total_kpi)*100, 2) if pd_total_kpi > 0 else 0
        tasa_ic       = round(np.mean(_tasa_interconexion_por_hora(res_kpi)), 2) if res_kpi else 100.0

        metricas_kpi += [
            ["% Energía distribuida", pct_e_kpi, "—", "—", "—"],
            ["Tasa interconexión [%]", tasa_ic,  "—", "—", "—"],
        ]
        for ri, row_data in enumerate(metricas_kpi, 2):
            for ci, val in enumerate(row_data, 1):
                ws_kpi.cell(row=ri, column=ci, value=val)
        _autowidth(ws_kpi)

        # ── Hoja SAIDI_SAIFI_mensual ──────────────────────────
        ws_mens = wb.create_sheet("SAIDI_SAIFI_mensual")
        if N_HORAS >= 720:
            headers_m = ["Mes", "SAIDI_base", "SAIDI_estrategia"]
            for ci, h in enumerate(headers_m, 1):
                ws_mens.cell(row=1, column=ci, value=h)
            _estilo_hdr(ws_mens, 1, 1, len(headers_m))

            saidi_b_m = []
            saidi_e_m = []
            h_start   = 0
            for mi, mh in enumerate(MESES_H):
                h_end   = min(h_start + mh, N_HORAS)
                subset  = {i: res_kpi[h_start + i] for i in range(h_end - h_start)
                           if res_kpi and h_start + i < len(res_kpi)}
                saidi_b_m.append(round(fn_SAIDI(subset, P_D_base, BUSES, range(1, len(subset)+1)), 3))
                if results_activos:
                    valid_act4 = [v for v in _results_to_list(results_activos) if v]
                    res_act4   = valid_act4[_pos_esc(esc_sel - 1, len(valid_act4))] if valid_act4 else []
                    subset_a   = {i: res_act4[h_start + i] for i in range(h_end - h_start)
                                  if res_act4 and h_start + i < len(res_act4)}
                    saidi_e_m.append(round(fn_SAIDI(subset_a, P_D_base, BUSES, range(1, len(subset_a)+1)), 3))
                else:
                    saidi_e_m.append("—")
                h_start = h_end
                if h_start >= N_HORAS:
                    break

            for ri, (mes, sb, se) in enumerate(zip(MESES_NOM[:len(saidi_b_m)], saidi_b_m, saidi_e_m), 2):
                ws_mens.cell(row=ri, column=1, value=mes)
                ws_mens.cell(row=ri, column=2, value=sb)
                ws_mens.cell(row=ri, column=3, value=se)
        else:
            ws_mens.cell(row=1, column=1, value="Datos insuficientes para agregación mensual (N_HORAS < 720)")
        _autowidth(ws_mens)

        # ── Hoja Medidas ───────────────────────────────────────
        ws_med = wb.create_sheet("Medidas")
        if results_activos:
            cap_dg_xls = _extraer_cap_dg(results_activos)
            headers_med = ["Gen", "Bus", "Zona", "CAP_sol [kW]", "CAP_eol [kW]", "CAP_fos [kW]"]
            for ci, h in enumerate(headers_med, 1):
                ws_med.cell(row=1, column=ci, value=h)
            _estilo_hdr(ws_med, 1, 1, len(headers_med))
            for ri, g in enumerate(GENS, 2):
                ws_med.cell(row=ri, column=1, value=g)
                ws_med.cell(row=ri, column=2, value=cap_dg_xls[g]["bus"])
                ws_med.cell(row=ri, column=3, value=cap_dg_xls[g]["zona"])
                ws_med.cell(row=ri, column=4, value=round(cap_dg_xls[g]["sol"][0], 2))
                ws_med.cell(row=ri, column=5, value=round(cap_dg_xls[g]["eol"][0], 2))
                ws_med.cell(row=ri, column=6, value=round(cap_dg_xls[g]["fos"][0], 2))
        else:
            ws_med.cell(row=1, column=1, value="Sin resultados de estrategia disponibles.")
        _autowidth(ws_med)

        # ── Hoja Costos ────────────────────────────────────────
        ws_cost = wb.create_sheet("Costos")
        if results_activos:
            costos_xls = _extraer_costos_objetivo(results_activos, results_sin, config_activo, data)
            items_cost = [
                ("CAPEX anualizado [USD]",  costos_xls["capex_anual"]),
                ("OPEX anual [USD]",        costos_xls["opex_anual"]),
                ("Costo total [USD]",       costos_xls["costo_total"]),
                ("Ahorro ENS [USD]",        costos_xls["ahorro_ens"]),
                ("Beneficio neto [USD]",    costos_xls["beneficio"]),
            ]
            payback_xls = (costos_xls["capex_anual"] /
                           (costos_xls["ahorro_ens"] - costos_xls["opex_anual"])
                           if (costos_xls["ahorro_ens"] - costos_xls["opex_anual"]) > 0 else None)
            items_cost.append(("Período recuperación [años]",
                                round(payback_xls, 2) if payback_xls else "No recuperable"))

            ws_cost.cell(row=1, column=1, value="Métrica")
            ws_cost.cell(row=1, column=2, value="Valor")
            _estilo_hdr(ws_cost, 1, 1, 2)
            for ri, (metric, val) in enumerate(items_cost, 2):
                ws_cost.cell(row=ri, column=1, value=metric)
                ws_cost.cell(row=ri, column=2, value=round(val, 2) if isinstance(val, float) else val)
        else:
            ws_cost.cell(row=1, column=1, value="Sin resultados de estrategia disponibles.")
        _autowidth(ws_cost)

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return buf

    historial_check = st.session_state.get("historial", {})
    if historial_check:
        buf_excel = _generar_excel()
        if buf_excel:
            st.download_button(
                "📥 Descargar reporte Excel",
                data=buf_excel,
                file_name="reporte_resiliencia.xlsx",
                mime="application/vnd.ms-excel"
            )
    else:
        st.warning("⚠️ Ejecuta al menos una estrategia de optimización para habilitar la exportación.")
