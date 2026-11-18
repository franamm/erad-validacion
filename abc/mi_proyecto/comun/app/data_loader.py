# ============================================================
# data_loader.py — CARGADOR PURO de los artefactos del pipeline
# ============================================================
# Este módulo ya NO calcula nada: solo lee lo que produjo
# pipeline.py y expone los nombres que consumen app_v7, los
# helpers_*, indicadores y grafo.
#
# Lo que ANTES hacía acá y ahora vive en pipeline.py:
#   - cargar app.pkl (744 MB) ................. E0/E1 (artefactos, 0.2 MB)
#   - recargar FaultlinMC.npy ................. E2
#   - leer el .nc de viento y adelgazar las
#     matrices veg/ref/ug en cada arranque .... E2 (precalculado)
#   - recalcular alphaDGsolar ................. E1
#   - definir todos los costos ................ E3
#   - baseline del caso base .................. E4 (sin fallas) / E5 (con fallas)
#
# Si falta algún artefacto, este módulo falla de inmediato con un
# mensaje que dice qué etapa correr. Antes, un artefacto ausente
# se degradaba en silencio (p. ej. el .nc faltante cambiaba la
# calibración de refuerzo/vegetación sin avisar).
# ============================================================

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import casos
import pipeline as P

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Dashboard Red de Distribución",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────
# CARGA DE ARTEFACTOS
# ─────────────────────────────────────────────────────────────

def _exigir(nombre, etapa):
    if not (P.ART_DIR / nombre).exists():
        # El comando lleva --artefactos: la app puede estar abierta en un caso
        # distinto del de validación (ERAD_ARTEFACTOS), y sin esa parte el
        # mensaje manda a generar la etapa en la carpeta equivocada.
        caso = f" --artefactos {P.ART_DIR.name}" if not casos.es_validacion(P.ART_DIR) else ""
        raise FileNotFoundError(
            f"Falta el artefacto '{nombre}' del caso '{P.ART_DIR.name}'. "
            f"Generarlo con:\n"
            f"    python pipeline.py{caso} --etapas {etapa}\n"
            f"(ver estado con: python pipeline.py{caso} --estado)")


@st.cache_resource
def _cargar_artefactos():
    # El E4 se puede elegir por etiqueta (ERAD_SIN_FALLAS=LIM-0.30-ALIM). Lo
    # fija run_barrido_medidas con su --sin-fallas, para que el desglose de
    # costos que calcula helpers_inversion use el MISMO caso sin fallas que el
    # objetivo del MILP, y para que un caso que solo tiene el E4 etiquetado
    # (lo normal en una red nueva) no pierda ese desglose por no tener el
    # sin_fallas.pkl histórico. Sin la variable, todo sigue como siempre.
    # Sin la variable y sin el histórico, se abre con el E4 etiquetado que haya
    # (ver casos.etiqueta_e4): una red nueva no tiene sin_fallas.pkl a secas.
    nombre_e4 = P.ruta_sin_fallas(casos.etiqueta_e4(P.ART_DIR)).name
    for nombre, etapa in (("red.pkl", "E0"), ("perfiles.npz", "E1"),
                          ("fallas.npz", "E2"), ("costos.pkl", "E3"),
                          (nombre_e4, "E4")):
        _exigir(nombre, etapa)
    red = P._cargar("red.pkl")
    costos = P._cargar("costos.pkl")
    fallas = np.load(P.ART_DIR / "fallas.npz")
    sin_fallas = P._cargar(nombre_e4)
    return red, costos, {k: fallas[k] for k in fallas.files}, sin_fallas


_red, _costos, _fallas, _sin_fallas = _cargar_artefactos()

# ── Red y conjuntos ──────────────────────────────────────────
net = _red["net"]
BUSES = _red["BUSES"]
LINES = _red["LINES"]
GENS = _red["GENS"]
ESS = _red["ESS"]
S = _red["S"]
bus_zone = _red["bus_zone"]
P_D_base = _red["P_D_base"]
N_eventos = _red["N_eventos"]
numesc = _red["numesc"]
sw_cand_list = _costos["sw_cand_list"]
# Enlaces candidatos a construir (medida `inv_tie`). Por defecto son las
# mismas tie-lines que usa la reconfiguración topológica.
tie_cand_list = list(_costos.get("tie_candidatas", _red.get("tie_indices", LINES[-3:])))
# Enlaces EXISTENTES: los que la reconfiguración topológica puede cerrar y que
# `construir_faultlin.py --enlaces-inmunes` excluye del sorteo de fallas.
# Se leen de red.pkl en vez de asumir "las últimas tres líneas". Hoy las dos
# cosas coinciden, pero si la red cambia el supuesto se rompe en silencio y
# el dibujo marcaría como enlace una línea que el modelo trata como normal.
# Lista vacía = red sin enlaces; el respaldo es solo para un red.pkl sin la clave.
tie_indices = list(_red["tie_indices"] if _red.get("tie_indices") is not None
                   else LINES[-3:])
buses_criticos = _red["buses_criticos"]
longitud_km_linea = _costos["longitud_km_linea"]

# ── Fallas ───────────────────────────────────────────────────
faultlinMC = _fallas["faultlinMC"]
repairMC = _fallas["repairMC"]
N_HORAS = int(faultlinMC.shape[0])
horas = range(1, N_HORAS + 1)

# ── Perfiles ─────────────────────────────────────────────────
# Vistas de solo lectura (np.broadcast_to): la forma es
# (horas, esc, bus) como esperan los optimizadores, pero NO se
# materializan los 105 MB por arreglo — el perfil real es de 8760
# valores. Ahorra ~315 MB de RAM respecto de la versión anterior.
demand_factor, alphaDGsolar, alphaDGeolica = P.expandir_perfiles(
    numesc, len(BUSES), copiar=False)

# ── Escenario del estudio congelado ──────────────────────────
_meta = json.loads((P.ART_DIR / "meta.json").read_text(encoding="utf-8"))
esc_esp = int(_meta.get("esc_congelado_estudio", 3))
esc_esp_ui = max(1, min(esc_esp + 1, numesc))     # 1-indexed para la UI

# ── Parámetros económicos ────────────────────────────────────
r_econ, n_econ = _costos["r"], _costos["años"]
frp = _costos["frp"]
frp_movil = _costos["frp_movil"]
C_sub = _costos["C_sub"]
VoLL = _costos["VoLL"]
VoLL_local = dict(VoLL)
for _b in buses_criticos:
    if _b in VoLL_local:
        VoLL_local[_b] = VoLL_local[_b] * 100

# ── Tope de capacidad de DG del caso sin fallas ──────────────
# El CAPEX/OPEX de DG sin fallas que E4 entrega al objetivo depende de cuánta
# capacidad se dejó instalar. Si la app resuelve con un tope distinto al que
# usó E4, el crédito de DG del objetivo queda referido a otra capacidad
# máxima: no da error, solo deja el resultado mal en silencio. Por eso el
# tope se toma del propio artefacto en vez de fijarlo aparte.
# None = artefacto anterior al registro de lim_dg; ahí manda el default de
# optimization_6_oficial y la app lo advierte.
lim_dg_e4 = (_sin_fallas.get("_meta") or {}).get("lim_dg")

C_incentivo_edrp = _costos["EDRP_incentivo"]
h_ess_4h = _costos["h_ess_4h"]
h_ess_5h = _costos["h_ess_5h"]
h_ess_6h = _costos["h_ess_6h"]

C_inv_DG_sol, C_op_DG_sol = _costos["DG_sol"]
C_inv_DG_eol, C_op_DG_eol = _costos["DG_eol"]
C_inv_DG_fos, C_op_DG_fos, C_comb_DG_fos = _costos["DG_fos"]
C_inv_ESS_4h, C_op_ESS_4h = _costos["ESS 4h"]
C_inv_ESS_5h, C_op_ESS_5h = _costos["ESS 5h"]
C_inv_ESS_6h, C_op_ESS_6h = _costos["ESS 6h"]
C_inv_MEG, C_op_MEG, C_comb_MEG = _costos["MEG"]
C_inv_mess, C_op_mess, C_tra_mess = _costos["MESS"]
C_inv_SWITCH, C_op_SWITCH = _costos["SWITCH"]
C_inv_RCS, C_op_RCS = _costos["RCS"]
C_inv_MS, C_op_MS = _costos["MS"]
# Enlaces nuevos (medida `inv_tie`). .get con default para no romper
# artefactos costos.pkl generados antes de que existiera la medida.
C_inv_TIE, C_op_TIE = _costos.get("TIE", [0.0, 0.0])
C_inv_veg, C_op_veg = _costos["VEG"][0], _costos["VEG"][1]
C_inv_ref, C_op_ref = _costos["REF"]
C_inv_ug, C_op_ug = _costos["UG"]


# ─────────────────────────────────────────────────────────────
# BASELINE POR ESCENARIO
# ─────────────────────────────────────────────────────────────
# Los 100 baselines juntos serían ~10 GB en memoria si se
# rehidratan a la vez, así que se carga solo el del escenario
# pedido. st.cache_data mantiene en caché los últimos usados.

ESCENARIOS_CON_BASELINE = set(P.escenarios_disponibles())


@st.cache_data
def escenarios_con_baseline(variante: str = "base"):
    """Escenarios que tienen baseline calculado para ese supuesto de
    aislación. Cada variante lleva su propio juego de baselines: mezclarlos
    haría que el 'ahorro de ENS' se calcule contra otra física."""
    return set(P.escenarios_disponibles(variante))


@st.cache_data
def variantes_con_baseline():
    return P.variantes_disponibles()


@st.cache_data
def escenarios_muestra():
    """Muestra estratificada PROPUESTA (0-idx) + metadatos del muestreo.
    No es necesariamente la que se corrió: para eso, `escenarios_barrido`."""
    return P.escenarios_muestra()


@st.cache_data
def escenarios_barrido(tanda, _sello=0.0):
    """Escenarios que esa tanda del barrido resolvió de verdad. Es el
    conjunto contra el que tiene que agregarse cualquier mapa de fallas que
    se compare con sus resultados por medida."""
    return P.escenarios_barrido(tanda)


@st.cache_data(show_spinner="Cargando caso base del escenario...")
def baseline_esc(esc_0idx: int, variante: str = "base"):
    """Baseline (caso sin estrategias, con fallas) del escenario dado,
    en el formato {esc: [dict por hora]} que consumen los helpers.
    Devuelve {} si ese escenario no tiene baseline calculado."""
    if esc_0idx not in escenarios_con_baseline(variante):
        return {}
    return {esc_0idx: P.cargar_baseline(esc_0idx, variante)}


@st.cache_data
def p_sub_total_esc(esc_0idx: int, variante: str = "base"):
    """ΣP_SUB del caso base para ese escenario — es el punto de
    comparación del término de ahorro de subestación del objetivo."""
    if esc_0idx not in escenarios_con_baseline(variante):
        return 0.0
    z = np.load(P.base_dir(variante) / f"esc_{esc_0idx:03d}.npz")
    return float(z["P_SUB"].sum())


# Baseline del escenario congelado: se mantiene expuesto con el
# nombre histórico para no romper a quien lo importe, pero la app
# usa baseline_esc(esc_seleccionado). NO usar este para comparar
# contra un escenario distinto del congelado.
results_sin = baseline_esc(esc_esp)
SAIDI_sin = {}


def _norm(d):
    """Re-indexa cualquier dict a 0-indexed según orden de claves."""
    if not d:
        return {}
    return {i: d[k] for i, k in enumerate(sorted(d.keys()))}


# ─────────────────────────────────────────────────────────────
# DICT `data` PARA LOS OPTIMIZADORES
# ─────────────────────────────────────────────────────────────

data = dict(_costos)
data.update({
    "net": net,
    "bus_zone": bus_zone,
    "P_D_base": P_D_base,
    "N_eventos": N_eventos,
    "numesc": numesc,
    "esc_esp": esc_esp,
    "horas": horas,
    "demand_factor": demand_factor,
    "alphaDGsolar": alphaDGsolar,
    "alphaDGeolica": alphaDGeolica,
    "faultlinMC": faultlinMC,
    "repairMC": repairMC,
    "faultlinMC_veg": _fallas["faultlinMC_veg"],
    "faultlinMC_ref": _fallas["faultlinMC_ref"],
    "faultlinMC_ug": _fallas["faultlinMC_ug"],
    "faultlinMC_veg_disponible": True,
    # Baseline sin fallas (E4) — constantes del objetivo de optimization_3
    "P_SUB_tot_sin_fallas": _sin_fallas["P_SUB_tot_sin_fallas"],
    "CAPEX_DG_sol_sin_fallas": _sin_fallas["CAPEX_DG_sol_sin_fallas"],
    "OPEX_DG_sol_sin_fallas": _sin_fallas["OPEX_DG_sol_sin_fallas"],
    "CAPEX_DG_eol_sin_fallas": _sin_fallas["CAPEX_DG_eol_sin_fallas"],
    "OPEX_DG_eol_sin_fallas": _sin_fallas["OPEX_DG_eol_sin_fallas"],
    # ΣP_SUB sin fallas CON cada tecnología. No entra al MILP: permite a
    # helpers_inversion separar `OPEX_DG_<tec>_sin_fallas` en OPEX puro y
    # crédito de subestación, que E4 guarda sumados en un solo escalar.
    "P_SUB_sf_sol": float(_sin_fallas["sin_fallas_DG_sol"]["P_SUB"].sum()),
    "P_SUB_sf_eol": float(_sin_fallas["sin_fallas_DG_eol"]["P_SUB"].sum()),
    # Las mismas series HORA A HORA, para restringir el término de subestación
    # a las horas con falla (la fracción atribuible a resiliencia).
    "P_SUB_sf_h":     _sin_fallas["sin_fallas"]["P_SUB"].sum(axis=1),
    "P_SUB_sf_sol_h": _sin_fallas["sin_fallas_DG_sol"]["P_SUB"].sum(axis=1),
    "P_SUB_sf_eol_h": _sin_fallas["sin_fallas_DG_eol"]["P_SUB"].sum(axis=1),
    # Baseline con fallas, POR ESCENARIO. optimization_3 elige el que
    # corresponde a config["esc_debug"]; antes era un escalar del
    # escenario congelado que se aplicaba a cualquier escenario.
    "P_SUB_tot_por_esc": {e: p_sub_total_esc(e) for e in sorted(ESCENARIOS_CON_BASELINE)},
})
# Ramales con fusible (ramales.py): las mismas claves que arma
# pipeline.construir_data, para que la app y la consola resuelvan lo mismo.
import ramales as _RAM
data.update(_RAM.datos_para_optimizador(_red, _fallas))
# Compatibilidad: valor del escenario congelado. optimization_3 lo usa
# solo como respaldo si el escenario pedido no está en P_SUB_tot_por_esc.
data["P_SUB_tot"] = data["P_SUB_tot_por_esc"].get(esc_esp, 0.0)
