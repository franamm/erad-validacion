"""
helpers_sinergias.py — Análisis factorial de interacción entre dos medidas.

Responde a la pregunta: cuando dos medidas (p. ej. DG eólico y ESS 5h) se
aplican por separado y luego en conjunto, ¿el efecto combinado es mayor
(sinergia), igual (aditividad) o menor (solapamiento) que la suma de los
efectos individuales?

Diseño factorial 2x2 sobre los mismos escenarios Monte Carlo:

    ∅   : caso base, sin medidas
    A   : solo la medida A
    B   : solo la medida B
    AB  : ambas medidas simultáneamente

Para un indicador X donde MENOS ES MEJOR (ENS, SAIDI, SAIFI) se define la
mejora respecto del caso base:

    Δ_A  = X_∅ − X_A        Δ_B  = X_∅ − X_B        Δ_AB = X_∅ − X_AB

y el término de interacción:

    I = Δ_AB − (Δ_A + Δ_B) = X_A + X_B − X_∅ − X_AB

    I > 0  →  sinergia      (superaditivo)
    I < 0  →  solapamiento  (subaditivo, retornos decrecientes)
    I ≈ 0  →  medidas aditivas / independientes

Todas las funciones son puras (numpy/pandas/plotly, sin Streamlit) para
poder reutilizarlas en notebooks o en la generación de figuras de la tesis.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data_loader import BUSES, horas, P_D_base
from indicadores import fn_SAIDI, fn_SAIFI, fn_ENS_total
from helpers_operativos import _results_to_list


# ─────────────────────────────────────────────────────────────
# Indicadores disponibles para el análisis
# ─────────────────────────────────────────────────────────────

def _ens(res):
    return fn_ENS_total(res, BUSES, horas)


def _saidi(res):
    return fn_SAIDI(res, P_D_base, BUSES, horas)


def _saifi(res):
    return fn_SAIFI(res, P_D_base, BUSES, horas)


INDICADORES = {
    "ENS [kWh]":            _ens,
    "SAIDI [h/cliente]":    _saidi,
    "SAIFI [int/cliente]":  _saifi,
}


# Claves de medidas en el diccionario `config` de cada estrategia del historial.
CLAVES_MEDIDAS = [
    "dg_sol", "dg_eol", "dg_fos",
    "ess_4h", "ess_5h", "ess_6h",
    "meg", "mess",
    "inv_switch_rcs", "inv_switch_ms",
    "reconfig_topo", "inv_tie", "prior_criticos", "edrp",
    "veg_mgmt", "refuerzo", "soterramiento",
]

NOMBRE_MEDIDA = {
    "dg_sol": "DG solar",
    "dg_eol": "DG eólico",
    "dg_fos": "DG fósil",
    "ess_4h": "ESS 4h",
    "ess_5h": "ESS 5h",
    "ess_6h": "ESS 6h",
    "meg": "MEG",
    "mess": "MESS",
    "inv_switch_rcs": "Switches RCS",
    "inv_switch_ms": "Switches MS",
    "reconfig_topo": "Reconfiguración",
    "inv_tie": "Enlaces nuevos",
    "prior_criticos": "Prior. críticos",
    "edrp": "EDRP",
    "veg_mgmt": "Gestión vegetación",
    "refuerzo": "Refuerzo estructural",
    "soterramiento": "Soterramiento",
}


def medidas_activas(cfg):
    """Conjunto de medidas activadas en una configuración de estrategia."""
    if not isinstance(cfg, dict):
        return set()
    return {k for k in CLAVES_MEDIDAS if cfg.get(k, False)}


def describir_medidas(cfg):
    """Texto legible con las medidas activas de una estrategia."""
    act = medidas_activas(cfg)
    if not act:
        return "sin medidas"
    return ", ".join(NOMBRE_MEDIDA.get(k, k) for k in CLAVES_MEDIDAS if k in act)


def sugerir_combinada(historial, lbl_a, lbl_b):
    """
    Busca en el historial una estrategia cuyo conjunto de medidas sea
    exactamente la unión de las medidas de A y B. Devuelve su label o None.
    """
    cfg_a = historial.get(lbl_a, {}).get("config", {})
    cfg_b = historial.get(lbl_b, {}).get("config", {})
    objetivo = medidas_activas(cfg_a) | medidas_activas(cfg_b)
    if not objetivo:
        return None
    for lbl, datos in historial.items():
        if lbl in (lbl_a, lbl_b):
            continue
        if medidas_activas(datos.get("config", {})) == objetivo:
            return lbl
    return None


def validar_diseno_factorial(cfg_a, cfg_b, cfg_ab):
    """
    Verifica que el trío A / B / AB forme un factorial 2x2 legítimo.
    Devuelve (es_valido, lista_de_advertencias).
    """
    ma, mb, mab = medidas_activas(cfg_a), medidas_activas(cfg_b), medidas_activas(cfg_ab)
    avisos = []

    if ma & mb:
        comunes = ", ".join(NOMBRE_MEDIDA.get(k, k) for k in sorted(ma & mb))
        avisos.append(
            f"A y B comparten medidas ({comunes}). El factorial 2x2 supone "
            "que A y B son conjuntos disjuntos de medidas."
        )
    if mab != (ma | mb):
        faltan = (ma | mb) - mab
        sobran = mab - (ma | mb)
        detalle = []
        if faltan:
            detalle.append("faltan: " + ", ".join(NOMBRE_MEDIDA.get(k, k) for k in sorted(faltan)))
        if sobran:
            detalle.append("sobran: " + ", ".join(NOMBRE_MEDIDA.get(k, k) for k in sorted(sobran)))
        avisos.append(
            "La estrategia combinada no corresponde exactamente a la unión "
            "de A y B (" + "; ".join(detalle) + "). La interacción calculada "
            "no será atribuible solo al par A–B."
        )
    if not ma:
        avisos.append("La estrategia A no tiene medidas activas.")
    if not mb:
        avisos.append("La estrategia B no tiene medidas activas.")

    return (len(avisos) == 0), avisos


# ─────────────────────────────────────────────────────────────
# Extracción de series por escenario
# ─────────────────────────────────────────────────────────────

def claves_escenarios_comunes(*results_dicts):
    """
    Escenarios presentes y no vacíos en TODAS las corridas entregadas.

    El pareo por escenario es indispensable: las cuatro corridas del
    factorial usan los mismos eventos de falla (common random numbers),
    lo que permite un análisis pareado de mucha menor varianza.
    """
    conjuntos = []
    for d in results_dicts:
        if not isinstance(d, dict):
            return []
        conjuntos.append({k for k, v in d.items() if v})
    if not conjuntos:
        return []
    comunes = set.intersection(*conjuntos)
    try:
        return sorted(comunes)
    except TypeError:
        return sorted(comunes, key=str)


def serie_indicador(results_dict, fn_indicador, claves):
    """
    Valor del indicador para cada escenario de `claves`, en ese orden.
    Los escenarios sin resultado válido quedan como NaN.
    """
    valores = []
    for k in claves:
        res = results_dict.get(k) if isinstance(results_dict, dict) else None
        if not res:
            valores.append(np.nan)
            continue
        try:
            valores.append(float(fn_indicador(res)))
        except Exception:
            valores.append(np.nan)
    return np.asarray(valores, dtype=float)


# ─────────────────────────────────────────────────────────────
# Interacción
# ─────────────────────────────────────────────────────────────

def interaccion_por_escenario(x_base, x_a, x_b, x_ab):
    """
    I_s = Δ_AB − (Δ_A + Δ_B) = X_A + X_B − X_∅ − X_AB, escenario a escenario.

    Válido para indicadores donde menos es mejor (ENS, SAIDI, SAIFI):
    I_s > 0 indica sinergia en ese escenario.
    """
    return np.asarray(x_a) + np.asarray(x_b) - np.asarray(x_base) - np.asarray(x_ab)


def bootstrap_ic(muestra, n_boot=5000, alpha=0.05, seed=1234):
    """
    Media e intervalo de confianza bootstrap percentil de una muestra pareada.
    Devuelve (media, limite_inferior, limite_superior).
    """
    m = np.asarray(muestra, dtype=float)
    m = m[np.isfinite(m)]
    if m.size == 0:
        return np.nan, np.nan, np.nan
    media = float(m.mean())
    if m.size < 2:
        return media, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m.size, size=(int(n_boot), m.size))
    medias = m[idx].mean(axis=1)
    lo, hi = np.percentile(medias, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return media, float(lo), float(hi)


def clasificar_interaccion(media, lo, hi):
    """Lectura cualitativa del término de interacción y su IC."""
    if not np.isfinite(media):
        return "—", "Sin datos suficientes."
    if np.isfinite(lo) and np.isfinite(hi) and lo <= 0.0 <= hi:
        return (
            "⚪ Aditivas",
            "El IC 95% incluye el cero: no hay evidencia estadística de "
            "interacción. Las medidas se comportan como independientes.",
        )
    if media > 0:
        return (
            "🟢 Sinergia",
            "El efecto conjunto supera la suma de los efectos individuales: "
            "las medidas se potencian mutuamente.",
        )
    return (
        "🟠 Solapamiento",
        "El efecto conjunto es menor que la suma de los individuales: las "
        "medidas compiten por el mismo beneficio (retornos decrecientes).",
    )


def resumen_interaccion(x_base, x_a, x_b, x_ab, n_boot=5000, seed=1234):
    """
    Todo el análisis de interacción de un indicador, en un diccionario.

    Incluye el contraste aditivo (principal) y el multiplicativo (robustez),
    este último necesario cuando las reducciones son grandes y el benchmark
    aditivo satura contra el piso del indicador (X = 0).
    """
    x_base = np.asarray(x_base, dtype=float)
    x_a = np.asarray(x_a, dtype=float)
    x_b = np.asarray(x_b, dtype=float)
    x_ab = np.asarray(x_ab, dtype=float)

    mask = np.isfinite(x_base) & np.isfinite(x_a) & np.isfinite(x_b) & np.isfinite(x_ab)
    n_val = int(mask.sum())

    m_base = float(np.mean(x_base[mask])) if n_val else np.nan
    m_a = float(np.mean(x_a[mask])) if n_val else np.nan
    m_b = float(np.mean(x_b[mask])) if n_val else np.nan
    m_ab = float(np.mean(x_ab[mask])) if n_val else np.nan

    d_a = m_base - m_a
    d_b = m_base - m_b
    d_ab = m_base - m_ab
    suma_ind = d_a + d_b

    i_serie = interaccion_por_escenario(x_base, x_a, x_b, x_ab)
    i_media, i_lo, i_hi = bootstrap_ic(i_serie[mask], n_boot=n_boot, seed=seed)

    # Índice de sinergia S = Δ_AB / (Δ_A + Δ_B)
    s_index = d_ab / suma_ind if abs(suma_ind) > 1e-12 else np.nan

    # Contraste multiplicativo (reducciones fraccionales)
    if np.isfinite(m_base) and abs(m_base) > 1e-12:
        r_a, r_b, r_ab = d_a / m_base, d_b / m_base, d_ab / m_base
        r_esp = 1.0 - (1.0 - r_a) * (1.0 - r_b)
        i_mult = r_ab - r_esp
    else:
        r_a = r_b = r_ab = r_esp = i_mult = np.nan

    etiqueta, lectura = clasificar_interaccion(i_media, i_lo, i_hi)

    return {
        "n_escenarios": n_val,
        "X_base": m_base, "X_A": m_a, "X_B": m_b, "X_AB": m_ab,
        "delta_A": d_a, "delta_B": d_b, "delta_AB": d_ab,
        "suma_individual": suma_ind,
        "I": i_media, "I_lo": i_lo, "I_hi": i_hi,
        "I_serie": i_serie,
        "S": s_index,
        "r_A": r_a, "r_B": r_b, "r_AB": r_ab,
        "r_esperado": r_esp, "I_multiplicativo": i_mult,
        "etiqueta": etiqueta, "lectura": lectura,
        "mask": mask,
    }


# ─────────────────────────────────────────────────────────────
# Atribución económica
# ─────────────────────────────────────────────────────────────

def shapley_dos_medidas(b_a, b_b, b_ab, b_base=0.0):
    """
    Valor de Shapley del beneficio conjunto entre dos medidas.

        φ_A = ½[(B_A − B_∅) + (B_AB − B_B)]
        φ_B = ½[(B_B − B_∅) + (B_AB − B_A)]
        φ_A + φ_B = B_AB − B_∅   (reparto exacto del beneficio conjunto)

    A diferencia de las contribuciones marginales, el reparto de Shapley es
    consistente: suma exactamente el beneficio de la cartera combinada.
    """
    phi_a = 0.5 * ((b_a - b_base) + (b_ab - b_b))
    phi_b = 0.5 * ((b_b - b_base) + (b_ab - b_a))
    return phi_a, phi_b


def tabla_economica(eco_a, eco_b, eco_ab, lbl_a, lbl_b, lbl_ab):
    """
    Contribuciones marginales y reparto de Shapley a partir de los
    diccionarios que devuelve `_extraer_costos_objetivo`.
    """
    def _g(eco, k):
        return float(eco.get(k, 0.0)) if isinstance(eco, dict) else 0.0

    b_a, b_b, b_ab = _g(eco_a, "beneficio"), _g(eco_b, "beneficio"), _g(eco_ab, "beneficio")
    c_a, c_b, c_ab = _g(eco_a, "costo_total"), _g(eco_b, "costo_total"), _g(eco_ab, "costo_total")

    phi_a, phi_b = shapley_dos_medidas(b_a, b_b, b_ab)

    filas = [
        {
            "Medida": lbl_a,
            "Beneficio en solitario [US$]": b_a,
            "Contribución marginal [US$]": b_ab - b_b,
            "Beneficio Shapley [US$]": phi_a,
            "Costo total [US$]": c_a,
        },
        {
            "Medida": lbl_b,
            "Beneficio en solitario [US$]": b_b,
            "Contribución marginal [US$]": b_ab - b_a,
            "Beneficio Shapley [US$]": phi_b,
            "Costo total [US$]": c_b,
        },
        {
            "Medida": f"{lbl_ab} (conjunto)",
            "Beneficio en solitario [US$]": b_ab,
            "Contribución marginal [US$]": np.nan,
            "Beneficio Shapley [US$]": phi_a + phi_b,
            "Costo total [US$]": c_ab,
        },
    ]
    df = pd.DataFrame(filas)

    interaccion_beneficio = b_ab - (b_a + b_b)
    return df, interaccion_beneficio


# ─────────────────────────────────────────────────────────────
# Tablas y figuras
# ─────────────────────────────────────────────────────────────

def tabla_factorial(resumenes, lbl_a, lbl_b, lbl_ab):
    """
    Tabla 4x(indicadores) del diseño factorial, más las filas de suma
    individual e interacción. `resumenes` = {nombre_indicador: resumen}.
    """
    filas = []
    for nombre, r in resumenes.items():
        filas.append({
            "Indicador": nombre,
            "Caso base": r["X_base"],
            lbl_a: r["X_A"],
            lbl_b: r["X_B"],
            lbl_ab: r["X_AB"],
            "Δ A": r["delta_A"],
            "Δ B": r["delta_B"],
            "Δ AB": r["delta_AB"],
            "Σ individual": r["suma_individual"],
            "Interacción I": r["I"],
            "S = ΔAB/Σ": r["S"],
            "Lectura": r["etiqueta"],
        })
    return pd.DataFrame(filas).set_index("Indicador")


def tabla_interaccion_ic(resumenes):
    """Interacción con IC bootstrap pareado, por indicador."""
    filas = []
    for nombre, r in resumenes.items():
        filas.append({
            "Indicador": nombre,
            "I (media)": r["I"],
            "IC 95% inf.": r["I_lo"],
            "IC 95% sup.": r["I_hi"],
            "I multiplicativo": r["I_multiplicativo"],
            "n escenarios": r["n_escenarios"],
            "Conclusión": r["etiqueta"],
        })
    return pd.DataFrame(filas).set_index("Indicador")


def df_por_escenario(claves, series_por_indicador):
    """
    Detalle escenario a escenario, para exportar y revisar la cola de la
    distribución (la sinergia suele concentrarse en los eventos severos).

    `series_por_indicador` = {nombre: (x_base, x_a, x_b, x_ab)}
    """
    df = pd.DataFrame({"Escenario": list(claves)})
    for nombre, (xb, xa, xbb, xab) in series_por_indicador.items():
        df[f"{nombre} — base"] = xb
        df[f"{nombre} — A"] = xa
        df[f"{nombre} — B"] = xbb
        df[f"{nombre} — AB"] = xab
        df[f"{nombre} — I"] = interaccion_por_escenario(xb, xa, xbb, xab)
    return df


def figura_waterfall(resumen, nombre_indicador, lbl_a, lbl_b, alto=420):
    """
    Cascada Base → efecto A → efecto B → interacción → combinada.

    Verifica visualmente la descomposición:
        X_AB = X_base − Δ_A − Δ_B − I
    """
    if not np.isfinite(resumen["X_base"]):
        return None

    valores = [
        resumen["X_base"],
        -resumen["delta_A"],
        -resumen["delta_B"],
        -resumen["I"],
        0,
    ]
    etiquetas_txt = [
        f"{resumen['X_base']:,.4g}",
        f"{-resumen['delta_A']:,.4g}",
        f"{-resumen['delta_B']:,.4g}",
        f"{-resumen['I']:,.4g}",
        f"{resumen['X_AB']:,.4g}",
    ]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Caso base", f"Efecto {lbl_a}", f"Efecto {lbl_b}", "Interacción", "Combinada"],
        y=valores,
        text=etiquetas_txt,
        connector={"line": {"color": "#999", "width": 1}},
        decreasing={"marker": {"color": "#2ca02c"}},
        increasing={"marker": {"color": "#d62728"}},
        totals={"marker": {"color": "#1f77b4"}},
        textposition="outside",
    ))
    fig.update_layout(
        yaxis_title=nombre_indicador,
        height=alto,
        margin=dict(l=60, r=30, t=40, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def figura_barras_factorial(resumen, nombre_indicador, lbl_a, lbl_b, lbl_ab, alto=380):
    """
    Las cuatro celdas del factorial más la barra de referencia
    'suma de efectos individuales' (lo que se esperaría si fueran aditivas).
    """
    x_esperado = resumen["X_base"] - resumen["suma_individual"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Caso base", lbl_a, lbl_b, lbl_ab],
        y=[resumen["X_base"], resumen["X_A"], resumen["X_B"], resumen["X_AB"]],
        marker_color=["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"],
        name="Observado",
        texttemplate="%{y:.3g}",
        textposition="outside",
    ))
    fig.add_hline(
        y=x_esperado,
        line_dash="dash",
        line_color="#7f7f7f",
        annotation_text="Esperado si fueran aditivas",
        annotation_position="top right",
    )
    fig.update_layout(
        yaxis_title=nombre_indicador,
        height=alto,
        margin=dict(l=60, r=30, t=40, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def figura_distribucion_interaccion(resumen, nombre_indicador, alto=360):
    """
    Distribución de I entre escenarios. Interesa la cola: es habitual que la
    sinergia sea nula en escenarios benignos y grande en los severos.
    """
    serie = np.asarray(resumen["I_serie"], dtype=float)
    serie = serie[np.isfinite(serie)]
    if serie.size == 0:
        return None

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=serie, nbinsx=min(30, max(5, serie.size // 2)),
                               marker_color="#1f77b4", name="I por escenario"))
    fig.add_vline(x=0, line_dash="dash", line_color="#d62728")
    if np.isfinite(resumen["I"]):
        fig.add_vline(x=resumen["I"], line_color="#2ca02c",
                      annotation_text="media", annotation_position="top")
    fig.update_layout(
        xaxis_title=f"Interacción I — {nombre_indicador}",
        yaxis_title="N° de escenarios",
        height=alto,
        margin=dict(l=60, r=30, t=40, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def matriz_interaccion(historial, results_base, fn_indicador, pares):
    """
    Matriz de interacción para varios pares (k x k), útil cuando ya hay
    más de dos medidas evaluadas y se quiere justificar una cartera.

    `pares` = lista de tuplas (lbl_a, lbl_b, lbl_ab).
    Devuelve un DataFrame con I por par.
    """
    etiquetas = sorted({l for p in pares for l in p[:2]})
    mat = pd.DataFrame(np.nan, index=etiquetas, columns=etiquetas, dtype=float)

    for lbl_a, lbl_b, lbl_ab in pares:
        claves = claves_escenarios_comunes(
            results_base,
            historial[lbl_a]["results"],
            historial[lbl_b]["results"],
            historial[lbl_ab]["results"],
        )
        if not claves:
            continue
        xb = serie_indicador(results_base, fn_indicador, claves)
        xa = serie_indicador(historial[lbl_a]["results"], fn_indicador, claves)
        xbb = serie_indicador(historial[lbl_b]["results"], fn_indicador, claves)
        xab = serie_indicador(historial[lbl_ab]["results"], fn_indicador, claves)
        r = resumen_interaccion(xb, xa, xbb, xab, n_boot=1000)
        mat.loc[lbl_a, lbl_b] = r["I"]
        mat.loc[lbl_b, lbl_a] = r["I"]

    return mat
