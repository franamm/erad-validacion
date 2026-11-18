"""Análisis posteriores a las corridas: activas, solapamiento, rangos y reglas.

Funciones puras (pandas/numpy, sin Streamlit), para poder contrastarlas con
las tablas de la tesis. Reproducen `graficos_validacion.ipynb` (celdas 9, 10,
11 y 13) y la matriz de `run_plan_fijo.py`: si se cambia una definición acá,
hay que cambiarla también allá.
"""
import numpy as np
import pandas as pd

NOMBRE_MEDIDA = {
    "reconfig_topo": "Reconfiguración topológica", "dg_sol": "DG solar",
    "dg_eol": "DG eólica", "dg_fos": "DG fósil", "ess_4h": "ESS 4h",
    "ess_5h": "ESS 5h", "ess_6h": "ESS 6h", "meg": "MEG", "mess": "MESS",
    "edrp": "EDRP", "refuerzo": "Refuerzo estructural",
    "soterramiento": "Soterramiento",
}
# Orden de la curva de ENS de la tesis: define el apilado del solapamiento.
ORDEN_MEDIDAS = ["reconfig_topo", "dg_sol", "dg_eol", "edrp", "refuerzo", "meg",
                 "dg_fos", "ess_4h", "ess_5h", "ess_6h", "mess", "soterramiento"]
# Orden de las columnas de la tabla de medidas activas del capítulo.
ORDEN_TABLA = ["reconfig_topo", "dg_sol", "dg_eol", "dg_fos", "ess_4h", "ess_5h",
               "ess_6h", "meg", "mess", "edrp", "refuerzo", "soterramiento"]
COLOR_CONJUNTO = "#1F2937"
NOMBRE_COMBO_TESIS = {
    "combo_dg_sol+ess_4h+ess_5h+ess_6h": "Solar + ESS 4/5/6h",
    "combo_dg_eol+ess_4h+ess_5h+ess_6h": "Eólica + ESS 4/5/6h",
    "combo_dg_fos+ess_4h+ess_5h+ess_6h": "Fósil + ESS 4/5/6h",
    "combo_reconfig_topo+ess_4h": "Reconfiguración + ESS 4h",
    "combo_reconfig_topo+dg_fos+ess_4h+ess_5h+ess_6h+meg+mess+edrp+refuerzo":
        "Sin renovables ni soterramiento",
    "combo_dg_sol+dg_eol": "Solar + eólica",
    "combo_dg_sol+dg_eol+dg_fos+ess_4h+ess_5h+ess_6h+meg+mess+edrp+refuerzo+soterramiento":
        "Todas excepto reconfiguración",
    "combo_reconfig_topo+dg_sol+dg_eol+dg_fos+ess_4h+ess_5h+ess_6h+meg+mess+edrp+refuerzo+soterramiento":
        "Librería completa",
}

# Columna del barrido y umbral sobre el que una medida habilitada cuenta como
# ACTIVA: el problema le asigna capacidad o la utiliza (celda 10 del notebook).
UMBRAL_ACTIVA = {
    "reconfig_topo": ("maniobras_tie", 0.0), "dg_sol": ("cap_dg_sol_kW", 1e-3),
    "dg_eol": ("cap_dg_eol_kW", 1e-3), "dg_fos": ("cap_dg_fos_kW", 1e-3),
    "ess_4h": ("cap_ess_4h_kWh", 1e-3), "ess_5h": ("cap_ess_5h_kWh", 1e-3),
    "ess_6h": ("cap_ess_6h_kWh", 1e-3), "meg": ("cap_meg_MW", 1e-6),
    "mess": ("cap_mess_MWh", 1e-6), "edrp": ("E_edrp", 1e-3),
    "refuerzo": ("km_ref", 1e-6), "soterramiento": ("km_ug", 1e-6),
}

INDICADORES_SOLAPAMIENTO = [("ens", "ENS", "kWh"), ("saidi", "SAIDI", "h/cliente"),
                            ("saifi", "SAIFI", "int./cliente")]


def miembros_de(combo):
    return [m for m in str(combo).removeprefix("combo_").split("+") if m]


def medidas_activas(fila, miembros):
    """Miembros de la combinación que el problema usa en esa corrida.

    `fila` es un Mapping con las columnas del barrido. Una medida sin umbral
    definido, o sin su columna en la fila, NO se descarta: queda activa, para
    no restarle al solapamiento algo que no se puede verificar.
    """
    activas = set()
    for m in miembros:
        col, umbral = UMBRAL_ACTIVA.get(m, (None, None))
        valor = fila.get(col) if col else None
        if col is None or valor is None or pd.isna(valor) or float(valor) > umbral:
            activas.add(m)
    return activas


def tabla_activas(dce, combos, escenarios):
    """Combinación × medida: en cuántos escenarios se activa (NaN = no es miembro)."""
    sub = dce[dce["escenario"].isin(escenarios)]
    filas = {}
    for combo in combos:
        ms = miembros_de(combo)
        corr = sub[sub["combinacion"] == combo]
        activas = [medidas_activas(f, ms) for _, f in corr.iterrows()]
        columnas = ORDEN_TABLA + [m for m in ms if m not in ORDEN_TABLA]
        filas[combo] = {m: (sum(m in a for a in activas) if m in ms else np.nan)
                        for m in columnas}
    return pd.DataFrame.from_dict(filas, orient="index")


def solapamiento_activas(ind, dce, combo, escenarios, indicador="ens"):
    """Abatimiento conjunto contra la suma de abatimientos individuales activos.

    `ind`: corridas individuales con `medida`, `escenario` y `d_<indicador>`.
    `dce`: corridas combinadas con `combinacion`, `escenario`, `d_<indicador>`
    y las columnas de `UMBRAL_ACTIVA`. Ambas ya enriquecidas contra el caso
    base del MISMO escenario (`barrido_vista.enriquecer`).

    En cada escenario suma el abatimiento individual (truncado en 0) de las
    medidas que la combinación activa; luego promedia sobre los escenarios.
    Si una medida activa no tiene corrida individual en un escenario, la suma
    queda indefinida: no se completa con ceros.
    """
    col = f"d_{indicador}"
    ms = miembros_de(combo)
    orden = [m for m in ORDEN_MEDIDAS if m in ms] + [m for m in ms if m not in ORDEN_MEDIDAS]
    corr = dce[(dce["combinacion"] == combo) & dce["escenario"].isin(escenarios)]
    corr = corr.set_index("escenario")
    faltan_comb = sorted(set(escenarios) - set(corr.index))
    d_ind = ind.set_index(["medida", "escenario"])[col]
    aportes = {m: [] for m in orden}
    faltan_ind = []
    activas_por_esc = {}
    for e in escenarios:
        if e not in corr.index:
            continue
        activas = medidas_activas(corr.loc[e], ms)
        activas_por_esc[e] = activas
        for m in orden:
            if m not in activas:
                aportes[m].append(0.0)
            elif (m, e) in d_ind.index and pd.notna(d_ind.loc[(m, e)]):
                aportes[m].append(max(float(d_ind.loc[(m, e)]), 0.0))
            else:
                faltan_ind.append((m, e))
                aportes[m].append(np.nan)
    serie = pd.Series({m: float(np.mean(v)) if v else np.nan for m, v in aportes.items()})
    conjunto = float(corr[col].mean()) if len(corr) else np.nan
    suma = float(serie.sum(skipna=False)) if len(serie) else np.nan
    return {
        "aportes": serie,
        "suma": suma,
        "conjunto": conjunto,
        "razon": conjunto / suma if suma and np.isfinite(suma) and suma > 0 else np.nan,
        "solapamiento": suma - conjunto,
        "activas": activas_por_esc,
        "faltan_combinacion": faltan_comb,
        "faltan_individuales": faltan_ind,
    }


# ── Rangos del plan ─────────────────────────────────────────────────────────
# (etiqueta de la medida, unidad, tipo de elemento) por clave de `plan`.
ETIQ_PLAN = {
    "CAP_DG_sol_kW": ("DG solar", "kW", "bus"),
    "CAP_DG_eol_kW": ("DG eólica", "kW", "bus"),
    "CAP_DG_fos_kW": ("DG fósil", "kW", "bus"),
    "CAP_ESS_kWh_4h": ("ESS 4h", "kWh", "bus"),
    "CAP_ESS_kWh_5h": ("ESS 5h", "kWh", "bus"),
    "CAP_ESS_kWh_6h": ("ESS 6h", "kWh", "bus"),
    "CAP_MEG_MW": ("MEG", "MW", "unidad"),
    "CAP_MESS_MWh": ("MESS", "MWh", "unidad"),
    "y_mess": ("MESS", "unidades", "unidad"),
    "x_ref": ("Refuerzo estructural", "línea tratada", "linea"),
    "x_ug": ("Soterramiento", "línea tratada", "linea"),
    "x_veg": ("Gestión de vegetación", "línea tratada", "linea"),
    "y_rcs": ("Switch RCS", "instalado", "linea"),
    "y_ms": ("Switch MS", "instalado", "linea"),
    "y_tie": ("Enlaces nuevos", "construido", "linea"),
}


def _orden(clave):
    try:
        return (0, int(clave), "")
    except (TypeError, ValueError):
        return (1, 0, str(clave))


def rangos_plan(registros, escenarios, nombre_linea=None, edrp_por_bus=None, umbral=1e-6):
    """Mínimo, máximo y media por elemento sobre los planes de cada escenario.

    `registros`: {escenario: .json de la corrida}. `edrp_por_bus`:
    {bus: {escenario: kWh}} desde `E_EDRP_bus` de los .npz. Solo aparece lo que
    se usa en al menos un escenario; donde no se instala cuenta como 0.
    Replica la celda 13 del notebook y la extiende a todas las claves de `plan`.
    """
    nombre_linea = nombre_linea or {}
    plan, ubic = {}, {}
    for e in escenarios:
        reg = registros.get(e) or {}
        for k, d in (reg.get("plan") or {}).items():
            for el, x in (d.items() if isinstance(d, dict) else [("_", d)]):
                plan.setdefault(k, {}).setdefault(str(el), {})[e] = float(x)
        for k in ("tie_horas_cerrada", "tie_maniobras"):
            for el, x in ((reg.get("ubicacion") or {}).get(k) or {}).items():
                ubic.setdefault(k, {}).setdefault(str(el), {})[e] = float(x)

    filas = []

    def agrega(medida, elemento, unidad, valores):
        v = np.array([valores.get(e, 0.0) for e in escenarios], float)
        if (np.abs(v) > umbral).any():
            filas.append(dict(Medida=medida, Elemento=elemento, Unidad=unidad,
                              Escenarios=int((np.abs(v) > umbral).sum()),
                              Min=v.min(), Max=v.max(), Media=v.mean()))

    def linea(el):
        try:
            return str(nombre_linea[int(el)]).removeprefix("Line ")
        except (KeyError, ValueError, TypeError):
            return str(el)

    claves = [k for k in ETIQ_PLAN if k in plan] + sorted(set(plan) - set(ETIQ_PLAN))
    for k in claves:
        medida, unidad, tipo = ETIQ_PLAN.get(k, (k, "", "unidad"))
        for el in sorted(plan[k], key=_orden):
            nombre = {"bus": f"Bus {el}", "linea": f"Línea {linea(el)}"}.get(tipo, f"Unidad {el}")
            agrega(medida, nombre, unidad, plan[k][el])
    for el in sorted(ubic.get("tie_horas_cerrada", {}), key=_orden):
        enlace = f"Enlace {linea(el)}"
        agrega("Reconfiguración", enlace, "h cerrado", ubic["tie_horas_cerrada"][el])
        agrega("Reconfiguración", enlace, "maniobras", ubic.get("tie_maniobras", {}).get(el, {}))
    for el in sorted(edrp_por_bus or {}, key=_orden):
        agrega("EDRP", f"Bus {el}", "kWh", edrp_por_bus[el])
    return pd.DataFrame(filas, columns=["Medida", "Elemento", "Unidad", "Escenarios",
                                        "Min", "Max", "Media"])


def edrp_por_bus(rutas_npz):
    """{bus: {escenario: kWh}} desde `E_EDRP_bus` de cada .npz ({escenario: ruta})."""
    out = {}
    for e, ruta in rutas_npz.items():
        with np.load(ruta) as z:
            if "E_EDRP_bus" not in z or "BUSES" not in z:
                continue
            for bus, x in zip(z["BUSES"], z["E_EDRP_bus"]):
                out.setdefault(str(int(bus)), {})[e] = float(x)
    return out


# ── Reglas de decisión ─────────────────────────────────────────────────────

REFERENCIAS = {
    "diagonal": "Plan diseñado para ese mismo escenario (información perfecta)",
    "mejor_candidato": "Mejor plan evaluado en ese escenario",
}


def reglas(M, beneficio=False, pesos=None, referencia="diagonal"):
    """Criterios sobre la matriz plan (fila) × escenario de evaluación (columna).

    Índice = escenario para el que se diseñó el plan; columnas = escenario
    donde se evalúa. `beneficio=False`: menor es mejor (costos, ENS…).

    Solo entran columnas completas (todos los planes evaluados), para que los
    promedios sean comparables entre filas. El arrepentimiento se mide contra
    `referencia`:
      - "diagonal": M[j, j], el plan optimizado para j. Columnas sin su
        diagonal quedan fuera del arrepentimiento y del EVPI.
      - "mejor_candidato": el mejor valor de la columna entre los planes.
    Los empates se conservan (tolerancia 1e-9).
    """
    if referencia not in REFERENCIAS:
        raise ValueError(f"Referencia desconocida: {referencia}")
    M = matriz_numerica(M)
    completas = [c for c in M.columns if M[c].notna().all()]
    M = M[completas]
    if M.empty or M.shape[1] == 0:
        raise ValueError("No hay escenarios evaluados para todos los planes.")
    w = _pesos(pesos, M.columns)

    if referencia == "diagonal":
        cols_r = [c for c in M.columns if c in M.index]
        ref = pd.Series({c: M.loc[c, c] for c in cols_r}, dtype=float)
    else:
        cols_r = list(M.columns)
        ref = M.max(axis=0) if beneficio else M.min(axis=0)
    R = (ref - M[cols_r]) if beneficio else (M[cols_r] - ref)

    wr = _pesos(pesos, cols_r) if cols_r else pd.Series(dtype=float)
    out = pd.DataFrame({
        "Valor esperado": M.mul(w, axis=1).sum(axis=1),
        "Mínimo": M.min(axis=1),
        "Máximo": M.max(axis=1),
        "Arrepentimiento esperado": R.mul(wr, axis=1).sum(axis=1) if cols_r else np.nan,
        "Arrepentimiento máximo": R.max(axis=1) if cols_r else np.nan,
    })
    if beneficio:
        criterios = {"Valor esperado": ("Valor esperado", "max"),
                     "Maximin": ("Mínimo", "max"), "Maximax": ("Máximo", "max")}
    else:
        criterios = {"Valor esperado": ("Valor esperado", "min"),
                     "Minimax": ("Máximo", "min"), "Minimin": ("Mínimo", "min")}
    criterios["Minimax regret"] = ("Arrepentimiento máximo", "min")
    elegidos = {}
    for regla, (col, sentido) in criterios.items():
        s = out[col].astype(float)
        if s.isna().all():
            elegidos[regla] = []
            continue
        mejor = s.max() if sentido == "max" else s.min()
        elegidos[regla] = list(s.index[np.isclose(s, mejor, rtol=1e-9, atol=1e-9)])

    # EVPI en la muestra: cuánto peor es comprometerse con UN plan (el de mejor
    # valor esperado) que decidir con información perfecta escenario a escenario.
    evpi = np.nan
    cols_d = [c for c in M.columns if c in M.index]
    if cols_d:
        wd = _pesos(pesos, cols_d)
        diag = sum(wd[c] * M.loc[c, c] for c in cols_d)
        ve = M[cols_d].mul(wd, axis=1).sum(axis=1)
        evpi = float(diag - ve.max()) if beneficio else float(ve.min() - diag)
    return {"criterios": out, "regret": R, "elegidos": elegidos,
            "columnas": list(M.columns), "columnas_regret": cols_r, "evpi": evpi}


def matriz_numerica(M):
    return M.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _pesos(pesos, columnas):
    columnas = list(columnas)
    if pesos is None:
        w = pd.Series(1.0, index=columnas)
    else:
        w = pd.Series(pesos, dtype=float).reindex(columnas)
    if w.isna().any() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Los pesos deben ser no negativos y cubrir todos los escenarios usados.")
    return w / w.sum()


# ── Muestra por percentil ──────────────────────────────────────────────────

def muestra_por_percentil(indicadores, n_estratos=10, criterio="ens"):
    """Centro de cada estrato de severidad, con `seleccionar_escenarios.estratificar`.

    `indicadores`: filas {escenario, ens, saidi, …} de la POBLACIÓN completa.
    Devuelve la tabla de la muestra (estrato, percentil, escenario, valor).
    """
    from seleccionar_escenarios import estratificar
    df = pd.DataFrame(indicadores).sort_values("escenario").reset_index(drop=True)
    if n_estratos > len(df):
        raise ValueError(f"Se pidieron {n_estratos} estratos y la población tiene {len(df)} escenarios.")
    pos, pcts = estratificar(df[criterio].to_numpy(float), n_estratos)
    sel = df.iloc[pos].copy()
    sel.insert(0, "Percentil", pcts)
    sel.insert(0, "Estrato", range(1, n_estratos + 1))
    return sel.reset_index(drop=True)


def muestra_mas_severa(indicadores, n=10, criterio="ens"):
    """Los `n` escenarios más severos de la población: el decil superior con n = 10 sobre 100.

    Es la regla de la tanda `sec-0-2-9-10 severo`. Ordena por `criterio` del caso
    base y devuelve del más al menos severo, con su percentil (100 = el peor).
    """
    df = pd.DataFrame(indicadores).sort_values("escenario").reset_index(drop=True)
    if n > len(df):
        raise ValueError(f"Se pidieron {n} escenarios y la población tiene {len(df)}.")
    orden = np.argsort(df[criterio].to_numpy(float), kind="stable")[::-1][:n]
    sel = df.iloc[orden].copy()
    ultimo = max(len(df) - 1, 1)
    sel.insert(0, "Percentil", [100 * (len(df) - 1 - k) / ultimo for k in range(n)])
    sel.insert(0, "Rango", range(1, n + 1))
    return sel.reset_index(drop=True)


def percentil_en_poblacion(indicadores, escenarios, criterio="ens"):
    """Posición de cada escenario en la población ordenada, en % (0 = menor)."""
    df = pd.DataFrame(indicadores).sort_values("escenario").reset_index(drop=True)
    orden = np.argsort(df[criterio].to_numpy(float), kind="stable")
    posicion = {int(df.loc[i, "escenario"]): p for p, i in enumerate(orden)}
    n = max(len(df) - 1, 1)
    return pd.DataFrame([{"escenario": e, criterio: float(df.loc[df.escenario == e, criterio].iloc[0])
                          if e in posicion else np.nan,
                          "Percentil": 100 * posicion[e] / n if e in posicion else np.nan}
                         for e in escenarios])
