# ============================================================
# barrido_vista.py — Pestaña de visualización del barrido de medidas
# ============================================================
# LEE, NO CALCULA. Todo lo que muestra esta pestaña lo produjo
# run_barrido_medidas.py y vive en artefactos/barrido/<variante>/.
#
# La separación es deliberada: el barrido son cientos de MILP de 8760 h
# (una por medida y escenario) que tardan horas. Streamlit re-ejecuta el
# script entero en cada interacción de widget y pierde el session_state
# al cerrar la pestaña, así que resolverlos desde la app significaría
# perder el trabajo a la primera distracción. El script de consola los
# guarda a medida que terminan y es reanudable; acá solo se leen.
#
# Contrato de entrada (lo que escribe run_barrido_medidas.py):
#   <barrido>/resumen_individual.csv    medida x escenario, fase 1
#   <barrido>/promedios.csv             promedio y representativo por medida
#   <barrido>/representativos.json      ranking completo de distancias
#   <barrido>/resumen_combinadas.csv    fase 3 (puede no existir todavía)
#   <barrido>/individual/<medida>/esc_XXX.{json,npz}
#   <barrido>/combinadas/<combo>/esc_XXX.{json,npz}
# ============================================================

import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from barrido_csv import leer_resumen

import pipeline as P
from run_barrido_medidas import AGREGADOS, INDICADORES, MEDIDAS

# Raíz donde viven las TANDAS de barrido: `artefactos/barrido`, más cualquier
# carpeta archivada del estilo `artefactos/barrido 01-09`.
RAIZ_BARRIDOS = P.ART_DIR
# Tanda canónica: es donde escriben las corridas nuevas que se lanzan desde acá.
DIR_BARRIDO = RAIZ_BARRIDOS / "barrido"
DIR_BARRIDO = RAIZ_BARRIDOS

def ruta_barrido(nombre):
    """Carpeta de un barrido. `nombre` viene relativo a `artefactos/`, o sea
    incluye la tanda: `barrido/sec-0-2-9-10`, `barrido 01-09/base`, etc."""
    return RAIZ_BARRIDOS / nombre

# Etiqueta legible y sentido de "mejor" para cada indicador.
ETIQUETAS = {
    "objetivo":  ("Función objetivo [USD/año]", "menor"),
    "ens":       ("ENS acumulada", "menor"),
    "saidi":     ("SAIDI [h/cliente]", "menor"),
    "saifi":     ("SAIFI [int./cliente]", "menor"),
    "asai":      ("ASAI [-]", "mayor"),
    "costo_ens": ("Costo de ENS [USD]", "menor"),
    # ── derivadas contra el caso base (ver `enriquecer`) ─────────────
    "d_ens":       ("ENS evitada [kWh]", "mayor"),
    "d_saidi":     ("SAIDI evitado [h/cliente]", "mayor"),
    "d_saifi":     ("SAIFI evitado [int./cliente]", "mayor"),
    "d_ens_pct":   ("ENS evitada [% del caso base]", "mayor"),
    "d_saidi_pct": ("SAIDI evitado [% del caso base]", "mayor"),
    "d_saifi_pct": ("SAIFI evitado [% del caso base]", "mayor"),
    "usd_kwh":     ("Costo por ENS evitada [USD/kWh]", "menor"),
    "usd_saidi":   ("Costo por SAIDI evitado [USD/(h/cliente)]", "menor"),
    "usd_saifi":   ("Costo por SAIFI evitado [USD/(int./cliente)]", "menor"),
    "bc":          ("Razón beneficio/costo [-]", "mayor"),
}

# Nombre con el que cada medida entra a las figuras que colorean por
# estrategia. NO son los de MEDIDAS a propósito: helpers_inversion.
# obtener_color_estrategia hace match por subcadena contra reglas como
# "switch rcs" o "gestion vegetacion", así que "Switches RCS" y "Gestion de
# vegetacion" caerían al gris de "no reconocida". Con estos nombres, una
# misma medida conserva su color entre el barrido y la pestaña Resultados.
NOMBRE_FIG = {
    "dg_sol": "DG Sol", "dg_eol": "DG Eól", "dg_fos": "DG Fos",
    "ess_4h": "ESS 4h", "ess_5h": "ESS 5h", "ess_6h": "ESS 6h",
    "reconfig_topo": "Reconfig topológica", "inv_tie": "Enlaces nuevos",
    "inv_switch_rcs": "Switch RCS", "inv_switch_ms": "Switch MS",
    "veg_mgmt": "Gestión vegetación", "refuerzo": "Refuerzo estructural",
    "soterramiento": "Soterramiento", "meg": "MEG", "mess": "MESS",
    "edrp": "EDRP", "prior_criticos": "Priorización cargas críticas",
}

# Resultados físicos. No tienen un "mejor" absoluto (más capacidad instalada
# no es ni bueno ni malo por sí solo), así que van sin sentido de color.
ETIQ_AGG = {
    "cap_dg_sol_kW":   "Capacidad DG solar [kW]",
    "cap_dg_eol_kW":   "Capacidad DG eólica [kW]",
    "cap_dg_fos_kW":   "Capacidad DG fósil [kW]",
    # Solo la parte que la resiliencia agrega sobre lo que el caso sin fallas
    # ya instalaba por economía. Es la MISMA base con la que el objetivo
    # calcula el CAPEX de DG, así que sin estas columnas la capacidad y su
    # costo se leen en unidades distintas.
    "cap_dg_sol_res_kW": "Capacidad DG solar por resiliencia [kW]",
    "cap_dg_eol_res_kW": "Capacidad DG eólica por resiliencia [kW]",
    "cap_ess_4h_kWh":  "Capacidad ESS 4h [kWh]",
    "cap_ess_5h_kWh":  "Capacidad ESS 5h [kWh]",
    "cap_ess_6h_kWh":  "Capacidad ESS 6h [kWh]",
    "cap_meg_MW":      "Capacidad MEG [MW]",
    "cap_mess_MWh":    "Capacidad MESS [MWh]",
    "n_rcs":           "Switches RCS instalados",
    "n_ms":            "Switches MS instalados",
    "n_tie_nuevas":    "Enlaces nuevos construidos",
    "n_mess":          "Unidades MESS instaladas",
    "n_veg":           "Líneas con gestión de vegetación",
    "n_ref":           "Líneas reforzadas",
    "n_ug":            "Líneas soterradas",
    "km_veg":          "km con gestión de vegetación",
    "km_ref":          "km reforzados",
    "km_ug":           "km soterrados",
    "E_dg_sol":        "Energía DG solar",
    "E_dg_eol":        "Energía DG eólica",
    "E_dg_fos":        "Energía DG fósil",
    "E_meg":           "Energía MEG",
    "E_ess_descarga":  "Energía descargada por ESS",
    "E_ess_carga":     "Energía cargada por ESS",
    "E_mess_descarga": "Energía descargada por MESS",
    "E_mess_carga":    "Energía cargada por MESS",
    "E_edrp":          "Energía desplazada por EDRP",
    "E_sub":           "Energía comprada a la subestación",
    "E_servida":       "Energía servida",
    "maniobras_tie":   "Maniobras de enlaces",
    "maniobras_sw":    "Maniobras de switches",
    "horas_isla":      "Horas con operación en isla",
}

# Series horarias del .npz: etiqueta y cómo agregar a diaria/mensual.
# "suma" para energía; "media" para conteos de estado, donde sumar 24 valores
# no tendría interpretación.
SERIES_POTENCIA = {
    "P_DG_sol":      ("DG solar", "suma"),
    "P_DG_eol":      ("DG eólica", "suma"),
    "P_DG_fos":      ("DG fósil", "suma"),
    "P_MEG":         ("MEG", "suma"),
    "P_ESS_4h":      ("ESS 4h (+ descarga / − carga)", "suma"),
    "P_ESS_5h":      ("ESS 5h (+ descarga / − carga)", "suma"),
    "P_ESS_6h":      ("ESS 6h (+ descarga / − carga)", "suma"),
    "P_MESS":        ("MESS (+ descarga / − carga)", "suma"),
    "EDRP":          ("EDRP", "suma"),
    "carga_servida": ("Carga servida", "suma"),
    "SOC_4h":        ("SOC ESS 4h", "media"),
    "SOC_5h":        ("SOC ESS 5h", "media"),
    "SOC_6h":        ("SOC ESS 6h", "media"),
    "SOC_MESS":      ("SOC MESS", "media"),
}
SERIES_OPERACION = {
    "fuera":          ("Líneas fuera de servicio", "media"),
    "dano":           ("Líneas averiadas", "media"),
    "tie_cerradas":   ("Enlaces cerrados", "media"),
    "maniobras_tie":  ("Maniobras de enlace", "suma"),
    "sw_en_servicio": ("Candidatas en servicio", "media"),
    "maniobras_sw":   ("Maniobras de switch", "suma"),
    "buses_isla":     ("Barras con inyección local", "media"),
}
ENERGIA_POR_BUS = {
    "E_DG_sol_bus": "DG solar", "E_DG_eol_bus": "DG eólica",
    "E_DG_fos_bus": "DG fósil", "E_MEG_bus": "MEG",
    "E_MESS_bus": "MESS", "E_EDRP_bus": "EDRP", "E_LC_bus": "ENS",
}


def _etiqueta(col):
    """Nombre legible de cualquier columna (indicador o resultado físico)."""
    if col in ETIQUETAS:
        return ETIQUETAS[col][0]
    return ETIQ_AGG.get(col, col)


# Parámetros de dimensionamiento que solo aplican si la medida está activa:
# (medida, título, [(clave, etiqueta, mín, máx, valor, paso)]). Los usan el
# lanzador de esta pestaña y la app v8.
CONDICIONALES = [
    ("meg", "MEG — generación móvil", [
        ("n_meg", "Nº de unidades", 1, 15, 3, 1),
        ("p_meg_max", "Capacidad máx. por unidad [kW]", 0.1, 5000.0, 300.0, 10.0),
        ("t_tra_meg", "Tiempo de traslado [h]", 0, 48, 2, 1)]),
    ("mess", "MESS — almacenamiento móvil", [
        ("n_mess", "Nº de unidades", 1, 20, 3, 1),
        ("h_mess", "Duración [h]", 1.0, 24.0, 3.0, 1.0),
        ("p_mess_max", "Potencia máx. [kW]", 0.1, 5000.0, 200.0, 10.0),
        ("t_tra_mess", "Tiempo de traslado [h]", 0, 48, 2, 1)]),
    ("inv_switch_ms", "Switches manuales", [
        ("t_maniobra_ms", "Tiempo de maniobra de la cuadrilla [h]", 0, 24, 1, 1)]),
]


def seccion_lanzar():
    """Configurar y lanzar un barrido, sin salir de la app.

    La app NO ejecuta el barrido: lanza `run_barrido_medidas.py` como proceso
    aparte (ver barrido_lanzador). Por eso la página sigue navegable mientras
    corre y el cálculo sobrevive a cerrar el navegador.
    """
    import barrido_lanzador as BL
    from aislacion import variante as _var

    st.subheader("Configurar y lanzar un barrido")
    st.caption(
        "El cálculo corre como proceso aparte, no dentro de la app: puedes "
        "seguir navegando, y si cierras el navegador sigue corriendo. Cada "
        "corrida se guarda apenas termina, así que es reanudable."
    )

    c1, c2 = st.columns(2)
    escenarios = c1.text_input(
        "Escenarios", "0,3,19,28,35,52,66,76,83,86", key="lz_esc",
        help="Índices Monte Carlo: '0-9' o '0,3,7'.")
    seccionadores = c1.text_input(
        "Seccionadores existentes", "0,2,9,10", key="lz_sec",
        help="Líneas con seccionador. Vacío = aislación perfecta, que deja "
             "sin efecto a switches y unidades móviles.")
    sin_fallas = c1.selectbox(
        "Artefacto E4 (caso sin fallas)", P.sin_fallas_disponibles(),
        format_func=lambda e: e or "sin_fallas.pkl (histórico)", key="lz_e4")
    medidas_sel = c2.multiselect(
        "Medidas", list(MEDIDAS), default=list(MEDIDAS),
        format_func=lambda m: f"{m} · {MEDIDAS[m]}", key="lz_med")
    fases = c2.multiselect("Fases", [1, 2, 3], default=[1, 2], key="lz_fases")
    criterio = c2.selectbox("Criterio del representativo",
                            ["objetivo", "ens", "multi"], key="lz_crit")

    with st.expander("Parámetros del modelo y del solver"):
        d1, d2, d3 = st.columns(3)
        horas = d1.number_input("Horizonte [h]", 24, 8760, 8760, step=24, key="lz_h")
        lim_dg = d1.number_input("Tope de DG (× demanda de la barra)", 0.1, 5.0,
                                 1.0, step=0.1, key="lz_lim")
        mip_gap = d2.number_input("MIPGap", 0.001, 0.20, 0.02, step=0.005,
                                  format="%.3f", key="lz_gap")
        time_limit = d2.number_input("Límite por corrida [s]", 60, 7200, 1800,
                                     step=60, key="lz_tl")
        completo = d3.checkbox("Guardar resultado completo (.pkl.gz)", True,
                               key="lz_comp",
                               help="~6 MB y ~9 s por corrida. Es lo que permite "
                                    "abrir la corrida en la pestaña Resultados.")
        microredes = d3.checkbox("Permitir operación en isla", False, key="lz_micro")
        forzar = d3.checkbox("Rehacer corridas ya guardadas", False, key="lz_forzar")

    if not medidas_sel or not fases:
        st.warning("Elige al menos una medida y una fase.")
        return

    # ── parámetros condicionales ────────────────────────────────────
    # Solo aparecen los que aplican a las medidas seleccionadas: mostrar el
    # dimensionamiento de MESS cuando MESS está apagado invita a creer que
    # ese número cambia algo. Los defaults son los de la UI de Optimización.
    extra = {}
    condicionales = CONDICIONALES
    activos = [(m, tit, campos) for m, tit, campos in condicionales if m in medidas_sel]
    if activos:
        with st.expander(f"Dimensionamiento de {', '.join(t for _, t, _ in activos)}",
                         expanded=True):
            for _m, titulo, campos in activos:
                st.markdown(f"**{titulo}**")
                for col, (clave, etiq, mn, mx, val, paso) in zip(
                        st.columns(len(campos)), campos):
                    entero = isinstance(val, int)
                    extra[clave] = col.number_input(
                        etiq, min_value=type(val)(mn), max_value=type(val)(mx),
                        value=val, step=type(val)(paso), key=f"lz_{clave}",
                        format=None if entero else "%.1f")
    else:
        st.caption("Las medidas seleccionadas no tienen parámetros de "
                   "dimensionamiento. MEG, MESS y switches manuales sí los "
                   "tienen, y aparecen acá al seleccionarlas.")

    sec_lista = [int(x) for x in seccionadores.replace(" ", "").split(",") if x]
    var = _var(bool(sec_lista), sec_lista)

    # La carpeta se deduce de la variante de aislación, pero el tope de DG, el
    # E4 y el horizonte también cambian los números y NO entran en ese nombre.
    # Por eso se deja editable: dos supuestos distintos necesitan carpetas
    # distintas, y el script se detiene si se intentan mezclar.
    sub = st.text_input(
        "Carpeta de resultados", var, key="lz_salida",
        help="Dentro de artefactos/barrido/. Cambia el nombre si vas a correr "
             "con otro tope de DG, otro E4 u otro horizonte: mezclar supuestos "
             "en una misma carpeta da una tabla sin sentido.")
    salida = DIR_BARRIDO / (sub.strip() or var)
    if sub.strip() != var:
        st.caption(f"↳ Carpeta propia. La deducida de la aislación sería `{var}`.")
    cmd = BL.construir_comando(
        escenarios=escenarios, medidas=medidas_sel, seccionadores=seccionadores,
        horas=horas, fases=",".join(str(f) for f in sorted(fases)),
        criterio=criterio, sin_fallas=sin_fallas, lim_dg=lim_dg, mip_gap=mip_gap,
        time_limit=time_limit, completo=completo, microredes=microredes,
        forzar=forzar, salida=str(salida), extra=extra)

    n_esc = len(_parse_lista(escenarios))
    st.markdown(f"**{len(medidas_sel) * n_esc} corridas** de fase 1 "
                f"({len(medidas_sel)} medidas × {n_esc} escenarios) → `{salida}`")
    # El comando se muestra SIEMPRE antes de lanzar: así queda reproducible,
    # se puede copiar a una consola y nada queda escondido tras un botón.
    st.code(BL.como_texto(cmd), language="bash")

    vivo, info = BL.esta_corriendo(salida)
    b1, b2, b3 = st.columns([1, 1, 3])

    if vivo:
        if b1.button("⏹️ Detener", key="lz_stop"):
            BL.detener(salida)
            st.rerun()
        if b2.button("🔄 Actualizar", key="lz_ref"):
            st.rerun()
        hechas, total = BL.progreso(salida, medidas_sel, _parse_lista(escenarios))
        transcurrido = time.time() - float(info.get("inicio_ts", time.time()))
        st.progress(hechas / total if total else 0.0,
                    text=f"{hechas} de {total} corridas · "
                         f"{transcurrido/60:.0f} min transcurridos"
                         + (f" · ~{transcurrido/hechas*(total-hechas)/60:.0f} min "
                            f"restantes" if hechas else ""))
        st.caption(f"PID {info.get('pid')} · iniciado {info.get('inicio')}")
        st.code(BL.cola_log(salida, 25) or "(sin salida todavía)", language="text")
    else:
        if b1.button("▶️ Lanzar barrido", key="lz_go", type="primary"):
            try:
                pid = BL.lanzar(cmd, salida)
                st.success(f"Barrido lanzado (PID {pid}). Puedes seguir usando "
                           f"la app o cerrar el navegador.")
                st.rerun()
            except Exception as e:                       # noqa: BLE001
                st.error(f"No se pudo lanzar: {type(e).__name__}: {e}")
        if BL.ruta_log(salida).exists():
            with b3.expander("Log de la última corrida"):
                st.code(BL.cola_log(salida, 40), language="text")


def _parse_lista(txt):
    """'0-9' | '0,3,7' -> [0, 3, 7]. Copia local de la del script."""
    out = []
    for parte in str(txt).replace(" ", "").split(","):
        if not parte:
            continue
        if "-" in parte:
            a, b = parte.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(parte))
    return sorted(set(out))


def _claves_int(resumen):
    """Devuelve las claves de `lineas` y `buses` a enteros.

    El resumen viaja por JSON, que convierte toda clave en texto, y
    `build_fig_grafo_resumen` indexa BUSCOORDS con enteros.
    """
    out = dict(resumen)
    for sub in ("lineas", "buses"):
        if isinstance(out.get(sub), dict):
            out[sub] = {int(k): v for k, v in out[sub].items()}
    return out


# Cómo se lee cada bloque de `ubicacion`: título y nombre de las dos columnas.
UBICACION = {
    "tie_horas_cerrada": ("Enlaces cerrados", "Enlace", "Horas cerrado"),
    "tie_maniobras":     ("Maniobras de enlace", "Enlace", "Maniobras"),
    "sw_horas_abierta":  ("Switches abiertos", "Línea", "Horas abierta"),
    "sw_maniobras":      ("Maniobras de switch", "Línea", "Maniobras"),
    "meg_horas_por_bus": ("MEG por barra", "Barra", "Horas"),
    "mess_horas_por_bus": ("MESS por barra", "Barra", "Horas"),
    "isla_horas_por_bus": ("Barras como fuente de isla", "Barra", "Horas"),
}


def _tabla_ubicacion(ubic):
    """Dónde actuó cada medida: qué enlace se cerró, qué switch se maniobró,
    a qué barra fue cada unidad móvil."""
    bloques = [(k, v) for k, v in UBICACION.items() if ubic.get(k)]
    if not bloques:
        return
    st.markdown("#### Dónde actuó la medida")
    for j in range(0, len(bloques), 3):
        for col, (clave, (titulo, c1, c2)) in zip(st.columns(3), bloques[j:j+3]):
            d = ubic[clave]
            col.markdown(f"**{titulo}**")
            col.dataframe(
                pd.DataFrame({c1: list(d), c2: list(d.values())}),
                use_container_width=True, hide_index=True)

    for clave, titulo in (("meg_horas_por_unidad_y_bus", "MEG: unidad → barra"),
                          ("mess_horas_por_unidad_y_bus", "MESS: unidad → barra")):
        if ubic.get(clave):
            with st.expander(titulo):
                filas = [{"Unidad": u, "Barra": int(b), "Horas": h}
                         for u, dd in ubic[clave].items() for b, h in dd.items()]
                st.dataframe(pd.DataFrame(filas).sort_values(["Unidad", "Horas"],
                                                            ascending=[True, False]),
                             use_container_width=True, hide_index=True)


def _cmap(col):
    """Escala de color según el sentido de 'mejor'. Los resultados físicos no
    tienen un mejor absoluto (más capacidad no es bueno ni malo por sí solo),
    así que usan una escala neutra."""
    if col not in ETIQUETAS:
        return "Blues"
    return "RdYlGn_r" if ETIQUETAS[col][1] == "menor" else "RdYlGn"


# ════════════════════════════════════════════════════════════
# Carga de artefactos
# ════════════════════════════════════════════════════════════

def listar_barridos():
    """Barridos con al menos la fase 1 escrita, en CUALQUIER tanda.

    Se recorren todas las carpetas `artefactos/barrido*` y no solo
    `artefactos/barrido`: al archivar una tanda renombrando su carpeta (por
    ejemplo a `barrido 01-09`) sus resultados desaparecían enteros de la app
    —esta función devolvía vacío y `render()` se quedaba solo con el
    lanzador, sin ninguna de las sub-pestañas— aunque los archivos siguieran
    intactos en disco.

    El nombre devuelto es relativo a `artefactos/`, así que lleva la tanda
    adentro y sirve directo para `ruta_barrido()`.

    SIN caché a propósito: es un listado de directorio (microsegundos) y el
    flujo normal es dejar la app abierta mientras el barrido corre en otra
    consola. Cacheado, un barrido nuevo no aparecería hasta reiniciar
    Streamlit.
    """
    if not RAIZ_BARRIDOS.exists():
        return []
    nombres = []
    for tanda in sorted(RAIZ_BARRIDOS.glob("barrido*")):
        if not tanda.is_dir():
            continue
        for d in sorted(tanda.iterdir()):
            if d.is_dir() and (d / "resumen_individual.csv").exists():
                nombres.append(f"{tanda.name}/{d.name}")
    return nombres


def sincronizar_barrido(clave_origen):
    """Los selectores de Barrido y Valor Esperado comparten la misma tanda."""
    destino = 've_barrido' if clave_origen == 'barr_nombre' else 'barr_nombre'
    st.session_state[destino] = st.session_state[clave_origen]


def sello(nombre):
    """mtime del resumen del barrido. Es la CLAVE de invalidación de caché,
    así que no puede estar cacheada ella misma."""
    ruta = ruta_barrido(nombre) / "resumen_individual.csv"
    return ruta.stat().st_mtime if ruta.exists() else 0.0


def cargar_barrido(nombre, _sello):
    """Lee los resúmenes pequeños desde disco en cada recarga.

    No cachear estas tablas: deben pasar siempre por leer_resumen, incluso
    después de concatenar CSV o de corregir su lector. El antiguo argumento
    `_sello` no invalidaba st.cache_data porque empezaba con guion bajo.
    Se conserva en la firma por compatibilidad con las dos vistas.
    """
    d = ruta_barrido(nombre)
    out = {"dir": str(d)}

    out["individual"] = leer_resumen(d / "resumen_individual.csv")
    out["promedios"] = (leer_resumen(d / "promedios.csv")
                        if (d / "promedios.csv").exists() else pd.DataFrame())
    out["combinadas"] = (leer_resumen(d / "resumen_combinadas.csv")
                         if (d / "resumen_combinadas.csv").exists() else pd.DataFrame())
    ruta_rep = d / "representativos.json"
    out["reps"] = (json.loads(ruta_rep.read_text(encoding="utf-8"))
                   if ruta_rep.exists() else {})

    # La variante de aislación se lee del registro de una corrida, NO del
    # nombre de la carpeta: con --salida el usuario puede llamarla como
    # quiera, y usar el nombre de carpeta haría que el caso base contra el
    # que se compara desaparezca en silencio.
    # De paso se lee el MIPGap, que fija cuánto puede empeorar una corrida
    # respecto del caso base por holgura de optimalidad (ver `enriquecer`).
    # El artefacto de E4 se lee por el mismo motivo: es el contrafactual del
    # CAPEX de DG, y cada etiqueta corresponde a otro tope de capacidad.
    out["variante"], out["mip_gap"], out["e4"] = "base", 0.02, ""
    out["lim_dg"] = 1.0
    for j in sorted(d.glob("individual/*/esc_*.json"))[:1]:
        reg = json.loads(j.read_text(encoding="utf-8"))
        out["variante"] = reg.get("variante_aislacion", "base")
        out["mip_gap"] = float(reg.get("mip_gap") or 0.02)
        out["e4"] = reg.get("e4_sin_fallas", "")
        out["lim_dg"] = float(reg.get("lim_dg") or 1.0)
    return out


@st.cache_data(show_spinner=False)
def cargar_series(nombre, subdir, etiqueta, esc, _sello=0.0):
    """Series horarias de una corrida: {P_LC, P_SUB, dano, fuera}."""
    ruta = ruta_barrido(nombre) / subdir / etiqueta / f"esc_{esc:03d}.npz"
    if not ruta.exists():
        return None
    z = np.load(ruta)
    return {k: z[k] for k in z.files}


@st.cache_data(show_spinner="Agregando los escenarios de la medida…")
def cargar_resumenes(nombre, subdir, etiqueta, _sello=0.0):
    """(resúmenes de red, escenarios) de TODAS las corridas de una medida.

    Cada .json del barrido trae su `resumen_red` —el dict de
    `grafo.resumir_simulacion`— así que el mapa agregado de la medida se
    arma leyendo unos pocos KB por escenario, sin abrir ningún .pkl.gz ni
    volver a resolver nada.

    Una corrida vieja puede no tener `resumen_red` (se agregó después):
    esas se saltan, y por eso se devuelve también la lista de escenarios
    que sí entraron — el conteo que se muestra tiene que ser el real.
    """
    carpeta = ruta_barrido(nombre) / subdir / etiqueta
    if not carpeta.is_dir():
        return [], []
    resumenes, escs = [], []
    for ruta in sorted(carpeta.glob("esc_*.json")):
        try:
            reg = json.loads(ruta.read_text(encoding="utf-8"))
        except Exception:                            # noqa: BLE001
            continue
        if reg.get("resumen_red"):
            resumenes.append(_claves_int(reg["resumen_red"]))
            escs.append(int(reg.get("escenario", ruta.stem.split("_")[1])))
    return resumenes, escs


@st.cache_data(show_spinner=False)
def ens_baseline(variante, esc, n_horas):
    """ENS del caso base (E5) del mismo escenario y supuesto de aislación,
    recortada al horizonte que efectivamente se simuló.

    Sin este recorte, comparar una corrida de 24 h contra un baseline de
    8760 h daría un 'ahorro' inventado del 99%.
    """
    ruta = P.base_dir(variante) / f"esc_{esc:03d}.npz"
    if not ruta.exists():
        return None
    return float(np.load(ruta)["P_LC"][:n_horas].sum())


@st.cache_data(show_spinner="Calculando indicadores del caso base...")
def cargar_base(variante, escenarios, n_horas):
    """Indicadores del caso base (E5) de cada escenario del barrido.

    Es la referencia de TODO lo que compara esta pestaña: sin ella, `ens`
    solo dice en qué nivel quedó cada estrategia, no cuánto evitó respecto
    de no hacer nada, y sin eso no hay costo por unidad de mejora.

    No re-resuelve nada (ver pipeline.indicadores_baseline): rehidrata los
    baselines ya calculados. La primera vez cuesta ~1 s por escenario y
    después queda cacheado en disco, así que se paga una sola vez por
    variante.
    """
    filas = []
    for e in escenarios:
        try:
            filas.append(P.indicadores_baseline(int(e), variante, n_horas or None))
        except FileNotFoundError:
            # Un escenario sin baseline no invalida el resto: se queda sin
            # Δ y las vistas que lo necesitan lo muestran como faltante.
            continue
    return pd.DataFrame(filas)


def enriquecer(df, base, mip_gap=0.02):
    """Agrega a cada corrida su Δ contra el caso base del MISMO escenario.

    La comparación es siempre intra-escenario: dos escenarios tienen otra
    secuencia de fallas, así que restarle a una corrida el baseline de otro
    escenario no significa nada.

    Devuelve el df intacto si falta el baseline; las vistas que dependen de
    las columnas nuevas comprueban su existencia.
    """
    if df.empty or base.empty:
        return df

    ren = {c: f"{c}_base" for c in ("ens", "saidi", "saifi", "costo_ens")}
    b = base.rename(columns=ren)[["escenario"] + list(ren.values())]
    out = df.merge(b, on="escenario", how="left")

    for ind in ("ens", "saidi", "saifi"):
        col_b = f"{ind}_base"
        out[f"d_{ind}"] = out[col_b] - out[ind]
        # En %, que es lo único comparable ENTRE escenarios: la ENS del
        # caso base va de ~1.900 a ~105.000 kWh según el escenario, así que
        # evitar 5.000 kWh no significa lo mismo en uno que en otro.
        out[f"d_{ind}_pct"] = np.where(out[col_b] > 0,
                                       out[f"d_{ind}"] / out[col_b], np.nan)

    if "cst_costo_total" in out:
        ct = out["cst_costo_total"]
        for ind, col in (("ens", "usd_kwh"), ("saidi", "usd_saidi"),
                         ("saifi", "usd_saifi")):
            d = out[f"d_{ind}"]
            out[col] = np.where(d > 0, ct / d, np.nan)
        out["bc"] = np.where(ct > 0, out["cst_ahorro_ens"] / ct, np.nan)

    # Corrida sospechosa: dejó más ENS que no hacer nada, por un margen que
    # el MIPGap no alcanza a explicar. El modelo siempre puede elegir no
    # instalar nada, así que su óptimo verdadero nunca es peor que el caso
    # base; con holgura de optimalidad sí puede quedar un poco peor, y esa
    # es exactamente la tolerancia que se le da acá. Lo que queda afuera son
    # corridas cortadas en un incumbente malo (típicamente por --time_limit),
    # que contaminan cualquier promedio en el que entren.
    out["sospechosa"] = out["ens"] > out["ens_base"] * (1.0 + float(mip_gap))

    # Segundo criterio, disponible solo en corridas nuevas: el solver reporta
    # cómo terminó y con qué gap. Un corte por tiempo NO es sospechoso por sí
    # solo —un incumbente al 3% sirve—, lo es cuando además quedó lejos del
    # óptimo pedido. Se suma al criterio físico, no lo reemplaza: el de arriba
    # funciona también sobre las corridas viejas, que no traen estas columnas.
    if "solver_term" in out.columns:
        cortada = ~out["solver_term"].isin(["optimal", "feasible"]) \
                  & out["solver_term"].notna()
        gap = pd.to_numeric(out.get("solver_gap"), errors="coerce")
        out["sospechosa"] |= cortada & (gap.isna() | (gap > 5 * float(mip_gap)))
    return out


@st.cache_data(show_spinner=False)
def capacidad_e4(etiqueta):
    """Capacidad de DG que el caso SIN FALLAS (E4) ya instalaba por economía.

    Es el contrafactual del CAPEX de DG: el objetivo cobra a la medida solo
    lo que instala POR ENCIMA de esto (helpers_inversion:447 y :453). Sin
    tenerlo a la vista, la capacidad instalada y su costo quedan en bases
    distintas — DG solar aparece con 29.324 kW y CAPEX cero, y no hay cómo
    saber que es porque instala exactamente lo mismo que instalaría sin
    fallas.
    """
    try:
        sf = P.cargar_sin_fallas("" if etiqueta in ("", "historico") else etiqueta)
    except (FileNotFoundError, Exception):          # noqa: BLE001
        return {}
    out = {}
    for tec in ("sol", "eol"):
        d = (sf.get(f"sin_fallas_DG_{tec}") or {}).get(f"CAP_DG_{tec}_kW") or {}
        if d:
            out[tec] = float(sum(d.values()))
        # Con las dos tecnologías el óptimo sin fallas es otro: comparten el
        # tope, y la eólica sola llena 13.423 kW pero junto a la solar se queda
        # en 13.029. Restar contra el caso de una sola tecnología daba
        # "capacidad por resiliencia" negativa en todo plan con ambas.
        d = (sf.get("sin_fallas_DG_ambas") or {}).get(f"CAP_DG_{tec}_kW") or {}
        if d:
            out[f"{tec}_ambas"] = float(sum(d.values()))
    return out


def _cap_resiliencia(df, cap_e4):
    """Agrega la capacidad de DG atribuible a resiliencia, in place.

    Solo se calcula donde la medida efectivamente instaló DG: en una corrida
    de refuerzo la capacidad solar es cero, y restarle el contrafactual daría
    un −29.324 kW sin sentido.

    Si la corrida instaló solar Y eólica, el contrafactual es el E4 con ambas
    tecnologías, no el de cada una por separado.
    """
    cols = ("cap_dg_sol_kW", "cap_dg_eol_kW")
    ambas = (np.all([pd.to_numeric(df[c], errors="coerce").fillna(0) > 0 for c in cols], axis=0)
             if all(c in df.columns for c in cols) else np.zeros(len(df), dtype=bool))
    for tec in ("sol", "eol"):
        col = f"cap_dg_{tec}_kW"
        if col in df.columns and tec in cap_e4:
            ref = np.where(ambas, cap_e4.get(f"{tec}_ambas", cap_e4[tec]), cap_e4[tec])
            df[f"cap_dg_{tec}_res_kW"] = np.where(
                df[col] > 0, df[col] - ref, 0.0)
    return df


def _nombre_fig(m):
    """Etiqueta de una estrategia para las figuras coloreadas por medida."""
    if isinstance(m, str) and m.startswith(PREF_COMBO):
        return _nombre(m)
    return NOMBRE_FIG.get(m, MEDIDAS.get(m, str(m)))


def _agregar(serie, gran, modo="suma"):
    """Serie horaria -> diaria/mensual.

    `modo="suma"` para energía (la ENS del día es la suma de sus horas);
    `modo="media"` para conteos de estado, como las líneas fuera de
    servicio, donde sumar 24 valores daría un número sin interpretación.
    """
    s = np.asarray(serie, dtype=float)
    paso = {"Horaria": 1, "Diaria": 24, "Mensual": 730}[gran]   # 730 h ~ 1 mes
    if paso == 1 or len(s) < paso:
        return s
    n = (len(s) // paso) * paso
    bloques = s[:n].reshape(-1, paso)
    return bloques.sum(axis=1) if modo == "suma" else bloques.mean(axis=1)


# Prefijo con el que una combinación entra a las tablas de medidas. Permite
# que todo lo que compara medidas (heatmaps, rankings, agregadores) trate a
# una combinación como una fila más, sin duplicar ninguna de esas funciones.
PREF_COMBO = "combo::"


def _nombre(m):
    if isinstance(m, str) and m.startswith(PREF_COMBO):
        return _nombre_combo(m[len(PREF_COMBO):])
    return MEDIDAS.get(m, m)


def _unir_combinadas(dfe, dce):
    """Individual y combinadas en una sola tabla, sobre escenarios comunes.

    Devuelve (tabla, escenarios). El recorte a los escenarios en que corrió la
    combinación NO es opcional: las individuales corrieron en 10 escenarios y
    la combinada en 4, y como la severidad entre escenarios varía hasta 6
    veces, promediar cada una sobre su propio conjunto compararía la dureza de
    los escenarios y no las medidas.

    Las corridas combinadas sospechosas quedan fuera: un incumbente cortado
    por tiempo entraría al ranking como si fuera un resultado.
    """
    if dce.empty or "sospechosa" not in dce.columns:
        return dfe, None
    val = dce[~dce["sospechosa"].fillna(False)]
    if val.empty or "combinacion" not in val.columns:
        return dfe, None
    escs = sorted(set(val["escenario"]) & set(dfe["escenario"]))
    if not escs:
        return dfe, None
    c = val[val["escenario"].isin(escs)].copy()
    c["medida"] = PREF_COMBO + c["combinacion"].astype(str)
    return (pd.concat([dfe[dfe["escenario"].isin(escs)], c], ignore_index=True),
            escs)


# ════════════════════════════════════════════════════════════
# Costo-efectividad
# ════════════════════════════════════════════════════════════

def _nombre_combo(c):
    """Identifica los componentes, incluso cuando dos carteras tienen igual tamaño."""
    ms = [x for x in str(c).removeprefix("combo_").split("+") if x]
    cortos = {
        "dg_sol": "GD solar", "dg_eol": "GD eólica", "dg_fos": "GD fósil",
        "ess_4h": "SAE 4 h", "ess_5h": "SAE 5 h", "ess_6h": "SAE 6 h",
        "reconfig_topo": "Reconfiguración", "meg": "GD móvil",
        "mess": "SAE móvil", "refuerzo": "Refuerzo",
        "veg_mgmt": "Vegetación", "soterramiento": "Soterramiento",
    }
    # Agrupar las duraciones sin ocultar qué alternativas SAE están presentes.
    sae = [m for m in ("ess_4h", "ess_5h", "ess_6h") if m in ms]
    etiquetas = []
    for m in ms:
        if len(sae) > 1 and m in sae:
            if m == next(x for x in ms if x in sae):
                etiquetas.append("SAE " + "/".join(x[4:-1] for x in sae) + " h")
        else:
            etiquetas.append(cortos.get(m, _nombre(m)))
    return " + ".join(etiquetas)


# Mejora mínima para que un costo unitario signifique algo. Bajo estos
# valores la "mejora" es ruido numérico del solver (un ΔSAIDI de 3e-15 en vez
# de 0 exacto), y dividir el costo por ella da cifras como 1e17 USD que
# aplastan la escala de cualquier tabla o gráfico. Se usan en el MISMO valor
# para la tabla y para las curvas, para que una medida no pueda aparecer en
# una y no en la otra.
UMBRAL_MEJORA = {"d_ens": 1.0, "d_saidi": 1e-3, "d_saifi": 1e-4}


def _unitario(costo, mejora, umbral):
    """Costo por unidad de mejora, o NaN si el cociente no significa nada.

    Se exige costo POSITIVO además de mejora suficiente. Con la contabilidad
    incremental, una estrategia puede terminar con costo total negativo —el
    crédito por la energía que deja de comprarse a la subestación supera al
    CAPEX+OPEX—, y ahí "USD por kWh evitado" no es un costo bajo: no existe.
    Dividir igual devolvería un número negativo que ordenaría la tabla como si
    fuera la opción más barata, cuando en realidad es de otra categoría.
    """
    costo, mejora = np.asarray(costo, dtype=float), np.asarray(mejora, dtype=float)
    return np.where((costo > 0) & (mejora > umbral), costo / np.where(mejora == 0, np.nan, mejora), np.nan)


def _resumen_eco(sub, col_caso):
    """Una fila por estrategia con mejora, costo y costo unitario.

    Los costos unitarios se recalculan del promedio de costo dividido por el
    promedio de mejora, NO como promedio de los cocientes: el promedio de
    razones pondera igual un escenario donde la medida evitó 100 kWh que
    otro donde evitó 10.000, y basta un escenario con mejora casi nula para
    que el costo unitario promedio se dispare.
    """
    _agg = dict(
        n=("escenario", "nunique"),
        d_ens=("d_ens", "mean"),
        d_saidi=("d_saidi", "mean"),
        d_saifi=("d_saifi", "mean"),
        d_ens_pct=("d_ens_pct", "mean"),
        capex=("cst_capex_anual", "mean"),
        opex=("cst_opex_anual", "mean"),
        costo_total=("cst_costo_total", "mean"),
        ahorro_ens=("cst_ahorro_ens", "mean"),
        ahorro_neto=("cst_ahorro_neto", "mean"),
    )
    # Desglose del OPEX. Solo existe en barridos rehechos desde que el helper
    # separa el término de subestación; en los anteriores no se agregan y el
    # bloque correspondiente simplemente no se dibuja.
    for _dest, _src in (("opex_medidas",     "cst_opex_medidas"),
                        ("termino_sub",      "cst_termino_subestacion"),
                        ("termino_sub_fall", "cst_termino_subestacion_falla"),
                        ("opex_resiliencia", "cst_opex_resiliencia")):
        if _src in sub.columns:
            _agg[_dest] = (_src, "mean")
    g = sub.groupby(col_caso, as_index=False).agg(**_agg)
    for ind, col in (("d_ens", "usd_kwh"), ("d_saidi", "usd_saidi"),
                     ("d_saifi", "usd_saifi")):
        g[col] = _unitario(g["costo_total"], g[ind], UMBRAL_MEJORA[ind])
    g["bc"] = np.where(g["costo_total"] > 0,
                       g["ahorro_ens"] / g["costo_total"], np.nan)
    return g.sort_values("usd_kwh", na_position="last")


# La tabla económica va SEPARADA en bloques, no como una sola tabla de 13
# columnas: mezclar mejora física, resultado en dinero y costo unitario en la
# misma grilla obliga a leer todo para encontrar una cosa. Cada bloque
# responde una pregunta distinta y se lee solo.
BLOQUES_ECO = {
    "Cuánto mejora": [
        ("d_ens",       "ENS evitada [kWh]",        "{:,.0f}"),
        ("d_ens_pct",   "ENS evitada [%]",          "{:.1%}"),
        ("d_saidi",     "SAIDI evitado [h/cl.]",    "{:.3f}"),
        ("d_saifi",     "SAIFI evitado [int./cl.]", "{:.3f}"),
    ],
    "Cuánto cuesta y cuánto devuelve": [
        ("capex",       "CAPEX anual [USD]",       "{:,.0f}"),
        # No es OPEX a secas: helpers_inversion:673 le suma el término de
        # subestación (lo que se deja de comprar, o se compra de más, frente
        # al caso base). Por eso puede salir negativo, y llamarlo "OPEX"
        # hacía que un crédito de energía se leyera como un costo de operación.
        ("opex",        "OPEX + energía de subestación [USD]", "{:,.0f}"),
        ("costo_total", "Costo total [USD/año]",   "{:,.0f}"),
        ("ahorro_ens",  "Ahorro de ENS [USD/año]", "{:,.0f}"),
        ("ahorro_neto", "Ahorro neto [USD/año]",   "{:,.0f}"),
        ("bc",          "B/C [-]",                 "{:.2f}"),
    ],
    # Desglose de la columna anterior. `opex` mezcla el OPEX de operar la medida
    # con el término de subestación, y ese término se calcula sobre las 8.760 h
    # contra un caso sin fallas evaluado a la capacidad óptima SIN fallas: si la
    # estrategia instala más que esa referencia, el excedente genera también en
    # las horas sanas y su ahorro de compra queda acreditado como resiliencia
    # sin serlo. La versión "solo horas con falla" elimina esa fuga.
    "De dónde sale ese OPEX": [
        ("opex_medidas",     "OPEX de la medida [USD/año]",              "{:,.0f}"),
        ("termino_sub",      "Energía de subestación, año completo [USD]", "{:,.0f}"),
        ("termino_sub_fall", "Energía de subestación, solo fallas [USD]",  "{:,.0f}"),
        ("opex_resiliencia", "OPEX atribuible a resiliencia [USD/año]",  "{:,.0f}"),
    ],
    "Cuánto cuesta cada unidad de mejora": [
        ("usd_kwh",   "USD/kWh evitado",              "{:,.2f}"),
        ("usd_saidi", "USD por h/cliente evitada",    "{:,.0f}"),
        ("usd_saifi", "USD por int./cliente evitada", "{:,.0f}"),
    ],
}
COLS_ECO = [c for bloque in BLOQUES_ECO.values() for c in bloque]

# Cómo colorear cada columna, cuando el color agrega algo. Se deja fuera lo
# que no tiene un "mejor" absoluto (CAPEX alto no es malo si compra mucha
# mejora), para que el color signifique siempre lo mismo.
GRADIENTES = {
    "B/C [-]": ("RdYlGn", dict(vmin=0, vmax=2)),
    "Ahorro neto [USD/año]": ("RdYlGn", {}),
    "USD/kWh evitado": ("RdYlGn_r", {}),
    "ENS evitada [%]": ("Greens", {}),
}


def _veredicto(fila):
    """Una palabra por medida, para no tener que leer la tabla entera.

    El corte es la razón beneficio/costo: sobre 1 el ahorro de ENS paga la
    medida por sí solo. Entre 0,5 y 1 la medida no se paga con ENS evitada
    pero está en el orden de magnitud, así que la decisión depende de
    beneficios que este modelo no valoriza (imagen, norma, continuidad de
    cargas críticas). Bajo 0,5 no hay caso.
    """
    if not (fila["d_ens"] > UMBRAL_MEJORA["d_ens"]):
        return "⚪ sin efecto"
    bc = fila["bc"]
    if pd.isna(bc):
        return "· sin costo"
    if bc >= 1:
        return "✅ se paga sola"
    if bc >= 0.5:
        return "🟡 marginal"
    return "🔴 no se justifica"


def _estilo(tabla, cols):
    """Formato y gradientes de un bloque de la tabla económica."""
    est = tabla.style.format({t: f for _, t, f in cols if t in tabla.columns},
                             na_rep="—")
    for _, titulo, _f in cols:
        if titulo in tabla.columns and titulo in GRADIENTES:
            cmap, kw = GRADIENTES[titulo]
            est = est.background_gradient(cmap=cmap, subset=[titulo], **kw)
    return est


def _curva_mac(g, col_delta, col_unitario, titulo, titulo_x, titulo_y,
               umbral_ruido, referencia=None, nombre_referencia=None):
    """Curva de abatimiento con la MISMA función que usa la pestaña
    Resultados, para que las dos figuras de la tesis salgan idénticas."""
    from helpers_inversion import construir_curva_abatimiento_2

    datos = pd.DataFrame({
        "Estrategia": g["Estrategia"],
        "Δ": g[col_delta],
        "Costo unitario": g[col_unitario],
        "Costo total": g["costo_total"],
    })
    return construir_curva_abatimiento_2(
        df=datos, columna_reduccion="Δ", columna_costo_unitario="Costo unitario",
        columna_costo_total="Costo total", columna_medida="Estrategia",
        titulo=titulo, titulo_x=titulo_x, titulo_y=titulo_y,
        escala_y="symlog", incluir_variaciones_negativas=False,
        umbral_ruido=umbral_ruido, mostrar_etiquetas=True,
        referencia=referencia, nombre_referencia=nombre_referencia,
        altura=620, fuente_titulo=20, fuente_ejes=15, fuente_ticks=13,
        fuente_leyenda=13, fuente_etiquetas=12)


def seccion_costo_efectividad(dfe, dce, reps, nombre, detalle):
    """Cuánto mejora cada medida, cuánto cuesta y cuánto cuesta cada unidad
    de mejora. Es la pregunta que ordena la cartera de inversión."""
    st.subheader("Costo-efectividad")

    if "d_ens" not in dfe.columns:
        st.warning(
            "No hay indicadores del caso base para esta variante, así que no "
            "se puede calcular cuánto evitó cada medida. Generar los baselines "
            "con:  `python pipeline.py --etapas E5 --escenarios ... "
            "--seccionadores ...`")
        return
    if "cst_costo_total" not in dfe.columns:
        st.warning("Este barrido es anterior a la descomposición de costos "
                   "(columnas `cst_*`). Rehacerlo para ver costo-efectividad.")
        return

    # VoLL efectivo, leído de los propios datos en vez de importarlo: es el
    # cociente entre el costo de ENS del caso base y su ENS física.
    voll = float(np.nanmedian(np.where(dfe["ens_base"] > 0,
                                       dfe["costo_ens_base"] / dfe["ens_base"],
                                       np.nan)))

    ambito = st.radio(
        "Ámbito", ["Promedio de escenarios", "Un escenario",
                   "Representativo por medida"],
        key="eco_ambito", horizontal=True,
        help="El promedio es el resultado robusto y el que conviene reportar. "
             "Un escenario permite además incluir las corridas combinadas, "
             "que solo existen para los representativos.")

    # Las corridas sospechosas se excluyen SIEMPRE en el resumen: son un
    # incumbente malo del solver, no un resultado, y dejar la decisión al
    # lector en la vista que pretende ser de una pasada es regalarle un
    # problema que no pidió. En Detalle sí se puede mirar lo que hay.
    excluir = True
    if detalle:
        excluir = st.checkbox(
            "Excluir corridas sospechosas", True, key="eco_excluir",
            help="Corridas que dejaron más ENS que no hacer nada, por encima "
                 "de lo que el MIPGap puede explicar. El modelo siempre puede "
                 "no instalar nada, así que eso solo pasa cuando la corrida se "
                 "cortó en un incumbente malo, casi siempre por --time_limit.")

    base_ind, base_comb = dfe, dce
    if excluir:
        base_ind = dfe[~dfe["sospechosa"].fillna(False)]
        if not dce.empty and "sospechosa" in dce:
            base_comb = dce[~dce["sospechosa"].fillna(False)]

    n_sosp = int(dfe["sospechosa"].fillna(False).sum()) + (
        int(dce["sospechosa"].fillna(False).sum())
        if not dce.empty and "sospechosa" in dce else 0)
    if n_sosp and not detalle:
        st.caption(f"⚠️ {n_sosp} corrida(s) sospechosa(s) excluida(s). "
                   f"El detalle está en la vista Detalle.")
    elif n_sosp:
        sosp = pd.concat([
            dfe[dfe["sospechosa"].fillna(False)][["medida", "escenario", "ens",
                                                  "ens_base", "segundos"]],
            (dce[dce["sospechosa"].fillna(False)]
             .assign(medida=lambda d: d["combinacion"])
             [["medida", "escenario", "ens", "ens_base", "segundos"]]
             if not dce.empty and "sospechosa" in dce else pd.DataFrame()),
        ])
        with st.expander(f"⚠️ {n_sosp} corrida(s) sospechosa(s) "
                         f"{'excluida(s)' if excluir else 'INCLUIDA(S)'}"):
            st.dataframe(sosp, use_container_width=True, hide_index=True)
            st.caption(
                "Dejaron más ENS que el caso base por encima de la holgura "
                "del MIPGap, lo que es imposible en un óptimo verdadero: no "
                "instalar nada siempre está disponible. Revisar `segundos` "
                "contra el `--time_limit` con que se corrió el barrido; casi "
                "siempre es una corrida cortada por tiempo.")

    # Corridas anteriores a la descomposición de costos: se descartan enteras
    # en vez de dejarlas aportar su mejora sin su costo, que inflaría la
    # costo-efectividad de la medida a la que pertenecen.
    n_sin_costo = int(base_ind["cst_costo_total"].isna().sum())
    base_ind = base_ind[base_ind["cst_costo_total"].notna()]
    if not base_comb.empty and "cst_costo_total" in base_comb:
        n_sin_costo += int(base_comb["cst_costo_total"].isna().sum())
        base_comb = base_comb[base_comb["cst_costo_total"].notna()]
    if n_sin_costo and detalle:
        st.caption(
            f"ℹ️ {n_sin_costo} corrida(s) sin descomposición de costos "
            f"(anteriores a las columnas `cst_*`) quedan fuera de esta "
            f"pestaña. Se recuperan con `--forzar` en esa medida.")

    # ── armar el conjunto de estrategias a comparar ──────────────────
    if ambito == "Un escenario":
        escs = sorted(base_ind["escenario"].unique())
        esc = st.selectbox("Escenario", escs, key="eco_esc")
        sub = base_ind[base_ind["escenario"] == esc].copy()
        sub["Estrategia"] = [_nombre_fig(m) for m in sub["medida"]]
        if not base_comb.empty:
            comb_esc = base_comb[base_comb["escenario"] == esc]
            if not comb_esc.empty and st.checkbox(
                    "Incluir las combinaciones corridas en este escenario",
                    True, key="eco_incl_comb"):
                cc = comb_esc.copy()
                cc["Estrategia"] = [_nombre_combo(c) for c in cc["combinacion"]]
                sub = pd.concat([sub, cc], ignore_index=True)
        pie = (f"Escenario {esc} · caso base: "
               f"{sub['ens_base'].iloc[0]:,.0f} kWh de ENS")
    elif ambito == "Representativo por medida":
        por_medida = (reps or {}).get("por_medida", {})
        if not por_medida:
            st.info("Falta la fase 2: no hay escenarios representativos.")
            return
        pares = {m: v["escenario"] for m, v in por_medida.items()}
        sub = base_ind[[m in pares and e == pares[m] for m, e in
                        zip(base_ind["medida"], base_ind["escenario"])]].copy()
        sub["Estrategia"] = [_nombre_fig(m) for m in sub["medida"]]
        pie = ("Cada medida en SU escenario representativo. Ojo: son "
               "escenarios distintos, con otra secuencia de fallas, así que "
               "la comparación entre medidas es indicativa.")
    else:
        sub = base_ind.copy()
        pie = (f"Promedio sobre {sub['escenario'].nunique()} escenarios. "
               f"Es el resultado robusto: cada medida se evaluó en todos.")
        unido, escs_comunes = (_unir_combinadas(base_ind, base_comb)
                               if not base_comb.empty else (base_ind, None))
        if escs_comunes and st.checkbox(
                f"Incluir las combinaciones (recorta a los "
                f"{len(escs_comunes)} escenarios en que corrieron)", False,
                key="eco_incl_comb_prom",
                help="Las combinaciones corrieron en menos escenarios. Para "
                     "promediarlas junto a las medidas sueltas hay que mirar "
                     "los mismos escenarios en ambos casos."):
            sub = unido.copy()
            pie = (f"Promedio sobre los {len(escs_comunes)} escenarios en que "
                   f"corrieron las combinaciones ({escs_comunes}), con las "
                   f"medidas individuales restringidas a esos mismos.")
        sub["Estrategia"] = [_nombre_fig(m) for m in sub["medida"]]

    if sub.empty:
        st.info("No quedan corridas que comparar con estos filtros.")
        return

    g = _resumen_eco(sub, "Estrategia")
    st.caption(pie)

    # ── titulares ────────────────────────────────────────────────────
    mejor = g.dropna(subset=["usd_kwh"]).head(1)
    k1, k2, k3 = st.columns(3)
    k1.metric("Se pagan solas", f"{int((g['bc'] >= 1).sum())} de {len(g)}",
              help="B/C ≥ 1: el ahorro de ENS supera el costo anual.")
    if not mejor.empty:
        k2.metric("Más barata por kWh evitado",
                  f"{mejor['usd_kwh'].iloc[0]:,.2f} USD/kWh",
                  help=str(mejor["Estrategia"].iloc[0]))
    k3.metric("VoLL de referencia", f"{voll:,.2f} USD/kWh",
              help="Bajo esta línea, evitar un kWh cuesta menos de lo que ese "
                   "kWh no servido le cuesta al sistema.")

    if detalle:
        _eco_detalle(g, sub, voll, nombre)
    else:
        _eco_resumen(g, voll)


def _eco_resumen(g, voll):
    """Lo mínimo para decidir: qué medida conviene, en qué orden y cuál no.

    Deliberadamente NO trae SAIDI/SAIFI, ni la descomposición CAPEX/OPEX, ni
    el gráfico costo-beneficio: todo eso está en Detalle. Una tabla de 13
    columnas no se lee, se hojea.
    """
    tabla = pd.DataFrame({
        "Estrategia": g["Estrategia"],
        "Veredicto": g.apply(_veredicto, axis=1),
        "ENS evitada [%]": g["d_ens_pct"],
        "USD/kWh evitado": g["usd_kwh"],
        "B/C [-]": g["bc"],
        "Ahorro neto [USD/año]": g["ahorro_neto"],
    })
    cols = [c for c in COLS_ECO if c[1] in tabla.columns]
    st.dataframe(_estilo(tabla, cols), use_container_width=True,
                 hide_index=True)
    st.caption(
        "Ordenada de la mejora más barata a la más cara. **B/C** = ahorro de "
        "ENS / costo total: sobre 1 la medida se paga sola con la energía que "
        "evita dejar de servir.")

    st.markdown("#### Curva de abatimiento de ENS")
    try:
        fig, datos = _curva_mac(
            g, "d_ens", "usd_kwh", "Curva de abatimiento de ENS",
            "Reducción acumulada de ENS [kWh]",
            "Costo por ENS evitada [USD/kWh]", UMBRAL_MEJORA["d_ens"],
            voll, "VoLL")
    except Exception as e:                          # noqa: BLE001
        st.warning(f"No se pudo construir la curva: {type(e).__name__}: {e}")
        return
    if fig is None:
        st.info("Ninguna estrategia reduce la ENS por sobre el umbral de "
                "ruido, así que no hay curva que dibujar.")
        return
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "El ancho de cada barra es cuánta ENS evita la medida; la altura, lo "
        "que cuesta cada kWh evitado. Leída de izquierda a derecha da el orden "
        "en que conviene invertir, y todo lo que queda bajo la línea del VoLL "
        "cuesta menos que la energía que ahorra. "
        "En **Detalle** están las curvas de SAIDI y SAIFI.")


def _cap_dg_detalle(sub):
    """Capacidad de DG: total instalada frente a la parte que agrega la
    resiliencia.

    Es la lectura que le falta al CAPEX de DG. El objetivo cobra solo la
    capacidad POR ENCIMA de la que el caso sin fallas ya instalaba por
    economía, así que sin esta tabla DG solar aparece con 29.324 kW y CAPEX
    cero, y no hay forma de saber que es porque no agrega ni un kW.
    """
    filas = []
    for tec, lab in (("sol", "DG solar"), ("eol", "DG eólica")):
        col, col_res = f"cap_dg_{tec}_kW", f"cap_dg_{tec}_res_kW"
        if col_res not in sub.columns or not (sub[col] > 0).any():
            continue
        s = sub[sub[col] > 0]
        total, res = float(s[col].mean()), float(s[col_res].mean())
        filas.append({
            "Tecnología": lab,
            "Instalada [kW]": total,
            "Ya instalada sin fallas (E4) [kW]": total - res,
            "Atribuible a resiliencia [kW]": res,
            "Fracción por resiliencia": res / total if total else np.nan,
        })
    if not filas:
        return
    with st.expander("Capacidad de DG: total vs. atribuible a resiliencia"):
        st.dataframe(
            pd.DataFrame(filas).style.format({
                "Instalada [kW]": "{:,.1f}",
                "Ya instalada sin fallas (E4) [kW]": "{:,.1f}",
                "Atribuible a resiliencia [kW]": "{:,.1f}",
                "Fracción por resiliencia": "{:.1%}"}, na_rep="—"),
            use_container_width=True, hide_index=True)
        st.caption(
            "El CAPEX de DG del objetivo cobra solo la última columna: la "
            "capacidad que la resiliencia agrega sobre lo que el caso sin "
            "fallas ya instalaba por economía. Por eso una tecnología puede "
            "aparecer con mucha capacidad y CAPEX cero.")


def _eco_detalle(g, sub, voll, nombre):
    """Todo lo demás, en bloques que se leen por separado."""
    for titulo, cols in BLOQUES_ECO.items():
        presentes = [c for c in cols if c[0] in g.columns]
        if not presentes:
            continue
        st.markdown(f"#### {titulo}")
        tabla = g[["Estrategia"] + [c for c, _, _ in presentes]].rename(
            columns={c: t for c, t, _ in presentes})
        st.dataframe(_estilo(tabla, presentes), use_container_width=True,
                     hide_index=True)
        if titulo.startswith("Cuánto cuesta y"):
            _cap_dg_detalle(sub)

    st.caption(
        "Los costos unitarios salen del promedio de costo dividido por el "
        "promedio de mejora, no del promedio de los cocientes: promediar "
        "razones pondera igual un escenario que evitó 100 kWh y otro que "
        "evitó 10.000.")
    completa = g[["Estrategia", "n"] + [c for c, _, _ in COLS_ECO
                                        if c in g.columns]].rename(
        columns={c: t for c, t, _ in COLS_ECO})
    st.download_button(
        "⬇️ Descargar tabla completa (CSV)",
        completa.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"barrido_{nombre}_costo_efectividad.csv", mime="text/csv",
        key="eco_dl")

    # ── curvas de abatimiento ────────────────────────────────────────
    st.divider()
    st.markdown("#### Curvas de costo marginal de abatimiento")
    st.caption(
        "El ancho de cada barra es cuánto mejora esa medida el indicador; la "
        "altura, lo que cuesta cada unidad de mejora. Se ordenan de más barata "
        "a más cara, así que leídas de izquierda a derecha dan el orden en que "
        "conviene invertir. Solo aparecen las medidas que mejoran el indicador.")

    t_ens, t_saidi, t_saifi = st.tabs(["ENS", "SAIDI", "SAIFI"])
    figuras = [
        (t_ens, "d_ens", "usd_kwh", "Curva de abatimiento de ENS",
         "Reducción acumulada de ENS [kWh]", "Costo por ENS evitada [USD/kWh]",
         voll, "VoLL"),
        (t_saidi, "d_saidi", "usd_saidi", "Curva de abatimiento de SAIDI",
         "Reducción acumulada de SAIDI [h/cliente]",
         "Costo por SAIDI evitado [USD/(h/cliente)]", None, None),
        (t_saifi, "d_saifi", "usd_saifi", "Curva de abatimiento de SAIFI",
         "Reducción acumulada de SAIFI [int./cliente]",
         "Costo por SAIFI evitado [USD/(int./cliente)]", None, None),
    ]
    for tab, col_d, col_u, tit, tx, ty, ref, nom_ref in figuras:
        with tab:
            try:
                fig, datos = _curva_mac(g, col_d, col_u, tit, tx, ty,
                                        UMBRAL_MEJORA[col_d], ref, nom_ref)
            except Exception as e:                  # noqa: BLE001
                st.warning(f"No se pudo construir la curva: "
                           f"{type(e).__name__}: {e}")
                continue
            if fig is None:
                st.info("Ninguna estrategia mejora este indicador por sobre el "
                        "umbral de ruido, así que no hay curva que dibujar.")
                continue
            st.plotly_chart(fig, use_container_width=True)
            fuera = sorted(set(g["Estrategia"]) - set(datos["Estrategia"]))
            if fuera:
                st.caption(f"Fuera de la curva (no mejoran el indicador): "
                           f"{', '.join(fuera)}.")

    # ── costo vs beneficio ───────────────────────────────────────────
    st.divider()
    st.markdown("#### Costo contra beneficio")
    from helpers_inversion import obtener_color_estrategia
    gp = g.dropna(subset=["costo_total", "ahorro_ens"]).copy()
    gp["Tamaño"] = gp["d_ens"].clip(lower=0).fillna(0)
    fig_cb = px.scatter(
        gp, x="costo_total", y="ahorro_ens", color="Estrategia",
        size="Tamaño", size_max=38, text="Estrategia",
        color_discrete_map={e: obtener_color_estrategia(e)
                            for e in gp["Estrategia"]},
        labels={"costo_total": "Costo total de la medida [USD/año]",
                "ahorro_ens": "Ahorro de ENS [USD/año]"})
    lim = float(max(gp["costo_total"].max(), gp["ahorro_ens"].max()) * 1.05)
    fig_cb.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                     line=dict(color="#888", dash="dash"))
    fig_cb.add_annotation(x=lim, y=lim, text="B/C = 1", showarrow=False,
                          xanchor="right", yanchor="bottom",
                          font=dict(color="#888"))
    fig_cb.update_traces(textposition="top center", showlegend=False)
    fig_cb.update_layout(height=520, margin=dict(t=30, b=0))
    st.plotly_chart(fig_cb, use_container_width=True)
    st.caption(
        "Sobre la diagonal, el ahorro de ENS paga la medida; bajo la diagonal, "
        "no. El tamaño del punto es la ENS evitada en kWh, así que un punto "
        "grande y bajo es una medida que sirve técnicamente pero no se "
        "justifica sola por costo de ENS.")


# ════════════════════════════════════════════════════════════
# Comparación entre medidas y entre escenarios
# ════════════════════════════════════════════════════════════

# Cómo se puede mirar una magnitud. La normalizada es el DEFAULT a propósito:
# la ENS del caso base va de ~1.900 a ~105.000 kWh según el escenario, así que
# una tabla de valores absolutos ordena los escenarios por dureza y esconde lo
# que uno quiere ver, que es qué tan bien le fue a la medida en cada uno.
ESCALAS = {
    "% del caso base": "pct",
    "Δ contra el caso base": "delta",
    "Valor absoluto": "abs",
}


def _severidad(base):
    """Qué tan duro es cada escenario, antes de aplicar ninguna medida.

    Es el contexto que faltaba para leer cualquier comparación entre
    escenarios: sin esto, un escenario aparece 'malo para todas las medidas'
    sin que se vea que simplemente le tocaron más fallas.
    """
    if base.empty:
        return
    b = base.sort_values("ens", ascending=False)
    fig = px.bar(b, x=[f"esc {int(e)}" for e in b["escenario"]], y="ens",
                 labels={"y": "ENS del caso base [kWh]", "x": ""},
                 hover_data={"saidi": ":.3f", "saifi": ":.3f"})
    fig.update_traces(marker_color="#9e9e9e")
    fig.update_layout(height=230, margin=dict(t=10, b=0),
                      yaxis_title="ENS del caso base [kWh]")
    st.plotly_chart(fig, use_container_width=True)
    peor, mejor = b.iloc[0], b.iloc[-1]
    st.caption(
        f"Sin ninguna medida, el escenario {int(peor['escenario'])} deja "
        f"{peor['ens']:,.0f} kWh sin servir y el {int(mejor['escenario'])} solo "
        f"{mejor['ens']:,.0f}: **{peor['ens']/mejor['ens']:.1f} veces más**. Por eso "
        f"las tablas de esta pestaña vienen normalizadas por defecto.")


def _pivote(dfe, ind, escala, base):
    """Tabla medida × escenario en la escala pedida, con su formato.

    Devuelve (pivote, formato, cmap, sentido). El Δ solo existe para los
    indicadores que tienen caso base; para los resultados físicos (capacidad
    instalada, maniobras) no hay contra qué restar y se cae a absoluto.
    """
    col_base = f"{ind}_base"
    if escala == "abs" or col_base not in dfe.columns:
        piv = dfe.pivot_table(index="medida", columns="escenario", values=ind)
        fmt = "{:,.3f}" if ind in ("saidi", "saifi", "asai") else "{:,.1f}"
        return piv, fmt, _cmap(ind), "abs"

    # Mejora = base − estrategia, así que un número ALTO siempre es bueno,
    # aunque el indicador de origen sea de los que conviene minimizar.
    d = dfe.assign(_v=dfe[col_base] - dfe[ind])
    if escala == "pct":
        d["_v"] = np.where(dfe[col_base] > 0, d["_v"] / dfe[col_base], np.nan)
        fmt = "{:.1%}"
    else:
        fmt = "{:,.3f}" if ind in ("saidi", "saifi", "asai") else "{:,.1f}"
    piv = d.pivot_table(index="medida", columns="escenario", values="_v")
    return piv, fmt, "RdYlGn", "mejora"


def _tabla_medida_escenario(dfe, ind, escala, base):
    """Heatmap medida × escenario, ordenado por el promedio."""
    piv, fmt, cmap, sentido = _pivote(dfe, ind, escala, base)
    if piv.empty:
        st.info("No hay datos para esa magnitud.")
        return
    piv = piv.assign(PROMEDIO=piv.mean(axis=1))
    # De mejor a peor: la pregunta es "¿cuáles sirven?", y una tabla
    # alfabética obliga a recorrerla entera para contestarla.
    piv = piv.sort_values("PROMEDIO", ascending=(sentido != "mejora"))
    nombres = pd.Index([_nombre(m) for m in piv.index], name="Medida")
    if nombres.is_unique:
        piv.index = nombres
    else:
        # Si dos claves comparten una etiqueta legible, conservar la original
        # evita colisiones en Styler y permite distinguir ambas corridas.
        piv.index = pd.MultiIndex.from_arrays(
            [nombres, piv.index], names=["Medida", "Caso"])
    piv.columns = [c if c == "PROMEDIO" else f"esc {c}" for c in piv.columns]

    cols_esc = [c for c in piv.columns if c != "PROMEDIO"]
    st.dataframe(
        piv.style
        .background_gradient(cmap=cmap, subset=cols_esc, axis=None)
        .background_gradient(cmap=cmap, subset=["PROMEDIO"])
        .format(fmt, na_rep="—"),
        use_container_width=True)
    if sentido == "mejora":
        st.caption("Verde = mejora frente a no hacer nada, en el mismo "
                   "escenario. El color es comparable entre celdas: toda la "
                   "tabla comparte escala.")


def _estabilidad_ranking(dfe, ind, medidas):
    """¿El orden de mérito entre medidas depende del escenario?

    Cada línea es una medida y su posición en el ranking de ese escenario.
    Líneas planas = el orden es robusto y elegir por un escenario
    representativo es defendible. Líneas que se cruzan = la medida que
    conviene depende de qué falle, y el representativo se vuelve frágil.
    """
    from helpers_inversion import obtener_color_estrategia

    col = f"d_{ind}" if f"d_{ind}" in dfe.columns else ind
    piv = dfe.pivot_table(index="medida", columns="escenario", values=col)
    if piv.shape[1] < 2:
        st.info("Hace falta más de un escenario para ver estabilidad.")
        return
    # rank 1 = la mejor de ese escenario. En las columnas de mejora (d_*) el
    # mejor es el mayor; en un indicador crudo depende de su sentido, que es
    # justo lo que guarda ETIQUETAS: ASAI se ordena al revés que SAIDI, y
    # asumir "menor es mejor" para todos dejaría el ranking de ASAI dado
    # vuelta sin que nada avise.
    asc = (False if col.startswith("d_")
           else ETIQUETAS.get(col, ("", "menor"))[1] == "menor")
    rk = piv.rank(axis=0, ascending=asc, method="min")
    orden = rk.mean(axis=1).sort_values().index

    fig = go.Figure()
    for m in orden:
        fig.add_scatter(
            x=[f"esc {c}" for c in rk.columns], y=rk.loc[m],
            mode="lines+markers", name=_nombre(m),
            line=dict(color=obtener_color_estrategia(_nombre_fig(m)), width=2),
            marker=dict(size=7),
            hovertemplate=f"{_nombre(m)}<br>%{{x}} · puesto %{{y:.0f}}<extra></extra>")
    fig.update_yaxes(autorange="reversed", title="Puesto en el ranking",
                     dtick=1)
    fig.update_layout(height=460, margin=dict(t=20, b=0),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    salto = (rk.max(axis=1) - rk.min(axis=1)).sort_values(ascending=False)
    estables = int((salto <= 1).sum())
    st.caption(
        f"{estables} de {len(salto)} medidas se mueven a lo más un puesto "
        f"entre escenarios. La más inestable es **{_nombre(salto.index[0])}**, "
        f"que oscila {int(salto.iloc[0])} puestos.")
    return salto


def _agregadores(dfe, ind="ens"):
    """La misma medida resumida con cuatro criterios distintos de agregación.

    El promedio es el valor esperado y es lo correcto si uno va a enfrentar
    muchos años parecidos a estos escenarios. Pero un plan de resiliencia se
    juzga también por cómo se comporta cuando sale mal, y ahí el promedio
    esconde justo lo que importa:

      Mediana       — no la mueve un escenario extremo.
      Peor caso     — criterio conservador (maximin): la medida que mejor se
                      porta en su peor escenario.
      Arrepentimiento máximo — cuánto se pierde, en el peor escenario, por
                      haber elegido esta medida y no la que resultó mejor en
                      ESE escenario. Es el criterio de decisión bajo
                      incertidumbre profunda, que es la situación de la tesis:
                      no se sabe qué escenario climático va a ocurrir.

    Si los cuatro rankings coinciden, decidir por el promedio es inocuo. Si no,
    la elección depende del criterio y hay que declararlo.
    """
    col = f"d_{ind}_pct"
    if col not in dfe.columns:
        return pd.DataFrame()
    piv = dfe.pivot_table(index="medida", columns="escenario", values=col)
    if piv.empty:
        return pd.DataFrame()
    # Arrepentimiento: contra la MEJOR medida de cada escenario, columna a
    # columna. No contra el óptimo global, que no existe acá.
    regret = piv.max(axis=0) - piv

    out = pd.DataFrame({
        "Promedio": piv.mean(axis=1),
        "Mediana": piv.median(axis=1),
        "Peor escenario": piv.min(axis=1),
        "Arrepentimiento máx.": regret.max(axis=1),
    })
    out["#Promedio"] = out["Promedio"].rank(ascending=False).astype(int)
    out["#Peor caso"] = out["Peor escenario"].rank(ascending=False).astype(int)
    out["#Arrepentim."] = out["Arrepentimiento máx."].rank(ascending=True).astype(int)
    out.insert(0, "Medida", [_nombre(m) for m in out.index])
    return out.sort_values("Promedio", ascending=False).reset_index(drop=True)


def _bloque_agregacion(dfe, ind="ens"):
    """¿Basta con el promedio para elegir entre medidas?"""
    agg = _agregadores(dfe, ind)
    if agg.empty:
        return
    st.dataframe(
        agg.style
        .format({"Promedio": "{:.1%}", "Mediana": "{:.1%}",
                 "Peor escenario": "{:.1%}", "Arrepentimiento máx.": "{:.1%}"},
                na_rep="—")
        .background_gradient(cmap="RdYlGn", subset=["Promedio", "Mediana",
                                                    "Peor escenario"])
        .background_gradient(cmap="RdYlGn_r", subset=["Arrepentimiento máx."]),
        use_container_width=True, hide_index=True)

    # El veredicto se calcula, no se afirma: si el podio cambia entre
    # criterios, decidir por el promedio SÍ altera la respuesta.
    top = {c: list(agg.sort_values(c, ascending=(c == "#Arrepentim."))["Medida"])[:3]
           for c in ("#Promedio", "#Peor caso", "#Arrepentim.")}
    podios = {tuple(v) for v in top.values()}
    if len(podios) == 1:
        st.success(
            f"Los tres criterios dan el mismo podio ({', '.join(top['#Promedio'])}), "
            f"así que decidir por el promedio no cambia la respuesta.")
    else:
        st.warning(
            "El podio **cambia según el criterio**: por promedio es "
            f"{', '.join(top['#Promedio'])}; por peor caso, "
            f"{', '.join(top['#Peor caso'])}; por arrepentimiento mínimo, "
            f"{', '.join(top['#Arrepentim.'])}. Con esta dispersión hay que "
            "declarar con qué criterio se eligió, porque el promedio solo no "
            "sostiene la decisión.")
    st.caption(
        "Todo en % de la ENS del caso base de cada escenario. "
        "**Peor escenario** es el criterio conservador; **arrepentimiento "
        "máximo** es cuánto se pierde, en el peor caso, por haber elegido esta "
        "medida en vez de la que resultó mejor en ese escenario — menos es "
        "mejor, y es el criterio habitual bajo incertidumbre profunda.")


def seccion_comparar(dfe, dce, base, detalle, nombre, medidas):
    """Cada medida por separado, medida × escenario, con o sin combinaciones."""
    st.subheader("Comparar medidas")

    hay_base = "d_ens" in dfe.columns
    if not hay_base:
        st.caption("Sin baselines de esta variante solo se pueden mostrar "
                   "valores absolutos, que no son comparables entre escenarios.")

    # Las combinaciones entran como una fila más, pero solo si se acepta el
    # recorte: corrieron en menos escenarios que las individuales.
    unido, escs_comunes = (_unir_combinadas(dfe, dce) if not dce.empty
                           else (dfe, None))
    if escs_comunes:
        incluir = st.checkbox(
            f"Incluir las combinaciones (recorta la comparación a los "
            f"{len(escs_comunes)} escenarios en que corrieron)", False,
            key="cmp_incluir_comb",
            help="Las combinaciones corrieron en menos escenarios que las "
                 "medidas sueltas. Para compararlas hay que mirar los mismos "
                 "escenarios en ambos casos, o la diferencia saldría de la "
                 "severidad de los escenarios y no de las medidas.")
        if incluir:
            dfe = unido
            st.caption(f"Comparación restringida a los escenarios "
                       f"{escs_comunes}. Las corridas combinadas sospechosas "
                       f"quedan fuera.")
    medidas = sorted(dfe["medida"].unique())

    if not detalle:
        st.markdown("#### Qué tan duro es cada escenario")
        _severidad(base)

        st.markdown("#### Cuánta ENS evita cada medida, escenario por escenario")
        _tabla_medida_escenario(dfe, "ens", "pct" if hay_base else "abs", base)

        st.markdown("#### ¿Cambia el orden de mérito según el escenario?")
        _estabilidad_ranking(dfe, "ens", medidas)

        if hay_base:
            st.markdown("#### ¿Basta con el promedio para elegir?")
            _bloque_agregacion(dfe, "ens")

        st.caption("En **Detalle** están las demás magnitudes, la dispersión "
                   "por medida y la tabla completa.")
        return

    # ── detalle ──────────────────────────────────────────────────────
    col_a1, col_a2, col_a3 = st.columns([1, 2, 1.4])
    grupo = col_a1.radio("Ver", ["Indicadores", "Resultados físicos"],
                         key="barr_grupo_a",
                         help="Los indicadores son SAIDI/SAIFI/ENS/ASAI y el "
                              "objetivo; los resultados físicos son capacidad "
                              "instalada, equipos, energía y maniobras.")
    # Las columnas derivadas (capacidad atribuible a resiliencia) van junto a
    # la capacidad total, no aparte: la pregunta "¿cuánto de esto es por
    # resiliencia?" se hace mirando la total.
    agregados_vista = []
    for a in AGREGADOS:
        agregados_vista.append(a)
        for extra in (f"{a[:-3]}_res_kW",) if a.endswith("_kW") else ():
            if extra in dfe.columns:
                agregados_vista.append(extra)
    opciones = (INDICADORES if grupo == "Indicadores"
                else [a for a in agregados_vista if a in dfe.columns])
    # Por defecto ENS y no `objetivo`: el objetivo no tiene caso base contra
    # el cual normalizar (el MILP del caso base no guarda su fn_objetivo), así
    # que abrir ahí dejaría la vista en valores absolutos, que es justo lo que
    # no se puede comparar entre escenarios.
    ind = col_a2.selectbox("Magnitud", opciones, key=f"barr_ind_a_{grupo}",
                           index=opciones.index("ens") if "ens" in opciones else 0,
                           format_func=_etiqueta)
    hay_base_ind = hay_base and f"{ind}_base" in dfe.columns
    escalas = list(ESCALAS) if hay_base_ind else ["Valor absoluto"]
    escala = ESCALAS[col_a3.selectbox(
        "Escala", escalas, key="barr_escala_a",
        help=None if hay_base_ind else
        f"{_etiqueta(ind)} no tiene caso base con el que comparar, así que "
        f"solo se puede ver en valor absoluto.")]

    st.markdown(f"**{_etiqueta(ind)}** — medida × escenario")
    _tabla_medida_escenario(dfe, ind, escala, base)

    with st.expander("Qué tan duro es cada escenario"):
        _severidad(base)

    st.markdown("#### Dispersión entre escenarios")
    col_disp = ind if escala == "abs" else None
    if col_disp is None:
        piv, _f, _c, _s = _pivote(dfe, ind, escala, base)
        largo = piv.stack().rename("valor").reset_index()
        largo.columns = ["medida", "escenario", "valor"]
        fig = px.box(largo, x="medida", y="valor", points="all",
                     hover_data=["escenario"],
                     labels={"medida": "", "valor": f"{_etiqueta(ind)} evitado"})
    else:
        fig = px.box(dfe, x="medida", y=ind, points="all",
                     hover_data=["escenario"],
                     labels={"medida": "", ind: _etiqueta(ind)})
    fig.update_xaxes(tickvals=medidas, ticktext=[_nombre(m) for m in medidas])
    fig.update_layout(height=420, margin=dict(t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cada punto es un escenario. Una caja alta y angosta es una "
               "medida que rinde parecido pase lo que pase; una caja larga "
               "es una medida cuyo resultado depende de qué falle.")

    st.markdown("#### Estabilidad del orden de mérito")
    salto = _estabilidad_ranking(dfe, ind if grupo == "Indicadores" else "ens",
                                 medidas)
    if salto is not None:
        st.dataframe(
            pd.DataFrame({"Medida": [_nombre(m) for m in salto.index],
                          "Puestos que oscila": salto.values}),
            use_container_width=True, hide_index=True)

    with st.expander("Tabla cruda (todas las columnas)"):
        st.dataframe(dfe, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Descargar resumen individual (CSV)",
        dfe.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"barrido_{nombre}_individual.csv", mime="text/csv")


# ════════════════════════════════════════════════════════════
# Robustez de la decisión
# ════════════════════════════════════════════════════════════

# De dónde sale "en qué elementos concretos actuó la medida". `plan` cubre la
# inversión fija (qué línea se refuerza, cuánta capacidad va en cada barra) y
# `ubicacion` cubre lo que varía en el tiempo (qué enlace se maniobró, a qué
# barra fue cada unidad móvil). Sin esto solo se sabe CUÁNTO se instaló, no
# DÓNDE, que es justo lo que decide si dos escenarios piden la misma obra.
ELEMENTO_MEDIDA = {
    "dg_sol":         ("plan", "CAP_DG_sol_kW", "Barra", "kW"),
    "dg_eol":         ("plan", "CAP_DG_eol_kW", "Barra", "kW"),
    "dg_fos":         ("plan", "CAP_DG_fos_kW", "Barra", "kW"),
    "ess_4h":         ("plan", "CAP_ESS_kWh_4h", "Barra", "kWh"),
    "ess_5h":         ("plan", "CAP_ESS_kWh_5h", "Barra", "kWh"),
    "ess_6h":         ("plan", "CAP_ESS_kWh_6h", "Barra", "kWh"),
    "refuerzo":       ("plan", "x_ref", "Línea", "—"),
    "veg_mgmt":       ("plan", "x_veg", "Línea", "—"),
    "soterramiento":  ("plan", "x_ug", "Línea", "—"),
    "inv_tie":        ("plan", "y_tie", "Enlace", "—"),
    "inv_switch_rcs": ("plan", "y_rcs", "Línea", "—"),
    "inv_switch_ms":  ("plan", "y_ms", "Línea", "—"),
    # Móviles y reconfiguración no tienen inversión por elemento: lo que se
    # decide es a dónde van y qué se maniobra, que vive en `ubicacion`.
    "meg":            ("ubicacion", "meg_horas_por_bus", "Barra", "horas"),
    "mess":           ("ubicacion", "mess_horas_por_bus", "Barra", "horas"),
    "reconfig_topo":  ("ubicacion", "tie_maniobras", "Enlace", "maniobras"),
}


def _clave_interna_medida(medida):
    """Convierte una etiqueta visible a la clave interna del barrido.

    Los artefactos históricos pueden guardar ``"DG eolica"`` mientras los
    diccionarios de la vista usan ``"dg_eol"``. Se ignoran tildes,
    mayúsculas y espacios para que ambas representaciones sean compatibles.
    """
    if medida in ELEMENTO_MEDIDA:
        return medida

    def normalizar(valor):
        texto = unicodedata.normalize("NFKD", str(valor))
        return " ".join(
            "".join(c for c in texto if not unicodedata.combining(c))
            .casefold().split()
        )

    buscada = normalizar(medida)
    for clave, etiqueta in MEDIDAS.items():
        if normalizar(etiqueta) == buscada:
            return clave
    return medida


def _elemento_medida(medida):
    """Devuelve la especificación aceptando clave interna o etiqueta."""
    return ELEMENTO_MEDIDA.get(_clave_interna_medida(medida))


@st.cache_data(show_spinner="Leyendo las carteras de inversión...")
def cargar_planes(nombre, _sello, subdir="individual"):
    """Qué eligió el solver en cada corrida, elemento por elemento.

    En `individual/` cada corrida tiene una sola medida encendida, así que se
    lee la variable que le corresponde. En `combinadas/` el solver pudo usar
    CUALQUIERA de las medidas del combo, así que se leen todas las claves y se
    conserva solo lo que quedó distinto de cero — es la única forma de saber
    qué instaló la combinación y dónde.

    Los .json pesan ~18 kB, así que leer las 125 es instantáneo; lo que no se
    puede es hacerlo en cada interacción de widget, de ahí la caché.
    """
    d = ruta_barrido(nombre) / subdir
    if not d.exists():
        return pd.DataFrame()
    combinada = subdir == "combinadas"
    filas = []
    for ruta in sorted(d.glob("*/esc_*.json")):
        try:
            reg = json.loads(ruta.read_text(encoding="utf-8"))
        except Exception:                           # noqa: BLE001
            continue
        if reg.get("status") != "optimal":
            continue
        if combinada:
            claves = [(m, s) for m, s in ELEMENTO_MEDIDA.items()
                      if m in set(reg.get("medidas_on") or [])]
        else:
            spec = _elemento_medida(reg.get("medida"))
            claves = [(reg.get("medida"), spec)] if spec else []
        for m, (fuente, clave, _etiq, _uni) in claves:
            for elem, val in ((reg.get(fuente) or {}).get(clave) or {}).items():
                filas.append({"caso": ruta.parent.name, "medida": m,
                              "escenario": int(reg["escenario"]),
                              "elemento": int(elem), "valor": float(val)})
    return pd.DataFrame(filas)


def _despliegue_combo(planes_c, combo, escs):
    """Qué instaló la combinación y en qué barras o líneas.

    Complementa a la tabla de retención, que dice CUÁNTO sobrevive de cada
    medida pero no dónde queda. Sin esto, una capacidad de 18.108 kW de DG
    solar no se puede llevar a un plan de obras.
    """
    if planes_c.empty:
        return pd.DataFrame()
    sub = planes_c[(planes_c["caso"] == combo)
                   & (planes_c["escenario"].isin(escs))
                   & (planes_c["valor"].abs() > 1e-9)]
    if sub.empty:
        return pd.DataFrame()
    # Denominador: los escenarios de la combinación, no aquellos donde hubo
    # algún despliegue. Si en uno el solver no instaló nada, "en todos los
    # escenarios" tiene que seguir significando los cuatro.
    n_esc = len(escs)
    filas = []
    for m, s in sub.groupby("medida"):
        _f, _c, etiq, uni = _elemento_medida(m)
        veces = s.groupby("elemento")["escenario"].nunique()
        nucleo = sorted(veces[veces >= n_esc].index)
        filas.append({
            "Medida": _nombre(m),
            "Unidad": uni,
            "Total (prom.)": float(s.groupby("escenario")["valor"].sum().mean()),
            "Elementos": len(veces),
            # El tipo de elemento va en el texto y no como nombre de columna:
            # cada medida usa uno distinto (barras, líneas, enlaces) y una
            # columna por tipo dejaría la tabla llena de huecos.
            "Dónde": f"{etiq}s " + ", ".join(str(e) for e in sorted(veces.index)),
            "En todos los escenarios": (", ".join(str(e) for e in nucleo)
                                        if nucleo else "—"),
        })
    return pd.DataFrame(filas).sort_values("Total (prom.)", ascending=False)


def _veredicto_cv(cv):
    if not np.isfinite(cv):
        return "—"
    if cv < 0.05:
        return "✅ estable"
    if cv < 0.25:
        return "🟡 moderada"
    return "🔴 depende del escenario"


def _estabilidad_magnitud(dfe, escenarios):
    """¿Cuánto instalar cambia según el escenario?

    Se mide sobre la variable de decisión de cada medida, con el coeficiente
    de variación entre escenarios: es adimensional, así que kW, km y maniobras
    quedan en la misma escala y se pueden ordenar juntas.
    """
    filas = []
    for m, (col, unidad) in VAR_MEDIDA.items():
        s = dfe[dfe["medida"] == m]
        if s.empty or col not in dfe.columns:
            continue
        v = s[col].astype(float)
        if v.abs().max() <= 1e-9:
            continue                                # la medida no instaló nada
        media, sd = v.mean(), v.std(ddof=0)
        cv = sd / abs(media) if abs(media) > 1e-9 else np.nan
        filas.append({
            "Medida": _nombre(m),
            "Decisión": f"{_etiqueta(col)}",
            "Unidad": unidad,
            "Promedio": media, "Mínimo": v.min(), "Máximo": v.max(),
            "CV": cv, "Veredicto": _veredicto_cv(cv),
        })
    # Sin filas no hay columna que ordenar: un barrido sin variables de
    # decisión guardadas llegaría acá con la lista vacía.
    return pd.DataFrame(filas).sort_values("CV") if filas else pd.DataFrame()


def _cartera_robusta(planes, n_corridas):
    """En cuántos escenarios se repite cada elemento elegido.

    Un elemento elegido en los N escenarios es núcleo robusto: la obra se
    justifica pase lo que pase. Uno elegido en uno o dos es oportunismo del
    escenario, y comprometerlo por el representativo es un riesgo.

    `n_corridas` son los escenarios en que la medida se CORRIÓ, no aquellos en
    que instaló algo. La diferencia importa: MESS solo instala en 1 de 10
    escenarios, y contra ese 1 sus barras parecerían elegidas "siempre".
    """
    if planes.empty:
        return pd.DataFrame(), pd.DataFrame()
    n = dict(n_corridas)
    g = (planes[planes["valor"].abs() > 1e-9]
         .groupby(["medida", "elemento"])
         .agg(veces=("escenario", "nunique"), valor=("valor", "mean"))
         .reset_index())
    g["n_esc"] = [n.get(m, 0) for m in g["medida"]]
    g["frecuencia"] = np.where(g["n_esc"] > 0, g["veces"] / g["n_esc"], np.nan)

    resumen = []
    for m, sub in g.groupby("medida"):
        nucleo = sub[sub["frecuencia"] >= 0.999]
        ocasional = sub[sub["frecuencia"] < 0.999]
        resumen.append({
            "Medida": _nombre(m),
            "Escenarios en que instala": f"{int(sub['veces'].max())} de {n.get(m, 0)}",
            "Elementos usados": len(sub),
            "Siempre (núcleo robusto)": len(nucleo),
            "Solo a veces": len(ocasional),
            "Núcleo": ", ".join(str(e) for e in sorted(nucleo["elemento"]))[:60]
                      or "—",
        })
    if not resumen:
        return g, pd.DataFrame()
    return g, pd.DataFrame(resumen).sort_values(
        "Siempre (núcleo robusto)", ascending=False)


def seccion_robustez(dfe, reps, planes, escenarios, medidas, detalle, nombre):
    """¿La decisión de inversión depende del escenario que se mire?"""
    st.subheader("Robustez de la decisión")
    st.caption(
        "La fase 2 elige un escenario representativo por medida. Esta pestaña "
        "verifica si esa elección importa: si la decisión de inversión es la "
        "misma en todos los escenarios, el representativo es inocuo; si "
        "cambia, comprometerse con uno solo es un riesgo.")

    # ── ¿cuánto instalar? ────────────────────────────────────────────
    st.markdown("#### ¿Cambia CUÁNTO se instala?")
    est = _estabilidad_magnitud(dfe, escenarios)
    if est.empty:
        st.info("Ninguna medida registró una variable de decisión comparable.")
    else:
        st.dataframe(
            est.style
            .format({"Promedio": "{:,.1f}", "Mínimo": "{:,.1f}",
                     "Máximo": "{:,.1f}", "CV": "{:.1%}"}, na_rep="—")
            .background_gradient(cmap="RdYlGn_r", subset=["CV"], vmin=0, vmax=0.5),
            use_container_width=True, hide_index=True)
        st.caption(
            "**CV** = desviación estándar entre escenarios dividida por el "
            "promedio. Bajo 5% el escenario no cambia la decisión; sobre 25% "
            "la magnitud a instalar depende de qué falle.")

    # ── ¿dónde invertir? ─────────────────────────────────────────────
    st.markdown("#### ¿Cambia DÓNDE se invierte?")
    n_corridas = dfe.groupby("medida")["escenario"].nunique().to_dict()
    detalle_el, resumen_el = _cartera_robusta(planes, n_corridas)
    if resumen_el.empty:
        st.info("No hay carteras por elemento guardadas en este barrido.")
    else:
        st.dataframe(resumen_el, use_container_width=True, hide_index=True)
        st.caption(
            "**Núcleo robusto** = elementos (líneas, barras o enlaces) que el "
            "solver eligió en TODOS los escenarios. Esa parte de la cartera se "
            "puede comprometer sin depender del representativo; el resto es "
            "específico del escenario.")

    if not detalle:
        st.caption("En **Detalle** está el mapa elemento × escenario de cada "
                   "medida y el ranking de cercanía al promedio.")
        return

    # ── detalle: mapa elemento × escenario ───────────────────────────
    st.divider()
    st.markdown("#### Dónde actuó cada medida, escenario por escenario")
    con_plan = sorted(planes["medida"].unique()) if not planes.empty else []
    if con_plan:
        m_sel = st.selectbox("Medida", con_plan, format_func=_nombre,
                             key="rob_medida")
        sub = planes[planes["medida"] == m_sel]
        piv = sub.pivot_table(index="elemento", columns="escenario",
                              values="valor", fill_value=0.0)
        _f, _c, etiq_el, unidad = _elemento_medida(m_sel)
        piv.index = [f"{etiq_el} {i}" for i in piv.index]
        piv.columns = [f"esc {c}" for c in piv.columns]
        # Denominador: los escenarios en que la medida se CORRIÓ, no aquellos
        # en que instaló algo (ver `_cartera_robusta`).
        piv["FRECUENCIA"] = (sub[sub["valor"].abs() > 1e-9]
                             .groupby("elemento")["escenario"].nunique()
                             .reindex([int(i.split()[-1]) for i in piv.index])
                             .fillna(0).values) / max(n_corridas.get(m_sel, 1), 1)
        st.dataframe(
            piv.style
            .background_gradient(cmap="Blues",
                                 subset=[c for c in piv.columns
                                         if c != "FRECUENCIA"], axis=None)
            .background_gradient(cmap="RdYlGn", subset=["FRECUENCIA"],
                                 vmin=0, vmax=1)
            .format("{:,.1f}", subset=[c for c in piv.columns
                                       if c != "FRECUENCIA"])
            .format("{:.0%}", subset=["FRECUENCIA"]),
            use_container_width=True)
        st.caption(f"Valor en {unidad}. La última columna es en qué fracción "
                   f"de los escenarios ese elemento fue elegido.")

    # ── detalle: el representativo ───────────────────────────────────
    st.divider()
    st.markdown("#### Escenario representativo de cada medida")
    if not reps:
        st.info("Falta la fase 2. Correr: "
                "`python run_barrido_medidas.py --escenarios ... --fases 2`")
        return
    por_medida = reps.get("por_medida", {})
    # El criterio "multi" es una distancia sobre varios indicadores a la vez,
    # así que no tiene una columna propia que mostrar: se reporta la ENS, que
    # es la que más se lee en la tesis.
    col_crit = reps["criterio"] if reps["criterio"] in INDICADORES else "ens"
    filas_r = []
    for m, v in sorted(por_medida.items()):
        prom = v["promedios"].get(col_crit, np.nan)
        val = v["valores_rep"].get(col_crit, np.nan)
        filas_r.append({
            "Medida": _nombre(m),
            "Esc. representativo": v["escenario"],
            f"Promedio ({col_crit})": prom,
            f"Valor del representativo ({col_crit})": val,
            "Desviación": abs(val - prom) / abs(prom) if prom else np.nan,
        })
    st.dataframe(
        pd.DataFrame(filas_r).style.format({
            f"Promedio ({col_crit})": "{:,.1f}",
            f"Valor del representativo ({col_crit})": "{:,.1f}",
            "Desviación": "{:.2%}"}),
        use_container_width=True, hide_index=True)

    m_sel2 = st.selectbox("Ranking de cercanía al promedio", medidas,
                          format_func=_nombre, key="barr_medida_b")
    v = por_medida.get(m_sel2)
    if v:
        rk = pd.DataFrame(v["ranking"])
        rk.columns = ["Escenario", "Distancia al promedio"]
        st.dataframe(rk, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# Medidas combinadas
# ════════════════════════════════════════════════════════════

# La variable de decisión con la que se mide "cuánto se desplegó" de cada
# medida, y su unidad. Es lo que permite comparar el despliegue de una medida
# SOLA contra su despliegue DENTRO de una combinación: si el solver instala
# menos de una medida cuando tiene las otras disponibles, esas dos compiten
# por la misma ENS; si instala más, se potencian.
VAR_MEDIDA = {
    "dg_sol":         ("cap_dg_sol_kW",  "kW"),
    "dg_eol":         ("cap_dg_eol_kW",  "kW"),
    "dg_fos":         ("cap_dg_fos_kW",  "kW"),
    "ess_4h":         ("cap_ess_4h_kWh", "kWh"),
    "ess_5h":         ("cap_ess_5h_kWh", "kWh"),
    "ess_6h":         ("cap_ess_6h_kWh", "kWh"),
    "meg":            ("cap_meg_MW",     "MW"),
    "mess":           ("cap_mess_MWh",   "MWh"),
    "inv_switch_rcs": ("n_rcs",          "switches"),
    "inv_switch_ms":  ("n_ms",           "switches"),
    "inv_tie":        ("n_tie_nuevas",   "enlaces"),
    "veg_mgmt":       ("km_veg",         "km"),
    "refuerzo":       ("km_ref",         "km"),
    "soterramiento":  ("km_ug",          "km"),
    "reconfig_topo":  ("maniobras_tie",  "maniobras"),
    "edrp":           ("E_edrp",         "kWh desplazados"),
}


def _lectura_retencion(sola, combo):
    """Qué significa la razón entre el despliegue solo y el combinado."""
    if sola <= 1e-9:
        return "🆕 solo aparece en conjunto"
    r = combo / sola
    if r >= 1.10:
        return "🔼 se despliega más en conjunto"
    if r >= 0.90:
        return "➖ igual"
    if r >= 0.05:
        return "🔽 se despliega menos"
    return "❌ la desplazan otras"


def _retencion(dfe, dce, combo, escs):
    """Despliegue de cada medida sola frente a su despliegue en la combinación.

    Es la pregunta "¿cuánto de la medida se sigue usando cuando están todas?",
    y se responde con las MISMAS variables de decisión del MILP, no con una
    proxy. Solo se comparan los escenarios en que existen las dos corridas:
    dos escenarios distintos tienen otra secuencia de fallas.
    """
    sub_c = dce[(dce["combinacion"] == combo) & (dce["escenario"].isin(escs))]
    if sub_c.empty:
        return pd.DataFrame()
    en_combo = set(str(sub_c["medidas"].iloc[0]).split("+"))

    filas = []
    for m, (col, unidad) in VAR_MEDIDA.items():
        if col not in dfe.columns or m not in en_combo:
            continue
        s = dfe[(dfe["medida"] == m) & (dfe["escenario"].isin(escs))]
        if s.empty:
            continue
        sola, junto = float(s[col].mean()), float(sub_c[col].mean())
        if max(abs(sola), abs(junto)) <= 1e-9:
            continue
        filas.append({
            "Medida": _nombre(m),
            "Unidad": unidad,
            "Sola": sola,
            "En combinación": junto,
            "Retención": junto / sola if sola > 1e-9 else np.nan,
            "Lectura": _lectura_retencion(sola, junto),
        })
    if not filas:
        return pd.DataFrame()
    return pd.DataFrame(filas).sort_values("Retención", ascending=False,
                                           na_position="first")


def _solapamiento(dfe, dce, combo, escs):
    """Suma de los beneficios individuales frente al beneficio real de la
    combinación, escenario por escenario.

    Si las medidas fueran independientes, las dos cantidades coincidirían.
    Que la suma sea varias veces mayor significa que están evitando la MISMA
    ENS: el beneficio no se suma, se solapa.
    """
    filas = []
    for e in sorted(escs):
        c = dce[(dce["combinacion"] == combo) & (dce["escenario"] == e)]
        ind = dfe[dfe["escenario"] == e]
        if c.empty or ind.empty:
            continue
        c = c.iloc[0]
        base_ens = float(c["ens_base"])
        if not base_ens > 0:
            continue
        filas.append({
            "escenario": int(e),
            "Combinación": float(c["d_ens"]) / base_ens,
            "Suma de individuales": float(ind["d_ens"].sum()) / base_ens,
            "Mejor individual": float(ind["d_ens"].max()) / base_ens,
            "mejor_medida": _nombre(ind.loc[ind["d_ens"].idxmax(), "medida"]),
            "costo_combo": float(c.get("cst_costo_total", np.nan)),
            "costo_mejor": float(ind.loc[ind["d_ens"].idxmax(), "cst_costo_total"])
            if "cst_costo_total" in ind else np.nan,
            "d_ens_combo": float(c["d_ens"]),
            "d_ens_mejor": float(ind["d_ens"].max()),
        })
    return pd.DataFrame(filas)


def _comparar_combos(dfe, val, escs):
    """Todas las combinaciones, una al lado de otra, sobre escenarios comunes.

    El selector de la pestaña muestra una combinación a la vez, que sirve para
    diseccionarla pero no para elegir entre ellas. Esta tabla contesta la
    pregunta anterior: de todas las carteras probadas, ¿cuál conviene?

    `escs` es la INTERSECCIÓN de los escenarios de todas las combinaciones. Si
    una corrió en cinco y otra en cuatro, promediar cada una sobre los suyos
    compararía escenarios de distinta dureza en vez de carteras.
    """
    if val.empty or not escs:
        return pd.DataFrame()
    ind = dfe[dfe["escenario"].isin(escs)]
    pct_medida = ind.groupby("medida")["d_ens_pct"].mean().to_dict()

    filas = []
    for combo, s in val[val["escenario"].isin(escs)].groupby("combinacion"):
        ms = [m for m in str(s["medidas"].iloc[0]).split("+") if m]
        f = {
            "Combinación": _nombre_combo(combo),
            "Medidas": len(ms),
            "ENS evitada [%]": float((s["d_ens"] / s["ens_base"]).mean()),
            # Cuánto evitarían esas mismas medidas por separado. Si es mucho
            # mayor que la columna anterior, la combinación está saturando y
            # las medidas se solapan; si es menor, se potencian.
            "Suma de sus medidas [%]": float(sum(pct_medida.get(m, 0.0)
                                                 for m in ms)),
            "SAIDI evitado": float(s["d_saidi"].mean()),
            "Costo total [USD/año]": float(s["cst_costo_total"].mean()),
            "Ahorro neto [USD/año]": float(s["cst_ahorro_neto"].mean()),
        }
        d_ens, costo = float(s["d_ens"].mean()), f["Costo total [USD/año]"]
        f["B/C [-]"] = (float(s["cst_ahorro_ens"].mean()) / costo
                        if costo > 0 else np.nan)
        f["USD/kWh evitado"] = float(_unitario(costo, d_ens,
                                                UMBRAL_MEJORA["d_ens"]))
        # Una estrategia con costo neto negativo no es "barata": es de otra
        # categoría, y hay que decirlo en vez de dejar una celda vacía.
        f["Nota"] = ("💚 costo neto negativo: la energía desplazada de la "
                     "subestación paga la inversión" if costo < 0 else "")
        filas.append(f)
    return pd.DataFrame(filas).sort_values("Ahorro neto [USD/año]",
                                           ascending=False)


def _casos_comb(dfe, sub_c, escs, combo, solo_mejor=False):
    """Promedios crudos del caso base, la combinación y las medidas solas.

    Todo restringido a los MISMOS escenarios en que corrió la combinación.
    Es la condición para que los promedios sean comparables: promediar la
    combinada sobre 4 escenarios y las individuales sobre 10 compararía dos
    poblaciones distintas, y como los escenarios difieren en severidad hasta
    6 veces, la diferencia saldría de ahí y no de las medidas.

    Devuelve una sola tabla con TODO en crudo; las vistas técnica y económica
    se arman de acá, para que no puedan discrepar entre sí.
    """
    ind = dfe[dfe["escenario"].isin(escs)]
    if ind.empty or sub_c.empty:
        return pd.DataFrame()

    def _m(s, col):
        return float(s[col].mean()) if col in s.columns else np.nan

    def fila(nombre_caso, tipo, s, base=False):
        f = {"Caso": nombre_caso, "Tipo": tipo,
             "n": int(s["escenario"].nunique())}
        # El caso base se describe con sus propias columnas *_base, que son
        # las del escenario sin ninguna medida.
        f["ens"] = _m(s, "ens_base" if base else "ens")
        f["saidi"] = _m(s, "saidi_base" if base else "saidi")
        f["saifi"] = _m(s, "saifi_base" if base else "saifi")
        f["desv_ens"] = float(s["ens_base" if base else "ens"].std(ddof=0))
        f["d_ens"] = 0.0 if base else _m(s, "d_ens")
        f["d_saidi"] = 0.0 if base else _m(s, "d_saidi")
        f["d_saifi"] = 0.0 if base else _m(s, "d_saifi")
        f["pct"] = 0.0 if base else float((s["d_ens"] / s["ens_base"]).mean())
        # Económicas: el caso base no invierte nada, pero sí tiene costo de
        # ENS — es justamente el costo de no hacer nada.
        f["capex"] = 0.0 if base else _m(s, "cst_capex_anual")
        f["opex"] = 0.0 if base else _m(s, "cst_opex_anual")
        f["costo_total"] = 0.0 if base else _m(s, "cst_costo_total")
        f["costo_ens"] = _m(s, "cst_costo_ens_base" if base
                            else "cst_costo_ens_estr")
        f["ahorro_ens"] = 0.0 if base else _m(s, "cst_ahorro_ens")
        f["ahorro_neto"] = 0.0 if base else _m(s, "cst_ahorro_neto")
        return f

    filas = [fila("Caso base (sin medidas)", "Referencia", sub_c, base=True),
             fila(_nombre_combo(combo), "Combinada", sub_c)]
    orden = ind.groupby("medida")["d_ens"].mean().sort_values(ascending=False)
    for m in (orden.index[:1] if solo_mejor else orden.index):
        filas.append(fila(_nombre(m), "Individual", ind[ind["medida"] == m]))

    g = pd.DataFrame(filas)
    # Cocientes de promedios, no promedios de cocientes: mismo criterio que
    # `_resumen_eco`, para que una medida no muestre un B/C acá y otro allá.
    for d, col in (("d_ens", "usd_kwh"), ("d_saidi", "usd_saidi"),
                   ("d_saifi", "usd_saifi")):
        g[col] = _unitario(g["costo_total"], g[d], UMBRAL_MEJORA[d])
    g["bc"] = np.where(g["costo_total"] > 0,
                       g["ahorro_ens"] / g["costo_total"], np.nan)
    return g


# Las dos vistas de `_casos_comb`: qué columna va en cada tabla y cómo se
# formatea. Separadas porque son dos preguntas distintas y una sola tabla de
# quince columnas no se lee.
VISTA_TECNICA = [
    ("ens", "ENS [kWh]", "{:,.0f}"),
    ("desv_ens", "Desv. ENS", "{:,.0f}"),
    ("saidi", "SAIDI [h/cl.]", "{:.3f}"),
    ("saifi", "SAIFI [int./cl.]", "{:.3f}"),
    ("pct", "ENS evitada [%]", "{:.1%}"),
    ("d_saidi", "SAIDI evitado", "{:.3f}"),
    ("d_saifi", "SAIFI evitado", "{:.3f}"),
]
VISTA_ECONOMICA = [
    ("capex", "CAPEX anual [USD]", "{:,.0f}"),
    ("opex", "OPEX + energía de subestación [USD]", "{:,.0f}"),
    ("costo_total", "Costo total [USD/año]", "{:,.0f}"),
    ("costo_ens", "Costo de ENS [USD/año]", "{:,.0f}"),
    ("ahorro_ens", "Ahorro de ENS [USD/año]", "{:,.0f}"),
    ("ahorro_neto", "Ahorro neto [USD/año]", "{:,.0f}"),
    ("bc", "B/C [-]", "{:.2f}"),
]
VISTA_UNITARIA = [
    ("usd_kwh", "USD/kWh evitado", "{:,.2f}"),
    ("usd_saidi", "USD por h/cliente evitada", "{:,.0f}"),
    ("usd_saifi", "USD por int./cliente evitada", "{:,.0f}"),
]


def _tabla_casos(g, vista, gradientes=()):
    """Una vista de `_casos_comb`, con sus columnas y su formato."""
    cols = [c for c, _, _ in vista if c in g.columns]
    if not cols:
        return
    tabla = g[["Caso", "Tipo"] + cols].rename(
        columns={c: t for c, t, _ in vista})
    est = tabla.style.format({t: f for c, t, f in vista if c in cols},
                             na_rep="—")
    for titulo, cmap in gradientes:
        if titulo in tabla.columns:
            est = est.background_gradient(cmap=cmap, subset=[titulo])
    st.dataframe(est, use_container_width=True, hide_index=True)


def seccion_combinadas(dfe, dce, detalle, nombre):
    """Fase 3: qué pasa cuando las medidas actúan juntas."""
    st.subheader("Medidas combinadas")

    if dce.empty:
        st.info("Falta la fase 3. Correr: "
                "`python run_barrido_medidas.py --escenarios ... --fases 3`")
        return
    if "d_ens" not in dce.columns:
        st.warning("Sin baselines no se puede medir el beneficio de la "
                   "combinación. Generar E5 para esta variante.")
        return

    n_sosp = int(dce["sospechosa"].fillna(False).sum())
    val = dce[~dce["sospechosa"].fillna(False)]
    if n_sosp:
        st.caption(f"⚠️ {n_sosp} corrida(s) combinada(s) excluida(s) por "
                   f"terminar peor que no hacer nada (incumbente cortado por "
                   f"tiempo).")
    if val.empty:
        st.warning("Todas las corridas combinadas quedaron descartadas.")
        return

    combos = sorted(val["combinacion"].unique())

    # ── comparación ENTRE combinaciones ──────────────────────────────
    if len(combos) > 1:
        st.markdown("#### Cuál combinación conviene")
        # Intersección: solo los escenarios que TODAS corrieron.
        escs_int = sorted(set.intersection(
            *[set(val[val["combinacion"] == c]["escenario"]) for c in combos]))
        comp_c = _comparar_combos(dfe, val, escs_int)
        if comp_c.empty:
            st.info("Las combinaciones no comparten escenarios, así que no se "
                    "pueden comparar entre sí.")
        else:
            st.dataframe(
                comp_c.style
                .format({"ENS evitada [%]": "{:.1%}",
                         "Suma de sus medidas [%]": "{:.0%}",
                         "SAIDI evitado": "{:.3f}",
                         "Costo total [USD/año]": "{:,.0f}",
                         "Ahorro neto [USD/año]": "{:,.0f}",
                         "B/C [-]": "{:.2f}",
                         "USD/kWh evitado": "{:,.2f}"}, na_rep="—")
                .background_gradient(cmap="RdYlGn",
                                     subset=["ENS evitada [%]",
                                             "Ahorro neto [USD/año]", "B/C [-]"])
                .background_gradient(cmap="RdYlGn_r",
                                     subset=["USD/kWh evitado"]),
                use_container_width=True, hide_index=True)
            st.caption(
                f"Sobre los escenarios que TODAS las combinaciones corrieron "
                f"({escs_int}), ordenadas por ahorro neto. **Suma de sus "
                f"medidas** es lo que evitarían por separado las medidas que "
                f"la componen: muy por encima de la columna anterior significa "
                f"que se solapan y la combinación está saturando; por debajo, "
                f"que se potencian.")

    st.divider()
    combo = (st.selectbox("Diseccionar una combinación", combos,
                          format_func=_nombre_combo, key="comb_sel")
             if len(combos) > 1 else combos[0])
    sub_c = val[val["combinacion"] == combo]
    escs = sorted(set(sub_c["escenario"]) & set(dfe["escenario"]))
    if not escs:
        st.warning("Los escenarios de esta combinación no tienen corridas "
                   "individuales con las que compararla.")
        return

    ms = [m for m in str(sub_c["medidas"].iloc[0]).split("+") if m]
    st.caption(f"**{len(ms)} medidas juntas** ({', '.join(_nombre(m) for m in ms)}) "
               f"sobre los escenarios {escs}.")

    sol = _solapamiento(dfe, dce, combo, escs)

    # ── titulares ────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("ENS evitada por la combinación",
              f"{sol['Combinación'].mean():.1%}")
    k2.metric("Mejor medida sola", f"{sol['Mejor individual'].mean():.1%}",
              help=", ".join(sorted(set(sol["mejor_medida"]))))
    factor = sol["Suma de individuales"].mean() / sol["Combinación"].mean()
    k3.metric("Solapamiento", f"{factor:.1f}×",
              help="Cuántas veces la suma de los beneficios individuales "
                   "supera al beneficio real de la combinación. 1,0× sería "
                   "independencia perfecta.")

    # ── promedios de la combinación ──────────────────────────────────
    st.markdown("#### Promedios sobre los escenarios de la combinación")
    g = _casos_comb(dfe, sub_c, escs, combo, solo_mejor=not detalle)
    if not g.empty:
        st.caption(
            f"Promedio sobre los {len(escs)} escenarios en que corrió la "
            f"combinación ({escs}), con las medidas individuales restringidas "
            f"a esos mismos escenarios — es la única forma de que los "
            f"promedios sean comparables."
            + ("" if detalle else " En **Detalle** aparecen las 12 medidas."))

        st.markdown("**Indicadores técnicos**")
        _tabla_casos(g, VISTA_TECNICA, [("ENS evitada [%]", "RdYlGn")])
        st.caption("**Desv. ENS** es la dispersión entre escenarios: mide qué "
                   "tan repetible es el resultado, no qué tan bueno es.")

        st.markdown("**Indicadores económicos**")
        _tabla_casos(g, VISTA_ECONOMICA,
                     [("Ahorro neto [USD/año]", "RdYlGn"), ("B/C [-]", "RdYlGn")])
        st.caption(
            "El **costo de ENS** del caso base es el costo de no hacer nada, y "
            "es contra ese que se mide el ahorro. **OPEX + energía de "
            "subestación** incluye el crédito o el sobrecosto por la energía "
            "que se deja de comprar —o se compra de más— frente al caso base, "
            "y por eso puede salir negativo.")

        st.markdown("**Costo por unidad de mejora**")
        _tabla_casos(g, VISTA_UNITARIA, [("USD/kWh evitado", "RdYlGn_r")])
        st.caption(
            "Cociente del costo promedio por la mejora promedio. La "
            "combinación se puede comparar directo con cada medida sola: si su "
            "USD/kWh es mucho mayor, está comprando el último tramo de ENS muy "
            "caro.")

    # ── ¿se suman los beneficios? ────────────────────────────────────
    st.markdown("#### ¿Se suman los beneficios o se solapan?")
    largo = sol.melt(id_vars="escenario",
                     value_vars=["Mejor individual", "Combinación",
                                 "Suma de individuales"],
                     var_name="Caso", value_name="ENS evitada")
    fig = px.bar(largo, x="escenario", y="ENS evitada", color="Caso",
                 barmode="group",
                 color_discrete_map={"Mejor individual": "#9ecae1",
                                     "Combinación": "#2ca02c",
                                     "Suma de individuales": "#d9d9d9"},
                 labels={"escenario": "Escenario"})
    fig.add_hline(y=1.0, line_dash="dash", line_color="#d62728",
                  annotation_text="100% de la ENS del caso base",
                  annotation_position="top left")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=420, margin=dict(t=30, b=0),
                      xaxis=dict(tickmode="linear"),
                      legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Si las medidas fueran independientes, la barra verde y la gris "
        f"coincidirían. La gris llega a {sol['Suma de individuales'].mean():.0%} "
        f"y la verde topa en 100%, que es todo lo que hay para evitar: las "
        f"medidas están evitando la MISMA energía, no energías distintas.")

    # ── cuánto cuesta el último tramo ────────────────────────────────
    if sol["costo_combo"].notna().any():
        d_ens_extra = (sol["d_ens_combo"] - sol["d_ens_mejor"]).mean()
        d_costo = (sol["costo_combo"] - sol["costo_mejor"]).mean()
        if d_ens_extra > 1:
            st.markdown("#### Lo que cuesta el último tramo")
            c1, c2, c3 = st.columns(3)
            c1.metric("ENS extra sobre la mejor medida sola",
                      f"{d_ens_extra:,.0f} kWh")
            c2.metric("Costo extra", f"{d_costo:,.0f} USD/año")
            c3.metric("Costo de ese tramo", f"{d_costo/d_ens_extra:,.0f} USD/kWh",
                      help="Comparar con el VoLL: sobre él, ese último tramo "
                           "cuesta más que la energía que salva.")
            st.caption(
                f"Pasar de **{sol['Mejor individual'].mean():.0%}** (la mejor "
                f"medida sola) a **{sol['Combinación'].mean():.0%}** cuesta "
                f"{d_costo/d_ens_extra:,.0f} USD por cada kWh adicional "
                f"evitado. La combinación completa es técnicamente perfecta y "
                f"económicamente muy pasada del punto de retornos "
                f"decrecientes.")

    # ── retención del despliegue ─────────────────────────────────────
    st.markdown("#### ¿Cuánto de cada medida se sigue usando en conjunto?")
    ret = _retencion(dfe, dce, combo, escs)
    if ret.empty:
        st.info("Ninguna medida de la combinación tiene variable de "
                "despliegue comparable.")
    else:
        st.dataframe(
            ret.style
            .format({"Sola": "{:,.1f}", "En combinación": "{:,.1f}",
                     "Retención": "{:.0%}"}, na_rep="—")
            .background_gradient(cmap="RdYlGn", subset=["Retención"],
                                 vmin=0, vmax=1.5),
            use_container_width=True, hide_index=True)
        st.caption(
            "Promedio sobre los escenarios comparables, en la propia variable "
            "de decisión del modelo. **Retención** = lo que el solver despliega "
            "teniendo todas las medidas disponibles, dividido por lo que "
            "despliega cuando esa medida está sola. Bajo 100% las otras "
            "medidas la desplazan; sobre 100% se potencian.")

    # ── qué se instaló y dónde ───────────────────────────────────────
    st.markdown("#### Qué instaló la combinación, y dónde")
    planes_c = cargar_planes(nombre, sello(nombre), "combinadas")
    desp = _despliegue_combo(planes_c, combo, escs)
    if desp.empty:
        st.info("Las corridas combinadas de este barrido no tienen cartera "
                "por elemento guardada.")
    else:
        st.dataframe(
            desp.style.format({"Total (prom.)": "{:,.1f}"}, na_rep="—"),
            use_container_width=True, hide_index=True)
        st.caption(
            "Promedio sobre los escenarios de la combinación. **Dónde** son "
            "todas las barras, líneas o enlaces que recibieron algo en algún "
            "escenario; la última columna, los que lo recibieron en TODOS — "
            "esa parte de la cartera no depende del escenario. Las medidas del "
            "combo que no aparecen son las que el solver descartó por completo.")

    if not detalle:
        st.caption("En **Detalle** están el mapa elemento × escenario, la "
                   "tabla completa de las corridas combinadas y la comparación "
                   "indicador por indicador.")
        return

    # ── mapa elemento × escenario de la combinación ──────────────────
    if not planes_c.empty:
        st.markdown("**Detalle por elemento**")
        sub_p = planes_c[(planes_c["caso"] == combo)
                         & (planes_c["escenario"].isin(escs))]
        con_algo = sorted(sub_p[sub_p["valor"].abs() > 1e-9]["medida"].unique())
        if con_algo:
            m_sel = st.selectbox("Medida", con_algo, format_func=_nombre,
                                 key="comb_medida_elem")
            s = sub_p[sub_p["medida"] == m_sel]
            piv = s.pivot_table(index="elemento", columns="escenario",
                                values="valor", fill_value=0.0)
            _f, _c, etiq, uni = _elemento_medida(m_sel)
            piv.index = [f"{etiq} {i}" for i in piv.index]
            piv.columns = [f"esc {c}" for c in piv.columns]
            st.dataframe(
                piv.style.background_gradient(cmap="Blues", axis=None)
                .format("{:,.1f}"),
                use_container_width=True)
            st.caption(f"Valor en {uni}, dentro de la combinación.")

    # ── detalle ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Comparación indicador por indicador")
    ind_c = st.selectbox("Indicador", [i for i in INDICADORES
                                       if f"d_{i}" in dce.columns] or INDICADORES,
                         key="barr_ind_c", format_func=_etiqueta)
    col = f"d_{ind_c}" if f"d_{ind_c}" in dce.columns else ind_c
    filas = []
    for e in escs:
        for m in sorted(dfe["medida"].unique()):
            r = dfe[(dfe["medida"] == m) & (dfe["escenario"] == e)]
            if not r.empty:
                filas.append({"Escenario": e, "Caso": _nombre(m),
                              "Tipo": "Individual", "valor": r[col].iloc[0]})
        c = sub_c[sub_c["escenario"] == e]
        if not c.empty:
            filas.append({"Escenario": e, "Caso": _nombre_combo(combo),
                          "Tipo": "Combinada", "valor": c[col].iloc[0]})
    comp = pd.DataFrame(filas)
    fig2 = px.bar(comp, x="Caso", y="valor", color="Tipo",
                  facet_row="Escenario" if comp["Escenario"].nunique() > 1 else None,
                  color_discrete_map={"Individual": "#9ecae1",
                                      "Combinada": "#2ca02c"},
                  labels={"valor": f"{_etiqueta(ind_c)} — mejora vs. caso base"})
    fig2.update_layout(height=300 * max(1, comp["Escenario"].nunique()),
                       xaxis_title="", margin=dict(t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Tabla completa de corridas combinadas"):
        st.dataframe(dce, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Descargar combinadas (CSV)",
        dce.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"barrido_{nombre}_combinadas.csv", mime="text/csv",
        key="comb_dl")


# ════════════════════════════════════════════════════════════
# Qué correr después
# ════════════════════════════════════════════════════════════

# Familia de cada medida. Sirve para priorizar pares de familias DISTINTAS:
# dos medidas de la misma familia (dos ESS, dos DG) compiten por la misma
# función y su combinación es predecible; el interés está en si una de red se
# potencia con una de generación, que es lo que no se sabe.
FAMILIA = {
    "dg_sol": "generación", "dg_eol": "generación", "dg_fos": "generación",
    "ess_4h": "almacenamiento", "ess_5h": "almacenamiento",
    "ess_6h": "almacenamiento",
    "meg": "móvil", "mess": "móvil",
    "reconfig_topo": "red", "inv_tie": "red",
    "inv_switch_rcs": "red", "inv_switch_ms": "red",
    "veg_mgmt": "endurecimiento", "refuerzo": "endurecimiento",
    "soterramiento": "endurecimiento",
    "edrp": "demanda", "prior_criticos": "demanda",
}


def _modelo_tiempo(dfe, dce):
    """Cuánto tarda una corrida de k medidas, estimado de lo ya corrido.

    Se interpolan los dos puntos que existen: el promedio de una corrida
    individual (k=1) y el de la combinación más grande ya resuelta. Es una
    estimación gruesa y se rotula como tal — sirve para decidir si una tanda
    cabe en una tarde, no para prometer una hora de término.
    """
    t1 = float(dfe["segundos"].mean()) if "segundos" in dfe else 120.0
    val = dce[~dce["sospechosa"].fillna(False)] if "sospechosa" in dce else dce
    if val.empty or "segundos" not in val:
        return lambda k: t1 * k
    k_obs = max(1, len(str(val["medidas"].iloc[0]).split("+")))
    t_obs = float(val["segundos"].mean())
    if k_obs <= 1:
        return lambda k: t1 * k
    pend = (t_obs - t1) / (k_obs - 1)
    return lambda k: max(t1, t1 + pend * (k - 1))


def _combos_ya_corridos(dce):
    """Conjuntos de medidas que ya tienen corrida, para no re-sugerirlos."""
    if dce.empty or "medidas" not in dce:
        return set()
    return {frozenset(str(m).split("+")) for m in dce["medidas"]}


def _sugerencias(dfe, dce, tipos, n_esc, hechos):
    """Qué corridas faltan y cuáles vale la pena hacer primero.

    El criterio central es la SATURACIÓN. Una combinación cuyas medidas, por
    separado, ya evitan más ENS de la que existe, va a llegar al 100% y no va
    a discriminar nada: es lo que pasó con la combinación de las 12 medidas,
    que da ENS = 0 en cuatro de cinco escenarios y no permite leer sinergia.
    Los pares informativos son los que suman cerca de —pero no mucho más
    que— el 100% de la ENS del caso base.
    """
    # B/C y USD/kWh como cociente de promedios, NO promedio de cocientes: es
    # el mismo criterio de `_resumen_eco`, y con el otro la misma medida
    # aparecería con un B/C distinto en esta pestaña que en la económica.
    agg = dfe.groupby("medida").agg(
        pct=("d_ens_pct", "mean"), d_ens=("d_ens", "mean"),
        ahorro=("cst_ahorro_ens", "mean"), costo=("cst_costo_total", "mean"))
    agg["bc"] = np.where(agg["costo"] > 0, agg["ahorro"] / agg["costo"], np.nan)
    agg["usd"] = _unitario(agg["costo"], agg["d_ens"], UMBRAL_MEJORA["d_ens"])
    g = agg.to_dict("index")
    medidas = [m for m in g if np.isfinite(g[m]["pct"])]
    t_de = _modelo_tiempo(dfe, dce)
    filas = []

    # ── pares ────────────────────────────────────────────────────────
    if "Pares" in tipos:
        for i, a in enumerate(medidas):
            for b in medidas[i + 1:]:
                if frozenset((a, b)) in hechos:
                    continue
                suma = g[a]["pct"] + g[b]["pct"]
                # Campana centrada en 0,9: por debajo la combinación queda
                # lejos del techo y el solapamiento apenas se nota; muy por
                # encima satura y el resultado es siempre 100%.
                informativo = float(np.exp(-((suma - 0.9) / 0.45) ** 2))
                distintas = FAMILIA.get(a) != FAMILIA.get(b)
                rentables = (g[a]["bc"] or 0) >= 1 and (g[b]["bc"] or 0) >= 1
                fams = {FAMILIA.get(a), FAMILIA.get(b)}
                complementarias = fams == {"generación", "almacenamiento"}
                score = (0.45 * informativo + 0.2 * distintas +
                         0.2 * rentables + 0.15 * complementarias)
                razon = []
                razon.append(
                    f"por separado suman {suma:.0%} de la ENS base"
                    + (" → va a saturar" if suma > 1.35 else
                       " → el solapamiento será medible" if suma > 0.5 else
                       " → poco margen para verse"))
                if complementarias:
                    razon.append("generación + almacenamiento: la "
                                 "complementariedad que insinúa la fase 3")
                elif not distintas:
                    razon.append(f"ambas son de la familia "
                                 f"'{FAMILIA.get(a)}': compiten por lo mismo")
                if rentables:
                    razon.append(f"ambas se pagan solas "
                                 f"(B/C {g[a]['bc']:.1f} y {g[b]['bc']:.1f})")
                filas.append({
                    "Tipo": "Par", "Medidas": [a, b],
                    "Qué correr": f"{_nombre(a)} + {_nombre(b)}",
                    "Por qué": " · ".join(razon), "interes": score,
                    "Corridas": n_esc, "min": t_de(2) * n_esc / 60.0,
                })

    # ── leave-one-out ────────────────────────────────────────────────
    if "Leave-one-out" in tipos and not dce.empty:
        val = dce[~dce["sospechosa"].fillna(False)]
        if not val.empty:
            todas = [m for m in str(val["medidas"].iloc[0]).split("+") if m]
            for m in todas:
                resto = [x for x in todas if x != m]
                if frozenset(resto) in hechos:
                    continue
                col = VAR_MEDIDA.get(m, (None, None))[0]
                desplegada = (col in val.columns
                              and float(val[col].mean()) > 1e-9)
                score = 0.85 if desplegada else 0.3
                razon = ("el solver la despliega cuando están todas, así que "
                         "su contribución marginal decide si entra a la "
                         "cartera final"
                         if desplegada else
                         "las otras medidas ya la desplazan por completo; el "
                         "leave-one-out solo lo confirmaría")
                filas.append({
                    "Tipo": "Leave-one-out", "Medidas": resto,
                    "Qué correr": f"Todas menos {_nombre(m)}",
                    "Por qué": razon, "interes": score,
                    "Corridas": n_esc, "min": t_de(len(resto)) * n_esc / 60.0,
                })

    # ── cartera incremental ──────────────────────────────────────────
    if "Cartera incremental" in tipos:
        orden = sorted([m for m in medidas
                        if np.isfinite(g[m].get("usd", np.nan))
                        and g[m]["pct"] > 0.01],
                       key=lambda m: g[m]["usd"])
        acum = []
        for m in orden[:6]:
            acum = acum + [m]
            if len(acum) < 2 or frozenset(acum) in hechos:
                continue
            pct_suma = sum(g[x]["pct"] for x in acum)
            filas.append({
                "Tipo": "Cartera", "Medidas": list(acum),
                "Qué correr": " + ".join(_nombre(x) for x in acum),
                "Por qué": f"las {len(acum)} más baratas por kWh, acumuladas "
                           f"(suman {pct_suma:.0%} por separado): dice hasta "
                           f"dónde conviene seguir agregando",
                "interes": 0.7 - 0.05 * len(acum),
                "Corridas": n_esc, "min": t_de(len(acum)) * n_esc / 60.0,
            })

    if not filas:
        return pd.DataFrame()
    return pd.DataFrame(filas).sort_values("interes", ascending=False)


def seccion_siguientes(dfe, dce, B, nombre, escenarios):
    """Qué corridas faltan, cuáles rinden más y con qué comando lanzarlas."""
    import barrido_lanzador as BL

    st.subheader("Qué correr después")
    st.caption(
        "Las combinaciones que faltan, ordenadas por lo que se aprendería de "
        "ellas. El criterio es evitar la saturación: una combinación cuyas "
        "medidas ya evitan por separado más ENS de la que existe llega al "
        "100% y no discrimina nada.")

    if "d_ens_pct" not in dfe.columns or "bc" not in dfe.columns:
        st.info("Hacen falta los baselines y la descomposición de costos para "
                "poder priorizar. Ver la pestaña de costo-efectividad.")
        return

    c1, c2, c3 = st.columns([2, 1.4, 1.2])
    txt_esc = c1.text_input(
        "Escenarios", ",".join(str(e) for e in escenarios), key="sug_esc",
        help="Correr las combinaciones sobre los MISMOS escenarios que las "
             "individuales es lo que permite comparar par contra partes y "
             "obtener un intervalo de confianza sobre la sinergia.")
    tipos = c2.multiselect("Tipo de estudio",
                           ["Pares", "Leave-one-out", "Cartera incremental"],
                           default=["Pares", "Leave-one-out"], key="sug_tipos")
    horas_pres = c3.number_input("Presupuesto [h]", 0.5, 200.0, 8.0, step=0.5,
                                 key="sug_horas",
                                 help="Se marcan las sugerencias que caben en "
                                      "ese tiempo, acumulando de arriba abajo.")

    escs = _parse_lista(txt_esc)
    if not escs or not tipos:
        st.warning("Elige al menos un escenario y un tipo de estudio.")
        return

    sug = _sugerencias(dfe, dce, tipos, len(escs), _combos_ya_corridos(dce))
    if sug.empty:
        st.success("No queda ninguna combinación pendiente de las que se "
                   "pueden sugerir con estos criterios.")
        return

    sug["h acum."] = sug["min"].cumsum() / 60.0
    sug["Cabe"] = np.where(sug["h acum."] <= horas_pres, "✅", "—")
    vista = sug[["Cabe", "Tipo", "Qué correr", "Por qué", "Corridas", "min",
                 "h acum."]].rename(columns={"min": "min. estimados"})
    st.dataframe(
        vista.style.format({"min. estimados": "{:,.0f}", "h acum.": "{:,.1f}"}),
        use_container_width=True, hide_index=True)
    st.caption(
        "El tiempo es una estimación gruesa, interpolada entre lo que tardó "
        "una corrida individual y lo que tardó la combinación ya resuelta. "
        "Sirve para saber si una tanda cabe en una tarde, no para prometer "
        "una hora de término.")

    # ── el comando ───────────────────────────────────────────────────
    st.markdown("#### Lanzar una tanda")
    cabe = sug[sug["Cabe"] == "✅"]
    opciones = list(sug["Qué correr"])
    elegidas = st.multiselect(
        "Combinaciones a correr", opciones,
        default=list(cabe["Qué correr"])[:3], key="sug_elegidas")
    if not elegidas:
        st.info("Elige al menos una combinación.")
        return

    combos = [list(sug.loc[sug["Qué correr"] == q, "Medidas"].iloc[0])
              for q in elegidas]
    n_total = len(combos) * len(escs)
    min_total = sum(sug.loc[sug["Qué correr"] == q, "min"].iloc[0]
                    for q in elegidas)

    sec = B["variante"][4:].replace("-", ",") if B["variante"] != "base" else ""
    # A la MISMA tanda que se está mirando: estas corridas amplían este
    # barrido, no arrancan uno nuevo.
    salida = ruta_barrido(nombre)
    cmd = BL.construir_comando(
        escenarios=txt_esc,
        # --medidas define el universo de la fase 3; con --no-todas solo se
        # corren los --combo pedidos, no además la combinación completa.
        medidas=sorted({m for c in combos for m in c}),
        seccionadores=sec, horas=int(dfe["horas"].iloc[0]),
        fases="3", sin_fallas="" if B["e4"] == "historico" else B["e4"],
        lim_dg=B.get("lim_dg", 1.0), mip_gap=B["mip_gap"],
        combos=combos, todas=False, combo_escenarios=txt_esc,
        salida=salida)

    st.markdown(f"**{n_total} corridas** · ~{min_total/60:,.1f} h estimadas → "
                f"`{salida}`")
    st.code(BL.como_texto(cmd), language="bash")

    vivo, info = BL.esta_corriendo(salida)
    if vivo:
        st.warning(f"Ya hay un barrido corriendo para esta carpeta "
                   f"(PID {info.get('pid')}). Espera o deténlo en la pestaña "
                   f"de configuración.")
        return
    if st.button("▶️ Lanzar estas combinaciones", key="sug_go", type="primary"):
        try:
            pid = BL.lanzar(cmd, salida)
            st.success(f"Lanzado (PID {pid}). El progreso se ve en la pestaña "
                       f"⚙️ Configurar y lanzar.")
        except Exception as e:                      # noqa: BLE001
            st.error(f"No se pudo lanzar: {type(e).__name__}: {e}")


# ════════════════════════════════════════════════════════════
# Render
# ════════════════════════════════════════════════════════════

def render():
    st.title("🧪 Barrido de medidas por escenario")

    barridos = listar_barridos()

    # El lanzador va primero y SIEMPRE: es lo que se usa cuando todavía no hay
    # nada calculado, y también para ampliar un barrido existente.
    if not barridos:
        seccion_lanzar()
        st.divider()
        st.info("Todavía no hay resultados que mostrar. Lanza un barrido acá "
                "arriba, o córrelo por consola con el comando que aparece.")
        return

    col_sel, col_vista, col_info, col_btn = st.columns([2, 1.3, 2.4, 1])
    nombre = col_sel.selectbox(
        "Barrido (tanda · variante de aislación)", barridos, key="barr_nombre",
        on_change=sincronizar_barrido, args=("barr_nombre",),
        format_func=lambda n: n.replace("/", "  ·  ", 1),
        help="Cada supuesto de aislación tiene su propia carpeta: los "
             "resultados de dos supuestos distintos NO son comparables. El "
             "primer tramo es la tanda (la carpeta `barrido*` de artefactos), "
             "así que también aparecen acá las tandas archivadas con otro "
             "nombre.")
    # Un solo conmutador para toda la página: el Resumen contesta las
    # preguntas de decisión (qué medida conviene, en qué orden, si el
    # escenario cambia la respuesta) y el Detalle abre todo lo demás.
    vista = col_vista.radio("Vista", ["Resumen", "Detalle"], key="barr_vista",
                            horizontal=True,
                            help="Resumen deja solo lo que se necesita para "
                                 "decidir; Detalle abre todas las magnitudes, "
                                 "las tablas completas y las descargas.")
    detalle = vista == "Detalle"
    if col_btn.button("🔄 Recargar", key="barr_recargar",
                      help="Vuelve a leer los artefactos desde disco"):
        st.cache_data.clear()
        st.rerun()

    B = cargar_barrido(nombre, sello(nombre))
    df = B["individual"]
    reps = B["reps"]

    col_info.caption(f"📁 `{B['dir']}`")
    if reps:
        col_info.caption(
            f"Criterio del representativo: **{reps.get('criterio', '?')}** · "
            f"escenarios representativos: "
            f"**{reps.get('escenarios_representativos', [])}**")

    medidas = sorted(df["medida"].unique())
    escenarios = sorted(df["escenario"].unique())
    n_horas = int(df["horas"].iloc[0]) if "horas" in df else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Medidas", len(medidas))
    c2.metric("Escenarios", len(escenarios))
    c3.metric("Corridas fase 1", len(df))
    c4.metric("Combinadas", len(B["combinadas"]))
    if len(df) < len(medidas) * len(escenarios):
        st.warning(
            f"⚠️ Faltan {len(medidas)*len(escenarios) - len(df)} corridas de la "
            f"fase 1 (sin solución, o el barrido quedó a medias). Los promedios "
            f"se calculan solo con las que sí resolvieron.")

    # Δ contra el caso base del mismo escenario. Es lo que convierte los
    # niveles del barrido en mejoras y, con los costos, en costos unitarios.
    base = cargar_base(B["variante"], tuple(int(e) for e in escenarios), n_horas)
    cap_e4 = capacidad_e4(B["e4"])
    planes = cargar_planes(nombre, sello(nombre))
    dfe = _cap_resiliencia(enriquecer(df, base, B["mip_gap"]), cap_e4)
    dce = _cap_resiliencia(enriquecer(B["combinadas"], base, B["mip_gap"]), cap_e4)
    if base.empty:
        st.warning(
            f"⚠️ No hay baselines (E5) de la variante '{B['variante']}' para "
            f"estos escenarios, así que no se puede calcular cuánto evitó cada "
            f"medida. La pestaña de costo-efectividad queda vacía.")

    st.divider()

    (sec_lanzar, sec_eco, sec_a, sec_b, sec_c, sec_sig, sec_d) = st.tabs([
        "⚙️ Configurar y lanzar", "💰 Costo-efectividad", "📋 Comparar medidas",
        "🎯 Robustez de la decisión", "🔗 Medidas combinadas",
        "🔭 Qué correr después", "🔍 Ver una corrida",
    ])

    with sec_lanzar:
        seccion_lanzar()

    with sec_eco:
        seccion_costo_efectividad(dfe, dce, reps, nombre, detalle)

    # ── A — medida x escenario ───────────────────────────────
    with sec_a:
        seccion_comparar(dfe, dce, base, detalle, nombre, medidas)

    # ── B — robustez de la decision ──────────────────────────
    with sec_b:
        seccion_robustez(dfe, reps, planes, escenarios, medidas, detalle, nombre)

    # ── C — fase 3, medidas combinadas ───────────────────────
    with sec_c:
        seccion_combinadas(dfe, dce, detalle, nombre)

    # ── Sugerencias de corridas ──────────────────────────────
    with sec_sig:
        seccion_siguientes(dfe, dce, B, nombre, escenarios)

    # ── D — series horarias de una corrida ───────────────────
    with sec_d:
        seccion_ver_corrida(nombre, B, df)


def seccion_ver_corrida(nombre, B, df, key_prefix="barr", cargar_en_resultados=True):
    """Series horarias, red y cartera de UNA corrida guardada.

    Vive fuera de `render()` para que la app v8 muestre exactamente la misma
    vista. `B` es lo que devuelve `cargar_barrido` y `df` su tabla de la
    fase 1. Además de las fases del barrido lee las evaluaciones de plan fijo
    (`plan_fijo/<combinación>/plan_esc_XXX/esc_YYY.*`, mismo formato), así que
    una inversión fija evaluada en otro escenario se mira igual que el resto.
    `cargar_en_resultados` muestra el botón que lleva la corrida al historial
    de la v7; la v8 no tiene ese historial.
    """
    st.subheader("Series horarias de una corrida guardada")
    medidas = sorted(df["medida"].unique()) if "medida" in df else []
    dir_pf = ruta_barrido(nombre) / "plan_fijo"
    planes_fijos = sorted(
        d.relative_to(dir_pf).as_posix()
        for d in dir_pf.glob("*/plan_esc_*")
        if d.is_dir() and any(d.glob("esc_*.json")))
    fases = ["Individual", "Combinada"] + (["Plan fijo"] if planes_fijos else [])
    col1, col2, col3 = st.columns(3)
    fuente = col1.radio("Fase", fases,
                        horizontal=True, key=f"{key_prefix}_fuente")
    if fuente == "Individual":
        subdir, opciones = "individual", medidas
        fmt = _nombre
    elif fuente == "Combinada":
        subdir = "combinadas"
        opciones = (sorted(B["combinadas"]["combinacion"].unique())
                    if not B["combinadas"].empty else [])
        fmt = str
    else:
        subdir, opciones = "plan_fijo", planes_fijos

        def fmt(caso):
            combo, plan = caso.rsplit("/", 1)
            return (f"{_nombre_combo(combo)} · inversión del escenario "
                    f"{int(plan.split('_')[-1])}")
    if not opciones:
        st.info("No hay corridas de esa fase todavía.")
        return

    etiqueta = col2.selectbox("Caso", opciones, format_func=fmt, key=f"{key_prefix}_caso_d")
    if etiqueta is None:
        st.info("Selecciona una corrida para ver sus series horarias.")
        return

    if fuente == "Plan fijo":
        # Escenarios de EVALUACIÓN de esa inversión fija.
        escs_serie = pd.Series([int(r.stem.split("_")[1])
                                for r in (dir_pf / etiqueta).glob("esc_*.json")],
                               dtype=float)
    else:
        escs_serie = (df.loc[df["medida"] == etiqueta, "escenario"]
                      if fuente == "Individual"
                      else B["combinadas"].loc[
                          B["combinadas"]["combinacion"] == etiqueta,
                          "escenario"])
    # Un resumen puede conservar el nombre de un caso aunque sus corridas
    # no tengan escenario válido (barrido incompleto o CSV histórico).
    # No se debe pasar el None del selectbox a int().
    escs_disp = sorted({int(e) for e in escs_serie.dropna()})
    if not escs_disp:
        st.info("Ese caso no tiene escenarios guardados para visualizar.")
        return

    esc = col3.selectbox("Escenario", escs_disp, key=f"{key_prefix}_esc_d")
    if esc is None:
        st.info("Selecciona un escenario para ver sus series horarias.")
        return
    esc = int(esc)

    S = cargar_series(nombre, subdir, etiqueta, esc, sello(nombre))
    if S is None:
        st.warning("No se encontró el .npz de esa corrida.")
        return

    gran = st.radio("Granularidad", ["Horaria", "Diaria", "Mensual"],
                    index=1, horizontal=True, key=f"{key_prefix}_gran")

    ens_h = S["P_LC"].sum(axis=1)
    ens_ag = _agregar(ens_h, gran)
    base_tot = ens_baseline(B["variante"], esc, len(ens_h))

    k1, k2, k3 = st.columns(3)
    k1.metric("ENS de la corrida", f"{ens_h.sum():,.1f}")
    if base_tot is not None:
        k2.metric("ENS del caso base", f"{base_tot:,.1f}")
        k3.metric("ENS evitada", f"{base_tot - ens_h.sum():,.1f}",
                  delta=f"{-(base_tot - ens_h.sum())/base_tot:.1%}"
                  if base_tot > 0 else None,
                  delta_color="inverse")
    else:
        k2.caption(f"Sin baseline E5 para el escenario {esc} en la "
                   f"variante '{B['variante']}'.")

    fig = go.Figure()
    fig.add_scatter(y=ens_ag, mode="lines", name="ENS (estrategia)",
                    line=dict(color="#d62728"))
    if base_tot is not None:
        zb = np.load(P.base_dir(B["variante"]) / f"esc_{esc:03d}.npz")
        fig.add_scatter(y=_agregar(zb["P_LC"][:len(ens_h)].sum(axis=1), gran),
                        mode="lines", name="ENS (caso base)",
                        line=dict(color="#7f7f7f", dash="dot"))
    fig.update_layout(height=380, yaxis_title="ENS",
                      xaxis_title={"Horaria": "Hora", "Diaria": "Día",
                                   "Mensual": "Mes"}[gran],
                      margin=dict(t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    eje_x = {"Horaria": "Hora", "Diaria": "Día", "Mensual": "Mes"}[gran]

    def _grafico(titulo, catalogo, key, altura=360):
        """Superpone las series elegidas de `catalogo` = {clave: (label, modo)}."""
        disp = [k for k in catalogo if k in S and np.abs(S[k]).sum() > 1e-9]
        if not disp:
            st.caption(f"{titulo}: todas las series están en cero en esta corrida.")
            return
        sel = st.multiselect(titulo, disp, default=disp, key=key,
                             format_func=lambda k: catalogo[k][0])
        if not sel:
            return
        f = go.Figure()
        for k in sel:
            f.add_scatter(y=_agregar(S[k], gran, modo=catalogo[k][1]),
                          mode="lines", name=catalogo[k][0])
        f.update_layout(height=altura, xaxis_title=eje_x,
                        margin=dict(t=30, b=0),
                        legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(f, use_container_width=True)

    st.markdown("#### Potencia por tecnología")
    st.caption("Solo aparecen las series con algún valor distinto de cero. "
               "En almacenamiento, positivo es descarga (inyección) y "
               "negativo es carga.")
    _grafico("Series de potencia", SERIES_POTENCIA, "barr_ser_pot")

    st.markdown("#### Operación de la red")
    st.caption("Las líneas fuera de servicio incluyen las sanas que arrastra "
               "la zona de aislación, no solo las averiadas.")
    _grafico("Series de operación", SERIES_OPERACION, "barr_ser_ope", altura=320)

    # ── energía acumulada por barra ─────────────────────────────
    if "BUSES" in S:
        filas_b = []
        for clave, lab in ENERGIA_POR_BUS.items():
            if clave in S and np.abs(S[clave]).sum() > 1e-9:
                for i, b in enumerate(S["BUSES"]):
                    filas_b.append({"Barra": int(b), "Magnitud": lab,
                                    "Energía": float(S[clave][i])})
        if filas_b:
            st.markdown("#### Energía acumulada por barra")
            fb = px.bar(pd.DataFrame(filas_b), x="Barra", y="Energía",
                        color="Magnitud", barmode="group")
            fb.update_layout(height=340, margin=dict(t=30, b=0),
                             xaxis=dict(tickmode="linear"))
            st.plotly_chart(fb, use_container_width=True)
            st.caption("Dónde terminó inyectando cada recurso y dónde quedó "
                       "la energía no suministrada.")

    ruta_json = ruta_barrido(nombre) / subdir / etiqueta / f"esc_{esc:03d}.json"
    if ruta_json.exists():
        reg = json.loads(ruta_json.read_text(encoding="utf-8"))

        # ── llevar la corrida a la pestaña Resultados ───────────
        # El .pkl.gz tiene el resultado COMPLETO en el formato nativo de
        # la app, así que basta con inyectarlo al historial: desde ahí
        # funciona toda la maquinaria de Resultados (comparaciones,
        # descomposición de costos, gráficos, exportación) sin duplicar
        # nada acá. Como el archivo es local, no hace falta subirlo.
        ruta_gz = ruta_json.with_suffix(".pkl.gz")
        if cargar_en_resultados and ruta_gz.exists():
            cg1, cg2 = st.columns([1, 3])
            if cg1.button("📥 Cargar en Resultados", key=f"cargar_{etiqueta}_{esc}"):
                try:
                    hist = st.session_state.get("historial", {})
                    nuevo = P.leer_historial(ruta_gz)
                    repetidos = [k for k in nuevo if k in hist]
                    hist.update(nuevo)
                    st.session_state["historial"] = hist
                    st.success(
                        f"Cargado en el historial: {', '.join(nuevo)}. "
                        + ("Se reemplazó una entrada con el mismo nombre. "
                           if repetidos else "")
                        + "Anda a la pestaña 📊 Resultados.")
                except Exception as e:               # noqa: BLE001
                    st.error(f"No se pudo cargar: {type(e).__name__}: {e}")
            cg2.caption(f"Resultado completo disponible "
                        f"({ruta_gz.stat().st_size/1e6:.1f} MB). Cargarlo lo deja "
                        f"disponible en Resultados junto al resto del historial.")
        elif not ruta_gz.exists():
            st.caption("Esta corrida no tiene el archivo completo (.pkl.gz). "
                       "Se genera con `--completo` al correr el barrido.")

        # ── resumen de red ──────────────────────────────────────
        # Es la MISMA figura de la pestaña Resultados: el barrido guarda el
        # dict de grafo.resumir_simulacion y acá solo se dibuja, para que
        # las dos vistas no se desincronicen.
        if reg.get("resumen_red"):
            st.markdown("#### Resumen de red")
            try:
                from grafo import build_fig_grafo_resumen, tablas_resumen, agregar_resumenes
                resumen = _claves_int(reg["resumen_red"])
                _res_grafos, _esc_grafos = cargar_resumenes(
                    nombre, subdir, etiqueta, sello(nombre))
                _titulo_grafo = f"{fmt(etiqueta)} — escenario {esc}"
                _sufijo_grafo = f"esc{esc:03d}"
                if len(_res_grafos) > 1:
                    _vista_rangos = st.radio(
                        "Vista del resumen de red",
                        ["Rangos entre escenarios", "Escenario seleccionado"],
                        horizontal=True, key=f"rangos_red_{subdir}_{etiqueta}")
                    if _vista_rangos == "Rangos entre escenarios":
                        resumen = agregar_resumenes(_res_grafos, _esc_grafos)
                        _titulo_grafo = (f"{fmt(etiqueta)} — rangos entre "
                                         f"{len(_esc_grafos)} escenarios")
                        _sufijo_grafo = "rangos"
                        st.caption(
                            "Cada cuadrito muestra al pasar el cursor el mínimo y máximo "
                            "de capacidad entre escenarios donde se instaló esa medida, "
                            "y en cuántos apareció. Escenarios: "
                            + ", ".join(map(str, _esc_grafos))
                            + ". El conjunto no representa una instalación única.")
                else:
                    st.caption("Solo hay un escenario con resumen guardado; no se puede formar un rango.")
                st.plotly_chart(
                    build_fig_grafo_resumen(
                        resumen,
                        cfg={"buses_criticos": []},
                        titulo=_titulo_grafo,
                        subtitulo=f"variante {B['variante']}"),
                    use_container_width=True)
                # tablas_resumen devuelve (df_lineas, df_buses) — el mismo
                # contenido del gráfico, que es lo que se exporta a CSV.
                df_l, df_b = tablas_resumen(resumen)
                for nom_tabla, tabla in (("Por línea", df_l), ("Por barra", df_b)):
                    if tabla is not None and len(tabla):
                        with st.expander(f"Detalle {nom_tabla.lower()}"):
                            st.dataframe(tabla, use_container_width=True,
                                         hide_index=True)
                            st.download_button(
                                f"⬇️ {nom_tabla} (CSV)",
                                tabla.to_csv(index=False).encode("utf-8-sig"),
                                file_name=f"barrido_{etiqueta}_{_sufijo_grafo}_"
                                          f"{nom_tabla.split()[-1].lower()}.csv",
                                mime="text/csv",
                                key=f"dl_{nom_tabla}_{etiqueta}_{esc}")
            except Exception as e:                       # noqa: BLE001
                st.warning(f"No se pudo dibujar el resumen de red: "
                           f"{type(e).__name__}: {e}")

        # ── mapa agregado de la medida ──────────────────────────
        # El resumen de arriba es UN escenario, y el Monte Carlo tira
        # fallas distintas cada vez: qué barra recibe GD cambia de una
        # corrida a otra. Fundir todos los escenarios de la medida
        # separa la decisión robusta (bus elegido en 10 de 10) del
        # ruido del escenario (bus elegido en 1 de 10), que es
        # justamente lo que hay que defender en la tesis.
        _agg_res, _agg_escs = cargar_resumenes(nombre, subdir, etiqueta,
                                               sello(nombre))
        if len(_agg_res) > 1:
            with st.expander(
                    f"🗺️ Mapa agregado de {fmt(etiqueta)} — "
                    f"{len(_agg_res)} escenarios fundidos"):
                try:
                    from grafo import (agregar_resumenes,
                                       build_fig_grafo_resumen,
                                       tablas_resumen)
                    agg = agregar_resumenes(_agg_res, _agg_escs)
                    t_agg = agg["totales"]
                    st.caption(
                        f"Escenarios: {', '.join(str(e) for e in _agg_escs)}. "
                        f"Grosor de línea = frecuencia de falla; el hover "
                        f"dice en cuántos escenarios falló cada línea y en "
                        f"cuántos recibió recurso cada barra. **Las "
                        f"capacidades son el promedio sobre los escenarios**, "
                        f"no una instalación concreta: no es un estado "
                        f"realizable de la red, es dónde converge la medida."
                    )
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Escenarios", f"{t_agg['n_esc']}")
                    m2.metric("Barras con GD", f"{t_agg['buses_con_dg']}",
                              help="En al menos un escenario.")
                    m3.metric("Barras con ESS", f"{t_agg['buses_con_ess']}",
                              help="En al menos un escenario.")
                    m4.metric("ENS media por escenario",
                              f"{t_agg['ens_media_kwh']:,.0f} kWh")
                    st.plotly_chart(
                        build_fig_grafo_resumen(
                            agg, cfg={"buses_criticos": []},
                            titulo=f"{fmt(etiqueta)} — mapa agregado "
                                   f"({t_agg['n_esc']} escenarios)",
                            subtitulo=f"variante {B['variante']} · "
                                      f"valor esperado, no un escenario "
                                      f"realizable"),
                        use_container_width=True,
                        key=f"agg_grafo_{subdir}_{etiqueta}")
                    dfa_l, dfa_b = tablas_resumen(agg)
                    for nom_t, tabla, cols in (
                        ("líneas", dfa_l, ["Línea", "Nombre",
                                           "Escenarios con falla",
                                           "% escenarios con falla",
                                           "Horas de falla por escenario",
                                           "Escenarios con medida fija",
                                           "Escenarios con seccionador nuevo",
                                           "Escenarios con enlace nuevo"]),
                        ("barras", dfa_b, ["Bus", "Zona", "Escenarios con GD",
                                           "Escenarios con ESS",
                                           "Escenarios con unidad móvil",
                                           "GD solar [kW]", "GD eólica [kW]",
                                           "GD fósil [kW]", "ESS 4h [kWh]",
                                           "ESS 5h [kWh]", "ESS 6h [kWh]",
                                           "ENS por escenario [kWh]"]),
                    ):
                        if tabla is None or not len(tabla):
                            continue
                        st.markdown(f"**Frecuencia por {nom_t}**")
                        st.dataframe(
                            tabla[[c for c in cols if c in tabla.columns]],
                            use_container_width=True, hide_index=True)
                        st.download_button(
                            f"⬇️ Agregado por {nom_t} (CSV)",
                            tabla.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"agregado_{etiqueta}_{nom_t}_"
                                      f"{t_agg['n_esc']}esc.csv",
                            mime="text/csv",
                            key=f"dl_agg_{nom_t}_{subdir}_{etiqueta}")
                except Exception as e:               # noqa: BLE001
                    st.warning(f"No se pudo agregar los escenarios: "
                               f"{type(e).__name__}: {e}")
        elif _agg_res:
            st.caption("Esta medida tiene una sola corrida con resumen de "
                       "red guardado: no hay nada que agregar todavía.")

        _tabla_ubicacion(reg.get("ubicacion") or {})

        no_nulos = [(a, reg[a]) for a in AGREGADOS
                    if a in reg and abs(float(reg[a])) > 1e-9]
        if no_nulos:
            st.markdown("#### Resultados físicos de la corrida")
            for j in range(0, len(no_nulos), 4):
                for c, (a, v) in zip(st.columns(4), no_nulos[j:j+4]):
                    entero = a.startswith(("n_", "horas_", "maniobras_"))
                    c.metric(ETIQ_AGG.get(a, a),
                             f"{v:,.0f}" if entero else f"{v:,.1f}")

        with st.expander("Cartera de inversión elegida por el solver"):
            if reg.get("plan"):
                for clave, d in reg["plan"].items():
                    st.markdown(f"**{clave}**")
                    st.dataframe(
                        pd.DataFrame({"elemento": list(d), "valor": list(d.values())}),
                        use_container_width=True, hide_index=True)
            else:
                st.caption("El solver no instaló nada: ninguna medida "
                           "resultó rentable en esta corrida.")
        with st.expander("Registro completo de la corrida (JSON)"):
            st.json(reg)
