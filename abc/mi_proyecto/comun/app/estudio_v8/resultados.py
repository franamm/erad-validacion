"""Pestaña 📊 Resultados: lee lo calculado y lo muestra con las vistas existentes.

Las figuras se muestran tal como las construye su función original (tamaño de
tesis, fondo transparente, colores por medida y por zona) y se descargan con
`formato_figuras._descargas_figura`. Las tablas económicas usan los bloques y
gradientes de `barrido_vista`.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import barrido_vista as BV
import graficos_valor_esperado as GVE
import pipeline as P
from formato_figuras import _descargas_figura
from estudio_v8 import analisis as A
from estudio_v8.core import ART_DIR, COMBINACION_TESIS, leer_json

COLOR_BASE, COLOR_CASO = "#9CA3AF", "#2a78d6"
MESES_H = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
PESTANAS = ["🕸️ Red y plan instalado", "📋 Indicadores", "⚡ ENS por zona y hora",
            "📉 Curvas MAC", "🔗 Solapamiento", "⚖️ Reglas de decisión", "🔍 Ver una corrida"]
ETIQ = {"cst_objetivo_plan": "Costo del plan [USD/año] (sin constante del E4)",
        "objetivo": "Función objetivo [USD/año]", "ens": "ENS [kWh]", "saidi": "SAIDI [h/cliente]",
        "saifi": "SAIFI [int./cliente]", "cst_costo_total": "Costo total de la medida [USD/año]"}


def navegacion(etiquetas, clave):
    """Pestañas nativas que calculan solo el contenido elegido."""
    import inspect
    if "on_change" in inspect.signature(st.tabs).parameters:
        tabs = st.tabs(etiquetas, key=clave, on_change="rerun")
        return next((label, tab) for label, tab in zip(etiquetas, tabs) if tab.open)
    seleccion = st.radio("Vista", etiquetas, horizontal=True, key=clave, label_visibility="collapsed")
    return seleccion, st.container()


def plantilla(fig, alto=900):
    """Formato de las figuras nuevas: el de graficos_valor_esperado.figuras_ens."""
    fig.update_layout(width=1600, height=alto, font=dict(size=30),
                      legend=dict(orientation="h", y=-.25, font=dict(size=25)),
                      margin=dict(l=90, r=40, t=110, b=80),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title_font=dict(size=30), tickfont=dict(size=30))
    fig.update_yaxes(title_font=dict(size=30), tickfont=dict(size=30))
    return fig


def figura(fig, nombre, key, *, scroll_zoom=False):
    config = {"scrollZoom": True} if scroll_zoom else None
    st.plotly_chart(fig, use_container_width=True, key=f"fig_{key}", config=config)
    _descargas_figura(fig, nombre, key=key)


def descarga_csv(df, nombre, key, index=False):
    st.download_button("⬇️ Descargar CSV", df.to_csv(index=index).encode("utf-8-sig"),
                       file_name=f"{nombre}.csv", mime="text/csv", key=f"csv_{key}")


def nombre_caso(tipo, caso):
    if tipo in ("combinadas", "Combinada"):
        return A.NOMBRE_COMBO_TESIS.get(caso) or BV._nombre_combo(caso)
    return BV._nombre(caso)


def _slug(caso):
    return str(caso).removeprefix("combo_").replace("+", "-")[:60]


def _validas(df):
    if df.empty or "sospechosa" not in df:
        return df
    return df[~df["sospechosa"].fillna(False)]


# ════════════════════════════════════════════════════════════
# Datos del estudio
# ════════════════════════════════════════════════════════════

def cargar_estudio(nombre):
    root = ART_DIR / nombre
    if (root / "resumen_individual.csv").exists():
        return root, BV.cargar_barrido(nombre, BV.sello(nombre))
    from barrido_csv import leer_resumen
    from estudio_v8.optimizacion import supuestos_estudio
    sup, _ = supuestos_estudio(root)
    comunes = {} if not sup else sup
    B = {"dir": str(root), "individual": pd.DataFrame(columns=["medida", "escenario"]),
         "promedios": pd.DataFrame(), "reps": {},
         "combinadas": (leer_resumen(root / "resumen_combinadas.csv")
                        if (root / "resumen_combinadas.csv").exists() else pd.DataFrame()),
         "variante": ("sec-" + "-".join(map(str, comunes["seccionadores"]))
                      if comunes.get("seccionadores") else "base"),
         "mip_gap": comunes.get("mip_gap", 0.02), "e4": comunes.get("e4", ""), "lim_dg": comunes.get("lim_dg")}
    # Igual que BV.cargar_barrido: la variante sale del registro de una corrida.
    # Reconstruirla desde los seccionadores falla con variantes con hash
    # (p. ej. `sec-80x-d184ce8a61` de El Melón) y deja el estudio sin caso base.
    import json
    for j in sorted(root.glob("combinadas/*/esc_*.json"))[:1]:
        reg = json.loads(j.read_text(encoding="utf-8"))
        B["variante"] = reg.get("variante_aislacion") or B["variante"]
        B["mip_gap"] = float(reg.get("mip_gap") or B["mip_gap"])
        B["e4"] = reg.get("e4_sin_fallas") or B["e4"]
        B["lim_dg"] = reg.get("lim_dg", B["lim_dg"])
    return root, B


def preparar(nombre):
    root, B = cargar_estudio(nombre)
    ind, comb = B["individual"], B["combinadas"]
    escs = sorted({int(e) for d in (ind, comb) if not d.empty and "escenario" in d
                   for e in d["escenario"].dropna()})
    horas = next((int(d["horas"].dropna().iloc[0]) for d in (ind, comb)
                  if not d.empty and "horas" in d and d["horas"].notna().any()), 8760)
    base = BV.cargar_base(B["variante"], tuple(escs), horas) if escs else pd.DataFrame()
    cap = BV.capacidad_e4(B["e4"])
    dfe = BV._cap_resiliencia(BV.enriquecer(ind, base, B["mip_gap"]), cap) if not ind.empty else ind
    dce = BV._cap_resiliencia(BV.enriquecer(comb, base, B["mip_gap"]), cap) if not comb.empty else comb
    return {"nombre": nombre, "root": root, "B": B, "dfe": dfe, "dce": dce, "base": base,
            "horas": horas, "escenarios": escs}


def filas_caso(d, tipo, caso):
    df = d["dce"] if tipo == "combinadas" else d["dfe"]
    col = "combinacion" if tipo == "combinadas" else "medida"
    return df[df[col] == caso] if not df.empty and col in df else df.iloc[:0]


@st.cache_data(show_spinner=False)
def _nombres_linea(mtime):
    return P._cargar("red.pkl")["net"].line["name"].to_dict()


def nombres_linea():
    ruta = ART_DIR / "red.pkl"
    return _nombres_linea(ruta.stat().st_mtime) if ruta.exists() else {}


def render():
    from pathlib import Path
    from estudio_v8.resultados_archivos import render as render_archivos
    from estudio_v8.optimizacion import estudios_disponibles
    st.header("📊 Resultados")
    elegido = Path(st.session_state.get("v8_caso", ART_DIR)).resolve()
    if elegido != ART_DIR.resolve():
        render_archivos(elegido)
        return
    estudios = estudios_disponibles()
    if not estudios:
        render_archivos(elegido)
        return
    st.sidebar.divider()
    st.sidebar.markdown("#### 📊 Qué mostrar")
    defecto = "barrido/sec-0-2-9-10 severo"
    if defecto not in estudios:
        # Otro caso (p. ej. El Melón): el estudio más reciente con planes fijos,
        # que es el que tiene el análisis completo; si no hay, el más reciente.
        con_pf = [n for n in estudios if (ART_DIR / n / "plan_fijo").is_dir()]
        defecto = max(con_pf or estudios, key=lambda n: (ART_DIR / n).stat().st_mtime)
    nombre = st.sidebar.selectbox("Estudio", estudios, key="res_estudio",
                                  index=estudios.index(defecto),
                                  format_func=lambda n: n.replace("/", "  ·  ", 1))
    if st.sidebar.button("🔄 Recargar", key="res_recargar", help="Vuelve a leer el disco"):
        st.cache_data.clear()
    d = preparar(nombre)
    casos = ([("combinadas", c) for c in sorted(d["dce"]["combinacion"].unique())]
             if not d["dce"].empty else [])
    casos += [("individual", m) for m in sorted(d["dfe"]["medida"].unique())] if not d["dfe"].empty else []
    if not casos:
        st.info("El estudio no tiene corridas legibles.")
        return
    preferido = ("combinadas", COMBINACION_TESIS)
    if preferido not in casos and ("combinadas", "todas") in casos:
        preferido = ("combinadas", "todas")   # nombre de la combinación completa en los demás casos
    tipo, caso = st.sidebar.selectbox("Caso (combinación o medida)", casos, key="res_caso",
                              index=casos.index(preferido) if preferido in casos else 0,
                              format_func=lambda k: ("🔗 " if k[0] == "combinadas" else "▪️ ") + nombre_caso(*k))
    sel = filas_caso(d, tipo, caso)
    val = _validas(sel)
    B = d["B"]
    st.caption(f"{len(val)}/{len(sel)} corridas válidas"
               + (f" · {len(sel) - len(val)} sospechosas excluidas" if len(sel) > len(val) else "")
               + f" · escenarios {sorted(int(e) for e in val['escenario'])} · variante {B['variante']}"
               + f" · E4 {B['e4'] or 'histórico'} · {d['horas']} h")
    if d["base"].empty:
        st.warning(f"Faltan los casos base E5 de la variante `{B['variante']}`: sin ellos no hay "
                   "mejoras, costos unitarios ni ENS comparada.")

    seccion, espacio = navegacion(PESTANAS, "resultados_seccion")
    with espacio:
        if seccion == PESTANAS[0]:
            tab_red(d, tipo, caso, val)
        elif seccion == PESTANAS[1]:
            tab_indicadores(d, tipo, caso)
        elif seccion == PESTANAS[2]:
            tab_ens(d, tipo, caso, val)
        elif seccion == PESTANAS[3]:
            tab_mac(d)
        elif seccion == PESTANAS[4]:
            tab_solapamiento(d, tipo, caso)
        elif seccion == PESTANAS[5]:
            tab_decision(d, tipo, caso)
        else:
            BV.seccion_ver_corrida(nombre, B, B["individual"], key_prefix="v8", cargar_en_resultados=False)


# ════════════════════════════════════════════════════════════
# Red y plan instalado
# ════════════════════════════════════════════════════════════

def tab_red(d, tipo, caso, val):
    from grafo import agregar_resumenes, build_fig_grafo_resumen, tablas_resumen
    escs = sorted(int(e) for e in val["escenario"])
    if not escs:
        st.info("No hay corridas válidas de este caso.")
        return
    a, b, c = st.columns([1.3, 1.3, 1])
    alcance = a.radio("Alcance", ["Todos los escenarios", "Un escenario"], horizontal=True, key="res_red_alcance")
    tiempo = b.radio("Tiempo", ["Resumen del horizonte", "Por hora"], horizontal=True, key="res_red_tiempo")
    if tiempo == "Por hora" and alcance != "Un escenario":
        st.caption("La vista por hora muestra el estado físico de un escenario concreto.")
        alcance = "Un escenario"
    esc = c.selectbox("Escenario", escs, key="res_red_esc") if alcance == "Un escenario" else None
    titulo_caso = nombre_caso(tipo, caso)

    if tiempo == "Resumen del horizonte":
        resumenes, ids = BV.cargar_resumenes(d["nombre"], tipo, caso, BV.sello(d["nombre"]))
        pares = [(r, i) for r, i in zip(resumenes, ids) if i in escs and (esc is None or i == esc)]
        if not pares:
            st.info("Estas corridas no guardaron el resumen de red.")
        else:
            if len(pares) > 1:
                resumen = agregar_resumenes([r for r, _ in pares], [i for _, i in pares])
                titulo, sufijo = f"{titulo_caso} — rangos entre {len(pares)} escenarios", "rangos"
                st.caption("Al pasar el cursor: capacidad mínima y máxima entre los escenarios donde se "
                           "instaló y en cuántos apareció. Es un resumen estadístico de planes distintos, "
                           "no una instalación única realizable.")
            else:
                resumen = pares[0][0]
                titulo, sufijo = f"{titulo_caso} — escenario {pares[0][1]}", f"esc{pares[0][1]:03d}"
            fig = build_fig_grafo_resumen(resumen, cfg={"buses_criticos": []}, titulo=titulo,
                                          subtitulo=f"variante {d['B']['variante']}")
            figura(fig, f"red_{_slug(caso)}_{sufijo}", "res_grafo_resumen", scroll_zoom=True)
            df_l, df_b = tablas_resumen(resumen)
            for nom, t in (("por línea", df_l), ("por barra", df_b)):
                if t is not None and len(t):
                    with st.expander(f"Detalle {nom}"):
                        st.dataframe(t, use_container_width=True, hide_index=True)
                        descarga_csv(t, f"red_{_slug(caso)}_{sufijo}_{nom.split()[-1]}", f"red_{nom}")
    else:
        _grafo_por_hora(d, tipo, caso, esc)

    st.markdown("#### Rangos del plan instalado")
    carpeta = d["root"] / tipo / caso
    regs = {e: leer_json(carpeta / f"esc_{e:03d}.json") for e in escs}
    npz = {e: carpeta / f"esc_{e:03d}.npz" for e in escs if (carpeta / f"esc_{e:03d}.npz").exists()}
    rangos = A.rangos_plan(regs, escs, nombres_linea(), A.edrp_por_bus(npz))
    if rangos.empty:
        st.info("El plan no instala ni usa ninguna medida en estos escenarios.")
        return
    st.dataframe(rangos.style.format({"Min": "{:,.1f}", "Max": "{:,.1f}", "Media": "{:,.1f}"}),
                 use_container_width=True, hide_index=True)
    st.caption(f"Sobre los {len(escs)} planes optimizados por escenario; donde no se instala cuenta como 0. "
               "'Escenarios' = en cuántos se usa. Frecuencia de la muestra, no probabilidad poblacional.")
    descarga_csv(rangos, f"rangos_{_slug(caso)}", "rangos")


def _grafo_por_hora(d, tipo, caso, esc):
    candidatos = [d["root"] / tipo / caso / f"esc_{esc:03d}.pkl.gz",
                  d["root"] / "detalle_horario" / tipo / caso / f"esc_{esc:03d}.pkl.gz"]
    ruta = next((r for r in candidatos if r.exists()), None)
    if ruta is None:
        st.info("Esta corrida no guardó el detalle horario (.pkl.gz). Se puede calcular solo este "
                "escenario, en `detalle_horario/`, sin tocar los resultados del estudio.")
        from estudio_v8.optimizacion import panel_detalle_horario
        panel_detalle_horario(d["root"], tipo, caso, esc, key=f"res_det_{esc}")
        return
    if not st.checkbox(f"Cargar el detalle horario ({ruta.stat().st_size / 1e6:.1f} MB)", key="res_cargar_hora"):
        return
    from grafo import build_fig_grafo
    from helpers_operativos import _results_to_list
    entrada = next(iter(P.leer_historial(ruta).values()))
    listas = [x for x in _results_to_list(entrada["results"]) if x]
    horas = listas[0]
    h = st.slider("Hora", 1, len(horas), 1, key="res_hora")
    from data_loader import LINES, faultlinMC
    fallas = {l: bool(faultlinMC[h - 1, esc, i]) for i, l in enumerate(LINES)}
    figura(build_fig_grafo(h, esc + 1, horas[h - 1], fallas, entrada["config"]),
           f"red_{_slug(caso)}_esc{esc:03d}_h{h}", "res_grafo_hora", scroll_zoom=True)


# ════════════════════════════════════════════════════════════
# Indicadores técnicos y económicos
# ════════════════════════════════════════════════════════════

def _estrategia(tipo, caso):
    return BV._nombre_combo(caso) if tipo == "combinadas" else BV._nombre_fig(caso)


def filas_comparables(d, claves):
    """Corridas válidas de los casos elegidos, sobre sus escenarios comunes."""
    piezas = []
    for tipo, caso in claves:
        f = _validas(filas_caso(d, tipo, caso)).copy()
        if f.empty:
            continue
        f["_tipo"] = "Combinada" if tipo == "combinadas" else "Individual"
        f["_caso"] = caso
        f["Estrategia"] = _estrategia(tipo, caso)
        piezas.append(f)
    if not piezas:
        return pd.DataFrame(), []
    comunes = sorted(set.intersection(*[set(p["escenario"]) for p in piezas]))
    return pd.concat([p[p["escenario"].isin(comunes)] for p in piezas], ignore_index=True), comunes


def tab_indicadores(d, tipo, caso):
    casos = ([("combinadas", c) for c in sorted(d["dce"]["combinacion"].unique())] if not d["dce"].empty else []) \
        + ([("individual", m) for m in sorted(d["dfe"]["medida"].unique())] if not d["dfe"].empty else [])
    defecto = [(tipo, caso)] + ([("individual", m) for m in A.miembros_de(caso) if ("individual", m) in casos]
                                if tipo == "combinadas" else [])
    elegidos = st.multiselect("Estrategias a comparar", casos, default=defecto, key="res_ind_casos",
                              format_func=lambda k: ("🔗 " if k[0] == "combinadas" else "▪️ ") + nombre_caso(*k))
    filas, comunes = filas_comparables(d, elegidos)
    if filas.empty or not comunes:
        st.info("Elige estrategias con escenarios comunes.")
        return
    if "d_ens" not in filas:
        st.warning("Sin casos base no se pueden calcular mejoras ni costos unitarios.")
        return
    st.caption(f"Promedio equiprobable sobre los {len(comunes)} escenarios comunes: {comunes}. "
               "Cada estrategia se compara con el caso base del mismo escenario.")

    st.markdown("#### Indicadores técnicos")
    cols = [c for c in ("ens", "saidi", "saifi", "asai") if c in filas]
    niveles = filas.groupby("Estrategia", sort=False)[cols].mean()
    base = d["base"][d["base"]["escenario"].isin(comunes)]
    if not base.empty:
        niveles = pd.concat([base[[c for c in cols if c in base]].mean().to_frame("Caso base").T, niveles])
    niveles = niveles.rename(columns={"ens": "ENS [kWh]", "saidi": "SAIDI [h/cliente]",
                                      "saifi": "SAIFI [int./cliente]", "asai": "ASAI [-]"})
    st.dataframe(niveles.style.format({"ENS [kWh]": "{:,.0f}", "SAIDI [h/cliente]": "{:.3f}",
                                       "SAIFI [int./cliente]": "{:.3f}", "ASAI [-]": "{:.6f}"}, na_rep="—"),
                 use_container_width=True)

    if "cst_costo_total" not in filas:
        st.info("Estas corridas son anteriores a la descomposición de costos (columnas cst_*).")
        return
    filas = filas[filas["cst_costo_total"].notna()]
    g = BV._resumen_eco(filas, "Estrategia")
    for titulo, bloque in BV.BLOQUES_ECO.items():
        presentes = [c for c in bloque if c[0] in g.columns]
        if not presentes:
            continue
        st.markdown(f"#### {titulo}")
        tabla = g[["Estrategia"] + [c for c, _, _ in presentes]].rename(columns={c: t for c, t, _ in presentes})
        st.dataframe(BV._estilo(tabla, presentes), use_container_width=True, hide_index=True)
        if titulo.startswith("Cuánto cuesta y"):
            BV._cap_dg_detalle(filas)
    st.caption("OPEX incluye el término de energía comprada a la subestación frente al caso base, por eso "
               "puede ser negativo. Los costos unitarios son E[costo] / E[mejora], no el promedio de cocientes, "
               "y quedan en blanco si la mejora no supera el umbral de ruido o el costo no es positivo.")
    completa = g.rename(columns={c: t for c, t, _ in BV.COLS_ECO})
    descarga_csv(completa, f"indicadores_{_slug(caso)}", "indicadores")
    with st.expander("Detalle por escenario"):
        cols_e = [c for c in ("Estrategia", "escenario", "ens", "d_ens", "d_ens_pct", "saidi", "d_saidi", "saifi",
                              "d_saifi", "cst_capex_anual", "cst_opex_anual", "cst_costo_total", "cst_ahorro_neto",
                              "usd_kwh", "solver_gap", "segundos") if c in filas]
        por_esc = filas[cols_e].sort_values(["Estrategia", "escenario"])
        st.dataframe(por_esc.style.format({"ens": "{:,.0f}", "d_ens": "{:,.0f}", "d_ens_pct": "{:.1%}",
                                           "saidi": "{:.3f}", "d_saidi": "{:.3f}", "saifi": "{:.3f}",
                                           "d_saifi": "{:.3f}", "cst_capex_anual": "{:,.0f}",
                                           "cst_opex_anual": "{:,.0f}", "cst_costo_total": "{:,.0f}",
                                           "cst_ahorro_neto": "{:,.0f}", "usd_kwh": "{:,.2f}",
                                           "solver_gap": "{:.2%}", "segundos": "{:,.0f}"}, na_rep="—"),
                     use_container_width=True, hide_index=True)
        descarga_csv(por_esc, f"indicadores_por_escenario_{_slug(caso)}", "indicadores_esc")


# ════════════════════════════════════════════════════════════
# ENS por zona y por hora
# ════════════════════════════════════════════════════════════

def _agregar_horario(horario, gran):
    if gran == "Horaria":
        return horario
    h = horario["Hora"].to_numpy() - 1
    periodo = h // 24 if gran == "Diaria" else np.searchsorted(np.cumsum(MESES_H), h, side="right")
    out = horario.drop(columns="Hora").groupby(periodo).sum().reset_index(drop=True)
    out.insert(0, "Hora", np.arange(1, len(out) + 1))
    return out


def tab_ens(d, tipo, caso, val):
    from estudio_v8.optimizacion import red
    escs = sorted(int(e) for e in val["escenario"])
    r = red()
    if not escs or r is None:
        st.info("Faltan corridas válidas o la red (E0).")
        return
    horas = int(val["horas"].iloc[0]) if "horas" in val else d["horas"]
    bases = [P.base_dir(d["B"]["variante"]) / f"esc_{e:03d}.npz" for e in escs]
    rutas = [d["root"] / tipo / caso / f"esc_{e:03d}.npz" for e in escs]
    faltan = [p.name for p in bases + rutas if not p.exists()]
    if faltan:
        st.warning(f"Faltan series o casos base ({len(faltan)} archivos); no se promedia una muestra parcial.")
        return
    try:
        horario, zonal = GVE.promediar_ens(rutas, bases, r["BUSES"], r["bus_zone"], horas)
    except (OSError, KeyError, ValueError) as exc:
        st.warning(f"No se pudo construir la ENS: {exc}")
        return
    nombre = nombre_caso(tipo, caso)
    st.caption(f"Promedio equiprobable de los mismos {len(escs)} escenarios para el caso base y la estrategia. "
               "ENS en kWh.")
    gran = st.radio("Agregación temporal", ["Horaria", "Diaria", "Mensual"], horizontal=True, key="res_ens_gran")
    zona, hora = GVE.figuras_ens(_agregar_horario(horario, gran), zonal, nombre)
    hora.update_xaxes(title={"Horaria": "Hora", "Diaria": "Día", "Mensual": "Mes"}[gran])
    t1, t2, t3 = st.tabs(["ENS por zona", "ENS en el tiempo", "ENS por escenario"])
    with t1:
        figura(zona, f"ens_zona_{_slug(caso)}", "res_ens_zona")
        descarga_csv(zonal, f"ens_zona_{_slug(caso)}", "ens_zona")
    with t2:
        figura(hora, f"ens_{gran.lower()}_{_slug(caso)}", "res_ens_hora")
        descarga_csv(_agregar_horario(horario, gran), f"ens_{gran.lower()}_{_slug(caso)}", "ens_hora")
    with t3:
        base = d["base"].set_index("escenario")["ens"] if not d["base"].empty else pd.Series(dtype=float)
        par = pd.DataFrame({"Caso base": base.reindex(escs).to_numpy(),
                            "Con medida": val.set_index("escenario")["ens"].reindex(escs).to_numpy()},
                           index=escs).sort_values("Caso base")
        par["Reducción %"] = 100 * (1 - par["Con medida"] / par["Caso base"])
        fig = go.Figure()
        x = [f"Esc. {e}" for e in par.index]
        fig.add_bar(name="Caso base", x=x, y=par["Caso base"], marker_color=COLOR_BASE)
        fig.add_bar(name=nombre, x=x, y=par["Con medida"], marker_color=COLOR_CASO,
                    text=[f"{-v:.0f} %" for v in par["Reducción %"]], textposition="outside",
                    textfont=dict(size=22))
        fig.update_layout(barmode="group", xaxis_title="Escenario (ordenado por ENS del caso base)",
                          yaxis_title="ENS [kWh]")
        figura(plantilla(fig, alto=800), f"ens_pareada_{_slug(caso)}", "res_ens_pareada")
        descarga_csv(par, f"ens_pareada_{_slug(caso)}", "ens_pareada", index=True)


# ════════════════════════════════════════════════════════════
# Curvas MAC
# ════════════════════════════════════════════════════════════

def tab_mac(d):
    """Las curvas de la pestaña Resultados de la v7 (sección E), con los valores esperados del estudio."""
    from curvas_mac import FORMATO_UNIDAD, curvas_mac_estrategias, tabla_costos_unidad
    from formato_figuras import _fig_pantalla
    if d["dfe"].empty:
        _falta_individuales(d, "las curvas MAC")
        return
    individuales = [("individual", m) for m in sorted(d["dfe"]["medida"].unique())]
    combos = [("combinadas", c) for c in sorted(d["dce"]["combinacion"].unique())] if not d["dce"].empty else []
    elegidos = st.multiselect("Estrategias en las curvas", individuales + combos, default=individuales,
                              key="res_mac_casos",
                              format_func=lambda k: ("🔗 " if k[0] == "combinadas" else "▪️ ") + nombre_caso(*k))
    filas, comunes = filas_comparables(d, elegidos)
    columnas = {"ens", "saidi", "saifi", "ens_base", "saidi_base", "saifi_base", "cst_costo_total"}
    if filas.empty or not comunes:
        st.info("Elige estrategias con escenarios comunes.")
        return
    if not columnas.issubset(filas.columns) or filas[list(columnas)].isna().any().any():
        st.warning("Faltan indicadores, casos base o costos: no se calculan promedios parciales.")
        return
    prom = GVE.resumir_indicadores(filas)
    df_mac = pd.DataFrame({
        "Estrategia": [_estrategia("combinadas" if t == "Combinada" else "individual", c)
                       for t, c in zip(prom["_tipo"], prom["_caso"])],
        "Δ ENS": prom["d_ens"], "Δ SAIDI": prom["d_saidi"], "Δ SAIFI": prom["d_saifi"],
        "Reducción ENS [USD/kWh]": prom["usd_kwh"],
        "Reducción SAIDI [USD/(h/año)]": prom["usd_saidi"],
        "Reducción SAIFI [USD/(int./año)]": prom["usd_saifi"]})
    st.caption(f"Valor esperado sobre los {len(comunes)} escenarios comunes {comunes}: reducción = "
               "E[caso base] − E[estrategia]; costo unitario = E[costo total] / reducción. Mismas curvas "
               "de la pestaña Resultados de la v7. Las anchuras comparan alternativas: su suma no es "
               "lo que lograría una cartera con todas.")
    if (prom["_tipo"] == "Combinada").any():
        st.warning("Con combinaciones en la curva, el eje X no es acumulable: una combinación ya contiene "
                   "a sus medidas sueltas.")
    curvas = curvas_mac_estrategias(df_mac)
    st.markdown("#### Costos por unidad de mejora")
    unidad = tabla_costos_unidad(df_mac, curvas)
    st.dataframe(unidad.style.format(FORMATO_UNIDAD, na_rep="—"), use_container_width=True)
    descarga_csv(unidad, "costos_por_unidad_de_mejora", "mac_unidad", index=True)
    for tab, k in zip(st.tabs(["ENS", "SAIDI", "SAIFI"]), ("ens", "saidi", "saifi")):
        with tab:
            fig, tabla = curvas[k]
            if fig is None:
                st.info(f"Ninguna estrategia reduce {k.upper()} lo suficiente para dibujar la curva.")
                continue
            st.plotly_chart(_fig_pantalla(fig, height=460), use_container_width=True, key=f"fig_res_mac_{k}")
            _descargas_figura(fig, f"curva_mac_{k}", key=f"res_mac_{k}")
            descarga_csv(tabla, f"curva_mac_{k}", f"mac_{k}")


def _falta_individuales(d, producto):
    st.info(f"Para {producto} faltan las medidas individuales de este estudio. Actívalo en "
            "⚙️ Optimización y análisis (casilla correspondiente) y ejecuta los pasos pendientes.")


# ════════════════════════════════════════════════════════════
# Solapamiento
# ════════════════════════════════════════════════════════════

def tab_solapamiento(d, tipo, caso):
    from helpers_inversion import obtener_color_estrategia
    if d["dce"].empty:
        st.info("El estudio no tiene combinaciones.")
        return
    if d["dfe"].empty:
        _falta_individuales(d, "el solapamiento")
        return
    if "d_ens" not in d["dce"]:
        st.warning("Faltan los casos base para calcular abatimientos.")
        return
    dfe, dce = _validas(d["dfe"]), _validas(d["dce"])
    combos = sorted(dce["combinacion"].unique())
    c1, c2 = st.columns([3, 1])
    defecto = combos if len(combos) <= 10 else [caso if caso in combos else combos[0]]
    elegidos = c1.multiselect("Combinaciones", combos, default=defecto, key="res_sol_combos",
                              format_func=lambda c: nombre_caso("combinadas", c))
    ind = c2.radio("Indicador", ["ens", "saidi", "saifi"], format_func=str.upper, horizontal=True, key="res_sol_ind")
    if not elegidos:
        return
    comunes = sorted(set.intersection(*[set(dce.loc[dce["combinacion"] == c, "escenario"]) for c in elegidos]))
    if not comunes:
        st.info("Las combinaciones elegidas no comparten escenarios.")
        return
    res = {c: A.solapamiento_activas(dfe, dce, c, comunes, ind) for c in elegidos}
    medias = {c: float(dce[(dce["combinacion"] == c) & dce["escenario"].isin(comunes)]["d_ens_pct"].mean())
              for c in elegidos}
    orden = sorted(elegidos, key=lambda c: medias[c])
    etiqueta, unidad = {"ens": ("ENS", "kWh"), "saidi": ("SAIDI", "h/cliente"), "saifi": ("SAIFI", "int./cliente")}[ind]
    base = float(d["base"][d["base"]["escenario"].isin(comunes)][ind].mean())

    cats_suma, cats_conj, cats = [], [], []
    for n, c in enumerate(orden, 1):
        lab = f"{n}. {nombre_caso('combinadas', c)}"
        cats_suma.append(f"{lab} · suma de activas")
        cats_conj.append(f"{lab} · conjunto")
        cats += [cats_suma[-1], cats_conj[-1]]
    fig = go.Figure()
    todas = A.ORDEN_MEDIDAS + sorted({m for c in orden for m in res[c]["aportes"].index} - set(A.ORDEN_MEDIDAS))
    for m in todas:
        valores = [res[c]["aportes"].get(m, 0.0) for c in orden]
        if not any(pd.notna(v) and v > 1e-6 for v in valores):
            continue
        nom = A.NOMBRE_MEDIDA.get(m, BV._nombre(m))
        fig.add_bar(y=cats_suma, x=valores, orientation="h", name=nom,
                    marker=dict(color=obtener_color_estrategia(nom), line=dict(color="white", width=1)))
    fig.add_bar(y=cats_conj, x=[res[c]["conjunto"] for c in orden], orientation="h",
                name="Abatimiento conjunto", marker_color=A.COLOR_CONJUNTO)
    fig.add_vline(x=base, line_dash="dot", line_color="#52514e", annotation_text="Caso base",
                  annotation_position="top")
    fig.update_layout(barmode="stack", xaxis_title=f"Abatimiento de {etiqueta} [{unidad}]",
                      yaxis=dict(categoryorder="array", categoryarray=list(reversed(cats))))
    plantilla(fig, alto=max(800, 60 * len(cats) + 350))
    fig.update_yaxes(automargin=True, tickfont=dict(size=22))
    fig.update_layout(margin=dict(l=40, r=40, t=110, b=220))
    st.caption(f"Sobre los {len(comunes)} escenarios comunes {comunes}. En cada escenario se suma el abatimiento "
               "individual (truncado en 0) SOLO de las medidas que la combinación activa, y luego se promedia. "
               "La diferencia con el conjunto describe falta de aditividad; no atribuye el aporte a cada medida.")
    figura(fig, f"solapamiento_{ind}", f"res_sol_{ind}")

    tabla = pd.DataFrame([{"N": n, "Combinación": nombre_caso("combinadas", c),
                           "Suma de individuales activas": res[c]["suma"], "Abatimiento conjunto": res[c]["conjunto"],
                           "Conjunto / suma": res[c]["razon"], "Solapamiento": res[c]["solapamiento"],
                           "Datos faltantes": len(res[c]["faltan_individuales"])}
                          for n, c in enumerate(orden, 1)])
    fmt = "{:,.0f}" if ind == "ens" else "{:.3f}"
    st.dataframe(tabla.style.format({"Suma de individuales activas": fmt, "Abatimiento conjunto": fmt,
                                     "Solapamiento": fmt, "Conjunto / suma": "{:.2f}"}, na_rep="—"),
                 use_container_width=True, hide_index=True)
    if any(res[c]["faltan_individuales"] for c in orden):
        st.warning("Hay medidas activas sin corrida individual en algún escenario: su suma queda indefinida.")
    descarga_csv(tabla, f"solapamiento_{ind}", "solapamiento")

    st.markdown("#### Medidas efectivamente activas")
    act = A.tabla_activas(dce, orden, comunes)
    act = act[[m for m in act.columns if act[m].notna().any()]]
    act.columns = [A.NOMBRE_MEDIDA.get(m, BV._nombre(m)) for m in act.columns]
    act.index = [f"{n}. {nombre_caso('combinadas', c)}" for n, c in enumerate(orden, 1)]
    st.dataframe(act.style.format("{:.0f}", na_rep="—").background_gradient(cmap="Blues", vmin=0, vmax=len(comunes)),
                 use_container_width=True)
    st.caption(f"En cuántos de los {len(comunes)} escenarios el problema asigna capacidad o usa cada medida "
               "habilitada. '—' = no forma parte de la combinación.")
    descarga_csv(act, "medidas_activas", "activas", index=True)

    miembros = sorted({m for c in orden for m in A.miembros_de(c)})
    claves = [("combinadas", c) for c in orden] + [("individual", m) for m in miembros
                                                   if m in set(dfe["medida"])]
    filas, comunes_f = filas_comparables(d, claves)
    if not filas.empty:
        from graficos_combinaciones import render_combinaciones
        render_combinaciones(filas, _descargas_figura)


# ════════════════════════════════════════════════════════════
# Reglas de decisión
# ════════════════════════════════════════════════════════════

ORDEN_SEVERIDAD = {"ens": "ENS del caso base", "costo_ens": "Costo de la ENS del caso base",
                   "saidi": "SAIDI del caso base", "saifi": "SAIFI del caso base", None: "Número original"}


def _orden_severidad(base, escenarios):
    """Escenarios de menor a mayor severidad según el caso base sin medidas.

    Filas y columnas de la matriz usan este mismo orden, así la diagonal (plan
    diseñado para su propio escenario) se conserva. Los escenarios sin
    baseline quedan al final, por número.
    """
    opciones = [k for k in ORDEN_SEVERIDAD if k is None or (not base.empty and k in base)]
    col = st.selectbox("Numerar escenarios (1 = más leve)", opciones, key="res_dec_orden",
                       format_func=ORDEN_SEVERIDAD.get,
                       help="Orden de filas y columnas de la matriz y de la numeración 1..N.")
    if col is None:
        return list(escenarios)
    sev = pd.to_numeric(base.set_index("escenario")[col], errors="coerce")
    sev.index = sev.index.astype(int)

    def clave(e):
        v = float(sev.get(e, np.nan))
        return (np.isnan(v), 0.0 if np.isnan(v) else v, e)
    return sorted(escenarios, key=clave)


def desvio_inversion(impuesto, evaluado, meg_existentes=0):
    """Como run_plan_fijo.desvio_plan, pero sin contar las unidades MEG existentes
    (las primeras `meg_existentes` de CAP_MEG_MW) cuando la evaluación usa MENOS
    capacidad que la impuesta: no tienen CAPEX y usar menos solo puede empeorar
    el plan en ese escenario, así que no lo favorece. Usar más sí cuenta."""
    m = 0.0
    for clave in set(impuesto) | set(evaluado):
        da, db = impuesto.get(clave) or {}, evaluado.get(clave) or {}
        if not isinstance(da, dict) or not isinstance(db, dict):
            continue
        for k in set(da) | set(db):
            a, b = float(da.get(k, 0.0) or 0.0), float(db.get(k, 0.0) or 0.0)
            if clave == "CAP_MEG_MW" and str(k).isdigit() and int(k) < meg_existentes and b <= a + 1e-6:
                continue
            m = max(m, abs(a - b))
    return m


def leer_decision(root, combo):
    """(diagonal, evaluaciones) de una combinación: {esc: registro} y {(plan, esc): registro}.

    Solo corridas `optimal`. Sin Streamlit: la usan la pestaña y el reporte."""
    carpeta_c, dir_pf = root / "combinadas" / combo, root / "plan_fijo" / combo
    diagonal = {int(p.stem.split("_")[1]): r for p in carpeta_c.glob("esc_*.json")
                if (r := leer_json(p)).get("status") == "optimal"}
    evaluaciones = {}
    for p in dir_pf.glob("plan_esc_*/esc_*.json"):
        r = leer_json(p)
        if r.get("status") == "optimal":
            evaluaciones[(int(p.parent.name.split("_")[-1]), int(p.stem.split("_")[1]))] = r
    return diagonal, evaluaciones


def matriz_decision(diagonal, evaluaciones, planes, escenarios, usar_resuelta=True):
    """Tabla larga plan × escenario con el origen de cada celda y el desvío de inversión.

    La celda (i, i) usa la diagonal re-resuelta si existe y `usar_resuelta`; si no, la
    corrida original de la combinación."""
    filas = []
    for i in planes:
        for j in escenarios:
            reg, origen = evaluaciones.get((i, j)), "plan fijo"
            if i == j and (reg is None or not usar_resuelta):
                reg, origen = diagonal.get(i), "combinación"
            if reg:
                fila = {**{k: v for k, v in reg.items() if not isinstance(v, (dict, list))},
                        "plan": i, "escenario": j, "origen": origen}
                if origen == "plan fijo" and i in diagonal and "desvio_plan_max" in reg:
                    fila["desvio_plan_max"] = desvio_inversion(diagonal[i].get("plan", {}), reg.get("plan", {}),
                                                               int(reg.get("meg_existentes") or 0))
                filas.append(fila)
    return pd.DataFrame(filas)


def excluir_desviadas(df, columnas):
    """Pone NaN en `columnas` de las evaluaciones que no conservaron la inversión impuesta.

    Devuelve (df, n_excluidas)."""
    if "desvio_plan_max" not in df:
        return df, 0
    malas = pd.to_numeric(df["desvio_plan_max"], errors="coerce") > 1e-3
    if malas.any():
        df = df.copy()
        df.loc[malas, [c for c in columnas if c in df]] = np.nan
    return df, int(malas.sum())


def tab_decision(d, tipo, caso):
    root = d["root"]
    combos = sorted(d["dce"]["combinacion"].unique()) if not d["dce"].empty else []
    if not combos:
        st.info("Las reglas de decisión evalúan los planes de una combinación; este estudio no tiene ninguna.")
        return
    combo = st.selectbox("Combinación cuyos planes se evalúan", combos, key="res_dec_combo",
                         index=combos.index(caso) if caso in combos else
                         (combos.index(COMBINACION_TESIS) if COMBINACION_TESIS in combos else 0),
                         format_func=lambda c: nombre_caso("combinadas", c))
    diagonal, evaluaciones = leer_decision(root, combo)
    if not evaluaciones:
        st.info("Todavía no hay evaluaciones de plan fijo: cada plan de la combinación (inversión fija) "
                "resuelto en los demás escenarios, con operación libre.")
        from estudio_v8.optimizacion import panel_decision
        panel_decision(root, combo, key="res_dec")
        return

    escenarios = _orden_severidad(d["base"], sorted({j for _, j in evaluaciones} | set(diagonal)))
    planes = [e for e in escenarios if e in diagonal]
    usar_resuelta = st.checkbox("Usar la diagonal re-resuelta cuando exista", True, key="res_dec_diag",
                                help="Así toda la matriz sale con el mismo E4 y el mismo criterio de "
                                     "criterios.csv. Desmarcado, la celda plan i en escenario i es la "
                                     "corrida original de la combinación, que puede llevar la constante "
                                     "de un E4 anterior y sesgar el arrepentimiento y el EVPI.")
    df = matriz_decision(diagonal, evaluaciones, planes, escenarios, usar_resuelta)
    indicadores = [c for c in ETIQ if c in df]
    c1, c2, c3 = st.columns([2, 1.3, 2])
    indicador = c1.selectbox("Indicador de la matriz (menor es mejor)", indicadores, key="res_dec_ind",
                             format_func=ETIQ.get)
    utilidad = c2.checkbox("Expresar como utilidad", False, key="res_dec_util",
                           help="Usa el negativo del indicador: mayor es mejor (Maximin / Maximax).")
    referencia = c3.radio("Arrepentimiento contra", list(A.REFERENCIAS), key="res_dec_ref",
                          format_func=A.REFERENCIAS.get)
    df, n_malas = excluir_desviadas(df, [indicador])
    if n_malas:
        st.warning(f"{n_malas} evaluaciones no conservaron la inversión impuesta y se excluyen.")
    M = df.pivot(index="plan", columns="escenario", values=indicador).reindex(index=planes, columns=escenarios)
    pesos_txt = st.text_input("Pesos de los escenarios (vacío = iguales)", key="res_dec_pesos",
                              help="Un peso por columna, en el orden de la matriz, separados por coma.")
    try:
        pesos = None
        if pesos_txt.strip():
            ws = [float(x) for x in pesos_txt.split(",")]
            if len(ws) != len(escenarios):
                raise ValueError(f"Indica {len(escenarios)} pesos, uno por escenario.")
            pesos = dict(zip(escenarios, ws))
        r = A.reglas(-M if utilidad else M, beneficio=utilidad, pesos=pesos, referencia=referencia)
    except ValueError as exc:
        st.info(str(exc))
        return

    # En pantalla los escenarios van numerados 1..N en el orden elegido (por defecto,
    # de menor a mayor severidad del caso base); la correspondencia con el esc_XXX
    # real queda en el expander de más abajo.
    num = {e: k + 1 for k, e in enumerate(escenarios)}
    st.caption(f"Cobertura: {int(M.notna().sum().sum())}/{M.size} celdas. Criterios sobre las columnas completas "
               f"{[num[c] for c in r['columnas']]}. Las celdas vacías no son ceros.")
    if len(r["columnas"]) < len(escenarios):
        st.warning("Ranking provisional: faltan evaluaciones.")
    fig = px.imshow(M.rename(index=num, columns=num), text_auto=".4s", aspect="auto",
                    color_continuous_scale="RdYlGn_r",
                    labels=dict(x="Escenario de evaluación", y="Plan diseñado para el escenario", color=""))
    ticks = list(num.values())
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=ticks)
    # imshow invierte el eje y (fila 1 arriba); aquí va de 1 abajo a N arriba, como el eje x.
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=ticks, autorange=True)
    fig.update_traces(textfont_size=20)
    figura(plantilla(fig, alto=max(700, 70 * len(planes) + 300)), f"matriz_{indicador}_{_slug(combo)}", "res_dec_matriz")

    elegidos = pd.DataFrame([{"Criterio": k,
                              "Plan elegido (escenario de diseño)": ", ".join(str(num[x]) for x in v) or "—"}
                             for k, v in r["elegidos"].items()])
    a, b = st.columns([2, 1])
    a.dataframe(elegidos, hide_index=True, use_container_width=True)
    b.metric("EVPI en la muestra", f"{r['evpi']:,.2f}" if np.isfinite(r["evpi"]) else "—",
             help="Cuánto peor es comprometerse con el plan de mejor valor esperado que decidir con "
                  "información perfecta escenario a escenario (diagonal).")
    st.caption("Costos: Minimax = menor peor resultado · Minimin = menor mejor resultado. "
               "Utilidad: Maximin = mayor mínimo · Maximax = mayor máximo. Minimax regret = menor pérdida máxima "
               "frente a la referencia elegida. Los empates se conservan.")
    st.caption("⚠️ Decisión pendiente: qué referencia de arrepentimiento reporta la tesis (diagonal o mejor candidato). "
               "Pueden dar rankings distintos.")
    crit = r["criterios"].rename(index=num)
    crit.index.name = "Plan"
    st.dataframe(crit.style.format("{:,.2f}", na_rep="—")
                 .highlight_min(subset=["Arrepentimiento máximo"], color="#c7f0d8"),
                 use_container_width=True)
    descarga_csv(crit, f"criterios_{indicador}", "criterios", index=True)
    with st.expander("Matriz de arrepentimiento"):
        st.dataframe(r["regret"].rename(index=num, columns=num).style.format("{:,.2f}")
                     .background_gradient(cmap="Reds"), use_container_width=True)
        st.caption("Valores negativos pequeños = diferencias dentro del MIPGap: la referencia no es un óptimo exacto.")
    with st.expander("Numeración de los escenarios"):
        tabla = pd.DataFrame({"N°": list(num.values()),
                              "Escenario original": [f"esc_{e:03d}" for e in num]})
        if not d["base"].empty:
            b = d["base"].assign(escenario=d["base"]["escenario"].astype(int)).set_index("escenario")
            for c in ("ens", "costo_ens", "saidi", "saifi"):
                if c in b:
                    tabla[f"{ORDEN_SEVERIDAD[c]}"] = [b[c].get(e, np.nan) for e in num]
        st.dataframe(tabla, hide_index=True, use_container_width=True)
        st.caption(f"La matriz y las tablas de arriba numeran los escenarios de 1 a N según "
                   f"«{ORDEN_SEVERIDAD[st.session_state.get('res_dec_orden')]}» (1 = más leve). "
                   "El plan N° k es el diseñado para el escenario N° k.")
    with st.expander("Calidad de las celdas"):
        cols = [c for c in ("plan", "escenario", "origen", "solver_gap", "solver_term", "desvio_plan_max", "segundos") if c in df]
        st.dataframe(df[cols].sort_values(["plan", "escenario"]), hide_index=True, use_container_width=True)
    faltan = len(planes) * len(escenarios) - int(M.notna().sum().sum())
    if faltan:
        with st.expander(f"Calcular las {faltan} celdas faltantes"):
            from estudio_v8.optimizacion import panel_decision
            panel_decision(root, combo, key="res_dec_faltan")
