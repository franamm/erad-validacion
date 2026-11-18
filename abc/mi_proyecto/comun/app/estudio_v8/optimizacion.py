"""Pestaña ⚙️ Optimización y análisis.

Arma los pasos de cálculo que faltan para los análisis elegidos y los lanza
como proceso aparte (`estudio_v8.jobs`). La página solo lee disco: se puede
cerrar el navegador y volver, y el avance sigue ahí.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import barrido_vista as BV
import pipeline as P
from aislacion import variante as variante_aislacion
from run_barrido_medidas import MEDIDAS
from estudio_v8 import analisis as A, estado as E, jobs, progreso as PR
from estudio_v8.core import APP_DIR, ART_DIR, MEDIDAS_TESIS, indices

ESTUDIO_TESIS = "barrido/sec-0-2-9-10 severo"
NUEVO = "➕ Nuevo estudio"
PY = sys.executable
# Orden con que se nombran las combinaciones: el de las carpetas existentes.
ORDEN = list(MEDIDAS_TESIS) + [m for m in MEDIDAS if m not in MEDIDAS_TESIS]
# Mismos rótulos y columnas que la pestaña Optimización de la v7.
TOGGLES = [
    [("dg_sol", "☀️ DG Solar"), ("dg_eol", "💨 DG Eólica"), ("dg_fos", "⛽ DG Fósil"), ("meg", "🚛 MEG")],
    [("ess_4h", "🔋 ESS 4h"), ("ess_5h", "🔋 ESS 5h"), ("ess_6h", "🔋 ESS 6h"), ("mess", "🚚 MESS")],
    [("prior_criticos", "Priorización de cargas críticas"), ("edrp", "📉 EDRP (Cargas Interrumpibles)"),
     ("veg_mgmt", "🌳 Gestión de vegetación"), ("refuerzo", "🔩 Refuerzo estructural"),
     ("soterramiento", "🕳️ Soterramiento")],
    [("reconfig_topo", "🔀 Reconfiguración topológica"), ("inv_tie", "🔗 Inversión en líneas de enlace"),
     ("inv_switch_rcs", "🔌 Inversión Switch RCS (automático)"),
     ("inv_switch_ms", "🔌 Inversión Switch MS (manual)")],
]
CLAVES_EXTRAS = [c for _, _, campos in BV.CONDICIONALES for c, *_ in campos]
SEG_POR_CORRIDA = 75     # GUIA_BARRIDO §5, 8760 h con MIPGap 2 %
SEG_POR_E5 = 60


# ════════════════════════════════════════════════════════════
# Lectura del estudio (cacheada unos segundos: son cientos de .json)
# ════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _red(art_dir, mtime):
    return P._cargar("red.pkl", art_dir)


def red(art_dir=None):
    carpeta = Path(art_dir or ART_DIR).resolve()
    ruta = carpeta / "red.pkl"
    return _red(str(carpeta), ruta.stat().st_mtime_ns) if ruta.exists() else None


@st.cache_data(show_spinner=False)
def _meta_e4(ruta, mtime_ns, tamano):
    import pickle
    with open(ruta, "rb") as archivo:
        return pickle.load(archivo).get("_meta") or {}


def meta_e4(art_dir, etiqueta):
    ruta = P.ruta_sin_fallas(etiqueta, art_dir)
    sello = ruta.stat()
    return _meta_e4(str(ruta), sello.st_mtime_ns, sello.st_size)


def incompatibilidad_e4(meta, sup):
    """Misma validación que el ejecutor, antes de ofrecer lanzar un trabajo."""
    from run_barrido_medidas import verificar_lim_dg
    try:
        verificar_lim_dg({"_meta": meta}, sup["lim_dg"], sup["e4"],
                         horas=sup["horas"], lim_dg_modo=sup["lim_dg_modo"],
                         lim_dg_bus=sup.get("lim_dg_bus"))
    except SystemExit as exc:
        # Solo el diagnóstico; los comandos del ejecutor omiten la ruta del caso.
        return str(exc).split("\n  ")[0].strip()
    return None


def numesc(art_dir=None):
    r = red(art_dir)
    return int(r["numesc"]) if r else 0


def estudios_disponibles(art_dir=None):
    nombres = set()
    for tanda in Path(art_dir or ART_DIR).glob("barrido*"):
        if tanda.is_dir():
            for d in tanda.iterdir():
                if d.is_dir() and ((d / "resumen_individual.csv").exists()
                                   or (d / "resumen_combinadas.csv").exists()
                                   or any(d.glob("combinadas/*/esc_*.json"))):
                    nombres.add(f"{tanda.name}/{d.name}")
    return sorted(nombres)


@st.cache_data(ttl=20, show_spinner=False)
def contenido(root):
    return E.contenido_estudio(root)


@st.cache_data(ttl=20, show_spinner=False)
def _supuestos_crudos(root):
    return E.supuestos_estudio(root)


@st.cache_data(ttl=20, show_spinner=False)
def segundos_promedio(root):
    seg = [float(r["segundos"]) for *_, r in E.registros_estudio(root)
           if r.get("status") == "optimal" and r.get("segundos")]
    return float(np.mean(seg)) if seg else SEG_POR_CORRIDA


@st.cache_data(show_spinner="Leyendo la configuración guardada del estudio…")
def _extras_guardados(ruta_gz, mtime):
    entrada = next(iter(P.leer_historial(Path(ruta_gz)).values()))
    cfg = entrada.get("config", {})
    return {k: cfg[k] for k in CLAVES_EXTRAS + ["gurobi_time_limit"] if k in cfg}


def extras_estudio(root):
    """Dimensionamiento de MEG/MESS/MS y tiempo límite, desde el .pkl.gz de la
    combinación más grande (el .json no los registra)."""
    gzs = sorted(Path(root).glob("combinadas/*/esc_*.pkl.gz"),
                 key=lambda p: -len(p.parent.name.split("+")))
    gzs = gzs or sorted(Path(root).glob("individual/*/esc_*.pkl.gz"))
    return _extras_guardados(str(gzs[0]), gzs[0].stat().st_mtime) if gzs else {}


def supuestos_estudio(root):
    """(supuestos para los comandos, supuestos mezclados). None si no hay corridas."""
    comunes, mezclados = _supuestos_crudos(str(root))
    if not comunes:
        return None, mezclados
    ex = extras_estudio(root)
    e4 = comunes.get("e4_sin_fallas")
    return {"seccionadores": E.seccionadores_de(comunes.get("variante_aislacion")),
            "e4": "" if e4 in (None, "historico") else e4,
            "lim_dg": comunes.get("lim_dg"), "lim_dg_modo": comunes.get("lim_dg_modo"),
            "horas": int(comunes.get("horas") or 8760),
            "mip_gap": float(comunes.get("mip_gap") or 0.02),
            "microredes": bool(comunes.get("microredes")),
            "time_limit": float(ex.get("gurobi_time_limit") or 1800),
            "extras": {k: ex[k] for k in CLAVES_EXTRAS if k in ex}}, mezclados


# ════════════════════════════════════════════════════════════
# Comandos (los mismos que se muestran en la guía)
# ════════════════════════════════════════════════════════════

def caso_de_estudio(root):
    """Carpeta del caso propietario de un estudio o trabajo."""
    ruta = Path(root).resolve()
    for padre in (ruta, *ruta.parents):
        if padre.name.startswith("barrido") or padre.name == "trabajos_v8":
            return padre.parent
    return ART_DIR


def _lista(xs):
    return ",".join(map(str, xs))


def _flags_gurobi(threads=0, mem_gb=0, method=None, presolve=None):
    """Recursos y algoritmo de Gurobi; lo que queda en 0/None lo decide el optimizador."""
    f = []
    if threads:
        f += ["--threads", str(int(threads))]
    if mem_gb:
        f += ["--mem_gb", str(mem_gb)]
    if method is not None:
        f += ["--method", str(int(method))]
    if presolve is not None:
        f += ["--presolve", str(int(presolve))]
    return f


def _flags(sup, *recursos):
    f = ["--horas", str(sup["horas"]), "--mip_gap", str(sup["mip_gap"]),
         "--time_limit", str(int(sup["time_limit"]))]
    if sup.get("lim_dg") is not None:
        f += ["--lim-dg", str(sup["lim_dg"])]
    if sup.get("lim_dg_modo"):
        f += ["--lim-dg-modo", sup["lim_dg_modo"]]
    if sup.get("e4"):
        f += ["--sin-fallas", sup["e4"]]
    if sup.get("seccionadores"):
        f += ["--seccionadores", _lista(sup["seccionadores"])]
    for k, v in (sup.get("extras") or {}).items():
        f += [f"--{k}", str(v)]
    if sup.get("microredes"):
        f += ["--microredes"]
    return f + _flags_gurobi(*recursos)


def cmd_barrido(root, sup, medidas, escenarios, fases, combo=False, completo=False,
                criterio="objetivo", recursos=(0, 0)):
    cmd = [PY, str(APP_DIR / "run_barrido_medidas.py"), "--medidas", _lista(medidas),
           "--escenarios", _lista(escenarios), "--fases", fases, "--salida", str(root),
           "--criterio", criterio, "--completo" if completo else "--no-completo",
           "--artefactos", str(caso_de_estudio(root))]
    if combo:
        cmd += ["--no-todas", "--combo", _lista(medidas), "--combo-escenarios", _lista(escenarios)]
    return cmd + _flags(sup, *recursos)


def cmd_recosteo(root, sup, escenarios, recursos=(0, 0)):
    return ([PY, str(APP_DIR / "run_barrido_medidas.py"), "--recostear-dg",
             "--escenarios", _lista(escenarios), "--salida", str(root),
             "--artefactos", str(caso_de_estudio(root))] + _flags(sup, *recursos))


def cmd_plan_fijo(root, combo, planes, escenarios, resolver_diagonal, sup, recursos=(0, 0),
                  solo_matriz=False):
    cmd = [PY, str(APP_DIR / "run_plan_fijo.py"), "--barrido", str(root), "--combinacion", combo,
           "--planes", _lista(planes), "--escenarios", _lista(escenarios),
           "--resolver-diagonal" if resolver_diagonal else "--no-resolver-diagonal",
           "--artefactos", str(caso_de_estudio(root))]
    if solo_matriz:
        return cmd + ["--solo-matriz"]
    cmd += ["--mip_gap", str(sup["mip_gap"]), "--time_limit", str(int(sup["time_limit"]))]
    return cmd + _flags_gurobi(*recursos)


def cmd_e5(seccionadores, escenarios, art_dir=None):
    cmd = [PY, str(APP_DIR / "pipeline.py"), "--etapas", "E5", "--escenarios", _lista(escenarios),
           "--artefactos", str(art_dir or ART_DIR)]
    return cmd + (["--seccionadores", _lista(seccionadores)] if seccionadores else [])


def paso(nombre, comando, rutas, seg, tipo, milp=True):
    rutas = [Path(r) for r in rutas]
    hechas = sum(r.exists() for r in rutas)
    return {"nombre": nombre, "comando": comando, "rutas": rutas, "total": len(rutas),
            "guardadas": hechas, "pendientes": len(rutas) - hechas, "seg": seg,
            "tipo": tipo, "milp": milp}


def armar_pasos(root, sup, medidas, escenarios, productos, opciones, combo=None):
    """Pasos mínimos, en orden, para obtener `productos` sobre esa muestra."""
    root = Path(root)
    art_dir = caso_de_estudio(root)
    medidas = [m for m in ORDEN if m in medidas]
    recursos = (opciones.get("threads", 0), opciones.get("mem_gb", 0),
                opciones.get("method"), opciones.get("presolve"))
    seg = opciones.get("seg", SEG_POR_CORRIDA)
    pasos = []
    var = variante_aislacion(bool(sup["seccionadores"]), sup["seccionadores"])
    faltan = [e for e in escenarios if not (P.base_dir(var, art_dir) / f"esc_{e:03d}.npz").exists()]
    if faltan:
        pasos.append(paso(f"Casos base E5 faltantes ({var})", cmd_e5(sup["seccionadores"], faltan, art_dir),
                          [P.base_dir(var, art_dir) / f"esc_{e:03d}.npz" for e in faltan], SEG_POR_E5, "e5"))
    combo = combo or ("combo_" + "+".join(medidas) if len(medidas) > 1 else None)
    if len(medidas) == 1 or {"mac", "solapamiento"} & set(productos):
        rutas = [r for *_, r in E.rutas_esperadas(root, medidas, escenarios)]
        pasos.append(paso(f"Medidas individuales ({len(medidas)} × {len(escenarios)})",
                          cmd_barrido(root, sup, medidas, escenarios, "1,2",
                                      completo=opciones.get("completo", False),
                                      criterio=opciones.get("criterio", "objetivo"), recursos=recursos),
                          rutas, seg, "individual"))
    if combo:
        rutas = [r for *_, r in E.rutas_esperadas(root, combos=[combo], escenarios=escenarios)]
        # El plan fijo recupera la configuración desde un .pkl.gz de la combinación.
        sin_gz = not any((root / "combinadas" / combo).glob("esc_*.pkl.gz"))
        pasos.append(paso(f"Combinación de {len(medidas)} medidas × {len(escenarios)} escenarios",
                          cmd_barrido(root, sup, medidas, escenarios, "3", combo=True,
                                      completo=opciones.get("completo", False)
                                      or ("decision" in productos and sin_gz),
                                      recursos=recursos),
                          rutas, seg * 2, "combo"))
    if "decision" in productos and combo:
        if opciones.get("recostear"):
            pasos.append(paso("Recostear DG de las corridas guardadas (sin MILP)",
                              cmd_recosteo(root, sup, escenarios, recursos), [], 0, "recosteo", milp=False))
        diag = opciones.get("resolver_diagonal", False)
        rutas = [r for t, c, j, r in E.rutas_esperadas(root, combos=[combo], planes=escenarios,
                                                       escenarios_plan=escenarios)
                 if t == "plan_fijo" and (diag or not c.endswith(f"plan_esc_{j:03d}"))]
        pasos.append(paso(f"Planes fijos ({len(escenarios)} planes × {len(escenarios)} escenarios)",
                          cmd_plan_fijo(root, combo, escenarios, escenarios, diag, sup, recursos),
                          rutas, seg * 2, "plan_fijo"))
    return pasos, combo


# ════════════════════════════════════════════════════════════
# Ejecución y seguimiento
# ════════════════════════════════════════════════════════════

def panel_ejecucion(salida, pasos, config, key):
    """Tabla de pasos, comandos exactos, botón y seguimiento en vivo."""
    salida = Path(salida)
    config = {**config, "artefactos": str(caso_de_estudio(salida))}
    info = jobs.estado(salida)
    corriendo = bool(info) and info.get("estado") in ("iniciando", "ejecutando") and jobs.vivo(info)
    necesarios = [p for p in pasos if p["pendientes"] or not p["milp"]]
    if pasos:
        st.dataframe(pd.DataFrame([{
            "Paso": p["nombre"],
            "Corridas": p["total"] if p["milp"] else "—",
            "Guardadas": p["guardadas"] if p["milp"] else "—",
            "Pendientes": p["pendientes"] if p["milp"] else "—",
            "Tiempo estimado": PR.formato_duracion(p["pendientes"] * p["seg"]) if p["milp"] else "segundos",
        } for p in pasos]), hide_index=True, use_container_width=True)
    with st.expander("Comandos exactos (se ejecutan en este orden)"):
        st.code("\n".join(subprocess.list2cmdline(p["comando"]) for p in necesarios)
                or "Nada pendiente.", language="powershell")
        st.caption(f"Registro y estado del trabajo: `{salida}`. Los comandos también se pueden "
                   "copiar a una consola; reanudar no repite corridas guardadas.")
    if corriendo:
        seguimiento(salida)
        return
    if not necesarios:
        st.success("Todo lo necesario ya está calculado. Revísalo en 📊 Resultados.")
    elif st.button(f"▶️ Ejecutar {len(necesarios)} paso(s) en segundo plano", type="primary",
                   key=f"{key}_lanzar"):
        try:
            jobs.lanzar([p["comando"] for p in necesarios], salida, config,
                        [r for p in necesarios for r in p["rutas"]])
            st.success("Trabajo iniciado. Puedes seguir navegando o cerrar el navegador.")
            time.sleep(0.5)
            st.rerun()
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
    seguimiento(salida)


def seguimiento(salida):
    info = jobs.estado(salida)
    if not info:
        return
    if info.get("estado") in ("iniciando", "ejecutando"):
        _seguimiento_vivo(str(salida))
    else:
        _mostrar_seguimiento(Path(salida), info)


@st.fragment(run_every=5)
def _seguimiento_vivo(salida):
    # Solo este bloque se refresca cada 5 s; el resto de la página no.
    salida = Path(salida)
    _mostrar_seguimiento(salida, jobs.estado(salida))


def _preliminares(esperados):
    filas = []
    for p in map(Path, esperados):
        if p.suffix != ".json" or not p.exists():
            continue
        r = E.leer_json(p)
        if r.get("status") != "optimal":
            continue
        caso = p.parent.name if not p.parent.name.startswith("plan_esc_") \
            else f"plan {int(p.parent.name.split('_')[-1])}"
        filas.append({"Caso": caso, "Escenario": int(p.stem.split("_")[1]),
                      "ENS [kWh]": r.get("ens"), "SAIDI [h/cl.]": r.get("saidi"),
                      "SAIFI [int./cl.]": r.get("saifi"), "Costo total [USD/año]": r.get("cst_costo_total"),
                      "Gap": r.get("solver_gap"), "Segundos": r.get("segundos"), "_t": p.stat().st_mtime})
    return pd.DataFrame(filas).sort_values("_t") if filas else pd.DataFrame()


def _mostrar_seguimiento(salida, info):
    est = info.get("estado", "desconocido")
    icono = {"ejecutando": "⏳", "iniciando": "⏳", "terminado": "✅", "incompleto": "⚠️",
             "fallido": "❌", "detenido": "⏹️", "interrumpido": "⚠️"}.get(est, "ℹ️")
    with st.container(border=True):
        n_pasos = len(info.get("comandos") or [1])
        st.markdown(f"**{icono} Trabajo {est}** · paso {info.get('paso', 1)} de {n_pasos}")
        a = PR.avance(info.get("esperados", []), info.get("inicio"))
        if a["total"]:
            st.progress(a["hechas"] / a["total"],
                        text=f"{a['hechas']} de {a['total']} corridas guardadas · {a['optimas']} con solución")
        vivo = est in ("ejecutando", "iniciando")
        c = st.columns(4)
        c[0].metric("Resolviendo hace", PR.formato_duracion(a["en_curso_s"]) if vivo else "—",
                    help="Tiempo desde que se guardó la última corrida (o desde el inicio).")
        c[1].metric("Promedio por corrida", PR.formato_duracion(a["promedio_s"]))
        c[2].metric("Restante estimado", PR.formato_duracion(a["restante_s"]) if vivo else "—")
        if vivo and a["actual"]:
            actual = Path(a["actual"])
            c[3].caption(f"En curso: **{actual.parent.name}** · escenario {actual.stem.split('_')[-1]}")
        texto = PR.cola(salida / "trabajo_v8.log")
        sol = PR.ultimo_solver(texto) if vivo else None
        if sol:
            st.caption(f"Gurobi · incumbente {sol['incumbente']:,.0f} · cota {sol['cota']:,.0f} · "
                       f"gap {sol['gap_pct']:.2f} %"
                       + (f" · {sol['segundos']} s de solver" if sol["segundos"] is not None else ""))
        prelim = _preliminares(info.get("esperados", []))
        if not prelim.empty:
            n, tot = len(prelim), a["total"]
            st.markdown(f"**Resultados {'preliminares' if n < tot else 'de este trabajo'} ({n}/{tot})**")
            k = st.columns(3)
            k[0].metric("ENS media [kWh]", f"{prelim['ENS [kWh]'].mean():,.0f}")
            k[1].metric("SAIDI medio [h/cl.]", f"{prelim['SAIDI [h/cl.]'].mean():,.3f}")
            if prelim["Costo total [USD/año]"].notna().any():
                k[2].metric("Costo total medio [USD/año]", f"{prelim['Costo total [USD/año]'].mean():,.0f}")
            st.dataframe(prelim.drop(columns="_t").tail(10).style.format({
                "ENS [kWh]": "{:,.0f}", "SAIDI [h/cl.]": "{:.3f}", "SAIFI [int./cl.]": "{:.3f}",
                "Costo total [USD/año]": "{:,.0f}", "Gap": "{:.2%}", "Segundos": "{:,.0f}"}, na_rep="—"),
                hide_index=True, use_container_width=True)
            st.caption("Promedios parciales sobre las corridas terminadas: cambian hasta completar la muestra.")
        if info.get("codigo_salida"):
            st.error(f"Un paso terminó con código {info['codigo_salida']}. Revisa el registro.")
        if vivo and st.button("⏹️ Detener", key=f"detener_{salida}"):
            jobs.detener(salida)
            st.rerun()
        with st.expander("Registro de ejecución"):
            st.code("\n".join(texto.splitlines()[-60:]) or "(sin salida todavía)", language="text")


# ════════════════════════════════════════════════════════════
# Paneles reutilizados desde Resultados
# ════════════════════════════════════════════════════════════

def panel_decision(root, combo, key):
    sup, mezclados = supuestos_estudio(root)
    if not sup:
        st.info("El estudio no tiene corridas con solución para recuperar sus supuestos.")
        return
    planes = sorted(e for t, c, e, r in E.registros_estudio(root)
                    if t == "combinadas" and c == combo and r.get("status") == "optimal")
    if len(planes) < 2:
        st.info("Se necesitan al menos dos escenarios resueltos de la combinación.")
        return
    c = st.columns(2)
    recostear = c[0].checkbox("Recostear DG antes (sin MILP)", _recosteo_sugerido(root, combo, sup),
                              key=f"{key}_recostear", help=AYUDA_RECOSTEO)
    diag = c[1].checkbox("Re-resolver la diagonal", False, key=f"{key}_diag", help=AYUDA_DIAGONAL)
    pasos, _ = armar_pasos(root, sup, A.miembros_de(combo), planes, {"decision"},
                           {"recostear": recostear, "resolver_diagonal": diag,
                            "seg": segundos_promedio(str(root))}, combo=combo)
    pasos = [p for p in pasos if p["tipo"] in ("e5", "recosteo", "plan_fijo")]
    panel_ejecucion(root, pasos, {"combinacion": combo, "planes": planes}, key)


def panel_detalle_horario(root, tipo, caso, esc, key):
    """Re-resuelve UN escenario guardando el .pkl.gz, en una carpeta aparte."""
    sup, _ = supuestos_estudio(root)
    if not sup:
        return
    salida = Path(root) / "detalle_horario"
    if tipo == "combinadas":
        medidas = A.miembros_de(caso)
        cmd = cmd_barrido(salida, sup, medidas, [esc], "3", combo=True, completo=True)
    else:
        cmd = cmd_barrido(salida, sup, [caso], [esc], "1", completo=True)
    ruta = salida / tipo / caso / f"esc_{esc:03d}.json"
    panel_ejecucion(salida, [paso(f"Detalle horario de escenario {esc}", cmd, [ruta],
                                  segundos_promedio(str(root)), "detalle")], {}, key)


AYUDA_RECOSTEO = ("Recalcula los costos de DG de las corridas guardadas con el E4 vigente, leyendo "
                  "sus .pkl.gz. No resuelve ningún MILP. Necesario si el E4 cambió después de "
                  "resolver la combinación.")
AYUDA_DIAGONAL = ("La celda plan i en escenario i ya está resuelta en la combinación. La matriz usa "
                  "cst_objetivo_plan, que no depende de la constante del E4, así que no hace falta "
                  "volver a resolverla. Actívalo solo si quieres toda la matriz con la misma pasada.")


def _recosteo_sugerido(root, combo, sup):
    jsons = list((Path(root) / "combinadas" / combo).glob("esc_*.json"))
    e4 = P.ruta_sin_fallas(sup["e4"], caso_de_estudio(root))
    return bool(jsons) and e4.exists() and min(p.stat().st_mtime for p in jsons) < e4.stat().st_mtime


# ════════════════════════════════════════════════════════════
# Pestaña
# ════════════════════════════════════════════════════════════

def _marcar(claves):
    for grupo in TOGGLES:
        for k, _ in grupo:
            st.session_state[f"opt_m_{k}"] = k in claves


@st.cache_data(show_spinner="Calculando la severidad de cada escenario…")
def _poblacion(variante, escenarios, horas, art_dir, sello):
    return [P.indicadores_baseline(e, variante, horas, art_dir, guardar_cache=False)
            for e in escenarios]


def poblacion(variante, escenarios, horas, art_dir):
    carpeta = P.base_dir(variante, art_dir)
    archivos = [carpeta / f"esc_{e:03d}.npz" for e in escenarios]
    archivos += [Path(art_dir) / n for n in ("red.pkl", "costos.pkl", "perfiles.npz")]
    sello = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in archivos if p.exists())
    return _poblacion(variante, tuple(escenarios), horas, str(art_dir), sello)


def render():
    art_dir = Path(st.session_state.get("v8_caso", ART_DIR)).resolve()
    if st.session_state.get("_opt_caso") != str(art_dir):
        for clave in list(st.session_state):
            if clave.startswith("opt_") or clave == "_opt_ini":
                del st.session_state[clave]
        st.session_state["_opt_caso"] = str(art_dir)
        st.session_state.pop("_opt_preparacion", None)
    from estudio_v8.core import leer_json
    preparacion = leer_json(art_dir / "trabajos_v8/pipeline/trabajo_v8.json").get("config", {})
    firma_preparacion = (str(art_dir), preparacion.get("e4"), tuple(preparacion.get("seccionadores", [])))
    if (preparacion.get("e4") in P.sin_fallas_disponibles(art_dir)
            and st.session_state.get("_opt_preparacion") != firma_preparacion):
        st.session_state["opt_e4"] = preparacion["e4"]
        st.session_state["opt_secc"] = _lista(preparacion.get("seccionadores", []))
        st.session_state["_opt_preparacion"] = firma_preparacion
    st.header("⚙️ Optimización y análisis")
    st.caption("Elige qué quieres obtener: se calcula solo lo que falta, como proceso aparte. "
               "Puedes cerrar el navegador; el avance y los resultados quedan en disco.")

    # ── 1 · estudio ──────────────────────────────────────────────────────
    st.caption(f"Artefactos del caso: `{art_dir}`")
    estudios = estudios_disponibles(art_dir)
    opciones = [NUEVO] + estudios
    with st.container(border=True):
        st.markdown("#### 1 · Estudio")
        c1, c2 = st.columns([2, 3])
        eleccion = c1.selectbox("Estudio", opciones, key="opt_estudio",
                                index=opciones.index(ESTUDIO_TESIS) if ESTUDIO_TESIS in opciones else 0,
                                format_func=lambda n: n.replace("/", "  ·  ", 1))
        if eleccion == NUEVO:
            nombre = c2.text_input(f"Carpeta nueva dentro de {art_dir.name}/barrido/", "estudio-v8", key="opt_nuevo")
            if not nombre.strip() or any(s in nombre for s in "/\\:") or nombre.strip() in (".", ".."):
                st.error("Usa un nombre de carpeta simple, sin separadores de ruta.")
                return
            root = art_dir / "barrido" / nombre.strip()
            if root.exists():
                st.warning("Esa carpeta ya existe: selecciónala en la lista para heredar sus supuestos.")
            cont = {"medidas": [], "combos": [], "escenarios": [], "planes_fijos": []}
            sup_est, mezclados = None, {}
        else:
            root = art_dir / eleccion
            cont = contenido(str(root))
            sup_est, mezclados = supuestos_estudio(root)
            m = c2.columns(4)
            m[0].metric("Medidas individuales", len(cont["medidas"]))
            m[1].metric("Combinaciones", len(cont["combos"]))
            m[2].metric("Escenarios", len(cont["escenarios"]))
            m[3].metric("Planes fijos", len(cont["planes_fijos"]))
            if mezclados:
                st.warning(f"Las corridas guardadas mezclan supuestos {list(mezclados)}; "
                           "no son comparables entre sí.")

    if st.session_state.get("_opt_ini") != str(root):
        grande = max(cont["combos"], key=lambda c: len(A.miembros_de(c))) if cont["combos"] else None
        sel = set(A.miembros_de(grande)) if grande else (set(cont["medidas"]) or set(MEDIDAS_TESIS))
        _marcar(sel)
        st.session_state["opt_mac"] = bool(cont["medidas"])
        st.session_state["opt_solap"] = bool(cont["medidas"] and cont["combos"])
        st.session_state["opt_decision"] = bool(cont["planes_fijos"])
        st.session_state["_opt_ini"] = str(root)

    # ── 2 · análisis ─────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("#### 2 · Qué análisis quieres")
        c = st.columns(4)
        c[0].checkbox("🕸️ Red, indicadores y ENS", value=True, disabled=True, key="opt_base",
                      help="Siempre: grafo con el plan instalado, indicadores técnicos y económicos, "
                           "ENS por zona y por hora, rangos del plan.")
        mac = c[1].checkbox("📉 Curvas MAC", key="opt_mac",
                            help="Agrega cada medida por separado en los mismos escenarios.")
        sol = c[2].checkbox("🔗 Solapamiento", key="opt_solap",
                            help="Agrega las medidas individuales para compararlas con la combinación.")
        dec = c[3].checkbox("⚖️ Reglas de decisión", key="opt_decision",
                            help="Agrega la evaluación de cada plan de la combinación en los demás "
                                 "escenarios (inversión fija, operación libre).")
        productos = {p for p, on in (("mac", mac), ("solapamiento", sol), ("decision", dec)) if on}

    # ── 3 · medidas ──────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("#### 3 · Medidas habilitadas")
        b = st.columns([1.4, 1, 1, 3])
        b[0].button("🎓 Las 12 de la tesis", on_click=_marcar, args=(set(MEDIDAS_TESIS),), key="opt_b12")
        b[1].button("✅ Marcar todas", on_click=_marcar, args=(set(MEDIDAS),), key="opt_btodas")
        b[2].button("↩️ Desmarcar", on_click=_marcar, args=(set(),), key="opt_bninguna")
        cols = st.columns(4)
        for col, grupo in zip(cols, TOGGLES):
            for k, lab in grupo:
                col.toggle(lab, key=f"opt_m_{k}")
        medidas = [k for k in ORDEN if st.session_state.get(f"opt_m_{k}")]
        st.caption(f"{len(medidas)} medida(s) habilitada(s). Habilitar no obliga a instalar: "
                   "el optimizador decide qué usar, y Resultados muestra qué se activó.")

    # ── 4 · supuestos ────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("#### 4 · Supuestos de la red y del solver")
        if sup_est:
            sup = dict(sup_est)
            st.caption("Heredados de las corridas guardadas: para cambiarlos, crea otro estudio.")
            k = st.columns(5)
            k[0].metric("Seccionadores", _lista(sup["seccionadores"]) or "ninguno")
            k[1].metric("Caso sin fallas (E4)", sup["e4"] or "histórico")
            k[2].metric("Tope DG", f"{sup['lim_dg']} · {sup['lim_dg_modo']}")
            k[3].metric("Horizonte", f"{sup['horas']} h")
            k[4].metric("MIPGap", f"{sup['mip_gap']:.1%}")
            if sup["extras"]:
                st.caption("Dimensionamiento guardado: " + " · ".join(f"{k}={v}" for k, v in sup["extras"].items()))
        else:
            c = st.columns(3)
            e4s = P.sin_fallas_disponibles(art_dir)
            if not e4s:
                st.error("Falta el caso sin fallas (E4). Prepáralo en 📥 Datos de entrada.")
                return
            e4 = c[0].selectbox("Caso sin fallas (E4)", e4s, key="opt_e4",
                                index=e4s.index("LIM-0.30-ALIM") if "LIM-0.30-ALIM" in e4s else 0,
                                format_func=lambda e: e or "sin_fallas.pkl (histórico)")
            meta = meta_e4(art_dir, e4)
            sello_e4 = (str(art_dir), e4, repr(sorted(meta.items())))
            if st.session_state.get("_opt_e4_supuestos") != sello_e4:
                st.session_state["opt_lim"] = float(meta.get("lim_dg", 1.0))
                st.session_state["opt_lim_modo"] = meta.get("lim_dg_modo") or "bus"
                st.session_state["opt_horas"] = int(meta.get("horas") or 8760)
                st.session_state["_opt_e4_supuestos"] = sello_e4
            st.caption("Tope DG, modo y horizonte iniciales leídos del E4 seleccionado.")
            sec_txt = c[0].text_input("Seccionadores existentes",
                                      _lista((red(art_dir) or {}).get("seccionadores_existentes", [])), key="opt_secc",
                                      help="Vacío = aislación perfecta.")
            try:
                secc = indices(sec_txt) if sec_txt.strip() else []
            except ValueError as exc:
                st.error(str(exc))
                return
            lim = c[1].number_input("Tope de DG [fracción de la demanda]", 0.0, max(5.0, float(meta.get("lim_dg", 1.0))), key="opt_lim")
            modo_lim = c[1].selectbox("Tope aplicado por", ["alimentador", "bus"], key="opt_lim_modo")
            horas = c[2].number_input("Horizonte [h]", 1, max(8760, int(meta.get("horas") or 8760)), step=1, key="opt_horas")
            micro = c[2].toggle("🏝️ Formación de microrredes", key="opt_micro",
                                help="Solo tiene efecto con Switch RCS o MS.")
            extras = {}
            for m, titulo, campos in BV.CONDICIONALES:
                if m in medidas:
                    st.markdown(f"**{titulo}**")
                    for col, (clave, etiq, mn, mx, val, pas) in zip(st.columns(len(campos)), campos):
                        extras[clave] = col.number_input(etiq, min_value=type(val)(mn), max_value=type(val)(mx),
                                                         value=val, step=type(val)(pas), key=f"opt_p_{clave}")
            sup = {"seccionadores": secc, "e4": e4, "lim_dg": lim, "lim_dg_modo": modo_lim,
                   "horas": int(horas), "mip_gap": 0.02, "microredes": micro, "time_limit": 1800,
                   "extras": extras}
        with st.expander("Solver y recursos"):
            c = st.columns(4)
            if not sup_est:
                sup["mip_gap"] = c[0].number_input("MIPGap", 0.001, 0.2, 0.02, format="%.3f", key="opt_gap")
            sup["time_limit"] = c[1].number_input("Límite por corrida [s]", 60, 86400,
                                                  int(sup["time_limit"]), step=60, key="opt_tl")
            threads = c[2].number_input("Hilos de Gurobi (0 = auto)", 0, 64, 0, key="opt_threads")
            mem = c[3].number_input("Memoria de Gurobi [GB] (0 = auto)", 0.0, 256.0, 0.0, key="opt_mem")
            c = st.columns(2)
            method = c[0].selectbox(
                "Algoritmo de la raíz (Method)", [None, 1, 2, 3, 0, -1], key="opt_method",
                format_func=lambda v: {None: "Por defecto (barrier)", -1: "Automático de Gurobi",
                                       0: "Simplex primal", 1: "Simplex dual (menos memoria)",
                                       2: "Barrier", 3: "Concurrente (más memoria)"}[v],
                help="Si Gurobi termina con 'Memory limit reached' sin explorar nodos, "
                     "probar simplex dual: barrier factoriza la matriz y eso pide mucha memoria.")
            presolve = c[1].selectbox(
                "Presolve", [None, 1, 0, -1, 2], key="opt_presolve",
                format_func=lambda v: {None: "Por defecto (agresivo)", -1: "Automático de Gurobi",
                                       0: "Apagado", 1: "Conservador (menos memoria)",
                                       2: "Agresivo"}[v])
            completo = st.checkbox("Guardar detalle horario completo (.pkl.gz, ~6 MB por corrida)", False,
                                   key="opt_completo",
                                   help="Solo lo usa el grafo por hora; se puede guardar después para un "
                                        "escenario desde Resultados.")
            criterio = st.selectbox("Criterio del escenario representativo (fase 2)",
                                    ["objetivo", "ens", "multi"], key="opt_criterio")

    try:
        meta = meta_e4(art_dir, sup["e4"])
    except (OSError, ValueError) as exc:
        st.error(f"No se pudo leer el E4 del caso seleccionado: {exc}")
        return
    error_e4 = incompatibilidad_e4(meta, sup)
    if error_e4:
        st.error(error_e4)
        st.info("Ajusta los supuestos al E4 guardado o elige otro E4 compatible antes de ejecutar.")
        return

    # ── 5 · escenarios ───────────────────────────────────────────────────
    var = variante_aislacion(bool(sup["seccionadores"]), sup["seccionadores"])
    with st.container(border=True):
        st.markdown("#### 5 · Escenarios")
        modos = (["Muestra del estudio"] if cont["escenarios"] else []) + \
            ["Por percentil de severidad del caso base", "Personalizada"]
        modo = st.radio("Cómo elegirlos", modos, horizontal=True, key="opt_modo_esc")
        n_pob = numesc(art_dir)
        disp = P.escenarios_disponibles(var, art_dir)
        completa = n_pob and len(disp) >= n_pob
        escenarios = []
        if modo == "Muestra del estudio":
            escenarios = cont["escenarios"]
            if completa:
                pct = A.percentil_en_poblacion(poblacion(var, disp, sup["horas"], art_dir), escenarios)
                st.dataframe(pct.set_index("escenario").T.style.format("{:,.1f}"), use_container_width=True)
                st.caption("Posición de cada escenario entre los casos base de la población (0 = el menos severo).")
            else:
                st.caption(f"Percentiles no disponibles: la variante `{var}` tiene {len(disp)}/{n_pob} casos base.")
        elif modo == "Personalizada":
            try:
                escenarios = indices(st.text_input("Escenarios (índices desde 0: 1,9,15 o 0-9)",
                                                   _lista(cont["escenarios"]) or "1", key="opt_esc_txt"))
            except ValueError as exc:
                st.error(str(exc))
                return
        else:
            c = st.columns(3)
            regla = c[0].radio("Regla", ["Decil más severo", "Centro de cada estrato"], key="opt_regla_muestra",
                               help="Decil más severo: los N escenarios con mayor severidad del caso base "
                                    "(regla de la tanda severa). Centro de cada estrato: uno por tramo de "
                                    "severidad, para estimar el promedio de la población.")
            n_estr = c[1].number_input("Escenarios en la muestra", 2, 50, 10, key="opt_n_estratos")
            crit = c[2].selectbox("Severidad", ["ens", "saidi"], key="opt_crit_estr",
                                  format_func=lambda x: {"ens": "ENS", "saidi": "SAIDI"}[x])
            if not completa:
                faltan = sorted(set(range(n_pob)) - set(disp))
                st.warning(f"Para ordenar por severidad hacen falta los casos base de toda la población. "
                           f"La variante `{var}` tiene {len(disp)} de {n_pob}.")
                panel_ejecucion(art_dir / "trabajos_v8" / f"E5_{var}",
                                [paso(f"Casos base E5 de la población ({var})", cmd_e5(sup["seccionadores"], faltan, art_dir),
                                      [P.base_dir(var, art_dir) / f"esc_{e:03d}.npz" for e in faltan], SEG_POR_E5, "e5")],
                                {}, key="opt_e5_pob")
            else:
                pob = poblacion(var, disp, sup["horas"], art_dir)
                severa = regla == "Decil más severo"
                muestra = (A.muestra_mas_severa(pob, int(n_estr), crit) if severa
                           else A.muestra_por_percentil(pob, int(n_estr), crit))
                escenarios = sorted(int(e) for e in muestra["escenario"])
                primera = "Rango" if severa else "Estrato"
                st.dataframe(muestra[[primera, "Percentil", "escenario", crit]].style.format(
                    {"Percentil": "{:.0f} %", crit: "{:,.1f}" if crit == "ens" else "{:.3f}"}),
                    hide_index=True, use_container_width=True)
                media_m, media_p = muestra[crit].mean(), np.mean([f[crit] for f in pob])
                if severa:
                    st.caption(f"Los {int(n_estr)} escenarios más severos de {len(pob)} (variante `{var}`). "
                               f"Media de la muestra {media_m:,.1f} contra {media_p:,.1f} de la población: "
                               "describe el peor tramo, no el promedio.")
                else:
                    st.caption(f"Centro de cada estrato (seleccionar_escenarios.estratificar). Media de la muestra "
                               f"{media_m:,.1f} contra {media_p:,.1f} de la población "
                               f"({100 * (media_m / media_p - 1):+.1f} %). Las colas quedan fuera a propósito.")
        if escenarios:
            st.caption(f"Escenarios del estudio (índices desde 0): **{_lista(escenarios)}**")

    # ── 6 · revisar y ejecutar ───────────────────────────────────────────
    if not medidas:
        st.warning("Habilita al menos una medida.")
        return
    if not escenarios:
        st.info("Elige los escenarios para ver qué hay que calcular.")
        return
    if n_pob and max(escenarios) >= n_pob:
        st.error(f"Hay escenarios fuera de la población ({n_pob}).")
        return
    with st.container(border=True):
        st.markdown("#### 6 · Revisar y ejecutar")
        opciones = {"threads": threads, "mem_gb": mem, "method": method, "presolve": presolve,
                    "completo": completo, "criterio": criterio,
                    "seg": segundos_promedio(str(root)) if root.exists() else SEG_POR_CORRIDA}
        if "decision" in productos:
            if len(medidas) < 2:
                st.warning("Las reglas de decisión evalúan los planes de una combinación: habilita dos o más medidas.")
                productos.discard("decision")
            else:
                combo_prev = "combo_" + "+".join(m for m in ORDEN if m in medidas)
                c = st.columns(2)
                opciones["recostear"] = c[0].checkbox("Recostear DG antes de evaluar planes (sin MILP)",
                                                      _recosteo_sugerido(root, combo_prev, sup),
                                                      key="opt_recostear", help=AYUDA_RECOSTEO)
                opciones["resolver_diagonal"] = c[1].checkbox("Re-resolver la diagonal", False,
                                                              key="opt_diag", help=AYUDA_DIAGONAL)
        pasos, combo = armar_pasos(root, sup, medidas, escenarios, productos, opciones)
        st.caption(("Caso principal: combinación `" + combo + "`") if combo
                   else f"Caso principal: medida `{medidas[0]}`")
        config = {"medidas": medidas, "escenarios": escenarios, "productos": sorted(productos),
                  "e4": sup["e4"], "seccionadores": sup["seccionadores"], "horas": sup["horas"],
                  "lim_dg": sup["lim_dg"], "lim_dg_modo": sup["lim_dg_modo"],
                  "extras": sup["extras"], "microredes": sup["microredes"]}
        panel_ejecucion(root, pasos, config, key="opt")
