"""Reúne, para un estudio, todos los números del reporte (sin Streamlit, sin optimizar).

Todo sale de los artefactos del caso activo y de las mismas funciones que usan las
pestañas de Resultados (`resultados.preparar`, `leer_decision`, `matriz_decision`,
`analisis.reglas`, `analisis.rangos_plan`, `graficos_valor_esperado.promediar_ens`),
así el reporte no puede contradecir a la app.

Configuración opcional por estudio: `<estudio>/reporte.json`
    {"nombre_red": "...", "individuales": ["barrido/...", ...],
     "grupos": {"Evento 1": [53, 54], "Evento 2": [91, 92]}}
`individuales` lista los estudios con corridas de medidas sueltas (en orden de
prioridad si una medida aparece en más de uno); por defecto, el propio estudio.
"""
import json
import pickle
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from estudio_v8 import analisis as A
from estudio_v8.core import ART_DIR, leer_json

INDICADOR_DECISION = "cst_objetivo_plan"
REGLAS_REPORTE = ("Valor esperado", "Minimax", "Minimax regret")
NOMBRE_INDICADOR = {"cst_objetivo_plan": "costo del plan (medidas + ENS)", "ens": "ENS",
                    "cst_costo_total": "costo anual de las medidas"}


def _cfg(root):
    ruta = root / "reporte.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def _git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, timeout=10).stdout.strip() or "—"
    except Exception:
        return "—"


def _carga(nombre):
    with open(ART_DIR / nombre, "rb") as f:
        return pickle.load(f)


def _json_caso(nombre):
    ruta = ART_DIR / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


# ── Red, amenaza y supuestos ────────────────────────────────────────────────

def datos_red():
    red = _carga("red.pkl")
    net = red["net"]
    ties = {int(l) for l in red.get("tie_indices", [])}
    lineas = [int(l) for l in net.line.index if int(l) not in ties]
    largo = {int(k): float(v) for k, v in red.get("longitud_km_linea", {}).items()}
    zona_bus = {int(b): int(z) for b, z in red["bus_zone"].items()}
    demanda = {int(b): float(v) for b, v in red.get("P_D_base", {}).items()}
    clientes = {int(b): float(v) for b, v in (red.get("clientes_bus") or {}).items()}
    zonas = []
    for z in sorted(set(zona_bus.values())):
        bs = [b for b, zz in zona_bus.items() if zz == z]
        ls = [l for l in lineas if zona_bus.get(int(net.line.at[l, "from_bus"])) == z]
        zonas.append({"Zona": f"ZG{z}", "Barras": len(bs), "Líneas": len(ls),
                      "km": sum(largo.get(l, 0.0) for l in ls),
                      "Demanda base [kW]": sum(demanda.get(b, 0.0) for b in bs),
                      "Clientes": sum(clientes.get(b, 0.0) for b in bs)})
    ram = red.get("ramales_fusible") or {}
    return {"barras": len(red["BUSES"]), "lineas": len(lineas), "km": sum(largo.get(l, 0.0) for l in lineas),
            "enlaces": len(ties), "seccionadores": len(red.get("seccionadores_existentes") or []),
            "subestaciones": len(red.get("S") or []), "ramales": len(ram.get("id", [])) if isinstance(ram, dict) else 0,
            "demanda_kW": sum(demanda.values()), "clientes": sum(clientes.values()),
            "zonas": pd.DataFrame(zonas), "zona_bus": zona_bus,
            "zona_linea": {int(l): zona_bus.get(int(net.line.at[l, "from_bus"])) for l in net.line.index},
            "largo": largo, "nombre_linea": net.line["name"].to_dict()}


def datos_amenaza():
    fm, fl = _json_caso("fallas_meta.json"), _json_caso("faultlin_meta.json")
    forma = fl.get("shape") or [None, None, None]
    return {"horas_reparacion": fm.get("horas_reparacion") or fl.get("horas_espera"),
            "fallas_poblacion": (fm.get("fallas") or fl.get("fallas_totales") or {}).get("base"),
            "horas": forma[0], "escenarios_poblacion": forma[1], "curva": fl.get("curva"),
            "escenario_clima": fl.get("escenario_clima"), "rho_ug": fm.get("rho_ug"),
            "ratio_ref": fm.get("ratio_clase3_clase5")}


def datos_costos():
    c = _carga("costos.pkl")
    caso = _json_caso("costos_caso.json")

    def v(x):
        return float(next(iter(x.values()))) if isinstance(x, dict) else float(x)

    filas = []

    def fila(medida, capex, u_capex, opex, u_opex, nota=""):
        filas.append({"Medida": medida, "CAPEX": capex, "Unidad CAPEX": u_capex,
                      "OPEX": opex, "Unidad OPEX": u_opex, "Nota": nota})

    for k, nom in (("DG_sol", "GD solar"), ("DG_eol", "GD eólica"), ("DG_fos", "GD fósil")):
        if k in c:
            fila(nom, v(c[k][0]), "USD/kW", v(c[k][1]), "USD/kW·año",
                 "más combustible" if k == "DG_fos" else "")
    for h in (4, 5, 6):
        k = f"ESS {h}h"
        if k in c:
            fila(f"Almacenamiento {h} h", v(c[k][0]), "USD/kW", v(c[k][1]), "USD/kW·año")
    if "MEG" in c:
        fila("Generador móvil (MEG)", float(c["MEG"][0]), "USD/kW", float(c["MEG"][2]), "USD/m³ diésel",
             f"traslado {float(c['MEG'][1]):,.0f} USD/h".replace(",", "."))
    if "MEG_arriendo" in c:
        fila("Arriendo de MEG", 0.0, "—", v(c["MEG_arriendo"][0]), "USD/evento", "incluye combustible")
    if "REF" in c:
        fila("Refuerzo estructural", float(c["REF"][0]), "USD/km", float(c["REF"][1]), "USD/km·año")
    if "UG" in c:
        fila("Soterramiento", float(c["UG"][0]), "USD/km", float(c["UG"][1]), "USD/km·año")
    if "SWITCH" in c:
        fila("Reconfiguración (enlaces existentes)", 0.0, "—", float(c["SWITCH"][1]), "USD/maniobra")
    if "EDRP_incentivo" in c:
        inc = [float(x) for x in c["EDRP_incentivo"]]
        fila("Respuesta de demanda (EDRP)", 0.0, "—", min(inc), "USD/kWh",
             f"{min(inc):.0f} a {max(inc):.0f} USD/kWh por bloque")
    return {"tabla": pd.DataFrame(filas), "voll": v(c.get("VoLL", 5)), "tasa": float(c.get("r", 0.1)),
            "anios": int(c.get("años", 30)), "frp": float(c.get("frp", np.nan)),
            "c_sub": v(c.get("C_sub", np.nan)), "fuente": caso.get("fuente", ""),
            "tipo_cambio": caso.get("tipo_cambio_clp_usd")}


# ── Medidas individuales ────────────────────────────────────────────────────

NOMBRE_MEDIDA = {**A.NOMBRE_MEDIDA, "dg_sol": "GD solar", "dg_eol": "GD eólica", "dg_fos": "GD fósil",
                 "meg_arriendo": "MEG con arriendo", "ess_4h": "Almacenamiento 4 h",
                 "ess_5h": "Almacenamiento 5 h", "ess_6h": "Almacenamiento 6 h", "mess": "Almacenamiento móvil (MESS)",
                 "meg": "Generador móvil (MEG)", "edrp": "Respuesta de demanda (EDRP)"}


def datos_individuales(estudios, base, mip_gap, e4):
    """Medidas sueltas, comparadas contra el caso base del estudio principal.

    Por medida se toma el estudio que la trae en más escenarios (empate: orden de
    `estudios`). Todas se re-enriquecen contra `base` (misma variante y E4 que los
    planes), como hace el notebook del caso; así una tanda sin variante registrada
    no queda sin mejoras. Entran los escenarios de la medida con más cobertura; las
    medidas que no los cubren todos se informan aparte, no se promedian parciales."""
    import barrido_vista as BV
    import graficos_valor_esperado as GVE
    from estudio_v8 import resultados as R
    mejor = {}
    for k, nombre in enumerate(estudios):
        if not (ART_DIR / nombre / "resumen_individual.csv").exists():
            continue
        crudo = BV.cargar_barrido(nombre, BV.sello(nombre))["individual"]
        for m, g in crudo.groupby("medida"):
            clave = (g["escenario"].nunique(), -k)
            if m not in mejor or clave > mejor[m][0]:
                mejor[m] = (clave, nombre, g)
    if not mejor:
        return None
    crudo = pd.concat([g for _, _, g in mejor.values()], ignore_index=True)
    dfe = BV._cap_resiliencia(BV.enriquecer(crudo, base.reset_index(), mip_gap), BV.capacidad_e4(e4))
    dfe = R._validas(dfe)
    cobertura = {m: set(g["escenario"].astype(int)) for m, g in dfe.groupby("medida")}
    escs = sorted(max(cobertura.values(), key=len))
    incompletas = sorted(m for m, s in cobertura.items() if not set(escs) <= s)
    dfe = dfe[dfe["escenario"].isin(escs) & ~dfe["medida"].isin(incompletas)].copy()
    dfe["_tipo"], dfe["_caso"] = "Individual", dfe["medida"]
    prom = GVE.resumir_indicadores(dfe)
    prom["Medida"] = prom["_caso"].map(lambda m: NOMBRE_MEDIDA.get(m, m))
    return {"dfe": dfe, "prom": prom, "escenarios": escs, "incompletas": incompletas,
            "fuentes": {m: n for m, (_, n, _) in mejor.items()}}


# ── Planes y reglas de decisión ─────────────────────────────────────────────

def datos_decision(root, combo, base, etiqueta):
    from estudio_v8 import resultados as R
    diagonal, evaluaciones = R.leer_decision(root, combo)
    if not evaluaciones:
        return None
    escenarios = sorted({j for _, j in evaluaciones} | set(diagonal))
    planes = [e for e in escenarios if e in diagonal]
    larga = R.matriz_decision(diagonal, evaluaciones, planes, escenarios, usar_resuelta=True)
    cols = ["cst_objetivo_plan", "objetivo", "ens", "saidi", "saifi", "cst_costo_total",
            "cst_capex_anual", "cst_opex_anual", "costo_ens", "cst_ahorro_neto"]
    larga, excluidas = R.excluir_desviadas(larga, cols)
    M = larga.pivot(index="plan", columns="escenario", values=INDICADOR_DECISION).reindex(index=planes,
                                                                                            columns=escenarios)
    r = A.reglas(M, referencia="diagonal")
    elegidos = {k: v for k, v in r["elegidos"].items() if k in REGLAS_REPORTE}
    votos = {p: sum(p in v for v in elegidos.values()) for p in planes}
    crit = r["criterios"]
    recomendado = max(planes, key=lambda p: (votos[p], -crit.loc[p, "Valor esperado"]))
    medias = larga.groupby("plan")[[c for c in cols if c in larga]].mean()
    alternativa = medias["ens"].idxmin() if "ens" in medias else None
    if alternativa == recomendado:
        alternativa = None
    return {"diagonal": diagonal, "evaluaciones": evaluaciones, "larga": larga, "planes": planes,
            "escenarios": escenarios, "M": M, "reglas": r, "elegidos": elegidos, "votos": votos,
            "recomendado": recomendado, "alternativa": alternativa, "medias": medias,
            "excluidas": excluidas, "etiqueta": etiqueta}


def resumen_inversion(reg, red):
    """Qué instala un plan, agrupado por medida (kW, km, unidades) y por elemento."""
    p = reg.get("plan") or {}
    out = {}

    def barras(clave):
        return {int(k): float(x) for k, x in (p.get(clave) or {}).items() if float(x) > 1e-3}

    for clave, nom in (("CAP_DG_sol_kW", "GD solar"), ("CAP_DG_eol_kW", "GD eólica"), ("CAP_DG_fos_kW", "GD fósil")):
        b = barras(clave)
        if b:
            out[nom] = {"total": sum(b.values()), "unidad": "kW", "n": len(b),
                        "elementos": sorted(b.items(), key=lambda t: -t[1])}
    for clave, nom in (("x_ref", "Refuerzo estructural"), ("x_ug", "Soterramiento")):
        ls = [int(k) for k, x in (p.get(clave) or {}).items() if float(x) > 0.5]
        if ls:
            out[nom] = {"total": sum(red["largo"].get(l, 0.0) for l in ls), "unidad": "km", "n": len(ls),
                        "elementos": sorted(((l, red["largo"].get(l, 0.0)) for l in ls), key=lambda t: -t[1])}
    meg = {k: float(x) for k, x in (p.get("CAP_MEG_MW") or {}).items() if float(x) > 1e-6}
    if meg:
        exist = int(reg.get("meg_existentes") or 0)
        nuevos = {k: x for k, x in meg.items() if not (str(k).isdigit() and int(k) < exist)}
        out["Generador móvil (MEG)"] = {"total": sum(meg.values()), "unidad": "kW", "n": len(meg),
                                       "existentes": len(meg) - len(nuevos), "nuevos": len(nuevos),
                                       "elementos": sorted(meg.items())}
    return out


def operacion_plan(regs):
    """Uso operativo de un plan en sus evaluaciones: enlaces, MEG, arriendos, EDRP."""
    enl, meg = {}, {}
    for r in regs:
        u = r.get("ubicacion") or {}
        for k, h in (u.get("tie_horas_cerrada") or {}).items():
            if float(h) > 0:
                enl[int(k)] = enl.get(int(k), 0) + 1
        for k, h in (u.get("meg_horas_por_bus") or {}).items():
            if float(h) > 0:
                meg[int(k)] = meg.get(int(k), 0) + 1

    def rng(k, f=1.0):
        xs = [float(r[k]) / f for r in regs if r.get(k) is not None]
        return (min(xs), max(xs)) if xs else None

    return {"enlaces": dict(sorted(enl.items(), key=lambda t: -t[1])), "meg_barras": dict(sorted(meg.items(), key=lambda t: -t[1])),
            "maniobras": rng("maniobras_tie"), "arriendos": rng("n_arriendos_meg"),
            "edrp_MWh": rng("E_edrp", 1000.0), "E_meg_MWh": rng("E_meg", 1000.0), "n": len(regs)}


# ── Caso base y ENS ─────────────────────────────────────────────────────────

def ens_promedio(rutas, rutas_base, red_pkl, horas):
    import graficos_valor_esperado as GVE
    if any(not Path(p).exists() for p in list(rutas_base) + list(rutas or [])):
        return None, None
    return GVE.promediar_ens(rutas, rutas_base, red_pkl["BUSES"], red_pkl["bus_zone"], horas)


def fallas_por_escenario(rutas_base):
    """Fallas (inicios de daño) por escenario y por línea, desde los npz del caso base."""
    por_esc, por_linea = {}, {}
    for e, ruta in rutas_base.items():
        with np.load(ruta) as z:
            if "dano_l" not in z:
                continue
            d = z["dano_l"] > 0
        ini = d & ~np.vstack([np.zeros((1, d.shape[1]), bool), d[:-1]])
        por_esc[e] = int(ini.sum())
        for l, c in enumerate(ini.sum(0)):
            if c:
                por_linea[l] = por_linea.get(l, 0.0) + c / len(rutas_base)
    return por_esc, por_linea


# ── Todo junto ──────────────────────────────────────────────────────────────

def reunir(estudio, combo="todas", plan=None):
    import pipeline as P
    from estudio_v8 import resultados as R
    root = ART_DIR / estudio
    cfg = _cfg(root)
    d = R.preparar(estudio)
    B = d["B"]
    red = datos_red()
    red_pkl = _carga("red.pkl")
    base = d["base"].copy()
    base["escenario"] = base["escenario"].astype(int)
    escs = sorted(int(e) for e in base["escenario"])
    etiqueta = {e: f"P{k}" for k, e in enumerate(escs, 1)}
    num_esc = {e: k for k, e in enumerate(escs, 1)}
    grupos_cfg = cfg.get("grupos") or {}
    grupo_de = {int(e): i for i, (_, es) in enumerate(grupos_cfg.items()) for e in es}

    rutas_base = {e: P.base_dir(B["variante"]) / f"esc_{e:03d}.npz" for e in escs}
    h_base, z_base = ens_promedio(None, [rutas_base[e] for e in escs], red_pkl, d["horas"])
    fallas_esc, fallas_linea = fallas_por_escenario({e: r for e, r in rutas_base.items() if r.exists()})

    dec = datos_decision(root, combo, base, etiqueta)
    sup = {}
    regs_combo = {int(p.stem.split("_")[1]): leer_json(p) for p in (root / "combinadas" / combo).glob("esc_*.json")}
    if regs_combo:
        r0 = next(iter(regs_combo.values()))
        sup = {k: r0.get(k) for k in ("horas", "lim_dg", "lim_dg_modo", "e4_sin_fallas", "variante_aislacion",
                                      "meg_existentes", "n_meg", "P_meg_max", "t_tra_meg", "medidas_on",
                                      "t_maniobra_tie", "radial_tie")}

    ind = datos_individuales(cfg.get("individuales") or [estudio], d["base"], B["mip_gap"], B["e4"])

    combinacion = None
    if ind is not None and not d["dce"].empty:
        dce = R._validas(d["dce"])
        # Combinaciones con nombre libre ("todas"): sus miembros salen de `medidas_on`.
        alias = combo
        if not str(combo).startswith("combo_") and sup.get("medidas_on"):
            alias = "combo_" + "+".join(sup["medidas_on"])
            dce = dce.assign(combinacion=dce["combinacion"].replace({combo: alias}))
        comunes = sorted(set(ind["escenarios"]) & set(dce.loc[dce["combinacion"] == alias, "escenario"]))
        if comunes:
            combinacion = {"escenarios": comunes, "miembros": A.miembros_de(alias),
                           "solapamiento": {k: A.solapamiento_activas(ind["dfe"], dce, alias, comunes, k)
                                            for k in ("ens", "saidi", "saifi")},
                           "activas": A.tabla_activas(dce, [alias], escs)}
    recomendado = None
    if dec is not None:
        p = plan if plan is not None else dec["recomendado"]
        if isinstance(p, str) and p.upper().startswith("P"):
            p = {v: k for k, v in etiqueta.items()}[p.upper()]
        p = int(p)
        regs_p = [dec["evaluaciones"].get((p, j)) or (dec["diagonal"].get(p) if j == p else None) for j in dec["escenarios"]]
        regs_p = [r for r in regs_p if r]
        rutas_p = []
        for j in dec["escenarios"]:
            ruta = root / "plan_fijo" / combo / f"plan_esc_{p:03d}" / f"esc_{j:03d}.npz"
            if not ruta.exists() and j == p:
                ruta = root / "combinadas" / combo / f"esc_{p:03d}.npz"
            rutas_p.append(ruta)
        h_p, z_p = ens_promedio(rutas_p, [rutas_base[j] for j in dec["escenarios"]], red_pkl, d["horas"])
        recomendado = {"plan": p, "etiqueta": etiqueta[p], "inversion": resumen_inversion(dec["diagonal"][p], red),
                       "operacion": operacion_plan(regs_p), "horario": h_p, "zonal": z_p,
                       "por_escenario": dec["larga"][dec["larga"]["plan"] == p].set_index("escenario"),
                       "forzado": plan is not None}
        if dec["alternativa"] is not None:
            a = dec["alternativa"]
            recomendado["alternativa"] = {"plan": a, "etiqueta": etiqueta[a],
                                          "inversion": resumen_inversion(dec["diagonal"][a], red)}
        dec["inversiones"] = {q: resumen_inversion(dec["diagonal"][q], red) for q in dec["planes"]}

    return {"meta": {"caso": ART_DIR.name, "ruta_caso": str(ART_DIR), "estudio": estudio, "combinacion": combo,
                     "nombre_red": cfg.get("nombre_red") or ART_DIR.name,
                     "fecha": datetime.now().strftime("%d-%m-%Y %H:%M"), "git": _git_hash(),
                     "variante": B["variante"], "e4": B["e4"], "horas": d["horas"]},
            "red": red, "amenaza": datos_amenaza(), "costos": datos_costos(), "supuestos": sup,
            "escenarios": escs, "num_esc": num_esc, "etiqueta": etiqueta,
            "grupos": grupos_cfg, "grupo_de": grupo_de,
            "base": base.set_index("escenario").select_dtypes("number"), "base_horario": h_base, "base_zonal": z_base,
            "fallas_esc": fallas_esc, "fallas_linea": fallas_linea,
            "individuales": ind, "combinacion": combinacion,
            "decision": dec, "recomendado": recomendado}
