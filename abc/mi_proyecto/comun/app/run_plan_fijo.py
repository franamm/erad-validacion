#!/usr/bin/env python
# ============================================================
# run_plan_fijo.py — matriz plan × escenario de una combinación del barrido
# ============================================================
# La fase 3 de run_barrido_medidas resuelve la combinación por escenario: el
# plan del escenario i es óptimo PARA i (información perfecta). Acá ese plan
# se fija y se evalúa en los demás escenarios j:
#
#   - inversión FIJA (capacidades DG/ESS/MEG, MESS, refuerzo, soterramiento,
#     poda, switches, enlaces nuevos) = la del plan i;
#   - operación LIBRE (despacho, ESS, EDRP, reconfiguración, móviles): es el
#     recurso con que ese plan enfrenta las fallas de j. Fijar también la
#     operación no tiene sentido, porque las fallas de j ocurren en otras
#     horas y otras líneas.
#
# Con la matriz M[i, j] se calculan criterios de selección del plan:
# valor esperado, peor caso (minimax), arrepentimiento (minimax regret) y el
# valor de la información perfecta.
#
# Resuelve con optimization_6_plan_fijo.py (copia del modelo oficial con el
# hook `plan_fijo`). Todo es reanudable: cada corrida se guarda apenas termina.
#
# Salida: artefactos/barrido/<barrido>/plan_fijo/<combinacion>/
#   plan_esc_XXX/esc_YYY.json|.npz   corrida del plan XXX en el escenario YYY
#   matriz_larga.csv                 una fila por (plan, escenario)
#   matriz_<indicador>.csv           filas = plan, columnas = escenario
#   criterios.csv                    un renglón por plan, con los criterios
#   resumen.json                     plan elegido por cada criterio
#
# USO
#   # ver qué se va a correr
#   python run_plan_fijo.py --barrido "sec-0-2-9-10 severo" --listar
#
#   # matriz completa 10 × 10 (la diagonal también se re-resuelve, para que
#   # toda la matriz use el mismo E4; --no-resolver-diagonal la toma del barrido)
#   python run_plan_fijo.py --barrido "sec-0-2-9-10 severo"
#
#   # solo algunos planes, o evaluar sobre escenarios que no están en el barrido
#   # (necesitan su baseline E5)
#   python run_plan_fijo.py --barrido "sec-0-2-9-10 severo" --planes 1,41 --escenarios 0-99
#
#   # rearmar matriz y criterios sin resolver nada
#   python run_plan_fijo.py --barrido "sec-0-2-9-10 severo" --solo-matriz
# ============================================================

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline as P
import run_barrido_medidas as RB
from aislacion import variante as _variante

INDICADORES_MATRIZ = ["objetivo", "ens", "saidi", "saifi", "costo_ens",
                      "cst_costo_total", "cst_ahorro_neto"]

COLS_LARGA = (["objetivo", "ens", "saidi", "saifi", "asai", "costo_ens"]
              + ["cst_capex_anual", "cst_opex_anual", "cst_costo_ens_estr",
                 "cst_ahorro_ens", "cst_costo_total", "cst_ahorro_neto",
                 "cst_objetivo_plan"]
              + ["cap_dg_sol_kW", "cap_dg_eol_kW", "cap_dg_fos_kW",
                 "cap_ess_4h_kWh", "cap_ess_5h_kWh", "cap_ess_6h_kWh",
                 "cap_meg_MW", "cap_mess_MWh", "n_ref", "n_ug", "n_veg",
                 "E_dg_sol", "E_dg_eol", "E_edrp",
                 "E_meg", "E_meg_arr", "n_arriendos_meg"]
              + ["solver_term", "solver_gap", "segundos"])


def _log(msg):
    print(f"[plan_fijo] {msg}", flush=True)


# ════════════════════════════════════════════════════════════
# Caché de la referencia sin fallas de la DG
# ════════════════════════════════════════════════════════════
# costos_de resuelve, por cada corrida con DG renovable, el caso sin fallas
# con la DG del plan (8760 h). Ese caso depende del plan y NO del escenario,
# así que en la matriz se repetiría idéntico en cada columna. Se envuelve acá
# para no tocar run_barrido_medidas.

_REF_CACHE = {}
_referencia_original = RB.referencia_dg_plan


def _referencia_cacheada(res, cfg, data):
    tecs = [t for t in ("sol", "eol") if cfg.get(f"dg_{t}")]
    h0 = res[0]
    clave = (len(res), cfg.get("lim_dg"), cfg.get("lim_dg_modo"), cfg.get("lim_dg_bus"),
             tuple((t, tuple(sorted((int(g), round(float(v), 4))
                                    for g, v in (h0.get(f"CAP_DG_{t}_kW") or {}).items())))
                   for t in tecs))
    if clave not in _REF_CACHE:
        _REF_CACHE[clave] = _referencia_original(res, cfg, data)
    return _REF_CACHE[clave]


RB.referencia_dg_plan = _referencia_cacheada


# ════════════════════════════════════════════════════════════
# Lectura del barrido
# ════════════════════════════════════════════════════════════

def elegir_combinacion(dir_comb, nombre):
    if nombre:
        d = dir_comb / nombre
        if not d.exists():
            raise SystemExit(f"No existe la combinacion '{nombre}' en {dir_comb}")
        return d
    # Por defecto, la de más medidas: es la cartera completa.
    cands = [d for d in dir_comb.iterdir() if d.is_dir() and any(d.glob("esc_*.json"))]
    if not cands:
        raise SystemExit(f"No hay combinaciones en {dir_comb}")
    return max(cands, key=lambda d: len(d.name.split("+")))


def cargar_planes(dir_c):
    """{escenario: registro .json} de las corridas con solución."""
    regs = {}
    for ruta in sorted(dir_c.glob("esc_*.json")):
        reg = json.loads(ruta.read_text(encoding="utf-8"))
        if reg.get("status") == "optimal":
            regs[int(reg["escenario"])] = reg
    return regs


def supuestos_de(regs):
    """Supuestos comunes a todas las corridas de la combinación."""
    claves = ("variante_aislacion", "e4_sin_fallas", "lim_dg", "lim_dg_modo",
              "lim_dg_bus", "horas")
    vals = {k: {json.dumps(r.get(k)) for r in regs.values()} for k in claves}
    distintos = {k: v for k, v in vals.items() if len(v) > 1}
    if distintos:
        raise SystemExit(f"Las corridas de la combinacion no comparten supuestos: {distintos}")
    return {k: json.loads(next(iter(v))) for k, v in vals.items()}


def config_plantilla(dir_c, regs):
    """Config con que se resolvió la combinación, leída del .pkl.gz.

    Es la fuente más fiel (trae n_meg, h_mess, T_tra, etc., que no viajan en
    el .json). El único dato que depende del escenario es `esc_debug`.
    """
    for e in sorted(regs):
        gz = dir_c / f"esc_{e:03d}.pkl.gz"
        if gz.exists():
            entrada = next(iter(P.leer_historial(gz).values()))
            return dict(entrada["config"])
    raise SystemExit("Ninguna corrida de la combinacion tiene .pkl.gz: no se puede "
                     "recuperar su config (re-correr la fase 3 con --completo).")


# ════════════════════════════════════════════════════════════
# Corridas
# ════════════════════════════════════════════════════════════

# Medidas de inversión que el modelo fija con `.fix()` después de construirlas. Con --liviano,
# la que el plan deja en cero se apaga en la config: no se crean sus variables ni restricciones
# (fijada en 0 o inexistente es lo mismo, pero Gurobi recibía ~38 M de filas y su presolve
# descartaba ~90 %). meg y meg_arriendo, reconfig_topo y edrp son operación y no se tocan.
MEDIDA_A_CLAVE_PLAN = {
    "dg_sol": "CAP_DG_sol_kW", "dg_eol": "CAP_DG_eol_kW", "dg_fos": "CAP_DG_fos_kW",
    "ess_4h": "CAP_ESS_kWh_4h", "ess_5h": "CAP_ESS_kWh_5h", "ess_6h": "CAP_ESS_kWh_6h",
    "inv_tie": "y_tie", "inv_switch_rcs": "y_rcs", "inv_switch_ms": "y_ms",
    "veg_mgmt": "x_veg", "refuerzo": "x_ref", "soterramiento": "x_ug", "mess": "y_mess",
}


def correr(args, dir_c, regs, planes, escenarios, salida):
    from optimization_6_plan_fijo import resolver_optimizacion

    sup = supuestos_de(regs)
    lim_bus = None if sup["lim_dg_bus"] in (None, "sin") else float(sup["lim_dg_bus"])
    e4 = "" if sup["e4_sin_fallas"] in (None, "historico") else sup["e4_sin_fallas"]

    tpl = config_plantilla(dir_c, regs)
    var = _variante(tpl.get("aislacion_fallas_fija", False),
                    tpl.get("seccionadores_existentes", []))
    if var != sup["variante_aislacion"]:
        raise SystemExit(f"Variante de la config ({var}) != la del .json "
                         f"({sup['variante_aislacion']})")
    medidas_tpl = sorted(m for m in RB.MEDIDAS if tpl.get(m))

    # Solver: se respetan los de la combinación salvo que se pidan otros.
    for clave, val in (("mip_gap", args.mip_gap), ("gurobi_time_limit", args.time_limit),
                       ("gurobi_threads", args.threads), ("gurobi_mem_gb", args.mem_gb),
                       ("gurobi_method", args.method), ("gurobi_presolve", args.presolve)):
        if val is not None:
            tpl[clave] = val

    propios = [i for i in planes if i not in getattr(args, "_ids_externos", set())]
    data, red = RB.construir_data(sorted(set(escenarios) | set(propios)), var, e4,
                                  sup["lim_dg"], sup["horas"],
                                  lim_dg_modo=sup["lim_dg_modo"], lim_dg_bus=lim_bus)
    if args.enlaces_nuevos:
        data, red = RB.agregar_enlaces_nuevos(
            data, red, RB.parse_enlaces_nuevos(args.enlaces_nuevos))
    firma = P.firma_modelo() if args.completo else None

    pendientes = [(i, j) for i in planes for j in escenarios
                  if not (i == j and j in regs and not args.resolver_diagonal)]
    _log(f"{len(pendientes)} corridas (planes {planes} x escenarios {escenarios})")

    for n, (i, j) in enumerate(pendientes, 1):
        reg_i = regs[i]
        if sorted(reg_i.get("medidas_on", [])) != medidas_tpl:
            # Un plan de OTRO barrido (--planes-de) puede traer inv_tie aunque la
            # plantilla no (o al revés). Vale mientras no haya enlaces nuevos
            # construidos que la red de la plantilla no tenga.
            extra = set(reg_i.get("medidas_on", [])) - set(medidas_tpl)
            tie_plan = {int(k) for k, v in (reg_i["plan"].get("y_tie") or {}).items() if v}
            if (i not in getattr(args, "_ids_externos", set())
                    or not extra <= {"inv_tie"}
                    or not tie_plan <= set(red["tie_indices"])):
                raise SystemExit(f"El plan {i} tiene otras medidas que la config plantilla")
        cfg = dict(tpl, esc_debug=j, plan_fijo=reg_i["plan"], _plan_origen=i)
        if args.liviano:
            apagadas = []
            for m, clave in MEDIDA_A_CLAVE_PLAN.items():
                if cfg.get(m) and not reg_i["plan"].get(clave):
                    cfg[m] = False
                    apagadas.append(m)
            if apagadas:
                _log(f"  liviano: sin construir {', '.join(apagadas)} (el plan {i} no las usa)")
        d = salida / f"plan_esc_{i:03d}"
        _log(f"[{n}/{len(pendientes)}] plan {i} -> escenario {j}")
        reg = RB.resolver_y_guardar(
            resolver_optimizacion, data, red, cfg, j,
            d / f"esc_{j:03d}.json", d / f"esc_{j:03d}.npz",
            etiqueta=f"plan_esc_{i:03d}", forzar=args.forzar,
            completo=args.completo, firma=firma)
        if reg and "plan_origen" not in reg:
            reg["plan_origen"] = i
            # Control: el plan resuelto tiene que ser el impuesto.
            reg["desvio_plan_max"] = desvio_plan(reg_i["plan"], reg.get("plan", {}))
            (d / f"esc_{j:03d}.json").write_text(
                json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
            if reg["desvio_plan_max"] > 1e-3:
                _log(f"  [!] el plan resuelto difiere del impuesto "
                     f"(desvio {reg['desvio_plan_max']:.4g})")


def desvio_plan(a, b):
    """Máxima diferencia absoluta entre dos `plan` ({clave: {id: valor}})."""
    m = 0.0
    for clave in set(a) | set(b):
        da, db = a.get(clave, {}), b.get(clave, {})
        for k in set(da) | set(db):
            m = max(m, abs(float(da.get(k, 0.0)) - float(db.get(k, 0.0))))
    return m


# ════════════════════════════════════════════════════════════
# Matriz y criterios
# ════════════════════════════════════════════════════════════

def armar_matriz(regs, planes, escenarios, salida, usar_diagonal_resuelta):
    filas = []
    for i in planes:
        for j in escenarios:
            ruta = salida / f"plan_esc_{i:03d}" / f"esc_{j:03d}.json"
            origen = "plan_fijo"
            reg = None
            if ruta.exists():
                reg = json.loads(ruta.read_text(encoding="utf-8"))
                if reg.get("status") != "optimal":
                    reg = None
            if i == j and j in regs and (reg is None or not usar_diagonal_resuelta):
                reg, origen = regs[j], "barrido"
            if reg is None:
                continue
            filas.append({"plan": i, "escenario": j, "diagonal": i == j, "origen": origen,
                          **{k: reg.get(k, np.nan) for k in COLS_LARGA}})
    return pd.DataFrame(filas)


def criterios(df, escenarios_completos):
    """Un renglón por plan. Solo usa escenarios evaluados para TODOS los planes,
    para que los promedios sean comparables entre planes."""
    sub = df[df["escenario"].isin(escenarios_completos)]
    M = sub.pivot(index="plan", columns="escenario", values="objetivo")
    # Óptimo con información perfecta de cada escenario: la diagonal.
    opt = pd.Series({j: M.loc[j, j] for j in M.columns if j in M.index})
    R = M[opt.index] - opt                   # arrepentimiento
    ind = {k: sub.pivot(index="plan", columns="escenario", values=k)
           for k in ("ens", "saidi", "saifi", "cst_capex_anual")}

    out = pd.DataFrame({
        "n_escenarios": M.notna().sum(axis=1),
        "E_objetivo": M.mean(axis=1),
        "std_objetivo": M.std(axis=1, ddof=0),
        "max_objetivo": M.max(axis=1),
        "E_regret": R.mean(axis=1),
        "max_regret": R.max(axis=1),
        "E_ens": ind["ens"].mean(axis=1),
        "max_ens": ind["ens"].max(axis=1),
        "prob_ens_pos": (ind["ens"] > 1e-6).mean(axis=1),
        "E_saidi": ind["saidi"].mean(axis=1),
        "max_saidi": ind["saidi"].max(axis=1),
        "E_saifi": ind["saifi"].mean(axis=1),
        "max_saifi": ind["saifi"].max(axis=1),
        "capex_anual": ind["cst_capex_anual"].mean(axis=1),
    })
    for c in ("E_objetivo", "max_objetivo", "E_regret", "max_regret", "E_ens", "E_saidi"):
        out[f"rank_{c}"] = out[c].rank(method="min").astype(int)

    # Valor esperado de la información perfecta (dentro de la muestra):
    # cuánto más cuesta decidir UN plan para todos que decidir escenario a escenario.
    wait_and_see = float(opt.mean()) if len(opt) else np.nan
    mejor_E = float(out["E_objetivo"].min())
    resumen = {
        "escenarios_usados": [int(e) for e in escenarios_completos],
        "ponderacion": "equiprobable",
        "elegido": {
            "valor_esperado": int(out["E_objetivo"].idxmin()),
            "minimax_objetivo": int(out["max_objetivo"].idxmin()),
            "minimax_regret": int(out["max_regret"].idxmin()),
            "min_E_ens": int(out["E_ens"].idxmin()),
            "min_E_saidi": int(out["E_saidi"].idxmin()),
        },
        "wait_and_see_E_objetivo": wait_and_see,
        "mejor_E_objetivo": mejor_E,
        "EVPI": mejor_E - wait_and_see,
    }
    return out, R, resumen


def escribir_resultados(df, regs, planes, escenarios, salida):
    df.to_csv(salida / "matriz_larga.csv", index=False, encoding="utf-8-sig")
    for k in INDICADORES_MATRIZ:
        if k in df:
            df.pivot(index="plan", columns="escenario", values=k).to_csv(
                salida / f"matriz_{k}.csv", encoding="utf-8-sig")

    # Escenarios con la columna llena para todos los planes, y con diagonal.
    completos = [j for j in escenarios
                 if j in planes and df[df["escenario"] == j]["plan"].nunique() == len(planes)]
    faltan = len(planes) * len(escenarios) - len(df)
    if faltan:
        _log(f"[!] faltan {faltan} celdas de la matriz; los criterios usan solo "
             f"los escenarios completos {completos}")
    if len(completos) < 2 or len(planes) < 2:
        _log("no hay columnas completas suficientes para calcular criterios")
        return

    crit, R, resumen = criterios(df, completos)
    crit.to_csv(salida / "criterios.csv", encoding="utf-8-sig")
    R.to_csv(salida / "matriz_regret.csv", encoding="utf-8-sig")
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")

    gaps = df["solver_gap"].astype(float)
    print()
    print("=" * 100)
    print("  CRITERIOS DE SELECCION DEL PLAN (inversion fija, operacion libre)")
    print("=" * 100)
    print(f"  {'plan':>5} {'E[obj]':>12} {'max obj':>12} {'E[regret]':>11} "
          f"{'max regret':>11} {'E[ENS]':>10} {'E[SAIDI]':>9} {'P(ENS>0)':>9}")
    for p, f in crit.sort_values("E_objetivo").iterrows():
        print(f"  {p:>5} {f['E_objetivo']:>12,.0f} {f['max_objetivo']:>12,.0f} "
              f"{f['E_regret']:>11,.0f} {f['max_regret']:>11,.0f} {f['E_ens']:>10,.1f} "
              f"{f['E_saidi']:>9.4f} {f['prob_ens_pos']:>9.0%}")
    print("-" * 100)
    for k, v in resumen["elegido"].items():
        print(f"  {k:<18}: plan del escenario {v}")
    print(f"  EVPI (muestra)    : {resumen['EVPI']:,.0f}  "
          f"(E[obj] mejor plan {resumen['mejor_E_objetivo']:,.0f} vs "
          f"informacion perfecta {resumen['wait_and_see_E_objetivo']:,.0f})")
    print(f"  MIPGap max        : {gaps.max():.2%}  -> diferencias menores que eso "
          f"son ruido del solver (el regret puede salir levemente negativo)")
    print("=" * 100)


# ════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Matriz plan x escenario con inversion fija.")
    ap.add_argument("--barrido", required=True,
                    help="carpeta dentro de artefactos/barrido/, ej: 'sec-0-2-9-10 severo'")
    ap.add_argument("--combinacion", default="",
                    help="subcarpeta de combinadas/ (default: la de mas medidas)")
    ap.add_argument("--planes", default="",
                    help="escenarios cuyo plan se fija (default: todos los de la combinacion)")
    ap.add_argument("--escenarios", default="",
                    help="escenarios donde se evalua (default: los mismos de --planes). "
                         "Puede incluir escenarios fuera del barrido si tienen baseline E5.")
    ap.add_argument("--resolver-diagonal", dest="resolver_diagonal",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="re-resolver tambien i == j y usarlo en la matriz (default). "
                         "Asi toda la matriz sale con el mismo E4 y el mismo MIPGap: el "
                         "`objetivo` del barrido lleva la constante del E4 con que se "
                         "resolvio, y mezclarlo con celdas nuevas sesga el regret. "
                         "--no-resolver-diagonal reutiliza la del barrido.")
    ap.add_argument("--solo-matriz", dest="solo_matriz", action="store_true",
                    help="no resuelve: rearma matriz y criterios con lo que haya en disco")
    ap.add_argument("--completo", action=argparse.BooleanOptionalAction, default=False,
                    help="guardar ademas el .pkl.gz (~7.6 MB por corrida; default: no)")
    ap.add_argument("--forzar", action="store_true", help="rehacer corridas ya guardadas")
    ap.add_argument("--liviano", action="store_true",
                    help="no construir las medidas que el plan impuesto deja en cero "
                         "(modelo mucho más chico; mismo resultado, ver MEDIDA_A_CLAVE_PLAN)")
    ap.add_argument("--listar", action="store_true", help="mostrar el plan y salir")
    ap.add_argument("--mip_gap", type=float, default=None)
    ap.add_argument("--time_limit", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--mem_gb", type=float, default=None)
    ap.add_argument("--method", type=int, default=None, choices=[-1, 0, 1, 2, 3])
    ap.add_argument("--presolve", type=int, default=None, choices=[-1, 0, 1, 2])
    ap.add_argument("--artefactos", default=None, help="Carpeta del caso de estudio")
    ap.add_argument("--planes-de", dest="planes_de", default="",
                    help="carpeta combinadas/<combinacion> de OTRO barrido (otro evento): "
                         "sus planes se evalúan también en los escenarios de este caso. "
                         "Se numeran con --offset-planes-de + su escenario.")
    ap.add_argument("--salida-sufijo", dest="salida_sufijo", default="",
                    help="sufijo de la carpeta plan_fijo/<combinacion>: con --planes-de "
                         "evita pisar la matriz propia del barrido")
    ap.add_argument("--salida-nombre", dest="salida_nombre", default="",
                    help="nombre corto de la carpeta plan_fijo/<nombre> (reemplaza al de la "
                         "combinación: en Windows las rutas largas superan los 260 caracteres)")
    ap.add_argument("--offset-planes-de", dest="offset_planes_de", type=int, default=100)
    ap.add_argument("--enlaces-nuevos", dest="enlaces_nuevos", default="",
                    help="igual que en run_barrido_medidas: los candidatos 'a-b:km' de la "
                         "corrida que generó los planes (necesario si este caso los usó)")
    args = ap.parse_args()
    if args.artefactos:
        P.usar_artefactos(args.artefactos)
        RB.ART_DIR = P.ART_DIR

    base = P.ART_DIR / "barrido" / args.barrido
    dir_c = elegir_combinacion(base / "combinadas", args.combinacion)
    regs = cargar_planes(dir_c)
    propios = sorted(regs)
    args._ids_externos = set()
    if args.planes_de:
        ext = {k + args.offset_planes_de: v
               for k, v in cargar_planes(Path(args.planes_de)).items()}
        if set(ext) & set(regs):
            raise SystemExit("--offset-planes-de choca con los planes propios")
        regs.update(ext)
        args._ids_externos = set(ext)
    planes = RB.parse_lista_int(args.planes) if args.planes else sorted(regs)
    escenarios = RB.parse_lista_int(args.escenarios) if args.escenarios else list(propios)
    sin_plan = [i for i in planes if i not in regs]
    if sin_plan:
        raise SystemExit(f"Sin corrida de la combinacion para los planes {sin_plan}")
    salida = base / "plan_fijo" / (args.salida_nombre or dir_c.name + args.salida_sufijo)

    print("=" * 100)
    print("  MATRIZ PLAN x ESCENARIO (inversion fija, operacion libre)")
    print("=" * 100)
    print(f"  combinacion : {dir_c.name}")
    print(f"  planes      : {planes}")
    print(f"  escenarios  : {escenarios}")
    print(f"  salida      : {salida.resolve()}")
    for i in planes:
        caps = {k: round(sum(v.values())) for k, v in regs[i]["plan"].items()}
        print(f"    plan {i:>3}: {caps}")
    print("=" * 100)
    if args.listar:
        return

    salida.mkdir(parents=True, exist_ok=True)
    if not args.solo_matriz:
        correr(args, dir_c, regs, planes, escenarios, salida)

    df = armar_matriz(regs, planes, escenarios, salida, args.resolver_diagonal)
    if df.empty:
        _log("matriz vacia")
        return
    escribir_resultados(df, regs, planes, escenarios, salida)
    _log(f"listo. Resultados en {salida}")


if __name__ == "__main__":
    main()
