"""Inventario de un estudio desde disco: qué corridas hay, cuáles faltan y con qué supuestos.

Sin Streamlit. Solo lee los `.json` de cada corrida (pocos KB), nunca los
`.npz` ni los `.pkl.gz`.
"""
from pathlib import Path

import pandas as pd

from estudio_v8.core import leer_json

SIMBOLO = {"optima": "✅", "sospechosa": "⚠️", "sin_solucion": "❌", "falta": "·"}
SUPUESTOS = ("variante_aislacion", "e4_sin_fallas", "lim_dg", "lim_dg_modo",
             "lim_dg_bus", "horas", "mip_gap", "microredes")


def estado_corrida(reg, ens_base=None):
    """optima | sospechosa | sin_solucion | falta, con los criterios de `barrido_vista.enriquecer`."""
    if not reg:
        return "falta"
    if reg.get("status") != "optimal":
        return "sin_solucion"
    gap_pedido = float(reg.get("mip_gap") or 0.02)
    term, gap = reg.get("solver_term"), reg.get("solver_gap")
    if term not in (None, "optimal", "feasible") and (gap is None or float(gap) > 5 * gap_pedido):
        return "sospechosa"
    if ens_base is not None and reg.get("ens") is not None \
            and float(reg["ens"]) > float(ens_base) * (1 + gap_pedido):
        return "sospechosa"
    return "optima"


def rutas_esperadas(root, medidas=(), escenarios=(), combos=(), planes=(), escenarios_plan=()):
    """[(tipo, caso, escenario, ruta .json)] en el orden en que corren los scripts."""
    root = Path(root)
    out = [("individual", m, e, root / "individual" / m / f"esc_{e:03d}.json")
           for m in medidas for e in escenarios]
    out += [("combinadas", c, e, root / "combinadas" / c / f"esc_{e:03d}.json")
            for c in combos for e in escenarios]
    for c in combos if planes else ():
        for i in planes:
            for j in escenarios_plan:
                out.append(("plan_fijo", f"{c}/plan_esc_{i:03d}", j,
                            root / "plan_fijo" / c / f"plan_esc_{i:03d}" / f"esc_{j:03d}.json"))
    return out


def inventario(esperadas, ens_base=None):
    """Una fila por corrida esperada, con su estado. `ens_base`: {escenario: ENS del caso base}."""
    filas = []
    for tipo, caso, e, ruta in esperadas:
        reg = leer_json(ruta) if Path(ruta).exists() else {}
        filas.append({"tipo": tipo, "caso": caso, "escenario": int(e),
                      "estado": estado_corrida(reg, (ens_base or {}).get(int(e))),
                      "segundos": reg.get("segundos"), "solver_gap": reg.get("solver_gap"),
                      "ruta": str(ruta)})
    return pd.DataFrame(filas, columns=["tipo", "caso", "escenario", "estado",
                                        "segundos", "solver_gap", "ruta"])


def resumen(inv):
    cuenta = inv["estado"].value_counts().to_dict() if len(inv) else {}
    total = len(inv)
    return {"total": total, "guardadas": total - cuenta.get("falta", 0),
            "optimas": cuenta.get("optima", 0), "sospechosas": cuenta.get("sospechosa", 0),
            "sin_solucion": cuenta.get("sin_solucion", 0), "pendientes": cuenta.get("falta", 0)}


def matriz_cobertura(inv, nombre=str):
    """Caso × escenario con los símbolos de estado, para mostrar de un vistazo."""
    if inv.empty:
        return pd.DataFrame()
    m = inv.assign(simbolo=inv["estado"].map(SIMBOLO),
                   Caso=[nombre(t, c) for t, c in zip(inv["tipo"], inv["caso"])])
    return m.pivot_table(index="Caso", columns="escenario", values="simbolo",
                         aggfunc="first", sort=False)


def registros_estudio(root):
    """Todos los `.json` de corridas del estudio: [(tipo, caso, escenario, registro)]."""
    root = Path(root)
    out = []
    patrones = (("individual", "individual/*/esc_*.json"), ("combinadas", "combinadas/*/esc_*.json"),
                ("plan_fijo", "plan_fijo/*/plan_esc_*/esc_*.json"))
    for tipo, patron in patrones:
        for ruta in sorted(root.glob(patron)):
            caso = ruta.parent.relative_to(root / tipo).as_posix()
            out.append((tipo, caso, int(ruta.stem.split("_")[1]), leer_json(ruta)))
    return out


def supuestos_estudio(root):
    """Supuestos comunes del estudio y si hay mezcla (que invalida comparaciones)."""
    valores = {k: set() for k in SUPUESTOS}
    for _tipo, _caso, _e, reg in registros_estudio(root):
        if reg.get("status") != "optimal":
            continue
        for k in SUPUESTOS:
            valores[k].add(repr(reg.get(k)))
    comunes = {k: eval(next(iter(v))) for k, v in valores.items() if len(v) == 1}  # noqa: S307 (repr propio)
    mezclados = {k: sorted(v) for k, v in valores.items() if len(v) > 1}
    return comunes, mezclados


def seccionadores_de(variante):
    """'sec-0-2-9-10' -> [0, 2, 9, 10]; 'base' -> []."""
    if not variante or not str(variante).startswith("sec-"):
        return []
    return [int(x) for x in str(variante)[4:].split("-") if x.isdigit()]


def contenido_estudio(root):
    """Medidas, combinaciones, escenarios y planes fijos presentes en disco."""
    medidas, combos, escs, planes = set(), set(), set(), set()
    for tipo, caso, e, reg in registros_estudio(root):
        if tipo == "individual":
            medidas.add(caso)
            escs.add(e)
        elif tipo == "combinadas":
            combos.add(caso)
            escs.add(e)
        else:
            planes.add(caso.split("/")[0])
    return {"medidas": sorted(medidas), "combos": sorted(combos),
            "escenarios": sorted(escs), "planes_fijos": sorted(planes)}
