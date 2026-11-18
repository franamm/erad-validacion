#!/usr/bin/env python
# ============================================================
# resumen_sensibilidad_voll.py — tabla de la sensibilidad al VoLL (caso de validación)
# ============================================================
# Lee las carpetas artefactos/<caso>/barrido/sensibilidad_voll_<valor>/ que deja
# correr_sensibilidad_voll.ps1 (combinación de las 12 medidas sobre los 10 escenarios
# severos, con un VoLL distinto en cada carpeta) y las compara con la corrida original
# de la tesis (VoLL = 5 USD/kWh, barrido "sec-0-2-9-10 severo").
#
# Por cada VoLL entrega: ENS esperada, reducción, costo de las medidas, relación
# beneficio-costo, y en cuántos escenarios se activa cada medida. Con eso se ve a qué
# VoLL entran o salen el refuerzo, el soterramiento, el almacenamiento, etc.
#
# También compara la corrida de control (control_voll_5, escenario 9) con la original:
# si no coincide, el código cambió el modelo y la sensibilidad no es comparable.
#
# USO
#   python resumen_sensibilidad_voll.py --artefactos <ruta a artefactos/validacion>
# ============================================================
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCIA = "sec-0-2-9-10 severo"      # tanda original de la tesis (VoLL = 5)
VOLL_ORIGINAL = 5.0
UMBRAL_ACTIVA = {   # columna del resultado y umbral desde el que la medida cuenta como activa
    "reconfig_topo": ("maniobras_tie", 0.0), "dg_sol": ("cap_dg_sol_kW", 1e-3),
    "dg_eol": ("cap_dg_eol_kW", 1e-3), "dg_fos": ("cap_dg_fos_kW", 1e-3),
    "ess_4h": ("cap_ess_4h_kWh", 1e-3), "ess_5h": ("cap_ess_5h_kWh", 1e-3),
    "ess_6h": ("cap_ess_6h_kWh", 1e-3), "meg": ("cap_meg_MW", 1e-6),
    "mess": ("cap_mess_MWh", 1e-6), "edrp": ("E_edrp", 1e-3),
    "refuerzo": ("km_ref", 1e-6), "soterramiento": ("km_ug", 1e-6)}


def leer(carpeta):
    regs = []
    for f in sorted(Path(carpeta).glob("esc_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("status") == "optimal":
            regs.append(r)
    return regs


def combinada_completa(base):
    """La combinación con más medidas de la carpeta `combinadas/` de una tanda."""
    cands = [d for d in (Path(base) / "combinadas").iterdir()
             if d.is_dir() and any(d.glob("esc_*.json"))]
    if not cands:
        raise SystemExit(f"No hay combinadas en {base}")
    return max(cands, key=lambda d: len(json.loads(next(d.glob("esc_*.json")).read_text("utf-8"))
                                        .get("medidas_on", [])))


def resumir(regs, voll):
    filas = []
    for r in regs:
        # cst_costo_ens_base es el costo del caso base calculado con el VoLL de la tesis (5): --voll
        # no lo reescala, así que la ENS base (que no depende del VoLL) se saca dividiendo por 5.
        base_ens = r["cst_costo_ens_base"] / VOLL_ORIGINAL
        fila = {"escenario": int(r["escenario"]), "ens_base": base_ens, "ens": r["ens"],
                "saidi": r["saidi"], "costo_medidas": r["cst_costo_total"],
                "km_ref": r.get("km_ref", 0.0), "km_ug": r.get("km_ug", 0.0),
                "dg_sol_kW": r.get("cap_dg_sol_kW", 0.0), "dg_eol_kW": r.get("cap_dg_eol_kW", 0.0),
                "gap": r.get("solver_gap", np.nan)}
        for m, (col, umbral) in UMBRAL_ACTIVA.items():
            fila[f"act_{m}"] = bool(r.get(col, 0.0) > umbral)
        filas.append(fila)
    return pd.DataFrame(filas).set_index("escenario").sort_index()


def fila_resumen(voll, d, fuente):
    ens_b, ens = d["ens_base"].mean(), d["ens"].mean()
    costo = d["costo_medidas"].mean()
    out = {"VoLL [USD/kWh]": voll, "fuente": fuente, "n escenarios": len(d),
           "ENS base [kWh]": ens_b, "ENS con medidas [kWh]": ens,
           "Reducción ENS [%]": 100 * (1 - ens / ens_b),
           "Costo medidas [USD/año]": costo,
           "Costo ENS evitada [USD/año]": voll * (ens_b - ens),
           "B/C": voll * (ens_b - ens) / costo if costo > 0 else np.nan,
           "Refuerzo medio [km]": d["km_ref"].mean(), "Soterramiento medio [km]": d["km_ug"].mean(),
           "DG solar media [kW]": d["dg_sol_kW"].mean(), "DG eólica media [kW]": d["dg_eol_kW"].mean(),
           "Gap máx. [%]": 100 * d["gap"].max()}
    for m in UMBRAL_ACTIVA:
        out[f"n_{m}"] = int(d[f"act_{m}"].sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--artefactos", default=os.environ.get("ERAD_ARTEFACTOS"),
                    help="carpeta artefactos/validacion (default: $ERAD_ARTEFACTOS)")
    args = ap.parse_args()
    if not args.artefactos:
        raise SystemExit("Falta --artefactos")
    barrido = Path(args.artefactos) / "barrido"

    filas = []
    ref = combinada_completa(barrido / REFERENCIA)
    d_ref = resumir(leer(ref), VOLL_ORIGINAL)
    filas.append(fila_resumen(VOLL_ORIGINAL, d_ref, "tesis (guardado, MEG 50 kW)"))
    print(f"Referencia (tesis): {ref.name[:40]}…  {len(d_ref)} escenarios")

    for carpeta in sorted((c for c in barrido.glob("sensibilidad_voll_*")
                           if re.search(r"_([0-9.]+)$", c.name)),
                          key=lambda p: float(re.search(r"_([0-9.]+)$", p.name).group(1))):
        voll = float(re.search(r"_([0-9.]+)$", carpeta.name).group(1))
        regs = leer(combinada_completa(carpeta))
        if regs:
            filas.append(fila_resumen(voll, resumir(regs, voll), "nueva (MEG 300 kW)"))

    tabla = pd.DataFrame(filas).sort_values(["VoLL [USD/kWh]", "fuente"]).reset_index(drop=True)
    tabla.to_csv(barrido / "resumen_sensibilidad_voll.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\nSENSIBILIDAD AL VoLL — combinación de las 12 medidas\n")
    print(tabla.round(2).T.to_string(header=False))
    print(f"\nGuardado: {barrido / 'resumen_sensibilidad_voll.csv'}")

    # Control: mismo VoLL (5) y mismo escenario que la tesis; tiene que dar lo mismo.
    ctl = barrido / "control_voll_5"
    if ctl.exists():
        regs = leer(combinada_completa(ctl))
        print("\nCONTROL (VoLL 5, comparación con la corrida original de la tesis)")
        for r in regs:
            o = next((x for x in leer(ref) if x["escenario"] == r["escenario"]), None)
            if o is None:
                continue
            for k in ("ens", "objetivo", "cst_costo_total"):
                dif = 100 * (r[k] - o[k]) / o[k] if o[k] else float("nan")
                print(f"  esc {r['escenario']} · {k:<16} nuevo {r[k]:>12.2f}  original {o[k]:>12.2f}  "
                      f"dif {dif:+.2f} %")
        print("  Aceptable: diferencias del orden del MIPGap (2 %). Mayores: el código cambió el modelo.")


if __name__ == "__main__":
    main()
