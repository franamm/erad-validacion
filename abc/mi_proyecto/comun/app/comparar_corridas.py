#!/usr/bin/env python
"""Compara la misma corrida (escenario) de dos carpetas del barrido.

Pensado para decidir si un cambio de supuestos (p. ej. la flota de MEG) altera
el plan: lista, por clave del plan, la inversión de cada lado y marca las que
difieren; luego indicadores, uso de MEG y arriendos.

    python comparar_corridas.py \\
        ../../caso_elmelon/datos/artefactos/elmelon/barrido/todas10_sin_ess_v2/combinadas/todas/esc_053.json \\
        ../../caso_elmelon/datos/artefactos/elmelon/barrido/todas10_sin_ess_v3/combinadas/todas/esc_053.json

`--tol` es la tolerancia relativa para decir que dos capacidades son iguales.
"""
import argparse
import json


def _cargar(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _dif(a, b, tol):
    return abs(a - b) > tol * max(abs(a), abs(b), 1e-9)


def comparar_plan(pa, pb, tol):
    """Imprime el plan de inversión lado a lado; devuelve las claves que difieren."""
    distintas = []
    for clave in sorted(set(pa) | set(pb)):
        da, db = pa.get(clave, {}), pb.get(clave, {})
        for i in sorted(set(da) | set(db), key=lambda s: (len(s), s)):
            va, vb = float(da.get(i, 0.0)), float(db.get(i, 0.0))
            if _dif(va, vb, tol):
                distintas.append((clave, i, va, vb))
    return distintas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="corrida de referencia (.json), p. ej. la v2")
    ap.add_argument("b", help="corrida nueva (.json), p. ej. la v3")
    ap.add_argument("--tol", type=float, default=0.01, help="tolerancia relativa (default 1 %%)")
    args = ap.parse_args()

    a, b = _cargar(args.a), _cargar(args.b)

    print("== Supuestos de MEG con que se corrió cada una (None = no registrado) ==")
    for k in ("n_meg", "P_meg_max", "t_tra_meg", "meg_existentes"):
        print(f"  {k:15s} A={a.get(k)!s:8s} B={b.get(k)!s:8s}")

    print("\n== Plan de inversión ==")
    dist = comparar_plan(a.get("plan", {}), b.get("plan", {}), args.tol)
    if not dist:
        print(f"  Idéntico (tolerancia {args.tol:.0%}).")
    for clave, i, va, vb in dist:
        print(f"  {clave:16s} id {i:>4s}   A={va:12.3f}   B={vb:12.3f}")

    print("\n== MEG (kW instalados; E en kWh) ==")
    exist_b = int(b.get("meg_existentes") or 0)
    p_b = float(b.get("P_meg_max") or 0.0)
    gratis = exist_b * p_b
    for nombre, r in (("A", a), ("B", b)):
        cap = float(r.get("cap_meg_MW", 0.0))
        print(f"  {nombre}: cap_meg={cap:9.2f}  E_meg={r.get('E_meg', 0):10.1f}  "
              f"E_meg_arr={r.get('E_meg_arr', 0):10.1f}  arriendos={r.get('n_arriendos_meg', 0)}")
    comprado_b = max(0.0, float(b.get("cap_meg_MW", 0.0)) - gratis)
    print(f"  B: existente (gratis) = {gratis:.1f} kW, comprado = {comprado_b:.1f} kW")

    print("\n== Indicadores ==")
    for k in ("objetivo", "ens", "saidi", "saifi", "costo_ens"):
        if k in a and k in b:
            va, vb = float(a[k]), float(b[k])
            print(f"  {k:10s} A={va:14.3f}   B={vb:14.3f}   ({(vb - va) / va:+.1%})" if va else
                  f"  {k:10s} A={va:14.3f}   B={vb:14.3f}")

    solo_existente = comprado_b < 1e-6 and float(b.get("E_meg", 0.0)) > 0
    print("\n== Veredicto sobre el plan ==")
    print("  Plan igual:", "sí" if not dist else "NO")
    print("  MEG: ", "solo la existente" if solo_existente else
          ("no se usó MEG propio" if float(b.get("E_meg", 0.0)) == 0 else "se compró MEG además de la existente"))


if __name__ == "__main__":
    main()
