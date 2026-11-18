"""Compara reconfig_topo sin radialidad (barrido/reconfig_prueba) con radialidad (barrido/radial_local)
hora a hora por evento. Uso (desde cualquier carpeta):
    python comparar_radial.py <carpeta artefactos del caso> [escenario]
"""
import sys, json
from pathlib import Path
import numpy as np

art = Path(sys.argv[1]); esc = int(sys.argv[2]) if len(sys.argv) > 2 else 53
rutas = {"sin radialidad": art / "barrido" / "reconfig_prueba" / "individual" / "reconfig_topo",
         "con radialidad": art / "barrido" / "radial_local" / "individual" / "reconfig_topo"}
datos = {}
for nombre, r in rutas.items():
    j = json.load(open(r / f"esc_{esc:03d}.json"))
    z = np.load(r / f"esc_{esc:03d}.npz", allow_pickle=True)
    datos[nombre] = (j, z)
    print(f"{nombre:15s} status={j['status']:8s} obj={j['objetivo']:>10,.0f}  costo_ens={j['costo_ens']:>10,.0f}  "
          f"maniobras_tie={j['maniobras_tie']:.0f}  saidi={j['saidi']:.2f}  gap={j['solver_gap']:.4f}")

z0 = datos["sin radialidad"][1]
dano = z0["dano"] > 0
bloques, i = [], 0
while i < len(dano):
    if dano[i]:
        j = i
        while j + 1 < len(dano) and dano[j + 1]:
            j += 1
        bloques.append((i, j)); i = j + 1
    else:
        i += 1
print(f"\nEscenario {esc}: {len(bloques)} eventos")
print(f"{'evento (h)':14s} {'enlaces·h cerrados':>22s} {'enlaces simult. max':>22s} {'P_LC kWh':>22s}")
print(f"{'':14s} {'sin':>10s} {'con':>11s} {'sin':>10s} {'con':>11s} {'sin':>10s} {'con':>11s}")
for a, b in bloques:
    f = []
    for nombre in rutas:
        z = datos[nombre][1]
        C = z["det_tie_cerrada"][a:b + 1]
        f.append((int(C.sum()), int(C.sum(axis=1).max()), float(z["P_LC"][a:b + 1].sum())))
    print(f"{a:5d}-{b:<8d} {f[0][0]:10d} {f[1][0]:11d} {f[0][1]:10d} {f[1][1]:11d} {f[0][2]:10,.0f} {f[1][2]:11,.0f}")
