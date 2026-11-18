# ============================================================
# construir_faultlin_anios.py — Matriz de fallas con VARIOS años de viento
# ============================================================
# `construir_faultlin.py` sortea todos los escenarios sobre UN año de viento.
# Este envoltorio lo corre una vez por cada año operativo de un NetCDF
# multianual (`comun/clima/construir_anios_observados.py`) y junta los
# resultados en el eje de escenarios:
#
#     escenario e  ->  año k = e // sorteos_por_anio
#
# Así cada año observado es un escenario climático y los sorteos son sus
# subescenarios. El sorteo en sí NO cambia: cada año es una corrida normal de
# construir_faultlin.py (mismos flags, semilla propia), así que E2 y lo que
# viene después leen los .npy combinados igual que siempre.
#
# Delta climático: entra multiplicado en `--factor-zona` junto con la
# normalización por zona, y queda registrado por separado en el meta.
#
# USO (desde dentro del caso):
#   python ../comun/app/construir_faultlin_anios.py --artefactos datos/artefactos/elmelon \
#       --red datos/artefactos/elmelon/red_nueva.pkl --sorteos-por-anio 10 --horas-espera 5 \
#       --curva dunn --dunn-alpha 1.13e-15 --normalizacion-zona 1.080,1.090,0.864 \
#       --sin-ajuste-altura --enlaces-inmunes --delta-estadistico p99
#
# NO sobrescribe los .npy del caso sin --forzar.
# ============================================================

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from rutas import CASO_DIR, DATOS_DIR

APP_DIR = Path(__file__).resolve().parent
HORAS_ANIO = 8760
VARIANTES = ("", "_veg", "_ref", "_ug", "_ramales")


def _log(msg):
    print(f"[faultlin-años] {msg}", flush=True)


def delta_arclim(ruta_csv, zonas, estadistico, modelo, ref, fut):
    """Cociente futuro/referencia del viento diario de ARClim, por zona.

    Referencia = período de los datos observados (no 1985–2014): el año observado ya
    contiene el clima de ese período, y compararlo contra 1985–2014 contaría dos veces
    el cambio ocurrido entre ambos.
    """
    d = pd.read_csv(ruta_csv, parse_dates=["fecha"])
    d["y"] = d.fecha.dt.year
    miembros = [c for c in d.columns if c not in ("zona", "fecha", "y")]
    out = {}
    for z in zonas:
        x = d[d.zona == z]
        h = x[x.y.between(*ref)]
        u = x[x.y.between(*fut)]
        if estadistico == "p99":
            r = u[miembros].quantile(0.99) / h[miembros].quantile(0.99)
        elif estadistico == "max_anual":
            r = u.groupby("y")[miembros].max().mean() / h.groupby("y")[miembros].max().mean()
        else:
            raise ValueError(estadistico)
        if modelo == "mediana":
            valor = float(r.median())
        else:
            sel = [c for c in miembros if c.startswith(modelo)]
            if not sel:
                raise ValueError(f"modelo '{modelo}' no está en {ruta_csv.name}")
            valor = float(r[sel[0]])
        out[z] = {"delta": valor, "min_modelos": float(r.min()), "max_modelos": float(r.max())}
    return out


def main():
    ap = argparse.ArgumentParser(description="FaultlinMC con varios años de viento (uno por bloque de escenarios)")
    ap.add_argument("--artefactos", required=True, help="carpeta de artefactos del caso (donde quedan los .npy)")
    ap.add_argument("--red", required=True, help="red.pkl o red_nueva.pkl")
    ap.add_argument("--viento", default=None,
                    help="NetCDF multianual. Default: Proyecto oficial/Outputs/BaseHoraEWE_<caso>_anios.nc")
    ap.add_argument("--sorteos-por-anio", type=int, default=10)
    ap.add_argument("--semilla", type=int, default=42, help="semilla del año k = semilla + 100·k")
    ap.add_argument("--horas-espera", type=int, required=True)
    ap.add_argument("--curva", default="dunn", choices=["lognormal", "dunn"])
    ap.add_argument("--dunn-alpha", type=float, default=None)
    ap.add_argument("--normalizacion-zona", default=None, help="factores por zona, ej. 1.080,1.090,0.864")
    ap.add_argument("--sin-ajuste-altura", action="store_true")
    ap.add_argument("--enlaces-inmunes", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--delta", default=None, help="delta por zona explícito, ej. 1.01,1.00,1.02")
    g.add_argument("--delta-estadistico", default=None, choices=["p99", "max_anual"],
                   help="calcular el delta desde datos/arclim_viento_diario_ssp585.csv")
    ap.add_argument("--delta-modelo", default="mediana", help="'mediana' o el nombre de un modelo, ej. AWI-CM-1-1-MR")
    ap.add_argument("--delta-ref", default="2017-2026")
    ap.add_argument("--delta-fut", default="2031-2050")
    ap.add_argument("--veg-rho", type=float, default=None,
                    help="poda como reducción fija (ej. 0.33) en vez de Canham; se pasa a construir_faultlin.py")
    ap.add_argument("--salida-dir", default=None,
                    help="carpeta donde escribir los .npy y el meta (default: --artefactos). Sirve para "
                         "variantes que no deben reemplazar la matriz vigente del caso.")
    ap.add_argument("--forzar", action="store_true")
    args = ap.parse_args()

    art = Path(args.salida_dir).resolve() if args.salida_dir else Path(args.artefactos).resolve()
    art.mkdir(parents=True, exist_ok=True)
    red = Path(args.red).resolve()
    viento = Path(args.viento).resolve() if args.viento else \
        CASO_DIR / "Proyecto oficial" / "Outputs" / f"BaseHoraEWE_{CASO_DIR.name.removeprefix('caso_')}_anios.nc"
    ds = xr.open_dataset(viento)
    n_horas_nc, n_zonas = ds["wind"].shape[0], ds["wind"].shape[1]
    zonas = [int(z) for z in ds["zone_id"].values] if "zone_id" in ds.coords else list(range(1, n_zonas + 1))
    if n_horas_nc % HORAS_ANIO:
        sys.exit(f"{viento.name} tiene {n_horas_nc} h, que no es múltiplo de {HORAS_ANIO}")
    n_anios = n_horas_nc // HORAS_ANIO
    etiquetas = [s.strip() for s in ds.attrs.get("anios_operativos", "").split(",")] \
        if ds.attrs.get("anios_operativos") else [f"año {k}" for k in range(n_anios)]
    ds.close()

    salidas = {v: art / f"FaultlinMC{v}.npy" for v in VARIANTES}
    existentes = [p.name for p in salidas.values() if p.exists()]
    if existentes and not args.forzar:
        sys.exit(f"Ya existen {existentes} en {art}. Usar --forzar para reemplazarlos.")

    # Factor por zona = normalización × delta
    norm = np.array([float(x) for x in args.normalizacion_zona.split(",")]) if args.normalizacion_zona \
        else np.ones(n_zonas)
    info_delta = None
    if args.delta:
        delta = np.array([float(x) for x in args.delta.split(",")])
        info_delta = {"origen": "explícito", "valores": delta.tolist()}
    elif args.delta_estadistico:
        ref = tuple(int(x) for x in args.delta_ref.split("-"))
        fut = tuple(int(x) for x in args.delta_fut.split("-"))
        csv = DATOS_DIR / "arclim_viento_diario_ssp585.csv"
        por_zona = delta_arclim(csv, zonas, args.delta_estadistico, args.delta_modelo, ref, fut)
        delta = np.array([por_zona[z]["delta"] for z in zonas])
        info_delta = {"origen": csv.name, "estadistico": args.delta_estadistico, "modelo": args.delta_modelo,
                      "referencia": args.delta_ref, "ventana_futura": args.delta_fut,
                      "por_zona": {str(z): v for z, v in por_zona.items()}}
    else:
        delta = np.ones(n_zonas)
    if len(norm) != n_zonas or len(delta) != n_zonas:
        sys.exit(f"normalización ({len(norm)}) y delta ({len(delta)}) deben tener {n_zonas} valores")
    factor = norm * delta
    _log(f"{n_anios} años × {args.sorteos_por_anio} sorteos = {n_anios * args.sorteos_por_anio} escenarios")
    _log(f"normalización {norm.round(4).tolist()} × delta {delta.round(4).tolist()} = "
         f"factor {factor.round(4).tolist()}")

    tmp = art / "_faultlin_por_anio"
    if tmp.exists():
        shutil.rmtree(tmp)
    metas = []
    for k in range(n_anios):
        dir_k = tmp / f"anio_{k:02d}"
        dir_k.mkdir(parents=True)
        cmd = [sys.executable, str(APP_DIR / "construir_faultlin.py"),
               "--artefactos", str(dir_k), "--red", str(red), "--viento", str(viento),
               "--horas", str(HORAS_ANIO), "--anio-offset", str(k * HORAS_ANIO),
               "--escenarios", str(args.sorteos_por_anio), "--semilla", str(args.semilla + 100 * k),
               "--escenario-clima", "with_EWE", "--curva", args.curva,
               "--horas-espera", str(args.horas_espera),
               "--factor-zona", ",".join(f"{x:.6f}" for x in factor),
               "--salida", str(dir_k / "FaultlinMC.npy")]
        if args.dunn_alpha is not None:
            cmd += ["--dunn-alpha", repr(args.dunn_alpha)]
        if args.sin_ajuste_altura:
            cmd.append("--sin-ajuste-altura")
        if args.enlaces_inmunes:
            cmd.append("--enlaces-inmunes")
        if args.veg_rho is not None:
            cmd += ["--veg-rho", repr(args.veg_rho)]
        _log(f"año {k} ({etiquetas[k]}): offset {k * HORAS_ANIO}, semilla {args.semilla + 100 * k}")
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-3000:])
            sys.exit(f"construir_faultlin.py falló en el año {k}")
        metas.append(json.loads((dir_k / "faultlin_meta.json").read_text(encoding="utf-8")))
        tot = metas[-1]["fallas_totales"]
        _log("   " + " · ".join(f"{v} {n}" for v, n in tot.items()))

    # Juntar en el eje de escenarios
    totales, por_anio = {}, {}
    for v, destino in salidas.items():
        partes = [tmp / f"anio_{k:02d}" / f"FaultlinMC{v}.npy" for k in range(n_anios)]
        if not all(p.exists() for p in partes):
            if any(p.exists() for p in partes):
                sys.exit(f"FaultlinMC{v}.npy existe solo en algunos años")
            continue
        arr = np.concatenate([np.load(p) for p in partes], axis=1)
        np.save(destino, arr)
        nombre = v.lstrip("_") or "base"
        totales[nombre] = int(arr.sum())
        por_anio[nombre] = [int(np.load(p).sum()) for p in partes]
        _log(f"→ {destino.name} {arr.shape} · {totales[nombre]} fallas")
    forma = list(np.load(salidas[""], mmap_mode="r").shape)

    meta = dict(metas[0])
    meta.update({
        "salidas": {(v.lstrip("_") or "base"): salidas[v].name for v in VARIANTES if salidas[v].exists()},
        "shape": forma,
        "fallas_totales": totales,
        "semilla": [args.semilla + 100 * k for k in range(n_anios)],
        "viento_nc": str(viento),
        "anio_offset": [k * HORAS_ANIO for k in range(n_anios)],
        "factor_zona": ",".join(f"{x:.6f}" for x in factor),
        "multianual": {
            "script": Path(__file__).name,
            "sorteos_por_anio": args.sorteos_por_anio,
            "anios": [{"k": k, "anio_operativo": etiquetas[k], "offset_horas": k * HORAS_ANIO,
                       "semilla": args.semilla + 100 * k,
                       "escenarios": [k * args.sorteos_por_anio, (k + 1) * args.sorteos_por_anio - 1],
                       "fallas": {n: por_anio[n][k] for n in por_anio}} for k in range(n_anios)],
            "escenario_a_anio": [etiquetas[e // args.sorteos_por_anio] for e in range(forma[1])],
            "normalizacion_zona": norm.tolist(),
            "delta_climatico": info_delta,
            "hora_0": "1 de octubre 00:00 (hora local fija UTC-4) de cada año operativo",
        },
    })
    (art / "faultlin_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(tmp)
    _log(f"→ faultlin_meta.json · siguiente: pipeline.py --artefactos {art} --etapas E2 --forzar")


if __name__ == "__main__":
    main()
