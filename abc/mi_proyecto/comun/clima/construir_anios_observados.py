#!/usr/bin/env python3
"""
Construye un NetCDF con VARIOS años operativos observados, uno detrás de otro.

POR QUÉ
-------
`construir_ano_operativo.py` arma un solo año (más el previo de arranque), y
con un solo año los resultados dependen de una o dos tormentas: en El Melón la
mitad de las fallas esperadas de cada año sale de 1 a 3 días, y el total
varía 3,4× entre años (notebook `02c_tormentas_elmelon.ipynb`). Este script
pone todos los años observados en un mismo archivo para que cada uno sea un
escenario climático distinto (como Panteli et al. 2017, que sortea un año de
reanálisis por simulación).

QUÉ PRODUCE
-----------
El mismo contrato que `BaseHoraEWE_arauco.nc`, para que `construir_faultlin.py`
lo lea sin cambios:

    dims  (time, lat, lon)
    lat   = point_id 0..Z-1 (zona 1..Z, en el orden de las columnas zg1, zg2, ...)
    lon   = 0 -> viento SOSTENIDO 10 m;  1 -> RÁFAGA 10 m

    El año k ocupa las horas [k*8760, (k+1)*8760): se lee con
    `construir_faultlin.py --anio-offset k*8760 --horas 8760`.

Cada año operativo va del 1 de octubre al 30 de septiembre y tiene 8760 h (se
descarta el 29 de febrero, igual que `construir_ano_operativo.py`).

NO aplica ningún delta climático: el archivo es el observado. El delta (y la
normalización por zona) entra como `--factor-zona` en `construir_faultlin.py`,
para que quede como parámetro de cada corrida y no escondido en el archivo.
La ráfaga ya es ráfaga: no multiplicar por ningún factor de ráfaga adicional.

HUECOS EN LOS EXTREMOS
----------------------
IFS en Open-Meteo parte el 1-ene-2017 (antes la API devuelve vacío), así que
al primer año operativo (2016/17) le faltan oct–dic de 2016: se copian las
mismas fechas del año siguiente. Y como la descarga es en UTC, al pasarla a
UTC-4 al último año le faltan las últimas horas del 30 de septiembre: se
copian las mismas horas del día anterior. Las dos cosas se hacen por defecto
(`--sin-rellenar` las desactiva) y quedan marcadas en la variable `relleno`
y en el resumen.

Uso (desde dentro del caso):
    python ../comun/clima/construir_anios_observados.py
    python ../comun/clima/construir_anios_observados.py --sin-rellenar
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from caso_clima import caso_dir, raiz_proyecto

HORAS_ANIO = 8760


def leer_csv(ruta: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    df = pd.read_csv(ruta)
    col_t = df.columns[0]
    df[col_t] = pd.to_datetime(df[col_t])
    df = df.set_index(col_t).sort_index()
    zonas = sorted(int(c[2:].split("_")[0]) for c in df.columns if c.startswith("zg") and c.endswith("_raf"))
    sost = pd.DataFrame({z: df[f"zg{z}_sost"] for z in zonas})
    raf = pd.DataFrame({z: df[f"zg{z}_raf"] for z in zonas})
    return sost, raf, zonas


def main() -> None:
    ap = argparse.ArgumentParser(description="Varios años operativos observados en un solo NetCDF.")
    ap.add_argument("--entrada-csv", default=None,
                    help="CSV horario por zona (columnas zgN_sost, zgN_raf). "
                         "Default: <caso>/datos/clima_historico_ifs_2017_2026.csv")
    ap.add_argument("--salida-nc", default=None,
                    help="Default: BaseHoraEWE_<caso>_anios.nc en Proyecto oficial/Outputs")
    ap.add_argument("--mes-inicio", type=int, default=10, help="Mes en que parte el año operativo (10 = octubre).")
    ap.add_argument("--sin-rellenar", dest="rellenar", action="store_false",
                    help="No completar el hueco del primer año; ese año se descarta si queda incompleto.")
    args = ap.parse_args()

    caso = caso_dir()
    nombre_caso = caso.name.removeprefix("caso_")
    salidas = Path(raiz_proyecto()) / "Outputs"
    salidas.mkdir(parents=True, exist_ok=True)
    entrada = Path(args.entrada_csv) if args.entrada_csv else caso / "datos" / "clima_historico_ifs_2017_2026.csv"
    salida = Path(args.salida_nc) if args.salida_nc else salidas / f"BaseHoraEWE_{nombre_caso}_anios.nc"

    sost, raf, zonas = leer_csv(entrada)
    relleno = pd.Series(False, index=raf.index)
    print(f"Caso    : {caso.name}")
    print(f"Entrada : {entrada.name}  ({raf.index.min()} → {raf.index.max()}, zonas {zonas})")

    # Hueco al inicio del primer año operativo
    m = args.mes_inicio
    primer_inicio = pd.Timestamp(year=raf.index.min().year - (raf.index.min().month < m), month=m, day=1)
    if args.rellenar and raf.index.min() > primer_inicio:
        falta = pd.date_range(primer_inicio, raf.index.min() - pd.Timedelta(hours=1), freq="h")
        origen = falta + pd.DateOffset(years=1)
        sost = pd.concat([sost.loc[origen].set_axis(falta), sost])
        raf = pd.concat([raf.loc[origen].set_axis(falta), raf])
        relleno = pd.concat([pd.Series(True, index=falta), relleno])
        print(f"Relleno : {len(falta)} h ({falta[0]:%d-%b-%Y} a {falta[-1]:%d-%b-%Y %H:%M}) copiadas del año siguiente")

    # Cola del último año: la descarga es en UTC y al pasar a UTC-4 faltan las últimas horas del
    # 30 de septiembre. Si faltan menos de 24 h, se copian las mismas horas del día anterior.
    ultimo_fin = pd.Timestamp(year=raf.index.max().year + (raf.index.max().month >= m), month=m, day=1) \
        - pd.Timedelta(hours=1)
    if args.rellenar and raf.index.max() < ultimo_fin \
            and (ultimo_fin - raf.index.max()) < pd.Timedelta(hours=24):
        falta = pd.date_range(raf.index.max() + pd.Timedelta(hours=1), ultimo_fin, freq="h")
        origen = falta - pd.Timedelta(hours=24)
        sost = pd.concat([sost, sost.loc[origen].set_axis(falta)])
        raf = pd.concat([raf, raf.loc[origen].set_axis(falta)])
        relleno = pd.concat([relleno, pd.Series(True, index=falta)])
        print(f"Relleno : {len(falta)} h ({falta[0]:%d-%b-%Y %H:%M} a {falta[-1]:%H:%M}) copiadas del día anterior")

    # Años operativos completos (sin 29-feb)
    sin_bis = ~((raf.index.month == 2) & (raf.index.day == 29))
    sost, raf, relleno = sost[sin_bis], raf[sin_bis], relleno[sin_bis]
    anio_op = raf.index.year + (raf.index.month >= m)          # oct-2016 a sep-2017 -> 2017
    anios = [a for a in np.unique(anio_op) if (anio_op == a).sum() == HORAS_ANIO]
    descartados = [int(a) for a in np.unique(anio_op) if a not in anios]
    sel = np.isin(anio_op, anios)
    sost, raf, relleno, anio_op = sost[sel], raf[sel], relleno[sel], anio_op[sel]
    etiquetas = [f"{a - 1}/{str(a)[2:]}" for a in anios]

    # Hora 0 de cada año = 1 de octubre 00:00 (hora local fija UTC-4)
    wind = np.stack([sost.to_numpy(), raf.to_numpy()], axis=2).astype("float32")   # (time, zona, lon)
    ds = xr.Dataset(
        data_vars={
            "wind": (("time", "lat", "lon"), wind,
                     {"units": "m/s", "description": "lon=0 sostenido 10 m; lon=1 RÁFAGA 10 m. "
                      "La ráfaga ya es ráfaga: no multiplicar por ningún factor adicional."}),
            "gust": (("time", "lat", "lon"), wind, {"units": "m/s", "description": "Alias de wind."}),
            "relleno": (("time",), relleno.to_numpy().astype("int8"),
                        {"description": "1 = hora copiada del año siguiente para completar el primer año"}),
            "anio_operativo": (("time",), anio_op.astype("int16"),
                               {"description": f"año en que termina el año operativo (parte en el mes {m})"}),
        },
        coords={
            "time": ("time", raf.index),
            "lat": ("lat", np.arange(len(zonas), dtype=int)),
            "lon": ("lon", np.array([0, 1], dtype=int)),
            "zone_id": ("lat", np.array(zonas, dtype=int)),
            "escenario": ("lon", np.array(["sostenido", "rafaga"], dtype="U10")),
        },
        attrs={
            "description": "Años operativos observados (reanálisis), uno detrás de otro. Sin delta climático.",
            "procedencia_horaria": f"{entrada.name} (Open-Meteo ecmwf_ifs, 10 m, hora local fija UTC-4)",
            "anios_operativos": ", ".join(etiquetas),
            "horas_por_anio": HORAS_ANIO,
            "mes_inicio": m,
            "aviso_offset": "el año k se lee con construir_faultlin.py --anio-offset k*8760 --horas 8760",
            "aviso_delta": "sin delta: aplicarlo con --factor-zona en construir_faultlin.py",
            "relleno_inicio": int(relleno.sum()),
        },
    )
    ds.to_netcdf(salida)

    resumen = pd.DataFrame({
        "k": range(len(anios)), "anio_operativo": etiquetas,
        "offset_horas": [k * HORAS_ANIO for k in range(len(anios))],
        "horas_relleno": [int(relleno[anio_op == a].sum()) for a in anios],
        **{f"rafaga_max_z{z}": [float(raf[z][anio_op == a].max()) for a in anios] for z in zonas},
    })
    resumen.to_csv(salida.with_name(salida.stem + "_resumen.csv"), index=False)
    meta = {"script": Path(__file__).name, "entrada": str(entrada), "salida": str(salida),
            "anios_operativos": etiquetas, "descartados_incompletos": descartados,
            "mes_inicio": m, "rellenar": args.rellenar, "horas_relleno": int(relleno.sum()),
            "zonas": zonas, "delta_climatico": None}
    salida.with_name(salida.stem + "_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                           encoding="utf-8")

    print(f"Años    : {', '.join(etiquetas)}  ({len(anios)} × {HORAS_ANIO} h)")
    if descartados:
        print(f"Descartados por incompletos (año en que terminan): {descartados}")
    print(resumen.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"\n  NetCDF : {salida}")


if __name__ == "__main__":
    main()
