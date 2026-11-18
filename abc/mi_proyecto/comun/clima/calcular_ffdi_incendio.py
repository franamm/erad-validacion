"""Capas FFDI y Fire para una base horaria operativa (p. ej. BaseHoraEWE_arauco_severo_clima_2050.nc).

Las bases operativas (construir_ano_operativo.py) traen wind, gust, tem, hur y prec,
pero no FFDI ni Fire. Este script las agrega con la MISMA fórmula de
Extreme_events_arclim_final.py (SMD, factor de sequía DF y FFDI de McArthur,
evento de incendio si FFDI diario > umbral, en la hora de máxima temperatura).

Diferencia con el script oficial: allá los valores diarios (tasmax, sfcWind, pr, hur)
vienen de ARClim diario; aquí no existen, así que se agregan desde la base horaria:
  Tmax = máximo diario de tem | viento = media diaria del SOSTENIDO | lluvia = suma
  diaria de prec | humedad = media diaria de hur.
El viento del FFDI es el sostenido (lon=0), igual que en el oficial, donde el FFDI se
calcula antes de aplicar el factor de ráfaga. La ráfaga (lon=1) no entra al índice.

Salida (con --salida): copia de la base con FFDI en ambos lon y Fire solo en lon=1, que
es la convención del oficial y el lon que lee construir_faultlin con with_EWE.
Sin --salida solo imprime el resumen (no escribe nada).

    python calcular_ffdi_incendio.py --entrada <base.nc> [--salida <base_incendio.nc>]
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def ffdi_diario(tmax, viento_ms, lluvia, hur):
    """FFDI diario de una zona. Réplica de Extreme_events_arclim_final.py (líneas 413-456)."""
    n = len(tmax)
    smd = np.zeros(n)
    ffdi = np.zeros(n)
    for d in range(n):
        R = lluvia[d - 365:d].sum() if d > 365 else lluvia[:min(365, n)].sum()
        prev = smd[d - 1] if d > 0 else 0.0
        a = (203.2 - prev) * (0.968 * np.exp(0.0875 * tmax[d] + 1.552) - 8.30) / 1000
        b = 1 + 10.88 * np.exp(-0.00173 * R)
        peff = lluvia[d] if lluvia[d] > 5 else 0
        smd[d] = max(prev - peff + a / b, 0)

        ventana = lluvia[d - 19:d + 1] if d >= 20 else lluvia[:min(20, n)]
        pacum = ventana.sum()
        N = len(ventana) - 1 - int(np.argmax(ventana)) if pacum > 2 else -1
        if pacum >= 2 and N >= 1:
            x = (N ** 1.3) / (N ** 1.3 + pacum - 2)
        elif pacum >= 2 and N == 0:
            x = (0.8 ** 1.3) / (0.8 ** 1.3 + pacum - 2)
        else:
            x = 1
        DF = max(10.5 * (1 - np.exp((-smd[d] + 30) / 40)) * (41 * x ** 2 + x) / (40 * x ** 2 + x + 1), 0)
        if DF > 0:
            ffdi[d] = 2 * np.exp(-0.45 + 0.987 * np.log(DF) - 0.0345 * min(hur[d], 100)
                                 + 0.0338 * tmax[d] + 0.0234 * viento_ms[d] * 3.6)
    return ffdi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", required=True, help="base horaria .nc (time, lat=zona, lon=0 sostenido/1 ráfaga)")
    ap.add_argument("--salida", default=None, help="si se da, escribe la base con FFDI y Fire")
    ap.add_argument("--umbral", type=float, default=15.0, help="umbral FFDI de incendio (igual que el oficial)")
    args = ap.parse_args()

    ds = xr.open_dataset(args.entrada)
    for v in ("tem", "hur", "wind", "prec"):
        if v not in ds:
            raise SystemExit(f"Falta la variable {v!r} en {args.entrada}. Hay: {list(ds.data_vars)}")
    horas, n_z = ds["tem"].shape[0], ds["tem"].shape[1]
    if horas % 24:
        raise SystemExit(f"{horas} horas no es múltiplo de 24")
    dias = horas // 24

    # lon=0 en todas las variables que entran al índice (el viento sostenido es wind[..., 0])
    def diario(v, f):
        return f(ds[v].values[:, :, 0].reshape(dias, 24, n_z), axis=1)
    tmax, hur_d = diario("tem", np.max), diario("hur", np.mean)
    viento, lluvia = diario("wind", np.mean), diario("prec", np.sum)

    ffdi_d = np.zeros((dias, n_z))
    for z in range(n_z):
        ffdi_d[:, z] = ffdi_diario(tmax[:, z], viento[:, z], lluvia[:, z], hur_d[:, z])

    tem_h = ds["tem"].values[:, :, 0].reshape(dias, 24, n_z)
    fire = np.zeros((horas, n_z, ds.sizes["lon"]), dtype="float32")
    hora_max = tem_h.argmax(axis=1)                       # (dias, zonas)
    for d, z in zip(*np.where(ffdi_d > args.umbral)):
        fire[d * 24 + hora_max[d, z], z, 1] = 1.0
    ffdi_h = np.repeat(ffdi_d, 24, axis=0)[:, :, None].repeat(ds.sizes["lon"], axis=2).astype("float32")

    anios = max(dias // 365, 1)
    print(f"{Path(args.entrada).name}: {dias} días, {n_z} zonas, umbral FFDI {args.umbral}")
    print(f"  Tmax diaria máx {tmax.max():.1f} °C | viento sostenido medio diario máx {viento.max():.1f} m/s")
    for z in range(n_z):
        porano = [int((ffdi_d[a * 365:(a + 1) * 365, z] > args.umbral).sum()) for a in range(anios)]
        print(f"  zona {int(ds['zone_id'].values[z])}: FFDI máx {ffdi_d[:, z].max():5.1f} | "
              f"días > {args.umbral:g} por año {porano} | días > 12: {int((ffdi_d[:, z] > 12).sum())}"
              f" | > 25: {int((ffdi_d[:, z] > 25).sum())}")

    if args.salida:
        sal = ds.copy(deep=True)
        sal["FFDI"] = (("time", "lat", "lon"), ffdi_h, {"units": "Adim"})
        sal["Fire"] = (("time", "lat", "lon"), fire, {"units": "Adim"})
        sal.attrs["note_incendio"] = ("FFDI/Fire calculados por calcular_ffdi_incendio.py desde la base horaria; "
                                      "viento sostenido (lon=0); Fire solo en lon=1.")
        sal.to_netcdf(args.salida)
        print(f"Escrito: {args.salida}")


if __name__ == "__main__":
    main()
