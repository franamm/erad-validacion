"""Viento WRF 1 km a 30 m (Explorador Eólico) en las barras de la red reducida del caso activo.

Verificación del recurso eólico: el modelo usa el viento IFS de cada año (el mismo de las fallas); esto
revisa con la simulación WRF de 1 km del Explorador (año 2015) si hay sitios de la red con más viento.

Uso (desde dentro del caso):  python ../comun/clima/descargar_wrf_barras.py --artefactos elmelon
Lee <caso>/datos/artefactos/<artefactos>/red_nueva.pkl y escribe en <caso>/datos/generacion/:
  wrf_viento30m_barras_2015.csv  (horario, una columna por celda de ~1 km con barras)
  wrf_celdas_barras.csv          (celda, lat, lon, zona, n.º de barras, viento medio)
  wrf_barras_celda.csv           (barra -> celda)
Una consulta por celda de 0,01° (varias barras caen en la misma).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explorador_api as E          # noqa: E402
from caso_clima import caso_dir     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artefactos", required=True, help="carpeta en datos/artefactos/ (p. ej. elmelon)")
    ap.add_argument("--anio", type=int, default=2015, help="año simulado del WRF (2010 o 2015)")
    ap.add_argument("--altura", type=int, default=30, help="altura del buje [m]")
    args = ap.parse_args()

    datos = Path(caso_dir()) / "datos"
    salida = datos / "generacion"
    red = pd.read_pickle(datos / "artefactos" / args.artefactos / "red_nueva.pkl")
    net = red["net"]
    geo = net.bus.geo.apply(lambda g: json.loads(g)["coordinates"])
    barras = pd.DataFrame({"bus": net.bus.index, "lon": [g[0] for g in geo], "lat": [g[1] for g in geo],
                           "zona": net.bus.index.map(red["bus_zone"])})
    barras["celda"] = barras.lat.round(2).astype(str) + "," + barras.lon.round(2).astype(str)
    celdas = barras.groupby("celda").agg(lat=("lat", "mean"), lon=("lon", "mean"), zona=("zona", "first"),
                                         barras=("bus", "size")).reset_index()
    print(f"{len(barras)} barras -> {len(celdas)} celdas WRF")

    s = E.sesion_web()
    series, filas, tiempo = {}, [], None
    for c in celdas.itertuples():
        req = {"action": {"action": "series", "interval": "hour", "stat": "mean", "tmy": False},
               "period": {"start": f"{args.anio}-01-01", "end": f"{args.anio}-12-31"},
               "export": {"label": f"vel{args.altura}_{c.Index}", "format": "json"},
               "variables": [{"id": "vel", "options": {"label": "Vel", "stat": "default", "hgt": args.altura}}],
               "position": [{"label": c.celda, "type": "point", "lat": float(c.lat), "lon": float(c.lon)}]}
        for _ in range(3):
            r = s.post(E.BASE + f"/api/proxy?sim={args.anio}", data=json.dumps(req),
                       headers={"Content-Type": "application/json"}, timeout=600)
            if r.status_code == 200 and "_ERROR" not in r.text[:200]:
                break
            print("  reintento", c.celda, r.status_code, r.text[:150])
            time.sleep(5)
        d = r.json()
        v = np.array(d["data"][0], dtype=float)
        series[c.celda], tiempo = v, d["time"]
        filas.append({"celda": c.celda, "lat": c.lat, "lon": c.lon, "zona": c.zona, "barras": c.barras,
                      f"media_{args.altura}m": float(np.nanmean(v))})
    salida.mkdir(exist_ok=True)
    pd.DataFrame(series, index=pd.to_datetime(tiempo, unit="s")).to_csv(
        salida / f"wrf_viento{args.altura}m_barras_{args.anio}.csv", index_label="time")
    pd.DataFrame(filas).to_csv(salida / "wrf_celdas_barras.csv", index=False)
    barras.to_csv(salida / "wrf_barras_celda.csv", index=False)
    print("guardado en", salida, len(series), "celdas")


if __name__ == "__main__":
    main()
