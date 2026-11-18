"""Generación FV horaria del Explorador Solar en los puntos de zona del caso activo.

Variable `pv` de api.minenergia.cl (radiación del modelo CLIRAD, PVWatts), misma configuración que el
caso de validación (caso_validacion/Proyecto oficial/Inputs/explorador_solar/parametros_pv_explorador.json):
panel fijo hacia el norte, inclinación = latitud del sitio, 1 kWp, pérdidas 14 %, inversor 0,96,
coeficiente de temperatura -0,45 %/°C, montaje abierto, vidrio. Años 2004-2016.

Uso (desde dentro del caso):  python ../comun/clima/descargar_pv_explorador.py
Lee <caso>/datos/puntos_zona_sobre_red.csv y escribe en <caso>/Proyecto oficial/Inputs/explorador_solar/
`explorador_pv_Z<zona>.csv` (time, pv) y `explorador_pv_Z<zona>_info.json` (sitio y solicitud).
No vuelve a descargar una zona que ya tiene archivo.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explorador_api as E          # noqa: E402
from caso_clima import caso_dir     # noqa: E402

CASO = Path(caso_dir())
SALIDA = CASO / "Proyecto oficial" / "Inputs" / "explorador_solar"
ANIOS = range(2004, 2017)
PV = {"Pcdo": 1, "coeft": -0.45, "use_corrected_temperature": False, "cover": "glass",
      "mounting": "open_rack_cell_glassback", "losses": 14, "eta_nom": 0.96, "DC_AC_ratio": 1.0}
RECEPTOR = {"type": "tilt", "azimuth": 0, "elev1": 99, "hsatmax": 45, "optimize": False}   # 99 = latitud del sitio


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    puntos = pd.read_csv(CASO / "datos" / "puntos_zona_sobre_red.csv")
    s = None
    for p in puntos.itertuples():
        destino = SALIDA / f"explorador_pv_Z{p.Zona}.csv"
        if destino.exists():
            print("ya existe", destino.name)
            continue
        s = s or E.sesion_web()
        partes, info = [], None
        for a in ANIOS:
            req = {"action": {"action": "series", "interval": "hour", "stat": "mean", "tmy": False},
                   "period": {"start": f"{a}-01-01", "end": f"{a}-12-31"},
                   "export": {"label": f"pv_Z{p.Zona}_{a}", "format": "json"},
                   "variables": [{"id": "pv", "options": {"label": "PV", "stat": "default",
                                                          "receptor": RECEPTOR, "pv": PV}}],
                   "position": [{"label": f"Z{p.Zona}", "type": "point", "lat": float(p.Lat), "lon": float(p.Lon)}]}
            for _ in range(3):
                r = s.post(E.BASE + "/api/proxy", data=json.dumps(req),
                           headers={"Content-Type": "application/json"}, timeout=600)
                if r.status_code == 200 and "_ERROR" not in r.text[:200]:
                    break
                print("  reintento", a, r.status_code, r.text[:120])
                time.sleep(10)
            d = r.json()
            info = info or {"site_info": d["site_info"][0], "variable_info": d["variable_info"][0], "solicitud": req}
            partes.append(pd.DataFrame({"time": pd.to_datetime(d["time"], unit="s"), "pv": d["data"][0]}))
            print(f"Z{p.Zona} {a}: {len(d['time'])} h, FP {pd.Series(d['data'][0], dtype=float).mean():.4f}")
        pd.concat(partes).to_csv(destino, index=False)
        (SALIDA / f"explorador_pv_Z{p.Zona}_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print("listo")


if __name__ == "__main__":
    main()
