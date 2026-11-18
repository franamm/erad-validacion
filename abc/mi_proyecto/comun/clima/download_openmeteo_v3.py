#!/usr/bin/env python3
"""
Descarga horaria historica desde Open-Meteo con RAFAGAS incluidas.

Sucesor de download_openmeteo_historical_ready_v2.py. Cambios respecto de v2:

  1. Baja `wind_gusts_10m` ademas de `wind_speed_10m`, y lo guarda como
     variable `gust` en el NetCDF. La rafaga es la magnitud que pide la curva
     de fragilidad; el sostenido sirve para el perfil de generacion eolica.

  2. FIJA EL MODELO en ecmwf_ifs en vez de usar best_match. Esto es critico:
     best_match combina IFS HRES + ERA5 + ERA5-Land "seamlessly" y cambia de
     fuente a mitad de la serie. Medido en Lebu, con best_match:
         2015-2016 -> las 3 zonas dan series IDENTICAS (ERA5 a 0.25 deg, ~25 km,
                      los 3 puntos caen en la misma celda) y 9.5 h/ano de
                      rafaga > 20.8 m/s
         2017-2019 -> las zonas se separan (IFS HRES, 9 km) y saltan a
                      230 h/ano. Un factor 24x que es cambio de modelo, no
                      meteorologia.
     Por eso el periodo por defecto parte en 2017, que es cuando IFS HRES esta
     disponible. era5_land NO sirve: devuelve viento vacio (n=0).

  3. Diagnostico de resolucion: reporta la celda que Open-Meteo devuelve para
     cada punto pedido y avisa si dos zonas caen en la MISMA celda, en cuyo
     caso sus series serian identicas y el contraste entre zonas seria ficticio.

  4. Diagnostico de homogeneidad: parte el registro en tercios y compara. Un
     salto grande entre tercios delata un cambio de modelo o de grilla.

  5. Mide el cociente Gf = gust / wind por zona, condicionado a horas ventosas.
     ATENCION a como se interpreta ese numero: ver la nota de abajo.

SOBRE LA PROCEDENCIA DE LOS DATOS
---------------------------------
Estos NO son datos observados. La Historical Weather API entrega reanalisis
(salida de modelo que asimila observaciones), no mediciones de estacion.

En consecuencia, el Gf que calcula este script es un DIAGNOSTICO DEL PRODUCTO
METEOROLOGICO, no una calibracion local:

  - La documentacion define wind_gusts_10m como "the maximum wind gusts of the
    preceding hour", pero NO especifica la duracion de la rafaga individual.
    El factor 1.263 del paper de referencia es explicitamente de 1 minuto.
    NO son intercambiables mientras no se establezca la duracion.
  - No esta calibrado contra postes ni contra estaciones locales.

Queda pendiente contrastarlo con una estacion DMC/DGA cercana con registro de
rafaga. Mientras tanto, reportarlo como propiedad del producto.

Y lo mas importante: si se usa `gust` directamente como variable de amenaza,
NO se la debe multiplicar por Gf. Seria contar la rafaga dos veces.

El archivo de salida Inputs/Hourlyhistoricdata.nc mantiene el contrato que
espera Hourly_climate_zone_humidity_daily_patch.py: variables t2m, wind, tp,
hur en dims (time, lat, lon). Las variables nuevas (gust, wind_dir) se agregan
sin romper nada aguas abajo.

Requisitos:
    pip install pandas requests xarray netcdf4 openpyxl
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import xarray as xr

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_START_DATE = "2017-01-01"   # IFS HRES parte en 2017
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_TIMEZONE = "America/Santiago"
DEFAULT_MODEL = "ecmwf_ifs"          # fijo, NO best_match (ver docstring)
DEFAULT_COORDS_FILE = "SystemzoneV1.xlsx"
DEFAULT_OUTPUT_NC = "Hourlyhistoricdata.nc"
DEFAULT_SLEEP_SECONDS = 1.0

# OJO: el default de la API es km/h. `wind_speed_unit=ms` es lo unico que
# fuerza m/s, y todo el resto de la cadena (umbral 20.8, el *3.6 del FFDI,
# el perfil de Hellmann) asume m/s. No quitar ese parametro.
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "precipitation",
]

RENOMBRE = {
    "temperature_2m": "t2m",           # degC
    "relative_humidity_2m": "hur",     # %
    "wind_speed_10m": "wind",          # m/s sostenido a 10 m
    "wind_gusts_10m": "gust",          # m/s rafaga a 10 m
    "wind_direction_10m": "wind_dir",  # grados
    "precipitation": "tp",             # mm/h
}

ATRIBUTOS = {
    "t2m": {"units": "degC", "description": "2 m air temperature (Open-Meteo Historical)"},
    "hur": {"units": "%", "description": "2 m relative humidity (Open-Meteo Historical)"},
    "wind": {"units": "m s-1", "description": "10 m sustained wind speed. REANALYSIS output (Open-Meteo Historical), not station data."},
    "gust": {"units": "m s-1", "description": "10 m wind gust, max of preceding hour. REANALYSIS output, not station data. Individual gust duration undocumented -- do NOT assume it matches the 1-minute convention of the reference paper."},
    "wind_dir": {"units": "degrees", "description": "10 m wind direction (Open-Meteo Historical)"},
    "tp": {"units": "mm", "description": "Hourly precipitation (Open-Meteo Historical)"},
}


@dataclass(frozen=True)
class ProjectDirs:
    root_dir: str
    input_dir: str
    output_dir: str
    coords_path: str


def resolve_project_dirs(coords_filename: str) -> ProjectDirs:
    from caso_clima import raiz_proyecto
    base_dir = raiz_proyecto()
    revisados: list[str] = []
    for root in [base_dir, os.path.abspath(os.path.join(base_dir, ".."))]:
        input_dir = os.path.join(root, "Inputs")
        output_dir = os.path.join(root, "Outputs")
        coords_path = os.path.join(input_dir, coords_filename)
        revisados.append(coords_path)
        if os.path.exists(coords_path):
            os.makedirs(output_dir, exist_ok=True)
            return ProjectDirs(root, input_dir, output_dir, coords_path)
    raise FileNotFoundError("No encontre el archivo de coordenadas. Revise:\n  - " + "\n  - ".join(revisados))


def read_coordinates(coords_path: str) -> pd.DataFrame:
    coords = pd.read_excel(coords_path)
    faltan = {"Lat", "Lon"} - set(coords.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en {coords_path}: {sorted(faltan)}")

    coords = coords.copy()
    coords["Lat"] = pd.to_numeric(coords["Lat"], errors="coerce")
    coords["Lon"] = pd.to_numeric(coords["Lon"], errors="coerce")
    coords = coords.dropna(subset=["Lat", "Lon"]).reset_index(drop=True)
    if coords.empty:
        raise ValueError("El archivo de coordenadas no tiene filas validas.")

    coords["point_id"] = np.arange(len(coords), dtype=int)
    if "Zona" in coords.columns:
        coords["zone_id"] = pd.to_numeric(coords["Zona"], errors="coerce").astype("Int64")
    else:
        coords["zone_id"] = coords["point_id"] + 1
    return coords


def descargar_punto(lat: float, lon: float, start_date: str, end_date: str,
                    timezone: str, model: str) -> tuple[pd.DataFrame, dict]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": timezone,
        "models": model,
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=300)
    r.raise_for_status()
    data = r.json()

    if "hourly" not in data:
        raise ValueError(f"Respuesta sin 'hourly' para lat={lat}, lon={lon}: {str(data)[:300]}")

    df = pd.DataFrame(data["hourly"])
    faltan = [v for v in HOURLY_VARIABLES if v not in df.columns]
    if faltan:
        raise ValueError(f"Open-Meteo no devolvio {faltan} para lat={lat}, lon={lon}, model={model}")

    df["time"] = pd.to_datetime(df["time"])
    df["Lat"] = lat
    df["Lon"] = lon

    # era5_land devuelve la estructura completa pero con viento todo nulo.
    # Sin este chequeo el fallo pasa desapercibido hasta mucho mas abajo.
    for v in ("wind_speed_10m", "wind_gusts_10m"):
        validos = int(df[v].notna().sum())
        if validos == 0:
            raise ValueError(
                f"El modelo '{model}' devolvio {v} completamente vacio para "
                f"lat={lat}, lon={lon}. Ese modelo no sirve para viento; use ecmwf_ifs.")
        if validos < 0.95 * len(df):
            print(f"   [!] {v}: solo {validos}/{len(df)} horas validas ({100*validos/len(df):.1f}%)")

    meta = {
        "lat_pedida": lat, "lon_pedida": lon,
        "lat_celda": data.get("latitude"), "lon_celda": data.get("longitude"),
        "elevacion_m": data.get("elevation"),
        "tz": data.get("timezone"), "modelo": model, "horas": len(df),
    }
    return df, meta


def factor_rafaga(df_punto: pd.DataFrame, percentil: float = 95.0) -> dict:
    """Cociente gust/wind condicionado a horas ventosas.

    DIAGNOSTICO DEL PRODUCTO, no calibracion local: la duracion de la rafaga
    de Open-Meteo no esta documentada, asi que este numero no es intercambiable
    con el 1.263 (1 minuto) del paper. Y si se usa `gust` directo como variable
    de amenaza, NO multiplicar por este factor.

    El cociente gust/wind se dispara cuando el sostenido tiende a cero, asi que
    promediarlo sobre todas las horas da un numero inflado y sin sentido fisico.
    Se condiciona a horas con sostenido >= percentil, que son las unicas que
    importan para eventos extremos.
    """
    w = df_punto["wind"].astype(float)
    g = df_punto["gust"].astype(float)
    umbral = np.percentile(w.dropna(), percentil)
    m = (w >= umbral) & (w > 0.5) & g.notna()
    ratio = (g[m] / w[m]).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "umbral_p95_ms": float(umbral),
        "horas_usadas": int(m.sum()),
        "Gf_medio": float(ratio.mean()),
        "Gf_mediana": float(ratio.median()),
        "Gf_p90": float(np.percentile(ratio, 90)),
        "wind_max": float(w.max()),
        "gust_max": float(g.max()),
    }


def homogeneidad(long_df, coords, umbral_ms: float = 20.8) -> pd.DataFrame:
    """Parte el registro en tercios y compara, por zona.

    Un salto grande entre tercios delata un cambio de modelo o de grilla dentro
    de la serie. Con best_match eso ocurre alrededor de 2017 (ERA5 -> IFS HRES)
    y llega a un factor 24x en horas sobre umbral: no es meteorologia.
    """
    filas = []
    for _, fila in coords.iterrows():
        z = int(fila["point_id"])
        p = long_df[long_df["point_id"] == z].sort_values("time")
        for etiqueta, trozo in zip(("T1", "T2", "T3"), np.array_split(p, 3)):
            w = trozo["wind"].astype(float).dropna()
            g = trozo["gust"].astype(float).dropna()
            if w.empty or g.empty:
                continue
            n = max(len(trozo) / 8760.0, 1e-9)
            filas.append({
                "zone_id": int(fila["zone_id"]), "tercio": etiqueta,
                "desde": trozo["time"].min().date(), "hasta": trozo["time"].max().date(),
                "media": w.mean(), "p99": np.percentile(w, 99),
                "gust_max": g.max(), "h_sobre_umbral_ano": (g > umbral_ms).sum() / n,
            })
    return pd.DataFrame(filas)


def construir_dataset(long_df: pd.DataFrame, coords: pd.DataFrame, model: str,
                      timezone: str, start_date: str, end_date: str) -> xr.Dataset:
    tiempos = np.array(sorted(long_df["time"].dropna().unique()), dtype="datetime64[ns]")
    lats = np.array(sorted(coords["Lat"].astype(float).unique()), dtype="float64")
    lons = np.array(sorted(coords["Lon"].astype(float).unique()), dtype="float64")
    idx_t = pd.Index(pd.to_datetime(tiempos), name="time")
    i_lat = {float(v): i for i, v in enumerate(lats)}
    i_lon = {float(v): i for i, v in enumerate(lons)}

    variables = ["t2m", "hur", "wind", "gust", "wind_dir", "tp"]
    data_vars = {}
    for var in variables:
        arr = np.full((len(tiempos), len(lats), len(lons)), np.nan, dtype="float32")
        for _, fila in coords.iterrows():
            la, lo = float(fila["Lat"]), float(fila["Lon"])
            p = long_df[(long_df["Lat"] == la) & (long_df["Lon"] == lo)]
            p = p.sort_values("time").set_index("time").reindex(idx_t)
            arr[:, i_lat[la], i_lon[lo]] = p[var].astype("float32").to_numpy()
        data_vars[var] = (("time", "lat", "lon"), arr, ATRIBUTOS[var])

    return xr.Dataset(
        data_vars=data_vars,
        coords={"time": ("time", tiempos), "lat": ("lat", lats), "lon": ("lon", lons)},
        attrs={
            "description": "Hourly reanalysis climate data with modelled gusts, for Hourly_climate_zone_*.py",
            "source": "Open-Meteo Historical Weather API",
            "model_requested": model,
            "timezone": timezone,
            "start_date": start_date,
            "end_date": end_date,
            "wind_units_note": "wind y gust en m/s (wind_speed_unit=ms). El default de la API es km/h.",
            "grid_note": "Solo los pares Lat-Lon pedidos estan llenos; el resto de la grilla es NaN.",
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Descarga historico Open-Meteo con rafagas.")
    p.add_argument("--coords-file", default=DEFAULT_COORDS_FILE)
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)
    p.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-nc", default=DEFAULT_OUTPUT_NC)
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dirs = resolve_project_dirs(args.coords_file)
    coords = read_coordinates(dirs.coords_path)

    print(f"Proyecto : {dirs.root_dir}")
    print(f"Coords   : {dirs.coords_path}")
    print(f"Periodo  : {args.start_date} a {args.end_date}")
    print(f"Puntos   : {len(coords)}")
    print()

    partes, metas, gf_filas = [], [], []
    for _, fila in coords.iterrows():
        z = int(fila["point_id"])
        la, lo = float(fila["Lat"]), float(fila["Lon"])
        zona = fila["zone_id"]
        print(f"Zona {zona} (punto {z + 1}/{len(coords)}): lat={la}, lon={lo}")
        df, meta = descargar_punto(la, lo, args.start_date, args.end_date, args.timezone, args.model)
        df = df.rename(columns=RENOMBRE)
        df["point_id"] = z
        partes.append(df)

        meta.update({"point_id": z, "zone_id": int(zona)})
        metas.append(meta)

        gf = factor_rafaga(df)
        gf.update({"point_id": z, "zone_id": int(zona), "Lat": la, "Lon": lo})
        gf_filas.append(gf)
        print(f"   celda devuelta: ({meta['lat_celda']}, {meta['lon_celda']})  elev {meta['elevacion_m']} m")
        print(f"   sostenido max {gf['wind_max']:.2f} m/s | rafaga max {gf['gust_max']:.2f} m/s | Gf {gf['Gf_medio']:.2f}")
        time.sleep(args.sleep)

    long_df = pd.concat(partes, ignore_index=True)

    # --- diagnostico de resolucion: zonas que comparten celda ---
    md = pd.DataFrame(metas)
    md["celda"] = list(zip(md["lat_celda"], md["lon_celda"]))
    colisiones = md.groupby("celda")["zone_id"].apply(list)
    colisiones = colisiones[colisiones.apply(len) > 1]
    print()
    if len(colisiones):
        print("*** AVISO DE RESOLUCION ***")
        for celda, zonas in colisiones.items():
            print(f"  Las zonas {zonas} caen en la MISMA celda {celda}: sus series seran identicas.")
        print("  El contraste entre esas zonas seria ficticio. Separe los puntos o use una fuente mas fina.")
    else:
        print(f"OK resolucion: las {len(md)} zonas caen en celdas distintas.")

    # --- diagnostico de homogeneidad temporal ---
    hg = homogeneidad(long_df, coords)
    print()
    print("HOMOGENEIDAD (tercios del registro) - un salto grande delata cambio de modelo:")
    print(hg.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    peor = 0.0
    for z, sub in hg.groupby("zone_id"):
        v = sub["h_sobre_umbral_ano"].to_numpy(float)
        if v.min() > 0:
            peor = max(peor, v.max() / v.min())
    if peor > 3.0:
        print(f"  *** AVISO: razon max/min entre tercios = {peor:.1f}x. Revise el modelo y el periodo.")
    else:
        print(f"  OK homogeneidad: razon max/min entre tercios = {peor:.1f}x.")

    # --- salidas ---
    ds = construir_dataset(long_df, coords, args.model, args.timezone, args.start_date, args.end_date)
    nc_path = os.path.join(dirs.input_dir, args.output_nc)
    ds.to_netcdf(nc_path)

    long_csv = os.path.join(dirs.output_dir, "Hourlyhistoricdata_openmeteo_long.csv")
    long_df.to_csv(long_csv, index=False)

    meta_csv = os.path.join(dirs.output_dir, "diagnostico_openmeteo_puntos.csv")
    md.drop(columns=["celda"]).to_csv(meta_csv, index=False)

    hg_csv = os.path.join(dirs.output_dir, "diagnostico_homogeneidad.csv")
    hg.to_csv(hg_csv, index=False)

    gf_csv = os.path.join(dirs.output_dir, "diagnostico_factor_rafaga.csv")
    gf_df = pd.DataFrame(gf_filas)
    gf_df.to_csv(gf_csv, index=False)

    print()
    print("COCIENTE gust/wind POR ZONA  [diagnostico del producto, NO calibracion local]")
    print("  No es intercambiable con el 1.263 del paper (ese es de 1 minuto).")
    print("  Si usas `gust` directo como amenaza, NO multipliques por este factor.")
    print(gf_df[["zone_id", "Lat", "Lon", "umbral_p95_ms", "Gf_medio", "Gf_mediana",
                 "wind_max", "gust_max"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print()
    print("Listo.")
    print(f"  NetCDF     : {nc_path}")
    print(f"  CSV largo  : {long_csv}")
    print(f"  Puntos     : {meta_csv}")
    print(f"  Factor Gf  : {gf_csv}")
    print(f"  Homogeneid.: {hg_csv}")


if __name__ == "__main__":
    main()
