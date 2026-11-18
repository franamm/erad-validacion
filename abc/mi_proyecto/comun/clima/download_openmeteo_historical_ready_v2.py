#!/usr/bin/env python3
"""
Descarga datos horarios historicos desde Open-Meteo Historical API y genera
un NetCDF compatible con Hourly_climate_zone.py.

Archivo principal generado:
    Inputs/Hourlyhistoricdata.nc

Estructura esperada por Hourly_climate_zone.py:
    variables: t2m, wind, tp, hur
    dimensiones: time, lat, lon
    coordenadas: lat y lon reales desde Inputs/SystemzoneV1.xlsx

Nota importante:
    Se usa best_match por defecto, no era5_land, porque era5_land puede no
    entregar todas las variables de viento requeridas para este flujo.

Requisitos:
    pip install pandas requests xarray netcdf4 openpyxl
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import xarray as xr


OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_START_DATE = "2017-01-01"
DEFAULT_END_DATE = "2021-12-31"
DEFAULT_TIMEZONE = "America/Santiago"
DEFAULT_MODEL = "best_match"
DEFAULT_COORDS_FILE = "SystemzoneV1.xlsx"
DEFAULT_OUTPUT_NC = "Hourlyhistoricdata.nc"
DEFAULT_SLEEP_SECONDS = 1.0

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]

RENAME_TO_HOURLY_CLIMATE_ZONE = {
    "temperature_2m": "t2m",           # degC
    "relative_humidity_2m": "hur",    # %
    "wind_speed_10m": "wind",         # m/s, forzado con wind_speed_unit=ms
    "wind_direction_10m": "wind_dir", # grados, no lo usa Hourly_climate_zone.py
    "precipitation": "tp",            # mm/h
}


@dataclass(frozen=True)
class ProjectDirs:
    root_dir: str
    input_dir: str
    output_dir: str
    coords_path: str


def resolve_project_dirs(coords_filename: str) -> ProjectDirs:
    """Encuentra Inputs/ y Outputs/ tanto si el script vive en la raiz como en /Scripts."""
    from caso_clima import raiz_proyecto
    base_dir = raiz_proyecto()
    candidates = [base_dir, os.path.abspath(os.path.join(base_dir, ".."))]

    checked: list[str] = []
    for root in candidates:
        input_dir = os.path.join(root, "Inputs")
        output_dir = os.path.join(root, "Outputs")
        coords_path = os.path.join(input_dir, coords_filename)
        checked.append(coords_path)
        if os.path.exists(coords_path):
            os.makedirs(output_dir, exist_ok=True)
            return ProjectDirs(root, input_dir, output_dir, coords_path)

    raise FileNotFoundError(
        "No encontre el archivo de coordenadas. Revise:\n  - " + "\n  - ".join(checked)
    )


def read_coordinates(coords_path: str) -> pd.DataFrame:
    coords = pd.read_excel(coords_path)
    required_cols = {"Lat", "Lon"}
    missing = required_cols - set(coords.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {coords_path}: {sorted(missing)}")

    coords = coords.copy()
    coords["Lat"] = pd.to_numeric(coords["Lat"], errors="coerce")
    coords["Lon"] = pd.to_numeric(coords["Lon"], errors="coerce")
    coords = coords.dropna(subset=["Lat", "Lon"]).reset_index(drop=True)
    if coords.empty:
        raise ValueError("El archivo de coordenadas no tiene filas validas con Lat y Lon.")

    coords["point_id"] = np.arange(len(coords), dtype=int)
    return coords


def download_point(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timezone: str,
    model: str,
) -> tuple[pd.DataFrame, dict]:
    """Descarga una serie horaria para una coordenada."""
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

    response = requests.get(OPEN_METEO_URL, params=params, timeout=180)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data:
        raise ValueError(f"Respuesta sin campo 'hourly' para lat={lat}, lon={lon}: {data}")

    df = pd.DataFrame(data["hourly"])
    if "time" not in df.columns:
        raise ValueError(f"Respuesta sin columna time para lat={lat}, lon={lon}")

    missing_vars = [v for v in HOURLY_VARIABLES if v not in df.columns]
    if missing_vars:
        raise ValueError(
            "Open-Meteo no devolvio todas las variables solicitadas para "
            f"lat={lat}, lon={lon}, model={model}. Faltan: {missing_vars}"
        )

    df["time"] = pd.to_datetime(df["time"])
    df["Lat"] = lat
    df["Lon"] = lon

    metadata = {
        "lat_input": lat,
        "lon_input": lon,
        "lat_returned": data.get("latitude"),
        "lon_returned": data.get("longitude"),
        "elevation_m": data.get("elevation"),
        "timezone_returned": data.get("timezone"),
        "utc_offset_seconds": data.get("utc_offset_seconds"),
        "model_requested": model,
        "rows": len(df),
    }
    return df, metadata


def build_grid_dataset(long_df: pd.DataFrame, coords: pd.DataFrame, model: str, timezone: str, start_date: str, end_date: str) -> xr.Dataset:
    """Convierte series por punto a grilla lat-lon compatible con .sel(lat=..., lon=...)."""
    required = {"time", "Lat", "Lon", "t2m", "hur", "wind", "tp"}
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"Faltan columnas para construir el NetCDF historico: {sorted(missing)}")

    time_values = np.array(sorted(long_df["time"].dropna().unique()), dtype="datetime64[ns]")
    lat_values = np.array(sorted(coords["Lat"].astype(float).unique()), dtype="float64")
    lon_values = np.array(sorted(coords["Lon"].astype(float).unique()), dtype="float64")

    time_index = pd.Index(pd.to_datetime(time_values), name="time")
    lat_to_i = {float(v): i for i, v in enumerate(lat_values)}
    lon_to_i = {float(v): i for i, v in enumerate(lon_values)}

    var_attrs = {
        "t2m": {"units": "degC", "description": "2 m air temperature from Open-Meteo Historical API"},
        "hur": {"units": "%", "description": "2 m relative humidity from Open-Meteo Historical API"},
        "wind": {"units": "m s-1", "description": "10 m wind speed from Open-Meteo Historical API"},
        "tp": {"units": "mm", "description": "Hourly precipitation from Open-Meteo Historical API"},
        "wind_dir": {"units": "degrees", "description": "10 m wind direction from Open-Meteo Historical API"},
    }

    variables = ["t2m", "hur", "wind", "tp"]
    if "wind_dir" in long_df.columns:
        variables.append("wind_dir")

    data_vars: dict[str, tuple[tuple[str, str, str], np.ndarray, dict[str, str]]] = {}
    for var in variables:
        arr = np.full((len(time_values), len(lat_values), len(lon_values)), np.nan, dtype="float32")
        for _, row in coords.iterrows():
            lat = float(row["Lat"])
            lon = float(row["Lon"])
            point = long_df[(long_df["Lat"] == lat) & (long_df["Lon"] == lon)]
            point = point.sort_values("time").set_index("time").reindex(time_index)
            arr[:, lat_to_i[lat], lon_to_i[lon]] = point[var].astype("float32").to_numpy()
        data_vars[var] = (("time", "lat", "lon"), arr, var_attrs.get(var, {}))

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", time_values),
            "lat": ("lat", lat_values),
            "lon": ("lon", lon_values),
        },
        attrs={
            "description": "Hourly historical climate data formatted for Hourly_climate_zone.py",
            "source": "Open-Meteo Historical Weather API",
            "model_requested": model,
            "timezone": timezone,
            "start_date": start_date,
            "end_date": end_date,
            "note": "Only requested Lat-Lon pairs are filled; non-requested combinations may be NaN.",
        },
    )


def check_requested_pairs(ds: xr.Dataset, coords: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in coords.iterrows():
        lat = float(row["Lat"])
        lon = float(row["Lon"])
        out = {"point_id": int(row["point_id"]), "Lat": lat, "Lon": lon}
        for var in ["t2m", "hur", "wind", "tp"]:
            series = ds[var].sel(lat=lat, lon=lon, method="nearest")
            out[f"{var}_valid_values"] = int(np.isfinite(series.values).sum())
            out[f"{var}_nan_values"] = int(np.isnan(series.values).sum())
        rows.append(out)
    return pd.DataFrame(rows)


def compare_profiles(ds: xr.Dataset, coords: pd.DataFrame, variables: Iterable[str]) -> pd.DataFrame:
    rows = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            lat_i, lon_i = float(coords.loc[i, "Lat"]), float(coords.loc[i, "Lon"])
            lat_j, lon_j = float(coords.loc[j, "Lat"]), float(coords.loc[j, "Lon"])
            row = {"point_i": i, "point_j": j, "Lat_i": lat_i, "Lon_i": lon_i, "Lat_j": lat_j, "Lon_j": lon_j}
            for var in variables:
                a = ds[var].sel(lat=lat_i, lon=lon_i, method="nearest").values.astype(float)
                b = ds[var].sel(lat=lat_j, lon=lon_j, method="nearest").values.astype(float)
                row[f"{var}_mean_abs_diff"] = float(np.nanmean(np.abs(a - b)))
                row[f"{var}_allclose"] = bool(np.allclose(a, b, equal_nan=True))
            rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga historico Open-Meteo listo para Hourly_climate_zone.py")
    parser.add_argument("--coords-file", default=DEFAULT_COORDS_FILE, help="Nombre del Excel dentro de Inputs/.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Por defecto best_match para asegurar variables de viento.")
    parser.add_argument("--output-nc", default=DEFAULT_OUTPUT_NC, help="Nombre del NetCDF a guardar en Inputs/.")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="Pausa entre llamadas a la API.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = resolve_project_dirs(args.coords_file)
    coords = read_coordinates(dirs.coords_path)

    print(f"Proyecto detectado en: {dirs.root_dir}")
    print(f"Coordenadas: {dirs.coords_path}")
    print(f"Modelo Open-Meteo solicitado: {args.model}")
    print(f"Puntos a descargar: {len(coords)}")

    all_points: list[pd.DataFrame] = []
    metadata_rows: list[dict] = []

    for _, row in coords.iterrows():
        point_id = int(row["point_id"])
        lat = float(row["Lat"])
        lon = float(row["Lon"])
        print(f"Descargando punto {point_id + 1}/{len(coords)}: lat={lat}, lon={lon}")
        df, metadata = download_point(lat, lon, args.start_date, args.end_date, args.timezone, args.model)
        df["point_id"] = point_id
        all_points.append(df)
        metadata["point_id"] = point_id
        metadata_rows.append(metadata)
        time.sleep(args.sleep)

    long_df = pd.concat(all_points, ignore_index=True)
    long_df = long_df.rename(columns=RENAME_TO_HOURLY_CLIMATE_ZONE)

    long_csv_path = os.path.join(dirs.output_dir, "Hourlyhistoricdata_openmeteo_long.csv")
    long_df.to_csv(long_csv_path, index=False)

    ds = build_grid_dataset(long_df, coords, args.model, args.timezone, args.start_date, args.end_date)
    nc_path = os.path.join(dirs.input_dir, args.output_nc)
    ds.to_netcdf(nc_path)

    metadata_path = os.path.join(dirs.output_dir, "diagnostico_openmeteo_puntos.csv")
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)

    check_path = os.path.join(dirs.output_dir, "diagnostico_openmeteo_validez_por_punto.csv")
    check_requested_pairs(ds, coords).to_csv(check_path, index=False)

    diff_path = os.path.join(dirs.output_dir, "diagnostico_openmeteo_diferencias_entre_puntos.csv")
    compare_profiles(ds, coords, ["t2m", "hur", "wind", "tp"]).to_csv(diff_path, index=False)

    print("\nListo.")
    print(f"NetCDF historico compatible guardado en: {nc_path}")
    print(f"CSV largo guardado en: {long_csv_path}")
    print(f"Diagnostico de puntos Open-Meteo: {metadata_path}")
    print(f"Diagnostico de valores validos: {check_path}")
    print(f"Diagnostico de diferencias entre puntos: {diff_path}")


if __name__ == "__main__":
    main()
