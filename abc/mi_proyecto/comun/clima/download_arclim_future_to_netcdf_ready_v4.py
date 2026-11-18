#!/usr/bin/env python3
"""
Descarga variables futuras diarias desde la API ARClim y genera los archivos
que espera Hourly_climate_zone.py:

    Inputs/Dailyfuturedata.nc
    Inputs/Dailyfuturedata_relativehumidity.nc

Variables base desde ARClim daily_series:
    tasmax  -> temperatura maxima diaria [degC]
    tasmin  -> temperatura minima diaria [degC]
    pr      -> precipitacion diaria [mm/dia]
    huss    -> humedad especifica [g/kg]
    ps      -> presion superficial [hPa]

Viento:
    El documento actual de la API ARClim lista sfcWind como velocidad de viento.
    Este script queda robusto:
        --wind-source auto       intenta sfcWind; si falla, intenta uas+vas.
        --wind-source sfcWind    descarga sfcWind directamente.
        --wind-source components descarga uas y vas y calcula sfcWind = sqrt(uas^2 + vas^2).

Como Hourly_climate_zone.py requiere tas y hur:
    tas = (tasmax + tasmin) / 2
    hur = humedad relativa estimada desde huss, ps y tas

Compatibilidad con Hourly_climate_zone.py:
    Dailyfuturedata.nc queda diario con variables tas, tasmax, tasmin, sfcWind y pr.

    Dailyfuturedata_relativehumidity.nc queda diario, con estructura:
        time = días entre start-year y end-year
        lat  = índice de punto z en el mismo orden de SystemzoneV1.xlsx
        lon  = dimensión dummy [0, 1], con datos duplicados

    En este archivo, lat no representa latitud real: representa el punto z.
    Por eso Hourly_climate_zone.py debe leer humedad como:
        humf1 = Bfuthur['hur'].sel(time=f"{y}-{m:02d}-{d:02d}").values[z, 0]

    Se conserva la opción --humidity-file-mode compatible si se quiere mantener
    temporalmente el indexado legado .values[0, z, 0], pero no se recomienda.

Requisitos:
    pip install pandas requests xarray netcdf4 openpyxl
"""

from __future__ import annotations

import argparse
import io
import os
import time
from dataclasses import dataclass
from functools import reduce
from typing import Literal

import numpy as np
import pandas as pd
import requests
import xarray as xr


ARCLIM_DAILY_SERIES_URL = "https://arclim.mma.gob.cl/api/daily_series/{variable}/{scenario}/"
DEFAULT_SCENARIO = "ssp585"
DEFAULT_START_YEAR = 2025
DEFAULT_END_YEAR = 2040
DEFAULT_COORDS_FILE = "SystemzoneV1.xlsx"
DEFAULT_AGGREGATION: Literal["mean", "median"] = "mean"
DEFAULT_SLEEP_SECONDS = 0.5
DEFAULT_WIND_SOURCE = "auto"
DEFAULT_HUMIDITY_FILE_MODE = "daily"

BASE_ARCLIM_VARIABLES = ["tasmax", "tasmin", "pr", "huss", "ps"]


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


def request_arclim_daily_series(variable: str, scenario: str, lat: float, lon: float) -> pd.DataFrame:
    """Descarga el CSV de ARClim para una variable y punto."""
    url = ARCLIM_DAILY_SERIES_URL.format(variable=variable, scenario=scenario)
    response = requests.get(url, params={"lat": lat, "lon": lon}, timeout=240)
    response.raise_for_status()

    text = response.text.strip()
    if not text:
        raise ValueError(f"ARClim devolvio una respuesta vacia para {variable}, {scenario}, lat={lat}, lon={lon}")

    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.ParserError:
        df = pd.read_csv(io.StringIO(text), sep=";")

    if df.empty:
        raise ValueError(f"ARClim devolvio un CSV vacio para {variable}, {scenario}, lat={lat}, lon={lon}")

    return df


def normalize_arclim_csv(
    raw: pd.DataFrame,
    variable: str,
    start_year: int,
    end_year: int,
    aggregation: Literal["mean", "median"],
) -> pd.DataFrame:
    """Convierte el CSV de ARClim a columnas time, variable, variable_n_members."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    lower_map = {c.lower(): c for c in df.columns}
    if {"year", "month", "day"}.issubset(lower_map):
        year_col, month_col, day_col = lower_map["year"], lower_map["month"], lower_map["day"]
    elif {"ano", "mes", "dia"}.issubset(lower_map):
        year_col, month_col, day_col = lower_map["ano"], lower_map["mes"], lower_map["dia"]
    else:
        year_col, month_col, day_col = df.columns[:3]

    date_cols = [year_col, month_col, day_col]
    for col in date_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=date_cols).copy()
    df[year_col] = df[year_col].astype(int)
    df[month_col] = df[month_col].astype(int)
    df[day_col] = df[day_col].astype(int)
    df = df[(df[year_col] >= start_year) & (df[year_col] <= end_year)].copy()

    if df.empty:
        raise ValueError(f"No quedaron datos de {variable} para {start_year}-{end_year}.")

    member_cols = [c for c in df.columns if c not in date_cols]
    if not member_cols:
        raise ValueError(f"No encontre columnas de modelos/GCM para {variable}.")

    for col in member_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    values = df[member_cols]
    if aggregation == "mean":
        aggregated = values.mean(axis=1, skipna=True)
    elif aggregation == "median":
        aggregated = values.median(axis=1, skipna=True)
    else:
        raise ValueError(f"Agregacion no soportada: {aggregation}")

    out = pd.DataFrame(
        {
            "time": pd.to_datetime(
                {"year": df[year_col], "month": df[month_col], "day": df[day_col]},
                errors="coerce",
            ),
            variable: aggregated.astype("float32"),
            f"{variable}_n_members": values.notna().sum(axis=1).astype("int16"),
        }
    )
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return out


def download_one_variable(
    variable: str,
    scenario: str,
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    aggregation: Literal["mean", "median"],
) -> pd.DataFrame:
    raw = request_arclim_daily_series(variable, scenario, lat, lon)
    return normalize_arclim_csv(raw, variable, start_year, end_year, aggregation)


def download_wind(
    scenario: str,
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
    aggregation: Literal["mean", "median"],
    wind_source: Literal["auto", "sfcWind", "components"],
) -> tuple[pd.DataFrame, str]:
    """Devuelve una tabla con columna sfcWind, independiente de la fuente original."""
    errors: list[str] = []

    if wind_source in ("auto", "sfcWind"):
        try:
            wind = download_one_variable("sfcWind", scenario, lat, lon, start_year, end_year, aggregation)
            return wind, "sfcWind"
        except Exception as exc:
            errors.append(f"sfcWind fallo: {exc}")
            if wind_source == "sfcWind":
                raise

    if wind_source in ("auto", "components"):
        try:
            uas = download_one_variable("uas", scenario, lat, lon, start_year, end_year, aggregation)
            vas = download_one_variable("vas", scenario, lat, lon, start_year, end_year, aggregation)
            merged = pd.merge(uas, vas, on="time", how="inner")
            merged["sfcWind"] = np.sqrt(merged["uas"].astype(float) ** 2 + merged["vas"].astype(float) ** 2).astype("float32")
            # Se conserva trazabilidad en el CSV largo, pero Hourly_climate_zone.py usara sfcWind.
            merged["sfcWind_n_members"] = np.minimum(merged.get("uas_n_members", 0), merged.get("vas_n_members", 0)).astype("int16")
            return merged, "uas_vas_components"
        except Exception as exc:
            errors.append(f"uas/vas fallo: {exc}")
            raise RuntimeError("No fue posible descargar viento desde ARClim. " + " | ".join(errors)) from exc

    raise ValueError(f"wind_source no soportado: {wind_source}")


def download_point_future(
    lat: float,
    lon: float,
    scenario: str,
    start_year: int,
    end_year: int,
    aggregation: Literal["mean", "median"],
    sleep_seconds: float,
    wind_source: Literal["auto", "sfcWind", "components"],
) -> tuple[pd.DataFrame, str]:
    """Descarga todas las variables ARClim para un punto y las fusiona por fecha."""
    series: list[pd.DataFrame] = []
    for variable in BASE_ARCLIM_VARIABLES:
        print(f"  ARClim {variable}: lat={lat}, lon={lon}")
        clean = download_one_variable(variable, scenario, lat, lon, start_year, end_year, aggregation)
        series.append(clean)
        time.sleep(sleep_seconds)

    print(f"  ARClim viento ({wind_source}): lat={lat}, lon={lon}")
    wind, wind_used = download_wind(scenario, lat, lon, start_year, end_year, aggregation, wind_source)
    series.append(wind)
    time.sleep(sleep_seconds)

    merged = reduce(lambda left, right: pd.merge(left, right, on="time", how="inner"), series)
    merged["Lat"] = lat
    merged["Lon"] = lon
    return merged, wind_used


def relative_humidity_from_specific_humidity(q_gkg: np.ndarray, ps_hpa: np.ndarray, tas_c: np.ndarray) -> np.ndarray:
    """
    Convierte humedad especifica a humedad relativa.

    q_gkg : humedad especifica en g/kg
    ps_hpa: presion superficial en hPa
    tas_c : temperatura media diaria en degC
    """
    q = np.asarray(q_gkg, dtype=float) / 1000.0  # kg/kg
    p = np.asarray(ps_hpa, dtype=float)          # hPa
    t = np.asarray(tas_c, dtype=float)           # degC

    epsilon = 0.622
    vapor_pressure_hpa = (q * p) / (epsilon + (1.0 - epsilon) * q)
    saturation_vapor_pressure_hpa = 6.112 * np.exp((17.67 * t) / (t + 243.5))
    rh = 100.0 * vapor_pressure_hpa / saturation_vapor_pressure_hpa
    return np.clip(rh, 0.0, 100.0).astype("float32")


def build_future_datasets(
    all_points_df: pd.DataFrame,
    coords: pd.DataFrame,
    humidity_file_mode: Literal["compatible", "daily"] = "compatible",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Construye Dailyfuturedata.nc y Dailyfuturedata_relativehumidity.nc.

    humidity_file_mode="daily" genera el archivo correcto: humedad diaria
    con dimension lat usada como indice de punto z.
    humidity_file_mode="compatible" replica el valor diario 24 veces para que
    funcione el indexado legado .values[0, z, 0], pero no se recomienda.
    """
    df = all_points_df.copy()
    df["tas"] = ((df["tasmax"] + df["tasmin"]) / 2.0).astype("float32")
    df["hur"] = relative_humidity_from_specific_humidity(df["huss"].values, df["ps"].values, df["tas"].values)

    time_values_daily = np.array(sorted(df["time"].dropna().unique()), dtype="datetime64[ns]")
    time_index_daily = pd.Index(pd.to_datetime(time_values_daily), name="time")

    lat_values = np.array(sorted(coords["Lat"].astype(float).unique()), dtype="float64")
    lon_values = np.array(sorted(coords["Lon"].astype(float).unique()), dtype="float64")
    lat_to_i = {float(v): i for i, v in enumerate(lat_values)}
    lon_to_i = {float(v): i for i, v in enumerate(lon_values)}

    future_vars = ["tas", "tasmax", "tasmin", "sfcWind", "pr"]
    future_attrs = {
        "tas": {"units": "degC", "description": "Daily mean air temperature approximated as (tasmax + tasmin)/2"},
        "tasmax": {"units": "degC", "description": "Daily maximum air temperature from ARClim"},
        "tasmin": {"units": "degC", "description": "Daily minimum air temperature from ARClim"},
        "sfcWind": {"units": "m s-1", "description": "Daily wind speed from ARClim. If source is uas/vas, computed as sqrt(uas^2+vas^2)."},
        "pr": {"units": "mm day-1", "description": "Daily precipitation from ARClim"},
    }

    data_vars: dict[str, tuple[tuple[str, str, str], np.ndarray, dict[str, str]]] = {}
    for var in future_vars:
        arr = np.full((len(time_values_daily), len(lat_values), len(lon_values)), np.nan, dtype="float32")
        for _, row in coords.iterrows():
            lat = float(row["Lat"])
            lon = float(row["Lon"])
            point = df[(df["Lat"] == lat) & (df["Lon"] == lon)].sort_values("time").set_index("time").reindex(time_index_daily)
            arr[:, lat_to_i[lat], lon_to_i[lon]] = point[var].astype("float32").to_numpy()
        data_vars[var] = (("time", "lat", "lon"), arr, future_attrs[var])

    ds_future = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": ("time", time_values_daily),
            "lat": ("lat", lat_values),
            "lon": ("lon", lon_values),
        },
        attrs={
            "description": "Daily future climate data from ARClim formatted for Hourly_climate_zone.py",
            "source": "ARClim daily_series API",
            "note": "Only requested Lat-Lon pairs are filled; non-requested combinations may be NaN.",
        },
    )

    # Humedad futura: el dato climático es diario, pero Hourly_climate_zone.py
    # no selecciona por lat/lon reales, sino por índice de punto z:
    #     Bfuthur['hur'].sel(time='YYYY-MM-DD').values[0, z, 0]
    # Por eso la dimensión lat de este archivo NO son latitudes reales; son puntos
    # en el mismo orden del Excel SystemzoneV1.xlsx.
    # La dimensión lon es dummy y se deja con longitud 2 para reproducir el formato
    # antiguo que observaste (time, 20 puntos, 2). Se duplican los valores en lon=0 y lon=1.
    if humidity_file_mode == "compatible":
        # Aunque la humedad es diaria, se repite 24 veces por día para que la selección
        # parcial .sel(time='YYYY-MM-DD') retorne forma (24, n_puntos, 2) y no elimine time.
        hur_times = []
        for day in pd.to_datetime(time_values_daily):
            hur_times.extend(pd.date_range(day, periods=24, freq="h"))
        time_values_hur = np.array(hur_times, dtype="datetime64[ns]")
        repeat_factor = 24
        compatibility_note = (
            "Daily values repeated 24 times only to preserve the legacy index "
            ".sel(time='YYYY-MM-DD').values[0,z,0]. The climate variable remains daily."
        )
    elif humidity_file_mode == "daily":
        time_values_hur = time_values_daily
        repeat_factor = 1
        compatibility_note = (
            "Strictly daily file. This is not compatible with the unmodified line "
            ".sel(time='YYYY-MM-DD').values[0,z,0]; use .values[z,0] instead."
        )
    else:
        raise ValueError(f"humidity_file_mode no soportado: {humidity_file_mode}")

    hur_arr = np.full((len(time_values_hur), len(coords), 2), np.nan, dtype="float32")
    for _, row in coords.iterrows():
        point_id = int(row["point_id"])
        lat = float(row["Lat"])
        lon = float(row["Lon"])
        point = df[(df["Lat"] == lat) & (df["Lon"] == lon)].sort_values("time").set_index("time").reindex(time_index_daily)
        daily_hur = point["hur"].astype("float32").to_numpy()
        values = np.repeat(daily_hur, repeat_factor)
        hur_arr[:, point_id, 0] = values
        hur_arr[:, point_id, 1] = values

    ds_hur = xr.Dataset(
        data_vars={
            "hur": (("time", "lat", "lon"), hur_arr, {"units": "%", "description": "Daily relative humidity estimated from ARClim huss, ps and tas"})
        },
        coords={
            "time": ("time", time_values_hur),
            "lat": ("lat", np.arange(len(coords), dtype=int)),
            "lon": ("lon", np.array([0, 1], dtype=int)),
            "Lat_real": ("lat", coords["Lat"].astype(float).to_numpy()),
            "Lon_real": ("lat", coords["Lon"].astype(float).to_numpy()),
        },
        attrs={
            "description": "Relative humidity future data formatted for Hourly_climate_zone.py",
            "source": "ARClim daily_series API",
            "humidity_file_mode": humidity_file_mode,
            "compatibility_note": compatibility_note,
        },
    )

    return ds_future, ds_hur


def validate_for_hourly_script(ds_future: xr.Dataset, ds_hur: xr.Dataset, coords: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Prueba selecciones equivalentes a las de Hourly_climate_zone.py."""
    rows = []
    test_dates = pd.to_datetime([f"{start_year}-01-01", f"{end_year}-12-31"])

    for _, row in coords.iterrows():
        z = int(row["point_id"])
        lat = float(row["Lat"])
        lon = float(row["Lon"])
        for date in test_dates:
            date_str = date.strftime("%Y-%m-%d")
            out = {"point_id": z, "Lat": lat, "Lon": lon, "date": date_str}
            for var in ["tas", "tasmax", "tasmin", "sfcWind", "pr"]:
                try:
                    value = ds_future[var].sel(time=date_str, lat=lat, lon=lon, method="nearest").values
                    out[f"{var}_ok"] = bool(np.isfinite(value).all())
                except Exception:
                    out[f"{var}_ok"] = False
            try:
                selected_hur = ds_hur["hur"].sel(time=date_str)
                if "time" in selected_hur.dims:
                    value_hur = selected_hur.values[0, z, 0]
                    out["hur_legacy_index_ok"] = bool(np.isfinite(value_hur))
                else:
                    # Archivo estrictamente diario: sirve con .values[z, 0], no con .values[0, z, 0].
                    value_hur = selected_hur.values[z, 0]
                    out["hur_daily_index_ok"] = bool(np.isfinite(value_hur))
                    out["hur_legacy_index_ok"] = False
            except Exception:
                out["hur_legacy_index_ok"] = False
            rows.append(out)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga futuro ARClim listo para Hourly_climate_zone.py")
    parser.add_argument("--coords-file", default=DEFAULT_COORDS_FILE, help="Nombre del Excel dentro de Inputs/.")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO, choices=["ssp245", "ssp585", "ref"])
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--aggregation", default=DEFAULT_AGGREGATION, choices=["mean", "median"], help="Como agregar las columnas GCM de ARClim.")
    parser.add_argument("--wind-source", default=DEFAULT_WIND_SOURCE, choices=["auto", "sfcWind", "components"], help="Fuente del viento: auto, sfcWind o components (uas/vas).")
    parser.add_argument("--humidity-file-mode", default=DEFAULT_HUMIDITY_FILE_MODE, choices=["daily", "compatible"], help="daily genera humedad diaria indexada por punto z; compatible mantiene temporalmente el indexado legacy .values[0,z,0].")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="Pausa entre llamadas a la API.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = resolve_project_dirs(args.coords_file)
    coords = read_coordinates(dirs.coords_path)

    print(f"Proyecto detectado en: {dirs.root_dir}")
    print(f"Coordenadas: {dirs.coords_path}")
    print(f"Escenario ARClim: {args.scenario}")
    print(f"Periodo descargado: {args.start_year}-{args.end_year}")
    print(f"Agregacion GCM: {args.aggregation}")
    print(f"Fuente de viento: {args.wind_source}")
    print(f"Modo archivo humedad: {args.humidity_file_mode}")

    all_points: list[pd.DataFrame] = []
    wind_sources_used: list[dict] = []
    for _, row in coords.iterrows():
        point_id = int(row["point_id"])
        lat = float(row["Lat"])
        lon = float(row["Lon"])
        print(f"\nDescargando punto {point_id + 1}/{len(coords)}: lat={lat}, lon={lon}")
        point_df, wind_used = download_point_future(lat, lon, args.scenario, args.start_year, args.end_year, args.aggregation, args.sleep, args.wind_source)
        point_df["point_id"] = point_id
        point_df["wind_source_used"] = wind_used
        all_points.append(point_df)
        wind_sources_used.append({"point_id": point_id, "Lat": lat, "Lon": lon, "wind_source_used": wind_used})

    all_points_df = pd.concat(all_points, ignore_index=True)
    long_csv_path = os.path.join(dirs.output_dir, f"Dailyfuturedata_arclim_{args.scenario}_long.csv")
    all_points_df.to_csv(long_csv_path, index=False)

    wind_source_path = os.path.join(dirs.output_dir, f"diagnostico_arclim_wind_source_{args.scenario}.csv")
    pd.DataFrame(wind_sources_used).to_csv(wind_source_path, index=False)

    ds_future, ds_hur = build_future_datasets(all_points_df, coords, args.humidity_file_mode)
    for ds in [ds_future, ds_hur]:
        ds.attrs["scenario"] = args.scenario
        ds.attrs["start_year"] = args.start_year
        ds.attrs["end_year"] = args.end_year
        ds.attrs["aggregation"] = args.aggregation
        ds.attrs["wind_source_requested"] = args.wind_source
        ds.attrs["humidity_file_mode"] = args.humidity_file_mode

    future_nc_path = os.path.join(dirs.input_dir, "Dailyfuturedata.nc")
    hur_nc_path = os.path.join(dirs.input_dir, "Dailyfuturedata_relativehumidity.nc")
    ds_future.to_netcdf(future_nc_path)
    ds_hur.to_netcdf(hur_nc_path)

    validation = validate_for_hourly_script(ds_future, ds_hur, coords, args.start_year, args.end_year)
    validation_path = os.path.join(dirs.output_dir, f"diagnostico_arclim_compatibilidad_{args.scenario}.csv")
    validation.to_csv(validation_path, index=False)

    print("\nListo.")
    print(f"NetCDF futuro diario compatible guardado en: {future_nc_path}")
    print(f"NetCDF humedad compatible guardado en: {hur_nc_path}")
    print(f"CSV largo ARClim guardado en: {long_csv_path}")
    print(f"Diagnostico fuente viento: {wind_source_path}")
    print(f"Diagnostico de compatibilidad guardado en: {validation_path}")


if __name__ == "__main__":
    main()
