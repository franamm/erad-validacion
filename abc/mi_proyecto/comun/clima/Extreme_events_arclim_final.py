#!/usr/bin/env python3
"""
Extreme_events_arclim_sin_olas_sequia.py

Version modificada de Extreme_events.py para trabajar con la nueva data futura
procesada desde ARClim, manteniendo el flujo del paper:

1) Lee la base horaria generada por Hourly_climate_zone.py:
      BaseFhorPELP4v20z2anos.nc

2) Lee la data futura diaria procesada desde ARClim:
      Dailyfuturedata.nc
      Dailyfuturedata_relativehumidity.nc

3) Deja de depender de los archivos AWI originales:
      tasmax_day_AWI-CM-1-1-MR_ssp585.nc
      sfcWind_day_AWI-CM-1-1-MR_ssp585.nc
      pr_day_AWI-CM-1-1-MR_ssp585.nc

4) Genera:
      BaseHoraEWEPELP20z.nc
      EWEwind_hora_anual.npy
      resumen_EWE_arclim.csv

Nota:
- Esta version elimina la aplicacion de olas de calor, olas de frio y sequia.
- Mantiene viento extremo, baja humedad relativa, FFDI e incendios.

Supuestos de formato ARClim procesado:
- Dailyfuturedata.nc contiene: tas, tasmax, tasmin, sfcWind, pr
  con dimensiones time, lat, lon.
- Dailyfuturedata_relativehumidity.nc contiene: hur
  con dimensiones time, lat, lon, donde lat representa el indice de punto z
  y lon es una dimension dummy [0, 1].

Requisitos:
    pip install numpy pandas xarray netcdf4 openpyxl
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


# -----------------------------
# Configuracion por defecto
# -----------------------------
DEFAULT_START_YEAR = 2029
DEFAULT_END_YEAR = 2030
DEFAULT_BASE_HOURLY_FILE = "BaseFhorPELP4v20z2anos.nc"
DEFAULT_FUTURE_DAILY_FILE = "Dailyfuturedata.nc"
DEFAULT_FUTURE_HUMIDITY_FILE = "Dailyfuturedata_relativehumidity.nc"
DEFAULT_COORDS_FILE = "SystemzoneV1.xlsx"
DEFAULT_OUTPUT_FILE = "BaseHoraEWEPELP20z.nc"


@dataclass(frozen=True)
class ProjectDirs:
    root_dir: str
    input_dir: str
    output_dir: str


def resolve_project_dirs(project_root: str | None = None) -> ProjectDirs:
    """Resuelve Inputs/ y Outputs/.

    Prioridad:
    1) --project-root/Inputs y --project-root/Outputs
    2) carpeta del script/Inputs y carpeta del script/Outputs
    3) carpeta madre/Inputs y carpeta madre/Outputs
    """
    candidates: list[str] = []

    if project_root:
        candidates.append(os.path.abspath(project_root))
    else:
        from caso_clima import raiz_proyecto
        base_dir = raiz_proyecto()
        candidates.extend([base_dir, os.path.abspath(os.path.join(base_dir, ".."))])

    checked = []
    for root in candidates:
        input_dir = os.path.join(root, "Inputs")
        output_dir = os.path.join(root, "Outputs")
        checked.append(input_dir)
        if os.path.isdir(input_dir):
            os.makedirs(output_dir, exist_ok=True)
            return ProjectDirs(root, input_dir, output_dir)

    raise FileNotFoundError(
        "No encontre carpeta Inputs/. Revise estas rutas:\n  - " + "\n  - ".join(checked)
    )


def find_file(filename: str, dirs: ProjectDirs, prefer: Iterable[str] = ("Inputs", "Outputs")) -> str:
    """Busca un archivo en Inputs y/o Outputs."""
    mapping = {"Inputs": dirs.input_dir, "Outputs": dirs.output_dir}
    checked = []
    for key in prefer:
        folder = mapping[key]
        path = os.path.join(folder, filename)
        checked.append(path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No encontre el archivo requerido:\n  - " + "\n  - ".join(checked))


def find_optional_file(filename: str, dirs: ProjectDirs, prefer: Iterable[str] = ("Inputs", "Outputs")) -> str | None:
    """Busca un archivo opcional. Si no existe, devuelve None."""
    mapping = {"Inputs": dirs.input_dir, "Outputs": dirs.output_dir}
    for key in prefer:
        path = os.path.join(mapping[key], filename)
        if os.path.exists(path):
            return path
    return None


def read_coordinates(path: str) -> pd.DataFrame:
    """Lee SystemzoneV1.xlsx preservando el orden original de las filas.

    Importante:
    - point_id = posición 0..N-1 usada internamente por los arreglos numpy.
    - zone_id = identificador real de la zona, tomado desde la columna "Zona" si existe.
      Si no existe, se usa point_id.

    Así se evita confundir la posición de columna con el número/nombre real de zona.
    """
    coords = pd.read_excel(path)
    required = {"Lat", "Lon"}
    missing = required - set(coords.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")

    coords = coords.copy()
    coords["original_row"] = np.arange(len(coords), dtype=int)
    coords["Lat"] = pd.to_numeric(coords["Lat"], errors="coerce")
    coords["Lon"] = pd.to_numeric(coords["Lon"], errors="coerce")

    # dropna preserva el orden original. reset_index solo compacta la posición interna.
    coords = coords.dropna(subset=["Lat", "Lon"]).reset_index(drop=True)
    coords["point_id"] = np.arange(len(coords), dtype=int)

    if "Zona" in coords.columns:
        coords["zone_id"] = pd.to_numeric(coords["Zona"], errors="coerce")
        if coords["zone_id"].isna().any():
            raise ValueError("La columna 'Zona' contiene valores no numéricos o vacíos.")
        coords["zone_id"] = coords["zone_id"].astype(int)
    else:
        coords["zone_id"] = coords["point_id"].astype(int)

    return coords


def to_celsius(values: np.ndarray) -> np.ndarray:
    """Convierte de K a degC solo si los valores parecen estar en Kelvin."""
    arr = np.asarray(values, dtype="float32")
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmedian(finite) > 100:
        return arr - np.float32(273.15)
    return arr


def precipitation_to_mm_day(da: xr.DataArray) -> xr.DataArray:
    """Convierte precipitación a mm/día solo si los atributos indican flujo.

    La data ARClim procesada por el script anterior debería venir como mm/day.
    Los archivos CMIP6 crudos suelen venir como kg m-2 s-1 y requieren *86400.
    """
    units = str(da.attrs.get("units", "")).lower().replace(" ", "")
    if ("kg" in units and "s-1" in units) or ("kgm-2s-1" in units):
        return da * 86400.0
    return da


def select_lon_for_dataset(ds: xr.Dataset | xr.DataArray, lon: float) -> float:
    """Ajusta longitud si el archivo está en 0..360 y el punto viene en -180..180."""
    if "lon" not in ds.coords:
        return lon
    lon_values = ds["lon"].values
    if np.nanmax(lon_values) > 180 and lon < 0:
        return lon + 360
    return lon


def select_daily_series(
    ds: xr.Dataset,
    var: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> xr.DataArray:
    """Selecciona serie diaria para una variable y punto usando nearest."""
    if var not in ds:
        raise KeyError(f"La variable {var!r} no existe en {list(ds.data_vars)}")

    lon_sel = select_lon_for_dataset(ds, lon)
    da = ds[var].sel(lat=lat, lon=lon_sel, method="nearest")
    da = da.sel(time=slice(start_date, end_date))
    return da


def read_future_humidity_daily(
    ds_hur: xr.Dataset,
    z: int,
    start_date: str,
    end_date: str,
) -> xr.DataArray:
    """Lee humedad futura diaria por indice de punto z.

    Soporta dos formatos:
    - daily: time diario, hur(time, lat=punto, lon=dummy)
    - compatible: time horario con valor diario repetido 24 veces
    """
    if "hur" not in ds_hur:
        raise KeyError(f"No existe variable 'hur'. Variables disponibles: {list(ds_hur.data_vars)}")

    da = ds_hur["hur"].sel(time=slice(start_date, pd.Timestamp(end_date) + pd.Timedelta(hours=23)))
    point = da.isel(lat=z, lon=0)

    # Si viene horario compatible, bajar a diario usando el primer valor del día.
    if point.sizes.get("time", 0) > (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1:
        point = point.resample(time="1D").first()

    point = point.sel(time=slice(start_date, end_date))
    point = point.clip(min=0, max=100)
    return point


def get_hourly_baseline(base_ds: xr.Dataset, var: str) -> np.ndarray:
    """Extrae lon=0 desde BaseFhorPELP4v20z2anos.nc."""
    if var not in base_ds:
        raise KeyError(f"No existe variable {var!r} en BaseFhor. Variables: {list(base_ds.data_vars)}")
    arr = base_ds[var].values
    if arr.ndim != 3:
        raise ValueError(f"{var} debe tener 3 dimensiones, pero tiene shape {arr.shape}")
    return arr[:, :, 0]


def validate_lengths(name: str, values: np.ndarray, expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} tiene largo {len(values)}, pero se esperaban {expected} días.")


def build_daily_future_arrays(
    ds_future: xr.Dataset,
    ds_hur: xr.Dataset,
    coords: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construye Tmax, viento, precipitación y humedad como matrices día x zona."""
    n_days = len(pd.date_range(start_date, end_date, freq="D"))
    zonas = len(coords)

    tmax = np.zeros((n_days, zonas), dtype="float32")
    wind = np.zeros((n_days, zonas), dtype="float32")
    prec = np.zeros((n_days, zonas), dtype="float32")
    hur = np.zeros((n_days, zonas), dtype="float32")

    for _, row in coords.iterrows():
        z = int(row["point_id"])
        lat = float(row["Lat"])
        lon = float(row["Lon"])

        s_tmax = select_daily_series(ds_future, "tasmax", lat, lon, start_date, end_date)
        s_wind = select_daily_series(ds_future, "sfcWind", lat, lon, start_date, end_date)
        s_prec = precipitation_to_mm_day(select_daily_series(ds_future, "pr", lat, lon, start_date, end_date))
        s_hur = read_future_humidity_daily(ds_hur, z, start_date, end_date)

        validate_lengths(f"tasmax punto {z}", s_tmax.values, n_days)
        validate_lengths(f"sfcWind punto {z}", s_wind.values, n_days)
        validate_lengths(f"pr punto {z}", s_prec.values, n_days)
        validate_lengths(f"hur punto {z}", s_hur.values, n_days)

        tmax[:, z] = to_celsius(s_tmax.values)
        wind[:, z] = np.asarray(s_wind.values, dtype="float32")
        prec[:, z] = np.asarray(s_prec.values, dtype="float32")
        hur[:, z] = np.clip(np.asarray(s_hur.values, dtype="float32"), 0, 100)

    return tmax, wind, prec, hur


def load_threshold_array(path: str, name: str, zonas: int) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{name} debe ser matriz 2D, pero tiene shape {arr.shape}")
    if arr.shape[1] != zonas:
        raise ValueError(
            f"{name} tiene {arr.shape[1]} zonas, pero SystemzoneV1.xlsx tiene {zonas}. "
            "Debes recalcular estos umbrales para tus nuevos puntos."
        )
    return arr.astype("float32")


def load_optional_threshold_array(path: str | None, name: str, zonas: int) -> np.ndarray | None:
    """Carga umbrales si existen. Si no existen, permite ejecutar sin ese EWE."""
    if path is None:
        print(f"Aviso: no se encontro {name}; se omitira ese tipo de evento si depende de este archivo.")
        return None
    return load_threshold_array(path, name, zonas)


def threshold_at(arr: np.ndarray, d: int, z: int) -> float:
    """Devuelve umbral para dia d y zona z, soportando arrays anuales o multianuales."""
    if arr.shape[0] == 0:
        raise ValueError("El arreglo de umbrales esta vacio.")
    idx = d if d < arr.shape[0] else d % arr.shape[0]
    return float(arr[idx, z])


def main() -> None:
    args = parse_args()
    dirs = resolve_project_dirs(args.project_root)

    input_coords = find_file(args.coords_file, dirs, prefer=("Inputs",))
    base_path = find_file(args.base_hourly_file, dirs, prefer=("Inputs", "Outputs"))
    future_path = find_file(args.future_daily_file, dirs, prefer=("Inputs",))
    hur_path = find_file(args.future_humidity_file, dirs, prefer=("Inputs",))


    coords = read_coordinates(input_coords)
    zonas = len(coords)
    la = coords["Lat"].astype(float).to_numpy()
    lo = coords["Lon"].astype(float).to_numpy()
    zone_id = coords["zone_id"].astype(int).to_numpy()
    point_id = coords["point_id"].astype(int).to_numpy()
    original_row = coords["original_row"].astype(int).to_numpy()

    print(f"Proyecto: {dirs.root_dir}")
    print(f"Input coords: {input_coords}")
    print(f"Base horaria: {base_path}")
    print(f"Futuro diario ARClim: {future_path}")
    print(f"Humedad diaria ARClim: {hur_path}")
    print(f"Zonas/puntos: {zonas}")

    Baseproyec = xr.open_dataset(base_path)
    Bfut = xr.open_dataset(future_path)
    Bhumr = xr.open_dataset(hur_path)


    study_start = pd.Timestamp(f"{args.start_year}-01-01")
    study_end = pd.Timestamp(f"{args.end_year}-12-31")
    start_date = study_start.strftime("%Y-%m-%d")
    end_date = study_end.strftime("%Y-%m-%d")

    # Base horaria generada por Hourly_climate_zone.py
    t2mfut = to_celsius(get_hourly_baseline(Baseproyec, "tem"))
    windfut = get_hourly_baseline(Baseproyec, "wind")
    tpfut = get_hourly_baseline(Baseproyec, "prec")
    hurfut = np.clip(get_hourly_baseline(Baseproyec, "hur"), 0, 100)

    horastot = t2mfut.shape[0]
    diastot = len(pd.date_range(start_date, end_date, freq="D"))
    expected_hours = diastot * 24
    if horastot != expected_hours:
        raise ValueError(
            f"La base horaria tiene {horastot} horas, pero el periodo {start_date} a {end_date} "
            f"requiere {expected_hours}. Ajusta --start-year/--end-year o regenera BaseFhor."
        )
    if t2mfut.shape[1] != zonas:
        raise ValueError(
            f"BaseFhor tiene {t2mfut.shape[1]} zonas, pero SystemzoneV1 tiene {zonas}."
        )

    Tmax, Vavg, Ptot, Hur = build_daily_future_arrays(Bfut, Bhumr, coords, start_date, end_date)

    # Base diaria: 0=temp max, 1=wind, 2=prec, 3=hur, 4=SMD, 5=DF, 6=FFDI, 7=Fire
    Basefull = np.zeros((diastot, 8, zonas), dtype="float32")
    Basefull[:, 0, :] = Tmax
    Basefull[:, 1, :] = Vavg
    Basefull[:, 2, :] = Ptot
    Basefull[:, 3, :] = Hur

    # Base horaria: 0=temp, 1=wind, 2=prec, 3=hur, 4=FFDI, 5=Fire
    varfintot = 6
    subesctot = 2  # 0 = tendencia sin EWE adicional, 1 = con EWE
    Basefullhora = np.zeros((horastot, varfintot, subesctot, zonas), dtype="float32")
    Basefullhora[:, 0, 0, :] = t2mfut
    Basefullhora[:, 1, 0, :] = windfut
    Basefullhora[:, 2, 0, :] = tpfut
    Basefullhora[:, 3, 0, :] = hurfut
    Basefullhora[:, :, 1, :] = Basefullhora[:, :, 0, :].copy()

    # Umbrales y factores EWE
    VVum = args.wind_threshold / args.wind_gust_factor
    Hum = args.humidity_threshold
    windEVE = args.wind_gust_factor
    hurEVE = args.humidity_scaling
    Fire_threshold = args.ffdi_threshold

    # Para reportes: se cuenta separadamente el primer año y segundo año.
    # En el paper usualmente se reporta el segundo año del horizonte 2029-2030.
    diabase = 365 if diastot >= 730 else 0
    iniEWE = diabase

    diavient = np.zeros(zonas, dtype="float32")
    vientno = np.zeros(zonas, dtype="float32")
    vientnohor = np.zeros(zonas, dtype="float32")
    diahumr = np.zeros(zonas, dtype="float32")
    humrno = np.zeros(zonas, dtype="float32")

    # Procesamiento por zona
    for z in range(zonas):
        for d in range(diastot):
            # SMD, DF y FFDI diario
            if d > 365:
                R = np.sum(Basefull[(d - 365):d, 2, z])
            else:
                R = np.sum(Basefull[0:min(365, diastot), 2, z])

            prev_smd = Basefull[d - 1, 4, z] if d > 0 else 0.0
            a = (203.2 - prev_smd) * (0.968 * np.exp(0.0875 * Basefull[d, 0, z] + 1.552) - 8.30) / 1000
            b = 1 + 10.88 * np.exp(-0.00173 * R)
            Peff = Basefull[d, 2, z] if Basefull[d, 2, z] > 5 else 0
            SMD = max(prev_smd - Peff + a / b, 0)
            Basefull[d, 4, z] = SMD

            if d >= 20:
                rain_window = Basefull[(d - 19):(d + 1), 2, z]
            else:
                rain_window = Basefull[0:min(20, diastot), 2, z]
            Pacum = np.sum(rain_window)
            if Pacum > 2:
                N = len(rain_window) - 1 - int(np.argmax(rain_window))
            else:
                N = -1
            if Pacum >= 2 and N >= 1:
                x = (N ** 1.3) / (N ** 1.3 + Pacum - 2)
            elif Pacum >= 2 and N == 0:
                x = (0.8 ** 1.3) / (0.8 ** 1.3 + Pacum - 2)
            else:
                x = 1

            DF = max(10.5 * (1 - np.exp((-SMD + 30) / 40)) * (41 * x ** 2 + x) / (40 * x ** 2 + x + 1), 0)
            Basefull[d, 5, z] = DF
            Basefull[d, 3, z] = min(Basefull[d, 3, z], 100)

            if DF > 0:
                FFDI = 2 * np.exp(
                    -0.45
                    + 0.987 * np.log(DF)
                    - 0.0345 * Basefull[d, 3, z]
                    + 0.0338 * Basefull[d, 0, z]
                    + 0.0234 * Basefull[d, 1, z] * 3.6
                )
            else:
                FFDI = 0.0
            Basefull[d, 6, z] = FFDI
            Basefullhora[d * 24:(d * 24 + 24), 4, :, z] = FFDI

            # Viento extremo horario
            wind_hours_mask = Basefullhora[d * 24:(d * 24 + 24), 1, 0, z] > VVum
            if np.any(wind_hours_mask):
                wind_day = Basefullhora[d * 24:(d * 24 + 24), 1, 1, z].copy()
                wind_day[wind_hours_mask] *= windEVE
                Basefullhora[d * 24:(d * 24 + 24), 1, 1, z] = wind_day
                Basefull[d, 1, z] = np.mean(wind_day)
                if d >= diabase:
                    diavient[z] += 1
                    vientno[z] += 1
                    vientnohor[z] += int(np.sum(wind_hours_mask))

            # Baja humedad relativa
            if Hur[d, z] < Hum:
                diahumr[z] += 1
                if d >= diabase:
                    humrno[z] += 1
                Basefull[d, 3, z] *= hurEVE
                Basefullhora[d * 24:(d * 24 + 24), 3, 1, z] *= hurEVE

            # Incendios
            if Basefull[d, 6, z] > Fire_threshold:
                Basefull[d, 7, z] = 1
                horaFIRE = int(np.argmax(Basefullhora[d * 24:(d * 24 + 24), 0, 1, z]))
                Basefullhora[d * 24 + horaFIRE, 5, 1, z] = 1
                if d >= iniEWE:
                    print(f"Fire event at zone {z}, day {d - iniEWE}, hour {horaFIRE}")

        print(f"Procesada zona {z + 1}/{zonas}")

    # Resumen por punto/zona
    summary = pd.DataFrame({
        "point_id": point_id,
        "zone_id": zone_id,
        "original_row_SystemzoneV1": original_row,
        "Lat": la,
        "Lon": lo,
        "extreme_wind_days": diavient,
        "extreme_wind_hours": vientnohor,
        "low_humidity_days": humrno,
        "fire_days": np.sum(Basefull[diabase:diastot, 7, :], axis=0),
        "fire_hours": np.sum(Basefullhora[diabase * 24:diastot * 24, 5, 1, :], axis=0),
        "max_temperature_EWE_C": np.max(Basefullhora[diabase * 24:diastot * 24, 0, 1, :], axis=0),
        "max_wind_EWE_ms": np.max(Basefullhora[diabase * 24:diastot * 24, 1, 1, :], axis=0),
    })

    print("\nResumen EWE por zona:")
    print(summary)

    # Salida: se conserva la convencion de BaseFhor:
    # dims time, lat, lon, donde lat = indice de punto y lon = escenario [0,1].
    # Por eso se transpone desde (time, escenario, zona) a (time, zona, escenario).
    out_time = pd.date_range(start_date, freq="h", periods=horastot)

    # Mantener el eje en el MISMO ORDEN que SystemzoneV1.xlsx.
    # La coordenada lat sigue siendo point_id para compatibilidad con BaseFhor/legado,
    # y zone_id guarda el identificador real de zona.
    latx = point_id.astype(int)
    lonx = np.array([0, 1], dtype=int)

    ds = xr.Dataset(
        data_vars=dict(
            tem=(("time", "lat", "lon"), np.float32(np.transpose(Basefullhora[:, 0, :, :], (0, 2, 1))), {"units": "degC"}),
            wind=(("time", "lat", "lon"), np.float32(np.transpose(Basefullhora[:, 1, :, :], (0, 2, 1))), {"units": "m/s"}),
            prec=(("time", "lat", "lon"), np.float32(np.transpose(Basefullhora[:, 2, :, :], (0, 2, 1))), {"units": "mm"}),
            hur=(("time", "lat", "lon"), np.float32(np.transpose(Basefullhora[:, 3, :, :], (0, 2, 1))), {"units": "%"}),
            FFDI=(("time", "lat", "lon"), np.float32(np.transpose(Basefullhora[:, 4, :, :], (0, 2, 1))), {"units": "Adim"}),
            Fire=(("time", "lat", "lon"), np.float32(np.transpose(Basefullhora[:, 5, :, :], (0, 2, 1))), {"units": "Adim"}),
        ),
        coords=dict(
            time=("time", out_time),
            lat=("lat", latx),
            lon=("lon", lonx),
            point_id=("lat", point_id),
            zone_id=("lat", zone_id),
            original_row_SystemzoneV1=("lat", original_row),
            Lat_real=("lat", la),
            Lon_real=("lat", lo),
            scenario=("lon", np.array(["baseline", "with_EWE"], dtype="U16")),
        ),
        attrs=dict(
            description="Hourly climate data with selected EWEs generated from ARClim-compatible future inputs",
            source_future="Dailyfuturedata.nc + Dailyfuturedata_relativehumidity.nc",
            source_hourly="BaseFhorPELP4v20z2anos.nc",
            start_year=args.start_year,
            end_year=args.end_year,
            note="Dimensions keep legacy names lat/lon: lat is point_id in the same row order as SystemzoneV1.xlsx; zone_id stores the original Zona identifier; lon is scenario index 0=baseline, 1=with_EWE. Heatwaves, coldwaves and drought scaling are intentionally disabled.",
        ),
    )

    output_path = os.path.join(dirs.output_dir, args.output_file)
    summary_path = os.path.join(dirs.output_dir, "resumen_EWE_arclim.csv")
    wind_path = os.path.join(dirs.output_dir, "EWEwind_hora_anual.npy")
    order_path = os.path.join(dirs.output_dir, "orden_zonas_BaseHoraEWEPELP20z.csv")

    zone_order = pd.DataFrame({
        "col_idx_numpy": point_id,
        "lat_coord_in_netcdf": latx,
        "zone_id": zone_id,
        "original_row_SystemzoneV1": original_row,
        "Lat": la,
        "Lon": lo,
    })

    ds.to_netcdf(output_path)
    summary.to_csv(summary_path, index=False)
    zone_order.to_csv(order_path, index=False)
    np.save(wind_path, vientnohor)

    print("\nShape de salida:")
    print(f"  tem:  {ds['tem'].shape}")
    print(f"  wind: {ds['wind'].shape}")
    print(f"  prec: {ds['prec'].shape}")
    print(f"  hur:  {ds['hur'].shape}")
    print("  Interpretacion: time x punto/zona x escenario")

    print("\nListo.")
    print(f"NetCDF EWE guardado en: {output_path}")
    print(f"Resumen CSV guardado en: {summary_path}")
    print(f"Orden de zonas guardado en: {order_path}")
    print(f"Horas de viento extremo guardadas en: {wind_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera BaseHoraEWEPELP20z.nc usando ARClim; sin olas de calor/frio ni sequia.")
    parser.add_argument("--project-root", default=None, help="Carpeta que contiene Inputs/ y Outputs/. Si se omite, se busca en la carpeta del script y en la carpeta madre.")
    parser.add_argument("--coords-file", default=DEFAULT_COORDS_FILE)
    parser.add_argument("--base-hourly-file", default=DEFAULT_BASE_HOURLY_FILE)
    parser.add_argument("--future-daily-file", default=DEFAULT_FUTURE_DAILY_FILE)
    parser.add_argument("--future-humidity-file", default=DEFAULT_FUTURE_HUMIDITY_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)

    # Umbrales/factores según el script original/paper
    parser.add_argument("--wind-threshold", type=float, default=20.8, help="Velocidad umbral máxima de viento tras ráfaga [m/s].")
    parser.add_argument("--wind-gust-factor", type=float, default=1.263)
    parser.add_argument("--humidity-threshold", type=float, default=15.0, help="Umbral de humedad relativa baja [%].")
    parser.add_argument("--humidity-scaling", type=float, default=0.7, help="Factor aplicado a humedad relativa baja.")
    parser.add_argument("--ffdi-threshold", type=float, default=15.0, help="Umbral FFDI para incendio.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
