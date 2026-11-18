import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

# Define relative paths
from caso_clima import raiz_proyecto
BASE_DIR = raiz_proyecto()  # <caso>/Proyecto oficial (ver caso_clima.py)
INPUT_DIR = os.path.join(BASE_DIR, "Inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs")
# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read input files
Bfut = xr.open_dataset(os.path.join(INPUT_DIR, "Dailyfuturedata.nc"))
Bfuthur = xr.open_dataset(os.path.join(INPUT_DIR, "Dailyfuturedata_relativehumidity.nc"))
Bhis = xr.open_dataset(os.path.join(INPUT_DIR, "Hourlyhistoricdata.nc"))
df2 = pd.read_excel(os.path.join(INPUT_DIR, "SystemzoneV1.xlsx"))



def read_future_humidity(Bfuthur, date_str, point_index):
    """Lee humedad futura diaria o compatible.

    En Dailyfuturedata_relativehumidity.nc, la dimension lat se usa como
    indice de punto z en el mismo orden del Excel de coordenadas, no como
    latitud geografica real. La dimension lon es dummy; se usa lon=0.
    """
    selected = Bfuthur['hur'].sel(time=date_str)
    if 'time' in selected.dims:
        # Formato compatible antiguo: seleccion por fecha devuelve 24 horas
        return float(selected.isel(time=0, lat=point_index, lon=0).values)
    # Formato correcto diario: seleccion por fecha devuelve matriz punto x dummy_lon
    return float(selected.isel(lat=point_index, lon=0).values)

# Extract coordinates
la = df2['Lat'].tolist()
lo = df2['Lon'].tolist()
lacon0 = range(len(la))

# Define time periods
yearBASE = range(2025, 2041)
year = range(2029, 2031)
year0 = range(2017, 2022)
month = range(1, 13)
daysA = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
daysB = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
dayZA = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
dayZB = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
danos = [365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365, 366, 365, 365, 365, 366]
danoZ = [0, 365, 730, 1095, 1461, 1826, 2191, 2556, 2922, 3287, 3652, 4017, 4383, 4748, 5113, 5478, 5844]

# Adapt danoZ for output years (2029 onward)
ubicacion = yearBASE.index(year[0])
danoZV2 = danoZ[ubicacion:]
danoZV2 = [valor - danoZV2[0] for valor in danoZV2]

# Initialize historical data array
baseanM = np.zeros((len(year0) * 31, 6 + 24 * 4, len(month)), dtype="float32")

# Initialize output arrays
time3 = pd.date_range('2029-01-01', freq='H', periods=365 * 24 * 2)
tas1 = np.zeros((len(time3), len(la), 2), dtype="float32")
wind1 = np.zeros((len(time3), len(la), 2), dtype="float32")
prec1 = np.zeros((len(time3), len(la), 2), dtype="float32")
hur1 = np.zeros((len(time3), len(la), 2), dtype="float32")

# Process historical and future data
iter = 0
for z in range(len(la)):
    largo = np.zeros(12, dtype=int)
    for m in month:
        y0cont0 = 0
        for y0 in year0:
            days = daysA
            for d in range(1, days[m - 1] + 1):
                for h in range(0, 24):
                    baseanM[y0cont0 * days[m - 1] + d - 1, 6 + h, m - 1] = Bhis['t2m'].sel(
                        time=f"{y0}-{m:02d}-{d:02d} {h:02d}:00:00", lat=la[z], lon=lo[z], method='nearest').values
                    baseanM[y0cont0 * days[m - 1] + d - 1, 30 + h, m - 1] = Bhis['wind'].sel(
                        time=f"{y0}-{m:02d}-{d:02d} {h:02d}:00:00", lat=la[z], lon=lo[z], method='nearest').values
                    baseanM[y0cont0 * days[m - 1] + d - 1, 54 + h, m - 1] = Bhis['tp'].sel(
                        time=f"{y0}-{m:02d}-{d:02d} {h:02d}:00:00", lat=la[z], lon=lo[z], method='nearest').values
                    ab = Bhis['hur'].sel(
                        time=f"{y0}-{m:02d}-{d:02d} {h:02d}:00:00", lat=la[z], lon=lo[z], method='nearest').values
                    baseanM[y0cont0 * days[m - 1] + d - 1, 78 + h, m - 1] = min(ab, 100)
                baseanM[y0cont0 * days[m - 1] + d - 1, 0, m - 1] = np.mean(baseanM[y0cont0 * days[m - 1] + d - 1, 6:30, m - 1])
                baseanM[y0cont0 * days[m - 1] + d - 1, 1, m - 1] = np.max(baseanM[y0cont0 * days[m - 1] + d - 1, 6:30, m - 1])
                baseanM[y0cont0 * days[m - 1] + d - 1, 2, m - 1] = np.min(baseanM[y0cont0 * days[m - 1] + d - 1, 6:30, m - 1])
                baseanM[y0cont0 * days[m - 1] + d - 1, 3, m - 1] = np.mean(baseanM[y0cont0 * days[m - 1] + d - 1, 30:54, m - 1])
                baseanM[y0cont0 * days[m - 1] + d - 1, 4, m - 1] = np.sum(baseanM[y0cont0 * days[m - 1] + d - 1, 54:78, m - 1])
                baseanM[y0cont0 * days[m - 1] + d - 1, 5, m - 1] = np.mean(baseanM[y0cont0 * days[m - 1] + d - 1, 78:102, m - 1])
                largo[m - 1] = y0cont0 * days[m - 1] + d - 1
            y0cont0 += 1
        ycont0 = 0
        for y in year:
            days = daysB if y in [2028, 2032, 2036, 2040, 2044] else daysA
            dayZ = dayZB if y in [2028, 2032, 2036, 2040, 2044] else dayZA
            for d in range(1, days[m - 1] + 1):
                ErrorT = np.zeros(largo[m - 1] + 1)
                ErrorW = np.zeros(largo[m - 1] + 1)
                ErrorP = np.zeros(largo[m - 1] + 1)
                ErrorH = np.zeros(largo[m - 1] + 1)
                tem1 = Bfut['tas'].sel(time=f"{y}-{m:02d}-{d:02d}", lat=la[z], lon=lo[z], method='nearest').values
                tma1 = Bfut['tasmax'].sel(time=f"{y}-{m:02d}-{d:02d}", lat=la[z], lon=lo[z], method='nearest').values
                tmi1 = Bfut['tasmin'].sel(time=f"{y}-{m:02d}-{d:02d}", lat=la[z], lon=lo[z], method='nearest').values
                win1 = Bfut['sfcWind'].sel(time=f"{y}-{m:02d}-{d:02d}", lat=la[z], lon=lo[z], method='nearest').values
                pre1 = Bfut['pr'].sel(time=f"{y}-{m:02d}-{d:02d}", lat=la[z], lon=lo[z], method='nearest').values
                humf1 = read_future_humidity(Bfuthur, f"{y}-{m:02d}-{d:02d}", z)
                humf1 = min(humf1, 100)
                for p in range(largo[m - 1] + 1):
                    ErrorT[p] = (tem1 - baseanM[p, 0, m - 1])**2 + (tma1 - baseanM[p, 1, m - 1])**2 + (tmi1 - baseanM[p, 2, m - 1])**2
                    ErrorW[p] = (win1 - baseanM[p, 3, m - 1])**2
                    ErrorP[p] = (pre1 - baseanM[p, 4, m - 1])**2
                    ErrorH[p] = (humf1 - baseanM[p, 5, m - 1])**2
                index = np.argmin(ErrorT)
                indexW = np.argmin(ErrorW)
                indexP = np.argmin(ErrorP)
                indexH = np.argmin(ErrorH)
                Bb = (tma1 - tem1) / (np.max(baseanM[index, 6:30, m - 1]) - np.mean(baseanM[index, 6:30, m - 1]))
                Aa = tma1 - Bb * np.max(baseanM[index, 6:30, m - 1])
                baseanM[index, 6:30, m - 1] = Aa + Bb * baseanM[index, 6:30, m - 1]
                aux1 = np.sum(baseanM[indexW, 30:54, m - 1])
                aux2 = win1 * 24 / aux1 if aux1 != 0 else win1 / 24
                baseanM[indexW, 30:54, m - 1] = aux2 * baseanM[indexW, 30:54, m - 1]
                aux1 = np.sum(baseanM[indexP, 54:78, m - 1])
                baseanM[indexP, 54:78, m - 1] = (pre1 / aux1) * baseanM[indexP, 54:78, m - 1] if aux1 != 0 else (pre1 / 24) * np.ones(24)
                aux1 = np.sum(baseanM[indexH, 78:102, m - 1])
                aux2 = humf1 * 24 / aux1
                baseanM[indexH, 78:102, m - 1] = aux2 * baseanM[indexH, 78:102, m - 1]
                horact = dayZ[m - 1] * 24 + (d - 1) * 24
                horact2 = danoZV2[ycont0] * 24 + horact
                tas1[horact2:(horact2 + 24), lacon0[z], 0] = baseanM[index, 6:30, m - 1]
                wind1[horact2:(horact2 + 24), lacon0[z], 0] = baseanM[indexW, 30:54, m - 1]
                prec1[horact2:(horact2 + 24), lacon0[z], 0] = baseanM[indexP, 54:78, m - 1]
                hur1[horact2:(horact2 + 24), lacon0[z], 0] = baseanM[indexH, 78:102, m - 1]
            ycont0 += 1
    iter += 1
    print(f"Iteration {iter} completed for location {z}")

# Create and export NetCDF dataset
latx = range(len(la))
lonx = range(0, 2)
ds = xr.Dataset(
    data_vars=dict(
        tem=(["time", "lat", "lon"], tas1, {"units": "K"}),
        prec=(["time", "lat", "lon"], prec1, {"units": "mm"}),
        wind=(["time", "lat", "lon"], wind1, {"units": "m/s"}),
        hur=(["time", "lat", "lon"], hur1, {"units": "%"})
    ),
    coords=dict(
        time=(["time"], time3),
        lat=(["lat"], latx),
        lon=(["lon"], lonx)
    ),
    attrs=dict(description="Hourly climate data for 2029-2030 at specified locations")
)
ds.to_netcdf(os.path.join(OUTPUT_DIR, "BaseFhorPELP4v20z2anos.nc"))

