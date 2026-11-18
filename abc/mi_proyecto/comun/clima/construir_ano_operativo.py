#!/usr/bin/env python3
"""
Construye el ano operativo (opcion B): ano observado + delta ARClim.

POR QUE ESTA OPCION Y NO EL REESCALADO POR ANALOGOS DEL PAPER
-------------------------------------------------------------
El metodo del paper (Hourly_climate_zone_*.py) fuerza la media diaria del
perfil horario al valor absoluto de ARClim. Eso funciona solo si ambas fuentes
comparten climatologia. Medido en Arauco, periodo pareado 2017-2025:

    Zona 1: ARClim / Open-Meteo = 1.00x   (coinciden)
    Zona 2:                       0.79x
    Zona 3:                       0.68x

Aplicando esos factores al ano 2024, las horas con rafaga > 20.8 m/s pasan de
140 / 119 / 30 a 140 / 6 / 0. Un sesgo de nivel de -21% borra el 95% de las
horas extremas, porque el umbral esta en la cola de la distribucion. El
contraste entre zonas pasa de 4.7x a 140x, que es un artefacto.

Esta version transfiere el DELTA en vez del NIVEL. El sesgo se cancela en el
cociente y la climatologia observada se conserva.

QUE PRODUCE
-----------
Un NetCDF con el mismo contrato que BaseHoraEWEPELP20z.nc, para que
construir_faultlin.py y construir_perfiles.py lo lean sin cambios:

    dims  (time, lat, lon)
    lat   = point_id 0..Z-1, en el orden del Excel de coordenadas
    lon   = 0 -> viento SOSTENIDO   (perfil de generacion eolica)
            1 -> RAFAGA             (variable de amenaza para fragilidad)

Ese mapeo hace que `construir_faultlin.py --escenario with_EWE` lea rafagas y
`--escenario baseline` lea sostenido, sin tocar una linea de ese script.

OJO: la rafaga ya es rafaga. NO multiplicarla por ningun factor de rafaga
adicional; seria contarla dos veces.

PROCEDENCIA
-----------
El horario es reanalisis (Open-Meteo / ECMWF IFS), no medicion de estacion.
El delta es ARClim ssp585 por miembro individual, NO la media del ensemble
(promediar miembros preserva la media y destruye la cola).

Uso:
    python construir_ano_operativo.py                          # 2023-2024, delta mediano
    python construir_ano_operativo.py --anio-operativo 2020    # ano severo
    python construir_ano_operativo.py --miembro AWI-CM-1-1-MR  # el modelo del paper
    python construir_ano_operativo.py --sin-delta              # solo observado
    python construir_ano_operativo.py --delta-clima            # + delta de T, lluvia y humedad

DELTA DE TEMPERATURA, LLUVIA Y HUMEDAD (--delta-clima, opcional)
----------------------------------------------------------------
Sin esta opcion el delta ARClim se calcula SOLO desde el viento y multiplica solo
wind/gust; t2m, hur y tp pasan tal cual del ano observado. Con --delta-clima, por
zona y con la misma ventana y seleccion de miembro que el viento:
    tem   : delta ADITIVO POR MES de tasmax y de tasmin (futuro - historico, K = degC).
            Cada dia se transforma linealmente, T' = T + dTmin + (dTmax-dTmin)*frac, con
            frac = (T-Tmin_dia)/(Tmax_dia-Tmin_dia): el minimo diario sube dTmin, el maximo
            dTmax y el resto se interpola. Es el ajuste lineal A*beta+B del paso I.A de
            Gutierrez et al. (2026), aplicado al delta y no al nivel absoluto (que en Arauco
            esta sesgado, ver arriba). Cambia el rango diario y respeta la estacionalidad.
    prec *= r_pr    r_pr = futuro/historico de pr (cociente anual, como el viento)
    hur   = RH * r_q * es(T)/es(T')   r_q = futuro/historico anual de huss (humedad especifica)
ARClim no expone humedad relativa por la API (hurs/hur dan error 500), solo huss. La
formula conserva la humedad especifica escalada por r_q y recalcula la relativa con la
temperatura nueva (Magnus, presion constante). pr y huss quedan con delta ANUAL: por mes
el cociente es inestable con 2 anios de ventana en los meses secos. Con esa ventana
futura (la misma del viento) el delta mensual por miembro es ruidoso.
Escribe a <salida>_clima.nc salvo que se pase --salida-nc, para no pisar la base vigente.
"""

from __future__ import annotations

import argparse
import io
import os
import time

import numpy as np
import pandas as pd
import requests
import xarray as xr

ARCLIM_URL = "https://arclim.mma.gob.cl/api/daily_series/{v}/{esc}/"
BASE_HIST = (1985, 2014)      # ventana historica de referencia dentro de ssp585
DEFAULT_FUT = (2029, 2030)    # ventana futura objetivo


def resolver_dirs(coords_file: str):
    from caso_clima import raiz_proyecto
    base = raiz_proyecto()
    for root in [base, os.path.abspath(os.path.join(base, ".."))]:
        inp, out = os.path.join(root, "Inputs"), os.path.join(root, "Outputs")
        if os.path.exists(os.path.join(inp, coords_file)):
            os.makedirs(out, exist_ok=True)
            return root, inp, out
    raise FileNotFoundError(f"No encontre Inputs/{coords_file}")


def leer_coords(path: str) -> pd.DataFrame:
    c = pd.read_excel(path)
    c["Lat"] = pd.to_numeric(c["Lat"], errors="coerce")
    c["Lon"] = pd.to_numeric(c["Lon"], errors="coerce")
    c = c.dropna(subset=["Lat", "Lon"]).reset_index(drop=True)
    c["point_id"] = np.arange(len(c), dtype=int)
    c["zone_id"] = pd.to_numeric(c["Zona"], errors="coerce").astype(int) if "Zona" in c else c["point_id"] + 1
    return c


def arclim(v: str, lat: float, lon: float, esc: str = "ssp585") -> pd.DataFrame:
    r = requests.get(ARCLIM_URL.format(v=v, esc=esc), params={"lat": lat, "lon": lon}, timeout=300)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text.strip()))
    df.columns = [str(c).strip() for c in df.columns]
    dc = list(df.columns[:3])
    mc = [c for c in df.columns if c not in dc]
    for c in dc + mc:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=dc)
    df["time"] = pd.to_datetime(dict(year=df[dc[0]].astype(int), month=df[dc[1]].astype(int),
                                     day=df[dc[2]].astype(int)), errors="coerce")
    return df.dropna(subset=["time"]).set_index("time")[mc]


def _exigir_ventana(d: pd.DataFrame, fut: tuple[int, int], v: str) -> None:
    """Falla si la ventana futura no tiene datos (ARClim llega hasta 2069), en vez de dar NaN."""
    n = d[(d.index.year >= fut[0]) & (d.index.year <= fut[1])].notna().any(axis=1).sum()
    if n == 0:
        raise SystemExit(f"ARClim {v}: sin datos en {fut[0]}-{fut[1]}; la serie cubre "
                         f"{d.index.min().year}-{d.index.max().year}. Usa --fut-y0/--fut-y1 dentro de ese rango.")


def delta_zona(lat: float, lon: float, fut: tuple[int, int], miembro: str | None, pausa: float) -> dict:
    """Delta = futuro/historico, por miembro, dentro de la MISMA serie ssp585.

    Comparar contra `ref` seria invalido: ref trae 1 sola serie de referencia y
    no comparte climatologia con los GCM, asi que el cociente mezclaria sesgo
    entre productos con senal climatica.
    """
    u = arclim("uas", lat, lon); time.sleep(pausa)
    v = arclim("vas", lat, lon); time.sleep(pausa)
    comunes = [c for c in u.columns if c in v.columns]
    ws = pd.DataFrame({c: np.sqrt(u[c] ** 2 + v[c] ** 2) for c in comunes})
    _exigir_ventana(ws, fut, "uas/vas")

    hist = ws[(ws.index.year >= BASE_HIST[0]) & (ws.index.year <= BASE_HIST[1])]
    futu = ws[(ws.index.year >= fut[0]) & (ws.index.year <= fut[1])]
    por_miembro = (futu.mean() / hist.mean()).dropna()

    if miembro:
        sel = [c for c in por_miembro.index if c.startswith(miembro)]
        if not sel:
            raise ValueError(f"Miembro '{miembro}' no esta. Disponibles: "
                             + ", ".join(sorted(c.split('_')[0] for c in por_miembro.index)))
        elegido, valor = sel[0], float(por_miembro[sel[0]])
    else:
        elegido, valor = "mediana_entre_miembros", float(por_miembro.median())

    return {"delta": valor, "miembro": elegido,
            "delta_min": float(por_miembro.min()), "delta_max": float(por_miembro.max()),
            "n_miembros": int(len(por_miembro))}


def delta_variable(v: str, modo: str, lat: float, lon: float, fut: tuple[int, int],
                   miembro: str | None) -> dict:
    """Delta futuro vs historico de una variable ARClim ssp585, por miembro.

    modo "dif": futuro - historico (temperatura). modo "ratio": futuro/historico.
    Si `miembro` no esta en esa variable (cada una tiene su propio conjunto de GCM),
    usa la mediana entre miembros y lo deja anotado en "miembro".
    """
    d = arclim(v, lat, lon)
    _exigir_ventana(d, fut, v)
    hist = d[(d.index.year >= BASE_HIST[0]) & (d.index.year <= BASE_HIST[1])].mean()
    futu = d[(d.index.year >= fut[0]) & (d.index.year <= fut[1])].mean()
    por_miembro = (futu - hist if modo == "dif" else futu / hist).dropna()
    sel = [c for c in por_miembro.index if miembro and c.startswith(miembro)]
    if sel:
        return {"delta": float(por_miembro[sel[0]]), "miembro": sel[0]}
    nota = "mediana_entre_miembros" if not miembro else f"mediana (sin {miembro} en {v})"
    return {"delta": float(por_miembro.median()), "miembro": nota}


def delta_mensual(v: str, lat: float, lon: float, fut: tuple[int, int],
                  miembro: str | None) -> dict:
    """Delta aditivo futuro - historico por MES calendario (12 valores), de una variable ARClim.

    Mismo criterio de miembro que delta_variable; con mediana entre miembros, la mediana
    se toma mes a mes.
    """
    d = arclim(v, lat, lon)
    _exigir_ventana(d, fut, v)
    h = d[(d.index.year >= BASE_HIST[0]) & (d.index.year <= BASE_HIST[1])]
    f = d[(d.index.year >= fut[0]) & (d.index.year <= fut[1])]
    por_mes = (f.groupby(f.index.month).mean() - h.groupby(h.index.month).mean())  # mes x miembro
    sel = [c for c in por_mes.columns if miembro and c.startswith(miembro)]
    if sel:
        return {"delta": por_mes[sel[0]].reindex(range(1, 13)).to_numpy(float), "miembro": sel[0]}
    nota = "mediana_entre_miembros" if not miembro else f"mediana (sin {miembro} en {v})"
    return {"delta": por_mes.median(axis=1).reindex(range(1, 13)).to_numpy(float), "miembro": nota}


def aplicar_delta_diario_lineal(t, meses, d_tmax, d_tmin):
    """Corre cada dia con una transformacion lineal T' = T + dTmin + (dTmax-dTmin)*frac.

    frac = (T - Tmin_dia)/(Tmax_dia - Tmin_dia), asi el minimo diario sube dTmin, el maximo
    dTmax y las horas intermedias se interpolan. Es el A*beta+B del paso I.A de Gutierrez
    aplicado al delta en vez de al nivel absoluto. t y meses: arrays (horas,), horas%24==0.
    """
    td = t.reshape(-1, 24).astype("float64")
    tmin, tmax = td.min(axis=1, keepdims=True), td.max(axis=1, keepdims=True)
    rango = tmax - tmin
    frac = np.where(rango > 1e-6, (td - tmin) / np.where(rango > 1e-6, rango, 1.0), 0.5)
    m = meses.reshape(-1, 24)[:, :1] - 1                       # mes de cada dia (0..11)
    return (td + d_tmin[m] + (d_tmax[m] - d_tmin[m]) * frac).reshape(-1)


def _es(t):
    """Presion de vapor de saturacion [hPa], Magnus."""
    return 6.112 * np.exp(17.67 * t / (t + 243.5))


def extraer_anio(ds: xr.Dataset, lat: float, lon: float, anio: int) -> pd.DataFrame:
    """8760 h de un anio. Descarta el 29-feb, igual que el codigo del paper."""
    s = ds.sel(lat=lat, lon=lon, method="nearest")
    t = pd.to_datetime(s.time.values)
    df = pd.DataFrame({v: s[v].values for v in ["wind", "gust", "t2m", "hur", "tp"]}, index=t)
    df = df[df.index.year == anio]
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    if len(df) != 8760:
        raise ValueError(f"El anio {anio} quedo con {len(df)} horas, se esperaban 8760.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Ano operativo = observado + delta ARClim (opcion B).")
    ap.add_argument("--coords-file", default="SystemzoneV1_arauco.xlsx")
    ap.add_argument("--entrada-nc", default="Hourlyhistoricdata_arauco.nc")
    ap.add_argument("--salida-nc", default="BaseHoraEWE_arauco.nc")
    ap.add_argument("--anio-previo", type=int, default=2023, help="Se mapea al primer anio de salida.")
    ap.add_argument("--anio-operativo", type=int, default=2024, help="Se mapea al segundo anio (el que lee construir_faultlin con offset 8760).")
    ap.add_argument("--anio-salida-1", type=int, default=2029)
    ap.add_argument("--fut-y0", type=int, default=DEFAULT_FUT[0])
    ap.add_argument("--fut-y1", type=int, default=DEFAULT_FUT[1])
    ap.add_argument("--miembro", default=None, help="GCM de ARClim, ej AWI-CM-1-1-MR. Omitir = mediana entre miembros.")
    ap.add_argument("--sin-delta", action="store_true", help="No aplicar delta (delta = 1.0).")
    ap.add_argument("--delta-clima", action="store_true",
                    help="Ademas del viento, aplica delta ARClim a temperatura (aditivo), lluvia (cociente) "
                         "y humedad (via humedad especifica). Ver docstring.")
    ap.add_argument("--umbral", type=float, default=20.8, help="Umbral de rafaga para el resumen [m/s].")
    ap.add_argument("--pausa", type=float, default=0.5)
    args = ap.parse_args()

    root, inp, out = resolver_dirs(args.coords_file)
    coords = leer_coords(os.path.join(inp, args.coords_file))
    ds = xr.open_dataset(os.path.join(inp, args.entrada_nc))

    print(f"Proyecto        : {root}")
    print(f"Anos observados : {args.anio_previo} -> {args.anio_salida_1}, "
          f"{args.anio_operativo} -> {args.anio_salida_1 + 1}  (operativo)")
    print(f"Ventana futura  : {args.fut_y0}-{args.fut_y1}")
    print(f"Delta           : {'DESACTIVADO' if args.sin_delta else (args.miembro or 'mediana entre miembros')}")
    print()

    n_z = len(coords)
    horas = 8760 * 2
    wind = np.zeros((horas, n_z), dtype="float32")
    gust = np.zeros((horas, n_z), dtype="float32")
    otras = {v: np.zeros((horas, n_z), dtype="float32") for v in ["t2m", "hur", "tp"]}
    filas = []

    for _, fila in coords.iterrows():
        z = int(fila["point_id"])
        la, lo = float(fila["Lat"]), float(fila["Lon"])

        if args.sin_delta:
            info = {"delta": 1.0, "miembro": "sin_delta", "delta_min": 1.0, "delta_max": 1.0, "n_miembros": 0}
        else:
            print(f"Zona {fila['zone_id']}: consultando ARClim...", flush=True)
            info = delta_zona(la, lo, (args.fut_y0, args.fut_y1), args.miembro, args.pausa)

        d = info["delta"]
        a = extraer_anio(ds, la, lo, args.anio_previo)
        b = extraer_anio(ds, la, lo, args.anio_operativo)
        obs = pd.concat([a, b])

        wind[:, z] = obs["wind"].to_numpy(dtype="float32") * d
        gust[:, z] = obs["gust"].to_numpy(dtype="float32") * d
        for v in otras:
            otras[v][:, z] = obs[v].to_numpy(dtype="float32")

        if args.delta_clima:
            if args.sin_delta:
                raise SystemExit("--delta-clima y --sin-delta se contradicen")
            fut = (args.fut_y0, args.fut_y1)
            dtx = delta_mensual("tasmax", la, lo, fut, args.miembro); time.sleep(args.pausa)
            dtn = delta_mensual("tasmin", la, lo, fut, args.miembro); time.sleep(args.pausa)
            rp = delta_variable("pr", "ratio", la, lo, fut, args.miembro); time.sleep(args.pausa)
            rq = delta_variable("huss", "ratio", la, lo, fut, args.miembro); time.sleep(args.pausa)
            t_obs = otras["t2m"][:, z].astype("float64")
            t_new = aplicar_delta_diario_lineal(t_obs, obs.index.month.to_numpy(),
                                                dtx["delta"], dtn["delta"])
            otras["hur"][:, z] = np.clip(otras["hur"][:, z] * rq["delta"]
                                         * _es(t_obs) / _es(t_new), 0, 100)
            otras["t2m"][:, z] = t_new
            otras["tp"][:, z] = otras["tp"][:, z] * rp["delta"]
            verano = [12, 1, 2]
            info.update({"dTmax_anual": float(np.mean(dtx["delta"])), "dTmin_anual": float(np.mean(dtn["delta"])),
                         "dTmax_verano": float(np.mean(dtx["delta"][[m - 1 for m in verano]])),
                         "miembro_T": dtx["miembro"],
                         "r_pr": rp["delta"], "miembro_pr": rp["miembro"],
                         "r_huss": rq["delta"], "miembro_huss": rq["miembro"]})
            print(f"   dTmax anual {info['dTmax_anual']:+.2f} (verano {info['dTmax_verano']:+.2f}) "
                  f"dTmin {info['dTmin_anual']:+.2f} C  r_pr={rp['delta']:.3f}  r_huss={rq['delta']:.3f}")

        g_op = gust[8760:, z]
        filas.append({
            "point_id": z, "zone_id": int(fila["zone_id"]), "Lat": la, "Lon": lo,
            "delta": d, "miembro": info["miembro"],
            "delta_min": info["delta_min"], "delta_max": info["delta_max"],
            "h_sobre_umbral": int((g_op > args.umbral).sum()),
            "dias_evento": int((pd.Series(g_op).groupby(np.arange(8760) // 24).max() > args.umbral).sum()),
            "gust_max": float(g_op.max()), "wind_max": float(wind[8760:, z].max()),
            **{k: info[k] for k in ("dTmax_anual", "dTmax_verano", "dTmin_anual", "miembro_T",
                                    "r_pr", "miembro_pr", "r_huss", "miembro_huss")
               if k in info},
            "tem_max": float(otras["t2m"][8760:, z].max()),
        })
        print(f"   delta={d:.4f}  (miembros {info['delta_min']:.3f}-{info['delta_max']:.3f})")

    # lon = 0 sostenido, 1 rafaga -> construir_faultlin lee rafaga con --escenario with_EWE
    apilado = {"wind": np.stack([wind, gust], axis=2)}
    for v, arr in otras.items():
        apilado[v] = np.stack([arr, arr], axis=2)

    t = pd.date_range(f"{args.anio_salida_1}-01-01", periods=horas, freq="h")
    dsx = xr.Dataset(
        data_vars={
            "wind": (("time", "lat", "lon"), apilado["wind"].astype("float32"),
                     {"units": "m/s", "description": "lon=0 sostenido 10 m; lon=1 RAFAGA 10 m. La rafaga ya es rafaga: no multiplicar por ningun factor adicional."}),
            "gust": (("time", "lat", "lon"), apilado["wind"].astype("float32"), {"units": "m/s", "description": "Alias de wind para lectura explicita."}),
            "tem": (("time", "lat", "lon"), apilado["t2m"].astype("float32"), {"units": "degC"}),
            "hur": (("time", "lat", "lon"), apilado["hur"].astype("float32"), {"units": "%"}),
            "prec": (("time", "lat", "lon"), apilado["tp"].astype("float32"), {"units": "mm"}),
        },
        coords={
            "time": ("time", t),
            "lat": ("lat", coords["point_id"].to_numpy(int)),
            "lon": ("lon", np.array([0, 1], dtype=int)),
            "zone_id": ("lat", coords["zone_id"].to_numpy(int)),
            "Lat_real": ("lat", coords["Lat"].to_numpy(float)),
            "Lon_real": ("lat", coords["Lon"].to_numpy(float)),
            "escenario": ("lon", np.array(["sostenido", "rafaga"], dtype="U10")),
        },
        attrs={
            "description": "Ano operativo: ano observado (reanalisis) + delta ARClim. Opcion B.",
            "metodo": ("delta multiplicativo del viento por zona; "
                       + ("+ dT aditivo, cociente de lluvia y humedad via huss (--delta-clima); "
                          if args.delta_clima else "temperatura, humedad y lluvia SIN delta; ")
                       + "NO reescalado por analogos"),
            "delta_clima": int(bool(args.delta_clima)),
            "anio_previo": args.anio_previo, "anio_operativo": args.anio_operativo,
            "ventana_futura": f"{args.fut_y0}-{args.fut_y1}",
            "miembro_arclim": filas[0]["miembro"] if filas else "n/a",
            "procedencia_horaria": "Open-Meteo / ECMWF IFS (reanalisis, NO estacion)",
            "aviso_rafaga": "lon=1 ya es rafaga. No aplicar factor de rafaga adicional.",
            "aviso_offset": "construir_faultlin.py con offset 8760 lee el anio operativo.",
        },
    )

    salida_nc = args.salida_nc
    if args.delta_clima and salida_nc == ap.get_default("salida_nc"):
        salida_nc = salida_nc.replace(".nc", "_clima.nc")      # no pisar la base vigente
    salida = os.path.join(out, salida_nc)
    dsx.to_netcdf(salida)
    resumen = pd.DataFrame(filas)
    resumen_path = os.path.join(out, "resumen_ano_operativo_clima.csv" if args.delta_clima
                                else "resumen_ano_operativo.csv")
    resumen.to_csv(resumen_path, index=False)

    print()
    print(f"RESUMEN DEL ANO OPERATIVO ({args.anio_operativo} -> {args.anio_salida_1 + 1}), umbral {args.umbral} m/s:")
    print(resumen[["zone_id", "delta", "h_sobre_umbral", "dias_evento", "gust_max", "wind_max"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()
    print(f"  NetCDF  : {salida}")
    print(f"  Resumen : {resumen_path}")
    print()
    print("  Para la matriz de fallas (lee la RAFAGA):")
    print(f"     python construir_faultlin.py --viento \"{salida}\" --escenario with_EWE --offset 8760")


if __name__ == "__main__":
    main()
