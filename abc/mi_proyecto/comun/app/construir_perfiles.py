# ============================================================
# construir_perfiles.py — Perfiles de generación por zona
# ============================================================
# Genera los perfiles de disponibilidad solar y eólica que consume E1, para
# los PUNTOS REALES del caso de estudio y con RESOLUCIÓN HORARIA ANUAL.
#
# Salida principal (--resolucion horaria, default):
#     artefactos/perfiles_horarios.npz   solar_por_zona / eolico_por_zona,
#                                        ambos (n_horas × n_zonas)
# Salida heredada (--resolucion diaria24):
#     Perfiles_solares_promedio_por_zona.xlsx
#     Perfiles_eolicos_promedio_por_zona.xlsx
#
# ── Por qué horaria y no un perfil diario ───────────────────────────────
# El estudio venía usando un perfil de 24 h repetido los 365 días. Tres
# razones para dejar de hacerlo, en orden de importancia:
#
# 1. CORRELACIÓN CON LAS FALLAS. Las fallas se sortean con el viento del
#    `.nc`. Con un perfil diario repetido, la generación eólica queda por
#    construcción NO CORRELACIONADA con el viento que causa la falla —
#    siendo la misma variable física. Medido sobre el .nc del estudio: en el
#    1% de horas de mayor viento (donde se concentran las fallas) el factor
#    de planta eólico de la zona 1 es 0,99, contra 0,69 de media anual. O
#    sea: el aerogenerador está a potencia nominal justo cuando caen las
#    líneas. Un perfil de 24 h no puede representar eso, y es exactamente lo
#    que decide si el eólico sirve como medida de resiliencia.
#
# 2. ESTACIONALIDAD. En Aysén (−47°) el día de verano dura ~17 h y el de
#    invierno ~7,5 h. Un perfil único hace enero idéntico a julio.
#
# 3. QUÉ DÍA REPRESENTA. El Excel vigente llega a 1,034 al mediodía: un
#    factor de planta de ~1 es un día despejado, no un promedio (el promedio
#    de 5 años en el mismo punto llega a 0,51). Con resolución horaria la
#    pregunta desaparece: no hay que elegir un día representativo.
#
# El artefacto `perfiles.npz` YA guarda (n_horas × n_buses), así que esto no
# cambia ningún formato: lo que aplanaba era `_perfil_diario_anual`.
#
# ── De dónde sale cada recurso ──────────────────────────────────────────
#   EÓLICO  → del MISMO `.nc` que genera las fallas (BaseHoraEWEPELP20z.nc),
#             mismo corte de horas y mismo eje de escenario. Es la única
#             forma de que la turbina y la falla vean el mismo viento.
#   SOLAR   → Open-Meteo Historical (el `.nc` no trae irradiancia), promediado
#             por (día del año, hora) sobre varios años y mapeado al horizonte
#             del estudio. Conserva estacionalidad y ciclo diario.
#             Alternativa: exportación del Explorador Solar.
#
# USO
#   python construir_perfiles.py --estado
#   python construir_perfiles.py                       # horaria, eólico del .nc
#   python construir_perfiles.py --fuente-eolico openmeteo
#   python construir_perfiles.py --resolucion diaria24 # Excel heredado
#   python construir_perfiles.py --solar Z1=z1.csv Z2=z2.csv Z3=z3.csv
# ============================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import casos

APP_DIR = Path(__file__).resolve().parent
from rutas import CASO_DIR, DATOS_DIR  # datos del caso activo (ver rutas.py)
ART_DIR = casos.art_dir_activo()      # caso activo, ver casos.py
RUTA_SOLAR_XLSX = DATOS_DIR / "Perfiles_solares_promedio_por_zona.xlsx"
RUTA_EOLICO_XLSX = DATOS_DIR / "Perfiles_eolicos_promedio_por_zona.xlsx"
RUTA_HORARIOS = ART_DIR / "perfiles_horarios.npz"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
TZ = "America/Santiago"

# — Modelo fotovoltaico —
# CF = (GHI/1000) · [1 + γ·(T_celda − 25)], con temperatura de celda por el
# modelo NOCT. Es el modelo de una línea que se usa cuando no se modelan
# inclinación ni seguimiento: suficiente acá, porque el perfil entra al MILP
# como fracción de la capacidad instalada, no como energía de una planta
# concreta.
GHI_STC = 1000.0        # W/m² en condiciones estándar
GAMMA_PMP = -0.004      # coeficiente de temperatura, silicio [1/°C]
NOCT = 45.0             # temperatura nominal de operación de celda [°C]
T_STC = 25.0

# — Curva de potencia eólica genérica (aerogenerador distribuido de MT) —
V_CUT_IN = 3.0
V_RATED = 12.0
V_CUT_OUT = 25.0
ALTURA_BUJE_M = 30.0
ALTURA_MEDICION_M = 10.0
EXP_HELLMANN = 1 / 7

# Los Excel heredados: el solar se guarda ×1000 porque E1 lo divide por 1000
# al leerlo (heredado de la celda 38 del notebook); el eólico no se divide.
# Es asimétrico, pero es el contrato vigente.
ESCALA_SOLAR_XLSX = 1000.0
ESCALA_EOLICO_XLSX = 1.0

CF_MEDIO_MINIMO = 0.05     # bajo esto, la tecnología no es viable en el punto

DIRS_CLIMA = [
    CASO_DIR / "Trend database with hourly resolution" / "Inputs",
    CASO_DIR / "Extreme Weather Events database" / "Inputs",
    CASO_DIR / "Proyecto oficial" / "Inputs",
]


def _log(msg):
    print(f"[perfiles] {msg}", flush=True)


# ════════════════════════════════════════════════════════════
# Puntos del caso de estudio
# ════════════════════════════════════════════════════════════

def puntos_del_estudio(ruta=None):
    """({zona: (lat, lon)}, ruta_de_origen).

    Se prefiere el paquete de la pestaña cuando existe: es el que está
    sincronizado con la red que el modelo tiene cargada.
    """
    if ruta:
        rutas = [Path(ruta)]
    else:
        rutas = [ART_DIR / "red_nueva.pkl"] + [d / "SystemzoneV1.xlsx" for d in DIRS_CLIMA]

    for r in rutas:
        if not r.exists():
            continue
        if r.suffix == ".pkl":
            import pickle
            with open(r, "rb") as f:
                pts = pickle.load(f).get("puntos_zona") or {}
            if pts:
                _log(f"puntos desde {r.name}")
                return {int(z): (float(a), float(b)) for z, (a, b) in pts.items()}, r
        else:
            df = pd.read_excel(r).dropna(subset=["Lat", "Lon"])
            if len(df):
                _log(f"puntos desde {r}")
                col_z = "Zona" if "Zona" in df.columns else None
                return ({int(row[col_z]) if col_z else i + 1:
                         (float(row["Lat"]), float(row["Lon"]))
                         for i, (_, row) in enumerate(df.iterrows())}, r)

    raise FileNotFoundError(
        "No encontré ni artefactos/red_nueva.pkl ni un SystemzoneV1.xlsx con "
        "columnas Lat/Lon.")


def _distancia_max_km(puntos):
    ps = list(puntos.values())
    peor = 0.0
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            dlat = (ps[i][0] - ps[j][0]) * 111.0
            dlon = (ps[i][1] - ps[j][1]) * 111.0 * np.cos(np.radians(ps[i][0]))
            peor = max(peor, float(np.hypot(dlat, dlon)))
    return peor


# ════════════════════════════════════════════════════════════
# Modelos recurso → factor de planta
# ════════════════════════════════════════════════════════════

def cf_solar(ghi, t2m=None):
    """Factor de planta fotovoltaico horario, en [0, 1]."""
    ghi = np.asarray(ghi, dtype=float)
    t_celda = 25.0 if t2m is None else (
        np.asarray(t2m, dtype=float) + (NOCT - 20.0) / 800.0 * ghi)
    return np.clip((ghi / GHI_STC) * (1.0 + GAMMA_PMP * (t_celda - T_STC)), 0.0, 1.0)


def cf_eolico(viento_10m, altura_buje=ALTURA_BUJE_M):
    """Factor de planta eólico horario, con curva de potencia genérica.

    Entre cut-in y nominal la potencia crece con el cubo de la velocidad —es
    la energía cinética del flujo—; sobre la nominal el control de paso la
    mantiene plana hasta el cut-out, donde la máquina se detiene. Ese
    cut-out importa acá más que en un estudio de energía: es la razón por la
    que un aerogenerador puede estar detenido justo en la tormenta que tira
    las líneas.
    """
    v = np.asarray(viento_10m, dtype=float) * (
        (altura_buje / ALTURA_MEDICION_M) ** EXP_HELLMANN)
    cf = np.zeros_like(v)
    rampa = (v >= V_CUT_IN) & (v < V_RATED)
    cf[rampa] = (v[rampa] ** 3 - V_CUT_IN ** 3) / (V_RATED ** 3 - V_CUT_IN ** 3)
    cf[(v >= V_RATED) & (v <= V_CUT_OUT)] = 1.0
    return np.clip(cf, 0.0, 1.0)


# ════════════════════════════════════════════════════════════
# Fuente A — el .nc de eventos extremos (eólico)
# ════════════════════════════════════════════════════════════

def eolico_desde_nc(ruta_nc, n_horas, offset, escenario_clima, altura_buje):
    """CF eólico horario por zona, del MISMO campo de viento que genera las
    fallas.

    Coherencia física: la turbina y la línea ven el mismo viento, que es
    justamente la correlación que se quiere medir al evaluar DG eólica como
    medida de resiliencia. Cualquier otra fuente la rompe.

    Pero *el mismo campo* no significa *la misma columna*. El `.nc` guarda dos
    estadísticos del mismo viento —sostenido en `lon=0`, ráfaga en `lon=1`— y
    cada uso toma el que le corresponde:

      - la curva de potencia, acá: SOSTENIDO. La IEC 61400 define potencia y
        cut-out sobre media de 10 min. Pasarle la ráfaga infla el factor de
        planta ~3x (0.18 -> 0.60 en Arauco), que es físicamente imposible para
        un aerogenerador distribuido.
      - la curva de fragilidad, en construir_faultlin.py: RÁFAGA, que es la
        convención de la literatura de fragilidad (ASCE / Darestani).
    """
    import xarray as xr

    ds = xr.open_dataset(ruta_nc)
    idx_lon = {"baseline": 0, "with_EWE": 1}[escenario_clima]
    total = ds["wind"].shape[0]
    if offset + n_horas > total:
        raise ValueError(
            f"El .nc tiene {total} horas y se pidieron {n_horas} desde la hora "
            f"{offset}. Ajustar --horas / --anio-offset.")
    viento = ds["wind"].values[offset:offset + n_horas, :, idx_lon]
    return cf_eolico(viento, altura_buje), viento


# ════════════════════════════════════════════════════════════
# Fuente B — Open-Meteo (solar, y eólico alternativo)
# ════════════════════════════════════════════════════════════

def descargar_openmeteo(lat, lon, anio_ini, anio_fin, pausa=1.0):
    """Serie horaria de irradiancia, temperatura y viento para un punto."""
    import time

    import requests

    r = requests.get(OPEN_METEO_URL, params={
        "latitude": lat, "longitude": lon,
        "start_date": f"{anio_ini}-01-01", "end_date": f"{anio_fin}-12-31",
        "hourly": "shortwave_radiation,temperature_2m,wind_speed_10m",
        "wind_speed_unit": "ms", "timezone": TZ,
    }, timeout=180)
    r.raise_for_status()
    h = r.json()["hourly"]
    time.sleep(pausa)
    df = pd.DataFrame({
        "time": pd.to_datetime(h["time"]),
        "ghi": pd.to_numeric(h["shortwave_radiation"], errors="coerce"),
        "t2m": pd.to_numeric(h["temperature_2m"], errors="coerce"),
        "wind": pd.to_numeric(h["wind_speed_10m"], errors="coerce"),
    })
    return df.dropna()


def climatologia_horaria(tiempos, valores, n_horas, anio_destino=2030):
    """Mapea una serie de varios años al horizonte del estudio.

    Promedia el recurso por (día del año, hora) sobre todos los años
    disponibles y luego reconstruye las `n_horas` del horizonte con su propio
    calendario. Así se conserva la estacionalidad y el ciclo diario —lo que
    un perfil de 24 h destruye— y a la vez se suaviza el ruido meteorológico
    de un año puntual, que no aporta nada porque el año de fallas es otro.
    """
    idx = pd.DatetimeIndex(tiempos)
    tabla = (pd.DataFrame({"doy": idx.dayofyear, "h": idx.hour,
                           "v": np.asarray(valores, dtype=float)})
             .groupby(["doy", "h"])["v"].mean())

    destino = pd.date_range(f"{anio_destino}-01-01", periods=n_horas, freq="h")
    clave = pd.MultiIndex.from_arrays([destino.dayofyear, destino.hour])
    serie = tabla.reindex(clave)

    if serie.isna().any():
        # 29 de febrero en años no bisiestos, o huecos de la API.
        serie = serie.to_frame("v")
        serie["h"] = destino.hour
        por_hora = serie.groupby("h")["v"].transform("mean")
        serie = serie["v"].fillna(por_hora).fillna(0.0)
    return serie.to_numpy(dtype=float)


# ════════════════════════════════════════════════════════════
# Fuente C — exportaciones del Explorador Solar / Eólico
# ════════════════════════════════════════════════════════════

# solar.minenergia.cl y eolico.minenergia.cl son aplicaciones web: la
# descarga se genera eligiendo el punto en el mapa y no hay endpoint estable
# que automatizar sin adivinarlo. El archivo se baja a mano y se lee acá.
ALIAS = {
    "time": ["fecha", "fecha/hora", "datetime", "date", "time", "tiempo",
             "fecha_hora", "periodo"],
    "ghi": ["ghi", "radiacion global", "radiación global", "global horizontal",
            "irradiancia", "rad_global"],
    "t2m": ["temperatura", "temp", "t2m", "tamb", "temperatura ambiente"],
    "wind": ["velocidad", "velocidad del viento", "wind", "ws", "viento",
             "vel_viento", "speed"],
    "cf": ["factor de planta", "factor_planta", "cf", "capacity factor",
           "generacion", "generación", "potencia"],
}


def _norm(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.strip().lower()


def _buscar_columna(df, clave):
    normalizadas = {_norm(c): c for c in df.columns}
    for alias in ALIAS[clave]:
        if alias in normalizadas:
            return normalizadas[alias]
    for alias in ALIAS[clave]:
        for norm, original in normalizadas.items():
            if alias in norm:
                return original
    return None


def leer_export_explorador(ruta):
    """Lee una exportación horaria del Explorador Solar o Eólico."""
    ruta = Path(ruta)
    if ruta.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(ruta)
    else:
        df = None
        for saltar in range(0, 12):     # los exports traen metadatos arriba
            try:
                intento = pd.read_csv(ruta, skiprows=saltar, sep=None, engine="python")
            except Exception:
                continue
            if _buscar_columna(intento, "time") is not None and intento.shape[1] >= 2:
                df = intento
                if saltar:
                    _log(f"  {ruta.name}: se saltaron {saltar} filas de encabezado")
                break
        if df is None:
            raise ValueError(
                f"No pude identificar la tabla dentro de '{ruta.name}'. Se espera "
                f"una columna de fecha/hora y una de dato horario.")

    salida = {}
    for clave in ("time", "ghi", "t2m", "wind", "cf"):
        col = _buscar_columna(df, clave)
        if col is not None:
            salida[clave] = df[col]
    if "time" not in salida:
        raise ValueError(f"'{ruta.name}': no encontré columna de fecha/hora. "
                         f"Columnas: {list(df.columns)}")
    if not ({"ghi", "wind", "cf"} & set(salida)):
        raise ValueError(f"'{ruta.name}': no encontré irradiancia, viento ni "
                         f"factor de planta. Columnas: {list(df.columns)}")

    out = pd.DataFrame(salida)
    out["time"] = pd.to_datetime(out["time"], errors="coerce", dayfirst=True)
    for c in out.columns:
        if c != "time":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["time"])


def _cf_desde_export(df, tipo, altura_buje):
    if "cf" in df:
        _log(f"  trae factor de planta: se usa directo, sin modelo de {tipo}")
        return np.clip(df["cf"].to_numpy(float), 0, 1)
    if tipo == "solar":
        return cf_solar(df["ghi"], df.get("t2m"))
    return cf_eolico(df["wind"], altura_buje)


# ════════════════════════════════════════════════════════════
# Diagnóstico
# ════════════════════════════════════════════════════════════

def _informe(nombre, cf_por_zona, km_max, viento_para_correlacion=None):
    """cf_por_zona: (n_horas × n_zonas)."""
    medios = cf_por_zona.mean(axis=0)
    detalle = ", ".join(f"Z{i+1}={v:.3f}" for i, v in enumerate(medios))
    _log(f"{nombre}: factor de planta medio por zona — {detalle}")

    if medios.max() > 0:
        disp = float((medios.max() - medios.min()) / medios.max())
        _log(f"  dispersión entre zonas: {disp:.1%}")
    else:
        disp = 0.0

    # Una dispersión alta significa dos cosas muy distintas: que las zonas
    # tienen recursos genuinamente distintos, o que alguna no tiene recurso.
    # Confundirlas lleva a instalar capacidad donde nunca va a generar.
    sin_recurso = [i + 1 for i, v in enumerate(medios) if v < CF_MEDIO_MINIMO]
    if sin_recurso:
        _log(f"  [!] zona(s) con CF medio < {CF_MEDIO_MINIMO:.0%}: {sin_recurso}. "
             f"Revisar si corresponde ofrecer esta tecnología ahí.")
    elif km_max and km_max < 150 and disp < 0.10:
        _log(f"  → zonas a {km_max:.0f} km con {disp:.1%} de diferencia: a esta "
             f"escala un solo perfil estaría justificado.")

    # Lo que un perfil de 24 h no puede mostrar: cuánto genera la tecnología
    # justo en las horas en que se rompen las líneas.
    if viento_para_correlacion is not None:
        v = viento_para_correlacion
        for z in range(cf_por_zona.shape[1]):
            umbral = np.percentile(v[:, z], 99)
            criticas = v[:, z] >= umbral
            if criticas.any():
                _log(f"  Z{z+1}: CF en el 1% de horas de mayor viento = "
                     f"{cf_por_zona[criticas, z].mean():.3f} "
                     f"(media anual {medios[z]:.3f})")
    return disp, medios


# ════════════════════════════════════════════════════════════
# Comparación entre fuentes
# ════════════════════════════════════════════════════════════
# Tres fuentes describen el mismo recurso en el mismo punto y NO coinciden.
# Cuál se usa cambia el resultado del estudio, así que conviene verlas juntas
# antes de elegir:
#
#   .nc del estudio   BaseHoraEWEPELP20z.nc — patrón horario de Open-Meteo
#                     mezclado con el nivel diario de ARClim (paso 3 del
#                     pipeline climático), proyectado a 2030. Es el que define
#                     las fallas.
#   Open-Meteo        reanálisis histórico directo, ~10 km de resolución.
#   Explorador        solar.minenergia.cl / eolico.minenergia.cl — WRF a ~1 km
#                     calibrado para Chile. Es la referencia nacional, y la
#                     única de las tres que resuelve topografía de valle. Se
#                     descarga a mano (son aplicaciones web sin API pública)
#                     y se pasa con --solar / --eolico.
#
# En terreno plano las tres deberían parecerse. Donde no se parecen —y en
# Aysén no se parecen— manda el Explorador.

def comparar_fuentes(puntos, args):
    """Tabla comparativa por zona y fuente. Devuelve un DataFrame."""
    zonas = sorted(puntos)
    filas = []

    # — .nc del estudio —
    try:
        cf_nc, viento_nc = eolico_desde_nc(args.viento, args.horas, args.offset,
                                           args.escenario_clima, args.altura_buje)
        for i, z in enumerate(zonas):
            if i >= viento_nc.shape[1]:
                break
            v, cf = viento_nc[:, i], cf_nc[:, i]
            filas.append(_fila_viento("nc_estudio", z, v, cf))
    except Exception as e:
        _log(f"[!] no se pudo leer el .nc: {type(e).__name__}: {e}")

    # — Open-Meteo —
    for z in zonas:
        lat, lon = puntos[z]
        try:
            _log(f"Z{z}: Open-Meteo {args.anios[0]}-{args.anios[1]}...")
            df = descargar_openmeteo(lat, lon, *args.anios)
        except Exception as e:
            _log(f"[!] Z{z} Open-Meteo: {type(e).__name__}: {e}")
            continue
        v = df["wind"].to_numpy(float)
        filas.append(_fila_viento("openmeteo", z, v, cf_eolico(v, args.altura_buje)))
        ghi = df["ghi"].to_numpy(float)
        filas.append(_fila_solar("openmeteo", z, ghi, cf_solar(ghi, df["t2m"])))

    # — Explorador (exports manuales) —
    for etiqueta, mapa, tipo in (("explorador_eolico", _parse_asignaciones(args.eolico), "eólico"),
                                 ("explorador_solar", _parse_asignaciones(args.solar), "solar")):
        for z, ruta in sorted(mapa.items()):
            try:
                df = leer_export_explorador(ruta)
            except Exception as e:
                _log(f"[!] Z{z} {etiqueta}: {type(e).__name__}: {e}")
                continue
            cf = _cf_desde_export(df, tipo, args.altura_buje)
            if tipo == "eólico":
                v = (df["wind"].to_numpy(float) if "wind" in df
                     else np.full(len(cf), np.nan))
                filas.append(_fila_viento(etiqueta, z, v, cf))
            else:
                ghi = (df["ghi"].to_numpy(float) if "ghi" in df
                       else np.full(len(cf), np.nan))
                filas.append(_fila_solar(etiqueta, z, ghi, cf))

    if not filas:
        _log("No se pudo construir ninguna fila de comparación.")
        return pd.DataFrame()

    df = pd.DataFrame(filas)
    return df.sort_values(["recurso", "zona", "fuente"]).reset_index(drop=True)


def _fila_viento(fuente, zona, v, cf):
    v = np.asarray(v, dtype=float)
    finito = v[np.isfinite(v)]
    # El indicador que importa para resiliencia: cuánto genera la turbina en
    # las horas de viento extremo, que son las que rompen las líneas.
    if finito.size:
        criticas = v >= np.nanpercentile(v, 99)
        cf_extremo = float(np.nanmean(cf[criticas])) if criticas.any() else np.nan
    else:
        cf_extremo = np.nan
    return {
        "recurso": "eólico", "zona": zona, "fuente": fuente, "n_horas": len(cf),
        "v_medio_ms": float(np.nanmean(finito)) if finito.size else np.nan,
        "v_p90_ms": float(np.nanpercentile(finito, 90)) if finito.size else np.nan,
        "v_max_ms": float(np.nanmax(finito)) if finito.size else np.nan,
        "cf_medio": float(np.nanmean(cf)),
        "cf_horas_viento_extremo": cf_extremo,
    }


def _fila_solar(fuente, zona, ghi, cf):
    ghi = np.asarray(ghi, dtype=float)
    finito = ghi[np.isfinite(ghi)]
    return {
        "recurso": "solar", "zona": zona, "fuente": fuente, "n_horas": len(cf),
        "ghi_medio_wm2": float(np.nanmean(finito)) if finito.size else np.nan,
        "ghi_max_wm2": float(np.nanmax(finito)) if finito.size else np.nan,
        "cf_medio": float(np.nanmean(cf)),
        "cf_p90": float(np.nanpercentile(cf, 90)),
    }


def mostrar_estado():
    print("=" * 66)
    print("PERFILES DE GENERACIÓN")
    print("=" * 66)
    try:
        puntos, origen = puntos_del_estudio()
        print(f"  puntos del estudio ({origen.name}):")
        for z, (la, lo) in sorted(puntos.items()):
            print(f"    Z{z}: {la:.4f}, {lo:.4f}")
        print(f"  distancia máxima entre zonas: {_distancia_max_km(puntos):.0f} km")
    except FileNotFoundError as e:
        print(f"  [!] {e}")

    if RUTA_HORARIOS.exists():
        z = np.load(RUTA_HORARIOS)
        print(f"  [x] HORARIO  {RUTA_HORARIOS.name}")
        for clave in ("solar_por_zona", "eolico_por_zona"):
            if clave in z.files:
                a = z[clave]
                medios = ", ".join(f"Z{i+1}={v:.3f}" for i, v in enumerate(a.mean(axis=0)))
                print(f"        {clave:<16} {a.shape}  CF medio: {medios}")
    else:
        print(f"  [ ] HORARIO  {RUTA_HORARIOS.name} (no existe)")

    for etiqueta, ruta, escala in (("solar", RUTA_SOLAR_XLSX, ESCALA_SOLAR_XLSX),
                                   ("eólico", RUTA_EOLICO_XLSX, ESCALA_EOLICO_XLSX)):
        if ruta.exists():
            df = pd.read_excel(ruta)
            cols = [c for c in df.columns if str(c).startswith("Z")]
            medios = ", ".join(f"{c}={df[c].mean()/escala:.3f}" for c in cols)
            print(f"  [x] 24 h     {ruta.name}  ({len(df)} h × {len(cols)} zonas) "
                  f"CF medio: {medios}")
        else:
            print(f"  [ ] 24 h     {ruta.name} (no existe)")
    print("=" * 66)


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def _parse_asignaciones(lista):
    """['Z1=a.csv', '2=b.csv'] -> {1: 'a.csv', 2: 'b.csv'}"""
    salida = {}
    for item in lista or []:
        if "=" not in item:
            sys.exit(f"Formato esperado Z<zona>=<archivo>, recibí '{item}'.")
        clave, valor = item.split("=", 1)
        salida[int(clave.strip().upper().lstrip("Z"))] = valor.strip()
    return salida


def main():
    import pipeline as P

    ap = argparse.ArgumentParser(
        description="Perfiles de generación por zona para los puntos del caso de estudio")
    ap.add_argument("--resolucion", default="horaria", choices=["horaria", "diaria24"],
                    help="horaria (default) escribe artefactos/perfiles_horarios.npz; "
                         "diaria24 escribe los Excel heredados de 24 h")
    ap.add_argument("--fuente-eolico", default="nc",
                    choices=["nc", "openmeteo", "explorador"],
                    help="nc (default) = el mismo viento que genera las fallas")
    ap.add_argument("--fuente-solar", default="openmeteo",
                    choices=["openmeteo", "explorador"],
                    help="el .nc no trae irradiancia, así que el solar no puede "
                         "salir de ahí")
    ap.add_argument("--anios", nargs=2, type=int, default=[2017, 2021],
                    metavar=("INI", "FIN"),
                    help="periodo histórico de Open-Meteo (default: el mismo del "
                         "paso 1 del pipeline climático)")
    ap.add_argument("--horas", type=int, default=8760)
    ap.add_argument("--anio-offset", dest="offset", type=int, default=8760,
                    help="hora inicial dentro del .nc; 8760 = segundo año (2030), "
                         "el mismo corte que usa la matriz de fallas")
    ap.add_argument("--anio-destino", type=int, default=2030,
                    help="calendario del horizonte, para mapear la climatología solar")
    ap.add_argument("--escenario-clima", default="baseline",
                    choices=["baseline", "with_EWE"],
                    help="eje lon del .nc: 'baseline' (default) = viento "
                         "SOSTENIDO, que es el que corresponde a la curva de "
                         "potencia (IEC 61400 la define sobre media de 10 min). "
                         "'with_EWE' = ráfaga, y NO debe usarse acá: infla el "
                         "factor de planta ~3x. La matriz de fallas sí usa "
                         "ráfaga, pero eso no obliga a que el perfil también: "
                         "es el mismo campo de viento y cada uso toma el "
                         "estadístico que le corresponde.")
    ap.add_argument("--altura-buje", type=float, default=ALTURA_BUJE_M)
    ap.add_argument("--puntos", default=None,
                    help="ruta a red_nueva.pkl o SystemzoneV1.xlsx (auto si se omite)")
    ap.add_argument("--viento", default=str(P.RUTA_WIND_NC))
    ap.add_argument("--solar", nargs="*", metavar="Z1=archivo",
                    help="para --fuente-solar explorador: export por zona")
    ap.add_argument("--eolico", nargs="*", metavar="Z1=archivo",
                    help="para --fuente-eolico explorador: export por zona")
    ap.add_argument("--comparar", action="store_true",
                    help="comparar las fuentes disponibles (.nc del estudio, "
                         "Open-Meteo y los exports del Explorador que se pasen "
                         "con --solar/--eolico) y salir sin escribir perfiles. "
                         "Deja artefactos/comparacion_fuentes.csv.")
    ap.add_argument("--forzar", action="store_true")
    ap.add_argument("--estado", action="store_true", help="mostrar estado y salir")
    args = ap.parse_args()

    if args.estado:
        mostrar_estado()
        return

    if args.comparar:
        puntos, origen = puntos_del_estudio(args.puntos)
        tabla = comparar_fuentes(puntos, args)
        if tabla.empty:
            return
        pd.set_option("display.width", 200)
        for recurso in ("eólico", "solar"):
            sub = tabla[tabla["recurso"] == recurso].dropna(axis=1, how="all")
            if sub.empty:
                continue
            print(f"\n── {recurso.upper()} " + "─" * 50)
            print(sub.drop(columns=["recurso"]).to_string(index=False,
                                                          float_format=lambda x: f"{x:.3f}"))
        ART_DIR.mkdir(parents=True, exist_ok=True)
        destino = ART_DIR / "comparacion_fuentes.csv"
        tabla.to_csv(destino, index=False)
        print()
        _log(f"→ {destino}")
        if not (args.solar or args.eolico):
            _log("Faltan los exports del Explorador. Se descargan a mano en "
                 "solar.minenergia.cl y eolico.minenergia.cl (elegir el punto en "
                 "el mapa → descargar serie horaria) y se pasan así:")
            _log("  python construir_perfiles.py --comparar \\")
            _log("      --solar Z1=sol_z1.csv Z2=sol_z2.csv Z3=sol_z3.csv \\")
            _log("      --eolico Z1=eol_z1.csv Z2=eol_z2.csv Z3=eol_z3.csv")
        return

    destinos = ([RUTA_HORARIOS] if args.resolucion == "horaria"
                else [RUTA_SOLAR_XLSX, RUTA_EOLICO_XLSX])
    existentes = [d.name for d in destinos if d.exists()]
    if existentes and not args.forzar:
        sys.exit(f"Ya existe(n) {existentes}. Rehacerlos cambia E1 y por lo tanto "
                 f"todo lo calculado con esos perfiles.\nSi es lo que quieres: --forzar.")

    puntos, origen = puntos_del_estudio(args.puntos)
    zonas = sorted(puntos)
    km = _distancia_max_km(puntos)
    _log(f"{len(zonas)} zona(s), distancia máxima {km:.0f} km, "
         f"horizonte {args.horas} h")

    n_h = args.horas
    solar = np.zeros((n_h, len(zonas)))
    eolico = np.zeros((n_h, len(zonas)))
    viento_nc = None

    # ── EÓLICO ──────────────────────────────────────────────
    if args.fuente_eolico == "nc":
        _log(f"eólico: desde {Path(args.viento).name} [{args.escenario_clima}], "
             f"horas {args.offset}..{args.offset + n_h} — el MISMO viento que "
             f"genera las fallas")
        cf_nc, viento_nc = eolico_desde_nc(args.viento, n_h, args.offset,
                                           args.escenario_clima, args.altura_buje)
        if cf_nc.shape[1] < len(zonas):
            sys.exit(f"El .nc trae {cf_nc.shape[1]} zonas y el estudio tiene "
                     f"{len(zonas)}. Regenerar el pipeline climático.")
        eolico = cf_nc[:, :len(zonas)]
        viento_nc = viento_nc[:, :len(zonas)]
    elif args.fuente_eolico == "openmeteo":
        for i, z in enumerate(zonas):
            lat, lon = puntos[z]
            _log(f"Z{z}: Open-Meteo eólico {args.anios[0]}-{args.anios[1]}...")
            df = descargar_openmeteo(lat, lon, *args.anios)
            eolico[:, i] = climatologia_horaria(
                df["time"], cf_eolico(df["wind"], args.altura_buje),
                n_h, args.anio_destino)
    else:
        mapa = _parse_asignaciones(args.eolico)
        if not mapa:
            sys.exit("--fuente-eolico explorador requiere --eolico Z1=archivo ...")
        for i, z in enumerate(zonas):
            if z not in mapa:
                sys.exit(f"Falta el archivo eólico de la zona {z}.")
            _log(f"Z{z}: leyendo {Path(mapa[z]).name} (eólico)...")
            df = leer_export_explorador(mapa[z])
            eolico[:, i] = climatologia_horaria(
                df["time"], _cf_desde_export(df, "eólico", args.altura_buje),
                n_h, args.anio_destino)

    # ── SOLAR ───────────────────────────────────────────────
    if args.fuente_solar == "openmeteo":
        for i, z in enumerate(zonas):
            lat, lon = puntos[z]
            _log(f"Z{z}: Open-Meteo solar {args.anios[0]}-{args.anios[1]} "
                 f"({lat:.4f}, {lon:.4f})...")
            df = descargar_openmeteo(lat, lon, *args.anios)
            solar[:, i] = climatologia_horaria(
                df["time"], cf_solar(df["ghi"], df["t2m"]), n_h, args.anio_destino)
    else:
        mapa = _parse_asignaciones(args.solar)
        if not mapa:
            sys.exit("--fuente-solar explorador requiere --solar Z1=archivo ...")
        for i, z in enumerate(zonas):
            if z not in mapa:
                sys.exit(f"Falta el archivo solar de la zona {z}.")
            _log(f"Z{z}: leyendo {Path(mapa[z]).name} (solar)...")
            df = leer_export_explorador(mapa[z])
            solar[:, i] = climatologia_horaria(
                df["time"], _cf_desde_export(df, "solar", args.altura_buje),
                n_h, args.anio_destino)

    # ── Diagnóstico y escritura ─────────────────────────────
    disp_sol, medios_sol = _informe("solar", solar, km)
    disp_eol, medios_eol = _informe("eólico", eolico, km, viento_nc)

    meta = {
        "resolucion": args.resolucion,
        "fuente_solar": args.fuente_solar,
        "fuente_eolico": args.fuente_eolico,
        "origen_puntos": str(origen),
        "puntos": {str(z): list(puntos[z]) for z in zonas},
        "zonas": zonas,
        "distancia_max_km": round(km, 1),
        "anios_openmeteo": args.anios,
        "anio_destino": args.anio_destino,
        "escenario_clima": args.escenario_clima,
        "anio_offset": args.offset,
        "altura_buje_m": args.altura_buje,
        "cf_medio": {"solar": [float(v) for v in medios_sol],
                     "eolico": [float(v) for v in medios_eol]},
    }

    if args.resolucion == "horaria":
        ART_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(RUTA_HORARIOS,
                            solar_por_zona=solar, eolico_por_zona=eolico,
                            zonas=np.array(zonas, dtype=int))
        _log(f"→ {RUTA_HORARIOS.name} ({RUTA_HORARIOS.stat().st_size/1e6:.2f} MB) "
             f"solar {solar.shape}, eólico {eolico.shape}")
    else:
        # Colapsar a 24 h para el formato heredado. Se usa el percentil 90 de
        # cada hora y no el promedio porque es lo que reproduce la convención
        # del Excel vigente (peak 1,03 = día despejado, no promedio anual).
        horas_del_dia = pd.date_range(f"{args.anio_destino}-01-01",
                                      periods=n_h, freq="h").hour
        def _colapsar(arr):
            return {zonas[i]: np.array([
                float(np.percentile(arr[horas_del_dia == h, i], 90))
                for h in range(24)]) for i in range(len(zonas))}
        for perf, ruta, escala in ((_colapsar(solar), RUTA_SOLAR_XLSX, ESCALA_SOLAR_XLSX),
                                   (_colapsar(eolico), RUTA_EOLICO_XLSX, ESCALA_EOLICO_XLSX)):
            df = pd.DataFrame({f"Z{z}": p * escala for z, p in sorted(perf.items())})
            df.to_excel(ruta, index=False)
            _log(f"→ {ruta.name}")
        meta["colapso_24h"] = "percentil 90 por hora del día"

    (ART_DIR / "perfiles_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    _log("→ artefactos/perfiles_meta.json")
    _log("siguiente paso:  python pipeline.py --fuente red_nueva --etapas E1 --forzar")


if __name__ == "__main__":
    main()
