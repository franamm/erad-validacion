# ============================================================
# crear_caso.py — La pestaña "🗺️ Origen de los datos", sin interfaz
# ============================================================
# Un caso nuevo (otra red) nace de dos archivos que tienen que hablar de las
# MISMAS zonas: `artefactos/<caso>/red_nueva.pkl`, que lee
# `pipeline.py --fuente red_nueva`, y `SystemzoneV1_<caso>.xlsx`, que lee la
# cadena climática. Hasta ahora solo la pestaña de Streamlit los escribía,
# así que en un servidor sin pantalla no había cómo empezar un caso.
#
# Este script hace lo mismo que el asistente de la pestaña, con la misma
# lógica (`entrada_red.py`, que no importa streamlit):
#
#   1. carga la red (CIGRE, pandapower, OpenDSS o CSV);
#   2. la adapta a lo que el MILP tiene escrito a fuego (buses contiguos,
#      cabeceras en 1 y 12, índices desde 0) y la valida;
#   3. fija zona por bus y un punto geográfico por zona;
#   4. escribe las dos salidas, más `crear_caso.json` con todo lo decidido.
#
# Numeración. Los buses y líneas que se pasan por línea de comandos
# (--criticos, --seccionadores, --ties, el CSV de --zonas) van en la
# numeración del ARCHIVO DE ENTRADA, que es la que la usuaria conoce. La
# adaptación renumera, así que el script traduce y deja la tabla de
# equivalencias en `crear_caso.json`: el bus 39 del archivo puede ser el 1
# del modelo, y cruzar resultados por número sin esa tabla daría cualquier
# cosa.
#
# USO
#   # 1) Mirar la red sin escribir nada
#   python crear_caso.py mi_red --red red.json --revisar
#
#   # 2) Crear el caso
#   python crear_caso.py mi_red --red red.json \
#       --zonas zonas.csv --puntos puntos.csv \
#       --criticos 5,17 --seccionadores ninguno \
#       --escenarios 100 --horas-reparacion 3 \
#       --perfil-demanda perfiles/demanda_mi_red.csv
#
#   # Después, el resto de la cadena: servidor/correr_caso.sh
# ============================================================

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import casos
import entrada_red as ER

from rutas import CASO_DIR, DATOS_DIR

APP_DIR = Path(__file__).resolve().parent
DIR_CLIMA_INPUTS = CASO_DIR / "Proyecto oficial" / "Inputs"

for _flujo in (sys.stdout, sys.stderr):      # ver pipeline._forzar_utf8_en_consola
    try:
        _flujo.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _log(msg=""):
    print(f"[crear_caso] {msg}" if msg else "", flush=True)


def _abortar(msg):
    sys.exit(f"\n[crear_caso] ERROR: {msg}")


def _lista_int(txt, nombre):
    """'1,4,7' / '0-5' / 'ninguno' -> lista. None si no se pasó."""
    if txt is None:
        return None
    txt = txt.strip().lower()
    if txt in ("", "ninguno", "ninguna", "none", "-"):
        return []
    out = []
    try:
        for parte in txt.split(","):
            parte = parte.strip()
            if "-" in parte:
                a, b = parte.split("-")
                out.extend(range(int(a), int(b) + 1))
            elif parte:
                out.append(int(parte))
    except ValueError:
        _abortar(f"--{nombre}: '{txt}' no es una lista de enteros (ej: 1,4,7 o 0-5).")
    return sorted(set(out))


def _leer_tabla(ruta):
    ruta = Path(ruta)
    if not ruta.exists():
        _abortar(f"No existe {ruta}")
    if ruta.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(ruta)
    return pd.read_csv(ruta)


# ════════════════════════════════════════════════════════════
# 1. Carga
# ════════════════════════════════════════════════════════════

def cargar_red(args):
    """(net, origen, avisos, extras). Marca la numeración original en `idx_original`.

    `extras` trae lo que un paquete del pipeline ya decidía (sus tie-lines),
    en numeración original.
    """
    avisos, extras = [], {}
    fmt = args.formato
    if fmt == "auto":
        if args.red == "cigre":
            fmt = "cigre"
        elif args.csv_buses:
            fmt = "csv"
        elif args.red and Path(args.red).suffix.lower() == ".dss":
            fmt = "opendss"
        else:
            fmt = "pandapower"

    if fmt == "cigre":
        net = ER.red_cigre(aerea=not args.cigre_subterranea)
        origen = "CIGRE MV de pandapower" + ("" if args.cigre_subterranea else " (línea aérea)")
    elif fmt == "pandapower":
        if not args.red:
            _abortar("--red es obligatorio para el formato pandapower.")
        net = ER.desde_pandapower(args.red)
        origen = f"pandapower: {Path(args.red).name}"
        # Un paquete del pipeline (red.pkl / red_nueva.pkl de otro caso) trae
        # las zonas en `bus_zone`, no en una columna de la red.
        if Path(args.red).suffix.lower() in (".pkl", ".p", ".pickle"):
            import pickle
            try:
                with open(args.red, "rb") as f:
                    obj = pickle.load(f)
            except Exception:                                   # noqa: BLE001
                obj = None
            if isinstance(obj, dict) and obj.get("bus_zone") and "zona" not in net.bus.columns:
                net.bus["zona"] = [int(obj["bus_zone"].get(int(b), 1)) for b in net.bus.index]
                origen += " (zonas del paquete)"
            if isinstance(obj, dict) and obj.get("tie_indices"):
                extras["tie_indices"] = [int(l) for l in obj["tie_indices"]]
    elif fmt == "opendss":
        if not args.red:
            _abortar("--red es obligatorio para OpenDSS (ruta al Master.dss).")
        try:
            net, avisos = ER.desde_opendss(args.red)
        except ImportError:
            _abortar("OpenDSS necesita `pip install opendssdirect.py`.")
        origen = f"OpenDSS: {Path(args.red).name}"
    elif fmt == "csv":
        if not (args.csv_buses and args.csv_lineas and args.csv_cargas):
            _abortar("El formato csv necesita --csv-buses, --csv-lineas y --csv-cargas "
                     f"(plantillas en entrada_red.PLANTILLA_CSV).")
        dfb = _leer_tabla(args.csv_buses)
        dfl = _leer_tabla(args.csv_lineas)
        net = ER.desde_csv(dfb, dfl, _leer_tabla(args.csv_cargas),
                           sn_mva_trafo=args.sn_mva_trafo or 25.0)
        # desde_csv crea los buses en orden; la numeración "original" es la
        # columna `bus` del CSV, no la posición.
        net.bus["idx_original"] = dfb["bus"].astype(int).to_numpy()
        if "line" in dfl.columns:
            net.line["idx_original"] = dfl["line"].astype(int).to_numpy()
        if not args.sn_mva_trafo:
            avisos.append("CSV sin --sn-mva-trafo: se declaró una cabecera de 25 MVA.")
        origen = f"CSV: {Path(args.csv_buses).name}"
    else:
        _abortar(f"Formato desconocido: {fmt}")

    if "idx_original" not in net.bus.columns:
        net.bus["idx_original"] = [int(b) for b in net.bus.index]
    if "idx_original" not in net.line.columns:
        net.line["idx_original"] = [int(l) for l in net.line.index]

    if net.trafo.empty:
        if args.sn_mva_trafo:
            ER.asegurar_trafo(net, args.sn_mva_trafo)
            avisos.append(f"La red no traía transformador: se declaró uno de "
                          f"{args.sn_mva_trafo} MVA entre los dos primeros buses.")
        else:
            _abortar("La red no trae transformador de cabecera y el MILP lee "
                     "net.trafo.sn_mva[0] como capacidad. Declararla con --sn-mva-trafo.")
    return net, origen, avisos, extras


# ════════════════════════════════════════════════════════════
# 2. Zonas y puntos
# ════════════════════════════════════════════════════════════

def marcar_zonas(net, ruta, n_zonas):
    """Escribe net.bus['zona'] ANTES de adaptar, en numeración original.

    Así la columna viaja con la renumeración y no hay que traducir nada.
    Devuelve el origen del reparto, para el registro.
    """
    if ruta:
        df = _leer_tabla(ruta)
        col_z = next((c for c in df.columns if str(c).lower() in ("zona", "zone")), None)
        if col_z is None:
            _abortar(f"{ruta}: falta la columna 'zona'.")
        if "bus" in df.columns:
            mapa = dict(zip(df["bus"].astype(int), df[col_z]))
            por_bus = net.bus["idx_original"].map(mapa)
        elif "name" in df.columns:
            mapa = dict(zip(df["name"].astype(str), df[col_z]))
            por_bus = net.bus["name"].astype(str).map(mapa)
        else:
            _abortar(f"{ruta}: necesita una columna 'bus' (índice del archivo) o 'name'.")
        faltan = net.bus.loc[por_bus.isna(), "idx_original"].tolist()
        if faltan:
            _abortar(f"{ruta} no asigna zona a {len(faltan)} bus(es): {faltan[:20]}"
                     f"{' ...' if len(faltan) > 20 else ''}")
        net.bus["zona"] = [ER._a_zona(v) for v in por_bus]
        return f"archivo {Path(ruta).name}"
    if "zona" in net.bus.columns and net.bus["zona"].notna().all():
        return "columna 'zona' de la red"
    if n_zonas:
        return f"reparto automático en {n_zonas} tramos de índice (REVISAR)"
    _abortar("Falta el reparto de zonas: --zonas archivo.csv (columnas bus,zona), "
             "o --n-zonas N para un reparto automático por índice (solo para probar).")


def leer_puntos(ruta, net, bus_zone):
    """{zona: (lat, lon)} desde archivo, o desde las coordenadas de la red."""
    if ruta:
        df = _leer_tabla(ruta)
        cols = {str(c).lower(): c for c in df.columns}
        if not {"lat", "lon"} <= set(cols):
            _abortar(f"{ruta}: necesita columnas Zona, Lat, Lon.")
        df = df.dropna(subset=[cols["lat"], cols["lon"]]).reset_index(drop=True)
        zonas = (df[cols["zona"]].astype(int) if "zona" in cols
                 else pd.Series(np.arange(1, len(df) + 1)))
        comunas = ({int(z): str(c) for z, c in zip(zonas, df[cols["comuna"]]) if pd.notna(c)}
                   if "comuna" in cols else {})
        return ({int(z): (float(a), float(b)) for z, a, b in
                 zip(zonas, df[cols["lat"]], df[cols["lon"]])},
                comunas, f"archivo {Path(ruta).name}")
    puntos = ER.puntos_por_zona(net, bus_zone)
    return puntos, {}, "centroide de las coordenadas de la red"


# ════════════════════════════════════════════════════════════
# 3. Traducción de numeración
# ════════════════════════════════════════════════════════════

def traductor(df, nombre):
    """original -> índice del modelo, para la tabla `df` (bus o line)."""
    mapa = {int(o): int(i) for i, o in zip(df.index, df["idx_original"])}

    def traducir(lista, flag):
        if lista is None:
            return None
        ajenos = [x for x in lista if x not in mapa]
        if ajenos:
            _abortar(f"--{flag}: {ajenos} no son {nombre} del archivo de entrada.")
        return sorted(mapa[x] for x in lista)
    return mapa, traducir


# ════════════════════════════════════════════════════════════
# 4. Perfiles
# ════════════════════════════════════════════════════════════

def revisar_perfiles(args, n_horas):
    """Rutas de perfiles del caso + avisos. Las relativas a app/ se guardan
    relativas, para que el paquete no amarre rutas de una máquina."""
    perfiles, avisos = {}, []
    for clave, ruta in (("demanda", args.perfil_demanda), ("solar", args.perfil_solar),
                        ("eolico", args.perfil_eolico)):
        if not ruta:
            continue
        p = Path(ruta).resolve()
        if not p.exists():
            _abortar(f"--perfil-{clave}: no existe {p}")
        if clave == "demanda":
            df = pd.read_csv(p)
            if "Power" not in df.columns:
                _abortar(f"{p.name}: E1 lee la columna 'Power' (factor por unidad "
                         f"que multiplica a P_D_base) y este CSV no la tiene.")
            if len(df) < n_horas:
                _abortar(f"{p.name}: {len(df)} filas, pero el horizonte es {n_horas} h. "
                         f"E1 toma el horizonte del largo de este archivo.")
            if df["Power"].max() > 1.5:
                avisos.append(f"Demanda: 'Power' llega a {df['Power'].max():.2f}. Es un "
                              f"factor sobre P_D_base, no kW: revisa que esté normalizado.")
        try:
            perfiles[clave] = str(p.relative_to(DATOS_DIR))
        except ValueError:
            perfiles[clave] = str(p)
    if "demanda" not in perfiles:
        avisos.append("Sin --perfil-demanda: el caso hereda Perfil_demanda.csv, la curva "
                      "de carga de la red de 15 barras. Sirve para probar el modelo, no "
                      "para concluir.")
    if not ({"solar", "eolico"} <= set(perfiles)):
        avisos.append("Solar/eólico: E1 abre los Excel de perfiles aunque después los "
                      "reemplace con perfiles_horarios.npz (construir_perfiles.py). Sin "
                      "esa etapa, el caso usa los perfiles de 24 h del estudio.")
    return perfiles, avisos


# ════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Crea un caso de estudio (red_nueva.pkl + SystemzoneV1) sin la app.")
    ap.add_argument("caso", help="nombre de la carpeta en artefactos/ (ej: mi_red)")
    g = ap.add_argument_group("red")
    g.add_argument("--red", help="archivo de red (.json/.p/.xlsx pandapower, Master.dss) o 'cigre'")
    g.add_argument("--formato", default="auto",
                   choices=["auto", "cigre", "pandapower", "opendss", "csv"])
    g.add_argument("--csv-buses"), g.add_argument("--csv-lineas"), g.add_argument("--csv-cargas")
    g.add_argument("--sn-mva-trafo", type=float, default=None,
                   help="capacidad de la cabecera si la red no trae transformador [MVA]")
    g.add_argument("--cigre-subterranea", action="store_true",
                   help="CIGRE con sus cables originales (por defecto se pasa a línea aérea)")

    g = ap.add_argument_group("zonas y puntos (numeración del archivo de entrada)")
    g.add_argument("--zonas", help="CSV/XLSX con columnas bus (o name) y zona")
    g.add_argument("--n-zonas", type=int, default=None,
                   help="reparto automático por índice, solo para probar")
    g.add_argument("--puntos", help="CSV/XLSX con Zona, Lat, Lon (y opcional Comuna)")
    g.add_argument("--sin-puntos", action="store_true",
                   help="permitir un caso sin puntos geográficos: no podrá usar la "
                        "cadena climática, solo fallas_sinteticas.py")

    g = ap.add_argument_group("parámetros del estudio")
    g.add_argument("--escenarios", type=int, default=100, help="escenarios Monte Carlo")
    g.add_argument("--horas", type=int, default=8760, help="horizonte")
    g.add_argument("--n-eventos", type=int, default=1,
                   help="multiplicador anual del costo de ENS")
    g.add_argument("--horas-reparacion", type=int, required=False, default=None,
                   help="horas que tarda en reponerse una línea (entra en SAIDI/ENS)")
    g.add_argument("--criticos", default=None,
                   help="buses críticos, ej 5,17, o 'ninguno'. Obligatorio: no hay "
                        "default que valga en otra red")
    g.add_argument("--ties", default="auto",
                   help="enlaces de respaldo: 'auto' (líneas con switch abierto), "
                        "una lista, o 'ninguno'")
    g.add_argument("--seccionadores", default=None,
                   help="líneas con seccionador, o 'ninguno'. Obligatorio")

    g = ap.add_argument_group("perfiles del caso")
    g.add_argument("--perfil-demanda", help="CSV con columna Power (p.u. de P_D_base)")
    g.add_argument("--perfil-solar", help="XLSX con columnas Z1..Zn (perfil 24 h, W)")
    g.add_argument("--perfil-eolico", help="XLSX con columnas Z1..Zn (perfil 24 h, p.u.)")

    ap.add_argument("--revisar", action="store_true",
                    help="mostrar la red adaptada y la validación, sin escribir nada")
    ap.add_argument("--forzar", action="store_true", help="sobrescribir un caso existente")
    args = ap.parse_args()

    ruta_caso, problema = casos.nombre_valido(args.caso)
    if ruta_caso is None:
        _abortar(problema)
    if casos.es_validacion(ruta_caso):
        _abortar("Ese es el caso de validación de la tesis: no se pisa desde acá.")
    paquete_ruta = ruta_caso / "red_nueva.pkl"
    if paquete_ruta.exists() and not (args.forzar or args.revisar):
        _abortar(f"{paquete_ruta} ya existe. --forzar para rehacerlo (invalida lo "
                 f"calculado aguas abajo: fallas, E4, E5 y barridos de este caso).")

    # ── 1. red ──────────────────────────────────────────────────────
    net, origen, avisos, extras = cargar_red(args)
    _log(f"red: {origen} — {len(net.bus)} buses, {len(net.line)} líneas, "
         f"{len(net.load)} cargas")
    reparto = marcar_zonas(net, args.zonas, args.n_zonas)

    net, cambios, ties_auto = ER.adaptar_para_milp(net)
    for c in cambios:
        _log(f"adaptación: {c}")

    mapa_bus, tr_bus = traductor(net.bus, "buses")
    mapa_lin, tr_lin = traductor(net.line, "líneas")

    errores = [m for n, m in ER.validar(net) if n == "error"]
    for nivel, msg in ER.validar(net):
        _log(f"validación [{nivel}]: {msg}")
    if errores:
        _abortar("La red no cumple lo que el MILP exige (ver arriba).")

    # ── 2. zonas ────────────────────────────────────────────────────
    # No `ER.bus_zone_inicial`: esa función es el punto de partida de una
    # tabla editable y, si la columna trae UNA sola zona, la descarta y
    # reparte en tramos. Acá no hay tabla que corregir después, y una red
    # entera en una sola zona climática es una respuesta válida.
    if "zona" in net.bus.columns and net.bus["zona"].notna().all():
        bus_zone = {int(b): ER._a_zona(net.bus.at[b, "zona"]) for b in net.bus.index}
    else:
        bus_zone = ER.bus_zone_inicial(net, args.n_zonas or 3)
    zonas = sorted(set(bus_zone.values()))
    if zonas != list(range(1, len(zonas) + 1)):
        # construir_faultlin toma el viento de la columna zona-1 del .nc, y
        # el .nc tiene una fila por zona en orden: zonas {1, 3} harían que la
        # zona 3 lea el viento de una columna que no existe (y se recorte a
        # la última, sin error).
        _abortar(f"Las zonas tienen que ser 1..Z sin huecos; vienen {zonas}.")
    res = ER.resumen(net)
    lineas_zona = pd.Series([bus_zone[int(net.line.from_bus[l])] for l in net.line.index])
    _log(f"zonas ({reparto}):")
    pdb = ER.p_d_base(net)
    for z in zonas:
        bz = [b for b, zz in bus_zone.items() if zz == z]
        _log(f"   Z{z}: {len(bz):>4} buses  {int((lineas_zona == z).sum()):>4} líneas  "
             f"{sum(pdb[b] for b in bz):>10,.0f} kW")
        if not (lineas_zona == z).any():
            avisos.append(f"La zona {z} no tiene ninguna línea (las líneas toman la zona "
                          f"de su from_bus): su punto climático no afecta a nada.")
    _log(f"demanda base {res['demanda_kw']:,.0f} kW · capacidad cabeceras "
         f"{(res['capacidad_kw'] or 0):,.0f} kW · {res['largo_km']:.1f} km de línea")

    puntos, comunas, origen_puntos = leer_puntos(args.puntos, net, bus_zone)
    sin_punto = [z for z in zonas if z not in puntos]
    if sin_punto and not args.sin_puntos:
        _abortar(f"Zonas sin punto geográfico: {sin_punto}. La cadena climática "
                 f"consulta ARClim y Open-Meteo con lat/lon reales: pasar --puntos "
                 f"(Zona,Lat,Lon). Con --sin-puntos el caso solo admite fallas sintéticas.")
    fuera = [z for z, (la, lo) in puntos.items() if not ER._en_chile(la, lo)]
    if fuera:
        _abortar(f"Puntos fuera de Chile en zonas {fuera}: ARClim solo cubre Chile "
                 f"(¿lat/lon invertidos?).")
    if puntos:
        _log(f"puntos ({origen_puntos}): "
             + "  ".join(f"Z{z}=({la:.4f}, {lo:.4f})" for z, (la, lo) in sorted(puntos.items())))

    # ── 3. parámetros en numeración del modelo ──────────────────────
    if args.criticos is None:
        _abortar("--criticos es obligatorio (una lista o 'ninguno'). Los críticos "
                 "multiplican su VoLL con la priorización y quedan fuera de EDRP.")
    if args.seccionadores is None:
        _abortar("--seccionadores es obligatorio (una lista o 'ninguno'). Vacío = "
                 "una falla deja fuera el alimentador completo en la variante realista.")
    criticos = tr_bus(_lista_int(args.criticos, "criticos"), "criticos")
    secc = tr_lin(_lista_int(args.seccionadores, "seccionadores"), "seccionadores")
    if args.ties.strip().lower() == "auto" and extras.get("tie_indices"):
        ties = tr_lin(extras["tie_indices"], "ties")
        _log(f"tie-lines del paquete de entrada: {extras['tie_indices']}")
    elif args.ties.strip().lower() == "auto":
        ties = list(ties_auto)
        if not ties:
            avisos.append("La red no trae switches abiertos: sin enlaces de respaldo, "
                          "la reconfiguración topológica no tiene qué cerrar. Pasar --ties.")
    else:
        ties = tr_lin(_lista_int(args.ties, "ties"), "ties")
    choque = set(ties) & set(secc)
    if choque:
        _abortar(f"Líneas a la vez enlace de respaldo y seccionador: {sorted(choque)}.")

    horas_rep = args.horas_reparacion
    if horas_rep is None:
        avisos.append("Sin --horas-reparacion: E2 tomará el valor con el que corra "
                      "construir_faultlin.py (--horas-espera).")
    perfiles, av_perf = revisar_perfiles(args, args.horas)
    avisos += av_perf

    _log()
    _log(f"escenarios MC {args.escenarios} · horizonte {args.horas} h · N_eventos "
         f"{args.n_eventos} · reparación {horas_rep if horas_rep else '(de faultlin)'} h")
    _log(f"críticos (modelo) {criticos} · tie-lines (modelo) {ties} · "
         f"seccionadores (modelo) {secc}")
    for a in avisos:
        _log(f"[!] {a}")

    if args.revisar:
        _log()
        _log("--revisar: no se escribió nada.")
        return

    # ── 4. salidas ──────────────────────────────────────────────────
    ruta_caso.mkdir(parents=True, exist_ok=True)
    (ruta_caso / ".caso").touch()
    net_geo = ER.aplicar_geo(net, bus_zone, puntos) if puntos else net
    ER.guardar_paquete(
        net_geo, bus_zone, puntos,
        numesc=args.escenarios, n_eventos=args.n_eventos, buses_criticos=criticos,
        tie_indices=ties, n_horas=args.horas, comunas=comunas,
        horas_reparacion=horas_rep, seccionadores=secc, perfiles=perfiles,
        ruta=paquete_ruta, origen=origen)
    _log(f"→ {paquete_ruta}")

    ruta_sz = None
    if puntos:
        ruta_sz = ER.exportar_systemzone(
            puntos, DIR_CLIMA_INPUTS / f"SystemzoneV1_{ruta_caso.name}.xlsx", comunas)
        _log(f"→ {ruta_sz}")

    registro = {
        "caso": ruta_caso.name,
        "creado": time.strftime("%Y-%m-%d %H:%M:%S"),
        "comando": sys.argv,
        "origen_red": origen,
        "reparto_zonas": reparto,
        "origen_puntos": origen_puntos if puntos else None,
        "adaptacion": cambios,
        "avisos": avisos,
        "parametros": {"escenarios": args.escenarios, "horas": args.horas,
                       "n_eventos": args.n_eventos, "horas_reparacion": horas_rep,
                       "criticos": criticos, "tie_indices": ties,
                       "seccionadores": secc, "perfiles": perfiles},
        "systemzone": str(ruta_sz) if ruta_sz else None,
        "resumen": res,
        # original -> modelo. Imprescindible para leer resultados por número.
        "bus_original_a_modelo": {str(k): v for k, v in sorted(mapa_bus.items())},
        "linea_original_a_modelo": {str(k): v for k, v in sorted(mapa_lin.items())},
    }
    (ruta_caso / "crear_caso.json").write_text(
        json.dumps(registro, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _log(f"→ {ruta_caso / 'crear_caso.json'} (parámetros y tabla de numeración)")


if __name__ == "__main__":
    main()
