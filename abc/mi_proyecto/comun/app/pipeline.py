# ============================================================
# pipeline.py — Construcción reproducible de los insumos de la app
# ============================================================
# Reemplaza la cadena implícita
#     Prueba 12 oficial.ipynb → app.pkl → caso_base_sin_fallas.py
#     → caso_base.py → data_loader.py
# por etapas explícitas, idempotentes y ejecutables por separado.
#
# USO
#   python pipeline.py --estado                 # qué hay y qué falta
#   python pipeline.py --etapas E0,E1,E2,E3     # datos (segundos, sin solver)
#   python pipeline.py --etapas E4              # caso sin fallas (3 corridas)
#   python pipeline.py --etapas E5              # baseline por escenario
#   python pipeline.py --etapas E5 --escenarios 0-19
#   python pipeline.py --etapas E2 --forzar     # recalcular una etapa
#
# INVARIANTES (no romper)
#   1. NUNCA se escribe app.pkl. Es el registro histórico del estudio
#      congelado y es de solo lectura para este script.
#   2. La fuente de verdad de las fallas es FaultlinMC.npy, NO el
#      app.pkl: la clave data["faultlinMC"] del pickle quedó en CEROS
#      cuando caso_base_sin_fallas.py hizo su dump (línea 158 sobre el
#      faultlinMC anulado de la línea 25). Hoy la app solo funciona
#      porque data_loader.py:75 recarga el .npy encima.
#   3. Este módulo NO importa data_loader: esa dependencia es justamente
#      la circularidad que el pipeline viene a romper.
# ============================================================

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import casos
import ramales as RAM

APP_DIR = Path(__file__).resolve().parent
from rutas import CASO_DIR, DATOS_DIR  # datos del caso activo (ver rutas.py)

# ART_DIR, BASE_DIR, RUTA_RED_NUEVA, RUTA_FAULTLIN y RUTA_FAULTLIN_MIT las fija
# `usar_artefactos`, que se llama al final del bloque de constantes. Ojo: NO
# reasignar ninguna de las cinco a mano. Hasta el 2026-09-22 `ART_DIR` era una
# constante y quien quería trabajar otra red hacía `pipeline.ART_DIR = otra`,
# que parecía funcionar pero dejaba `BASE_DIR` apuntando a la carpeta original:
# `base_dir()`, `escenarios_disponibles()` y `variantes_disponibles()` seguían
# leyendo los baselines del caso viejo, en silencio.


def _forzar_utf8_en_consola():
    """La consola de Windows usa cp1252 y revienta con '→', 'Σ' o '⚠️'.

    No es cosmético: `print` lanza UnicodeEncodeError y mata el proceso. Un
    barrido de horas no puede morir por un carácter de un mensaje de
    progreso, y parchear los mensajes uno por uno solo garantiza que el
    próximo que alguien escriba vuelva a romper. Se arregla el flujo, no
    los textos.

    Se hace al importar (no en `main`) para cubrir de una vez a todos los
    puntos de entrada de consola que dependen de este módulo: pipeline.py,
    run_barrido_medidas.py y run_estocastico.py. Bajo Streamlit stdout
    puede no soportar reconfigure(); ahí no se hace nada y no pasa nada,
    porque Streamlit maneja Unicode por su cuenta.
    """
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_forzar_utf8_en_consola()


def base_dir(variante="base", art_dir=None):
    """Carpeta de baselines del supuesto de aislación `variante`.

    Dos supuestos de aislación distintos NO son comparables entre sí: si
    el caso base se calcula con aislación perfecta y la estrategia con
    aislación realista, el "ahorro de ENS" sale negativo aunque la medida
    sirva. Por eso cada variante guarda sus propios baselines en vez de
    sobrescribir los de la otra — además, la comparación entre ambos
    supuestos es un resultado en sí.

    La variante "base" (sin aislación) apunta a la carpeta histórica, así
    que los baselines ya calculados siguen siendo válidos.
    """
    base = Path(art_dir) / "baselines" if art_dir is not None else BASE_DIR
    return base if variante == "base" else base / variante


def clientes_denominador_saifi(red, art_dir=None):
    """Total comunal configurado; admite paquetes antiguos sin la clave."""
    valor = red.get("clientes_denominador")
    if valor is not None:
        return float(valor)
    ruta = Path(art_dir or ART_DIR) / "saifi_config.json"
    if ruta.exists():
        try:
            return float(json.loads(ruta.read_text(encoding="utf-8"))[
                "clientes_denominador"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return None

def ruta_sin_fallas(etiqueta="", art_dir=None):
    """Artefacto de E4 (caso sin fallas), opcionalmente etiquetado.

    Mismo criterio que `base_dir` para los baselines: dos supuestos de
    modelación distintos NO son comparables, así que cada uno guarda su
    propio artefacto en vez de pisar al otro.

    El caso concreto que obligó a esto es el tope de capacidad de DG
    (`lim_dg`): el CAPEX/OPEX de DG sin fallas que E4 entrega al objetivo
    depende de cuánta capacidad se dejó instalar, así que un E4 calculado
    con 0.3·P_D_base no sirve para una estrategia que puede instalar hasta
    1.0·P_D_base. Cada valor de `lim_dg` merece su propia etiqueta, y el
    valor usado queda registrado en `salida["_meta"]["lim_dg"]`.

    Sin etiqueta apunta al nombre histórico, así que nada de lo que ya
    existe cambia de sitio.
    """
    return Path(art_dir or ART_DIR) / (f"sin_fallas_{etiqueta}.pkl" if etiqueta else "sin_fallas.pkl")


def cargar_sin_fallas(etiqueta=""):
    ruta = ruta_sin_fallas(etiqueta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el artefacto de E4 '{ruta.name}'. Generarlo con:\n"
            f"    python pipeline.py --etapas E4"
            + (f" --etiqueta {etiqueta}" if etiqueta else ""))
    with open(ruta, "rb") as f:
        return pickle.load(f)


def sin_fallas_disponibles(art_dir=None):
    """Etiquetas de E4 con artefacto calculado ('' = el histórico)."""
    carpeta = Path(art_dir or ART_DIR)
    out = [""] if (carpeta / "sin_fallas.pkl").exists() else []
    out += sorted(p.stem[len("sin_fallas_"):] for p in carpeta.glob("sin_fallas_*.pkl"))
    return out


# ════════════════════════════════════════════════════════════
# Archivo de resultados de la app (.pkl.gz)
# ════════════════════════════════════════════════════════════
# La app exporta e importa su historial en este formato. El barrido escribe
# EXACTAMENTE el mismo, para que una corrida de consola se pueda abrir en la
# pestaña Resultados sin código nuevo. La firma vive acá, y no duplicada en
# app_v7, porque si las dos versiones divergen la app rechaza en silencio los
# archivos que ella misma podría leer.

VERSION_ARCHIVO_RESULTADOS = 2


def firma_modelo():
    """Datos mínimos para detectar resultados de un modelo incompatible."""
    red = _cargar("red.pkl")
    fallas = np.load(ART_DIR / "fallas.npz")
    n_horas = int(fallas["faultlinMC"].shape[0])
    n_esc, n_buses = int(red["numesc"]), len(red["BUSES"])
    return {
        "N_HORAS": n_horas,
        "numesc": n_esc,
        "BUSES": [int(x) for x in red["BUSES"]],
        "LINES": [int(x) for x in red["LINES"]],
        "shape_faultlinMC": tuple(int(x) for x in fallas["faultlinMC"].shape),
        "shape_demand_factor": (n_horas, n_esc, n_buses),
        "P_D_base": {int(k): float(v) for k, v in red["P_D_base"].items()},
    }


def escribir_historial(ruta, historial, firma=None):
    """Escribe un historial en el .pkl.gz que lee la app.

    compresslevel=1 como en la app: comprime 16x los dicts horarios y tarda
    la mitad que el nivel 6, que solo gana un 18% de tamaño.
    """
    import gzip
    payload = {
        "version_archivo": VERSION_ARCHIVO_RESULTADOS,
        "fecha_exportacion": time.strftime("%Y-%m-%d %H:%M:%S"),
        "firma_modelo": firma if firma is not None else firma_modelo(),
        "historial": historial,
    }
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(ruta), mode="wb", compresslevel=1, mtime=0) as g:
        pickle.dump(payload, g, protocol=pickle.HIGHEST_PROTOCOL)
    return ruta


def leer_historial(ruta):
    """Lee un .pkl.gz de resultados y devuelve su `historial`."""
    import gzip
    with gzip.GzipFile(filename=str(ruta), mode="rb") as g:
        payload = pickle.load(g)
    if not isinstance(payload, dict) or "historial" not in payload:
        raise ValueError(f"'{Path(ruta).name}' no contiene un historial válido.")
    return payload["historial"]


RUTA_APP_PKL = DATOS_DIR / "app.pkl"                 # solo lectura
RUTA_WIND_NC = CASO_DIR / "Scenario Database" / "Inputs" / "BaseHoraEWEPELP20z.nc"
RUTA_PERFIL_SOLAR = DATOS_DIR / "Perfiles_solares_promedio_por_zona.xlsx"
RUTA_PERFIL_EOLICO = DATOS_DIR / "Perfiles_eolicos_promedio_por_zona.xlsx"
RUTA_PERFIL_DEMANDA = DATOS_DIR / "Perfil_demanda.csv"

# Fuente alternativa de E0/E1: el paquete que exporta la pestaña "🗺️ Datos
# de entrada" (entrada_red.guardar_paquete). Existe para poder cambiar de
# red sin fabricar un app.pkl nuevo de 740 MB: de ese pickle E0 solo rescata
# seis claves, y E1 tres perfiles que igual viven en archivos sueltos
# (Perfil_demanda.csv y los dos Excel de perfiles).
FUENTES = ("app_pkl", "red_nueva")


def usar_artefactos(ruta):
    """Apunta TODO este módulo a la carpeta de artefactos de un caso.

    Es el único lugar que reasigna esas rutas, y hay dos motivos para que
    sea uno solo:

      * media docena de ellas se derivan entre sí (`BASE_DIR` cuelga de
        `ART_DIR`), así que reasignar una a mano deja el resto apuntando al
        caso anterior, que es exactamente el bug que tenía el notebook de
        prueba;
      * la carpeta queda publicada en ERAD_ARTEFACTOS, de modo que
        cualquier subproceso que se lance después (los jobs de la app,
        `construir_faultlin.py`, `run_barrido_medidas.py`) herede el mismo
        caso sin que haya que acordarse de pasárselo.

    ⚠️ NO llamar desde código de Streamlit. El proceso de la app tiene un
    solo `pipeline` importado, compartido con `data_loader`, `barrido_vista`
    y `estudio_v8`: cambiarlo en caliente movería la carpeta bajo los pies
    de esas vistas. Para MIRAR otro caso está `casos.estado_caso(ruta)`, y
    para CALCULAR sobre él, un subproceso con `--artefactos`.
    """
    global ART_DIR, BASE_DIR, RUTA_RED_NUEVA, RUTA_FAULTLIN, RUTA_FAULTLIN_MIT, RUTA_FAULTLIN_RAM
    ART_DIR = casos.resolver_ruta(ruta)
    BASE_DIR = ART_DIR / "baselines"
    RUTA_RED_NUEVA = ART_DIR / "red_nueva.pkl"
    # El caso de validación conserva su matriz en app/, fuera de artefactos/.
    # Ver `casos.ruta_faultlin`: ese archivo no se mueve de sitio.
    RUTA_FAULTLIN = casos.ruta_faultlin(ART_DIR)
    RUTA_FAULTLIN_MIT = {
        v: RUTA_FAULTLIN.with_name(f"{RUTA_FAULTLIN.stem}_{v}.npy")
        for v in ("veg", "ref", "ug")}
    # Ramales con fusible (ver ramales.py): solo existe en casos que los declaran.
    RUTA_FAULTLIN_RAM = RUTA_FAULTLIN.with_name(f"{RUTA_FAULTLIN.stem}_ramales.npy")
    os.environ[casos.VAR_ENTORNO] = str(ART_DIR)
    return ART_DIR


usar_artefactos(casos.art_dir_activo())

# Horas que tarda en reponerse una línea fallada. Es un PARÁMETRO DE ENTRADA
# del estudio, no una constante del código: define cuánto dura la
# interrupción y por lo tanto entra directo en SAIDI, ENS y en el valor de
# toda medida que acorte la indisponibilidad.
#
# Se usa en dos lugares que tienen que decir lo mismo:
#   - `construir_faultlin.py --horas-espera`: bloquea a la línea mientras se
#     repara, para que no vuelva a fallar dentro de la misma tormenta.
#   - E2 acá: construye `repairMC`, la matriz con la que el MILP sabe cuándo
#     la línea vuelve a estar disponible.
# Por eso E2 toma por defecto el valor que dejó registrado la corrida del
# Monte Carlo (`artefactos/faultlin_meta.json`) en vez de imponer el suyo.
# El 4 solo aplica si no hay registro: es el valor con el que se generó el
# `repairMC` del app.pkl histórico (verificado).
HORAS_REPARACION_DEFAULT = 4
HORAS_REPARACION = HORAS_REPARACION_DEFAULT   # compatibilidad con importadores
S = [0, 1]               # subestaciones (buses 1 y 12)
TIPO_CAMBIO = 922.75     # CLP/USD, único para E3 y los costos por caso


def horas_reparacion_registradas():
    """Horas de espera con las que se simuló la matriz de fallas vigente,
    o None si esa corrida no dejó registro (el .npy histórico)."""
    ruta = ART_DIR / "faultlin_meta.json"
    if not ruta.exists():
        return None
    try:
        return int(json.loads(ruta.read_text(encoding="utf-8"))["horas_espera"])
    except Exception:
        return None


# ════════════════════════════════════════════════════════════
# Utilidades
# ════════════════════════════════════════════════════════════

def _log(msg):
    print(f"[pipeline] {msg}", flush=True)


def _guardar(nombre, obj):
    ART_DIR.mkdir(parents=True, exist_ok=True)
    ruta = ART_DIR / nombre
    with open(ruta, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    _log(f"→ {nombre} ({ruta.stat().st_size/1e6:.1f} MB)")
    return ruta


def _cargar(nombre, art_dir=None):
    with open(Path(art_dir or ART_DIR) / nombre, "rb") as f:
        return pickle.load(f)


def _existe(nombre):
    return (ART_DIR / nombre).exists()


def _leer_pkl_congelado():
    """Lectura ÚNICA del app.pkl. Solo se rescatan las claves que no se
    pueden regenerar: la red, la topología, y los resultados del estudio
    congelado. Todo lo demás se reconstruye en las etapas siguientes."""
    _log(f"leyendo {RUTA_APP_PKL.name} (solo lectura, ~15 s)...")
    with open(RUTA_APP_PKL, "rb") as f:
        return pickle.load(f)


def _leer_red_nueva():
    """Paquete de red exportado desde la pestaña '🗺️ Datos de entrada'."""
    if not RUTA_RED_NUEVA.exists():
        raise FileNotFoundError(
            f"No existe '{RUTA_RED_NUEVA.name}'. Se genera en la pestaña "
            f"'🗺️ Datos de entrada' de la app (paso 5), o con "
            f"entrada_red.guardar_paquete() desde consola.")
    _log(f"leyendo {RUTA_RED_NUEVA.name}...")
    with open(RUTA_RED_NUEVA, "rb") as f:
        return pickle.load(f)


def _fuente_ok(fuente):
    if fuente not in FUENTES:
        raise ValueError(f"Fuente desconocida '{fuente}'. Válidas: {list(FUENTES)}")
    return fuente


# ════════════════════════════════════════════════════════════
# E0 — Red y topología
# ════════════════════════════════════════════════════════════

def etapa_E0(forzar=False, _d=None, fuente="app_pkl"):
    if _existe("red.pkl") and not forzar:
        _log("E0 red: ya existe, se omite (--forzar para rehacer)")
        return
    _fuente_ok(fuente)
    if fuente == "red_nueva":
        d = _leer_red_nueva()
    else:
        d = _d if _d is not None else _leer_pkl_congelado()
    net = d["net"]

    BUSES = list(net.bus.index)
    LINES = list(net.line.index)
    # Con app.pkl las tie-lines son, por convención del estudio, las tres
    # últimas líneas. El paquete de red las trae elegidas explícitamente en
    # la pestaña, porque en una red arbitraria "las tres últimas" no
    # significa nada.
    # Con el paquete, una lista VACÍA significa "esta red no tiene enlaces de
    # respaldo" y se respeta: con `or` dejaba abiertas tres líneas
    # cualesquiera. El camino del app.pkl queda EXACTAMENTE como estaba, para
    # que rehacer E0 del caso de validación no pueda cambiar nada.
    explicito = fuente == "red_nueva"
    if explicito and d.get("tie_indices") is not None:
        tie_indices = set(d["tie_indices"])
    else:
        tie_indices = set(d.get("tie_indices") or LINES[-3:])
    # Mismo criterio que data_loader:463 — candidatas = todas menos las tie
    sw_cand_list = [l for l in LINES if l not in tie_indices]

    red = {
        "net": net,
        "BUSES": BUSES,
        "LINES": LINES,
        "GENS": BUSES,
        "ESS": BUSES,
        "S": S,
        "bus_zone": d["bus_zone"],
        "P_D_base": d["P_D_base"],
        "tie_indices": sorted(tie_indices),
        "sw_cand_list": sw_cand_list,
        "longitud_km_linea": {l: float(net.line.length_km.iloc[i])
                              for i, l in enumerate(LINES)},
        "N_eventos": d["N_eventos"],
        "numesc": int(d["numesc"]),
        # [6, 11, 14] son las barras críticas de la red de 15 barras. En el
        # paquete, una lista VACÍA es una respuesta ("ninguna barra crítica"),
        # no un hueco: con `or` se convertía en tres barras arbitrarias de otra
        # red, que además EDRP excluye de la respuesta de demanda. El camino
        # del app.pkl no cambia.
        "buses_criticos": list(d["buses_criticos"]
                               if explicito and d.get("buses_criticos") is not None
                               else (d.get("buses_criticos") or [6, 11, 14])),
        # Líneas con seccionador: dato de la red que habilita la variante de
        # aislación realista (`config_medida` la lee de acá). El app.pkl
        # histórico no la trae, así que queda vacía y el comportamiento del
        # estudio no cambia: sin seccionadores declarados, la aislación fija
        # deja fuera al alimentador completo, que es el supuesto conservador.
        "seccionadores_existentes": list(d.get("seccionadores_existentes") or []),
    }
    # Opcionales de una red reducida (caso El Melón): ramales con fusible como
    # elementos de falla propios (ramales.py) y el largo EXPUESTO de cada línea
    # (sin los km de ramales ni de cable), que es el que usa la curva de Dunn.
    # Solo se escriben si el paquete los trae: el red.pkl de los casos que no
    # los tienen queda igual que antes.
    if d.get(RAM.CLAVE_RED):
        RAM.validar(d[RAM.CLAVE_RED], BUSES)
        red[RAM.CLAVE_RED] = d[RAM.CLAVE_RED]
        _log(f"E0 ramales con fusible: {len(d[RAM.CLAVE_RED]['id'])}")
    if d.get("largo_expuesto_km"):
        red["largo_expuesto_km"] = {int(l): float(v) for l, v in d["largo_expuesto_km"].items()}
    if d.get("clientes_bus"):
        red["clientes_bus"] = {int(b): float(c) for b, c in d["clientes_bus"].items()}
        _log(f"E0 clientes por barra: {sum(red['clientes_bus'].values()):.0f}")
    if d.get("clientes_denominador") is not None:
        red["clientes_denominador"] = float(d["clientes_denominador"])
        _log(f"E0 denominador SAIFI: {red['clientes_denominador']:.0f} clientes")
    else:
        cli_total = clientes_denominador_saifi({}, art_dir)
        if cli_total is not None:
            red["clientes_denominador"] = cli_total
            _log(f"E0 denominador SAIFI desde saifi_config.json: {cli_total:.0f} clientes")
    _guardar("red.pkl", red)


# ════════════════════════════════════════════════════════════
# E1 — Perfiles (compactos)
# ════════════════════════════════════════════════════════════

def etapa_E1(forzar=False, _d=None, fuente="app_pkl"):
    """demand_factor y alphaDG* están guardados en el pickle como
    np.tile de (8760, 100, 15) float64 = 105 MB cada uno, pero el
    contenido real es un perfil de 8760 (demanda, solar) o de 8760×3
    zonas (eólico). Se guardan compactos y se expanden al vuelo."""
    if _existe("perfiles.npz") and not forzar:
        _log("E1 perfiles: ya existe, se omite")
        return
    _fuente_ok(fuente)
    if fuente == "red_nueva":
        return _etapa_E1_desde_archivos()
    d = _d if _d is not None else _leer_pkl_congelado()

    df = d["demand_factor"]
    demanda = df[:, 0, 0].copy()
    if not np.allclose(df, demanda[:, None, None]):
        raise RuntimeError(
            "demand_factor NO es uniforme en escenario/bus: no se puede "
            "compactar sin perder información. Revisar antes de seguir.")

    sol = d["alphaDGsolar"]
    solar = sol[:, 0, 0].copy()
    if not np.allclose(sol, solar[:, None, None]):
        raise RuntimeError("alphaDGsolar no es uniforme entre buses.")

    # Eólico sí varía por zona: se guarda una columna por bus (el mapeo
    # bus→zona ya está aplicado) para no re-derivarlo y arriesgar un
    # desfase respecto del artefacto congelado.
    eolico_por_bus = d["alphaDGeolica"][:, 0, :].copy()

    # Perfiles horarios por zona (solar del Explorador Solar, eólico del
    # viento sostenido del .nc) en vez de los de 24 h del app.pkl, igual que
    # en la rama red_nueva. Hasta el 2026-09-14 esta rama no llamaba a
    # _preferir_perfiles_horarios, así que todas las tandas de barrido
    # anteriores corrieron con el perfil diario de 24 h del Excel.
    extra = {}
    fuente_perf = "perfil diario de 24 h (app.pkl)"
    if _existe("red.pkl"):
        red = _cargar("red.pkl")
        BUSES, bus_zone = red["BUSES"], red["bus_zone"]
        solar_por_bus = np.tile(solar[:, None], (1, len(BUSES)))
        solar_por_bus, eolico_por_bus, fuente_perf = _preferir_perfiles_horarios(
            solar_por_bus, eolico_por_bus, BUSES, bus_zone, len(demanda))
        extra["solar_por_bus"] = solar_por_bus.astype(np.float64)
        solar = solar_por_bus[:, 0].copy()
    else:
        _log("  [!] E1 sin red.pkl: no se pueden asignar perfiles horarios por "
             "zona, se usan los de 24 h del app.pkl (correr E0 antes)")

    ART_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ART_DIR / "perfiles.npz",
        demanda=demanda.astype(np.float64),
        # `solar` 1-D se mantiene por compatibilidad con lectores antiguos;
        # `expandir_perfiles` prefiere `solar_por_bus` cuando está.
        solar=solar.astype(np.float64),
        eolico_por_bus=eolico_por_bus.astype(np.float64),
        **extra,
    )
    _log(f"→ perfiles.npz ({(ART_DIR/'perfiles.npz').stat().st_size/1e6:.1f} MB, "
         f"{fuente_perf})")


def _perfiles_del_caso():
    """Archivos de perfiles a usar: los del paquete del caso, o los compartidos.

    El paquete que exporta la pestaña puede declarar sus propios archivos
    (clave "perfiles"). Los de `app/` quedan como respaldo, que es lo correcto
    para el caso de validación —son los suyos— pero para una red nueva son un
    préstamo, y la pestaña lo advierte.
    """
    rutas = {"demanda": RUTA_PERFIL_DEMANDA, "solar": RUTA_PERFIL_SOLAR,
             "eolico": RUTA_PERFIL_EOLICO}
    if not RUTA_RED_NUEVA.exists():
        return rutas
    try:
        with open(RUTA_RED_NUEVA, "rb") as f:
            propios = (pickle.load(f).get("perfiles") or {})
    except Exception:
        return rutas
    for clave, ruta in propios.items():
        # "demanda_por_bus" (opcional): .npz con un factor horario por barra
        # (`buses`, `factor` de 8760 × barras); reemplaza a "demanda".
        if (clave in rutas or clave == "demanda_por_bus") and ruta:
            p = Path(ruta)
            if not p.is_absolute():
                p = DATOS_DIR / p
            rutas[clave] = p
            _log(f"E1 perfil '{clave}' del caso: {p.name}")
    return rutas


def _perfil_diario_anual(perfil_24h, n_horas):
    """Repite un perfil diario hasta cubrir el horizonte (celda 38 del
    notebook: `np.tile(perfil, (Horastot + 1) // 24)`)."""
    veces = int(np.ceil(n_horas / len(perfil_24h)))
    return np.tile(np.asarray(perfil_24h, dtype=float), veces)[:n_horas]


def _dispersion_perfiles(df_zonas):
    """Spread relativo de la energía diaria entre columnas Z* de un Excel
    de perfiles: (max - min) / max."""
    cols = [c for c in df_zonas.columns if str(c).startswith("Z")]
    energias = np.array([float(df_zonas[c].sum()) for c in cols])
    if energias.size < 2 or energias.max() <= 0:
        return 0.0, dict(zip(cols, energias))
    return float((energias.max() - energias.min()) / energias.max()), dict(zip(cols, energias))


def _distancia_max_km(puntos):
    """Mayor distancia entre pares de puntos (lat, lon), aproximada."""
    ps = list(puntos.values())
    if len(ps) < 2:
        return 0.0
    peor = 0.0
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            dlat = (ps[i][0] - ps[j][0]) * 111.0
            dlon = (ps[i][1] - ps[j][1]) * 111.0 * np.cos(np.radians(ps[i][0]))
            peor = max(peor, float(np.hypot(dlat, dlon)))
    return peor


def _avisar_perfiles_vs_puntos(solar_zonas, eolico_zonas, umbral_km=150.0,
                               umbral_spread=0.10):
    """Avisa si los perfiles por zona no pueden corresponder a los puntos.

    Existe por un problema concreto y detectado en los datos del estudio:
    `Perfiles_solares_promedio_por_zona.xlsx` tiene Z1/Z2/Z3 con 39% de
    diferencia en energía diaria — un spread que solo se explica con puntos
    a cientos de km (la columna suelta de SystemzoneV1.xlsx los identifica
    como Iquique / Valparaíso / Concepción). Los puntos que hoy alimentan al
    pipeline climático, en cambio, están a ~45 km entre sí. Un perfil solar
    distinto por zona a esa escala no tiene respaldo físico: son perfiles de
    otra definición de zonas que quedó pegada al archivo.

    Esto no se puede arreglar solo desde el código: hay que decidir si los
    perfiles se regeneran para los puntos actuales o si los puntos vuelven a
    ser los de la definición original. Lo que sí se puede es no dejar que
    pase inadvertido.
    """
    if not RUTA_RED_NUEVA.exists():
        return
    try:
        with open(RUTA_RED_NUEVA, "rb") as f:
            puntos = pickle.load(f).get("puntos_zona") or {}
    except Exception:
        return

    d_km = _distancia_max_km(puntos)
    if d_km == 0 or d_km > umbral_km:
        return

    for nombre, df in (("solar", solar_zonas), ("eólico", eolico_zonas)):
        spread, energias = _dispersion_perfiles(df)
        if spread > umbral_spread:
            detalle = ", ".join(f"{k}={float(v):.2f}" for k, v in energias.items())
            _log(f"  [!] perfil {nombre}: las zonas están a {d_km:.0f} km entre sí "
                 f"pero los perfiles difieren {spread:.0%} en energía diaria "
                 f"({detalle}).")
            _log(f"      Esos perfiles vienen de OTRA definición de zonas. "
                 f"Revisar antes de usar los resultados (ver INSTRUCTIVO_DATOS.md §7).")


def _preferir_perfiles_horarios(solar_por_bus, eolico_por_bus, BUSES, bus_zone, n_horas):
    """Usa los perfiles horarios en vez de los tilados, si existen.

    `construir_perfiles.py --resolucion horaria` deja
    `artefactos/perfiles_horarios.npz` con una serie de 8760 h por zona, en
    vez de un perfil de 24 h repetido los 365 días.

    Qué gana el modelo con eso. El perfil entra como factor de planta
    normalizado que acota la generación:
        P_DG_sol[t,g] <= alphaDGsolar[t,esc,g] * CAP_DG_sol[g]
    así que su valor en cada hora ES la energía disponible de esa hora.
    Con un perfil de 24 h repetido, enero y julio quedan idénticos: en Aysén
    (−47°) el día de verano dura ~17 h y el de invierno ~7,5 h, de modo que
    el modelo ve sol donde no lo hay y no lo ve donde sí. Como las fallas
    caen en horas concretas del año, la hora del corte deja de ser
    informativa.

    Para el eólico la ganancia es menor de lo que parece: el modelo ya
    multiplica por `a_eol_wind[t,g]`, un binario que apaga toda la generación
    eólica de una zona en cualquier hora con una línea dañada ahí. En esas
    horas el perfil da igual. Sigue importando para las zonas no afectadas y
    para las horas alrededor del evento, que son las que definen con qué
    llega cargado el almacenamiento.

    Devuelve (solar, eolico, etiqueta_de_fuente).
    """
    ruta = ART_DIR / "perfiles_horarios.npz"
    if not ruta.exists():
        return solar_por_bus, eolico_por_bus, "perfil diario de 24 h (Excel)"

    z = np.load(ruta)
    zonas = [int(x) for x in z["zonas"]] if "zonas" in z.files else None

    def _por_bus(matriz_zona):
        columnas = ({zona: i for i, zona in enumerate(zonas)} if zonas
                    else {i + 1: i for i in range(matriz_zona.shape[1])})
        salida = np.zeros((n_horas, len(BUSES)))
        for i, b in enumerate(BUSES):
            zona = int(bus_zone.get(int(b), 1))
            col = columnas.get(zona)
            if col is None:                      # zona sin serie: se deja en 0
                _log(f"  [!] la zona {zona} no está en {ruta.name}: perfil en cero")
                continue
            serie = matriz_zona[:, col]
            if len(serie) < n_horas:             # se recicla si el horizonte es mayor
                serie = np.resize(serie, n_horas)
            salida[:, i] = serie[:n_horas]
        return salida

    solar = _por_bus(z["solar_por_zona"])
    eolico = _por_bus(z["eolico_por_zona"])
    _log(f"E1 perfiles horarios: {ruta.name} "
         f"(CF medio solar {solar.mean():.3f}, eólico {eolico.mean():.3f})")
    return solar, eolico, f"horario ({ruta.name})"


def _eolico_por_anio(BUSES, bus_zone, n_horas):
    """Eólico por año de viento, si `perfiles_horarios.npz` lo trae.

    Con varios años de clima (construir_faultlin_anios.py) el escenario e usa
    el año e // sorteos_por_anio, y la turbina tiene que ver el viento de ese
    mismo año, como las líneas. El archivo trae `eolico_por_zona_anio`
    (años × horas × zonas) y `sorteos_por_anio`. Devuelve
    (años × horas × barras, sorteos_por_anio, etiquetas de año) o None.
    """
    ruta = ART_DIR / "perfiles_horarios.npz"
    if not ruta.exists():
        return None
    z = np.load(ruta)
    if "eolico_por_zona_anio" not in z.files:
        return None
    por_zona = z["eolico_por_zona_anio"]
    zonas = [int(x) for x in z["zonas"]]
    col = [zonas.index(int(bus_zone.get(int(b), 1))) for b in BUSES]
    arr = np.ascontiguousarray(por_zona[:, :n_horas, :][:, :, col], dtype=np.float32)
    anios = [str(a) for a in z["anios"]] if "anios" in z.files else list(range(len(arr)))
    _log(f"E1 eólico por año: {len(arr)} años × {n_horas} h × {len(BUSES)} barras "
         f"(CF medio {arr.mean():.4f}; {int(z['sorteos_por_anio'])} escenarios por año)")
    return arr, int(z["sorteos_por_anio"]), anios


class PerfilPorAnio:
    """Perfil (horas, escenarios, barras) donde el escenario e lee el año e // sorteos.

    Se indexa como el arreglo que reemplaza (`alphaDGeolica[t, esc, i]`, y
    también con rebanadas), pero guarda un perfil por año en vez de uno por
    escenario: con 10 años y 100 escenarios son 10 veces menos memoria que
    materializarlo (unos 700 MB en El Melón). `np.asarray` lo materializa.
    """

    def __init__(self, por_anio, sorteos_por_anio, num_esc):
        self._b = np.moveaxis(np.asarray(por_anio), 0, 1)            # (horas, años, barras), vista
        self.anio_de_escenario = np.arange(num_esc) // int(sorteos_por_anio)
        if self.anio_de_escenario.max() >= self._b.shape[1]:
            raise ValueError(f"{num_esc} escenarios con {sorteos_por_anio} por año piden "
                             f"{self.anio_de_escenario.max() + 1} años; hay {self._b.shape[1]}")
        self.shape = (self._b.shape[0], int(num_esc), self._b.shape[2])
        self.ndim, self.dtype = 3, self._b.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, clave):
        if not isinstance(clave, tuple):
            clave = (clave,)
        clave = clave + (slice(None),) * (3 - len(clave))
        t, e, b = clave
        return self._b[t, self.anio_de_escenario[e], b]

    def __array__(self, dtype=None, copy=None):
        a = self._b[:, self.anio_de_escenario, :]
        return a.astype(dtype) if dtype is not None else a


def _etapa_E1_desde_archivos():
    """E1 para una red nueva: los perfiles se reconstruyen desde los
    archivos sueltos en vez de rescatarse del app.pkl.

    Es exactamente lo que hacía el notebook antes de congelar el pickle
    (celda 38): demanda desde `Perfil_demanda.csv`, solar y eólico desde
    los dos Excel de perfiles promedio por zona, repetidos día a día y
    asignados a cada bus según su zona.

    Diferencia deliberada con el camino app.pkl: acá el perfil SOLAR
    también se guarda por bus. En el artefacto congelado quedó un único
    perfil solar para las tres zonas (la columna Z3), tal como salió de
    data_loader; se respeta ese valor histórico donde ya está, pero no se
    replica el error en una red nueva.
    """
    if not _existe("red.pkl"):
        raise RuntimeError("E1 desde archivos requiere E0 (red.pkl).")
    red = _cargar("red.pkl")
    BUSES, bus_zone = red["BUSES"], red["bus_zone"]

    # Perfiles DEL CASO si el paquete los declaró; si no, los compartidos de
    # app/. Importa: sin esto una red nueva hereda la demanda del estudio de
    # validación sin que nada lo diga, y se terminaría concluyendo sobre una
    # red con la curva de carga de otra.
    rutas = _perfiles_del_caso()
    ruta_por_bus = rutas.pop("demanda_por_bus", None)
    if ruta_por_bus is not None:
        rutas.pop("demanda")                    # la reemplaza el factor por barra
    # Con perfiles horarios propios del caso, los Excel de 24 h no hacen falta
    # (de lo contrario una red nueva exigiría los de la validación).
    horarios = (ART_DIR / "perfiles_horarios.npz").exists()
    if horarios:
        rutas.pop("solar"); rutas.pop("eolico")
    faltan = [str(p) for p in [*rutas.values(), *([ruta_por_bus] if ruta_por_bus else [])]
              if not p.exists()]
    if faltan:
        raise FileNotFoundError(f"E1 no encuentra estos archivos de perfiles: {faltan}")

    demanda_por_bus = None
    if ruta_por_bus is not None:
        demanda_por_bus, demanda = _demanda_por_bus(ruta_por_bus, red)
        _log(f"E1 demanda: factor por barra desde {ruta_por_bus.name} "
             f"({demanda_por_bus.shape[0]} h × {demanda_por_bus.shape[1]} barras, "
             f"máx {demanda_por_bus.max():.3f})")
    else:
        demanda = pd.read_csv(rutas["demanda"], header=0)["Power"].to_numpy(dtype=float)
        _log(f"E1 demanda: {len(demanda)} h desde {rutas['demanda'].name} "
             f"[{demanda.min():.2f}, {demanda.max():.2f}]")
    n_horas = len(demanda)

    solar_por_bus = np.zeros((n_horas, len(BUSES)))
    eolico_por_bus = np.zeros((n_horas, len(BUSES)))
    if not horarios:
        # Los perfiles solares del Excel vienen en W; el notebook los divide por
        # 1000 para dejarlos como fracción de la capacidad instalada.
        solar_zonas = pd.read_excel(rutas["solar"]) / 1000.0
        eolico_zonas = pd.read_excel(rutas["eolico"])

        def _columna(df, zona):
            col = f"Z{zona}"
            if col not in df.columns:
                disponibles = [c for c in df.columns if str(c).startswith("Z")]
                if not disponibles:
                    raise ValueError(f"El Excel de perfiles no tiene columnas Z*: {list(df.columns)}")
                col = disponibles[min(zona, len(disponibles)) - 1]
                _log(f"  [!] no hay columna Z{zona}; se usa '{col}'")
            return df[col].to_numpy(dtype=float)

        for i, b in enumerate(BUSES):
            z = int(bus_zone.get(int(b), 1))
            solar_por_bus[:, i] = _perfil_diario_anual(_columna(solar_zonas, z), n_horas)
            eolico_por_bus[:, i] = _perfil_diario_anual(_columna(eolico_zonas, z), n_horas)
        _avisar_perfiles_vs_puntos(solar_zonas, eolico_zonas)

    solar_por_bus, eolico_por_bus, fuente_perf = _preferir_perfiles_horarios(
        solar_por_bus, eolico_por_bus, BUSES, bus_zone, n_horas)
    por_anio = _eolico_por_anio(BUSES, bus_zone, n_horas)
    extra = {}
    if por_anio is not None:
        arr, sorteos, anios = por_anio
        extra = {"eolico_por_bus_anio": arr, "sorteos_por_anio": sorteos, "anios_eolico": np.array(anios)}

    ART_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ART_DIR / "perfiles.npz",
        demanda=demanda,
        # `solar` 1-D se mantiene por compatibilidad con lectores antiguos;
        # `expandir_perfiles` prefiere `solar_por_bus` cuando está.
        solar=solar_por_bus[:, 0].copy(),
        solar_por_bus=solar_por_bus,
        eolico_por_bus=eolico_por_bus,
        # idem: `demanda` queda como el agregado, y `demanda_por_bus` manda
        **({"demanda_por_bus": demanda_por_bus} if demanda_por_bus is not None else {}),
        # `eolico_por_bus` queda como el promedio de los años; manda el por año
        **extra,
    )
    _log(f"→ perfiles.npz ({(ART_DIR/'perfiles.npz').stat().st_size/1e6:.1f} MB, "
         f"solar y eólico por bus{', demanda por bus' if demanda_por_bus is not None else ''}"
         f"{', eólico por año' if extra else ''} — {fuente_perf})")


def _demanda_por_bus(ruta, red):
    """Factor horario de demanda por barra, en el orden de red["BUSES"].

    El archivo (lo arma el notebook del caso) trae `buses` y `factor`
    (horas × barras): demanda de la barra en la hora t = P_D_base × factor.
    Una barra del paquete que no esté en el archivo es un error, para que
    no quede con factor 0 sin que nadie lo note. Devuelve además el perfil
    agregado (ponderado por P_D_base), que se guarda como `demanda` para los
    lectores que esperan uno solo.
    """
    z = np.load(ruta)
    pos = {int(b): i for i, b in enumerate(z["buses"])}
    BUSES, P_D = red["BUSES"], red["P_D_base"]
    faltan = [b for b in BUSES if int(b) not in pos]
    if faltan:
        raise ValueError(f"{Path(ruta).name}: faltan las barras {faltan[:10]} del paquete")
    f = np.asarray(z["factor"], dtype=np.float64)[:, [pos[int(b)] for b in BUSES]]
    w = np.array([max(float(P_D.get(b, 0.0)), 0.0) for b in BUSES])
    agregado = f @ w / w.sum() if w.sum() > 0 else f.mean(axis=1)
    return f, agregado


# Factor de planta mínimo de DG solar/eólica. Valores en (0, ALPHA_DG_MIN) se
# llevan a 0: son el amanecer/atardecer solar (hasta 7,7e-07 en El Melón) y la
# cola baja de la curva eólica, y como coeficiente de `P_DG ≤ α·CAP` estiraban
# el rango de la matriz a [8e-07, 8e+07], que es lo que hacía fallar a barrier
# ("Numerical trouble"). Con 1e-4 la energía que se pierde es a lo más 0,03 %
# (perfil eólico promedio de El Melón, el de E4); con 1e-3 sería 1,8 % en ese
# mismo perfil, que ya movería el caso sin fallas. 0 desactiva el recorte.
ALPHA_DG_MIN = 1e-4


def _recortar_alpha(a):
    """Copia de `a` con los factores de planta residuales en 0 (ver ALPHA_DG_MIN)."""
    a = np.asarray(a)
    if ALPHA_DG_MIN <= 0:
        return a
    return np.where((a > 0) & (a < ALPHA_DG_MIN), 0.0, a)


def expandir_perfiles(num_esc, n_buses, copiar=True, eolico_promedio=False):
    """Devuelve demand_factor y alphaDG* en la forma (horas, esc, bus)
    que esperan optimization_base / optimization_3.

    copiar=False devuelve VISTAS de solo lectura (np.broadcast_to): la
    forma es la misma pero no se materializan los 105 MB por arreglo,
    porque el contenido real es un perfil de 8760 valores repetido. Los
    optimizadores solo leen, así que la vista basta; un intento de
    escritura levantaría error en vez de corromper el perfil en silencio.

    Si E1 guardó el eólico por año de viento (`eolico_por_bus_anio`), el
    eólico sale como `PerfilPorAnio`: cada escenario lee el año de su clima
    (se indexa igual, sin materializar). `eolico_promedio=True` usa en cambio
    el promedio de los años para todos, que es lo que corresponde al caso
    sin fallas de E4 (no tiene año).
    """
    z = np.load(ART_DIR / "perfiles.npz")
    n_h = len(z["demanda"])
    if "demanda_por_bus" in z.files:
        dem = np.broadcast_to(z["demanda_por_bus"][:, None, :], (n_h, num_esc, n_buses))
    else:
        dem = np.broadcast_to(z["demanda"][:, None, None], (n_h, num_esc, n_buses))
    # El recorte de ALPHA_DG_MIN va sobre el perfil base, antes de expandir:
    # así cuesta 8760 × barras y llega igual a todos los optimizadores.
    if "solar_por_bus" in z.files:
        sol = np.broadcast_to(_recortar_alpha(z["solar_por_bus"])[:, None, :], (n_h, num_esc, n_buses))
    else:
        sol = np.broadcast_to(_recortar_alpha(z["solar"])[:, None, None], (n_h, num_esc, n_buses))
    if "eolico_por_bus_anio" in z.files and not eolico_promedio:
        eol = PerfilPorAnio(_recortar_alpha(z["eolico_por_bus_anio"]), int(z["sorteos_por_anio"]), num_esc)
        if copiar:
            return np.ascontiguousarray(dem), np.ascontiguousarray(sol), eol
        return dem, sol, eol
    eol = np.broadcast_to(_recortar_alpha(z["eolico_por_bus"])[:, None, :], (n_h, num_esc, n_buses))
    if copiar:
        return np.ascontiguousarray(dem), np.ascontiguousarray(sol), np.ascontiguousarray(eol)
    return dem, sol, eol


# ════════════════════════════════════════════════════════════
# E2 — Fallas: matriz base, reparaciones y matrices mitigadas
# ════════════════════════════════════════════════════════════

RHO_UG = 0.95
_DARESTANI_MU_C5 = [6.597, -1.285e-2, -8.189e-2, 6.850e-3, -6.540e-2, 4.979e-5,
                    -1.368e-3, 5.333e-3, -7.649e-7, -3.429e-6, -1.405e-4,
                    2.438e-4, 1.743e-3, -1.836e-5, 5.301e-4]
_DARESTANI_MU_C3 = [6.705, -1.128e-2, -7.239e-2, 6.587e-3, -6.723e-2, 4.004e-5,
                    -1.302e-3, 4.629e-3, -1.403e-6, -1.669e-5, -1.367e-4,
                    2.341e-4, 1.692e-3, -1.123e-5, 5.356e-4]
_CANHAM_A_S, _CANHAM_B_S, _CANHAM_C_S = -2.261, 0.426, 1.140
DBH_ANTES_PODA_CM = 30.0
DBH_DESPUES_PODA_CM = 10.0
BETA_REF = 0.25


def _mu_log_mediana_darestani(a, theta_deg=90.0, Ac=2.0, t=50.0, H=15.24):
    return (a[0] + a[1]*theta_deg + a[2]*Ac + a[3]*t + a[4]*H
            + a[5]*theta_deg**2 + a[6]*Ac*theta_deg + a[7]*Ac**2
            + a[8]*t*theta_deg + a[9]*t*Ac + a[10]*t**2
            + a[11]*theta_deg*H + a[12]*Ac*H + a[13]*t*H + a[14]*H**2)


def _cargar_wind20kv(n_horas):
    """Serie de viento real (misma fuente que generó FaultlinMC.npy:
    BaseHoraEWEPELP20z.nc, año 2030), ajustada a 12 m de altura de poste.
    None si el .nc no está en este entorno."""
    try:
        import xarray as xr
        ds = xr.open_dataset(RUTA_WIND_NC)
        wind_ms = ds["wind"].values[8760:8760 + n_horas, :, 0]
        return wind_ms * ((12 / 10) ** (1 / 7))
    except Exception as e:
        _log(f"  [!] no se pudo leer el .nc de viento ({e.__class__.__name__}); "
             f"se usará el fallback de % plano")
        return None


def _cargar_mitigadas_directas(forma):
    """Matrices por medida simuladas con su propia curva de fragilidad.

    Las escribe `construir_faultlin.py` junto a la base. Devuelve None si
    falta alguna o si no calzan con la forma de la matriz base: media
    mitigación cargada sería peor que ninguna, porque las tres medidas se
    compararían contra físicas distintas sin avisar.

    No basta con que los archivos existan: `FaultlinMC_veg.npy` ya está en
    el repo desde antes, generado por adelgazamiento en la celda 15 del
    notebook. Por eso se exige que `faultlin_meta.json` declare que la
    corrida que los escribió usó las curvas directas — el archivo suelto no
    dice de dónde salió, el meta sí.
    """
    meta_ruta = ART_DIR / "faultlin_meta.json"
    if not meta_ruta.exists():
        return None
    try:
        metodo = json.loads(meta_ruta.read_text(encoding="utf-8")) \
                     .get("mitigadas", {}).get("metodo")
    except Exception:
        return None
    if metodo != "curva_directa_numeros_comunes":
        return None

    rutas = dict(RUTA_FAULTLIN_MIT)
    faltan = [v for v, r in rutas.items() if not r.exists()]
    if faltan:
        _log(f"  [!] faultlin_meta.json declara curvas directas pero faltan "
             f"{faltan}: se adelgaza.")
        return None

    salida = {}
    for v, r in rutas.items():
        arr = np.load(r)
        if arr.shape != tuple(forma):
            _log(f"  [!] {r.name} tiene forma {arr.shape} y la base {tuple(forma)}: "
                 f"se ignoran las mitigadas directas y se adelgaza.")
            return None
        salida[v] = arr
    return salida


def _adelgazar(faultlinMC, rho, seed):
    """Adelgazamiento de Bernoulli: cada falla ya sorteada se 'cura' con
    probabilidad rho. rho puede ser escalar o (horas, líneas)."""
    rng = np.random.default_rng(seed)
    fallas = faultlinMC == 1
    if np.isscalar(rho):
        curadas = rng.random(faultlinMC.shape) < rho
    else:
        curadas = rng.random(faultlinMC.shape) < rho[:, np.newaxis, :]
    return np.where(fallas & curadas, 0, faultlinMC).astype(faultlinMC.dtype)


def etapa_E2(forzar=False, _d=None, horas_reparacion=None):
    if _existe("fallas.npz") and not forzar:
        _log("E2 fallas: ya existe, se omite")
        return
    if not _existe("red.pkl"):
        raise RuntimeError("E2 requiere E0 (red.pkl).")

    # Prioridad: lo que pidió la usuaria > lo que usó el Monte Carlo >
    # el valor histórico. El orden importa: si E2 impusiera su propio número,
    # una línea podría figurar reparada en `repairMC` mientras el sorteo la
    # seguía considerando fuera de servicio (o al revés), y el MILP operaría
    # sobre una disponibilidad que nunca existió.
    registrado = horas_reparacion_registradas()
    if horas_reparacion is None:
        horas_reparacion = registrado if registrado is not None else HORAS_REPARACION_DEFAULT
    horas_reparacion = int(horas_reparacion)
    if registrado is not None and registrado != horas_reparacion:
        _log(f"  [!] la matriz de fallas se simuló con {registrado} h de espera "
             f"y E2 va a construir repairMC con {horas_reparacion} h. "
             f"Son el mismo fenómeno físico: revisar si es intencional.")
    _log(f"E2 horas de reparación: {horas_reparacion}")
    red = _cargar("red.pkl")
    net, LINES = red["net"], red["LINES"]
    bus_zone = red["bus_zone"]

    faultlinMC = np.load(RUTA_FAULTLIN)
    n_horas = faultlinMC.shape[0]
    _log(f"E2 faultlinMC {faultlinMC.shape} — {int(faultlinMC.sum())} fallas")

    # repairMC: verificado que reproduce exactamente el del app.pkl
    repairMC = np.zeros_like(faultlinMC)
    for h in range(n_horas - horas_reparacion):
        repairMC[h + horas_reparacion] = np.maximum(
            repairMC[h + horas_reparacion], faultlinMC[h])

    wind = _cargar_wind20kv(n_horas)
    fuente_viento = "nc_real" if wind is not None else "fallback_plano"

    def _rho_por_zona_a_linea(rho_por_zona):
        out = np.zeros((n_horas, len(LINES)))
        for i, l in enumerate(LINES):
            z = bus_zone.get(int(net.line.from_bus[l]), 1) - 1
            z = min(max(z, 0), rho_por_zona.shape[1] - 1)
            out[:, i] = rho_por_zona[:, z]
        return out

    def _p_falla_lognormal(v, theta, beta):
        from scipy.stats import norm
        v = np.asarray(v, dtype=float)
        out = np.zeros_like(v)
        safe = v > 0
        out[safe] = norm.cdf(np.log(v[safe] / theta) / beta)
        return out

    # — REFUERZO ESTRUCTURAL — desplazamiento de la curva de fragilidad
    ratio = np.exp(_mu_log_mediana_darestani(_DARESTANI_MU_C3)
                   - _mu_log_mediana_darestani(_DARESTANI_MU_C5))
    theta_ref = 30.0 * ratio
    if wind is None:
        rho_ref = np.full((n_horas, len(LINES)), 0.42)   # Hu et al. 2021
    else:
        mu0 = _p_falla_lognormal(wind, 30.0, 0.25)
        muref = _p_falla_lognormal(wind, theta_ref, BETA_REF)
        with np.errstate(divide="ignore", invalid="ignore"):
            rz = np.where(mu0 > 1e-6, 1 - muref / np.maximum(mu0, 1e-12), 0.0)
        rho_ref = _rho_por_zona_a_linea(np.clip(rz, 0.0, 1.0))

    # — GESTIÓN DE VEGETACIÓN — modelo de caída de árboles de Canham
    if wind is None:
        rho_veg = np.full((n_horas, len(LINES)), 0.30)   # Cerrai et al. 2019
    else:
        def _p_ftr(v, dbh, v_max):
            Sv = np.clip(v / v_max, 0.0, 1.0)
            return 1 / (1 + np.exp(-(_CANHAM_A_S + _CANHAM_C_S * Sv * dbh ** _CANHAM_B_S)))
        v_max = wind.max()
        p_antes = _p_ftr(wind, DBH_ANTES_PODA_CM, v_max)
        p_desp = _p_ftr(wind, DBH_DESPUES_PODA_CM, v_max)
        with np.errstate(divide="ignore", invalid="ignore"):
            rz = np.where(p_antes > 1e-6, 1 - p_desp / np.maximum(p_antes, 1e-12), 0.0)
        rho_veg = _rho_por_zona_a_linea(np.clip(rz, 0.0, 1.0))

    # Si construir_faultlin.py ya simuló cada medida con SU PROPIA curva de
    # fragilidad (y con los mismos números aleatorios que la base), esas
    # matrices son mejores que adelgazar: la línea podada puede fallar en una
    # hora en que su gemela sin podar todavía se repara, cosa que el
    # adelgazamiento no puede representar porque solo sabe borrar fallas ya
    # sorteadas. El adelgazamiento queda como respaldo — es lo único posible
    # sobre el FaultlinMC.npy histórico, que se generó sin semilla.
    directas = _cargar_mitigadas_directas(faultlinMC.shape)
    if directas:
        veg, ref, ug = directas["veg"], directas["ref"], directas["ug"]
        metodo = "curva_directa"
        _log("E2 mitigadas: leídas de FaultlinMC_{veg,ref,ug}.npy (curva propia)")
    else:
        # Semillas idénticas a data_loader para reproducir sus matrices
        veg = _adelgazar(faultlinMC, rho_veg, seed=20260803)
        ref = _adelgazar(faultlinMC, rho_ref, seed=20260810)
        ug = _adelgazar(faultlinMC, RHO_UG, seed=20260811)
        metodo = "adelgazamiento"

    # Ramales con fusible: mismo horizonte, escenarios y horas de reparación que
    # las líneas. Si la red los declara y falta su matriz, se corta acá: correr
    # E2 sin ellos volvería en silencio al supuesto de que el fusible no aísla.
    extra_ram = {}
    if red.get(RAM.CLAVE_RED):
        if not RUTA_FAULTLIN_RAM.exists():
            raise RuntimeError(
                f"red.pkl declara {len(red[RAM.CLAVE_RED]['id'])} ramales con fusible pero no existe "
                f"{RUTA_FAULTLIN_RAM.name}. Correr construir_faultlin.py --artefactos ... --forzar.")
        f_ram = np.load(RUTA_FAULTLIN_RAM)
        esperado = (n_horas, faultlinMC.shape[1], len(red[RAM.CLAVE_RED]["id"]))
        if f_ram.shape != esperado:
            raise RuntimeError(f"{RUTA_FAULTLIN_RAM.name} tiene forma {f_ram.shape} y se esperaba "
                               f"{esperado} (horas, escenarios, ramales): rehacer construir_faultlin.py.")
        extra_ram = {"faultlinMC_ramales": f_ram,
                     "repairMC_ramales": RAM.matriz_reparacion(f_ram, horas_reparacion)}
        _log(f"E2 ramales con fusible: {f_ram.shape} — {int(f_ram.sum())} fallas")

    ART_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ART_DIR / "fallas.npz",
        faultlinMC=faultlinMC, repairMC=repairMC,
        faultlinMC_veg=veg, faultlinMC_ref=ref, faultlinMC_ug=ug,
        **extra_ram,
    )
    meta = {
        "fuente_viento": fuente_viento,
        "metodo_mitigadas": metodo,
        "theta_ref": float(theta_ref),
        "ratio_clase3_clase5": float(ratio),
        "rho_ug": RHO_UG,
        "horas_reparacion": horas_reparacion,
        "semillas": ({"veg": 20260803, "ref": 20260810, "ug": 20260811}
                     if metodo == "adelgazamiento" else "ver faultlin_meta.json"),
        "fallas": {
            "base": int(faultlinMC.sum()), "veg": int(veg.sum()),
            "ref": int(ref.sum()), "ug": int(ug.sum()),
            **({"ramales": int(extra_ram["faultlinMC_ramales"].sum())} if extra_ram else {}),
        },
    }
    (ART_DIR / "fallas_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"→ fallas.npz | viento={fuente_viento} | fallas base={meta['fallas']['base']} "
         f"veg={meta['fallas']['veg']} ref={meta['fallas']['ref']} ug={meta['fallas']['ug']}")


# ════════════════════════════════════════════════════════════
# E3 — Parámetros económicos
# ════════════════════════════════════════════════════════════

def etapa_E3(forzar=False, _d=None):
    """Traslado literal del bloque data_loader:349-527. Estos valores
    PISAN los del notebook a propósito: son las correcciones posteriores
    al estudio congelado."""
    if _existe("costos.pkl") and not forzar:
        _log("E3 costos: ya existe, se omite")
        return
    if not _existe("red.pkl"):
        raise RuntimeError("E3 requiere E0 (red.pkl).")
    red = _cargar("red.pkl")
    GENS, ESS, BUSES = red["GENS"], red["ESS"], red["BUSES"]
    LINES, sw_cand_list = red["LINES"], red["sw_cand_list"]

    r_econ, n_econ = 0.1, 30
    frp = r_econ * (1 + r_econ) ** n_econ / ((1 + r_econ) ** n_econ - 1)
    frp_movil = r_econ * (1 + r_econ) ** 10 / ((1 + r_econ) ** 10 - 1)

    VoLL = {b: 5 for b in BUSES}

    c = {
        "r": r_econ, "años": n_econ, "frp": frp, "frp_movil": frp_movil,
        "VoLL": VoLL,
        "C_sub": {s: 102.975 / TIPO_CAMBIO for s in S},   # tarifa inyección CGE / USD
        "DG_sol": [{g: 863 for g in GENS}, {g: 863 * 0.01 for g in GENS}],
        "DG_eol": [{g: 1504 for g in GENS}, {g: 1504 * 0.015 for g in GENS}],
        "DG_fos": [{g: 576 for g in GENS}, {g: 0.0238 for g in GENS},
                   {g: 999.54 / 0.85 for g in GENS}],   # al 18-06-2026
        # ESS 2h: valores del notebook, NUNCA redefinidos por data_loader.
        # El modelo vigente (optimization_3) usa ESS 5h y no lee esta clave,
        # pero optimization_base sí la exige aunque la medida esté apagada.
        "ESS 2h": [{e: 1427 for e in ESS}, {e: 34.401 for e in ESS}],
        "ESS 4h": [{e: 722 for e in ESS}, {e: 722 * 0.02 for e in ESS}],
        "ESS 5h": [{e: 805 for e in ESS}, {e: 805 * 0.02 for e in ESS}],
        "ESS 6h": [{e: 935 for e in ESS}, {e: 935 * 0.02 for e in ESS}],
        "MEG": [300, 20, 999.54 / 0.85],
        "MESS": [50, 1000, 20],
        "SWITCH": [10000, 5.5],
        "RCS": [15000, 5.5],
        "MS": [6000, 5.5],
        "EDRP_incentivo": [1.0, 2.0, 3.0, 4.0],
        "alpha_edrp": 0.20,
        "h_ess_4h": 4, "h_ess_5h": 5, "h_ess_6h": 6,
    }

    # Medidas por km de línea
    long_km = red["longitud_km_linea"]
    CICLO_VEG_ANIOS = 4
    c["VEG"] = [3000 / 1.60934, (700 / 1.60934) / CICLO_VEG_ANIOS, long_km]
    c["REF"] = [430_000 / 1.60934, 1_700 / 1.60934]     # NY DPS/IEc 2023, overhead
    c["UG"] = [3_600_000 / 1.60934, 900 / 1.60934]      # NY DPS/IEc 2023, underground
    # Enlaces nuevos (tie-lines candidatas a construir): [CAPEX USD/km,
    # OPEX fijo USD/km·año]. Una línea nueva cuesta al menos lo que reforzar una
    # existente: CAPEX = el de REF (misma fuente, NY DPS/IEc 2023); OPEX fijo 2 %
    # del CAPEX/año, como en El Melón. Antes: placeholder de 200 kUSD/km.
    c["TIE"] = [c["REF"][0], c["REF"][0] * 0.02]
    c["tie_candidatas"] = list(red["tie_indices"])
    c["longitud_km_linea"] = long_km
    c["lineas_veg"] = list(sw_cand_list)
    c["lineas_ref"] = list(sw_cand_list)
    c["lineas_ug"] = list(sw_cand_list)
    c["sw_cand_list"] = list(sw_cand_list)

    _aplicar_costos_caso(c, GENS)
    _guardar("costos.pkl", c)


# Costos propios de un caso: `artefactos/<caso>/costos_caso.json`, en CLP tal
# como los informa la empresa, con su fuente. Solo se reemplaza lo que el
# archivo trae (CAPEX; los OPEX y el resto quedan como arriba). Sin archivo,
# E3 queda idéntico al de la validación. Claves desconocidas = error, para que
# un nombre mal escrito no deje un costo viejo en silencio.
_CLAVES_COSTOS_CASO = {
    "fuente", "notas", "tipo_cambio_clp_usd", "C_sub_clp_kwh", "diesel_clp_m3",
    "RCS_clp", "MS_clp", "REF_clp_km", "UG_clp_km", "TIE_clp_km",
    "MEG_arriendo_clp_evento", "vida_meg_anios",
}


def _aplicar_costos_caso(c, GENS):
    ruta = ART_DIR / "costos_caso.json"
    if not ruta.exists():
        return
    cc = json.loads(ruta.read_text(encoding="utf-8"))
    otras = set(cc) - _CLAVES_COSTOS_CASO
    if otras:
        raise ValueError(f"{ruta.name}: claves desconocidas {sorted(otras)}; "
                         f"válidas: {sorted(_CLAVES_COSTOS_CASO)}")
    tc = float(cc.get("tipo_cambio_clp_usd", TIPO_CAMBIO))
    usd = lambda k: float(cc[k]) / tc
    if "C_sub_clp_kwh" in cc:
        c["C_sub"] = {s: usd("C_sub_clp_kwh") for s in S}
    if "diesel_clp_m3" in cc:
        # USD/m³ ÷ 0,85 kg/L = USD/t; el modelo lo multiplica por 0,000242 t/kWh
        comb = usd("diesel_clp_m3") / 0.85
        c["DG_fos"][2] = {g: comb for g in GENS}
        c["MEG"][2] = comb
    for clave, k in (("RCS_clp", "RCS"), ("MS_clp", "MS"), ("REF_clp_km", "REF"),
                     ("UG_clp_km", "UG"), ("TIE_clp_km", "TIE")):
        if clave in cc:
            c[k][0] = usd(clave)
    if "MEG_arriendo_clp_evento" in cc:
        c["MEG_arriendo"] = [usd("MEG_arriendo_clp_evento")]
    if "vida_meg_anios" in cc:
        n = int(cc["vida_meg_anios"])
        c["frp_meg"] = c["r"] * (1 + c["r"]) ** n / ((1 + c["r"]) ** n - 1)
    c["_costos_caso"] = {"archivo": ruta.name, "fuente": cc.get("fuente", ""),
                         "tipo_cambio": tc}
    _log(f"E3 costos del caso desde {ruta.name} (TC {tc:g} CLP/USD)")


# ════════════════════════════════════════════════════════════
# Ensamblado del dict `data` que esperan los optimizadores
# ════════════════════════════════════════════════════════════

def construir_data(sin_fallas=False, extras=None):
    """Arma el dict `data` con la misma forma que consumen
    optimization_base y optimization_3_corregido, desde los artefactos."""
    red = _cargar("red.pkl")
    costos = _cargar("costos.pkl")
    fallas = np.load(ART_DIR / "fallas.npz")

    num_esc, n_buses = red["numesc"], len(red["BUSES"])
    # sin fallas (E4) no hay año de clima: eólico promedio de los años
    dem, sol, eol = expandir_perfiles(num_esc, n_buses, eolico_promedio=sin_fallas)

    f = fallas["faultlinMC"]
    data = dict(costos)
    data.update({
        "net": red["net"],
        "bus_zone": red["bus_zone"],
        "P_D_base": red["P_D_base"],
        "N_eventos": red["N_eventos"],
        "numesc": num_esc,
        "demand_factor": dem,
        "alphaDGsolar": sol,
        "alphaDGeolica": eol,
        "faultlinMC": np.zeros_like(f) if sin_fallas else f,
        "repairMC": np.zeros_like(f) if sin_fallas else fallas["repairMC"],
        "faultlinMC_veg": fallas["faultlinMC_veg"],
        "faultlinMC_ref": fallas["faultlinMC_ref"],
        "faultlinMC_ug": fallas["faultlinMC_ug"],
    })
    data.update(RAM.datos_para_optimizador(red, fallas, sin_fallas=sin_fallas))
    if extras:
        data.update(extras)
    return data, red


# Tope de capacidad de DG por barra, como fracción de su demanda base.
# Se fija acá y NO en cada optimizador para que E4/E5 (optimization_base) y
# las estrategias (optimization_6_oficial) no puedan quedar con supuestos
# distintos: si el caso sin fallas se calcula con 0.3 y la estrategia con 1.0,
# el CAPEX/OPEX de DG del objetivo queda referido a otra capacidad máxima.
from limites_dg import MODOS as MODOS_LIM_DG

LIM_DG_DEFAULT = 1.0
LIM_DG_MODO_DEFAULT = "bus"


def config_medida(red, horas, esc, lim_dg=LIM_DG_DEFAULT,
                  lim_dg_modo=LIM_DG_MODO_DEFAULT, lim_dg_bus=None,
                  lim_dg_conjunto=False, **medidas):
    """Config base de optimization_base/6 con todas las medidas apagadas;
    `medidas` enciende las que correspondan."""
    cfg = {
        "lim_dg": float(lim_dg),
        # Agrupación del tope de DG: "bus" (histórico, una barra por grupo)
        # o un agregado por "alimentador"/"zona"/"seccion" que libera el
        # emplazamiento dentro del grupo. Ver `limites_dg.py`.
        "lim_dg_modo": str(lim_dg_modo),
        "lim_dg_bus": None if lim_dg_bus is None else float(lim_dg_bus),
        # True: solar + eólica comparten UN tope (suma ≤ lim_dg·demanda del grupo).
        # False (histórico): cada una tiene el suyo. Ver limites_dg.aplicar_tope_conjunto.
        "lim_dg_conjunto": bool(lim_dg_conjunto),
        "dg_sol": False, "dg_eol": False, "dg_fos": False,
        "ess_2h": False, "ess_4h": False, "ess_5h": False, "ess_6h": False,
        "reconfig_topo": False,
        # Enlaces normalmente abiertos de ESTA red (E0 los toma del paquete
        # de red o, con app.pkl, de la convención "las tres últimas"). Los
        # optimizadores los daban por fijos en LINES[-3:]; ahora los leen de
        # acá, que es lo único correcto en una red arbitraria.
        "tie_indices": list(red["tie_indices"]),
        "inv_tie": False, "tie_candidatas": red["tie_indices"],
        "prior_criticos": False, "buses_criticos": red["buses_criticos"],
        "meg": False, "n_meg": 0, "P_meg_max": 0,
        "T_tra_meg": {b: 0 for b in red["BUSES"]},
        "mess": False, "n_mess": 0, "h_mess": 0,
        "E_cap_mess": 0, "E_cap_mess_total": 0, "P_mess_max": 0, "T_tra_mess": 0,
        "inv_switch_rcs": False, "inv_switch_ms": False,
        "sw_candidatas": red["sw_cand_list"], "T_maniobra_MS": 0,
        # Aislación de la falla por seccionadores preexistentes. Apagado =
        # se saca de servicio solo la línea averiada (aislación perfecta
        # implícita). Encendido = la falla deja fuera a toda la sección
        # delimitada por `seccionadores_existentes` (índices de LINES).
        "aislacion_fallas_fija": False,
        "seccionadores_existentes": red.get("seccionadores_existentes", []),
        "edrp": False,
        "numesc": red["numesc"],
        "esc_debug": int(esc),
        "horas": int(horas),
    }
    cfg.update(medidas)
    return cfg


# ════════════════════════════════════════════════════════════
# E4 — Caso SIN FALLAS (faultlinMC = 0)
# ════════════════════════════════════════════════════════════
# Con faultlinMC en ceros el resultado NO depende del escenario:
# demand_factor y alphaDG* son idénticos para los 100 escenarios
# (verificado en E1). Por eso son 3 corridas totales, no 3 por escenario.
#
# Orden interno obligatorio:
#   1) base sin fallas          → P_SUB_tot_sin_fallas
#   2) DG solar (usa ese valor) → CAPEX/OPEX_DG_sol_sin_fallas
#   3) DG eólica                → CAPEX/OPEX_DG_eol_sin_fallas
# Es la secuencia que estaba comentada en caso_base_sin_fallas.py:120-145.

def _p_sub_total(res_lista):
    return sum(sum(h.get("P_SUB", {}).get(s, 0) for s in S) for h in res_lista)


def _capex_opex_dg(res_estrategia, res_base, data, tecnologia):
    """Réplica EXACTA de helpers_inversion._extraer_costos_objetivo_DG
    para la parte que consume optimization_3 (capex_anual / opex_anual).
    Se reimplementa acá porque importar helpers_inversion arrastraría
    data_loader y caso_base, que es la circularidad que el pipeline rompe.

    OJO (documentado, NO corregido para no alterar la continuidad):
    `opex_anual` NO es solo OPEX — incluye `cost_ahorro_sub`, el crédito
    por dejar de comprar a la subestación. Por eso sale NEGATIVO. Como
    optimization_3:1034 lo usa con signo cambiado (`-OPEX_sin_fallas`),
    termina sumando ~2.8-3.2 M al objetivo. Es una CONSTANTE: no altera
    la solución óptima, solo desplaza el valor reportado del objetivo.
    """
    GENS = list(data["net"].bus.index)
    frp = data["frp"]
    C_inv = data["DG_sol"][0] if tecnologia == "sol" else data["DG_eol"][0]
    C_op = data["DG_sol"][1] if tecnologia == "sol" else data["DG_eol"][1]
    clave_cap = "CAP_DG_sol_kW" if tecnologia == "sol" else "CAP_DG_eol_kW"

    hora0 = res_estrategia[0]
    capex = sum(hora0.get(clave_cap, {}).get(g, 0) * C_inv[g] * frp for g in GENS)
    opex = sum(hora0.get(clave_cap, {}).get(g, 0) * C_op[g] for g in GENS)

    C_sub = data["C_sub"]
    cost_ahorro_sub = -(data["P_SUB_tot_sin_fallas"]
                        - _p_sub_total(res_estrategia)) * C_sub.get(0, 1.0)
    return {"capex_anual": capex, "opex_anual": opex + cost_ahorro_sub}


def _slim_sin_fallas(res, BUSES, tecnologia=None):
    """Reduce un resultado de E4 a lo que realmente se usa después.

    El dict-por-hora completo pesa ~80 MB por caso (241 MB los tres) y
    NINGÚN módulo lo consume: data_loader solo lee los escalares. Se
    conservan además las capacidades instaladas y las series horarias de
    P_SUB/P_LC, que son lo que se necesita para graficar o justificar el
    caso sin fallas en la tesis — ~1 MB por caso en vez de 80.
    """
    slim = {
        "P_SUB": np.array([[h.get("P_SUB", {}).get(s, 0.0) for s in S] for h in res],
                          dtype=np.float64),
        "P_LC": np.array([[h.get("P_LC", {}).get(b, 0.0) for b in BUSES] for h in res],
                         dtype=np.float64),
        "horas": len(res),
    }
    for tec in ([tecnologia] if isinstance(tecnologia, str)
                else list(tecnologia or [])):
        clave = f"CAP_DG_{tec}_kW"
        slim[clave] = dict(res[0].get(clave, {}))
    return slim


def _capex_opex_dg_multi(res_estrategia, data, tecnologias):
    """CAPEX/OPEX de VARIAS tecnologías DG sobre UNA MISMA corrida.

    No es la suma de `_capex_opex_dg` por tecnología. El término
    `cost_ahorro_sub` —el crédito por dejar de comprar a la subestación—
    depende del P_SUB total de esa corrida, no de cada tecnología por
    separado, así que sumarlo una vez por tecnología lo duplicaría. Es
    exactamente el bug que `ECUACIONES_OBJETIVO_DG.md` §8.3 señala en
    `optimization_base`, y por eso acá se suma UNA sola vez, fuera del
    bucle de tecnologías.
    """
    GENS = list(data["net"].bus.index)
    frp = data["frp"]
    hora0 = res_estrategia[0]

    capex = opex = 0.0
    for tec in tecnologias:
        C_inv = data["DG_sol"][0] if tec == "sol" else data["DG_eol"][0]
        C_op = data["DG_sol"][1] if tec == "sol" else data["DG_eol"][1]
        clave_cap = "CAP_DG_sol_kW" if tec == "sol" else "CAP_DG_eol_kW"
        capex += sum(hora0.get(clave_cap, {}).get(g, 0) * C_inv[g] * frp for g in GENS)
        opex += sum(hora0.get(clave_cap, {}).get(g, 0) * C_op[g] for g in GENS)

    C_sub = data["C_sub"]
    cost_ahorro_sub = -(data["P_SUB_tot_sin_fallas"]
                        - _p_sub_total(res_estrategia)) * C_sub.get(0, 1.0)
    return {"capex_anual": capex, "opex_anual": opex + cost_ahorro_sub}


def etapa_E4(forzar=False, horas=None, etiqueta="", lim_dg=LIM_DG_DEFAULT,
             lim_dg_modo=LIM_DG_MODO_DEFAULT, lim_dg_bus=None,
             lim_dg_conjunto=False, **_):
    destino = ruta_sin_fallas(etiqueta)
    tope = dict(lim_dg=lim_dg, lim_dg_modo=lim_dg_modo, lim_dg_bus=lim_dg_bus,
                lim_dg_conjunto=lim_dg_conjunto)
    if destino.exists() and not forzar:
        _log(f"E4 sin fallas: {destino.name} ya existe, se omite")
        return
    for req in ("red.pkl", "costos.pkl"):
        if not _existe(req):
            raise RuntimeError(f"E4 requiere {req} (correr E0-E3 primero).")
    from optimization_base import resolver_optimizacion_base

    data, red = construir_data(sin_fallas=True)
    n_horas = horas or data["demand_factor"].shape[0]
    esc = 0   # irrelevante sin fallas, pero el solver necesita un índice

    _log(f"E4/1 caso base sin fallas ({n_horas} h, lim_dg={lim_dg}, "
         f"modo={lim_dg_modo}, lim_dg_bus={lim_dg_bus}, conjunto={lim_dg_conjunto})...")
    t = time.time()
    r1 = resolver_optimizacion_base(data=data, config=config_medida(red, n_horas, esc, **tope))
    base = [h for h in r1.get(esc, []) if h]
    p_sub_sf = _p_sub_total(base)
    _log(f"      P_SUB_tot_sin_fallas = {p_sub_sf:,.2f} kW  ({time.time()-t:.0f} s)")

    data["P_SUB_tot_sin_fallas"] = p_sub_sf

    salida = {
        "P_SUB_tot_sin_fallas": p_sub_sf,
        "sin_fallas": _slim_sin_fallas(base, red["BUSES"]),
    }

    for tec, flag in (("sol", "dg_sol"), ("eol", "dg_eol")):
        _log(f"E4/{2 if tec=='sol' else 3} sin fallas con DG {tec}...")
        t = time.time()
        r = resolver_optimizacion_base(
            data=data, config=config_medida(red, n_horas, esc, **tope, **{flag: True}))
        res = [h for h in r.get(esc, []) if h]
        co = _capex_opex_dg(res, base, data, tec)
        salida[f"sin_fallas_DG_{tec}"] = _slim_sin_fallas(res, red["BUSES"], tec)
        salida[f"CAPEX_DG_{tec}_sin_fallas"] = co["capex_anual"]
        salida[f"OPEX_DG_{tec}_sin_fallas"] = co["opex_anual"]
        _log(f"      CAPEX={co['capex_anual']:,.2f}  OPEX={co['opex_anual']:,.2f}  "
             f"({time.time()-t:.0f} s)")

    # E4/4 — AMBAS DG a la vez. Es la referencia que corresponde a cualquier
    # corrida que encienda las dos, y no es la suma de las dos anteriores:
    # solar y eólica compiten por el MISMO tope de `lim_dg` por alimentador,
    # así que el óptimo conjunto sin fallas instala menos de cada una que sus
    # óptimos individuales. Restar las dos constantes por separado infla el
    # crédito. Ver ECUACIONES_OBJETIVO_DG.md §8.2.
    #
    # Las combinaciones que llevan UNA sola DG no necesitan referencia propia:
    # con el tope activo (que es el caso hoy, E4 instala exactamente el 30 %
    # del alimentador en las dos tecnologías) el óptimo sin fallas de esa DG
    # no cambia por agregarle ESS u otra medida — ya está en el techo.
    _log("E4/4 sin fallas con AMBAS DG...")
    t = time.time()
    r = resolver_optimizacion_base(
        data=data,
        config=config_medida(red, n_horas, esc, **tope, dg_sol=True, dg_eol=True))
    res = [h for h in r.get(esc, []) if h]
    if res:
        co = _capex_opex_dg_multi(res, data, ("sol", "eol"))
        salida["sin_fallas_DG_ambas"] = _slim_sin_fallas(res, red["BUSES"], ("sol", "eol"))
        salida["CAPEX_DG_ambas_sin_fallas"] = co["capex_anual"]
        salida["OPEX_DG_ambas_sin_fallas"] = co["opex_anual"]
        cs = sum(salida["sin_fallas_DG_ambas"]["CAP_DG_sol_kW"].values())
        ce = sum(salida["sin_fallas_DG_ambas"]["CAP_DG_eol_kW"].values())
        indiv = (sum(salida["sin_fallas_DG_sol"]["CAP_DG_sol_kW"].values())
                 + sum(salida["sin_fallas_DG_eol"]["CAP_DG_eol_kW"].values()))
        _log(f"      CAPEX={co['capex_anual']:,.2f}  OPEX={co['opex_anual']:,.2f}  "
             f"({time.time()-t:.0f} s)")
        _log(f"      capacidad conjunta {cs:,.1f} sol + {ce:,.1f} eol = {cs+ce:,.1f} kW "
             f"vs {indiv:,.1f} kW sumando los individuales "
             f"({100*((cs+ce)/indiv-1):+.1f} %)")
        capex_indiv = (salida["CAPEX_DG_sol_sin_fallas"]
                       + salida["CAPEX_DG_eol_sin_fallas"])
        _log(f"      CAPEX conjunto {co['capex_anual']:,.0f} vs {capex_indiv:,.0f} "
             f"sumando los individuales ({100*(co['capex_anual']/capex_indiv-1):+.1f} %) "
             f"← lo que corrige §8.2")
    else:
        _log("      SIN SOLUCIÓN — no se escribe la referencia conjunta")

    # Trazabilidad: el límite de capacidad de DG es justamente lo que hace
    # que dos artefactos de E4 no sean comparables, así que queda anotado
    # junto a las cifras que produjo.
    salida["_meta"] = {
        "etiqueta": etiqueta,
        "generado": time.strftime("%Y-%m-%d %H:%M:%S"),
        "horas": n_horas,
        "lim_dg": float(lim_dg),
        "lim_dg_modo": str(lim_dg_modo),
        "lim_dg_bus": None if lim_dg_bus is None else float(lim_dg_bus),
        "lim_dg_conjunto": bool(lim_dg_conjunto),
    }
    _guardar(destino.name, salida)




# ════════════════════════════════════════════════════════════
# E5 — Baseline CON FALLAS, por escenario
# ════════════════════════════════════════════════════════════
# Se guarda en formato de arrays, no como lista de dicts por hora: el
# dict completo pesa ~100 MB por escenario (≈10 GB los 100), mientras
# que P_LC/P_SUB/daño en arrays son ~0.5 MB. La función
# cargar_baseline() los rehidrata al formato que esperan los helpers.

_CAMPOS_BASELINE = ("P_LC", "P_SUB", "dano")


def etapa_E5(forzar=False, escenarios=None, horas=None,
             seccionadores=None, **_):
    for req in ("red.pkl", "costos.pkl", "fallas.npz"):
        if not _existe(req):
            raise RuntimeError(f"E5 requiere {req} (correr E0-E3 primero).")
    from optimization_base import resolver_optimizacion_base
    from aislacion import variante as _variante

    data, red = construir_data(sin_fallas=False)
    n_horas = horas or data["demand_factor"].shape[0]
    BUSES = red["BUSES"]
    escenarios = escenarios if escenarios is not None else range(red["numesc"])

    # Sin --seccionadores se reproduce el supuesto histórico (aislación
    # perfecta) y se escribe en la carpeta de siempre.
    sec = sorted(seccionadores or [])
    usar_aisl = bool(sec)
    var = _variante(usar_aisl, sec)
    destino_dir = base_dir(var)
    _log(f"E5 variante '{var}'"
         + (f" — seccionadores en {sec}" if sec else " — sin aislación")
         + f" → {destino_dir}")

    destino_dir.mkdir(parents=True, exist_ok=True)
    for esc in escenarios:
        destino = destino_dir / f"esc_{esc:03d}.npz"
        if destino.exists() and not forzar:
            _log(f"E5 esc {esc}: ya existe, se omite")
            continue
        t = time.time()
        cfg = config_medida(red, n_horas, esc)
        cfg["aislacion_fallas_fija"] = usar_aisl
        cfg["seccionadores_existentes"] = sec
        r = resolver_optimizacion_base(data=data, config=cfg)
        res = [h for h in r.get(esc, []) if h]
        if not res:
            _log(f"E5 esc {esc}: SIN SOLUCIÓN, se omite")
            continue

        # float64, no float32: en float32 la suma de 8760×15 valores
        # acumula ~5e-8 de error relativo, suficiente para que el ENS
        # del escenario congelado no reproduzca su valor histórico al
        # cuarto decimal. El ahorro de disco (53 vs 105 MB los 100) no
        # justifica perder reproducibilidad exacta.
        P_LC = np.array([[h.get("P_LC", {}).get(b, 0.0) for b in BUSES] for h in res],
                        dtype=np.float64)
        P_SUB = np.array([[h.get("P_SUB", {}).get(s, 0.0) for s in S] for h in res],
                         dtype=np.float64)
        dano = np.array([h.get("n°lineas_danadas", 0) for h in res], dtype=np.int16)
        # Líneas fuera de servicio = averiadas + arrastradas por la zona de
        # aislación. Sin aislación coincide con `dano`; con aislación es el
        # número que corresponde graficar en la curva de resiliencia.
        fuera = np.array([h.get("n°lineas_fuera_servicio",
                                h.get("n°lineas_danadas", 0)) for h in res],
                         dtype=np.int16)

        # Detalle POR LÍNEA, no solo el conteo: sin esto el grafo del caso
        # base (pestaña Entradas) no puede pintar la zona de aislación,
        # porque `cargar_baseline` no tendría de dónde sacar qué línea
        # quedó fuera. Son matrices booleanas de 8760×n_líneas: comprimidas
        # ocupan del orden de kilobytes.
        _LINES = red["LINES"]
        _n_l = len(_LINES)
        _pos_l = {l: i for i, l in enumerate(_LINES)}

        def _matriz(clave):
            m = np.zeros((len(res), _n_l), dtype=np.uint8)
            for i, h in enumerate(res):
                for x in h.get(clave, []):
                    j = _pos_l.get(x, _pos_l.get(int(x)) if str(x).lstrip("-").isdigit() else None)
                    if j is not None:
                        m[i, j] = 1
            return m

        np.savez_compressed(
            destino, P_LC=P_LC, P_SUB=P_SUB, dano=dano, fuera=fuera,
            dano_l=_matriz("lineas_danadas_indices"),
            fuera_l=_matriz("lineas_fuera_servicio_indices"),
            seccionadores=np.array(sec, dtype=np.int32),
        )
        _log(f"E5 esc {esc}: ENS={P_LC.sum():,.2f} kW  ΣP_SUB={P_SUB.sum():,.2f} kW  "
             f"({time.time()-t:.0f} s) → {destino.name}")


def cargar_baseline(esc, variante="base", art_dir=None):
    """Rehidrata el baseline del escenario `esc` al formato lista-de-dicts
    por hora que consumen indicadores.py y helpers_*.py.

    `variante` selecciona el supuesto de aislación (ver `base_dir`).
    """
    ruta = base_dir(variante, art_dir) / f"esc_{esc:03d}.npz"
    if not ruta.exists():
        _extra = ""
        if variante != "base":
            _sec = variante[4:].replace("-", ",")
            _extra = f" --seccionadores {_sec}"
        raise FileNotFoundError(
            f"No hay baseline para el escenario {esc} en la variante "
            f"'{variante}'. Correr: python pipeline.py --etapas E5 "
            f"--escenarios {esc}{_extra}")
    z = np.load(ruta)
    red = _cargar("red.pkl", art_dir)
    BUSES, P_D_base = red["BUSES"], red["P_D_base"]
    _perf = np.load(Path(art_dir or ART_DIR) / "perfiles.npz")
    # Misma demanda que ve el modelo (expandir_perfiles): el factor por barra
    # si existe; el agregado solo cuando no hay perfil por barra.
    perf = _perf["demanda_por_bus"] if "demanda_por_bus" in _perf.files else _perf["demanda"]
    perf_bus = perf.ndim == 2

    P_LC, P_SUB, dano = z["P_LC"], z["P_SUB"], z["dano"]
    # `fuera` y el detalle por línea solo existen en baselines generados
    # desde que hay aislación; en los antiguos `fuera` coincide con `dano`
    # y el detalle simplemente no se agrega (el grafo cae al comportamiento
    # anterior, pintando la falla desde faultlinMC).
    fuera = z["fuera"] if "fuera" in z.files else dano
    LINEAS = red["LINES"]
    dano_l = z["dano_l"] if "dano_l" in z.files else None
    fuera_l = z["fuera_l"] if "fuera_l" in z.files else None
    sec_l = z["seccionadores"].tolist() if "seccionadores" in z.files else []
    _sec_set = set(sec_l)
    salida = []
    for i in range(P_LC.shape[0]):
        P_D = {b: P_D_base[b] * (perf[i, bi] if perf_bus else perf[i])
               for bi, b in enumerate(BUSES)}
        p_lc = {b: float(P_LC[i, bi]) for bi, b in enumerate(BUSES)}
        extra = {}
        if dano_l is not None and fuera_l is not None:
            _dan = [LINEAS[j] for j in np.where(dano_l[i] == 1)[0]]
            _fue = [LINEAS[j] for j in np.where(fuera_l[i] == 1)[0]]
            _dset = set(_dan)
            extra = {
                "lineas_danadas_indices": _dan,
                "lineas_fuera_servicio_indices": _fue,
                "lineas_seccionadas_indices": [x for x in _fue
                                               if x not in _dset and x in _sec_set],
                "lineas_aisladas_indices": [x for x in _fue
                                            if x not in _dset and x not in _sec_set],
                "seccionadores": sec_l,
            }
        salida.append({
            "escenario": esc, "hora": i + 1, "status": "optimal",
            "n°lineas_fuera_servicio": int(fuera[i]),
            **extra,
            "ENS_MWh": float(P_LC[i].sum()),
            "carga_servida_MW": sum(P_D[b] - p_lc[b] for b in BUSES),
            "n°lineas_danadas": int(dano[i]),
            "P_LC": p_lc,
            "P_SUB": {s: float(P_SUB[i, si]) for si, s in enumerate(red["S"])},
            "P_D": P_D,
            "Pb": {b: P_D[b] - p_lc[b] for b in BUSES},
        })
    return salida


def indicadores_baseline(esc, variante="base", n_horas=None, art_dir=None, guardar_cache=True):
    """SAIDI/SAIFI/ENS/ASAI y costo de ENS del CASO BASE de un escenario.

    Es la referencia contra la que se mide cualquier medida. Sin ella solo
    se pueden comparar NIVELES entre estrategias ("esta deja menos ENS que
    aquella"), nunca cuánto mejoró cada una respecto de no hacer nada, que
    es lo único que permite un costo por unidad de mejora y una curva de
    abatimiento.

    No re-resuelve nada: rehidrata el baseline de E5 (`cargar_baseline`) y
    le aplica las MISMAS funciones de indicadores.py que usa el barrido
    para las estrategias, así que los dos lados de la resta son coherentes.

    `n_horas` recorta el horizonte. Importa: comparar una corrida de 24 h
    contra un baseline de 8760 h daría una "mejora" inventada del 99%.

    El resultado se cachea en <baselines>/<variante>/indicadores_base.json,
    porque rehidratar 8760 horas cuesta ~1 s por escenario y esto se pide
    una vez por cada escenario del barrido, en cada vista.
    """
    ruta = base_dir(variante, art_dir) / "indicadores_base.json"
    cache = {}
    if ruta.exists():
        try:
            cache = json.loads(ruta.read_text(encoding="utf-8"))
        except Exception:                       # noqa: BLE001 — caché corrupta
            cache = {}                          # se regenera, no es un dato
    clave = f"esc_{int(esc):03d}|{int(n_horas) if n_horas else 'todo'}"
    if clave in cache:
        return cache[clave]

    # Import local: indicadores.py no depende de nada del pipeline, pero
    # tenerlo arriba haría que importar pipeline arrastre el módulo entero
    # para una función que casi nadie llama.
    from indicadores import fn_ASAI, fn_ENS_total, fn_SAIDI, fn_SAIFI

    res = cargar_baseline(esc, variante, art_dir)
    if n_horas:
        res = res[:int(n_horas)]
    red = _cargar("red.pkl", art_dir)
    costos = _cargar("costos.pkl", art_dir)
    BUSES, P_D_base = red["BUSES"], red["P_D_base"]
    cli = red.get("clientes_bus")              # None: regla de clientes por kW
    cli_total = clientes_denominador_saifi(red, art_dir)
    horas_idx = range(1, len(res) + 1)

    saidi = float(fn_SAIDI(res, P_D_base, BUSES, horas_idx, clientes=cli))
    VoLL, N_ev = costos.get("VoLL", {}), red.get("N_eventos", 1)
    out = {
        "escenario": int(esc),
        "variante": variante,
        "horas": len(res),
        "ens": float(fn_ENS_total(res, BUSES, horas_idx)),
        "saidi": saidi,
        "saifi": float(fn_SAIFI(res, P_D_base, BUSES, horas_idx, clientes=cli,
                                clientes_denominador=cli_total)),
        # Sobre el horizonte simulado, no sobre 8760 fijo (mismo criterio
        # que run_barrido_medidas.indicadores_de).
        "asai": float(fn_ASAI(saidi, len(res))),
        "costo_ens": float(sum(VoLL.get(b, 1.0) * h.get("P_LC", {}).get(b, 0.0)
                               for h in res for b in BUSES) * N_ev),
    }

    if not guardar_cache:
        return out
    cache[clave] = out
    ruta.parent.mkdir(parents=True, exist_ok=True)
    # Escritura atómica: la app puede estar leyendo esta caché mientras un
    # barrido corre en consola, y un JSON a medio escribir la dejaría sin
    # referencia hasta que alguien lo borre a mano.
    tmp = ruta.with_name(ruta.name + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(ruta)
    return out


def escenarios_disponibles(variante="base", art_dir=None):
    d = base_dir(variante, art_dir)
    if not d.exists():
        return []
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("esc_*.npz"))


def escenarios_muestra():
    """La muestra estratificada de escenarios (`seleccionar_escenarios.py`).

    OJO: es la muestra que el script PROPONE, no necesariamente la que se
    corrió. Para agregar fallas y comparar contra los resultados de una
    tanda hay que usar `escenarios_barrido(tanda)`, que lee los escenarios
    que esa tanda efectivamente resolvió — pueden no coincidir (una tanda
    puede haberse lanzado con otra lista, o quedar a medias).

    Devuelve (escenarios 0-idx, metadatos) o ([], {}) si no está el archivo.
    """
    ruta = ART_DIR / "escenarios_estratificados.json"
    if not ruta.exists():
        return [], {}
    try:
        d = json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return [], {}
    escs = sorted(int(e) for e in (d.get("escenarios") or []))
    meta = {k: d.get(k) for k in ("criterio", "n_estratos", "n_poblacion",
                                  "media_muestra", "media_poblacion",
                                  "sesgo_pct", "cola_excluida")}
    return escs, meta


def escenarios_barrido(tanda):
    """Escenarios (0-idx) que una tanda del barrido resolvió de verdad.

    Se leen de su `resumen_individual.csv`, o sea de lo que quedó en disco,
    no de la muestra que se planificó. Es EL conjunto contra el que se
    compara un mapa de fallas agregado: si el barrido evaluó las medidas
    sobre estos 10 escenarios, agregar las fallas sobre otros (los 100, o
    la muestra teórica) compara dos poblaciones distintas y el mapa deja de
    explicar los resultados que acompaña.

    `tanda` es la ruta relativa a `artefactos/`, igual que en la vista del
    barrido: `barrido/sec-0-2-9-10 severo`, `barrido 01-09/base`, etc.
    """
    if not tanda:
        return []
    ruta = ART_DIR / tanda / "resumen_individual.csv"
    if not ruta.exists():
        return []
    try:
        df = pd.read_csv(ruta, usecols=["escenario"])
    except Exception:                                    # noqa: BLE001
        return []
    return sorted({int(e) for e in df["escenario"].dropna()})


def variantes_disponibles():
    """Variantes de aislación que tienen al menos un baseline calculado."""
    out = ["base"] if escenarios_disponibles("base") else []
    if BASE_DIR.exists():
        for d in sorted(BASE_DIR.iterdir()):
            if d.is_dir() and any(d.glob("esc_*.npz")):
                out.append(d.name)
    return out


# ════════════════════════════════════════════════════════════
# E6 — Índice / estado
# ════════════════════════════════════════════════════════════

def _json_meta_previo():
    """meta.json que dejó la corrida anterior, o None."""
    ruta = ART_DIR / "meta.json"
    if not ruta.exists():
        return None
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return None


def etapa_E6(forzar=False, duraciones=None, **_):
    disp = escenarios_disponibles()
    # Los tiempos medidos se ACUMULAN entre corridas: cada invocación solo
    # conoce las etapas que ella corrió, y perder las anteriores dejaría a la
    # app sin con qué estimar las que ya no va a volver a correr.
    medidos = dict(((_json_meta_previo() or {}).get("duraciones") or {}))
    medidos.update(duraciones or {})
    meta = {
        "generado": time.strftime("%Y-%m-%d %H:%M:%S"),
        "etapas": {
            "E0_red": _existe("red.pkl"),
            "E1_perfiles": _existe("perfiles.npz"),
            "E2_fallas": _existe("fallas.npz"),
            "E3_costos": _existe("costos.pkl"),
            "E4_sin_fallas": _existe("sin_fallas.pkl"),
            "E5_baselines": len(disp),
        },
        "escenarios_con_baseline": disp,
        "esc_congelado_estudio": 3,
        "fuentes": {
            "fallas": str(RUTA_FAULTLIN.name),
            "congelado": str(RUTA_APP_PKL.name) + " (solo lectura)",
            "red_nueva": (str(RUTA_RED_NUEVA.name) if RUTA_RED_NUEVA.exists()
                          else None),
        },
        # Tiempos medidos por etapa en ESTA carpeta y ESTA máquina.
        "duraciones": medidos,
    }
    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    _log("→ meta.json")
    return meta


def mostrar_estado():
    print("=" * 62)
    print(f"ARTEFACTOS EN {ART_DIR}")
    print("=" * 62)
    for nombre, etapa in (("red.pkl", "E0"), ("perfiles.npz", "E1"),
                          ("fallas.npz", "E2"), ("costos.pkl", "E3"),
                          ("sin_fallas.pkl", "E4")):
        ruta = ART_DIR / nombre
        if ruta.exists():
            print(f"  [x] {etapa}  {nombre:<16} {ruta.stat().st_size/1e6:>8.1f} MB")
        else:
            print(f"  [ ] {etapa}  {nombre:<16}    (falta)")
    _otros_e4 = [e for e in sin_fallas_disponibles() if e]
    if _otros_e4:
        print("         otras variantes de E4 (sin fallas):")
        for e in _otros_e4:
            _r = ruta_sin_fallas(e)
            try:
                with open(_r, "rb") as _f:
                    _m = pickle.load(_f).get("_meta", {})
                _lim = f"Lim_DG={_m.get('lim_dg')}" if _m.get("lim_dg") else "sin meta"
            except Exception:
                _lim = "ilegible"
            print(f"           {e:<28} {_r.stat().st_size/1e6:>6.1f} MB  ({_lim})")
    disp = escenarios_disponibles()
    print(f"  [{'x' if disp else ' '}] E5  baselines       {len(disp)} escenario(s)")
    _otras = [v for v in variantes_disponibles() if v != "base"]
    if _otras:
        print("         variantes de aislación:")
        for v in _otras:
            _d = escenarios_disponibles(v)
            print(f"           {v:<28} {len(_d)} escenario(s)")
    if disp:
        print(f"         {disp if len(disp) <= 20 else str(disp[:20]) + ' ...'}")
        # Precisión: los baselines generados antes del cambio a float64
        # arrastran ~5e-8 de error relativo en las sumas (ENS, ΣP_SUB).
        # No cambia ninguna cifra reportable, pero impide reproducir el
        # valor histórico del escenario congelado al cuarto decimal.
        f32 = [e for e in disp
               if np.load(BASE_DIR / f"esc_{e:03d}.npz")["P_LC"].dtype == np.float32]
        if f32:
            print(f"         [!] {len(f32)} en float32 (precisión antigua). "
                  f"Para rehacerlos en float64:")
            print(f"             python pipeline.py --etapas E5 --forzar")
    print("=" * 62)


# ════════════════════════════════════════════════════════════

ETAPAS = {"E0": etapa_E0, "E1": etapa_E1, "E2": etapa_E2,
          "E3": etapa_E3, "E4": etapa_E4, "E5": etapa_E5, "E6": etapa_E6}


def _parse_escenarios(txt):
    if not txt:
        return None
    out = []
    for parte in txt.split(","):
        if "-" in parte:
            a, b = parte.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(parte))
    return out


def main():
    ap = argparse.ArgumentParser(description="Pipeline de insumos de la app de resiliencia")
    ap.add_argument("--etapas", default="", help="lista separada por comas, ej: E0,E1,E2,E3")
    ap.add_argument("--escenarios", default="", help="para E5, ej: 0-19 o 3,7,42")
    ap.add_argument("--horas", type=int, default=None, help="horizonte reducido (pruebas)")
    ap.add_argument("--forzar", action="store_true", help="rehacer aunque exista")
    ap.add_argument("--estado", action="store_true", help="mostrar qué hay y salir")
    ap.add_argument("--lim-dg", dest="lim_dg", type=float, default=LIM_DG_DEFAULT,
                    help=f"para E4: tope de capacidad de DG, como fracción de "
                         f"la demanda base del grupo (default: {LIM_DG_DEFAULT}). "
                         f"Usar junto con --etiqueta para no pisar el artefacto "
                         f"calculado con otro tope.")
    ap.add_argument("--lim-dg-modo", dest="lim_dg_modo", default=LIM_DG_MODO_DEFAULT,
                    choices=list(MODOS_LIM_DG),
                    help=f"para E4: sobre qué grupo se agrega el tope de DG. "
                         f"'bus' (default) = el histórico, una barra por grupo, "
                         f"cada barra topada por su propia demanda. "
                         f"'alimentador'/'zona'/'seccion' = tope agregado por "
                         f"grupo, que deja el emplazamiento libre dentro de él. "
                         f"Ver limites_dg.py.")
    ap.add_argument("--lim-dg-bus", dest="lim_dg_bus", type=float, default=None,
                    help="para E4: tope adicional POR BARRA, en múltiplos de su "
                         "demanda base. Recomendado con --lim-dg-modo distinto "
                         "de 'bus': sin él, el agregado permite colgar toda la "
                         "holgura del grupo de una barra cuyo lateral no la "
                         "soporta.")
    ap.add_argument("--lim-dg-conjunto", dest="lim_dg_conjunto", action="store_true",
                    help="para E4: solar + eólica comparten UN tope (la suma ≤ "
                         "--lim-dg · demanda del grupo) en vez de uno por "
                         "tecnología. Usar con --etiqueta (p. ej. LIM-0.30-ALIM-CONJ).")
    ap.add_argument("--etiqueta", default="",
                    help="para E4: guarda el caso sin fallas como "
                         "sin_fallas_<etiqueta>.pkl en vez de pisar el "
                         "histórico. Usar cuando cambió un supuesto del "
                         "modelo (p. ej. el límite de capacidad de DG).")
    ap.add_argument("--seccionadores", default="",
                    help="para E5: líneas con seccionador, ej: 0,1,2,9. "
                         "El baseline se guarda en su propia variante y NO "
                         "sobrescribe el histórico (sin aislación).")
    ap.add_argument("--horas-reparacion", dest="horas_reparacion", type=int, default=None,
                    help="para E2: horas que tarda en reponerse una línea fallada. "
                         "Es un parámetro del estudio (entra en SAIDI y ENS). Sin "
                         "este flag se toma el valor con el que se simuló la matriz "
                         "de fallas (artefactos/faultlin_meta.json), y si esa corrida "
                         f"no dejó registro, {HORAS_REPARACION_DEFAULT} (el histórico).")
    ap.add_argument("--fuente", default="app_pkl", choices=list(FUENTES),
                    help="para E0/E1: de dónde sale la red. 'app_pkl' (default) "
                         "es el estudio congelado, y su red es la CIGRE de 15 "
                         "barras, así que SOLO vale para el caso de validación: "
                         "en cualquier otro caso se rechaza. 'red_nueva' lee "
                         "<caso>/red_nueva.pkl, el paquete que exporta la "
                         "pestaña 'Datos de entrada', y es lo que corresponde "
                         "para una red propia. Las demás etapas no cambian: "
                         "leen el red.pkl del caso.")
    ap.add_argument("--artefactos", default=None,
                    help="carpeta de artefactos del caso sobre el que trabajar. "
                         "Sin este flag se usa $ERAD_ARTEFACTOS y, si no está, "
                         "artefactos/ (el caso de validación de la tesis). Cada "
                         "caso tiene su propia red, sus fallas y sus baselines, "
                         "así que apuntar al caso equivocado no da error: da "
                         "resultados de otra red.")
    args = ap.parse_args()

    # Antes que cualquier lectura, incluido --estado: si esto quedara más abajo,
    # `pipeline.py --artefactos X --estado` reportaría el caso por defecto.
    if args.artefactos:
        usar_artefactos(args.artefactos)
    _log(f"caso: {ART_DIR}")

    if args.estado or not args.etapas:
        mostrar_estado()
        if not args.etapas:
            return

    pedidas = [e.strip().upper() for e in args.etapas.split(",") if e.strip()]
    desconocidas = [e for e in pedidas if e not in ETAPAS]
    if desconocidas:
        sys.exit(f"Etapas desconocidas: {desconocidas}. Válidas: {list(ETAPAS)}")

    # E0-E3 comparten la lectura del app.pkl: se hace una sola vez.
    # Con --fuente red_nueva no se toca app.pkl en absoluto: la red viene del
    # paquete y los perfiles de los archivos sueltos.
    _d = None
    if args.fuente == "app_pkl" and any(e in pedidas for e in ("E0", "E1")):
        # `app.pkl` es el registro congelado del estudio de la tesis y la red que
        # trae es la CIGRE de 15 barras. Usarlo como fuente de OTRO caso llenaría
        # ese caso con la red del estudio sin dar error: se descubriría mucho
        # después, cuando los resultados no calzan con la red que se subió.
        if not casos.es_validacion(ART_DIR):
            sys.exit(
                f"--fuente app_pkl lee '{RUTA_APP_PKL.name}', el estudio congelado, "
                f"y su red es la de 15 barras del caso de validación. El caso "
                f"'{ART_DIR.name}' tiene que salir de su propio paquete:\n"
                f"    python pipeline.py --artefactos {ART_DIR.name} "
                f"--fuente red_nueva --etapas {args.etapas}\n"
                f"Ese paquete ({RUTA_RED_NUEVA.name}) lo escribe la pestaña "
                f"'Datos de entrada' al terminar el asistente de red.")
        _d = _leer_pkl_congelado()

    t0 = time.time()
    # Cuánto tardó cada etapa, para que la app deje de estimar a ojo. Los
    # tiempos dependen de la red (15 barras vs 179), del horizonte y de la
    # máquina: el mismo caso base de 720 h tardó 18,7 s en un equipo y 60,2 s
    # en otro. Una constante en el código no puede representar eso; lo medido
    # en este caso y esta máquina, sí.
    duraciones = {}
    for e in pedidas:
        _log(f"--- {e} ---")
        t_etapa = time.time()
        kwargs = {"forzar": args.forzar}
        if e in ("E0", "E1", "E2", "E3"):
            kwargs["_d"] = _d
        if e in ("E0", "E1"):
            kwargs["fuente"] = args.fuente
        if e == "E2":
            kwargs["horas_reparacion"] = args.horas_reparacion
        if e == "E5":
            kwargs["escenarios"] = _parse_escenarios(args.escenarios)
            kwargs["seccionadores"] = _parse_escenarios(args.seccionadores)
        if e == "E4":
            kwargs["etiqueta"] = args.etiqueta
            kwargs["lim_dg"] = args.lim_dg
            kwargs["lim_dg_modo"] = args.lim_dg_modo
            kwargs["lim_dg_bus"] = args.lim_dg_bus
            kwargs["lim_dg_conjunto"] = args.lim_dg_conjunto
        if e in ("E4", "E5"):
            kwargs["horas"] = args.horas
        ETAPAS[e](**kwargs)
        duraciones[e] = {
            "segundos": round(time.time() - t_etapa, 1),
            "horas": int(args.horas) if args.horas else None,
            "escenarios": (len(_parse_escenarios(args.escenarios)) or None
                           if e == "E5" else None),
            "cuando": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    etapa_E6(duraciones=duraciones)
    _log(f"listo en {time.time()-t0:.1f} s")
    mostrar_estado()


if __name__ == "__main__":
    main()
