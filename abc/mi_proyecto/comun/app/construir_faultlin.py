# ============================================================
# construir_faultlin.py — Matriz de fallas por Monte Carlo
# ============================================================
# Rescata a script de consola la simulación que hasta ahora solo existía
# COMENTADA dentro de "Prueba 12 oficial.ipynb" (celdas 11-14). Ese es el
# problema que este archivo resuelve: `FaultlinMC.npy` es la fuente de
# verdad de las fallas para todo el estudio (pipeline.py:20-24), pero el
# código que lo produjo no se podía volver a correr sin descomentar celdas
# a mano y sin saber cuál de las tres variantes del notebook era la vigente.
#
# La vigente es la lognormal (celda 14): θ=30 m/s, β=0.25. Se sabe porque
# es la misma curva con la que pipeline.py:428 calibra el refuerzo
# estructural — si la matriz base viniera de otra curva, el "ahorro" del
# refuerzo estaría medido contra una física distinta de la que generó las
# fallas.
#
# USO
#   python construir_faultlin.py --estado          # qué hay, sin escribir
#   python construir_faultlin.py --salida FaultlinMC_nueva.npy
#   python construir_faultlin.py --escenarios 100 --horas 8760
#
# NO sobrescribe FaultlinMC.npy sin --forzar: el .npy actual es el que
# generó todos los baselines de E5 y todo el barrido. Rehacerlo invalida
# esos artefactos (ver INSTRUCTIVO_DATOS.md, §6).
# ============================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import casos
import ramales as RAM

APP_DIR = Path(__file__).resolve().parent
from rutas import CASO_DIR, DATOS_DIR  # datos del caso activo (ver rutas.py)
# Carpeta del caso activo (ver casos.py). `--artefactos` la mueve, y con ella
# se mueven la red que se lee, el .npy que se escribe y faultlin_meta.json: una
# matriz de fallas solo sirve para la red con la que se sorteó, así que las tres
# cosas tienen que vivir en la misma carpeta.
ART_DIR = casos.art_dir_activo()
RUTA_WIND_NC = CASO_DIR / "Scenario Database" / "Inputs" / "BaseHoraEWEPELP20z.nc"

# — Parámetros de la curva de fragilidad vigente (celda 14 del notebook) —
THETA_DEFAULT = 30.0      # mediana de la lognormal [m/s]
BETA_DEFAULT = 0.25       # dispersión logarítmica
ALTURA_POSTE_M = 12.0     # altura del conductor de 20 kV
ALTURA_MEDICION_M = 10.0  # altura de la medición de viento del .nc
EXP_HELLMANN = 1 / 7      # perfil de viento en potencia (terreno abierto)
DUNN_ALPHA = 2e-17        # [1/km] Dunn et al. 2018, vía Shi et al. 2021 ec. (5)
DUNN_EXP = 9.91           # exponente del viento [m/s] en la misma curva
LARGO_FUENTE = "length_km"  # o "largo_expuesto_km" si red.pkl lo trae (cargar_red)
HORAS_ESPERA_DEFAULT = 3  # `linespera` del notebook: horas fuera tras fallar
ANIO_OFFSET_DEFAULT = 8760  # el .nc parte en 2029; el estudio usa 2030


def _log(msg):
    print(f"[faultlin] {msg}", flush=True)


# ════════════════════════════════════════════════════════════
# Insumos
# ════════════════════════════════════════════════════════════

def cargar_red(ruta=None):
    """Red desde artefactos/red.pkl (E0) o desde el paquete que exporta la
    pestaña 'Datos de entrada' (red_nueva.pkl).

    Devuelve (LINES, zona_por_linea, largo_km) donde zona_por_linea[i] es la
    zona climática 1..Z de la línea i, tomada de su bus de origen — el mismo
    criterio de `df_lines['zone']` en el notebook y de
    `_rho_por_zona_a_linea` en pipeline.py:405.

    `largo_km` sólo lo usa la curva empírica de Dunn, que es la única en que
    la probabilidad escala con el largo del tramo.
    """
    import pickle

    if ruta is None:
        ruta = ART_DIR / "red.pkl"
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe '{ruta}'. Generar la red primero:\n"
            f"    python pipeline.py --etapas E0")
    with open(ruta, "rb") as f:
        d = pickle.load(f)

    net = d["net"]
    LINES = d.get("LINES", list(net.line.index))
    bus_zone = d["bus_zone"]
    zona_por_linea = np.array(
        [int(bus_zone.get(int(net.line.from_bus[l]), 1)) for l in LINES], dtype=int)
    largo_km = np.asarray(net.line.length_km.loc[LINES].values, dtype=float)
    # Red reducida: el largo que se expone al viento no es el eléctrico. Los km
    # de los ramales podados van con su propio elemento de falla (ramales.py) y
    # los de cable no fallan por viento; `largo_expuesto_km` ya los descuenta.
    # Sin la clave (casos anteriores) se usa length_km, como siempre.
    global LARGO_FUENTE
    LARGO_FUENTE = "length_km"
    if d.get("largo_expuesto_km"):
        LARGO_FUENTE = "largo_expuesto_km"
        exp = d["largo_expuesto_km"]
        largo_km = np.asarray([float(exp.get(int(l), 0.0)) for l in LINES], dtype=float)
    # Posición (no índice de línea) de los enlaces dentro de LINES, para
    # poder anularlos con --enlaces-inmunes.
    ties = set(int(t) for t in d.get("tie_indices", []))
    tie_pos = [i for i, l in enumerate(LINES) if int(l) in ties]
    return LINES, zona_por_linea, largo_km, tie_pos


def cargar_ramales(ruta=None):
    """Ramales con fusible de la red (ver ramales.py), o None si no declara."""
    import pickle

    ruta = Path(ruta) if ruta is not None else ART_DIR / "red.pkl"
    with open(ruta, "rb") as f:
        d = pickle.load(f)
    ram = d.get(RAM.CLAVE_RED)
    if not ram:
        return None
    RAM.validar(ram)
    return {"id": list(ram["id"]), "km": np.asarray(ram["km"], dtype=float),
            "zona": np.asarray(ram["zona"], dtype=int)}


def cargar_viento(ruta_nc, horas, offset, escenario_clima="baseline"):
    """Serie horaria de viento por zona, ajustada a la altura del poste.

    `escenario_clima` elige el eje `lon` del NetCDF de eventos extremos:
      - "baseline" (lon=0): la tendencia climática SIN la amplificación de
        ráfaga de los eventos extremos.
      - "with_EWE" (lon=1): con la ráfaga aplicada (factor 1.263 sobre las
        horas que superan el umbral, ver Extreme_events_arclim_final.py).

    El default es "baseline" porque es el que usó el notebook
    (`Base085.wind.values[8760:17520, :, 0]`) y por lo tanto el que generó
    el FaultlinMC.npy histórico. Es una decisión heredada, no obviamente
    la correcta: ver INSTRUCTIVO_DATOS.md §3.4.
    """
    import xarray as xr

    ds = xr.open_dataset(ruta_nc)
    idx_lon = {"baseline": 0, "with_EWE": 1}[escenario_clima]
    total = ds["wind"].shape[0]
    if offset + horas > total:
        raise ValueError(
            f"El .nc tiene {total} horas y se pidieron {horas} desde la hora "
            f"{offset} (faltan {offset + horas - total}). Revisar --horas / "
            f"--anio-offset.")
    wind = ds["wind"].values[offset:offset + horas, :, idx_lon]
    factor = (ALTURA_POSTE_M / ALTURA_MEDICION_M) ** EXP_HELLMANN
    return np.asarray(wind, dtype=float) * factor


# ════════════════════════════════════════════════════════════
# Curva de fragilidad y Monte Carlo
# ════════════════════════════════════════════════════════════

def p_falla_lognormal(viento_ms, theta=THETA_DEFAULT, beta=BETA_DEFAULT):
    """P(falla | viento) con la lognormal calibrada. v<=0 -> 0."""
    from scipy.special import ndtr

    v = np.asarray(viento_ms, dtype=float)
    out = np.zeros_like(v)
    seguro = v > 0
    out[seguro] = ndtr(np.log(v[seguro] / theta) / beta)
    return out


def p_falla_dunn(viento_ms, largo_km, alpha=None, exp=DUNN_EXP):
    """Curva EMPÍRICA de líneas de distribución:  p = α · l[km] · v^exp.

    Dunn et al. (2018), ajustada sobre 12.000 fallas reales de líneas de
    distribución por viento en el Reino Unido (2003-2010). La citan
    Shi et al. (2021), IEEE Trans. Power Syst. 36(4), ec. (5), y el paper
    de IJEPES S0142061521010759 con el mismo ajuste.

    Dos cosas la distinguen de la lognormal estructural:

    - **Escala con el largo del tramo.** Es la única que lo hace, y evita
      tener que suponer un número de postes por línea.
    - **Incluye la caída de árboles**, que es el mecanismo dominante de
      falla en distribución (Karagiannakis et al. 2025, §4.3). Las curvas
      analíticas (Panteli, Darestani) modelan sólo la rotura estructural
      del poste, así que multiplicar por un factor de vegetación sobre
      ellas mezcla dos cosas distintas.

    Ojo: siendo ley de potencia con exponente ~10, satura en p=1 a 48,4 m/s
    para un tramo de 1 km (antes si el tramo es más largo), de ahí el clip.
    """
    v = np.asarray(viento_ms, dtype=float)
    alpha = DUNN_ALPHA if alpha is None else alpha        # --dunn-alpha reasigna el global
    l = np.asarray(largo_km, dtype=float)
    return np.minimum(alpha * l * np.maximum(v, 0.0) ** exp, 1.0)


# ════════════════════════════════════════════════════════════
# Curvas de las medidas de hardening
# ════════════════════════════════════════════════════════════
# Los modelos son los mismos que E2 usa hoy para adelgazar la matriz base
# (pipeline.py:336-346 y 421-446); se importan de ahí en vez de copiarlos,
# para que no puedan desincronizarse. Lo que cambia es CUÁNDO se aplican:
# antes se "curaban" fallas ya sorteadas con una probabilidad ρ; ahora la
# medida entra como una curva de fragilidad propia y se sortea con ella.
#
# Los dos caminos coinciden en valor esperado —E2 calcula justamente
# ρ = 1 − p_mitigada/p_base— pero sortear con la curva es preferible por
# dos razones: la mitigada puede fallar en horas en que la base todavía se
# repara (el adelgazamiento no puede representar eso), y no depende de que
# la matriz base sea reproducible.

def theta_refuerzo(theta_base=THETA_DEFAULT):
    """Mediana de la curva desplazada por refuerzo estructural [m/s].

    Darestani et al.: se compara la mediana logarítmica de un poste clase 5
    (el existente) contra uno clase 3 (el reforzado); la razón entre ambas
    desplaza la curva hacia vientos más altos.

    `theta_base` DEBE ser la θ de la curva base de la corrida, no la
    constante del módulo: la razón clase 3 / clase 5 es un desplazamiento
    relativo, y anclarlo a THETA_DEFAULT hacía que con `--theta` mayor que
    30 el "refuerzo" quedara por debajo de la base y aumentara las fallas.
    """
    import pipeline as P

    razon = np.exp(P._mu_log_mediana_darestani(P._DARESTANI_MU_C3)
                   - P._mu_log_mediana_darestani(P._DARESTANI_MU_C5))
    return float(theta_base) * float(razon), float(razon)


def factor_vegetacion(viento_ms):
    """Factor multiplicativo sobre P(falla) por gestión de vegetación.

    Modelo de caída de árboles de Canham: la probabilidad de que un árbol
    caiga sobre la línea crece con su DBH y con el viento normalizado. La
    poda lleva el DBH efectivo del corredor de 30 cm a 10 cm; el cociente
    entre ambas probabilidades es lo que queda de riesgo.
    """
    import pipeline as P

    v = np.asarray(viento_ms, dtype=float)
    v_max = float(v.max()) if v.size and v.max() > 0 else 1.0

    def _p_ftr(dbh):
        s = np.clip(v / v_max, 0.0, 1.0)
        return 1 / (1 + np.exp(-(P._CANHAM_A_S + P._CANHAM_C_S * s * dbh ** P._CANHAM_B_S)))

    antes = _p_ftr(P.DBH_ANTES_PODA_CM)
    despues = _p_ftr(P.DBH_DESPUES_PODA_CM)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(antes > 1e-12, despues / np.maximum(antes, 1e-12), 1.0)
    return np.clip(factor, 0.0, 1.0)


def curvas_mitigadas(wind_zonas, col_zona, theta, beta, curva="lognormal",
                     largo_km=None, veg_rho=None):
    """P(falla) por hora y línea para la red base y cada medida.

    Devuelve {"base", "veg", "ref", "ug"} con arreglos (horas, n_líneas).

    `curva` elige la física de la línea base:
      - "lognormal": estructural, Φ(ln(v/θ)/β). Necesita θ y β.
      - "dunn":      empírica, α·l·v^exp. Necesita `largo_km`.

    En ambos casos el refuerzo es el MISMO desplazamiento relativo de
    Darestani (poste clase 5 → clase 3): la curva se evalúa en v/razón. Bajo
    la lognormal eso equivale a mover θ; bajo la ley de potencia, a dividir
    la probabilidad por razón^exp. Es la misma medida física expresada en
    las dos parametrizaciones, no dos supuestos distintos.

    `veg_rho`: si se da, la poda reduce la probabilidad de falla en una
    fracción FIJA (p_veg = p_base·(1 − veg_rho)), como el soterramiento, en
    vez del factor de Canham que depende del viento. Sin él, Canham.
    """
    import pipeline as P

    viento_linea = wind_zonas[:, col_zona]              # (horas, n_líneas)
    _, razon = theta_refuerzo(theta)

    if curva == "dunn":
        if largo_km is None:
            raise ValueError("la curva 'dunn' necesita el largo de cada línea")
        pf_base = p_falla_dunn(viento_linea, largo_km)
        pf_ref = p_falla_dunn(viento_linea / razon, largo_km)
        meta_curva = {"curva": "dunn", "alpha": DUNN_ALPHA, "exp": DUNN_EXP,
                      "factor_ref": float(razon ** -DUNN_EXP)}
    else:
        pf_base = p_falla_lognormal(viento_linea, theta, beta)
        # El refuerzo desplaza la mediana, no la dispersión: β es el mismo de
        # la curva base. `P.BETA_REF` valía justo BETA_DEFAULT, así que la
        # distinción nunca se notó mientras no se pasara `--beta`.
        pf_ref = p_falla_lognormal(viento_linea, theta * razon, beta)
        meta_curva = {"curva": "lognormal", "theta": theta, "beta": beta,
                      "theta_ref": theta * razon, "beta_ref": beta}

    if veg_rho is None:
        pf_veg = pf_base * factor_vegetacion(wind_zonas)[:, col_zona]
    else:
        pf_veg = pf_base * (1.0 - veg_rho)
    pf_ug = pf_base * (1.0 - P.RHO_UG)

    meta_curva.update({"razon_clase3_clase5": razon, "rho_ug": P.RHO_UG,
                       "veg_modelo": "canham" if veg_rho is None else "fijo",
                       "veg_rho": veg_rho,
                       "dbh_antes_cm": P.DBH_ANTES_PODA_CM,
                       "dbh_despues_cm": P.DBH_DESPUES_PODA_CM})
    return {"base": pf_base, "veg": pf_veg, "ref": pf_ref, "ug": pf_ug}, meta_curva


def simular(pf_por_variante, num_esc, horas_espera, semilla=None):
    """Monte Carlo de fallas de línea, hora a hora, para varias variantes
    de la red a la vez.

    pf_por_variante : {nombre: (horas, n_lineas)} probabilidad de falla de
                      cada línea en cada hora bajo esa variante (base, con
                      poda, con refuerzo, soterrada...).

    Dos cosas que este bucle hace y que importan:

    **Reparación.** Una línea que falla queda fuera `horas_espera` horas y
    no puede volver a fallar mientras se repara — el contador `restline`
    del notebook. Es la única memoria del proceso: sin ella, una tormenta
    de 10 h generaría 10 fallas seguidas en la misma línea.

    **Números aleatorios comunes.** El sorteo uniforme se hace UNA vez por
    hora y lo usan todas las variantes. Así la comparación entre "sin
    medida" y "con medida" es pareada: la medida se evalúa contra la misma
    secuencia de ráfagas, no contra otro año simulado. Como la curva
    mitigada está siempre por debajo de la base, toda falla mitigada es
    también falla base en el mismo sorteo — la misma monotonía que
    garantizaba el adelgazamiento de Bernoulli de E2, pero derivada de la
    física en vez de un ρ empírico.

    Las trayectorias sí se separan después, y debe ser así: una línea que
    con poda no falló está disponible antes que su gemela sin poda, y puede
    fallar en una hora en que la otra todavía se repara.
    """
    nombres = list(pf_por_variante)
    n_horas, n_lin = pf_por_variante[nombres[0]].shape
    rng = np.random.default_rng(semilla)

    fallas = {v: np.zeros((n_horas, num_esc, n_lin), dtype=np.int8) for v in nombres}
    restline = {v: np.zeros((num_esc, n_lin), dtype=np.int16) for v in nombres}

    # Unión de columnas con probabilidad no despreciable en alguna variante:
    # define de qué tamaño se sortea, para que el consumo del generador sea
    # idéntico corra una variante o cuatro.
    activas_h = np.zeros((n_horas, n_lin), dtype=bool)
    for pf in pf_por_variante.values():
        activas_h |= pf > 1e-9

    for h in range(n_horas):
        for v in nombres:
            r = restline[v]
            r[r > 0] -= 1

        activas = activas_h[h]
        if not np.any(activas):
            continue

        sorteo = rng.random((num_esc, int(activas.sum())))   # común a todas
        columnas = np.flatnonzero(activas)

        for v in nombres:
            pf_act = pf_por_variante[v][h][activas]
            disponibles = restline[v][:, activas] == 0
            falla_act = disponibles & (sorteo < pf_act[np.newaxis, :])
            if not np.any(falla_act):
                continue
            fallas[v][h][:, activas] = falla_act.astype(np.int8)
            for j, col in enumerate(columnas):
                restline[v][falla_act[:, j], col] = horas_espera

    return fallas


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def mostrar_estado(args):
    print("=" * 62)
    print("INSUMOS PARA LA MATRIZ DE FALLAS")
    print("=" * 62)
    for etiqueta, ruta in (("red (E0)", ART_DIR / "red.pkl"),
                           ("viento (.nc)", Path(args.viento))):
        marca = "x" if ruta.exists() else " "
        print(f"  [{marca}] {etiqueta:<16} {ruta}")
    raiz = (DATOS_DIR / args.salida).with_suffix("")
    for etiqueta, ruta in [("base", DATOS_DIR / args.salida)] + [
            (v, Path(f"{raiz}_{v}.npy")) for v in ("veg", "ref", "ug")]:
        if ruta.exists():
            a = np.load(ruta, mmap_mode="r")
            print(f"  [x] {etiqueta:<16} {ruta.name:<24} shape={a.shape} "
                  f"fallas={int(np.asarray(a).sum())}")
        else:
            print(f"  [ ] {etiqueta:<16} {ruta.name} (no existe)")
    meta = ART_DIR / "faultlin_meta.json"
    if meta.exists():
        print("\n  Última corrida registrada:")
        print("   ", json.dumps(json.loads(meta.read_text(encoding="utf-8")),
                                indent=2, ensure_ascii=False).replace("\n", "\n    "))
    print("=" * 62)


def main():
    global DUNN_ALPHA                  # --dunn-alpha lo reasigna
    ap = argparse.ArgumentParser(
        description="Matriz de fallas de línea por Monte Carlo sobre el viento del .nc de EWE")
    ap.add_argument("--salida", default="FaultlinMC.npy",
                    help="nombre del .npy a escribir dentro de app/ (default: FaultlinMC.npy)")
    ap.add_argument("--red", default=None,
                    help="ruta a red.pkl o red_nueva.pkl (default: artefactos/red.pkl)")
    ap.add_argument("--viento", default=str(RUTA_WIND_NC),
                    help="NetCDF de eventos extremos (BaseHoraEWEPELP20z.nc)")
    ap.add_argument("--horas", type=int, default=8760)
    ap.add_argument("--escenarios", type=int, default=100)
    ap.add_argument("--anio-offset", dest="offset", type=int, default=ANIO_OFFSET_DEFAULT,
                    help=f"hora inicial dentro del .nc (default: {ANIO_OFFSET_DEFAULT} = "
                         f"segundo año, 2030)")
    ap.add_argument("--escenario-clima", default="baseline", choices=["baseline", "with_EWE"],
                    help="eje lon del .nc: baseline (histórico) o with_EWE (con ráfaga)")
    ap.add_argument("--enlaces-inmunes", dest="enlaces_inmunes", action="store_true",
                    help="anular las fallas de las líneas de enlace "
                         "(`tie_indices` de red.pkl). Se aplica DESPUÉS del "
                         "sorteo, así que el resto de la matriz queda idéntico "
                         "a la misma corrida sin el flag: con la misma semilla "
                         "sólo desaparecen las columnas de enlace.")
    ap.add_argument("--curva", default="lognormal", choices=["lognormal", "dunn"],
                    help="física de la línea base: 'lognormal' (estructural, "
                         "Panteli/Darestani, usa --theta y --beta) o 'dunn' "
                         "(empírica, escala con el largo del tramo e incluye "
                         "caída de árboles)")
    ap.add_argument("--dunn-alpha", dest="dunn_alpha", type=float, default=None,
                    help=f"α de la curva de Dunn (default {DUNN_ALPHA:g}, literatura)")
    ap.add_argument("--factor-zona", dest="factor_zona", default=None,
                    help="factores de viento por zona separados por coma (normalización), ej. 1.08,1.09,0.864")
    ap.add_argument("--sin-ajuste-altura", dest="sin_ajuste_altura", action="store_true",
                    help="no llevar el viento de 10 m a la altura del poste (usar si α se calibró a 10 m)")
    ap.add_argument("--theta", type=float, default=THETA_DEFAULT)
    ap.add_argument("--beta", type=float, default=BETA_DEFAULT)
    ap.add_argument("--horas-espera", type=int, default=HORAS_ESPERA_DEFAULT,
                    help=f"horas que una línea queda fuera tras fallar (default: "
                         f"{HORAS_ESPERA_DEFAULT})")
    ap.add_argument("--semilla", type=int, default=None,
                    help="semilla del sorteo. Sin ella la corrida NO es reproducible "
                         "(así se generó el .npy histórico).")
    ap.add_argument("--sin-mitigadas", dest="mitigadas", action="store_false",
                    help="calcular SOLO la matriz base. Por defecto se calculan "
                         "también las de vegetación, refuerzo y soterramiento, "
                         "sorteadas con su propia curva de fragilidad y con los "
                         "mismos números aleatorios que la base.")
    ap.add_argument("--veg-rho", dest="veg_rho", type=float, default=None,
                    help="poda como reducción FIJA de la probabilidad de falla (ej. 0.33), "
                         "en vez del factor de Canham que depende del viento")
    ap.add_argument("--forzar", action="store_true",
                    help="sobrescribir los .npy si ya existen")
    ap.add_argument("--estado", action="store_true", help="mostrar insumos y salir")
    ap.add_argument("--artefactos", default=None,
                    help="carpeta de artefactos del caso. Mueve de una vez la red "
                         "que se lee, el .npy que se escribe y faultlin_meta.json. "
                         "Sin el flag: $ERAD_ARTEFACTOS y, si no está, artefactos/.")
    args = ap.parse_args()

    # Antes de cualquier lectura, incluido --estado.
    if args.artefactos:
        global ART_DIR
        ART_DIR = casos.resolver_ruta(args.artefactos)
        # La salida por defecto pasa a la carpeta del caso: la matriz del caso de
        # validación vive en app/FaultlinMC.npy por historia, pero repetir eso en
        # cada caso nuevo dejaría todas las matrices juntas en app/, con nombres
        # que no dicen a qué red pertenecen.
        if args.salida == "FaultlinMC.npy":
            args.salida = str(ART_DIR / "FaultlinMC.npy")
    _log(f"caso: {ART_DIR}")

    if args.estado:
        mostrar_estado(args)
        return

    salida = DATOS_DIR / args.salida
    salidas = {"base": salida}
    if args.mitigadas:
        raiz = salida.with_suffix("")
        salidas.update({v: Path(f"{raiz}_{v}.npy") for v in ("veg", "ref", "ug")})
    ramales = cargar_ramales(args.red)
    if ramales is not None:
        salidas["ramales"] = salida.with_name(f"{salida.stem}_ramales.npy")

    existentes = [s.name for s in salidas.values() if s.exists()]
    if existentes and not args.forzar:
        sys.exit(
            f"Ya existe(n): {existentes}. Rehacerlos invalida los baselines de E5 y "
            f"todo el barrido calculado con la matriz actual.\n"
            f"Si es lo que quieres: --forzar (o --salida con otro nombre).")

    LINES, zona_por_linea, largo_km, tie_pos = cargar_red(args.red)
    _log(f"red: {len(LINES)} líneas, zonas {sorted(set(zona_por_linea.tolist()))}, "
         f"{largo_km.sum():.1f} km ({largo_km.min():.2f}–{largo_km.max():.2f} por tramo)")
    if ramales is not None:
        _log(f"ramales con fusible: {len(ramales['id'])}, {ramales['km'].sum():.1f} km, "
             f"zonas {sorted(set(ramales['zona'].tolist()))}")

    wind = cargar_viento(args.viento, args.horas, args.offset, args.escenario_clima)
    # α calibrado localmente (caso El Melón: con la SEC, ver notebook 03). Sin el flag, el
    # de la literatura. La ráfaga calibrada es la de 10 m: --sin-ajuste-altura no la lleva a 12 m.
    if args.dunn_alpha is not None:
        DUNN_ALPHA = float(args.dunn_alpha)
    if args.sin_ajuste_altura:
        wind = wind / ((ALTURA_POSTE_M / ALTURA_MEDICION_M) ** EXP_HELLMANN)
    # Normalización por zona: quita el sesgo permanente del reanálisis entre zonas
    # (cada zona conserva su serie horaria, ver notebook 03 sección 14).
    if args.factor_zona:
        f_z = np.array([float(x) for x in args.factor_zona.split(",")])
        if len(f_z) != wind.shape[1]:
            sys.exit(f"--factor-zona trae {len(f_z)} valores y el .nc tiene {wind.shape[1]} zonas")
        wind = wind * f_z[np.newaxis, :]
        _log(f"factor de viento por zona: {f_z.tolist()}")
    _log(f"viento: {wind.shape} (horas × zonas) desde {Path(args.viento).name} "
         f"[{args.escenario_clima}], máx {wind.max():.2f} m/s a {ALTURA_POSTE_M:.0f} m")

    n_zonas_nc = wind.shape[1]
    if ramales is not None and ramales["zona"].max() > n_zonas_nc:
        sys.exit(f"Los ramales tienen zonas hasta {ramales['zona'].max()} pero el .nc solo trae "
                 f"{n_zonas_nc} puntos.")
    if zona_por_linea.max() > n_zonas_nc:
        sys.exit(
            f"La red tiene zonas hasta {zona_por_linea.max()} pero el .nc solo trae "
            f"{n_zonas_nc} puntos. Regenerar el pipeline climático con un "
            f"SystemzoneV1.xlsx que tenga una fila por zona de la red.")

    # zona 1..Z -> columna 0..Z-1 del arreglo de viento
    col_zona = np.clip(zona_por_linea - 1, 0, n_zonas_nc - 1)
    curvas, params_medidas = curvas_mitigadas(wind, col_zona, args.theta, args.beta,
                                              curva=args.curva, largo_km=largo_km,
                                              veg_rho=args.veg_rho)
    if not args.mitigadas:
        curvas = {"base": curvas["base"]}

    if args.curva == "dunn":
        _log(f"curva: Dunn empírica  p = {DUNN_ALPHA:g}·l[km]·v^{DUNN_EXP} "
             f"(Dunn et al. 2018 vía Shi et al. 2021)")
    else:
        _log(f"curva: lognormal estructural  θ={args.theta:.2f} m/s  β={args.beta:.3f}")
    _log(f"P(falla) máx por línea: base={curvas['base'].max():.4f}"
         + "".join(f" | {v}={curvas[v].max():.4f}" for v in curvas if v != "base"))
    if args.mitigadas:
        razon = params_medidas["razon_clase3_clase5"]
        if args.curva == "dunn":
            _log(f"  refuerzo: p ×{params_medidas['factor_ref']:.3f} "
                 f"(v/{razon:.4f}, poste clase 5 → clase 3)")
        else:
            _log(f"  refuerzo: θ {args.theta:.1f} → {params_medidas['theta_ref']:.2f} m/s "
                 f"(×{razon:.4f}, poste clase 5 → clase 3)")

    _log(f"simulando {args.horas} h × {args.escenarios} escenarios × {len(LINES)} "
         f"líneas × {len(curvas)} variante(s)...")
    fallas = simular(curvas, args.escenarios, args.horas_espera, args.semilla)

    # Enlaces inmunes: se anulan DESPUÉS del sorteo, nunca antes. Sacarlos de
    # las columnas activas cambiaría cuántos números consume el generador por
    # hora y desplazaría toda la matriz; anularlos después deja el resto bit a
    # bit igual a la corrida sin el flag.
    #
    # El motivo no es que un enlace no pueda romperse —es conductor aéreo como
    # cualquier otro— sino que en el caso base está normalmente abierto y con
    # `z_tie` fijo en 0: su falla no puede producir ENS, así que sólo infla la
    # cuenta de fallas y ensucia la correlación entre fallas y consecuencia.
    if args.enlaces_inmunes:
        if not tie_pos:
            _log("  [!] --enlaces-inmunes pedido pero red.pkl no declara "
                 "tie_indices; no se anula nada")
        else:
            quitadas = 0
            for arr in fallas.values():
                quitadas += int(arr[:, :, tie_pos].sum())
                arr[:, :, tie_pos] = 0
            lineas_tie = [LINES[i] for i in tie_pos]
            _log(f"  enlaces inmunes: líneas {lineas_tie} anuladas "
                 f"({quitadas} fallas descartadas en total)")

    # Ramales con fusible: su propio sorteo, con la misma curva que la red base y
    # el viento de su zona. Va aparte (otro generador, semilla + 1) para que la
    # matriz de las líneas sea la misma que sin ramales con la misma semilla.
    # No tienen variantes mitigadas: el MILP no decide medidas sobre ramales.
    if ramales is not None:
        col_r = np.clip(ramales["zona"] - 1, 0, n_zonas_nc - 1)
        viento_r = wind[:, col_r]
        if args.curva == "dunn":
            pf_r = p_falla_dunn(viento_r, ramales["km"])
        else:
            _log("  [!] curva lognormal: la probabilidad de un ramal no depende de su largo")
            pf_r = p_falla_lognormal(viento_r, args.theta, args.beta)
        sem_r = None if args.semilla is None else args.semilla + 1
        fallas["ramales"] = simular({"ramales": pf_r}, args.escenarios, args.horas_espera, sem_r)["ramales"]

    totales = {}
    for variante, arr in fallas.items():
        salidas[variante].parent.mkdir(parents=True, exist_ok=True)
        np.save(salidas[variante], arr)
        totales[variante] = int(arr.sum())
        _log(f"→ {salidas[variante].name}  shape={arr.shape}  "
             f"fallas={totales[variante]} ({totales[variante]/args.escenarios:.1f} "
             f"por escenario)")

    if args.mitigadas:
        base = max(totales["base"], 1)
        _log("  reducción de fallas vs. base: "
             + ", ".join(f"{v} {1 - totales[v]/base:.0%}"
                         for v in ("veg", "ref", "ug") if v in totales))

    ART_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "salidas": {v: s.name for v, s in salidas.items() if v in fallas},
        "shape": list(fallas["base"].shape),
        "fallas_totales": totales,
        "curva": args.curva,
        # θ y β sólo describen la corrida si la curva es la lognormal; con
        # 'dunn' se declaran los parámetros de la ley de potencia y se deja
        # explícito que los otros no se usaron.
        **({"theta_ms": args.theta, "beta": args.beta}
           if args.curva == "lognormal"
           else {"dunn_alpha": DUNN_ALPHA, "dunn_exp": DUNN_EXP,
                 "theta_ms": None, "beta": None,
                 "largo_km_por_linea": largo_km.round(4).tolist()}),
        "enlaces_inmunes": bool(args.enlaces_inmunes),
        "factor_zona": args.factor_zona, "ajuste_altura": not args.sin_ajuste_altura,
        "lineas_enlace": [LINES[i] for i in tie_pos],
        "largo": LARGO_FUENTE,
        "ramales": (None if ramales is None else
                    {"n": len(ramales["id"]), "km": float(ramales["km"].sum()),
                     "semilla": None if args.semilla is None else args.semilla + 1}),
        "altura_poste_m": ALTURA_POSTE_M,
        "horas_espera": args.horas_espera,
        "semilla": args.semilla,
        "viento_nc": str(args.viento),
        "escenario_clima": args.escenario_clima,
        "anio_offset": args.offset,
        "zonas_por_linea": zona_por_linea.tolist(),
        "mitigadas": {
            "metodo": ("curva_directa_numeros_comunes" if args.mitigadas else None),
            **(params_medidas if args.mitigadas else {}),
        },
    }
    (ART_DIR / "faultlin_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    _log("→ artefactos/faultlin_meta.json")
    _log("siguiente paso:  python pipeline.py --etapas E2 --forzar"
         + ("  (tomará las mitigadas de los .npy, sin adelgazar)"
            if args.mitigadas else ""))


if __name__ == "__main__":
    main()