#!/usr/bin/env python3
"""Muestra estratificada de escenarios Monte Carlo para el barrido.

Automatiza lo que hasta ahora se hacía a mano y quedaba documentado sólo en
prosa (`GUIA_BARRIDO.md` §"Por qué esos tres parámetros"): ordenar los N
escenarios por severidad del caso base, partirlos en estratos y tomar el del
centro de cada uno.

    python seleccionar_escenarios.py
    python seleccionar_escenarios.py --seccionadores 0,2,9,10
    python seleccionar_escenarios.py --criterio saidi --n-estratos 5

POR QUÉ ESTRATIFICAR Y NO TOMAR LOS PRIMEROS N
----------------------------------------------
El barrido no puede correr las ~15 medidas sobre los 100 escenarios: son
cientos de MILP de 8760 h. Hay que elegir una muestra, y la pregunta es cuál.

Tomar `0-9` funciona sólo si el generador no tiene estructura, y aun así deja
la cobertura al azar: la muestra puede quedar apelotonada en el centro o
sesgada hacia un extremo sin que nada lo avise. El centro de decil garantiza
que cada tramo de severidad esté representado exactamente una vez, y que la
media de la muestra reproduzca la de la población — que es lo que la fase 2
del barrido necesita, porque elige el representativo por cercanía al promedio.

POR QUÉ ENS Y NO CANTIDAD DE FALLAS
------------------------------------
El número de fallas de línea es la severidad de la *amenaza*; la ENS es la
severidad de la *consecuencia*, que es lo que las medidas mejoran. No son lo
mismo: un escenario con muchas fallas en líneas poco críticas puede tener
menos ENS que uno con pocas fallas bien ubicadas. Se estratifica por lo que
se va a medir.

LO QUE ESTA MUESTRA NO CUBRE
-----------------------------
El centro de decil excluye las colas a propósito: el peor y el mejor
escenario nunca entran. Es lo correcto para estimar el promedio, pero
significa que el barrido no dice nada sobre el peor caso. El script imprime
cuáles quedaron fuera para que se corran aparte si se quiere reportar la cola.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import casos

APP_DIR = Path(__file__).resolve().parent
from rutas import DATOS_DIR  # datos del caso activo (ver rutas.py)
ART_DIR = casos.art_dir_activo()      # caso activo, ver casos.py

CRITERIOS = ("ens", "saidi", "fallas")


def _log(msg):
    print(f"[escenarios] {msg}", flush=True)


# ════════════════════════════════════════════════════════════
# Insumos
# ════════════════════════════════════════════════════════════

def carpeta_baselines(seccionadores=None):
    """Carpeta de baselines del supuesto de aislación pedido.

    Se delega en `aislacion.variante` y `pipeline.base_dir` para no duplicar
    la convención de nombres: si cambia allá, este script la sigue.
    """
    from aislacion import variante as _variante
    from pipeline import base_dir

    sec = sorted(seccionadores or [])
    var = _variante(bool(sec), sec)
    return base_dir(var), var


def leer_baselines(carpeta):
    """{escenario: P_LC (horas, buses)} de todos los `esc_*.npz` de la carpeta."""
    archivos = sorted(Path(carpeta).glob("esc_*.npz"))
    if not archivos:
        raise FileNotFoundError(
            f"No hay baselines en '{carpeta}'. Correr:\n"
            f"    python pipeline.py --etapas E5"
            + ("" if "base" in str(carpeta) else " --seccionadores ..."))
    out = {}
    for f in archivos:
        m = re.search(r"esc_(\d+)", f.name)
        if m:
            out[int(m.group(1))] = np.load(f)["P_LC"]
    return out


def demanda_por_hora_bus():
    """P_D[h, b] = P_D_base[b] · demanda[h], igual que `optimization_base:271`.

    Hace falta para el SAIDI: la fracción de clientes interrumpidos es
    P_LC/P_D, y el `.npz` del baseline sólo guarda P_LC.
    """
    import pickle

    with open(ART_DIR / "red.pkl", "rb") as f:
        red = pickle.load(f)
    P_D_base = np.array([float(red["P_D_base"][b]) for b in red["BUSES"]])
    demanda = np.load(ART_DIR / "perfiles.npz")["demanda"]
    return demanda[:, None] * P_D_base[None, :], P_D_base


# ════════════════════════════════════════════════════════════
# Indicadores por escenario
# ════════════════════════════════════════════════════════════

def indicadores(baselines, n_horas=None):
    """ENS, SAIDI y fallas por escenario, desde los baselines ya resueltos.

    El SAIDI reproduce la definición de `indicadores.fn_SAIDI`: la fracción
    equivalente de clientes sin suministro en la barra b a la hora h es
    q = P_LC/P_D, y se pondera por los clientes de la barra. Como los
    clientes se aproximan proporcionales a P_D_base, la constante
    `clientes_por_kW` se cancela en el cociente y no hace falta.
    """
    P_D, P_D_base = demanda_por_hora_bus()
    peso = P_D_base / P_D_base.sum()

    fallas = None
    ruta_f = DATOS_DIR / "FaultlinMC.npy"
    if ruta_f.exists():
        fallas = np.load(ruta_f, mmap_mode="r")

    filas = []
    for esc in sorted(baselines):
        p_lc = baselines[esc]
        h = p_lc.shape[0] if n_horas is None else min(n_horas, p_lc.shape[0])
        p_lc = p_lc[:h]
        pd_ = P_D[:h]

        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(pd_ > 0, p_lc / pd_, 0.0)
        q = np.clip(q, 0.0, 1.0)

        fila = {
            "escenario": esc,
            "ens": float(p_lc.sum()),
            "saidi": float((q * peso[None, :]).sum()),
        }
        if fallas is not None and esc < fallas.shape[1]:
            fila["fallas"] = int(np.asarray(fallas[:h, esc, :]).sum())
        filas.append(fila)
    return filas


# ════════════════════════════════════════════════════════════
# Estratificación
# ════════════════════════════════════════════════════════════

def estratificar(valores, n_estratos):
    """Índices del centro de cada estrato, sobre `valores` ya ordenables.

    El centro del estrato i de n es el percentil (2i+1)/(2n) — es decir 5,
    15, ... 95 para n=10. Se toma el elemento en esa posición del orden, no
    el más cercano al valor del percentil: así cada estrato aporta
    exactamente uno y ninguno se repite.
    """
    orden = np.argsort(valores, kind="stable")
    n = len(orden)
    sel = []
    for i in range(n_estratos):
        pct = (2 * i + 1) / (2 * n_estratos)
        pos = int(round(pct * (n - 1)))
        while pos in sel and pos < n - 1:      # colisión por muestra chica
            pos += 1
        sel.append(pos)
    return [int(orden[p]) for p in sel], [(2 * i + 1) * 100 / (2 * n_estratos)
                                          for i in range(n_estratos)]


def _fmt(x):
    """Decimales según magnitud: la ENS son decenas de miles y el SAIDI es del
    orden de 1, así que un formato fijo aplasta uno de los dos."""
    a = abs(x)
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 1:
        return f"{x:,.2f}"
    return f"{x:,.4f}"


def correlacion(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Muestra estratificada de escenarios para el barrido.")
    ap.add_argument("--criterio", default="ens", choices=CRITERIOS,
                    help="severidad sobre la que se estratifica (default: ens, "
                         "la consecuencia; 'fallas' es la amenaza y 'saidi' la "
                         "duración equivalente)")
    ap.add_argument("--n-estratos", type=int, default=10,
                    help="cuántos escenarios devolver (default: 10, deciles)")
    ap.add_argument("--seccionadores", default=None,
                    help="líneas con seccionador, ej: 0,2,9,10. Elige la "
                         "carpeta de baselines; sin esto usa la variante "
                         "histórica sin aislación.")
    ap.add_argument("--horas", type=int, default=None,
                    help="recortar el horizonte (pruebas)")
    ap.add_argument("--salida", default="escenarios_estratificados.json",
                    help="nombre del JSON dentro de artefactos/")
    ap.add_argument("--json-solo", action="store_true",
                    help="imprimir sólo la lista de escenarios, para pipes")
    args = ap.parse_args()

    sec = ([int(x) for x in args.seccionadores.split(",") if x.strip() != ""]
           if args.seccionadores else [])
    carpeta, var = carpeta_baselines(sec)
    base = leer_baselines(carpeta)
    filas = indicadores(base, args.horas)

    if args.criterio == "fallas" and "fallas" not in filas[0]:
        raise SystemExit("No está FaultlinMC.npy; no se puede estratificar por "
                         "fallas. Usar --criterio ens.")

    v = np.array([f[args.criterio] for f in filas], dtype=float)
    escs = np.array([f["escenario"] for f in filas])
    if args.n_estratos > len(escs):
        raise SystemExit(f"Se pidieron {args.n_estratos} estratos pero sólo hay "
                         f"{len(escs)} baselines en '{carpeta}'.")

    pos, pcts = estratificar(v, args.n_estratos)
    sel = sorted(int(escs[p]) for p in pos)
    lista = ",".join(map(str, sel))

    if args.json_solo:
        print(lista)
        return

    _log(f"baselines: {len(escs)} escenarios en {carpeta}  (variante '{var}')")
    _log(f"criterio: {args.criterio}")
    print()
    print(f"  población: min {_fmt(v.min())}  mediana {_fmt(np.median(v))}  "
          f"max {_fmt(v.max())}  media {_fmt(v.mean())}")
    print()
    print(f"  {'estrato':>7} {'pct':>5} {'esc':>5} {args.criterio:>14}")
    print("  " + "-" * 34)
    for i, (p, pc) in enumerate(zip(pos, pcts), 1):
        print(f"  {i:>7} {pc:>4.0f}% {escs[p]:>5} {_fmt(v[p]):>14}")

    med_m = v[pos].mean()
    sesgo = 100 * (med_m / v.mean() - 1) if v.mean() else 0.0
    print()
    print(f"  media muestra {_fmt(med_m)}  vs población {_fmt(v.mean())}  "
          f"({sesgo:+.1f} %)")

    # Las dos limitaciones que GUIA_BARRIDO.md §9 pide declarar, calculadas
    # en vez de recordadas.
    if args.criterio != "fallas" and "fallas" in filas[0]:
        r = correlacion(v, [f["fallas"] for f in filas])
        print(f"  correlación {args.criterio}–fallas: {r:.3f}")
    otro = "saidi" if args.criterio == "ens" else "ens"
    if otro in filas[0]:
        r = correlacion(v, [f[otro] for f in filas])
        print(f"  correlación {args.criterio}–{otro}: {r:.3f}"
              + ("   ← la muestra NO queda estratificada en ese otro indicador"
                 if r < 0.98 else ""))

    peor, mejor = int(escs[v.argmax()]), int(escs[v.argmin()])
    print()
    print(f"  colas EXCLUIDAS a propósito: peor esc {peor} ({_fmt(v.max())}), "
          f"mejor esc {mejor} ({_fmt(v.min())})")

    meta = {
        "criterio": args.criterio,
        "n_estratos": args.n_estratos,
        "variante_baselines": var,
        "carpeta_baselines": str(carpeta),
        "n_poblacion": int(len(escs)),
        "escenarios": sel,
        "por_estrato": [
            {"estrato": i, "percentil": pc, "escenario": int(escs[p]),
             args.criterio: float(v[p])}
            for i, (p, pc) in enumerate(zip(pos, pcts), 1)],
        "media_muestra": float(med_m),
        "media_poblacion": float(v.mean()),
        "sesgo_pct": float(sesgo),
        "cola_excluida": {"peor": peor, "mejor": mejor},
        "indicadores_por_escenario": filas,
    }
    (ART_DIR / args.salida).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    _log(f"→ artefactos/{args.salida}")
    print()
    print(f"  --escenarios {lista}")


if __name__ == "__main__":
    main()
