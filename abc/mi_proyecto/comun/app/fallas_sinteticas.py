"""
fallas_sinteticas.py — Matriz de fallas de juguete, para probar el modelo
=========================================================================

⚠️ **NO SIRVE PARA LA TESIS.** Las fallas de un resultado publicable salen de
`construir_faultlin.py`: viento real del `.nc` de eventos extremos, curva de
fragilidad y Monte Carlo. Este módulo sortea tormentas inventadas.

Existe por un problema concreto: para comprobar que el MILP corre sobre una
red nueva hace falta una matriz de fallas de la forma correcta, y armarla por
la cadena buena exige antes los puntos geográficos, la descarga de clima y el
NetCDF. Eso son horas y APIs externas para responder "¿el modelo resuelve esta
red o no?". Con esto se responde en un minuto y se sigue.

Qué se conserva del modelo bueno, porque si no la prueba no prueba nada:

  * **Coherencia espacial y temporal.** Un Bernoulli independiente línea a
    línea repartiría las fallas por toda la red y por todo el año, y una red
    de distribución no falla así. Acá cada evento golpea una o varias zonas
    completas, que es lo que hace que la reconfiguración y los recursos
    móviles tengan algo que resolver.
  * **Enlaces inmunes** (`enlaces_inmunes: true` del estudio) y la ventana de
    reparación que impide que una línea vuelva a fallar mientras se repone.

Y una decisión deliberada: la **primera tormenta va en la hora 6**. Si todas
cayeran al azar en el año, un horizonte corto de prueba (24 h) no tendría
ninguna falla, el LP sería trivial y "corre" no diría nada.

Cómo queda marcado que no está validado, de forma estructural y no como
comentario: se escribe `fallas_meta.json` con `fuente_viento = "sintetico"` y
`no_validado = true`. Es el MISMO archivo que escribe E2, así que el aviso que
la app ya muestra cuando las fallas no vienen del `.nc` real
(`entrada_red_vista._aviso_viento_e2`) aparece solo, en toda la app, sin
código nuevo.

USO
    python fallas_sinteticas.py --artefactos artefactos_prueba
    python fallas_sinteticas.py --artefactos artefactos_prueba --horas 720 --escenarios 5
    python fallas_sinteticas.py --artefactos artefactos_prueba --estado
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import casos

APP_DIR = Path(__file__).resolve().parent
ART_DIR = casos.art_dir_activo()

# Valores por defecto: los que se usaron para la prueba de MV Oberrhein
# (`prueba_oberrhein.ipynb`). Con 181 líneas dan ~110 fallas al año, o sea
# 6,9e-5 por línea·hora, del mismo orden que el 5,5e-5 de la matriz real del
# estudio — la comparación que hace creíble la prueba.
N_TORMENTAS_DEFAULT = 12
P_LINEA_DEFAULT = 0.08
P_MULTIZONA_DEFAULT = 0.30
HORAS_REPARACION_DEFAULT = 4
HORA_PRIMERA_TORMENTA = 6


def _log(msg):
    print(f"[fallas_sint] {msg}", flush=True)


def generar(n_horas, numesc, zona_por_linea, tie_indices, *,
            n_tormentas=N_TORMENTAS_DEFAULT, p_linea=P_LINEA_DEFAULT,
            p_multizona=P_MULTIZONA_DEFAULT,
            horas_reparacion=HORAS_REPARACION_DEFAULT, semilla=7):
    """(faultlinMC, repairMC, bitacora) de forma (n_horas, numesc, n_líneas).

    `zona_por_linea` es un arreglo con la zona de cada línea (la de su bus de
    origen, misma convención que `pipeline` y `construir_faultlin`).

    Cada escenario se sortea con `semilla + s`: son realizaciones
    INDEPENDIENTES del año. Si compartieran el sorteo, E5 resolvería N veces
    el mismo año y el promedio sobre escenarios no significaría nada.
    """
    zona_por_linea = np.asarray(zona_por_linea)
    n_lineas = len(zona_por_linea)
    zonas = sorted(set(int(z) for z in zona_por_linea))
    tie = set(int(l) for l in tie_indices)
    expuestas = np.array([l for l in range(n_lineas) if l not in tie])
    if expuestas.size == 0:
        raise ValueError("Todas las líneas son enlaces: no hay ninguna que pueda fallar.")

    fault = np.zeros((n_horas, numesc, n_lineas), dtype=np.int8)
    bitacora = []

    for s in range(numesc):
        rng = np.random.default_rng(semilla + s)
        # Horas candidatas separadas por la ventana de reparación, para que dos
        # eventos no se solapen sobre la misma línea.
        paso = horas_reparacion + 1
        candidatas = np.arange(horas_reparacion * 3, max(n_horas - horas_reparacion, 1), paso)
        n_extra = max(0, min(n_tormentas - 1, len(candidatas)))
        horas = [HORA_PRIMERA_TORMENTA] if n_horas > HORA_PRIMERA_TORMENTA else []
        if n_extra:
            horas += sorted(int(h) for h in rng.choice(candidatas, size=n_extra, replace=False))

        for h in horas:
            # Cada rama consume azar SOLO cuando hay una elección real que
            # hacer. No es una micro-optimización: con dos zonas, "multizona"
            # es forzosamente las dos, así que sortear un subconjunto gastaría
            # números sin decidir nada y correría el flujo aleatorio. Gracias a
            # esto, con dos zonas la matriz sale idéntica a la de
            # `prueba_oberrhein.ipynb`, que es la referencia con la que se
            # verificó este módulo.
            if len(zonas) > 1 and rng.random() < p_multizona:
                if len(zonas) == 2:
                    golpeadas = set(zonas)
                else:
                    k = int(rng.integers(2, len(zonas) + 1))
                    golpeadas = set(int(z) for z in rng.choice(zonas, size=k, replace=False))
            else:
                golpeadas = {zonas[int(rng.integers(0, len(zonas)))]}
            en_zona = expuestas[np.isin(zona_por_linea[expuestas], list(golpeadas))]
            caidas = en_zona[rng.random(len(en_zona)) < p_linea]
            fault[h, s, caidas] = 1
            bitacora.append({"escenario": s, "hora": int(h),
                             "zonas": sorted(golpeadas), "lineas": int(len(caidas))})

    repair = np.zeros_like(fault)
    for h in range(n_horas - horas_reparacion):
        repair[h + horas_reparacion] = np.maximum(repair[h + horas_reparacion], fault[h])
    return fault, repair, bitacora


def zona_por_linea_de(red):
    """Zona de cada línea = zona de su bus de origen (convención del estudio)."""
    net, LINES = red["net"], red["LINES"]
    bus_zone = red["bus_zone"]
    return np.array([int(bus_zone.get(int(net.line.from_bus[l]), 1)) for l in LINES])


def escribir(art_dir, fault, repair, bitacora, params):
    """`fallas.npz` + `fallas_meta.json` en la carpeta del caso.

    Las cinco claves son exactamente las que lee `pipeline.construir_data`.
    Las tres mitigadas quedan IGUALES a la base: sin curva de fragilidad no
    hay forma de mitigar nada, así que vegetación, refuerzo y soterramiento
    van a dar ahorro cero por construcción. El `fallas_meta.json` lo declara
    para que no se lea como un resultado.
    """
    art_dir = Path(art_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(art_dir / "fallas.npz",
                        faultlinMC=fault, repairMC=repair,
                        faultlinMC_veg=fault, faultlinMC_ref=fault,
                        faultlinMC_ug=fault)
    meta = {
        "generado": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generador": "fallas_sinteticas.py",
        "no_validado": True,
        "advertencia": ("Fallas inventadas, sin viento ni curva de fragilidad. "
                        "Sirven para probar que el modelo corre, no para "
                        "concluir nada sobre resiliencia."),
        "fuente_viento": "sintetico",
        "metodo_mitigadas": "copia_de_la_base_sin_mitigacion",
        "horas_reparacion": int(params["horas_reparacion"]),
        "forma": [int(x) for x in fault.shape],
        "fallas": {"base": int(fault.sum()), "veg": int(fault.sum()),
                   "ref": int(fault.sum()), "ug": int(fault.sum())},
        "parametros": params,
        "eventos": bitacora,
    }
    (art_dir / "fallas_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return art_dir / "fallas.npz"


def mostrar_estado(art_dir):
    art_dir = Path(art_dir)
    print("=" * 62)
    print(f"FALLAS EN {art_dir}")
    print("=" * 62)
    npz = art_dir / "fallas.npz"
    if not npz.exists():
        print("  [ ] fallas.npz (no existe)")
    else:
        z = np.load(npz)
        f = z["faultlinMC"]
        print(f"  [x] fallas.npz  {f.shape} — {int(f.sum())} fallas")
    meta = art_dir / "fallas_meta.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        print(f"      fuente_viento={m.get('fuente_viento')} "
              f"no_validado={m.get('no_validado', False)}")
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--artefactos", default=None,
                    help="carpeta del caso (default: $ERAD_ARTEFACTOS o artefactos/)")
    ap.add_argument("--red", default=None, help="ruta a red.pkl (default: <caso>/red.pkl)")
    ap.add_argument("--horas", type=int, default=None,
                    help="horizonte. Default: el n_horas del caso, o 8760.")
    ap.add_argument("--escenarios", type=int, default=None,
                    help="columnas de la matriz. Default: el numesc de red.pkl.")
    ap.add_argument("--tormentas", type=int, default=N_TORMENTAS_DEFAULT)
    ap.add_argument("--p-linea", dest="p_linea", type=float, default=P_LINEA_DEFAULT,
                    help="probabilidad de falla de cada línea de la zona golpeada")
    ap.add_argument("--p-multizona", dest="p_multizona", type=float,
                    default=P_MULTIZONA_DEFAULT,
                    help="probabilidad de que el evento golpee más de una zona")
    ap.add_argument("--horas-reparacion", dest="horas_reparacion", type=int,
                    default=HORAS_REPARACION_DEFAULT)
    ap.add_argument("--semilla", type=int, default=7)
    ap.add_argument("--forzar", action="store_true", help="sobrescribir fallas.npz")
    ap.add_argument("--estado", action="store_true")
    args = ap.parse_args()

    global ART_DIR
    if args.artefactos:
        ART_DIR = casos.resolver_ruta(args.artefactos)
    _log(f"caso: {ART_DIR}")

    if args.estado:
        mostrar_estado(ART_DIR)
        return

    if casos.es_validacion(ART_DIR):
        sys.exit(
            "Este es el caso de validación de la tesis: sus fallas vienen del "
            "viento real y no se pisan con fallas inventadas.\n"
            "Si de verdad quieres una matriz sintética, créala en otro caso:\n"
            "    python fallas_sinteticas.py --artefactos artefactos_prueba")

    ruta_red = Path(args.red) if args.red else ART_DIR / "red.pkl"
    if not ruta_red.exists():
        sys.exit(f"Falta {ruta_red}. Correr antes:\n"
                 f"    python pipeline.py --artefactos {ART_DIR} --etapas E0")
    with open(ruta_red, "rb") as f:
        red = pickle.load(f)

    destino = ART_DIR / "fallas.npz"
    if destino.exists() and not args.forzar:
        sys.exit(f"Ya existe {destino}. Rehacerla invalida E4/E5 de este caso. "
                 f"Si es lo que quieres: --forzar")

    numesc = args.escenarios or int(red.get("numesc") or 1)
    n_horas = args.horas or int(_n_horas_del_caso(ART_DIR) or 8760)
    zona_por_linea = zona_por_linea_de(red)
    tie = red.get("tie_indices") or []

    _log(f"red: {len(zona_por_linea)} líneas, zonas "
         f"{sorted(set(zona_por_linea.tolist()))}, {len(tie)} enlaces inmunes")
    _log(f"sorteando {args.tormentas} tormentas/año × {numesc} escenario(s) "
         f"sobre {n_horas} h (semilla {args.semilla})")

    fault, repair, bitacora = generar(
        n_horas, numesc, zona_por_linea, tie, n_tormentas=args.tormentas,
        p_linea=args.p_linea, p_multizona=args.p_multizona,
        horas_reparacion=args.horas_reparacion, semilla=args.semilla)

    params = {"n_tormentas": args.tormentas, "p_linea": args.p_linea,
              "p_multizona": args.p_multizona, "semilla": args.semilla,
              "horas_reparacion": args.horas_reparacion,
              "enlaces_inmunes": [int(l) for l in tie]}
    ruta = escribir(ART_DIR, fault, repair, bitacora, params)

    tasa = fault.sum() / fault.size
    _log(f"→ {ruta.name}: {int(fault.sum())} fallas, "
         f"{fault.sum()/numesc/len(zona_por_linea):.2f} por línea·año, "
         f"tasa {tasa:.2e} por línea·hora "
         f"(la matriz real del estudio: 5.5e-05)")
    _log("⚠️ fallas NO validadas: solo para comprobar que el modelo corre.")
    _log(f"siguiente paso:\n"
         f"    python pipeline.py --artefactos {ART_DIR} --etapas E4")


def _n_horas_del_caso(art_dir):
    """Horizonte que declaró el paquete de la pestaña, si existe."""
    meta = casos.estado_caso(art_dir).get("paquete") or {}
    return meta.get("n_horas")


if __name__ == "__main__":
    main()
