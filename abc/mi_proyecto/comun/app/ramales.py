"""
ramales.py — Ramales con fusible: fallas que cortan solo la carga del ramal.

En la red reducida los ramales con fusible se podan: su carga queda sumada en
la barra del troncal de la que cuelgan y sus km, en el largo expuesto de la
línea. Sin este módulo, una falla dentro del ramal se modela como falla de la
línea y deja fuera la sección completa, cuando en la operación real se funde
el fusible y cae SOLO el ramal (validado con la SEC y la red completa de PF en
`caso_elmelon/datos/01_red_elmelon.ipynb`).

Aquí cada ramal es un elemento de falla propio, con su Monte Carlo
(`construir_faultlin.py`) y sus horas de reparación (E2). Mientras un ramal
está fundido, su carga es un corte forzado: el optimizador recibe un piso para
`P_LC[t, b]` igual a la fracción de la demanda de la barra que está detrás del
fusible. No agrega barras, líneas ni variables al MILP, y sin ramales el piso
es 0, así que el modelo queda idéntico al de antes.

Formato (clave "ramales_fusible" del paquete y de red.pkl):
    {"id":    [nombre del fusible, ...],
     "km":    [km aéreos propios del ramal, ...],
     "zona":  [zona climática 1..Z, ...],
     "carga": [(i_ramal, bus, fracción de P_D_base[bus]), ...],
     "padre": [índice del ramal que lo contiene, o -1, ...]}   (opcional)

La carga de un ramal incluye la de los ramales anidados dentro de él (si se
funde el de afuera, cae todo). Con "padre", un ramal anidado no suma mientras su
ramal contenedor esté fundido, para no contar dos veces la misma carga.

Lo usan `pipeline.py` (E0, E2, construir_data), `data_loader.py`,
`optimization_base.py` y `optimization_6_oficial.py`, para que todos lean lo
mismo.
"""

import numpy as np

CLAVE_RED = "ramales_fusible"
CLAVES_DATA = ("faultlinMC_ramales", "repairMC_ramales", "ramales_carga", "ramales_padre")


def validar(ramales, BUSES=None):
    """Revisa el formato y devuelve el número de ramales. Lanza ValueError."""
    if not ramales:
        return 0
    for k in ("id", "km", "zona", "carga"):
        if k not in ramales:
            raise ValueError(f"ramales_fusible sin la clave '{k}'")
    n = len(ramales["id"])
    if len(ramales["km"]) != n or len(ramales["zona"]) != n:
        raise ValueError("ramales_fusible: 'id', 'km' y 'zona' deben tener el mismo largo")
    buses = set(BUSES) if BUSES is not None else None
    for i, b, fr in ramales["carga"]:
        if not 0 <= int(i) < n:
            raise ValueError(f"ramales_fusible: carga con ramal {i} fuera de rango")
        if buses is not None and int(b) not in buses:
            raise ValueError(f"ramales_fusible: carga en la barra {b}, que no está en la red")
        if not 0.0 <= float(fr) <= 1.0 + 1e-9:
            raise ValueError(f"ramales_fusible: fracción {fr} fuera de [0, 1] (ramal {i}, barra {b})")
    padre = ramales.get("padre")
    if padre is not None:
        if len(padre) != n or any(not -1 <= int(p) < n or int(p) == i for i, p in enumerate(padre)):
            raise ValueError("ramales_fusible: 'padre' debe tener un índice válido (o -1) por ramal")
    return n


def matriz_reparacion(faultlin, horas_reparacion):
    """Igual que el `repairMC` de las líneas en E2: marca la hora de reposición."""
    rep = np.zeros_like(faultlin)
    n_horas = faultlin.shape[0]
    for h in range(n_horas - horas_reparacion):
        rep[h + horas_reparacion] = np.maximum(rep[h + horas_reparacion], faultlin[h])
    return rep


def datos_para_optimizador(red, fallas, sin_fallas=False):
    """Claves de `data` para los optimizadores, o {} si el caso no tiene ramales.

    `fallas` es el contenido de fallas.npz (o un dict con las mismas claves).
    Con `sin_fallas` las matrices van en cero, igual que las de las líneas.
    """
    ramales = red.get(CLAVE_RED) if red else None
    if not ramales or "faultlinMC_ramales" not in fallas:
        return {}
    f = fallas["faultlinMC_ramales"]
    r = fallas["repairMC_ramales"]
    return {
        "faultlinMC_ramales": np.zeros_like(f) if sin_fallas else f,
        "repairMC_ramales": np.zeros_like(r) if sin_fallas else r,
        "ramales_carga": [(int(i), int(b), float(fr)) for i, b, fr in ramales["carga"]],
        "ramales_padre": [int(p) for p in ramales.get("padre") or [-1] * len(ramales["id"])],
    }


def piso_corte(data, esc, horas):
    """{(t, bus): fracción de la demanda que está detrás de fusibles fundidos}.

    El daño se acumula igual que el de las líneas en los optimizadores: un ramal
    que falla queda fundido hasta su hora de reposición en `repairMC_ramales`.
    La fracción se topa en 1: varios ramales de la misma barra fundidos a la
    vez no pueden exigir cortar más que su demanda (sería infactible).
    Devuelve {} si no hay ramales, y entonces el piso es 0 en todas partes.
    """
    f = data.get("faultlinMC_ramales")
    carga = data.get("ramales_carga")
    if f is None or not carga:
        return {}
    rep = data["repairMC_ramales"]
    padre = data.get("ramales_padre")
    # ancestros[i] = ramales que contienen a i (de adentro hacia afuera)
    ancestros = {}
    if padre:
        for i in range(len(padre)):
            cadena, p = [], int(padre[i])
            while p >= 0 and p not in cadena:
                cadena.append(p); p = int(padre[p])
            if cadena:
                ancestros[i] = cadena
    por_ramal = {}
    for i, b, fr in carga:
        por_ramal.setdefault(int(i), []).append((int(b), float(fr)))
    acum = np.zeros(f.shape[2], dtype=bool)
    piso = {}
    for t in horas:
        acum[rep[t, esc, :] == 1] = False
        acum |= f[t, esc, :] == 1
        if not acum.any():
            continue
        por_bus = {}
        for i in np.flatnonzero(acum):
            if any(acum[a] for a in ancestros.get(int(i), ())):
                continue                       # ya cuenta dentro del ramal que lo contiene
            for b, fr in por_ramal.get(int(i), ()):
                por_bus[b] = por_bus.get(b, 0.0) + fr
        for b, v in por_bus.items():
            if v > 0:
                piso[t, b] = min(1.0, v)
    return piso
