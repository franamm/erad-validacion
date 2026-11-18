"""
grafo_layout_pp.py
Coordenadas originales de la red CIGRE MV (pandapower)
y polígonos por zona para el grafo de app_fixed_new.py
"""
import numpy as np

# ── Coordenadas originales de net.bus['geo'] en create_cigre_network_mv ──
BUSCOORDS = {
    0:  (7.0,  16.0),   # Bus HV (subestación)
    1:  (4.0,  15.0),
    2:  (4.0,  13.0),
    3:  (4.0,  11.0),
    4:  (2.5,   9.0),
    5:  (1.0,   7.0),
    6:  (1.0,   3.0),
    7:  (8.0,   3.0),
    8:  (8.0,   8.0),
    9:  (6.0,   5.0),
    10: (4.0,   5.0),
    11: (4.0,   7.0),
    12: (10.0,  15.0),
    13: (10.0,  11.0),
    14: (10.0,   8.0),
}

# def _zona_polygon(bus_list, pad_x=0.75, pad_y=0.75):
#     """Polígono octagonal que engloba los buses de una zona."""
#     pts  = np.array([BUSCOORDS[b] for b in bus_list], dtype=float)
#     xmin, ymin = pts.min(axis=0)
#     xmax, ymax = pts.max(axis=0)

#     if xmax - xmin < 1.0:          # buses colineales (ej. Zona 3)
#         cx = (xmin + xmax) / 2
#         xmin, xmax = cx - pad_x, cx + pad_x
#     else:
#         xmin -= pad_x;  xmax += pad_x

#     ymin -= pad_y;  ymax += pad_y

#     r  = min(0.45, (xmax - xmin) * 0.12, (ymax - ymin) * 0.06)
#     xs = [xmin+r, xmax-r, xmax,   xmax,   xmax-r, xmin+r, xmin,   xmin,   xmin+r]
#     ys = [ymin,   ymin,   ymin+r, ymax-r, ymax,   ymax,   ymax-r, ymin+r, ymin  ]
#     return xs, ys

# ORIGINAL
# _ZONA_BUSES = {
#     1: [0, 1, 2, 12],
#     2: [3, 4, 5, 11, 13],
#     3: [6, 7, 8, 9, 10, 14],
# }
_ZONA_BUSES = {
    1: [ 1, 2, 3, 4],
    2: [7, 8, 12, 13, 14],
    3: [5, 6, 9, 10, 11],
}

# ZONE_POLYGONS = {z: _zona_polygon(buses) for z, buses in _ZONA_BUSES.items()}

def _clip_polygon(poly, normal, limit):
    """Recorta un polígono mediante normal·p <= limit."""
    result = []

    for p1, p2 in zip(poly, poly[1:] + poly[:1]):
        v1 = np.dot(normal, p1) - limit
        v2 = np.dot(normal, p2) - limit

        inside1 = v1 <= 1e-9
        inside2 = v2 <= 1e-9

        if inside1:
            result.append(p1)

        if inside1 != inside2:
            t = v1 / (v1 - v2)
            intersection = p1 + t * (p2 - p1)
            result.append(intersection)

    return result


# def build_zone_polygons(zone_buses, pad=0.8):
#     """Genera polígonos Voronoi adyacentes según el centro de cada zona."""

#     # Centro geométrico de los buses de cada zona
#     centers = {
#         z: np.mean(
#             [BUSCOORDS[b] for b in buses],
#             axis=0
#         )
#         for z, buses in zone_buses.items()
#     }

#     # Rectángulo exterior que contiene toda la red
#     coords = np.array(list(BUSCOORDS.values()), dtype=float)

#     xmin, ymin = coords.min(axis=0) - pad
#     xmax, ymax = coords.max(axis=0) + pad

#     boundary = [
#         np.array([xmin, ymin]),
#         np.array([xmax, ymin]),
#         np.array([xmax, ymax]),
#         np.array([xmin, ymax]),
#     ]

#     polygons = {}

#     for zone, center in centers.items():
#         polygon = [p.copy() for p in boundary]

#         # Recortar según la distancia a los demás centros
#         for other_zone, other_center in centers.items():
#             if other_zone == zone:
#                 continue

#             normal = 2 * (other_center - center)
#             limit = (
#                 np.dot(other_center, other_center)
#                 - np.dot(center, center)
#             )

#             polygon = _clip_polygon(
#                 polygon,
#                 normal,
#                 limit
#             )

#         xs = [p[0] for p in polygon]
#         ys = [p[1] for p in polygon]

#         # Cerrar explícitamente el polígono
#         xs.append(xs[0])
#         ys.append(ys[0])

#         polygons[zone] = (xs, ys)

#     return polygons

# ZONE_POLYGONS = build_zone_polygons(_ZONA_BUSES)

def build_zone_map(zone_buses, resolution=300, pad=0.8):
    """Genera un mapa continuo según el bus zonificado más cercano."""
    coords = np.array(list(BUSCOORDS.values()), dtype=float)

    xmin, ymin = coords.min(axis=0) - pad
    xmax, ymax = coords.max(axis=0) + pad

    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    xx, yy = np.meshgrid(x, y)

    zone_map = np.zeros_like(xx)

    buses_zoned = [
        (bus, zone)
        for zone, buses in zone_buses.items()
        for bus in buses
    ]

    min_distance = np.full_like(xx, np.inf, dtype=float)

    for bus, zone in buses_zoned:
        bx, by = BUSCOORDS[bus]
        distance = (xx - bx)**2 + (yy - by)**2

        mask = distance < min_distance
        zone_map[mask] = zone
        min_distance[mask] = distance[mask]

    return x, y, zone_map

ZONE_X, ZONE_Y, ZONE_MAP = build_zone_map(_ZONA_BUSES)


# ── Funciones de compatibilidad con app_fixed_new.py ─────────────────────

def build_bus_coords_like_simple_plotly(net, bus_zone=None, root=0):
    """Retorna BUSCOORDS originales ignorando geodata modificada."""
    return dict(BUSCOORDS)

def build_line_segments(net, buscoords):
    segments = {}
    for l in net.line.index:
        fb = int(net.line.from_bus[l])
        tb = int(net.line.to_bus[l])
        if fb in buscoords and tb in buscoords:
            segments[l] = (buscoords[fb], buscoords[tb])
    return segments

def zone_bands_from_coords(buscoords, bus_zone):
    """Retorna los polígonos precalculados (ignora argumentos)."""
    return dict(ZONE_POLYGONS)

# ── Layout para redes distintas de CIGRE MV ──────────────────────────────
# BUSCOORDS de arriba es un dibujo a mano de las 15 barras de CIGRE. Cualquier
# otra red (p. ej. MV Oberrhein, 179 barras) se dibuja con sus coordenadas
# geográficas de pandapower, reescaladas al mismo marco de la figura.

# Marco en unidades de dibujo que ocupa la red genérica (x0, x1, y0, y1).
_MARCO_GENERICO = (0.0, 11.0, 2.5, 20.5)


def es_red_cigre(net):
    """True si `net` es la CIGRE MV de 15 barras que dibuja BUSCOORDS."""
    return set(int(b) for b in net.bus.index) == set(BUSCOORDS)


def _coords_geo(net):
    """{bus: (x, y)} desde la geodata de pandapower, o {} si no trae.

    pandapower 3.x guarda un GeoJSON en `net.bus.geo`; las 2.x, columnas x/y
    en `net.bus_geodata`. Se aceptan ambas.
    """
    import json
    coords = {}
    if "geo" in net.bus.columns:
        for b, g in net.bus.geo.items():
            if isinstance(g, str) and g:
                try:
                    x, y = json.loads(g)["coordinates"][:2]
                    coords[int(b)] = (float(x), float(y))
                except Exception:
                    pass
    geod = getattr(net, "bus_geodata", None)
    if not coords and geod is not None and len(geod):
        for b, fila in geod.iterrows():
            if np.isfinite(fila.x) and np.isfinite(fila.y):
                coords[int(b)] = (float(fila.x), float(fila.y))
    return coords


def _coords_topologicas(net):
    """Respaldo sin geodata: layout de resortes, determinista."""
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(int(b) for b in net.bus.index)
    for _, f in net.line.iterrows():
        g.add_edge(int(f.from_bus), int(f.to_bus))
    for _, f in net.trafo.iterrows():
        g.add_edge(int(f.hv_bus), int(f.lv_bus))
    pos = nx.spring_layout(g, seed=0)
    return {b: (float(p[0]), float(p[1])) for b, p in pos.items()}


def coords_red_generica(net, marco=_MARCO_GENERICO):
    """Coordenadas de dibujo de cualquier red, dentro de `marco`.

    Conserva la proporción: si la geodata está en grados (lon/lat) la x se
    corrige por cos(latitud) para que la red no salga estirada.
    """
    crudas = _coords_geo(net)
    if len(crudas) < len(net.bus):
        crudas = _coords_topologicas(net)

    pts = np.array(list(crudas.values()), dtype=float)
    if np.all(np.abs(pts[:, 0]) <= 180) and np.all(np.abs(pts[:, 1]) <= 90) \
            and np.ptp(pts, axis=0).max() < 5:
        pts[:, 0] *= np.cos(np.radians(pts[:, 1].mean()))

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    x0, x1, y0, y1 = marco
    escala = min((x1 - x0) / span[0], (y1 - y0) / span[1])
    # Centrado en el marco.
    off_x = x0 + ((x1 - x0) - span[0] * escala) / 2
    off_y = y0 + ((y1 - y0) - span[1] * escala) / 2
    return {
        b: (float(off_x + (p[0] - lo[0]) * escala),
            float(off_y + (p[1] - lo[1]) * escala))
        for b, p in zip(crudas.keys(), pts)
    }


def mapa_zonas(coords, bus_zone, radio, lineas=(), resolution=400):
    """Mapa de zona por celda: la del elemento más cercano, NaN lejos de la red.

    Igual que `build_zone_map`, pero recortado a `radio` alrededor de cada
    bus y de cada línea, para que la mancha de zona siga la forma de los
    alimentadores en vez de llenar todo el rectángulo.

    `lineas` = pares (from_bus, to_bus). Solo cuentan las que tienen las dos
    puntas en la misma zona: un enlace entre zonas no pertenece a ninguna.
    """
    pts = np.array([coords[b] for b in coords], dtype=float)
    xmin, ymin = pts.min(axis=0) - radio
    xmax, ymax = pts.max(axis=0) + radio
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    xx, yy = np.meshgrid(x, y)
    zone_map = np.full_like(xx, np.nan)
    min_d = np.full_like(xx, np.inf)

    # Cada bus es un segmento de largo cero: una sola fórmula para ambos.
    segmentos = [(coords[b], coords[b], bus_zone.get(b, 1)) for b in coords]
    for fb, tb in lineas:
        if fb in coords and tb in coords and bus_zone.get(fb, 1) == bus_zone.get(tb, 1):
            segmentos.append((coords[fb], coords[tb], bus_zone.get(fb, 1)))

    for (ax_, ay_), (bx_, by_), z in segmentos:
        dx, dy = bx_ - ax_, by_ - ay_
        largo2 = dx * dx + dy * dy
        if largo2 > 0:
            t = np.clip(((xx - ax_) * dx + (yy - ay_) * dy) / largo2, 0.0, 1.0)
        else:
            t = 0.0
        d = (xx - ax_ - t * dx) ** 2 + (yy - ay_ - t * dy) ** 2
        m = d < min_d
        zone_map[m] = z
        min_d[m] = d[m]
    zone_map[min_d > radio ** 2] = np.nan
    return x, y, zone_map
