"""
grafo.py — Construcción del grafo visual de la red (Plotly).
Exporta: build_fig_grafo(hora, esc_sel, res_hora, fmc_hora)
"""
from turtle import color

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from data_loader import (
    net, BUSES, LINES, bus_zone, P_D_base, faultlinMC, N_HORAS, sw_cand_list,
    tie_indices as _tie_indices
)
from grafo_layout_pp import (
    BUSCOORDS as BUSCOORDS_CIGRE, ZONE_X, ZONE_Y, ZONE_MAP, _ZONA_BUSES,
    es_red_cigre, coords_red_generica, mapa_zonas,
)

# ── Layout según la red cargada ───────────────────────────────────────────────
# CIGRE MV (15 barras) conserva su dibujo a mano: polígonos de zona, nube del
# "Sistema" y transformadores verticales. Cualquier otra red se dibuja con su
# geodata (ver grafo_layout_pp.coords_red_generica): nodos más chicos, sin
# número encima (va en el hover), zonas como manchas alrededor de la red y la
# subestación marcada sobre su propia barra.
ES_CIGRE = es_red_cigre(net)
BUSCOORDS = dict(BUSCOORDS_CIGRE) if ES_CIGRE else coords_red_generica(net)
# Barras de AT de los transformadores: no tienen carga ni líneas y en la
# geodata caen encima de la barra de MT, así que no se dibujan. En CIGRE es
# la 0, que ya se excluía a mano.
_BUSES_OCULTOS = {int(b) for b in net.trafo.hv_bus} | {0}
_BUSES_SUBESTACION = [int(b) for b in net.trafo.lv_bus]
# Rango de los ejes: el marco de CIGRE incluye la nube del "Sistema" arriba.
if ES_CIGRE:
    X_RANGO, Y_RANGO = (-0.5, 11.5), (2.0, 21.5)
else:
    _xs = [c[0] for b, c in BUSCOORDS.items() if b not in _BUSES_OCULTOS]
    _ys = [c[1] for b, c in BUSCOORDS.items() if b not in _BUSES_OCULTOS]
    X_RANGO = (min(_xs) - 0.8, max(_xs) + 0.8)
    Y_RANGO = (min(_ys) - 0.8, max(_ys) + 1.6)   # arriba: etiquetas S/E y zona


def _tam_nodo(pdb, max_pd):
    """Diámetro del nodo en px, creciente con la demanda base de la barra."""
    if ES_CIGRE:
        return 20 + 22 * pdb / max_pd if max_pd > 0 else 28
    return 5 + 5 * pdb / max_pd if max_pd > 0 else 7


# Grosor del borde de los nodos, en px.
BORDE_NODO = 2.5 if ES_CIGRE else 1.0
# Marcadores de estrategia junto al nodo (cuadrados del resumen, rombo por
# hora), en px. En redes grandes van a escala de los nodos chicos.
TAM_RECUADRO = 10 if ES_CIGRE else 5
TAM_ROMBO = 12 if ES_CIGRE else 6


def _texto_nodo(b):
    """Número de la barra sobre el nodo. En redes grandes no cabe: va al hover."""
    if ES_CIGRE:
        return f"{b}"
    return ""

# ── Paletas de color ──────────────────────────────────────────────────────────
ZONE_FILL   = {1: "rgba(59,130,246,0.12)",  2: "rgba(245,158,11,0.12)",   3: "rgba(16,185,129,0.12)"}
ZONE_BORDER = {1: "rgba(29,78,216,0.55)",   2: "rgba(180,83,9,0.55)",     3: "rgba(4,120,87,0.55)"}
ZONE_LABEL  = {1: "#1D4ED8", 2: "#B45309",  3: "#047857"}
ZONE_NAME   = {1: "Zona 1",  2: "Zona 2",   3: "Zona 3"}
NODEFILL    = {1: "#BFDBFE", 2: "#FDE68A",  3: "#A7F3D0"}
NODESTROKE  = {1: "#1D4ED8", 2: "#B45309",  3: "#047857"}
# Codificación visual de las líneas:
#   fault      rojo    sólido    — averiada, requiere cuadrilla
#   seccionada azul    punteada  — TIENE seccionador y se abrió para aislar
#                                  la falla (es la maniobra, la acción)
#   aislada    naranjo sólido    — sana y sin seccionador: perdió suministro
#                                  por arrastre (el daño colateral)
#   tieopen    gris    punteada  — enlace normalmente abierto
#   tieclosed  azul    sólido    — enlace cerrado para reponer
#   ok         verde   sólido    — en servicio
# El punteado marca "abierto por maniobra"; el sólido, "así quedó".
LINE_COLORS = {"ok": "#16A34A", "fault": "#DC2626", "tiefault": "#DC2626", "aislada": "#F59E0B", "seccionada": "#2563EB", "tieclosed": "#2563EB", "tieopen": "#9CA3AF", "tieclosedsw": "#25EBC0", "tieopensw": "#9CA3AF"}


# Medidas de tratamiento por línea (decisiones x_veg/x_ref/x_ug del MILP).
MEDIDA_LINEA_STYLE = {
    "veg": dict(color="#05EF5B", label="🌳 Gestión de vegetación"),
    "ref": dict(color="#57534E", label="🏗️ Refuerzo estructural"),
    "ug":  dict(color="#7C3AED", label="⛏️ Soterramiento"),
}

# Enlaces desde red.pkl (vía data_loader), no "las últimas tres líneas": si la
# red cambia, el dibujo tiene que seguir marcando los mismos enlaces que el
# modelo, no los que queden al final de la tabla.
tie_indices_set = set(_tie_indices)


# ── Geometría compartida (zonas y sistema externo) ────────────────────────────
# Se extrajo del cuerpo de build_fig_grafo para que el grafo resumen dibuje
# exactamente el mismo fondo sin duplicar coordenadas.
_p3 = np.array(BUSCOORDS_CIGRE[3])
_p8 = np.array(BUSCOORDS_CIGRE[8])
TRIPLE_X, TRIPLE_Y = (_p3 + _p8) / 2
TRIPLE_Y = TRIPLE_Y - 1.5
TRIPLE_X = TRIPLE_X + 1

ZONE_POLYGONS = {
    1: (
        [0.25, 7.0, TRIPLE_X, 0.25, 0.25],
        [16.75, 16.75, TRIPLE_Y, TRIPLE_Y, 16.75],
    ),
    2: (
        [7.0, 12.75, 12.75, 7.0, 7.0, 7.0, TRIPLE_X, 7.0],
        [16.75, 16.75, 2.25, 2.25, 5.0, 7.0, TRIPLE_Y, 16.75],
    ),
    3: (
        [0.25, TRIPLE_X, 7.0, 7.0, 7.0, 0.25, 0.25],
        [TRIPLE_Y, TRIPLE_Y, 7.0, 5.0, 2.25, 2.25, TRIPLE_Y],
    ),
}


def _add_zonas_generica(fig):
    """Zonas de una red cualquiera: mancha por zona alrededor de sus barras.

    Cada celda toma la zona de la barra más cercana y se deja vacía lejos de
    la red, así la mancha sigue la forma de los alimentadores sin que las
    zonas se monten una sobre otra.
    """
    coords = {b: c for b, c in BUSCOORDS.items() if b not in _BUSES_OCULTOS}
    lineas = [(int(f.from_bus), int(f.to_bus)) for _, f in net.line.iterrows()]
    x, y, zmap = mapa_zonas(coords, bus_zone, radio=0.28, lineas=lineas)
    zonas = sorted({int(bus_zone.get(b, 1)) for b in coords})
    if not zonas:
        return
    # Escala de colores discreta: un tramo plano por zona.
    zmin, zmax = zonas[0] - 0.5, zonas[-1] + 0.5
    escala = []
    for z in range(zonas[0], zonas[-1] + 1):
        col = ZONE_FILL.get(z, "rgba(156,163,175,0.12)")
        escala += [[(z - 0.5 - zmin) / (zmax - zmin), col],
                   [(z + 0.5 - zmin) / (zmax - zmin), col]]
    fig.add_trace(go.Heatmap(
        x=x, y=y, z=zmap, zmin=zmin, zmax=zmax, colorscale=escala,
        showscale=False, hoverinfo="skip", zsmooth=False,
    ))
    # Nombre de cada zona en la leyenda, no sobre el dibujo: en una red real
    # las zonas se entrelazan y una etiqueta flotante cae encima de la vecina.
    for z in zonas:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=14,
                        color=NODEFILL.get(z, "#E5E7EB"),
                        line=dict(color=NODESTROKE.get(z, "#374151"), width=2)),
            name=ZONE_NAME.get(z, f"Zona {z}"), showlegend=True, hoverinfo="skip",
        ))


def _add_zonas(fig):
    """Polígonos y etiquetas de zona."""
    if not ES_CIGRE:
        _add_zonas_generica(fig)
        return
    for z, (xs, ys) in ZONE_POLYGONS.items():
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself",
            fillcolor=ZONE_FILL[z],
            line=dict(color=ZONE_BORDER[z], width=2, dash="dot"),
            mode="lines", showlegend=False, hoverinfo="skip",
        ))
        cx   = (max(xs) + min(xs)) / 2 - 1.5
        cy_t = max(ys) - 0.35
        if z == 3:
            fig.add_annotation(
                x=cx + 1.5, y=cy_t - 5, text=f"<b>{ZONE_NAME[z]}</b>",
                showarrow=False,
                font=dict(size=13, color=ZONE_LABEL[z], family="Arial"),
                xanchor="center",
            )
        else:
            fig.add_annotation(
                x=cx, y=cy_t, text=f"<b>{ZONE_NAME[z]}</b>",
                showarrow=False,
                font=dict(size=13, color=ZONE_LABEL[z], family="Arial"),
                xanchor="center",
            )


def add_vertical_transformer(fig, x, y_from, y_to, label="Transformador"):
    color = "#334155"

    y_mid = (y_from + y_to) / 2 + 0.725
    gap_low = y_mid - 0.22 - 0.22
    gap_high = y_mid + 0.22 + 0.22 + 0.725

    y_to_line = y_to - 0.04 + 0.18

    fig.add_trace(go.Scatter(
        x=[x, x, None, x, x],
        y=[y_from, gap_low, None, gap_high, y_to_line],
        mode="lines",
        line=dict(color=color, width=2.2),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=[x, x],
        y=[y_mid - 0.12, y_mid + 0.12],
        mode="markers",
        marker=dict(
            symbol="circle-open",
            size=18,
            color=color,
            line=dict(color=color, width=2.4),
        ),
        showlegend=False,
        hovertemplate=f"{label}<extra></extra>",
    ))


def _add_subestaciones_generica(fig, buscoords):
    """Marca cada subestación sobre su barra de MT (redes no CIGRE).

    La nube del "Sistema" con transformadores verticales solo calza con el
    dibujo a mano de CIGRE: en una red geográfica los cruzaría encima.
    """
    for k, b in enumerate(_BUSES_SUBESTACION):
        if b not in buscoords:
            continue
        xb, yb = buscoords[b]
        fig.add_trace(go.Scatter(
            x=[xb], y=[yb], mode="markers",
            marker=dict(symbol="square", size=16, color="#334155",
                        line=dict(color="#0F172A", width=1.5)),
            showlegend=False,
            hovertemplate=f"Subestación {k + 1} (trafo AT/MT) · Bus {b}<extra></extra>",
        ))
        fig.add_annotation(
            x=xb, y=yb, text=f"<b>S/E {k + 1}</b>", showarrow=True,
            arrowhead=0, arrowcolor="#334155", ax=0, ay=-34,
            font=dict(size=12, color="#0F172A", family="Arial"),
            bgcolor="rgba(255,255,255,0.85)",
        )


def add_external_system(fig, buscoords):
    if not ES_CIGRE:
        _add_subestaciones_generica(fig, buscoords)
        return
    x1, y1 = buscoords[1]
    x12, y12 = buscoords[12]

    y_port = 17.00

    path = (
        "M 1.65,18.75 "
        "C 1.55,19.45 1.75,20.15 2.45,20.45 "
        "C 4.55,20.90 6.15,20.95 7.35,20.75 "
        "C 8.75,20.55 10.35,20.35 11.10,19.55 "
        "C 11.75,18.85 11.45,18.05 10.65,17.85 "
        "C 9.65,17.60 8.55,18.00 7.75,18.25 "
        "C 6.85,18.55 6.10,18.55 5.25,18.15 "
        "C 4.40,17.75 2.45,17.75 1.95,18.15 "
        "C 1.70,18.35 1.62,18.55 1.65,18.75 Z"
    )
    fig.add_shape(
        type="path",
        path=path,
        fillcolor="#ffdac1",
        line=dict(color="#f89d48", width=1.8),
        layer="above",
    )

    fig.add_annotation(
        x=6.35, y=19.65,
        text="<b>Sistema</b>",
        showarrow=False,
        font=dict(size=15, color="#000000", family="Arial"),
    )

    add_vertical_transformer(fig, x1, y1, y_port, label="Transformador 1-Sistema")
    add_vertical_transformer(fig, x12, y12, y_port, label="Transformador 12-Sistema")

# tie_indices_set = set(net.line.index) 


def build_fig_grafo(hora_grafo: int, esc_sel: int, res_hora: dict | None = None, fmc_hora: dict | None = None, cfg: dict | None = None, buses_criticos=None):
    """Construye y retorna el fig_grafo para la hora y escenario dados."""
    if fmc_hora is None:
        fmc_hora = {}
    if res_hora is None:
        res_hora = {}
    if cfg is None:
        cfg = {}

    usar_prior_crit = bool(cfg.get("prior_criticos", False))

    if buses_criticos is None:
        buses_criticos = cfg.get("buses_criticos", [])

    BUSES_CRITICOS = set(int(b) for b in buses_criticos)
    P_LC_hora = {}
    for _bus_key, _plc in (res_hora.get("P_LC", {}) or {}).items():
        if isinstance(_bus_key, tuple) and _bus_key:
            _bus_key = _bus_key[-1]
        try:
            P_LC_hora[int(_bus_key)] = float(_plc)
        except (TypeError, ValueError):
            continue
    pb_hora = {}
    for _bus_key, _pb in (res_hora.get("Pb", {}) or {}).items():
        if isinstance(_bus_key, tuple) and _bus_key:
            _bus_key = _bus_key[-1]
        try:
            pb_hora[int(_bus_key)] = float(_pb)
        except (TypeError, ValueError):
            continue
    # `Pb` es la carga efectivamente servida. Usarlo junto con P_LC evita
    # marcar como desconectado un bus que solo tiene un recorte parcial.
    buses_sin_suministro = set()
    for _b, _plc in P_LC_hora.items():
        if _plc <= 0.01:
            continue
        if _b in pb_hora:
            if pb_hora[_b] <= 0.01:
                buses_sin_suministro.add(_b)
        else:
            buses_sin_suministro.add(_b)
    label_activos = res_hora.get("_label", "Caso base")
    hay_estrategia_activa = False


    ZONE_FILL   = {1: "rgba(59,130,246,0.12)",  2: "rgba(245,158,11,0.12)",  3: "rgba(16,185,129,0.12)"}
    ZONE_BORDER = {1: "rgba(29,78,216,0.55)",   2: "rgba(180,83,9,0.55)",    3: "rgba(4,120,87,0.55)"}
    ZONE_LABEL  = {1: "#1D4ED8",                2: "#B45309",                 3: "#047857"}
    ZONE_NAME   = {1: "Zona 1",                 2: "Zona 2",                  3: "Zona 3"}
    NODEFILL    = {1: "#BFDBFE",                2: "#FDE68A",                 3: "#A7F3D0"}
    NODESTROKE  = {1: "#1D4ED8",                2: "#B45309",                 3: "#047857"}
    # Ver el bloque LINE_COLORS de nivel de módulo para la codificación.
    LINE_COLORS = {"ok": "#16A34A", "fault": "#DC2626", "tiefault": "#DC2626", "aislada": "#F59E0B", "seccionada": "#2563EB", "tieclosed": "#2563EB", "tieopen": "#9CA3AF", "tieclosedsw": "#25EBC0", "tieopensw": "#9CA3AF"}

    # Medidas de tratamiento por línea (gestión de vegetación / refuerzo
    # estructural / soterramiento) — decisión x_veg/x_ref/x_ug del MILP,
    # antes no se dibujaban en el grafo.
    MEDIDA_LINEA_STYLE = {
        "veg": dict(color="#05EF5B", label="🌳 Gestión de vegetación"),
        "ref": dict(color="#57534E", label="🏗️ Refuerzo estructural"),
        "ug":  dict(color="#7C3AED", label="⛏️ Soterramiento"),
    }

    # ── Red eléctrica ────────────────────────────────────────
    fig_grafo = go.Figure()
    # 1) Zonas (mismo fondo que el grafo resumen; ver _add_zonas)
    _add_zonas(fig_grafo)

    # 2) Aristas
    z_tie_hr = res_hora.get("z_tie", {}) if res_hora else {}
    z_sw_hr  = res_hora.get("w_sw",{}) if res_hora else {}
    x_veg_hr = res_hora.get("x_veg", {}) if res_hora else {}
    x_ref_hr = res_hora.get("x_ref", {}) if res_hora else {}
    x_ug_hr  = res_hora.get("x_ug", {}) if res_hora else {}

    def _medida_activa_linea(dic, l, l_int):
        """True si la línea (por índice l o l_int) fue tratada con la medida."""
        try:
            if dic.get(l, 0) not in (0, False, None) and float(dic.get(l, 0)) > 0.5:
                return True
        except Exception:
            pass
        try:
            if dic.get(l_int, 0) not in (0, False, None) and float(dic.get(l_int, 0)) > 0.5:
                return True
        except Exception:
            pass
        return False

    medidas_lineas_presentes = set()
    # Enlaces averiados en esta hora: la leyenda sólo muestra la entrada
    # correspondiente si de verdad hay alguno, igual que las de
    # seccionador y aislación.
    hay_enlace_averiado = False

    lineas_danadas_res = set()
    if res_hora:
        lineas_danadas_res = set(res_hora.get("lineas_danadas_indices", []))

    # Normalización para evitar problemas int / np.int64 / str
    fmc_hora_norm = {}
    for k, v in fmc_hora.items():
        try:
            fmc_hora_norm[int(k)] = bool(v)
        except Exception:
            fmc_hora_norm[k] = bool(v)

    lineas_danadas_norm = set()
    for k in lineas_danadas_res:
        try:
            lineas_danadas_norm.add(int(k))
        except Exception:
            lineas_danadas_norm.add(k)

    # Conjuntos de la zona de aislación. Solo vienen con
    # `aislacion_fallas_fija` activo; sin esa opción llegan vacíos y el
    # grafo se ve igual que antes.
    def _norm_set(clave):
        crudo = set(res_hora.get(clave, [])) if res_hora else set()
        salida = set(crudo)
        for k in crudo:
            try:
                salida.add(int(k))
            except Exception:
                pass
        return salida

    def _norm_lista(vals):
        salida = set(vals or [])
        for k in list(salida):
            try:
                salida.add(int(k))
            except Exception:
                pass
        return salida

    # Líneas fuera de servicio sin estar averiadas. Algunos resultados
    # antiguos solo guardan el conjunto total fuera de servicio; derivar de
    # ahí evita dibujar como operativas las líneas sanas que quedaron sin
    # suministro tras abrir un seccionador.
    lineas_aisladas_norm = _norm_set("lineas_aisladas_indices")
    if not lineas_aisladas_norm:
        fuera_norm = _norm_set("lineas_fuera_servicio_indices")
        lineas_aisladas_norm = fuera_norm - lineas_danadas_norm

    # Qué líneas llevan seccionador. Se toma del resultado si viene, y si no
    # del `cfg` de la corrida: así el grafo funciona igual con resultados
    # calculados antes de que el optimizador reportara esta lista.
    lineas_con_sec_norm = (_norm_set("seccionadores")
                           | _norm_lista(cfg.get("seccionadores_existentes")))

    # Las que se ABRIERON para aislar: son las que llevan seccionador y
    # quedaron fuera. Si el resultado ya trae la lista explícita se usa esa
    # (es la autoritativa); si no, se deriva cruzando aisladas ∩ con
    # seccionador, que da lo mismo y no exige recalcular nada.
    lineas_seccionadas_norm = _norm_set("lineas_seccionadas_indices")
    if not lineas_seccionadas_norm:
        lineas_seccionadas_norm = lineas_aisladas_norm & lineas_con_sec_norm
    # El naranjo queda solo para el arrastre: sin seccionador con qué maniobrar.
    lineas_aisladas_norm = lineas_aisladas_norm - lineas_seccionadas_norm

    for l in LINES:
        try:
            l_int = int(l)
            fb = int(net.line.from_bus[l])
            tb = int(net.line.to_bus[l])
        except Exception:
            continue

        if fb not in BUSCOORDS or tb not in BUSCOORDS:
            continue

        x0, y0 = BUSCOORDS[fb]
        x1, y1 = BUSCOORDS[tb]

        is_tie = l in tie_indices_set or l_int in tie_indices_set

        # Falla desde faultlinMC o desde resultados de optimización
        danada = (
            fmc_hora_norm.get(l, False)
            or fmc_hora_norm.get(l_int, False)
            or l in lineas_danadas_res
            or l_int in lineas_danadas_norm
        )

        aislada    = (
            l in lineas_aisladas_norm or l_int in lineas_aisladas_norm
            or (not is_tie and (fb in buses_sin_suministro or tb in buses_sin_suministro))
        )
        seccionada = (l in lineas_seccionadas_norm or l_int in lineas_seccionadas_norm)
        tiene_sec  = (l in lineas_con_sec_norm     or l_int in lineas_con_sec_norm)

        # La falla tiene prioridad visual sobre todo. Pero un ENLACE dañado no
        # es lo mismo que un tramo de alimentador caído: el enlace está
        # normalmente abierto y no transportaba carga, así que su falla no
        # produce clientes sin suministro — sólo lo deja indisponible para
        # reconfigurar. Se pinta rojo (está averiado) y punteado, respetando
        # la convención del archivo: el punteado marca "abierto".
        if danada:
            cat = "tiefault" if is_tie else "fault"
            ww, dash_e = (4.0, "10px,6px") if is_tie else (4.8, "solid")
            hay_enlace_averiado = hay_enlace_averiado or is_tie

        # Lleva seccionador y se abrió para aislar la falla: es la maniobra.
        # Punteada, igual que un enlace abierto, porque el punteado marca
        # "abierto".
        elif seccionada:
            cat = "seccionada"
            ww, dash_e = 4.0, "8px,5px"

        # Sana y sin seccionador: perdió suministro por arrastre. Sólida,
        # porque no se abrió nada — el punteado se reserva para "abierto".
        elif aislada:
            cat = "aislada"
            ww, dash_e = 4.0, "solid"

        elif is_tie:
            # closed = (z_tie_hr.get(l, 0) > 0.5 or z_sw_hr.get(l_int, 0) > 0.5)
            closed = (z_tie_hr.get(l, 0) > 0.5)
            cat = "tieclosed" if closed else "tieopen"
            ww, dash_e = (3.5, "solid") if closed else (3.0, "10px,6px")

        elif (l in sw_cand_list or l_int in sw_cand_list) and (z_sw_hr.get(l, 0) > 0.5 or z_sw_hr.get(l_int, 0) > 0.5):
            closed = (z_sw_hr.get(l, 0) > 0.5 or z_sw_hr.get(l_int, 0) > 0.5)
            cat = "ok" if closed else "tieopensw"
            ww, dash_e = (3.5, "solid") if closed else (3.0, "10px,6px")

        else:
            cat = "ok"
            ww, dash_e = 2.5, "solid"

        # Medida de tratamiento aplicada a esta línea (a lo más una, el
        # MILP impone x_veg + x_ref + x_ug <= 1 por línea candidata).
        medida_linea = None
        if _medida_activa_linea(x_veg_hr, l, l_int):
            medida_linea = "veg"
        elif _medida_activa_linea(x_ref_hr, l, l_int):
            medida_linea = "ref"
        elif _medida_activa_linea(x_ug_hr, l, l_int):
            medida_linea = "ug"

        medida_txt = f"<br>{MEDIDA_LINEA_STYLE[medida_linea]['label']} aplicada" if (medida_linea and cat not in ("fault", "tiefault")) else ""

        ESTADO_TXT = {
            "ok": "en servicio",
            "fault": "averiada (requiere reparación)",
            "seccionada": "seccionador ABIERTO — aísla la falla",
            "aislada": "sana, sin suministro por arrastre de la falla",
            "tieclosed": "enlace cerrado",
            "tieopen": "enlace abierto",
            "tiefault": "enlace averiado — no disponible para reconfigurar",
            "tieclosedsw": "enlace cerrado (switch)",
            "tieopensw": "abierta por maniobra",
        }
        estado_txt = ESTADO_TXT.get(cat, cat)
        if tiene_sec and cat != "seccionada":
            estado_txt += " · con seccionador (cerrado)"

        fig_grafo.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(
                color=LINE_COLORS[cat],
                width=ww,
                dash=dash_e
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>Línea {l}</b>"
                f"<br>Bus origen: {fb}"
                f"<br>Bus destino: {tb}"
                f"<br>Estado: {estado_txt}"
                f"{medida_txt}"
                "<extra></extra>"
            ),
            legendgroup=cat,
        ))

        # Solo dos capas para líneas: la original arriba, y la medida
        # (vegetación/refuerzo/soterramiento) encima si corresponde. La
        # falla mantiene prioridad visual: si está dañada no se dibuja la medida.
        if medida_linea and cat not in ("fault", "tiefault"):
            medidas_lineas_presentes.add(medida_linea)
            estilo = MEDIDA_LINEA_STYLE[medida_linea]
            fig_grafo.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(color=estilo["color"], width=ww + 2.5, dash=dash_e),
                showlegend=False,
                hovertemplate=(
                    f"<b>Línea {l}</b>"
                    f"<br>{estilo['label']} aplicada"
                    "<extra></extra>"
                ),
                legendgroup=f"medida_{medida_linea}",
            ))

    add_external_system(fig_grafo, BUSCOORDS)
    buses_a_dibujar = [b for b in BUSCOORDS.keys() if b not in _BUSES_OCULTOS]
    # 3) Nodos
    max_pd = max(P_D_base.values()) if P_D_base else 1
    def _valor_dict_bus(dic, b, tol=0.01):
        """
        Devuelve el valor asociado al bus b cuando el diccionario usa:
        - clave simple: b
        - clave tupla: (..., b)
        """
        vals = []

        if not isinstance(dic, dict):
            return vals

        for k, v in dic.items():
            try:
                v = float(v)
            except Exception:
                continue

            if abs(v) <= tol:
                continue

            if k == b:
                vals.append((k, v))
            elif isinstance(k, tuple) and len(k) >= 1 and k[-1] == b:
                vals.append((k, v))

        return vals


    def _gen_parts_bus(res_hora, b):
        """
        Lee las variables de estrategia presentes en res_hora
        y arma textos para el hover del bus.
        """
        gen_parts = []

        # DG Solar
        for _, v in _valor_dict_bus(res_hora.get("P_DG_sol", {}), b):
            gen_parts.append(f"☀ DG solar: {v:.1f} kW")

        # DG Eólica
        for _, v in _valor_dict_bus(res_hora.get("P_DG_eol", {}), b):
            gen_parts.append(f"💨 DG eólica: {v:.1f} kW")

        # DG Fósil
        for _, v in _valor_dict_bus(res_hora.get("P_DG_fos", {}), b):
            gen_parts.append(f"⛽ DG fósil: {v:.1f} kW")

        # ESS fijo por duración
        for dur in ["5h", "4h", "6h"]:
            for _, v in _valor_dict_bus(res_hora.get(f"P_ESS_{dur}", {}), b):
                if v > 0:
                    gen_parts.append(f"🔋 ESS descarga: {v:.1f} kW ({dur})")
                else:
                    gen_parts.append(f"🔋 ESS carga: {abs(v):.1f} kW ({dur})")

        # MEG móvil
        for k, v in _valor_dict_bus(res_hora.get("P_MEG", {}), b):
            if isinstance(k, tuple):
                unidad = k[0]
                gen_parts.append(f"🚛 MEG {unidad}: {v:.1f} kW")
            else:
                gen_parts.append(f"🚛 MEG: {v:.1f} kW")

        # MESS móvil: dejo varios nombres posibles por si el optimizador usa otra clave
        for key_mess in ["P_mess"]:
            for k, v in _valor_dict_bus(res_hora.get(key_mess, {}), b):
                if isinstance(k, tuple):
                    unidad = k[0]
                    gen_parts.append(f"🚚 MESS {unidad+1}: {v:.1f} kW")
                else:
                    gen_parts.append(f"🚚 MESS: {v:.1f} kW")

        # EDRP / carga interrumpible
        for key_edrp in ["P_EDRP", "P_DR", "P_curt"]:
            for k, v in _valor_dict_bus(res_hora.get(key_edrp, {}), b):
                gen_parts.append(f"📉 EDRP: {v:.1f} kW")

        return gen_parts


    for b in buses_a_dibujar:
        if b not in BUSCOORDS:
            continue

        xb, yb = BUSCOORDS[b]
        zona_b = bus_zone.get(b, 1)
        pdb = P_D_base.get(b, 0)

        carga_no_servida = P_LC_hora.get(b, 0)
        sin_sum = carga_no_servida > 0.01
        is_slack = (b == 0)

        size_b = _tam_nodo(pdb, max_pd)

        es_critico = usar_prior_crit and (b in BUSES_CRITICOS)

        fill_col = (
            "#F97316" if sin_sum
            else "#8B5CF6" if es_critico
            else "#1E3A5F" if is_slack
            else NODEFILL.get(zona_b, "#E5E7EB")
        )

        strk_col = (
            "#7C2D12" if sin_sum
            else "#4C1D95" if es_critico
            else "#F0F9FF" if is_slack
            else NODESTROKE.get(zona_b, "#374151")
        )

        txt_col = "#FFFFFF" if (sin_sum or is_slack or es_critico) else "#1F2937"

        gen_parts = _gen_parts_bus(res_hora, b)

        estado_servicio = (
            "<br><b style='color:#DC2626'>⚠ Carga no suministrada</b>"
            if sin_sum
            else "<br>✓ Servido"
        )

        hover_txt = (
            f"<b>Bus {b}</b>" +
            (" · Slack" if is_slack else f" · {ZONE_NAME.get(zona_b, '')}") +
            f"<br>PD base: {pdb:.1f} kW" +
            f"<br>Carga no servida: {carga_no_servida:.2f} kW" +
            ("<br>" + "<br>".join(gen_parts) if gen_parts else estado_servicio)
        )

        fig_grafo.add_trace(go.Scatter(
            x=[xb],
            y=[yb],
            mode="markers+text",
            marker=dict(
                size=size_b,
                color=fill_col,
                line=dict(color=strk_col, width=BORDE_NODO),
                symbol="circle"
            ),
            text=[_texto_nodo(b)],
            textposition="middle center",
            textfont=dict(size=9, color=txt_col, family="Arial Black"),
            showlegend=False,
            hovertemplate=hover_txt + "<extra></extra>",
            name=f"Bus {b}",
        ))

        # Marcador de "estrategia activa" (DG/ESS/MEG/MESS/EDRP) — capa del
        # medio: va entre el nodo base y el duplicado de respaldo.
        if gen_parts:
            hay_estrategia_activa = True
            fig_grafo.add_trace(go.Scatter(
                x=[xb + (0.28 if ES_CIGRE else 0.12)],
                y=[yb + (0.28 if ES_CIGRE else 0.12)],
                mode="markers",
                marker=dict(
                    size=TAM_ROMBO - 2, color="#2563EB",
                    line=dict(color="white", width=2 if ES_CIGRE else 0.5),
                    symbol="diamond"
                ),
                showlegend=False,
                hovertemplate=(
                    f"<b>Bus {b}</b><br>" +
                    "<br>".join(gen_parts) +
                    "<extra></extra>"
                ),
                name="Estrategia activa",
            ))

    # Traza duplicada (respaldo): redibuja cada nodo encima.
    for b in buses_a_dibujar:
        if b not in BUSCOORDS:
            continue
        xb, yb   = BUSCOORDS[b]
        zona_b   = bus_zone.get(b, 1)
        pdb      = P_D_base.get(b, 0)
        sin_sum  = P_LC_hora.get(b, 0) > 0.01
        is_slack = (b == 0)

        size_b   = _tam_nodo(pdb, max_pd)
        es_critico = usar_prior_crit and (b in BUSES_CRITICOS)

        fill_col = (
            "#F97316" if sin_sum
            else "#8B5CF6" if es_critico
            else "#1E3A5F" if is_slack
            else NODEFILL.get(zona_b, "#E5E7EB")
        )

        strk_col = (
            "#7C2D12" if sin_sum
            else "#4C1D95" if es_critico
            else "#F0F9FF" if is_slack
            else NODESTROKE.get(zona_b, "#374151")
        )

        txt_col = "#FFFFFF" if (sin_sum or is_slack or es_critico) else "#1F2937"

        hover_txt = (
            f"<b>Bus {b}</b>" +
            (" · Slack" if is_slack else f" · {ZONE_NAME.get(zona_b, '')}") +
            f"<br>PD base: {pdb:.1f} kW" +
            f"<br>Carga no servida: {P_LC_hora.get(b, 0):.2f} kW" +
            "<br>✓ Servido"
        )

        fig_grafo.add_trace(go.Scatter(
            x=[xb], y=[yb], mode="markers+text",
            marker=dict(size=size_b, color=fill_col,
                        line=dict(color=strk_col, width=BORDE_NODO), symbol="circle"),
            text=[_texto_nodo(b)], textposition="middle center",
            textfont=dict(size=9, color=txt_col, family="Arial Black"),
            showlegend=False,
            hovertemplate=hover_txt + "<extra></extra>",
            name=f"Bus {b}",
        ))

    # ── Diagnóstico temporal ───────────────────────────────────
    # Cuenta cuántos nodos/líneas tienen algo que pintar en esta hora/
    # escenario. Si todo da 0 aquí, el problema es que res_hora no trae
    # los valores (config, hora, o el solver no aplicó la medida) — no
    # es un problema de capas/z-order en el grafo. Borrar cuando se
    # confirme cuál es la causa.
    n_bus_estrategia = sum(1 for b in buses_a_dibujar if _gen_parts_bus(res_hora, b))
    st.caption(
        f"🔍 Debug grafo — hora {hora_grafo}, esc {esc_sel}: "
        f"nodos con estrategia activa: {n_bus_estrategia}/{len(buses_a_dibujar)} · "
        f"líneas con medida aplicada: {len(medidas_lineas_presentes)} "
        f"(tipos presentes: {sorted(medidas_lineas_presentes) or '—'}) · "
        f"claves en res_hora: {sorted(res_hora.keys()) if res_hora else '—'}"
    )

    # 4) Leyenda
    _leyenda = [
        ("Línea operativa",  LINE_COLORS["ok"],       "solid", 3),
        ("Línea en falla",   LINE_COLORS["fault"],     "solid", 3),
    ]
    if lineas_seccionadas_norm:
        _leyenda.append(("Seccionador abierto (aísla la falla)",
                         LINE_COLORS["seccionada"], "8px,5px", 3.4))
    if lineas_aisladas_norm:
        _leyenda.append(("Sin suministro por aislación",
                         LINE_COLORS["aislada"], "solid", 3.4))
    _leyenda += [
        ("Línea de enlace cerrada", LINE_COLORS["tieclosed"], "solid", 2.5),
        ("Línea de enlace abierta", LINE_COLORS["tieopen"],   "dot",   2),
    ]
    if hay_enlace_averiado:
        _leyenda.append(("Enlace averiado (no reconfigurable)",
                         LINE_COLORS["tiefault"], "10px,6px", 3.4))
    for lbl, col, dsh, ww in _leyenda:
        fig_grafo.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=col, width=ww, dash=dsh),
            name=lbl, showlegend=True,
        ))
    fig_grafo.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color="#F97316", line=dict(color="#7C2D12", width=2)),
        name="Nodo sin suministro", showlegend=True,
    ))
    label_activos    = "Caso base"
    if usar_prior_crit and BUSES_CRITICOS:
        fig_grafo.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=12,
                color="#8B5CF6",
                line=dict(color="#4C1D95", width=2)
            ),
            name="Nodo crítico",
            showlegend=True,
        ))
    if hay_estrategia_activa:
        fig_grafo.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                size=10,
                color="#2563EB",   # o "#D97706" si quieres que coincida con el rombo del bus
                line=dict(color="white", width=1.5),
                symbol="diamond"
            ),
            name="Estrategia activa",
            showlegend=True,
            hoverinfo="skip",
        ))

    # Medidas de tratamiento por línea (vegetación / refuerzo / soterramiento)
    for medida_key in ("veg", "ref", "ug"):
        if medida_key not in medidas_lineas_presentes:
            continue
        estilo = MEDIDA_LINEA_STYLE[medida_key]
        fig_grafo.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=estilo["color"], width=5, dash="solid"),
            name=estilo["label"],
            showlegend=True,
            hoverinfo="skip",
        ))


    # 5) Layout
    fig_grafo.update_layout(
        # title=dict(
        #     text=f"<b>Red CIGRE MV · Hora {hora_grafo}</b>  <span style='font-size:13px;color:#6B7280'>{label_activos}</span>",
        #     font=dict(size=16, family="Arial"), x=0.01, xanchor="left",
        # ),
        xaxis=dict(visible=False, range=list(X_RANGO), fixedrange=False),
        yaxis=dict(visible=False, range=[1.5, 17.0], scaleanchor="x", scaleratio=1.0, fixedrange=False),
        height=720, plot_bgcolor="#F9FAFB", paper_bgcolor="#FFFFFF",
        margin=dict(l=0, r=0, t=55, b=90),
        legend=dict(
            orientation="v", yanchor="bottom", y=-0.11, xanchor="left", x=0,
            font=dict(size=15, family="Arial"),
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#E5E7EB", borderwidth=1,
            itemsizing="constant",
        ),
        dragmode="zoom",
    )
    fig_grafo.update_yaxes(range=list(Y_RANGO))
    return fig_grafo


# ═════════════════════════════════════════════════════════════════════════════
# GRAFO RESUMEN DE LA SIMULACIÓN
# ═════════════════════════════════════════════════════════════════════════════
# build_fig_grafo() muestra UNA hora. Lo de abajo agrega TODO el horizonte de
# un escenario en una sola figura descargable: qué líneas fallaron (y cuántas
# horas), dónde se instaló GD / ESS / MEG / MESS y qué líneas recibieron
# medidas fijas (vegetación / refuerzo / soterramiento / seccionadores /
# enlaces nuevos).

# Recursos que se dibujan como "badge" alrededor del nodo. El orden fija la
# posición del badge, así que un mismo recurso queda siempre en el mismo lugar.
# `label` va en el hover (HTML del navegador, muestra el emoji); `leyenda` va
# en la leyenda de la figura, sin emoji porque kaleido no trae fuente emoji y
# en el PNG exportado saldrían como cuadrados vacíos.
# `borde` es un tono oscuro del propio color: un contorno blanco sobre el fondo
# claro del grafo se come el ícono y lo vuelve invisible.
RECURSO_STYLE = {
    "dg_sol": dict(symbol="star",          color="#F59E0B", borde="#92400E",
                   label="☀️ GD solar instalada",  leyenda="GD solar instalada"),
    "dg_eol": dict(symbol="triangle-up",   color="#0EA5E9", borde="#075985",
                   label="💨 GD eólica instalada", leyenda="GD eólica instalada"),
    "dg_fos": dict(symbol="square",        color="#78716C", borde="#292524",
                   label="⛽ GD fósil instalada",  leyenda="GD fósil instalada"),
    "ess":    dict(symbol="hexagon",       color="#10B981", borde="#065F46",
                   label="🔋 ESS instalado",       leyenda="ESS instalado"),
    "meg":    dict(symbol="triangle-right",color="#DB2777", borde="#831843",
                   label="🚛 MEG asignado",        leyenda="MEG asignado (móvil)"),
    "mess":   dict(symbol="diamond",       color="#2563EB", borde="#1E3A8A",
                   label="🚚 MESS asignado",       leyenda="MESS asignado (móvil)"),
    "edrp":   dict(symbol="x",             color="#6B7280", borde="#1F2937",
                   label="📉 EDRP activado",       leyenda="EDRP activado"),
}
MEDIDA_LINEA_LEYENDA = {
    "veg": "Gestión de vegetación",
    "ref": "Refuerzo estructural",
    "ug":  "Soterramiento",
}
# El grafo resumen usa UN marcador por cosa, no uno por tipo de medida: un
# rombo azul en el bus ("acá se instaló algo") y un halo gris en la línea
# ("acá hubo intervención"). El qué se detalla en el texto bajo el grafo.
COLOR_MEDIDA_BUS = "#2563EB"
COLOR_INTERVENCION = "#94A3B8"


def _recursos_recuadros(d):
    """Recursos presentes, distinguiendo las duraciones de almacenamiento."""
    recursos = set(d.get("recursos") or [])
    if "ess" in recursos:
        duraciones = [f"ess_{h}" for h in ("4h", "5h", "6h")
                      if _num(d.get(f"cap_ess_{h}")) > 0.01]
        if duraciones:
            recursos.remove("ess")
            recursos.update(duraciones)
    return [r for r in RECUADRO_STYLE if r in recursos]


RECUADRO_STYLE = {k: (v["color"], v["leyenda"]) for k, v in RECURSO_STYLE.items()}
RECUADRO_STYLE.update({
    "ess_4h": ("#10B981", "ESS 4 h"),
    "ess_5h": ("#8B5CF6", "ESS 5 h"),
    "ess_6h": ("#F43F5E", "ESS 6 h"),
})

_SWITCH_CORTO = {"rcs": "RCS telecomandado", "ms": "MS manual"}

# Buses conectados al sistema externo: ahí el rombo va bajo el nodo, porque
# arriba está el transformador y no se vería.
_BUSES_ROMBO_ABAJO = {1, 12}

# Tamaño de la figura. El PNG se exporta con este mismo tamaño y solo se sube
# la escala: si se pide otro width/height, los marcadores (que van en px) dejan
# de calzar con la geometría (que va en unidades de dato) y el grafo se
# desarma.
FIG_ANCHO, FIG_ALTO = 1250, 1010
_MARGEN_T, _MARGEN_B = 80, 250   # el margen inferior aloja la leyenda vertical
_Y_RANGO = Y_RANGO

SWITCH_STYLE = {
    "rcs": dict(color="#0F766E", label="🎛️ Seccionador telecomandado (RCS)",
                leyenda="Seccionador telecomandado (RCS)"),
    "ms":  dict(color="#92400E", label="🔧 Seccionador manual (MS)",
                leyenda="Seccionador manual (MS)"),
}


# Versión de la estructura que devuelve `resumir_simulacion`. Súbela cuando
# agregues o cambies llaves de `resumen["lineas"]` / `resumen["buses"]`: quien
# cachee un resumen debe incluirla en su clave, para no reusar uno armado por
# una versión anterior del código (que no tendría las llaves nuevas).
#   1 → estructura original
#   2 → agrega horas_seccionada / tiene_sec (seccionador abierto vs. arrastre)
#   3 → horas_seccionada se deriva de `seccionadores`, así que también sale
#       con resultados calculados antes de que existiera la lista explícita
# 4: seccionadores existentes incluso sin resultados horarios.
RESUMEN_SCHEMA = 4


def _num(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _activo(x):
    return _num(x) > 0.5


def resumir_simulacion(res_lista, faultlin_esc=None, seccionadores=None):
    """Agrega los resultados horarios de UN escenario en un resumen único.

    `res_lista` es la lista de dicts por hora que devuelve el optimizador para
    el escenario (la misma que consume build_fig_grafo hora a hora).
    `faultlin_esc` es opcional: matriz (horas × líneas) de faultlinMC para ese
    escenario, usada solo como respaldo si aún no hay resultados.

    Devuelve un dict con dos sub-dicts (`lineas`, `buses`) más `totales`.
    Las variables de inversión (CAP_*, x_veg/x_ref/x_ug, y_rcs/y_ms, y_tie)
    son constantes en el horizonte: se toman con `max` sobre las horas para no
    depender de que la hora 0 traiga el valor.
    """
    horas_res = [h for h in (res_lista or []) if h]
    n_horas = len(horas_res)

    # ── Estructura por línea ──────────────────────────────────
    lineas = {}
    for l in LINES:
        try:
            l_int = int(l)
            fb = int(net.line.from_bus[l])
            tb = int(net.line.to_bus[l])
        except Exception:
            continue
        try:
            nombre = str(net.line.at[l, "name"])
        except Exception:
            nombre = f"L{l_int}"
        lineas[l_int] = dict(
            nombre=nombre, from_bus=fb, to_bus=tb,
            es_tie=(l in tie_indices_set or l_int in tie_indices_set),
            es_candidata=(l in sw_cand_list or l_int in sw_cand_list),
            horas_falla=0, horas_aislada=0, horas_seccionada=0,
            horas_fuera=0, horas_tie_cerrada=0,
            medida=None, switch=None, tie_nueva=False, tiene_sec=False,
        )

    # ── Estructura por bus ────────────────────────────────────
    buses = {}
    for b in BUSES:
        b_int = int(b)
        buses[b_int] = dict(
            zona=bus_zone.get(b, bus_zone.get(b_int, "—")),
            pd_base=_num(P_D_base.get(b, P_D_base.get(b_int, 0.0))),
            ens_kwh=0.0, horas_sin_sum=0,
            cap_dg_sol=0.0, cap_dg_eol=0.0, cap_dg_fos=0.0,
            cap_ess_4h=0.0, cap_ess_5h=0.0, cap_ess_6h=0.0,
            e_dg_sol=0.0, e_dg_eol=0.0, e_dg_fos=0.0, e_ess_dis=0.0,
            cap_meg=0.0, horas_meg=0, e_meg=0.0,
            cap_mess=0.0, horas_mess=0, e_mess=0.0,
            e_edrp=0.0, horas_edrp=0,
        )

    sec_arg = list(seccionadores or [])
    for sec in sec_arg:
        li = int(sec)
        if li in lineas:
            lineas[li]["tiene_sec"] = True

    def _b(k):
        """Bus destino de una clave simple o de tupla (unidad, bus)."""
        if isinstance(k, tuple) and k:
            k = k[-1]
        try:
            return int(k)
        except Exception:
            return None

    for hd in horas_res:
        # Fallas y aislación
        for l in hd.get("lineas_danadas_indices", []) or []:
            li = _b(l)
            if li in lineas:
                lineas[li]["horas_falla"] += 1
        # Qué líneas llevan seccionador: del resultado si viene, si no del
        # argumento `seccionadores` (lo pasa la app desde el cfg de la
        # corrida). Permite separar maniobra de arrastre también en
        # resultados calculados antes de que el optimizador lo reportara.
        _sec_h = {_b(s) for s in (hd.get("seccionadores") or sec_arg or [])}
        _sec_h.discard(None)
        for li in _sec_h:
            if li in lineas:
                lineas[li]["tiene_sec"] = True

        _aisl = [_b(l) for l in (hd.get("lineas_aisladas_indices") or [])]
        if not _aisl:
            fuera = {_b(l) for l in (hd.get("lineas_fuera_servicio_indices") or [])}
            danadas = {_b(l) for l in (hd.get("lineas_danadas_indices") or [])}
            _aisl = sorted(l for l in fuera - danadas if l is not None)
        _secc = hd.get("lineas_seccionadas_indices")
        if _secc is None:
            # Derivado: la que tiene seccionador y quedó fuera es la que se
            # abrió; el resto perdió suministro por arrastre.
            _secc = [li for li in _aisl if li in _sec_h]
        else:
            _secc = [_b(l) for l in _secc]
        _secc_set = set(_secc)

        for li in _aisl:
            if li in lineas and li not in _secc_set:
                lineas[li]["horas_aislada"] += 1
        for li in _secc:
            if li in lineas:
                lineas[li]["horas_seccionada"] += 1
        for l in hd.get("lineas_fuera_servicio_indices", []) or []:
            li = _b(l)
            if li in lineas:
                lineas[li]["horas_fuera"] += 1

        # Medidas fijas por línea (constantes en el horizonte)
        for clave, medida in (("x_veg", "veg"), ("x_ref", "ref"), ("x_ug", "ug")):
            for l, v in (hd.get(clave, {}) or {}).items():
                li = _b(l)
                if li in lineas and _activo(v):
                    lineas[li]["medida"] = medida
        for clave, tipo in (("y_rcs", "rcs"), ("y_ms", "ms")):
            for l, v in (hd.get(clave, {}) or {}).items():
                li = _b(l)
                if li in lineas and _activo(v):
                    lineas[li]["switch"] = tipo
        for l, v in (hd.get("y_tie", {}) or {}).items():
            li = _b(l)
            if li in lineas and _activo(v):
                lineas[li]["tie_nueva"] = True
        for l, v in (hd.get("z_tie", {}) or {}).items():
            li = _b(l)
            if li in lineas and _activo(v):
                lineas[li]["horas_tie_cerrada"] += 1

        # Carga no servida. Si un bus queda totalmente sin carga servida,
        # marcar también sus líneas sanas adyacentes como sin suministro;
        # algunas corridas guardadas no incluyen las listas de aislación.
        buses_sin_suministro = set()
        _pb_h = {_b(k): _num(v) for k, v in (hd.get("Pb", {}) or {}).items()}
        for k, v in (hd.get("P_LC", {}) or {}).items():
            bi = _b(k)
            if bi in buses:
                val = _num(v)
                if val > 0.01:
                    buses[bi]["ens_kwh"] += val
                    buses[bi]["horas_sin_sum"] += 1
                    if bi in _pb_h:
                        if _pb_h[bi] <= 0.01:
                            buses_sin_suministro.add(bi)
                    else:
                        buses_sin_suministro.add(bi)
        _dan_h = {_b(l) for l in (hd.get("lineas_danadas_indices") or [])}
        for li, dl in lineas.items():
            if (not dl["es_tie"] and li not in _dan_h and li not in _secc_set
                    and li not in _aisl and
                    (dl["from_bus"] in buses_sin_suministro or
                     dl["to_bus"] in buses_sin_suministro)):
                dl["horas_aislada"] += 1

        # Capacidades instaladas (máximo sobre el horizonte)
        for clave, campo in (("CAP_DG_sol_kW", "cap_dg_sol"),
                             ("CAP_DG_eol_kW", "cap_dg_eol"),
                             ("CAP_DG_fos_kW", "cap_dg_fos"),
                             ("CAP_ESS_kWh_4h", "cap_ess_4h"),
                             ("CAP_ESS_kWh_5h", "cap_ess_5h"),
                             ("CAP_ESS_kWh_6h", "cap_ess_6h")):
            for k, v in (hd.get(clave, {}) or {}).items():
                bi = _b(k)
                if bi in buses:
                    buses[bi][campo] = max(buses[bi][campo], _num(v))

        # Energía inyectada por cada recurso
        for clave, campo in (("P_DG_sol", "e_dg_sol"),
                             ("P_DG_eol", "e_dg_eol"),
                             ("P_DG_fos", "e_dg_fos")):
            for k, v in (hd.get(clave, {}) or {}).items():
                bi = _b(k)
                if bi in buses:
                    buses[bi][campo] += max(_num(v), 0.0)
        for dur in ("4h", "5h", "6h"):
            for k, v in (hd.get(f"P_ESS_{dur}", {}) or {}).items():
                bi = _b(k)
                if bi in buses:
                    buses[bi]["e_ess_dis"] += max(_num(v), 0.0)

        # MEG: x_re fija el bus, CAP_MEG_MW la capacidad de la unidad
        cap_meg_k = hd.get("CAP_MEG_MW", {}) or {}
        for k, v in (hd.get("x_re", {}) or {}).items():
            if not _activo(v) or not isinstance(k, tuple):
                continue
            bi = _b(k)
            if bi in buses:
                buses[bi]["cap_meg"] = max(buses[bi]["cap_meg"], _num(cap_meg_k.get(k[0], 0.0)))
        for k, v in (hd.get("P_MEG", {}) or {}).items():
            bi = _b(k)
            if bi in buses and _num(v) > 0.01:
                buses[bi]["e_meg"] += _num(v)
                buses[bi]["horas_meg"] += 1

        # MESS: u_mess da la posición hora a hora
        cap_mess_k = hd.get("CAP_MESS_MWh", {}) or {}
        for k, v in (hd.get("u_mess", {}) or {}).items():
            if not _activo(v) or not isinstance(k, tuple):
                continue
            bi = _b(k)
            if bi in buses:
                buses[bi]["cap_mess"] = max(buses[bi]["cap_mess"], _num(cap_mess_k.get(k[0], 0.0)))
                buses[bi]["horas_mess"] += 1
        for k, v in (hd.get("P_mess", {}) or {}).items():
            bi = _b(k)
            if bi in buses:
                buses[bi]["e_mess"] += max(_num(v), 0.0)

        # EDRP
        for k, v in (hd.get("EDRP_total", {}) or {}).items():
            bi = _b(k)
            if bi in buses and _num(v) > 0.01:
                buses[bi]["e_edrp"] += _num(v)
                buses[bi]["horas_edrp"] += 1

    # Respaldo: sin resultados, al menos las fallas del Monte Carlo
    if not horas_res and faultlin_esc is not None:
        try:
            arr = np.asarray(faultlin_esc)
            n_horas = int(arr.shape[0])
            for li, l in enumerate(LINES):
                l_int = int(l)
                if l_int in lineas and li < arr.shape[1]:
                    lineas[l_int]["horas_falla"] = int((arr[:, li] > 0).sum())
        except Exception:
            pass

    # ── Recursos presentes por bus (para los badges del grafo) ──
    for bi, d in buses.items():
        recursos = []
        if d["cap_dg_sol"] > 0.01:
            recursos.append("dg_sol")
        if d["cap_dg_eol"] > 0.01:
            recursos.append("dg_eol")
        if d["cap_dg_fos"] > 0.01:
            recursos.append("dg_fos")
        if max(d["cap_ess_4h"], d["cap_ess_5h"], d["cap_ess_6h"]) > 0.01:
            recursos.append("ess")
        if d["cap_meg"] > 0.01 or d["horas_meg"] > 0:
            recursos.append("meg")
        if d["cap_mess"] > 0.01 or d["horas_mess"] > 0:
            recursos.append("mess")
        if d["e_edrp"] > 0.01:
            recursos.append("edrp")
        d["recursos"] = recursos

    totales = dict(
        n_horas=n_horas,
        lineas_con_falla=sum(1 for d in lineas.values() if d["horas_falla"] > 0),
        horas_falla_totales=sum(d["horas_falla"] for d in lineas.values()),
        lineas_aisladas=sum(1 for d in lineas.values() if d["horas_aislada"] > 0),
        lineas_con_medida=sum(1 for d in lineas.values() if d["medida"]),
        lineas_con_switch=sum(1 for d in lineas.values() if d["switch"]),
        tie_nuevas=sum(1 for d in lineas.values() if d["tie_nueva"]),
        buses_con_dg=sum(1 for d in buses.values()
                         if any(r in d["recursos"] for r in ("dg_sol", "dg_eol", "dg_fos"))),
        buses_con_ess=sum(1 for d in buses.values() if "ess" in d["recursos"]),
        buses_con_movil=sum(1 for d in buses.values()
                            if any(r in d["recursos"] for r in ("meg", "mess"))),
        buses_sin_sum=sum(1 for d in buses.values() if d["horas_sin_sum"] > 0),
        ens_total_kwh=sum(d["ens_kwh"] for d in buses.values()),
        cap_dg_total_kw=sum(d["cap_dg_sol"] + d["cap_dg_eol"] + d["cap_dg_fos"]
                            for d in buses.values()),
        cap_ess_total_kwh=sum(d["cap_ess_4h"] + d["cap_ess_5h"] + d["cap_ess_6h"]
                              for d in buses.values()),
    )

    return dict(lineas=lineas, buses=buses, totales=totales)


# ════════════════════════════════════════════════════════════════
# AGREGACIÓN SOBRE ESCENARIOS
# ════════════════════════════════════════════════════════════════
# Un escenario suelto no dice dónde está el riesgo: el Monte Carlo tira
# fallas distintas cada vez, y elegir "uno bonito" para la tesis es elegir
# el resultado. Agregar los N escenarios da el mapa de exposición, que es
# el valor esperado del proceso y no depende de qué escenario se mire.
#
# ⚠️ Lo que sale de acá NO es un estado realizable de la red: una línea
# puede aparecer a la vez con horas de falla y de seccionamiento porque
# eso pasó en escenarios DISTINTOS. Sirve para "dónde falla" y "dónde se
# invierte", no para narrar una secuencia operativa: para eso está el
# escenario representativo (run_barrido_medidas.py, fase 2).

# Campos que se ACUMULAN al fundir escenarios (todos son conteos o energías).
_AGG_SUMA_LINEA = ("horas_falla", "horas_aislada", "horas_seccionada",
                   "horas_fuera", "horas_tie_cerrada")
_AGG_SUMA_BUS   = ("ens_kwh", "horas_sin_sum", "e_dg_sol", "e_dg_eol",
                   "e_dg_fos", "e_ess_dis", "e_meg", "horas_meg", "e_mess",
                   "horas_mess", "e_edrp", "horas_edrp")
# Capacidades: se PROMEDIAN sobre los escenarios. Sumarlas daría la
# capacidad de instalar N veces la misma central; el promedio es la
# capacidad esperada en ese bus, que es lo que se dimensiona.
_AGG_MEDIA_BUS  = ("cap_dg_sol", "cap_dg_eol", "cap_dg_fos", "cap_ess_4h",
                   "cap_ess_5h", "cap_ess_6h", "cap_meg", "cap_mess")
# Cómo se nombran esas capacidades en las columnas de rango del CSV.
_CAP_TABLA = (("cap_dg_sol", "GD solar", "kW"), ("cap_dg_eol", "GD eólica", "kW"),
              ("cap_dg_fos", "GD fósil", "kW"), ("cap_ess_4h", "ESS 4h", "kWh"),
              ("cap_ess_5h", "ESS 5h", "kWh"), ("cap_ess_6h", "ESS 6h", "kWh"),
              ("cap_meg", "MEG", "kW"), ("cap_mess", "MESS", "kWh"))


def agregar_resumenes(resumenes, etiquetas=None):
    """Funde N resúmenes de escenario en uno solo, con el mismo formato.

    Devuelve un dict `{lineas, buses, totales}` idéntico en forma al de
    `resumir_simulacion` —así lo dibuja `build_fig_grafo_resumen` sin
    tocar nada— más las frecuencias de selección, que son la información
    nueva que aporta agregar: en cuántos de los N escenarios falló cada
    línea y en cuántos el solver puso un recurso en cada barra.

    `totales["n_horas"]` es la suma de las horas de todos los escenarios,
    así que la fracción `horas_falla / n_horas` que usa la figura para el
    grosor sigue siendo comparable con la de un escenario suelto.
    """
    vals = [r for r in (resumenes or []) if r and r.get("lineas")]
    n_esc = len(vals)
    if not n_esc:
        return dict(lineas={}, buses={}, totales=dict(n_horas=0, n_esc=0))
    etiquetas = list(etiquetas or [])[:n_esc]

    lineas, buses = {}, {}
    # Los conteos por línea/bus se llevan aparte porque no son campos del
    # formato original: se inyectan al final para no romper a quien lea
    # el resumen esperando el esquema de un escenario.
    votos_medida, votos_switch = {}, {}

    for r in vals:
        for l_int, d in (r.get("lineas") or {}).items():
            li = int(l_int)
            acc = lineas.setdefault(li, {
                **{k: v for k, v in d.items()
                   if k in ("nombre", "from_bus", "to_bus", "es_tie", "es_candidata")},
                **{k: 0 for k in _AGG_SUMA_LINEA},
                "medida": None, "switch": None, "tie_nueva": False,
                "tiene_sec": False, "esc_con_falla": 0, "esc_con_medida": 0,
                "esc_con_switch": 0, "esc_con_tie_nueva": 0, "esc_con_sec": 0,
            })
            for k in _AGG_SUMA_LINEA:
                acc[k] += int(d.get(k, 0) or 0)
            if (d.get("horas_falla") or 0) > 0:
                acc["esc_con_falla"] += 1
            if d.get("medida"):
                acc["esc_con_medida"] += 1
                votos_medida.setdefault(li, {})
                votos_medida[li][d["medida"]] = votos_medida[li].get(d["medida"], 0) + 1
            if d.get("switch"):
                acc["esc_con_switch"] += 1
                votos_switch.setdefault(li, {})
                votos_switch[li][d["switch"]] = votos_switch[li].get(d["switch"], 0) + 1
            if d.get("tie_nueva"):
                acc["esc_con_tie_nueva"] += 1
            if d.get("tiene_sec"):
                acc["esc_con_sec"] += 1
                acc["tiene_sec"] = True

        for b_int, d in (r.get("buses") or {}).items():
            bi = int(b_int)
            acc = buses.setdefault(bi, {
                "zona": d.get("zona", "—"), "pd_base": _num(d.get("pd_base")),
                **{k: 0.0 for k in _AGG_SUMA_BUS},
                **{k: 0.0 for k in _AGG_MEDIA_BUS},
                # Rango de la capacidad ENTRE LOS ESCENARIOS QUE LA
                # INSTALARON. El promedio solo dice cuánto sale en promedio;
                # el rango dice si el dimensionamiento es estable (240-260 kW)
                # o si el escenario lo manda (39-393 kW), que es lo que hay
                # que saber para proponer un tamaño. Los escenarios que no
                # instalan nada quedan fuera del rango a propósito: si
                # entraran, el mínimo sería 0 siempre y el rango no diría
                # nada — cuántos instalan ya lo cuenta `esc_con_*`.
                **{f"{k}_min": None for k in _AGG_MEDIA_BUS},
                **{f"{k}_max": 0.0 for k in _AGG_MEDIA_BUS},
                **{f"{k}_n": 0 for k in _AGG_MEDIA_BUS},
                "esc_con_dg": 0, "esc_con_ess": 0, "esc_con_movil": 0,
                "esc_sin_sum": 0,
            })
            for k in _AGG_SUMA_BUS:
                acc[k] += _num(d.get(k))
            for k in _AGG_MEDIA_BUS:
                v = _num(d.get(k))
                acc[k] += v                       # se divide por n_esc abajo
                if v > 0.01:
                    acc[f"{k}_n"] += 1
                    acc[f"{k}_max"] = max(acc[f"{k}_max"], v)
                    acc[f"{k}_min"] = (v if acc[f"{k}_min"] is None
                                       else min(acc[f"{k}_min"], v))
            rec = set(d.get("recursos") or [])
            if rec & {"dg_sol", "dg_eol", "dg_fos"}:
                acc["esc_con_dg"] += 1
            if "ess" in rec:
                acc["esc_con_ess"] += 1
            if rec & {"meg", "mess"}:
                acc["esc_con_movil"] += 1
            if (d.get("horas_sin_sum") or 0) > 0:
                acc["esc_sin_sum"] += 1

    # La medida/switch que se dibuja es la MODA: en el mapa agregado el
    # halo dice "acá el solver interviene", y la frecuencia con que lo
    # elige va en el hover. Empate → la primera por orden alfabético, que
    # es arbitrario pero estable entre corridas.
    for li, d in lineas.items():
        if votos_medida.get(li):
            d["medida"] = max(sorted(votos_medida[li]), key=lambda m: votos_medida[li][m])
        if votos_switch.get(li):
            d["switch"] = max(sorted(votos_switch[li]), key=lambda s: votos_switch[li][s])
        d["tie_nueva"] = d["esc_con_tie_nueva"] > 0

    for d in buses.values():
        for k in _AGG_MEDIA_BUS:
            d[k] = d[k] / n_esc
        recursos = []
        if max(d["cap_dg_sol"], d["cap_dg_eol"], d["cap_dg_fos"]) > 0.01:
            for campo, nom in (("cap_dg_sol", "dg_sol"), ("cap_dg_eol", "dg_eol"),
                               ("cap_dg_fos", "dg_fos")):
                if d[campo] > 0.01:
                    recursos.append(nom)
        if max(d["cap_ess_4h"], d["cap_ess_5h"], d["cap_ess_6h"]) > 0.01:
            recursos.append("ess")
        if d["cap_meg"] > 0.01 or d["horas_meg"] > 0:
            recursos.append("meg")
        if d["cap_mess"] > 0.01 or d["horas_mess"] > 0:
            recursos.append("mess")
        if d["e_edrp"] > 0.01:
            recursos.append("edrp")
        d["recursos"] = recursos

    n_horas = sum(int((r.get("totales") or {}).get("n_horas", 0) or 0) for r in vals)
    totales = dict(
        n_horas=n_horas, n_esc=n_esc, escenarios=etiquetas,
        lineas_con_falla=sum(1 for d in lineas.values() if d["horas_falla"] > 0),
        horas_falla_totales=sum(d["horas_falla"] for d in lineas.values()),
        lineas_aisladas=sum(1 for d in lineas.values() if d["horas_aislada"] > 0),
        lineas_con_medida=sum(1 for d in lineas.values() if d["medida"]),
        lineas_con_switch=sum(1 for d in lineas.values() if d["switch"]),
        tie_nuevas=sum(1 for d in lineas.values() if d["tie_nueva"]),
        buses_con_dg=sum(1 for d in buses.values() if d["esc_con_dg"] > 0),
        buses_con_ess=sum(1 for d in buses.values() if d["esc_con_ess"] > 0),
        buses_con_movil=sum(1 for d in buses.values() if d["esc_con_movil"] > 0),
        buses_sin_sum=sum(1 for d in buses.values() if d["horas_sin_sum"] > 0),
        ens_total_kwh=sum(d["ens_kwh"] for d in buses.values()),
        # Promedios por escenario: son los que se citan en el texto ("ENS
        # esperada"), y sumar 19 escenarios da un número sin unidad útil.
        ens_media_kwh=sum(d["ens_kwh"] for d in buses.values()) / n_esc,
        cap_dg_total_kw=sum(d["cap_dg_sol"] + d["cap_dg_eol"] + d["cap_dg_fos"]
                            for d in buses.values()),
        cap_ess_total_kwh=sum(d["cap_ess_4h"] + d["cap_ess_5h"] + d["cap_ess_6h"]
                              for d in buses.values()),
    )
    return dict(lineas=lineas, buses=buses, totales=totales)


def resumir_fallas_mc(esc_lista=None, seccionadores=None, faultlin=None):
    """Mapa de fallas del Monte Carlo, sin necesidad de optimizar nada.

    Es el agregado del caso base sobre TODOS los escenarios: se arma
    directo de `faultlinMC` (horas × escenarios × líneas), así que no
    depende de que haya baselines resueltos y cubre los 100 escenarios,
    no solo los que tienen caso base en disco.

    Con `seccionadores` además reparte las horas fuera de servicio entre
    la línea averiada y las que arrastra la zona de aislación, que es la
    diferencia que justifica invertir en seccionadores.
    """
    arr = faultlinMC if faultlin is None else faultlin
    arr = np.asarray(arr)
    n_esc_tot = arr.shape[1]
    escs = sorted({int(e) for e in (esc_lista if esc_lista is not None
                                    else range(n_esc_tot))
                   if 0 <= int(e) < n_esc_tot})
    if not escs:
        return dict(lineas={}, buses={}, totales=dict(n_horas=0, n_esc=0))

    sec = sorted({int(s) for s in (seccionadores or [])})
    if sec:
        from aislacion import adyacencia_lineas, zona_aislada
        _fb = {l: int(net.line.from_bus[l]) for l in LINES}
        _tb = {l: int(net.line.to_bus[l]) for l in LINES}
        _ady = adyacencia_lineas(LINES, _fb, _tb)
        _tie = {int(t) for t in _tie_indices}

    lineas = {}
    for pos, l in enumerate(LINES):
        try:
            l_int, fb, tb = int(l), int(net.line.from_bus[l]), int(net.line.to_bus[l])
        except Exception:
            continue
        try:
            nombre = str(net.line.at[l, "name"])
        except Exception:
            nombre = f"L{l_int}"
        lineas[l_int] = dict(
            nombre=nombre, from_bus=fb, to_bus=tb, pos=pos,
            es_tie=(l in _tie_indices or l_int in _tie_indices),
            es_candidata=(l in sw_cand_list or l_int in sw_cand_list),
            horas_falla=0, horas_aislada=0, horas_seccionada=0, horas_fuera=0,
            horas_tie_cerrada=0, medida=None, switch=None, tie_nueva=False,
            tiene_sec=(l_int in sec), esc_con_falla=0, esc_con_medida=0,
            esc_con_switch=0, esc_con_tie_nueva=0,
            esc_con_sec=(len(escs) if l_int in sec else 0),
        )

    for e in escs:
        m = arr[:, e, :]                       # horas × líneas
        for l_int, d in lineas.items():
            col = m[:, d["pos"]] > 0
            n = int(col.sum())
            d["horas_falla"] += n
            if n:
                d["esc_con_falla"] += 1
        if not sec:
            continue
        # Solo las horas CON falla: el resto la red está completa. Son
        # pocas (≈2 % de las 8760), así que el recorrido del grafo por
        # hora sale barato comparado con vectorizarlo.
        for h in np.flatnonzero(m.any(axis=1)):
            dam = m[h]
            fuera = zona_aislada(dam, LINES, _fb, _tb, _ady, sec, _tie)
            for l_int, d in lineas.items():
                p = d["pos"]
                if not fuera[p]:
                    continue
                d["horas_fuera"] += 1
                if dam[p]:
                    continue                   # la averiada ya se contó
                if l_int in sec:
                    d["horas_seccionada"] += 1  # se abrió para aislar
                else:
                    d["horas_aislada"] += 1     # arrastre, sin con qué cortar

    for d in lineas.values():
        d.pop("pos", None)
        if not sec:
            d["horas_fuera"] = d["horas_falla"]

    buses = {int(b): dict(zona=bus_zone.get(b, bus_zone.get(int(b), "—")),
                          pd_base=_num(P_D_base.get(b, P_D_base.get(int(b), 0.0))),
                          ens_kwh=0.0, horas_sin_sum=0, recursos=[],
                          esc_con_dg=0, esc_con_ess=0, esc_con_movil=0,
                          **{k: 0.0 for k in _AGG_MEDIA_BUS},
                          e_dg_sol=0.0, e_dg_eol=0.0, e_dg_fos=0.0,
                          e_ess_dis=0.0, e_meg=0.0, horas_meg=0, e_mess=0.0,
                          horas_mess=0, e_edrp=0.0, horas_edrp=0)
              for b in BUSES}

    totales = dict(
        n_horas=int(arr.shape[0]) * len(escs), n_esc=len(escs),
        escenarios=[e + 1 for e in escs],
        lineas_con_falla=sum(1 for d in lineas.values() if d["horas_falla"] > 0),
        horas_falla_totales=sum(d["horas_falla"] for d in lineas.values()),
        lineas_aisladas=sum(1 for d in lineas.values() if d["horas_aislada"] > 0),
        lineas_con_medida=0, lineas_con_switch=0, tie_nuevas=0,
        buses_con_dg=0, buses_con_ess=0, buses_con_movil=0, buses_sin_sum=0,
        ens_total_kwh=0.0, ens_media_kwh=0.0,
        cap_dg_total_kw=0.0, cap_ess_total_kwh=0.0,
    )
    return dict(lineas=lineas, buses=buses, totales=totales)


def build_fig_grafo_resumen(resumen, cfg=None, buses_criticos=None,
                            titulo=None, subtitulo=None, mostrar_seccionadores=False):
    """Figura Plotly con el resumen de toda la simulación (ver resumir_simulacion)."""
    cfg = cfg or {}
    lineas = resumen.get("lineas", {})
    buses = resumen.get("buses", {})
    tot = resumen.get("totales", {})
    n_horas = max(int(tot.get("n_horas", 0)), 1)
    # >0 solo si el resumen viene de `agregar_resumenes`/`resumir_fallas_mc`:
    # activa las frecuencias entre escenarios en los hovers.
    n_esc = int(tot.get("n_esc", 0) or 0)

    usar_prior_crit = bool(cfg.get("prior_criticos", False))
    if buses_criticos is None:
        buses_criticos = cfg.get("buses_criticos", [])
    BUSES_CRITICOS = set(int(b) for b in buses_criticos)

    fig = go.Figure()
    _add_zonas(fig)

    # ── Líneas ────────────────────────────────────────────────
    # Diseño deliberadamente pobre en símbolos: el dibujo solo responde "¿falló?"
    # y "¿tiene intervención?". QUÉ intervención se lee en el detalle de abajo.
    hay_falla = hay_aislada = hay_seccionada = hay_intervencion = False
    sec_config = {int(l) for l in cfg.get("seccionadores_existentes", []) or []}
    sec_x, sec_y, sec_labels = [], [], []

    for l_int, d in lineas.items():
        fb, tb = d["from_bus"], d["to_bus"]
        if fb not in BUSCOORDS or tb not in BUSCOORDS:
            continue
        x0, y0 = BUSCOORDS[fb]
        x1, y1 = BUSCOORDS[tb]

        frac = d["horas_falla"] / n_horas
        if mostrar_seccionadores and (d.get("tiene_sec") or int(l_int) in sec_config):
            sec_x.append((x0 + x1) / 2)
            sec_y.append((y0 + y1) / 2)
            sec_labels.append(f"SEC L{l_int}")
        intervenida = bool(d["medida"] or d["switch"] or d["tie_nueva"])

        # Un único halo neutro para cualquier intervención fija (vegetación,
        # refuerzo, soterramiento, seccionador o enlace nuevo). Va DEBAJO de la
        # línea de estado, así se ve a la vez si la línea falló.
        if intervenida:
            hay_intervencion = True
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                line=dict(color=COLOR_INTERVENCION, width=11),
                opacity=0.5, showlegend=False, hoverinfo="skip",
            ))

        if d["horas_falla"] > 0:
            hay_falla = True
            col, dash = LINE_COLORS["fault"], "solid"
            ww = 2.5 + 4.5 * min(frac * 6, 1.0)   # grosor ∝ severidad de la falla
        # Misma convención que el grafo por hora: punteado = abierto por
        # maniobra (seccionador), sólido = así quedó (sin suministro).
        elif d.get("horas_seccionada", 0) > 0:
            hay_seccionada = True
            col, dash, ww = LINE_COLORS["seccionada"], "8px,5px", 3.6
        elif (d.get("horas_aislada", 0) > 0 or
              _num(buses.get(d["from_bus"], buses.get(str(d["from_bus"]), {})).get("horas_sin_sum", 0)) > 0 or
              _num(buses.get(d["to_bus"], buses.get(str(d["to_bus"]), {})).get("horas_sin_sum", 0)) > 0):
            hay_aislada = True
            col, dash, ww = LINE_COLORS["aislada"], "solid", 3.6
        elif d["es_tie"]:
            col = (LINE_COLORS["tieclosed"] if d.get("horas_tie_cerrada", 0) > 0
                   else LINE_COLORS["ok"])
            dash = "solid" if d.get("horas_tie_cerrada", 0) > 0 else "dash"
            ww = 3.2
        else:
            col, dash, ww = LINE_COLORS["ok"], "solid", 2.4

        partes = [
            f"<b>Línea {l_int}</b> · {d['nombre']}",
            f"Bus {fb} → Bus {tb}",
            f"Horas en falla: {d['horas_falla']} de {n_horas} · {100*frac:.1f} %",
        ]
        # En el mapa agregado la frecuencia entre escenarios es EL dato: las
        # horas totales dependen de cuántos escenarios se fundieron, "falló
        # en 12 de 19" no.
        if n_esc:
            partes.append(f"Falló en {d.get('esc_con_falla', 0)} de {n_esc} "
                          f"escenarios · {100*d.get('esc_con_falla', 0)/n_esc:.0f} %")
            partes.append(f"Horas de falla por escenario: "
                          f"{d['horas_falla']/n_esc:.1f}")
            if d.get("esc_con_medida"):
                partes.append(f"Intervenida en {d['esc_con_medida']} de {n_esc} "
                              f"escenarios")
            if d.get("esc_con_switch"):
                partes.append(f"Switch instalado en {d['esc_con_switch']} de "
                              f"{n_esc} escenarios")
            if d.get("esc_con_tie_nueva"):
                partes.append(f"Enlace nuevo en {d['esc_con_tie_nueva']} de "
                              f"{n_esc} escenarios")
        # `.get`: el `resumen` puede venir persistido (artefactos del barrido,
        # sesiones guardadas) de antes de que existieran estas llaves.
        if d.get("horas_seccionada"):
            partes.append(f"Horas con el seccionador abierto: {d['horas_seccionada']}")
        if d.get("horas_aislada"):
            partes.append(f"Horas sin servicio por aislación: {d['horas_aislada']}")
        if d.get("tiene_sec") and not d.get("horas_seccionada"):
            partes.append("Tiene seccionador (nunca se abrió)")
        if d["medida"]:
            partes.append(MEDIDA_LINEA_STYLE[d["medida"]]["label"])
        if d["switch"]:
            partes.append(SWITCH_STYLE[d["switch"]]["label"])
        if d["tie_nueva"]:
            partes.append("🔗 Enlace nuevo construido")
        if d["es_tie"]:
            partes.append(f"Horas de enlace cerrado: {d['horas_tie_cerrada']}")

        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(color=col, width=ww, dash=dash),
            showlegend=False,
            hovertemplate="<br>".join(partes) + "<extra></extra>",
        ))

    add_external_system(fig, BUSCOORDS)

    # ── Nodos ─────────────────────────────────────────────────
    # Se dibujan en tres capas: nodo base → rombo de medida → nodo repetido
    # encima. Así el rombo asoma por detrás del nodo en vez de flotar suelto.
    if sec_x:
        fig.add_trace(go.Scatter(
            x=sec_x, y=sec_y, mode="markers+text",
            marker=dict(symbol="square", size=10, color="#FFFFFF",
                        line=dict(color="#2563EB", width=2)),
            text=sec_labels, textposition="top center",
            textfont=dict(size=12, color="#2563EB"),
            name="Seccionador preexistente", showlegend=True,
            hovertemplate="%{text}<br>Ubicación del seccionador preexistente<extra></extra>",
        ))

    max_pd = max(P_D_base.values()) if P_D_base else 1
    hay_medida_bus = False
    hay_sin_sum = False
    capa_base, capa_rombo = [], []
    recursos_presentes = {r for d in buses.values() for r in _recursos_recuadros(d)}
    distinguir_medidas = len(recursos_presentes) > 1

    # px por unidad de dato: el eje y es el que manda (rango más largo). Sirve
    # para sacar el rombo justo fuera del borde del nodo, que se dibuja en px y
    # crece con la demanda del bus.
    _px_unidad = (FIG_ALTO - _MARGEN_T - _MARGEN_B) / (_Y_RANGO[1] - _Y_RANGO[0])

    for b in [x for x in BUSCOORDS.keys() if x not in _BUSES_OCULTOS]:
        d = buses.get(int(b))
        if d is None:
            continue
        xb, yb = BUSCOORDS[b]
        zona_b = d["zona"] if isinstance(d["zona"], int) else bus_zone.get(b, 1)
        pdb = d["pd_base"]
        size_b = _tam_nodo(pdb, max_pd)

        sin_sum = d["horas_sin_sum"] > 0
        hay_sin_sum = hay_sin_sum or sin_sum
        es_critico = usar_prior_crit and (int(b) in BUSES_CRITICOS)

        fill_col = ("#F97316" if sin_sum else
                    "#8B5CF6" if es_critico else
                    NODEFILL.get(zona_b, "#E5E7EB"))
        strk_col = ("#7C2D12" if sin_sum else
                    "#4C1D95" if es_critico else
                    NODESTROKE.get(zona_b, "#374151"))
        txt_col = "#FFFFFF" if (sin_sum or es_critico) else "#1F2937"

        partes = [
            f"<b>Bus {b}</b> · {ZONE_NAME.get(zona_b, '')}",
            f"PD base: {pdb:.1f} kW",
            f"ENS acumulada: {d['ens_kwh']:.1f} kWh",
            f"Horas sin suministro: {d['horas_sin_sum']} de {n_horas}",
        ]
        # En el agregado NO se muestra el promedio de la capacidad: un solo
        # número esconde si el dimensionamiento es estable o si lo manda el
        # escenario. Se muestra el RANGO entre los escenarios que la
        # instalaron, más en cuántos la instalaron.
        def _cap(campo, unidad="kW"):
            """'336.4 kW' en un escenario; '39–393 kW (10/10 esc)' agregado."""
            if not n_esc:
                return f"{d[campo]:.1f} {unidad}"
            lo, hi, cu = d.get(f"{campo}_min"), d.get(f"{campo}_max"), d.get(f"{campo}_n")
            if lo is None:                    # resumen agregado sin los rangos
                return f"{d[campo]:.1f} {unidad} (promedio)"
            rango = (f"{lo:.1f} {unidad}" if abs(hi - lo) < 0.05
                     else f"{lo:.1f}–{hi:.1f} {unidad}")
            return f"{rango} en {cu} de {n_esc} esc"

        if n_esc:
            partes.append(f"ENS por escenario: {d['ens_kwh']/n_esc:,.0f} kWh")
            if d.get("esc_con_dg"):
                partes.append(f"Recibió GD en {d['esc_con_dg']} de {n_esc} escenarios")
            if d.get("esc_con_ess"):
                partes.append(f"Recibió ESS en {d['esc_con_ess']} de {n_esc} escenarios")
            if d.get("esc_con_movil"):
                partes.append(f"Recibió unidad móvil en {d['esc_con_movil']} de "
                              f"{n_esc} escenarios")
            partes.append("<i>Capacidades = rango entre los escenarios que "
                          "la instalaron</i>")
        if d["cap_dg_sol"] > 0.01:
            partes.append(f"☀️ GD solar: {_cap('cap_dg_sol')} · "
                          f"{d['e_dg_sol']:.0f} kWh generados")
        if d["cap_dg_eol"] > 0.01:
            partes.append(f"💨 GD eólica: {_cap('cap_dg_eol')} · "
                          f"{d['e_dg_eol']:.0f} kWh generados")
        if d["cap_dg_fos"] > 0.01:
            partes.append(f"⛽ GD fósil: {_cap('cap_dg_fos')} · "
                          f"{d['e_dg_fos']:.0f} kWh generados")
        for dur in ("4h", "5h", "6h"):
            if d[f"cap_ess_{dur}"] > 0.01:
                partes.append(f"🔋 ESS {dur}: {_cap(f'cap_ess_{dur}', 'kWh')}")
        if "ess" in d["recursos"]:
            partes.append(f"🔋 Energía descargada: {d['e_ess_dis']:.0f} kWh")
        if "meg" in d["recursos"]:
            partes.append(f"🚛 MEG: {_cap('cap_meg')} · "
                          f"{d['horas_meg']} h activo · {d['e_meg']:.0f} kWh")
        if "mess" in d["recursos"]:
            partes.append(f"🚚 MESS: {_cap('cap_mess', 'kWh')} · "
                          f"{d['horas_mess']} h en el bus · {d['e_mess']:.0f} kWh")
        if "edrp" in d["recursos"]:
            partes.append(f"📉 EDRP: {d['e_edrp']:.0f} kWh en {d['horas_edrp']} h")
        if not d["recursos"]:
            partes.append("— Sin medidas instaladas")

        traza_nodo = dict(
            x=[xb], y=[yb], mode="markers+text",
            marker=dict(size=size_b, color=fill_col,
                        line=dict(color=strk_col, width=BORDE_NODO), symbol="circle"),
            text=[_texto_nodo(b)], textposition="middle center",
            textfont=dict(size=9, color=txt_col, family="Arial Black"),
            showlegend=False,
            hovertemplate="<br>".join(partes) + "<extra></extra>",
            name=f"Bus {b}",
        )
        capa_base.append(traza_nodo)

        # Un solo rombo por bus: marca "acá se instaló algo". El qué se lee en
        # el hover y en las tablas, no en el dibujo. El desplazamiento sale del
        # radio real del nodo, que crece con la demanda: con un offset fijo el
        # rombo quedaba tapado justo en los buses más grandes.
        if d["recursos"] and distinguir_medidas:
            recursos_bus = _recursos_recuadros(d)
            cantidad = len(recursos_bus)
            # Corona alrededor del nodo: deja espacio para las esquinas
            # de los cuadrados y entre recursos cuando coinciden muchos.
            radio_px = size_b / 2 + TAM_RECUADRO + 1
            if cantidad > 1:
                radio_px = max(radio_px, 0.8 * TAM_RECUADRO / np.sin(np.pi / cantidad))
            radio = radio_px / _px_unidad
            inicio = -np.pi / 4 if int(b) in _BUSES_ROMBO_ABAJO else np.pi / 4
            for i, recurso in enumerate(recursos_bus):
                angulo = inicio + 2 * np.pi * i / cantidad
                color, etiqueta = RECUADRO_STYLE[recurso]
                campo_recurso = {
                    "dg_sol": "cap_dg_sol", "dg_eol": "cap_dg_eol",
                    "dg_fos": "cap_dg_fos", "ess_4h": "cap_ess_4h",
                    "ess_5h": "cap_ess_5h", "ess_6h": "cap_ess_6h",
                    "meg": "cap_meg", "mess": "cap_mess",
                }.get(recurso)
                detalle_recurso = ""
                if campo_recurso:
                    unidad_recurso = "kWh" if recurso.startswith("ess") or recurso == "mess" else "kW"
                    detalle_recurso = _cap(campo_recurso, unidad_recurso) + "<br>"
                capa_rombo.append(dict(
                    x=[xb + radio * np.cos(angulo)],
                    y=[yb + radio * np.sin(angulo)],
                    mode="markers", name=f"{etiqueta} · Bus {b}",
                    marker=dict(symbol="square", size=TAM_RECUADRO, color=color,
                                line=dict(color="white", width=1 if ES_CIGRE else 0.5)),
                    legendgroup=recurso, showlegend=False,
                    hovertemplate=f"<b>{etiqueta} · Bus {b}</b><br>"
                                  + detalle_recurso + "<br>".join(partes) + "<extra></extra>",
                ))
        elif d["recursos"]:
            hay_medida_bus = True
            radio = (size_b / 2) / _px_unidad
            paso = (radio + (0.10 if ES_CIGRE else 0.05)) / (2 ** 0.5)
            dy = -paso if int(b) in _BUSES_ROMBO_ABAJO else paso
            capa_rombo.append(dict(
                x=[xb + paso], y=[yb + dy], mode="markers",
                marker=dict(symbol="diamond", size=TAM_ROMBO, color=COLOR_MEDIDA_BUS,
                            line=dict(color="white", width=2 if ES_CIGRE else 0.5)),
                showlegend=False,
                hovertemplate="<br>".join(partes) + "<extra></extra>",
            ))

    # Capa 1: nodos · capa 2: rombos · capa 3: nodos otra vez, encima.
    for tr in capa_base:
        fig.add_trace(go.Scatter(**tr))
    for tr in capa_rombo:
        fig.add_trace(go.Scatter(**tr))
    for tr in capa_base:
        fig.add_trace(go.Scatter(**tr))

    # ── Leyenda (corta a propósito: 4 a 6 entradas) ───────────
    # Cada entrada se dibuja con el mismo trazo que usa en el grafo: los
    # enlaces van segmentados, así que la muestra de la leyenda va segmentada.
    entradas_linea = [("Línea sin fallas", LINE_COLORS["ok"], "solid", 2.4)]
    if any(d["es_tie"] and d.get("horas_tie_cerrada", 0) <= 0 for d in lineas.values()):
        entradas_linea.append(("Tie-line sin cierres registrados", LINE_COLORS["ok"], "dash", 3.2))
    if any(d["es_tie"] and d.get("horas_tie_cerrada", 0) > 0 for d in lineas.values()):
        entradas_linea.append(("Tie-line usada durante la simulación", LINE_COLORS["tieclosed"], "solid", 3.2))
    if hay_falla:
        entradas_linea.append(("Línea con falla", LINE_COLORS["fault"], "solid", 5))
    if hay_seccionada:
        entradas_linea.append(("Seccionador abierto (aísla la falla)",
                               LINE_COLORS["seccionada"], "8px,5px", 3.6))
    if hay_aislada:
        entradas_linea.append(("Línea sin servicio por aislación",
                               LINE_COLORS["aislada"], "solid", 3.6))
    if hay_intervencion:
        entradas_linea.append(("Línea intervenida", COLOR_INTERVENCION, "solid", 9))
    for lbl, col, dsh, ww in entradas_linea:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(color=col, width=ww, dash=dsh),
                                 name=lbl, showlegend=True, hoverinfo="skip"))

    if distinguir_medidas:
        for recurso in RECUADRO_STYLE:
            if recurso not in recursos_presentes:
                continue
            color, etiqueta = RECUADRO_STYLE[recurso]
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(symbol="square", size=10, color=color,
                            line=dict(color="white", width=1)),
                name=etiqueta, legendgroup=recurso,
                showlegend=True, hoverinfo="skip",
            ))
    if hay_medida_bus:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(symbol="diamond", size=12, color=COLOR_MEDIDA_BUS,
                                             line=dict(color="white", width=2)),
                                 name="Bus con medida instalada",
                                 showlegend=True, hoverinfo="skip"))
    if hay_sin_sum:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color="#F97316",
                                             line=dict(color="#7C2D12", width=2)),
                                 name="Bus con carga no suministrada",
                                 showlegend=True, hoverinfo="skip"))
    if usar_prior_crit and BUSES_CRITICOS:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color="#8B5CF6",
                                             line=dict(color="#4C1D95", width=2)),
                                 name="Bus crítico", showlegend=True, hoverinfo="skip"))

    titulo = titulo or "Resumen de la simulación"
    if subtitulo is None:
        subtitulo = (
            f"{tot.get('n_horas', 0)} h simuladas · "
            f"{tot.get('lineas_con_falla', 0)} líneas con falla · "
            f"{tot.get('buses_con_dg', 0)} buses con GD · "
            f"{tot.get('buses_con_ess', 0)} con ESS · "
            f"{tot.get('lineas_con_medida', 0)} líneas con medida fija · "
            f"ENS {tot.get('ens_total_kwh', 0.0):,.0f} kWh"
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{titulo}</b><br><span style='font-size:14px;color:#6B7280'>{subtitulo}</span>",
            font=dict(size=20, family="Arial"), x=0.01, xanchor="left",
        ),
        xaxis=dict(visible=False, range=list(X_RANGO), fixedrange=False),
        yaxis=dict(visible=False, range=list(_Y_RANGO), scaleanchor="x",
                   scaleratio=1.0, fixedrange=False),
        width=FIG_ANCHO, height=FIG_ALTO,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=_MARGEN_T, b=_MARGEN_B),
        legend=dict(orientation="v", yanchor="top", y=-0.02, xanchor="left", x=0,
                    font=dict(size=13, family="Arial", color="#111827"),
                    bgcolor="#FFFFFF", itemsizing="constant"),
        dragmode="zoom",
    )
    # Reservar una fila por entrada sin reducir el área del grafo:
    # la geometría de los marcadores depende de su altura en píxeles.
    n_leyenda = sum(tr.showlegend is True for tr in fig.data)
    margen_leyenda = max(_MARGEN_B, 30 + 24 * n_leyenda)
    fig.update_layout(
        height=FIG_ALTO + margen_leyenda - _MARGEN_B,
        margin=dict(b=margen_leyenda),
    )
    return fig


def tablas_resumen(resumen):
    """(df_lineas, df_buses) con el mismo contenido del grafo resumen, para
    exportar a CSV/Excel."""
    import pandas as pd

    _tot = resumen.get("totales", {}) or {}
    n_horas = max(int(_tot.get("n_horas", 0)), 1)
    # Resumen agregado sobre escenarios: las columnas de frecuencia son las
    # que se citan en la tesis ("L1 falla en 93 de 100 escenarios"), así que
    # tienen que salir en el CSV y no solo en el hover.
    n_esc = int(_tot.get("n_esc", 0) or 0)

    filas_l = []
    for l_int, d in sorted(resumen.get("lineas", {}).items()):
        extra_l = {}
        if n_esc:
            extra_l = {
                "Escenarios con falla": d.get("esc_con_falla", 0),
                "% escenarios con falla": round(
                    100 * d.get("esc_con_falla", 0) / n_esc, 1),
                "Horas de falla por escenario": round(d["horas_falla"] / n_esc, 2),
                "Escenarios con medida fija": d.get("esc_con_medida", 0),
                "Escenarios con seccionador nuevo": d.get("esc_con_switch", 0),
                "Escenarios con enlace nuevo": d.get("esc_con_tie_nueva", 0),
            }
        filas_l.append({
            "Línea": l_int,
            "Nombre": d["nombre"],
            "Bus origen": d["from_bus"],
            "Bus destino": d["to_bus"],
            "Horas en falla": d["horas_falla"],
            "% horas en falla": round(100 * d["horas_falla"] / n_horas, 2),
            "Horas con seccionador abierto": d.get("horas_seccionada", 0),
            "Horas sin servicio por aislación": d.get("horas_aislada", 0),
            "Medida fija": MEDIDA_LINEA_LEYENDA.get(d["medida"], "—"),
            "Seccionador instalado": _SWITCH_CORTO.get(d["switch"], "—"),
            "Enlace nuevo": "Sí" if d["tie_nueva"] else "—",
            "Es enlace (tie)": "Sí" if d["es_tie"] else "—",
            "Horas de enlace cerrado": d["horas_tie_cerrada"],
            **extra_l,
        })

    filas_b = []
    for b_int, d in sorted(resumen.get("buses", {}).items()):
        extra_b = {}
        if n_esc:
            extra_b = {
                "Escenarios con GD": d.get("esc_con_dg", 0),
                "Escenarios con ESS": d.get("esc_con_ess", 0),
                "Escenarios con unidad móvil": d.get("esc_con_movil", 0),
                "ENS por escenario [kWh]": round(d["ens_kwh"] / n_esc, 2),
            }
            # Rango de cada capacidad entre los escenarios que la instalaron.
            # Van como columnas numéricas aparte (mín/máx/nº) y no como el
            # texto "39–393" del hover: en el CSV el rango tiene que poder
            # graficarse y ordenarse, no solo leerse.
            for campo, etiq, uni in _CAP_TABLA:
                lo, hi = d.get(f"{campo}_min"), d.get(f"{campo}_max")
                extra_b[f"{etiq} mín [{uni}]"] = round(lo, 2) if lo is not None else 0.0
                extra_b[f"{etiq} máx [{uni}]"] = round(hi or 0.0, 2)
                extra_b[f"{etiq} nº esc"] = d.get(f"{campo}_n", 0)
        filas_b.append({
            "Bus": b_int,
            "Zona": d["zona"],
            "PD base [kW]": round(d["pd_base"], 2),
            "ENS acumulada [kWh]": round(d["ens_kwh"], 2),
            "Horas sin suministro": d["horas_sin_sum"],
            "GD solar [kW]": round(d["cap_dg_sol"], 2),
            "GD eólica [kW]": round(d["cap_dg_eol"], 2),
            "GD fósil [kW]": round(d["cap_dg_fos"], 2),
            "ESS 4h [kWh]": round(d["cap_ess_4h"], 2),
            "ESS 5h [kWh]": round(d["cap_ess_5h"], 2),
            "ESS 6h [kWh]": round(d["cap_ess_6h"], 2),
            "MEG [kW]": round(d["cap_meg"], 2),
            "Horas MEG activo": d["horas_meg"],
            "MESS [kWh]": round(d["cap_mess"], 2),
            "Horas MESS en el bus": d["horas_mess"],
            "Energía GD [kWh]": round(d["e_dg_sol"] + d["e_dg_eol"] + d["e_dg_fos"], 2),
            "Energía ESS descargada [kWh]": round(d["e_ess_dis"], 2),
            "Energía EDRP [kWh]": round(d["e_edrp"], 2),
            **extra_b,
        })

    return pd.DataFrame(filas_l), pd.DataFrame(filas_b)
