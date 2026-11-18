"""
grafo_layout_pp.py
Coordenadas originales de la red CIGRE MV (pandapower)
y polígonos por zona para el grafo de app_fixed_new.py
"""
import numpy as np
import plotly.graph_objects as go
from app_v6 import BUSES, LINES, tie_indices_set, fmc_hora, P_D_base, bus_zone, hora_grafo, res_hora

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

_ZONA_BUSES = {
    1: [1, 2, 3, 4],
    2: [7, 8, 12, 13, 14],
    3: [5, 6, 9, 10, 11],
}

ZONE_FILL   = {1: "rgba(59,130,246,0.12)",  2: "rgba(245,158,11,0.12)",  3: "rgba(16,185,129,0.12)"}
ZONE_BORDER = {1: "rgba(29,78,216,0.55)",   2: "rgba(180,83,9,0.55)",    3: "rgba(4,120,87,0.55)"}
ZONE_LABEL  = {1: "#1D4ED8",                2: "#B45309",                 3: "#047857"}
ZONE_NAME   = {1: "Zona 1",                 2: "Zona 2",                  3: "Zona 3"}
NODEFILL    = {1: "#BFDBFE",                2: "#FDE68A",                 3: "#A7F3D0"}
NODESTROKE  = {1: "#1D4ED8",                2: "#B45309",                 3: "#047857"}
LINE_COLORS = {"ok": "#16A34A", "fault": "#DC2626", "tieclosed": "#2563EB", "tieopen": "#9CA3AF"}

# ── Red eléctrica ────────────────────────────────────────
fig_grafo = go.Figure()
p3 = np.array(BUSCOORDS[3])
p8 = np.array(BUSCOORDS[8])
triple_x, triple_y = (p3 + p8) / 2  # (6, 8)
triple_y = triple_y - 1.5
triple_x = triple_x +1
# Punto donde la línea 3–8 cruza la frontera Zona 1–Zona 2

ZONE_POLYGONS = {
1: ([0.25, 7.0, triple_x, 0.25, 0.25],
    [16.75, 16.75, triple_y, triple_y, 16.75],),
2: ([7.0, 12.75, 12.75, 7.0, 7.0, 7.0, triple_x, 7.0],
    [16.75, 16.75, 2.25, 2.25, 5.0, 7.0, triple_y, 16.75],),
3: ([0.25, triple_x, 7.0, 7.0, 7.0, 0.25, 0.25],
    [triple_y, triple_y, 7.0, 5.0, 2.25, 2.25, triple_y],),
}

# 1) Polígonos de zona
for z, (xs, ys) in ZONE_POLYGONS.items():
    fig_grafo.add_trace(go.Scatter(
        x=xs, y=ys, fill="toself",
        fillcolor=ZONE_FILL[z],
        line=dict(color=ZONE_BORDER[z], width=2, dash="dot"),
        mode="lines", showlegend=False, hoverinfo="skip",
    ))
    cx   = (max(xs) + min(xs)) / 2 - 1.5
    cy_t = max(ys) - 0.35
    if z == 3:
        fig_grafo.add_annotation(x=cx+1.5, y=cy_t-5, text=f"<b>{ZONE_NAME[z]}</b>",
        showarrow=False,
        font=dict(size=13, color=ZONE_LABEL[z], family="Arial"),
        xanchor="center",
    )
    else:
        fig_grafo.add_annotation(x=cx, y=cy_t, text=f"<b>{ZONE_NAME[z]}</b>",
            showarrow=False,
            font=dict(size=13, color=ZONE_LABEL[z], family="Arial"),
            xanchor="center",
        )
def add_transformer_symbol(fig, bus_a, bus_b, buscoords):
    x1, y1 = buscoords[bus_a]
    x2, y2 = buscoords[bus_b]
    color = "#64748b"

    def point(t):
        return (
            x1 + t * (x2 - x1),
            y1 + t * (y2 - y1),
        )

    # Posición de los círculos
    c1 = point(0.45)
    c2 = point(0.55)

    # Extremos de las líneas, dejando libre el símbolo
    line_end_1 = point(0.38)
    line_start_2 = point(0.62)

    fig.add_trace(go.Scatter(
        x=[x1, line_end_1[0], None, line_start_2[0], x2],
        y=[y1, line_end_1[1], None, line_start_2[1], y2],
        mode="lines",
        line=dict(color=color, width=2),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=[c1[0], c2[0]],
        y=[c1[1], c2[1]],
        mode="markers",
        marker=dict(
            symbol="circle-open",
            size=18,
            color=color,
            line=dict(color=color, width=2.5),
        ),
        hovertemplate=f"Transformador {bus_a}-{bus_b}<extra></extra>",
        showlegend=False,
    ))

def add_vertical_transformer(fig, x, y_from, y_to, label="Transformador"):
    color = "#334155"

    y_mid = (y_from + y_to) / 2 + 0.725
    gap_low = y_mid - 0.22-0.22
    gap_high = y_mid + 0.22+0.22+0.725

    y_to_line = y_to - 0.04+0.18

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
# 2) Aristas
for l in LINES:
    try:
        fb = int(net.line.from_bus[l]);  tb = int(net.line.to_bus[l])
    except Exception:
        continue
    if fb not in BUSCOORDS or tb not in BUSCOORDS:
        continue
    x0, y0 = BUSCOORDS[fb];  x1, y1 = BUSCOORDS[tb]
    is_tie  = l in tie_indices_set

    if is_tie:
        cat    = "tieopen"
        ww, dash_e = (3.0, "10px,6px")
    else:
        danada = fmc_hora.get(l, False)
        cat    = "fault" if danada else "ok"
        ww, dash_e = (3.5, "solid") if danada else (2.5, "solid")

    fig_grafo.add_trace(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None], mode="lines",
        line=dict(color=LINE_COLORS[cat], width=ww, dash=dash_e),
        showlegend=False, hoverinfo="skip", legendgroup=cat,
    ))

def add_external_system(fig, buscoords):
    # Buses de conexión
    x1, y1 = buscoords[1]
    x12, y12 = buscoords[12]

    # Posición de conexión inferior del sistema
    y_port = 17.00

    # Misma forma, desplazada hacia arriba
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
        line=dict(color = "#f89d48", width=1.8),
        layer="above",
    )

    fig.add_annotation(
        x=6.35,
        y=19.65,
        text="<b>Sistema</b>",
        showarrow=False,
        font=dict(size=15, color="#000000", family="Arial"),
    )

    # Transformadores verticales hacia el sistema
    add_vertical_transformer(fig, x1, y1, y_port, label="Transformador 1-Sistema")
    add_vertical_transformer(fig, x12, y12, y_port, label="Transformador 12-Sistema")
    
add_external_system(fig_grafo, BUSCOORDS)
buses_a_dibujar = [b for b in BUSCOORDS.keys() if b != 0]

# 3) Nodos
max_pd = max(P_D_base.values()) if P_D_base else 1
P_LC_hora = {}

for b in buses_a_dibujar:
    if b not in BUSCOORDS:
        continue
    xb, yb   = BUSCOORDS[b]
    zona_b   = bus_zone.get(b, 1)
    pdb      = P_D_base.get(b, 0)
    sin_sum  = P_LC_hora.get(b, 0) > 0.01
    is_slack = (b == 0)

    size_b   = 20 + 22 * pdb / max_pd if max_pd > 0 else 28
    fill_col = "#F97316" if sin_sum else ("#1E3A5F" if is_slack else NODEFILL.get(zona_b, "#E5E7EB"))
    strk_col = "#7C2D12" if sin_sum else ("#F0F9FF" if is_slack else NODESTROKE.get(zona_b, "#374151"))
    txt_col  = "#FFFFFF" if (sin_sum or is_slack) else "#1F2937"

    gen_parts = []
    
    hover_txt = (
        f"<b>Bus {b}</b>" +
        (" · Slack" if is_slack else f" · {ZONE_NAME.get(zona_b, '')}") +
        f"<br>PD base: {pdb:.1f} kW" +
        f"<br>Carga no servida: {P_LC_hora.get(b, 0):.2f} kW" +
        ("<br>" + "<br>".join(gen_parts) if gen_parts else "<br>✓ Servido")
    )

    fig_grafo.add_trace(go.Scatter(
        x=[xb], y=[yb], mode="markers+text",
        marker=dict(size=size_b, color=fill_col,
                    line=dict(color=strk_col, width=2.5), symbol="circle"),
        text=[f"{b}"], textposition="middle center",
        textfont=dict(size=9, color=txt_col, family="Arial Black"),
        showlegend=False,
        hovertemplate=hover_txt + "<extra></extra>",
        name=f"Bus {b}",
    ))
    

# 4) Leyenda
for lbl, col, dsh, ww in [
    ("Línea operativa",  LINE_COLORS["ok"],       "solid", 3),
    ("Línea en falla",   LINE_COLORS["fault"],     "solid", 3),
    ("Tie-line cerrada", LINE_COLORS["tieclosed"], "solid", 2.5),
    ("Tie-line abierta", LINE_COLORS["tieopen"],   "dot",   2),
]:
    fig_grafo.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=col, width=ww, dash=dsh),
        name=lbl, showlegend=True,
    ))
fig_grafo.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(size=12, color="#F97316", line=dict(color="#7C2D12", width=2)),
    name="Bus sin suministro", showlegend=True,
))
label_activos    = "Caso base"

# 5) Layout
fig_grafo.update_layout(
    title=dict(
        text=f"<b>Red CIGRE MV · Hora {hora_grafo}</b>  <span style='font-size:13px;color:#6B7280'>{label_activos}</span>",
        font=dict(size=16, family="Arial"), x=0.01, xanchor="left",
    ),
    xaxis=dict(visible=False, range=[-0.5, 11.5], fixedrange=True),
    yaxis=dict(visible=False, range=[1.5, 17.0], scaleanchor="x", scaleratio=1.0, fixedrange=True),
    height=720, plot_bgcolor="#F9FAFB", paper_bgcolor="#FFFFFF",
    margin=dict(l=0, r=0, t=55, b=90),
    legend=dict(
        orientation="h", yanchor="bottom", y=-0.11, xanchor="left", x=0,
        font=dict(size=12, family="Arial"),
        bgcolor="rgba(255,255,255,0.9)", bordercolor="#E5E7EB", borderwidth=1,
        itemsizing="constant",
    ),
    dragmode=False,
)
fig_grafo.update_yaxes(range=[2.0, 21.5])

#========================
### Grafo Resultados ###
#========================

fig_grafo_res = fig_grafo.copy()

# 3) Nodos
max_pd = max(P_D_base.values()) if P_D_base else 1
for b in BUSES:
    if b not in BUSCOORDS:
        continue
    xb, yb   = BUSCOORDS[b]
    zona_b   = bus_zone.get(b, 1)
    pdb      = P_D_base.get(b, 0)
    sin_sum  = P_LC_hora.get(b, 0) > 0.01
    is_slack = (b == 0)

    size_b   = 20 + 22 * pdb / max_pd if max_pd > 0 else 28
    fill_col = "#F97316" if sin_sum else ("#1E3A5F" if is_slack else NODEFILL.get(zona_b, "#E5E7EB"))
    strk_col = "#7C2D12" if sin_sum else ("#F0F9FF" if is_slack else NODESTROKE.get(zona_b, "#374151"))
    txt_col  = "#FFFFFF" if (sin_sum or is_slack) else "#1F2937"

    gen_parts = []
    if res_hora:
        # DG Solar — g es el índice = bus donde está instalado
        for g, v in res_hora.get("P_DG_sol", {}).items():
            if g == b and v > 0.01:
                gen_parts.append(f"☀ {v:.1f} kW")

        # DG Eólica
        for g, v in res_hora.get("P_DG_eol", {}).items():
            if g == b and v > 0.01:
                gen_parts.append(f"💨 {v:.1f} kW")

        # DG Fósil
        for g, v in res_hora.get("P_DG_fos", {}).items():
            if g == b and v > 0.01:
                gen_parts.append(f"⛽ {v:.1f} kW")

        # ESS — mostrar potencia neta (positiva = descargando)
        for dur in ["2h", "4h", "6h"]:
            for e, v in res_hora.get(f"P_ESS_{dur}", {}).items():
                if e == b and abs(v) > 0.01:
                    simbolo = "🔋↑" if v > 0 else "🔋↓"
                    gen_parts.append(f"{simbolo} {abs(v):.1f} kW ({dur})")

        # MEG — ya usa (k, bb) así que no cambia
        for (k, bb), v in res_hora.get("P_MEG", {}).items():
            if bb == b and v > 0.01:
                gen_parts.append(f"🚛 {v:.1f} kW")

    hover_txt = (
        f"<b>Bus {b}</b>" +
        (" · Slack" if is_slack else f" · {ZONE_NAME.get(zona_b, '')}") +
        f"<br>PD base: {pdb:.1f} kW" +
        f"<br>Carga no servida: {P_LC_hora.get(b, 0):.2f} kW" +
        ("<br>" + "<br>".join(gen_parts) if gen_parts else "<br>✓ Servido")
    )

    fig_grafo_res.add_trace(go.Scatter(
        x=[xb], y=[yb], mode="markers+text",
        marker=dict(size=size_b, color=fill_col,
                    line=dict(color=strk_col, width=2.5), symbol="circle"),
        text=[f"{b}"], textposition="middle center",
        textfont=dict(size=9, color=txt_col, family="Arial Black"),
        showlegend=False,
        hovertemplate=hover_txt + "<extra></extra>",
        name=f"Bus {b}",
    ))

# 4) Leyenda
for lbl, col, dsh, ww in [
    ("Línea operativa",  LINE_COLORS["ok"],       "solid", 3),
    ("Línea en falla",   LINE_COLORS["fault"],     "solid", 3),
    ("Tie-line cerrada", LINE_COLORS["tieclosed"], "solid", 2.5),
    ("Tie-line abierta", LINE_COLORS["tieopen"],   "dot",   2),
]:
    fig_grafo_res.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=col, width=ww, dash=dsh),
        name=lbl, showlegend=True,
    ))
fig_grafo_res.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(size=12, color="#F97316", line=dict(color="#7C2D12", width=2)),
    name="Bus sin suministro", showlegend=True,
))

# 5) Layout
fig_grafo_res.update_layout(
    title=dict(
        text=f"<b>Red CIGRE MV · Hora {hora_grafo}</b>  <span style='font-size:13px;color:#6B7280'>{label_activos}</span>",
        font=dict(size=16, family="Arial"), x=0.01, xanchor="left",
    ),
    xaxis=dict(visible=False, range=[-0.5, 11.5], fixedrange=True),
    yaxis=dict(visible=False, range=[1.5, 17.0], scaleanchor="x", scaleratio=1.0, fixedrange=True),
    height=720, plot_bgcolor="#F9FAFB", paper_bgcolor="#FFFFFF",
    margin=dict(l=0, r=0, t=55, b=90),
    legend=dict(
        orientation="h", yanchor="bottom", y=-0.11, xanchor="left", x=0,
        font=dict(size=12, family="Arial"),
        bgcolor="rgba(255,255,255,0.9)", bordercolor="#E5E7EB", borderwidth=1,
        itemsizing="constant",
    ),
    dragmode=False,
)