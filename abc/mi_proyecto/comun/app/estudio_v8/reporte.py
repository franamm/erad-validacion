"""Reporte del estudio: modelo neutro del documento, figuras y armado de secciones.

`construir(D, carpeta_figuras)` devuelve una lista de bloques que `reporte_docx` y
`reporte_pdf` dibujan igual; así Word y PDF no pueden divergir. `D` es la salida de
`reporte_datos.reunir`. Las figuras son matplotlib (mismo estilo que las fichas del
caso) para que el reporte no dependa de un navegador.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from estudio_v8 import reporte_textos as T
from estudio_v8.reporte_datos import NOMBRE_MEDIDA
from estudio_v8.reporte_textos import P, miles, pct, usd, mwh, km

# ── Modelo del documento ────────────────────────────────────────────────────


@dataclass
class Bloque:
    tipo: str                      # titulo | parrafo | vinetas | tabla | figura | destacado | salto | portada
    texto: object = None           # str, párrafo (lista de segmentos) o lista de párrafos
    nivel: int = 1
    tabla: pd.DataFrame = None
    alinear: list = field(default_factory=list)   # "l"/"r" por columna
    ruta: Path = None
    ancho_cm: float = 16.0
    pie: str = ""
    meta: dict = field(default_factory=dict)


def titulo(t, n=1):
    return Bloque("titulo", t, nivel=n)


def parrafo(p):
    return Bloque("parrafo", p if isinstance(p, list) else P(p))


def vinetas(ps):
    return Bloque("vinetas", ps)


def tabla(df, pie="", alinear=None):
    al = alinear or ["l"] + ["r"] * (df.shape[1] - 1)
    return Bloque("tabla", tabla=df, alinear=al, pie=pie)


def figura(ruta, pie, ancho_cm=16.0):
    return Bloque("figura", ruta=Path(ruta), pie=pie, ancho_cm=ancho_cm)


def destacado(p):
    return Bloque("destacado", p if isinstance(p, list) else P(p))


# ── Figuras ─────────────────────────────────────────────────────────────────

COLOR_ZONA = ["#3B82F6", "#F59E0B", "#10B981", "#8B5CF6", "#EF4444"]
COLOR_BASE_H, COLOR_PLAN_H = "#C0504D", "#4F81BD"
TINTA, TINTA_2, MUTE, GRID = "#0b0b0b", "#52514e", "#898781", "#e3e2de"
COLOR_MEDIDA = {"reconfig_topo": "#2ECC71", "dg_sol": "#F1C40F", "meg_arriendo": "#00695C",
                "refuerzo": "#7F8C8D", "soterramiento": "#4A3523", "edrp": "#D62728", "meg": "#E67E22",
                "dg_fos": "#8C564B", "dg_eol": "#3498DB", "mess": "#17BECF", "ess_4h": "#8E44AD",
                "ess_5h": "#B5B82A", "ess_6h": "#E377C2"}


def _mpl():
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.edgecolor": MUTE, "axes.labelcolor": TINTA_2,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 7.5})
    return plt


def _fmt_miles(dec=0):
    import matplotlib.ticker as mt
    return mt.FuncFormatter(lambda v, _: miles(v, dec))


def _guardar(fig, ruta):
    fig.savefig(ruta, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return ruta


def fig_zona(zonal, ruta, etiqueta=None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    x = np.arange(len(zonal)); w = 0.38
    con = "Con medida" in zonal
    for i in range(len(zonal)):
        c = COLOR_ZONA[i % len(COLOR_ZONA)]
        b = zonal["Caso base"].iloc[i] / 1000
        xb = x[i] - (w / 2 if con else 0)
        ax.bar(xb, b, w * 0.95, color=c, alpha=0.30, hatch="////", edgecolor=c, linewidth=0.6, zorder=3)
        ax.annotate(miles(b, 1), (xb, b), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=TINTA_2)
        if con:
            m = zonal["Con medida"].iloc[i] / 1000
            ax.bar(x[i] + w / 2, m, w * 0.95, color=c, alpha=0.9, zorder=3)
            ax.annotate(miles(m, 1), (x[i] + w / 2, m), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(x, [z.replace("Zona ", "ZG") for z in zonal["Zona"]])
    ax.set_ylabel("ENS [MWh/año]")
    alto = max(zonal["Caso base"].max(), zonal["Con medida"].max() if con else 0) / 1000
    ax.set_ylim(0, alto * 1.38); ax.grid(True, axis="y"); ax.set_axisbelow(True)
    import matplotlib as mpl
    h = [mpl.patches.Patch(facecolor="#bbbbbb", alpha=0.4, hatch="////", edgecolor="#888", label="Caso base")]
    if con:
        h.append(mpl.patches.Patch(facecolor="#777", label=etiqueta or "Con medidas"))
    ax.legend(handles=h, loc="upper center", ncol=2)
    return _guardar(fig, ruta)


def fig_hora(horario, ruta, etiqueta=None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.plot(horario["Hora"], horario["Caso base"], color=COLOR_BASE_H, lw=0.7, label="Caso base")
    if "Con medida" in horario:
        ax.plot(horario["Hora"], horario["Con medida"], color=COLOR_PLAN_H, lw=0.7, label=etiqueta or "Con medidas")
    ax.set_xlabel("Hora del año"); ax.set_ylabel("ENS [kWh]")
    ax.yaxis.set_major_formatter(_fmt_miles()); ax.xaxis.set_major_formatter(_fmt_miles())
    cols = [c for c in ("Caso base", "Con medida") if c in horario]
    ax.set_ylim(0, horario[cols].max().max() * 1.3)
    ax.grid(True, axis="y"); ax.set_axisbelow(True)
    ax.legend(loc="upper center", ncol=2)
    return _guardar(fig, ruta)


MAC = {"ens": ("d_ens", "usd_kwh", 1e-3, "Reducción de ENS [MWh/año]", "Costo por ENS evitada [USD/kWh]",
               lambda v: miles(v, 2), "MWh"),
       "saidi": ("d_saidi", "usd_saidi", 1, "Reducción de SAIDI [h/cliente]", "Costo por reducción de SAIDI\n[miles de USD/(h/cliente)]",
                 lambda v: miles(v / 1000, 1), "h/cliente"),
       "saifi": ("d_saifi", "usd_saifi", 1, "Reducción de SAIFI [int./cliente]", "Costo por reducción de SAIFI\n[miles de USD/(int./cliente)]",
                 lambda v: miles(v / 1000, 0), "int./cliente")}


def fig_mac(prom, ind, base_val, ruta):
    plt = _mpl()
    col_d, col_c, f, ex, ey, fmt, ub = MAC[ind]
    ok = prom[prom[col_c].notna() & (prom[col_d] > 0)].sort_values(col_c)
    if ok.empty:
        return None
    total = (ok[col_d] * f).sum(); alto = ok[col_c].max()
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x0, ult_x, ult_h, nivel = 0.0, -1e9, -1e9, 0
    for _, r in ok.iterrows():
        w, h = r[col_d] * f, r[col_c]
        ax.bar(x0, h, width=w, align="edge", color=COLOR_MEDIDA.get(r["_caso"], "#999"), edgecolor="white",
               linewidth=0.6, zorder=3, label=r["Medida"])
        cx = x0 + w / 2
        nivel = nivel + 1 if (cx - ult_x) < 0.07 * total and abs(h - ult_h) < 0.08 * alto else 0
        ax.annotate(fmt(h), (cx, h), xytext=(0, 2 + 9 * nivel), textcoords="offset points", ha="center", fontsize=7)
        ult_x, ult_h = cx, h; x0 += w
    x_max = max(x0, base_val) * 1.02
    ax.axvline(base_val, color=TINTA_2, ls=(0, (2, 2)), lw=0.9, zorder=4)
    der = base_val > 0.7 * x_max
    ax.text(base_val, alto * 1.22, (f"Caso base: {miles(base_val, 2)} {ub} " if der else f" Caso base: {miles(base_val, 2)} {ub}"),
            fontsize=7, color=TINTA_2, ha="right" if der else "left", va="top")
    ax.set_ylim(0, alto * 1.25); ax.set_xlim(0, x_max)
    ax.set_xlabel(ex); ax.set_ylabel(ey)
    ax.xaxis.set_major_formatter(_fmt_miles(2 if ind == "saifi" else 0))
    ax.yaxis.set_major_formatter(__import__("matplotlib.ticker", fromlist=["x"]).FuncFormatter(lambda v, _: fmt(v)))
    ax.grid(True, axis="y"); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    return _guardar(fig, ruta)


def fig_solapamiento(C, ruta):
    plt = _mpl()
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for ax, (k, nom, f, dec) in zip(axs, (("ens", "ENS [MWh]", 1e-3, 0), ("saidi", "SAIDI [h/cliente]", 1, 1),
                                         ("saifi", "SAIFI [int./cliente]", 1, 2))):
        s = C["solapamiento"][k]
        izq = 0.0
        for m, v in s["aportes"].items():
            if pd.notna(v) and v > 1e-9:
                ax.barh(1, v * f, left=izq, color=COLOR_MEDIDA.get(m, "#999"), edgecolor="white", lw=0.5)
                izq += v * f
        ax.barh(0, s["conjunto"] * f, color="#1F2937")
        ax.set_yticks([0, 1], ["Juntas", "Suma de\nsueltas"])
        ax.set_title(nom, fontsize=8.5)
        ax.annotate(miles(s["conjunto"] * f, dec), (s["conjunto"] * f, 0), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7)
        ax.annotate(miles(izq, dec), (izq, 1), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7)
        ax.set_xlim(0, izq * 1.25); ax.grid(True, axis="x"); ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(_fmt_miles(dec))
    import matplotlib as mpl
    activos = [m for m in C["solapamiento"]["ens"]["aportes"].index if C["solapamiento"]["ens"]["aportes"].get(m, 0) > 1e-9]
    h = [mpl.patches.Patch(color=COLOR_MEDIDA.get(m, "#999"), label=NOMBRE_MEDIDA.get(m, m)) for m in activos]
    h.append(mpl.patches.Patch(color="#1F2937", label="Abatimiento conjunto"))
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    return _guardar(fig, ruta)


def fig_matriz(M, etiqueta, num_esc, ruta, recomendado=None):
    plt = _mpl()
    Z = M.to_numpy(float) / 1000
    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(M) + 1.4))
    im = ax.imshow(Z, cmap="RdYlGn_r", aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isfinite(Z[i, j]):
                ax.text(j, i, miles(Z[i, j]), ha="center", va="center", fontsize=6.5,
                        fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(M.shape[1]), [num_esc[e] for e in M.columns])
    ax.set_yticks(range(M.shape[0]), [etiqueta[p] for p in M.index])
    if recomendado is not None and recomendado in M.index:
        i = list(M.index).index(recomendado)
        ax.add_patch(__import__("matplotlib.patches", fromlist=["x"]).Rectangle((-0.5, i - 0.5), Z.shape[1], 1,
                                                                                fill=False, ec=TINTA, lw=1.4))
    ax.set_xlabel("Escenario de evaluación"); ax.set_ylabel("Plan")
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Costo del plan [miles de USD/año]", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    return _guardar(fig, ruta)


def fig_pareada(base, plan, num_esc, etiqueta, ruta):
    plt = _mpl()
    escs = list(base.index)
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(len(escs)); w = 0.38
    ax.bar(x - w / 2, base.to_numpy() / 1000, w, color="#9CA3AF", label="Caso base", zorder=3)
    ax.bar(x + w / 2, plan.to_numpy() / 1000, w, color=COLOR_PLAN_H, label=etiqueta, zorder=3)
    for i, (b, p) in enumerate(zip(base.to_numpy(), plan.to_numpy())):
        ax.annotate(f"−{miles(100 * (1 - p / b), 0)} %", (x[i] + w / 2, p / 1000), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=6.5)
    ax.set_xticks(x, [num_esc[e] for e in escs]); ax.set_xlabel("Escenario")
    ax.set_ylabel("ENS [MWh/año]"); ax.grid(True, axis="y"); ax.set_axisbelow(True)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.15))
    return _guardar(fig, ruta)


def fig_mapa(inversion, operacion, fallas_linea, ruta, n_esc):
    """Mapa de la red (coordenadas de las barras) con la inversión y la operación de un plan.

    Líneas por zona; en negro discontinuo, las que fallan en el caso base (grosor = fallas).
    None si la red no trae coordenadas."""
    import json
    import matplotlib as mpl
    from estudio_v8.reporte_datos import _carga
    plt = _mpl()
    red = _carga("red.pkl")
    net = red["net"]
    if "geo" not in net.bus:
        return None
    try:
        xy = {int(b): tuple(json.loads(g)["coordinates"][:2]) for b, g in net.bus["geo"].items()}
    except (TypeError, ValueError, KeyError):
        return None
    zona = {int(b): int(z) for b, z in red["bus_zone"].items()}
    ties = {int(l) for l in red.get("tie_indices", [])}
    seg = {int(l): (xy[int(net.line.at[l, "from_bus"])], xy[int(net.line.at[l, "to_bus"])]) for l in net.line.index}
    lat = np.mean([p[1] for p in xy.values()])
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    for l, (a, b) in seg.items():
        if l in ties:
            continue
        ax.plot([a[0], b[0]], [a[1], b[1]], color=COLOR_ZONA[(zona[int(net.line.at[l, 'from_bus'])] - 1) % 5],
                lw=1.0, alpha=0.5, zorder=2)
    fmax = max(fallas_linea.values()) if fallas_linea else 1
    for l, c in fallas_linea.items():
        if l in seg:
            (a, b) = seg[l]
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#111", lw=0.4 + 2.0 * c / fmax, ls=(0, (2.4, 1.4)), alpha=0.8, zorder=3)
    for nom, color in (("Refuerzo estructural", "#B91C1C"), ("Soterramiento", "#7c4a1e")):
        for l, _ in (inversion.get(nom) or {}).get("elementos", []):
            (a, b) = seg[int(l)]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=3.2, solid_capstyle="round", zorder=4)
            ax.scatter([(a[0] + b[0]) / 2], [(a[1] + b[1]) / 2], s=28, color=color, zorder=4)
    for nom, color in (("GD solar", "#FDE047"), ("GD eólica", "#60A5FA"), ("GD fósil", "#8c564b")):
        els = (inversion.get(nom) or {}).get("elementos", [])
        for bb, kw in els:
            ax.scatter(*xy[int(bb)], s=10 + kw / 5, facecolor=color, edgecolor=TINTA, lw=0.4, zorder=6, alpha=0.9)
        for bb, kw in els[:4]:
            ax.annotate(f"B{bb}", xy[int(bb)], xytext=(4, 3), textcoords="offset points", fontsize=6.5, zorder=8)
    for bb, c in operacion.get("meg_barras", {}).items():
        ax.scatter(*xy[int(bb)], s=40, marker="D", facecolor="#e67e22", edgecolor=TINTA, lw=0.5, zorder=7)
    for l, c in operacion.get("enlaces", {}).items():
        (a, b) = seg[int(l)]
        col = "#08306b" if c >= 0.8 * n_esc else "#3a7fd0" if c >= 0.4 * n_esc else "#a9cdef"
        ax.scatter([(a[0] + b[0]) / 2], [(a[1] + b[1]) / 2], s=22, color=col, edgecolor="white", lw=0.8, zorder=7)
        if c >= 0.8 * n_esc:
            ax.annotate(f"E{l}", ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), xytext=(-4, 4), textcoords="offset points",
                        fontsize=6.5, ha="right", fontweight="bold", zorder=8)
    ax.set_aspect(1 / np.cos(np.radians(lat))); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    L = mpl.lines.Line2D
    h = [L([], [], color="#111", lw=1.4, ls=(0, (2.4, 1.4)), label="Línea que falla (grosor: n.º de fallas)"),
         L([], [], marker="o", ls="", markerfacecolor="#FDE047", markeredgecolor=TINTA, label="GD solar (tamaño: kW)"),
         L([], [], marker="o", ls="", markerfacecolor="#8c564b", markeredgecolor=TINTA, label="GD fósil"),
         L([], [], color="#B91C1C", lw=3, label="Refuerzo"), L([], [], color="#7c4a1e", lw=3, label="Soterramiento"),
         L([], [], marker="D", ls="", markerfacecolor="#e67e22", markeredgecolor=TINTA, label="Barra donde opera el MEG"),
         L([], [], marker="o", ls="", markerfacecolor="#08306b", markeredgecolor="white", label="Enlace cerrado (≥ 80 % esc.)"),
         L([], [], marker="o", ls="", markerfacecolor="#a9cdef", markeredgecolor="white", label="Enlace cerrado (menos)")]
    h += [L([], [], color=COLOR_ZONA[z - 1], lw=1.4, label=f"Red ZG{z}") for z in sorted(set(zona.values()))]
    ax.legend(handles=h, loc="upper left", fontsize=6.5, frameon=True, framealpha=0.9, edgecolor="none")
    return _guardar(fig, ruta)


# ── Tablas ──────────────────────────────────────────────────────────────────

def _f(df, formatos):
    out = df.copy()
    for c, fmt in formatos.items():
        if c in out:
            out[c] = out[c].map(lambda v: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt(v))
    return out


def tabla_planes(D):
    dec, et = D["decision"], D["etiqueta"]
    filas = []
    for p in dec["planes"]:
        inv = dec["inversiones"][p]
        g = lambda k: inv.get(k, {}).get("total", 0.0)
        filas.append({"Plan": et[p], "Grupo": next((k for k, v in D["grupos"].items() if p in v), "—"),
                      "GD solar [kW]": g("GD solar"), "GD fósil [kW]": g("GD fósil"), "GD eólica [kW]": g("GD eólica"),
                      "Refuerzo [km]": g("Refuerzo estructural"), "Soterramiento [km]": g("Soterramiento"),
                      "MEG nuevos": inv.get("Generador móvil (MEG)", {}).get("nuevos", 0)})
    df = pd.DataFrame(filas)
    df = df[[c for c in df if c in ("Plan", "Grupo") or (df[c] != 0).any()]]
    if not D["grupos"]:
        df = df.drop(columns="Grupo")
    return _f(df, {"GD solar [kW]": lambda v: miles(v), "GD fósil [kW]": lambda v: miles(v),
                   "GD eólica [kW]": lambda v: miles(v), "Refuerzo [km]": lambda v: miles(v, 2),
                   "Soterramiento [km]": lambda v: miles(v, 3), "MEG nuevos": lambda v: miles(v)})


def tabla_medias(D, frontera):
    dec, et = D["decision"], D["etiqueta"]
    m = dec["medias"]
    b = D["base"].loc[dec["escenarios"]].mean()
    df = pd.DataFrame({"Plan": [et[p] for p in m.index],
                       "ENS [MWh/año]": m["ens"] / 1000, "ΔENS": 100 * (1 - m["ens"] / b["ens"]),
                       "SAIDI [h]": m["saidi"], "SAIFI [int.]": m["saifi"],
                       "Costo medidas [USD/año]": m["cst_costo_total"], "Costo del plan [USD/año]": m["cst_objetivo_plan"],
                       "Frontera": ["sí" if p in frontera else "" for p in m.index]})
    return _f(df, {"ENS [MWh/año]": lambda v: miles(v, 1), "ΔENS": lambda v: f"−{pct(v)}",
                   "SAIDI [h]": lambda v: miles(v, 2), "SAIFI [int.]": lambda v: miles(v, 2),
                   "Costo medidas [USD/año]": lambda v: miles(v), "Costo del plan [USD/año]": lambda v: miles(v)})


# ── Armado ──────────────────────────────────────────────────────────────────

def construir(D, carpeta):
    from estudio_v8 import pareto as PA
    carpeta = Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
    meta, dec, R = D["meta"], D["decision"], D["recomendado"]
    et, num = D["etiqueta"], D["num_esc"]
    B = []
    nfig = [0]

    def fig(ruta, pie, ancho=16.0):
        if ruta is None:
            return
        nfig[0] += 1
        B.append(figura(ruta, f"Figura {nfig[0]}. {pie}", ancho))

    B.append(Bloque("portada", meta={
        "titulo": "Reporte de resiliencia frente a eventos climáticos extremos",
        "subtitulo": meta["nombre_red"],
        "filas": [("Caso", meta["caso"]), ("Estudio", meta["estudio"]), ("Combinación de medidas", meta["combinacion"]),
                  ("Escenarios", f"{len(D['escenarios'])}"), ("Variante de la red", meta["variante"]),
                  ("Fecha", meta["fecha"]), ("Versión del código", meta["git"])]}))

    # 1. Resumen ejecutivo
    B.append(titulo("1. Resumen ejecutivo"))
    if dec is None:
        B.append(parrafo("El estudio no tiene evaluaciones de plan fijo: no se puede recomendar un plan."))
    else:
        B.append(destacado(T.frase_recomendacion(D)))
        B.append(titulo("Qué instala y cómo opera", 2))
        B.append(vinetas(T.vinetas_inversion(R["inversion"], R["operacion"], len(dec["escenarios"]))))
        B.append(titulo("Por qué este plan", 2))
        B.append(parrafo(T.frase_por_que(D)))
        alt = T.frase_alternativa(D)
        if alt:
            B.append(parrafo(alt))
        B.append(titulo("Advertencias", 2))
        B.append(vinetas(T.advertencias(D)))
    B.append(Bloque("salto"))

    # 2. Red
    B.append(titulo("2. Red y datos de entrada"))
    for p in T.texto_red(D):
        B.append(parrafo(p))
    z = D["red"]["zonas"].copy()
    B.append(tabla(_f(z, {"km": lambda v: miles(v, 1), "Demanda base [kW]": lambda v: miles(v),
                          "Clientes": lambda v: miles(v)}), "Tabla. Red por zona."))

    # 3. Amenaza y escenarios
    B.append(titulo("3. Amenaza y escenarios"))
    for p in T.texto_amenaza(D):
        B.append(parrafo(p))
    b = D["base"].loc[D["escenarios"]]
    te = pd.DataFrame({"N.º": [num[e] for e in D["escenarios"]],
                       "Escenario original": [f"esc_{e:03d}" for e in D["escenarios"]],
                       "Grupo": [next((k for k, v in D["grupos"].items() if e in v), "—") for e in D["escenarios"]],
                       "Fallas": [D["fallas_esc"].get(e, np.nan) for e in D["escenarios"]],
                       "ENS base [MWh]": b["ens"].to_numpy() / 1000, "SAIDI base [h]": b["saidi"].to_numpy(),
                       "SAIFI base": b["saifi"].to_numpy()})
    if not D["grupos"]:
        te = te.drop(columns="Grupo")
    B.append(tabla(_f(te, {"Fallas": lambda v: miles(v), "ENS base [MWh]": lambda v: miles(v, 1),
                           "SAIDI base [h]": lambda v: miles(v, 2), "SAIFI base": lambda v: miles(v, 2)}),
                   "Tabla. Escenarios del estudio y su severidad sin medidas."))

    # 4. Supuestos
    B.append(titulo("4. Supuestos económicos y de modelación"))
    c, s = D["costos"], D["supuestos"]
    sup = [("Valor de la energía no suministrada (VoLL)", f"{miles(c['voll'], 0)} USD/kWh"),
           ("Tasa de descuento y vida útil", f"{pct(100 * c['tasa'], 0)}, {c['anios']} años (anualidad {miles(c['frp'], 4)})"),
           ("Costo de la energía de la subestación", f"{miles(c['c_sub'], 3)} USD/kWh"),
           ("Horizonte de cada escenario", f"{miles(s.get('horas') or meta['horas'])} horas"),
           ("Reparación de una línea dañada", f"{D['amenaza'].get('horas_reparacion') or '—'} horas"),
           ("Tope de generación distribuida", f"{pct(100 * (s.get('lim_dg') or 0), 0)} de la demanda por {s.get('lim_dg_modo') or 'red'}"),
           ("Generadores móviles", f"{s.get('meg_existentes') or 0} existente(s), hasta {s.get('n_meg') or '—'} unidades de "
                                   f"{miles(s.get('P_meg_max') or 0)} kW, traslado {s.get('t_tra_meg') or '—'} h"),
           ("Medidas habilitadas", ", ".join(NOMBRE_MEDIDA.get(m, m) for m in (s.get("medidas_on") or [])))]
    if c.get("tipo_cambio"):
        sup.append(("Tipo de cambio", f"{miles(c['tipo_cambio'], 2)} CLP/USD"))
    B.append(tabla(pd.DataFrame(sup, columns=["Supuesto", "Valor"]), "Tabla. Supuestos del estudio.", ["l", "l"]))
    tc = c["tabla"].copy()
    tc["CAPEX"] = [f"{miles(v)} {u}" if u != "—" else "—" for v, u in zip(tc["CAPEX"], tc["Unidad CAPEX"])]
    tc["OPEX"] = [f"{miles(v, 2)} {u}" for v, u in zip(tc["OPEX"], tc["Unidad OPEX"])]
    B.append(tabla(tc[["Medida", "CAPEX", "OPEX", "Nota"]], "Tabla. Costos unitarios por medida"
                   + (f" ({c['fuente'].split(',')[0]})." if c.get("fuente") else "."), ["l", "r", "r", "l"]))

    # 5. Caso base
    B.append(titulo("5. Caso base: la red sin medidas"))
    for p in T.texto_base(D):
        B.append(parrafo(p))
    if D["base_zonal"] is not None:
        fig(fig_zona(D["base_zonal"], carpeta / "base_zona.png"), "ENS esperada del caso base por zona.", 11)
        fig(fig_hora(D["base_horario"], carpeta / "base_hora.png"), "ENS esperada del caso base en el año.", 15)

    # 6. Medidas individuales
    I = D["individuales"]
    if I is not None:
        B.append(titulo("6. Medidas individuales"))
        for p in T.texto_individuales(D):
            B.append(parrafo(p))
        pr = I["prom"].sort_values("usd_kwh", na_position="last")
        ti = pd.DataFrame({"Medida": pr["Medida"], "ΔENS [MWh]": pr["d_ens"] / 1000, "ΔENS": 100 * pr["d_ens_pct"],
                           "Costo [USD/año]": pr["cst_costo_total"], "USD/kWh": pr["usd_kwh"],
                           "USD/(h/cliente)": pr["usd_saidi"], "USD/(int./cliente)": pr["usd_saifi"]})
        B.append(tabla(_f(ti, {"ΔENS [MWh]": lambda v: miles(v, 1), "ΔENS": lambda v: pct(v),
                               "Costo [USD/año]": lambda v: miles(v), "USD/kWh": lambda v: miles(v, 2),
                               "USD/(h/cliente)": lambda v: miles(v), "USD/(int./cliente)": lambda v: miles(v)}),
                       f"Tabla. Medidas individuales: mejora y costo por unidad de mejora (media de "
                       f"{len(I['escenarios'])} escenarios)."))
        bb = D["base"].loc[I["escenarios"]].mean()
        for k, nom in (("ens", "ENS"), ("saidi", "SAIDI"), ("saifi", "SAIFI")):
            base_val = bb[k] / 1000 if k == "ens" else bb[k]
            fig(fig_mac(I["prom"], k, base_val, carpeta / f"mac_{k}.png"), f"Curva de costos de abatimiento de {nom}.")

    # 7. Combinación
    C = D["combinacion"]
    if C is not None:
        B.append(titulo("7. Combinación de medidas: los efectos no se suman"))
        B.append(parrafo(P(f"Con todas las medidas habilitadas a la vez, cada medida se usa menos que sola porque "
                           f"compiten por las mismas fallas. Comparación sobre {len(C['escenarios'])} escenarios:")))
        B.append(vinetas(T.texto_combinacion(D)))
        fig(fig_solapamiento(C, carpeta / "solapamiento.png"),
            "Suma de los abatimientos de las medidas evaluadas por separado frente al abatimiento conjunto.")
    B.append(Bloque("salto"))

    if dec is not None:
        # 8. Planes
        B.append(titulo("8. Planes candidatos"))
        for p in T.texto_planes(D):
            B.append(parrafo(p))
        B.append(tabla(tabla_planes(D), "Tabla. Inversión de cada plan candidato."))
        prom = dec["medias"].copy()
        base_p = D["base"].loc[dec["escenarios"]].mean()
        fr, _ = PA.frontera_pareto(prom, ["ens", "cst_costo_total"])
        fr = fr.sort_values("cst_costo_total")
        B.append(tabla(tabla_medias(D, list(fr.index)), "Tabla. Valor esperado de cada plan en todos los escenarios."))
        B.append(titulo("Fronteras de Pareto", 2))
        B.append(parrafo(T.texto_pareto(D, list(fr.index))))
        # Panel "función objetivo" = costo del plan (medidas + ENS); el caso base solo tiene el costo de su ENS.
        base_p["cst_costo_total"] = 0.0
        base_p["objetivo"] = base_p["costo_ens"]
        gr = D["grupo_de"] or None
        f_ = PA.figura_pareto(prom.assign(objetivo=prom["cst_objetivo_plan"]),
                              base_p, et, grupo=gr, nombres_grupo=list(D["grupos"]) or None,
                              titulo=f"Valor esperado de los {len(dec['escenarios'])} escenarios")
        fig(_guardar(f_, carpeta / "pareto_esperado.png"), "Fronteras de Pareto de los planes en valor esperado.")
        for g, escs in D["grupos"].items():
            pg = PA.promediar_planes(dec["larga"], escs, ("ens", "saidi", "saifi", "cst_objetivo_plan", "cst_costo_total"))
            bg = D["base"].loc[escs].mean(); bg["cst_costo_total"] = 0.0; bg["objetivo"] = bg["costo_ens"]
            frg, _ = PA.frontera_pareto(pg, ["ens", "cst_costo_total"])
            B.append(parrafo(T.texto_pareto(D, list(frg.sort_values("cst_costo_total").index), g)))
            f_ = PA.figura_pareto(pg.assign(objetivo=pg["cst_objetivo_plan"]), bg, et, grupo=gr,
                                  nombres_grupo=list(D["grupos"]), titulo=f"{g}: {len(escs)} escenarios")
            fig(_guardar(f_, carpeta / f"pareto_{g.replace(' ', '_')}.png"), f"Fronteras de Pareto: {g}.")

        # 9. Reglas
        B.append(Bloque("salto"))
        B.append(titulo("9. Reglas de decisión"))
        for p in T.texto_reglas(D):
            B.append(parrafo(p))
        fig(fig_matriz(dec["M"], et, num, carpeta / "matriz.png", R["plan"]),
            "Costo del plan (filas) evaluado en cada escenario (columnas), en miles de USD/año. En negrita, cada plan en su "
            "propio escenario; recuadrado, el plan recomendado.")
        cr = dec["reglas"]["criterios"].copy()
        tcr = pd.DataFrame({"Plan": [et[p] for p in cr.index], "Valor esperado": cr["Valor esperado"],
                            "Peor escenario": cr["Máximo"], "Arrepentimiento máximo": cr["Arrepentimiento máximo"]})
        for col, regla in (("Valor esperado", "Valor esperado"), ("Peor escenario", "Minimax"),
                           ("Arrepentimiento máximo", "Minimax regret")):
            tcr[col] = [miles(v) + (" ✔" if p in dec["elegidos"].get(regla, []) else "") for p, v in zip(cr.index, tcr[col])]
        B.append(tabla(tcr, "Tabla. Criterios de decisión sobre el costo del plan [USD/año]; ✔ = plan elegido por la regla."))
        if np.isfinite(dec["reglas"]["evpi"]):
            B.append(parrafo(P("El valor esperado de la información perfecta (EVPI) es ",
                               (f"{usd(dec['reglas']['evpi'])}/año", True),
                               ": es lo máximo que valdría saber de antemano qué escenario ocurrirá.")))
        B.append(Bloque("salto"))

        # 10. Plan recomendado
        B.append(titulo(f"10. Plan recomendado: {R['etiqueta']}"))
        B.append(destacado(T.frase_recomendacion(D)))
        m, b = T.indicadores_plan(D)
        comp = [("ENS [MWh/año]", b["ens"] / 1000, m["ens"] / 1000, 1), ("SAIDI [h/cliente]", b["saidi"], m["saidi"], 2),
                ("SAIFI [int./cliente]", b["saifi"], m["saifi"], 2), ("CAPEX anual [USD]", 0, m["cst_capex_anual"], 0),
                ("OPEX anual [USD]", 0, m["cst_opex_anual"], 0), ("Costo de la ENS [USD/año]", b["costo_ens"], m["costo_ens"], 0),
                ("Beneficio neto [USD/año]", np.nan, m["cst_ahorro_neto"], 0)]
        filas = []
        alt = R.get("alternativa")
        ma = T.indicadores_plan(D, alt["plan"])[0] if alt else None
        for nom, vb, vp, d_ in comp:
            fila = {"Indicador": nom, "Caso base": miles(vb, d_) if np.isfinite(vb) else "—", R["etiqueta"]: miles(vp, d_)}
            if alt:
                clave = {"ENS [MWh/año]": ("ens", 1e-3), "SAIDI [h/cliente]": ("saidi", 1), "SAIFI [int./cliente]": ("saifi", 1),
                         "CAPEX anual [USD]": ("cst_capex_anual", 1), "OPEX anual [USD]": ("cst_opex_anual", 1),
                         "Costo de la ENS [USD/año]": ("costo_ens", 1), "Beneficio neto [USD/año]": ("cst_ahorro_neto", 1)}[nom]
                fila[f"{alt['etiqueta']} (alternativa)"] = miles(ma[clave[0]] * clave[1], d_)
            filas.append(fila)
        B.append(tabla(pd.DataFrame(filas), f"Tabla. Plan {R['etiqueta']} frente al caso base (valor esperado de "
                                             f"{len(dec['escenarios'])} escenarios)."))
        fig(fig_mapa(R["inversion"], R["operacion"], D["fallas_linea"], carpeta / "mapa_plan.png", len(dec["escenarios"])),
            f"Red con la inversión del plan {R['etiqueta']} y su operación en los {len(dec['escenarios'])} escenarios.", 15)
        inv_det = []
        for nom, x in R["inversion"].items():
            if x["unidad"] == "km":
                els = ", ".join(f"L{el} ({km(v)})" for el, v in x["elementos"])
                total = km(x["total"])
            elif nom.startswith("Generador"):
                els = (f"{x.get('existentes', 0)} existente(s), {x.get('nuevos', 0)} nueva(s)")
                total = f"{miles(x['total'])} kW"
            else:
                els = ", ".join(f"B{el} ({miles(v)})" for el, v in x["elementos"] if v >= 1)
                chicas = sum(1 for _, v in x["elementos"] if v < 1)
                els += f" y {chicas} barras con menos de 1 kW" if chicas else ""
                total = f"{miles(x['total'])} kW"
            inv_det.append({"Medida": nom, "Total": total, "Dónde (cantidad)": els})
        B.append(tabla(pd.DataFrame(inv_det), f"Tabla. Detalle de la inversión del plan {R['etiqueta']}; barras en kW.",
                       ["l", "r", "l"]))
        if R["zonal"] is not None:
            fig(fig_zona(R["zonal"], carpeta / "plan_zona.png", f"Plan {R['etiqueta']}"),
                f"ENS esperada por zona: caso base y plan {R['etiqueta']}.", 11)
            fig(fig_hora(R["horario"], carpeta / "plan_hora.png", f"Plan {R['etiqueta']}"),
                f"ENS esperada en el año: caso base y plan {R['etiqueta']}.", 15)
        pe = R["por_escenario"].reindex(dec["escenarios"])
        fig(fig_pareada(D["base"].loc[dec["escenarios"], "ens"], pe["ens"], num, f"Plan {R['etiqueta']}",
                        carpeta / "plan_pareada.png"), f"ENS por escenario: caso base y plan {R['etiqueta']}.")

    # 11. Limitaciones
    B.append(titulo("11. Alcance y limitaciones"))
    B.append(vinetas([
        P("El modelo representa la operación con flujo de potencia DC horario y fallas por viento; no incluye otras "
          "amenazas ni restricciones de tensión."),
        P("Los costos son anualizados con la tasa y vida útil de la sección 4; los OPEX incluyen la energía comprada a la "
          "subestación frente al caso base, por eso pueden ser negativos."),
        P("Las medidas que el modelo no elige en ningún escenario no se descartan en general: no se justifican con estos "
          "costos, este VoLL y estas fallas."),
        P(f"Evaluaciones excluidas por no conservar la inversión impuesta: {dec['excluidas'] if dec else 0}.")]))

    # Anexos
    if dec is not None:
        from estudio_v8 import analisis as A
        rg = A.rangos_plan(dec["diagonal"], dec["planes"])          # índices de línea, como L164 / E199
        n = len(dec["planes"])
        rg = rg[(rg["Escenarios"] >= max(2, int(np.ceil(0.3 * n)))) & (rg["Unidad"] != "maniobras")]
        if not rg.empty:
            B.append(Bloque("salto"))
            B.append(titulo("Anexo A. Elementos que se repiten entre los planes candidatos"))
            rg = rg.assign(Medida=rg["Medida"].str.replace("DG ", "GD "),
                           Elemento=rg["Elemento"].str.replace("Bus ", "B").str.replace("Línea ", "L")
                           .str.replace("Enlace ", "E").str.replace("Unidad ", "Unidad "),
                           Escenarios=[f"{e} de {n}" for e in rg["Escenarios"]])
            rg = rg.sort_values(["Medida", "Escenarios"], ascending=[True, False])
            B.append(tabla(_f(rg, {"Min": lambda v: miles(v, 1), "Max": lambda v: miles(v, 1), "Media": lambda v: miles(v, 1)}),
                           f"Tabla. Elementos presentes en al menos el 30 % de los {n} planes; mínimo, máximo y media entre "
                           "planes (donde no se instala cuenta como 0). La lista completa está en la pestaña Red y plan "
                           "instalado de la app.", ["l", "l", "l", "r", "r", "r", "r"]))
    return B
