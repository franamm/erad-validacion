# ============================================================
# grafico_curva_potencia.py — Figura de la curva de potencia eólica
# ============================================================
# Dos paneles que comparten el eje de velocidad de viento:
#
#   (a) la curva de potencia que usa `construir_perfiles.cf_eolico`
#   (b) dónde cae el viento REAL del estudio: todas las horas del año
#       contra las horas en que efectivamente falla alguna línea
#
# El panel (b) es el que justifica al (a). No es un eje doble —son dos
# paneles apilados sobre el mismo eje x— porque superponer una densidad y
# un factor de planta en dos escalas verticales distintas es exactamente
# la figura que no se puede leer.
#
# USO
#   python grafico_curva_potencia.py
#   python grafico_curva_potencia.py --salida "../../../Tesis_Francisca_Muñoz_M/Figuras/Metodologia/curva_potencia.png"
#   python grafico_curva_potencia.py --formato pdf --dpi 300
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import construir_faultlin as CF
import construir_perfiles as CP

APP_DIR = Path(__file__).resolve().parent
from rutas import DATOS_DIR  # datos del caso activo (ver rutas.py)

# Paleta validada (light): slots categóricos 1 y 2 de la paleta de referencia.
# Verificada con validate_palette.js — ΔE CVD 24.7, normal 33.6, contraste ≥3:1.
AZUL = "#2a78d6"      # serie 1 — la curva, y "todas las horas"
NARANJA = "#eb6834"   # serie 2 — "horas con falla"
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_MUTED = "#898781"
EJE = "#c3c2b7"


def curva(v_buje):
    """CF en función de la velocidad A LA ALTURA DEL BUJE.

    `cf_eolico` recibe viento a 10 m y escala internamente; acá se quiere
    graficar contra la velocidad que ve la máquina, así que se deshace ese
    escalamiento en la entrada en vez de reimplementar la curva.
    """
    v10 = np.asarray(v_buje, float) / ((CP.ALTURA_BUJE_M / CP.ALTURA_MEDICION_M)
                                       ** CP.EXP_HELLMANN)
    return CP.cf_eolico(v10)


def datos_viento(horas, offset, escenario):
    """Viento a la altura del buje y máscara de falla, ambos POR ZONA.

    La máscara es (hora, zona), no (hora,): marca que falló una línea *de esa
    zona*, no que falló alguna en el sistema.

    La diferencia no es sutil. El viento extremo es local: cuando la zona 2
    tiene tormenta y se le caen las líneas, las zonas 1 y 3 suelen estar en
    calma. Con una máscara por hora, cada tormenta arrastra también esas dos
    zonas en calma al histograma — el efecto sale 1,5x cuando en realidad es
    3,5x, y aparece un modo espurio en vientos bajos que no son fallas con
    poco viento sino vecinas que no fallaron.
    """
    import pickle

    w12 = CF.cargar_viento(CF.RUTA_WIND_NC, horas, offset, escenario)
    w10 = w12 / ((CF.ALTURA_POSTE_M / CF.ALTURA_MEDICION_M) ** CF.EXP_HELLMANN)
    w_buje = w10 * ((CP.ALTURA_BUJE_M / CP.ALTURA_MEDICION_M) ** CP.EXP_HELLMANN)

    fallas = np.asarray(np.load(CF.DATOS_DIR / "FaultlinMC.npy", mmap_mode="r")[:horas])
    with open(CF.ART_DIR / "red.pkl", "rb") as f:
        red = pickle.load(f)
    net, LINES, bus_zone = red["net"], red["LINES"], red["bus_zone"]
    zona_linea = np.array(
        [int(bus_zone.get(int(net.line.from_bus[l]), 1)) for l in LINES]) - 1

    falla_zona = np.zeros(w_buje.shape, dtype=bool)
    for z in range(w_buje.shape[1]):
        cols = np.flatnonzero(zona_linea == z)
        if cols.size:
            falla_zona[:, z] = fallas[:, :, cols].any(axis=(1, 2))
    return w_buje, falla_zona


def figura(w_buje, h_falla, titulo=True):
    import matplotlib.pyplot as plt

    v = np.linspace(0, 30, 601)
    cf = curva(v)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.15], "hspace": 0.16})
    fig.patch.set_facecolor(SUPERFICIE)

    # ── (a) la curva ────────────────────────────────────────
    ax1.set_facecolor(SUPERFICIE)
    ax1.plot(v, cf, color=AZUL, linewidth=2.0, solid_capstyle="round", zorder=3)

    # Los tres regímenes, marcados en tinta y no en color de serie. Etiqueta
    # de una línea: con dos renglones chocan contra el título del panel.
    for x, txt in ((CP.V_CUT_IN, f"cut-in {CP.V_CUT_IN:.0f}"),
                   (CP.V_RATED, f"nominal {CP.V_RATED:.0f}"),
                   (CP.V_CUT_OUT, f"cut-out {CP.V_CUT_OUT:.0f} m/s")):
        ax1.axvline(x, color=EJE, linewidth=1.0, linestyle=(0, (4, 4)), zorder=1)
        ax1.annotate(txt, xy=(x, 1.015), xycoords=("data", "axes fraction"),
                     ha="center", va="bottom", fontsize=8, color=TINTA_MUTED)

    ax1.annotate("P ∝ v³", xy=(6.4, 0.42), fontsize=9.5, color=TINTA_2,
                 ha="right", va="center", style="italic")
    ax1.annotate("potencia nominal", xy=(18.5, 0.93), fontsize=9,
                 color=TINTA_2, ha="center", va="top")

    ax1.set_ylim(-0.04, 1.10)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_ylabel("Factor de planta [–]", fontsize=9.5, color=TINTA_2)
    ax1.set_title("Curva de potencia del aerogenerador",
                  fontsize=11.5, color=TINTA, loc="left", pad=24,
                  fontweight="medium")

    # ── (b) dónde cae el viento real ────────────────────────
    ax2.set_facecolor(SUPERFICIE)
    bins = np.arange(0, 30.5, 1.0)

    # Dos rellenos superpuestos con alpha se mezclan en un tercer color que no
    # es ninguna de las dos series: la identidad se pierde justo donde las
    # distribuciones se cruzan, que es lo que hay que leer. Se separan por
    # forma en vez de por transparencia — el contexto va relleno y recesivo,
    # la serie que cuenta la historia va como contorno grueso.
    todas = w_buje.ravel()
    con_falla = w_buje[h_falla].ravel()

    ax2.hist(todas, bins=bins, weights=np.full(todas.size, 100.0 / todas.size),
             color=AZUL, alpha=0.30, label="Todas las horas del año",
             edgecolor=SUPERFICIE, linewidth=0.8, zorder=2)
    ax2.hist(con_falla, bins=bins,
             weights=np.full(con_falla.size, 100.0 / con_falla.size),
             histtype="step", color=NARANJA, linewidth=2.0,
             label="Horas con falla en esa zona", zorder=3)

    ax2.set_xlim(0, 30)
    ax2.set_xlabel("Velocidad de viento a la altura del buje "
                   f"({CP.ALTURA_BUJE_M:.0f} m) [m/s]", fontsize=9.5, color=TINTA_2)
    ax2.set_ylabel("% de horas", fontsize=9.5, color=TINTA_2)
    ax2.set_title("Distribución del viento del escenario climático",
                  fontsize=10, color=TINTA, loc="left", pad=8)
    leg = ax2.legend(frameon=False, fontsize=9, loc="upper right",
                     handlelength=1.2, handleheight=0.9)
    for t in leg.get_texts():
        t.set_color(TINTA_2)

    # ── ejes recesivos, en los dos paneles ──────────────────
    for ax in (ax1, ax2):
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(EJE)
            ax.spines[lado].set_linewidth(1.0)
        ax.tick_params(colors=TINTA_MUTED, labelsize=8.5, length=3, width=1.0)
        ax.grid(axis="y", color=EJE, linewidth=0.7, alpha=0.55, zorder=0)
        ax.set_axisbelow(True)

    return fig


def main():
    ap = argparse.ArgumentParser(
        description="Figura de la curva de potencia eólica y del viento del estudio")
    ap.add_argument("--salida", default=None,
                    help="ruta del archivo (default: curva_potencia_eolica.<formato> "
                         "junto a este script)")
    ap.add_argument("--formato", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--horas", type=int, default=8760)
    ap.add_argument("--anio-offset", dest="offset", type=int,
                    default=CF.ANIO_OFFSET_DEFAULT)
    ap.add_argument("--escenario-clima", default="baseline",
                    choices=["baseline", "with_EWE"])
    args = ap.parse_args()

    w_buje, h_falla = datos_viento(args.horas, args.offset, args.escenario_clima)
    cf_todas, cf_falla = curva(w_buje).mean(), curva(w_buje[h_falla]).mean()
    print(f"[grafico] viento {w_buje.shape} | zona-hora con falla en esa zona: "
          f"{int(h_falla.sum())} de {h_falla.size}")
    print(f"[grafico] viento medio [m/s buje]: todas {w_buje.mean():.1f} | "
          f"con falla {w_buje[h_falla].mean():.1f}")
    print(f"[grafico] CF eólico: todas {cf_todas:.3f} | con falla {cf_falla:.3f} "
          f"-> {cf_falla / cf_todas:.2f}x")

    fig = figura(w_buje, h_falla)
    salida = Path(args.salida) if args.salida else (
        DATOS_DIR / f"curva_potencia_eolica.{args.formato}")
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=args.dpi, bbox_inches="tight",
                facecolor=SUPERFICIE)
    print(f"[grafico] → {salida}")


if __name__ == "__main__":
    main()
