"""Frontera de Pareto de los planes candidatos (puro: pandas, numpy, matplotlib).

Portado de `caso_elmelon/datos/graficos_resultados_elmelon_10esc.ipynb` (celdas 15 y 16):
mismos objetivos, mismos paneles y mismos colores, para que la app y el reporte
dibujen lo mismo que la tesis.
"""
import numpy as np
import pandas as pd

C_FRONTERA, C_DOMINADA, C_BASE = "#2a78d6", "#53565B", "#B91C1C"

# (x, y, título, etiqueta x, etiqueta y); columnas de la tabla de planes promediados.
PANELES = [
    ("ens", "objetivo", "ENS vs función objetivo", "ENS [kWh/año]", "Función objetivo [USD/año]"),
    ("ens", "cst_costo_total", "ENS vs costo anual de las medidas", "ENS [kWh/año]", "CAPEX + OPEX [USD/año]"),
    ("saidi", "cst_costo_total", "SAIDI vs costo anual de las medidas", "SAIDI [h/cliente·año]",
     "CAPEX + OPEX [USD/año]"),
    ("saidi", "saifi", "SAIDI vs SAIFI", "SAIDI [h/cliente·año]", "SAIFI [int./cliente·año]"),
]


def frontera_pareto(df, objetivos, rtol=1e-6):
    """(no dominados ordenados por los objetivos, dominados). Todos los objetivos se minimizan."""
    X = df[list(objetivos)].to_numpy(float)
    if len(X) == 0:
        return df, df
    tol = rtol * np.maximum(np.ptp(X, axis=0), 1e-12)
    no_peor = (X[None] <= X[:, None, :] + tol).all(-1)
    mejor = (X[None] < X[:, None, :] - tol).any(-1)
    no_dom = ~(no_peor & mejor).any(1)
    return df[no_dom].sort_values(list(objetivos)), df[~no_dom]


def promediar_planes(larga, escenarios, columnas=("ens", "saidi", "saifi", "objetivo", "cst_costo_total")):
    """Valor esperado (equiprobable) de cada plan sobre `escenarios`; índice = plan."""
    sub = larga[larga["escenario"].isin(escenarios)]
    cols = [c for c in columnas if c in sub]
    completos = sub.groupby("plan")["escenario"].nunique() == len(set(escenarios))
    return sub.groupby("plan")[cols].mean()[completos]


def figura_pareto(prom, base, etiqueta, grupo=None, titulo="", paneles=PANELES, plan_propio=None,
                  nombres_grupo=None):
    """Figura 2×2 con la frontera de cada par de objetivos.

    prom: planes promediados (índice = plan). base: Serie con los mismos indicadores del caso
    sin medidas (costo 0). etiqueta: {plan: "P4"}. grupo: {plan: 0/1…} para el marcador
    (círculo, triángulo…). plan_propio: plan de diseño del escenario (estrella), si aplica."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    marcas = ["o", "^", "s", "D"]
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 8.2))
    for ax, (x, y, tit, ex, ey) in zip(axs.flat, paneles):
        if x not in prom or y not in prom:
            ax.set_visible(False)
            continue
        D = prom[[x, y]].copy()
        fr, dom = frontera_pareto(D, [x, y])
        for d_, color, alpha, s in ((dom, C_DOMINADA, 0.3, 28), (fr, C_FRONTERA, 1.0, 36)):
            for p, fila in d_.iterrows():
                m = marcas[(grupo or {}).get(p, 0) % len(marcas)]
                ax.scatter(fila[x], fila[y], s=s, marker=m, color=color, alpha=alpha, edgecolor="none", zorder=2)
                ax.annotate(etiqueta.get(p, str(p)), (fila[x], fila[y]), textcoords="offset points",
                            xytext=(5, 4), fontsize=7.5, color="#333" if color == C_DOMINADA else C_FRONTERA)
        if plan_propio is not None and plan_propio in D.index:
            ax.scatter(D.loc[plan_propio, x], D.loc[plan_propio, y], s=60, marker="*", color="#1F2937", zorder=3)
        bx, by = float(base.get(x, np.nan)), float(base.get(y, 0.0))
        if np.isfinite(bx):
            ax.scatter(bx, by, s=34, color=C_BASE, zorder=3)
            ax.annotate("Sin medidas", (bx, by), textcoords="offset points", xytext=(-6, 6), ha="right",
                        fontsize=7.5, color=C_BASE)
            union, _ = frontera_pareto(pd.concat([D, pd.DataFrame([[bx, by]], columns=[x, y], index=["_base"])]),
                                       [x, y])
        else:
            union = fr
        if len(union) > 1:
            ax.plot(union[x], union[y], color=C_FRONTERA, lw=1, zorder=1)
        ax.set_title(tit, fontsize=10)
        ax.set_xlabel(ex, fontsize=8.5); ax.set_ylabel(ey, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, color="#e3e2de", lw=0.6); ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.yaxis.set_major_formatter(_miles(y))
        ax.xaxis.set_major_formatter(_miles(x))
    h = [Line2D([], [], marker="o", ls="", color=C_FRONTERA, label="Frontera de Pareto"),
         Line2D([], [], marker="o", ls="", color=C_DOMINADA, alpha=0.4, label="Planes dominados"),
         Line2D([], [], marker="o", ls="", color=C_BASE, label="Caso sin medidas")]
    if grupo and len(set(grupo.values())) > 1:
        for g in sorted(set(grupo.values())):
            nom = nombres_grupo[g] if nombres_grupo and g < len(nombres_grupo) else f"grupo {g + 1}"
            h.append(Line2D([], [], marker=marcas[g % len(marcas)], ls="", color="#555", label=f"Planes del {nom}"))
    if plan_propio is not None:
        h.append(Line2D([], [], marker="*", ls="", color="#1F2937", markersize=9, label="Plan del propio escenario"))
    fig.legend(handles=h, loc="lower center", ncol=min(len(h), 5), fontsize=8, frameon=False)
    if titulo:
        fig.suptitle(titulo, fontsize=11, color="#444")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97 if titulo else 1))
    return fig


def _miles(col):
    import matplotlib.ticker as mt
    dec = 2 if col == "saifi" else (1 if col == "saidi" else 0)
    return mt.FuncFormatter(lambda v, _: f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", "."))
