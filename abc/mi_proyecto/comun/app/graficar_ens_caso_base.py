"""Figuras de ENS esperada del CASO BASE (sin ninguna medida).

Es el diagnóstico contra el que se compara todo lo demás: cuánta energía deja
de servirse si no se hace nada, promediada sobre los escenarios del barrido.
La app dibuja lo mismo en la pestaña de valor esperado, pero siempre pegada a
una medida; para la tesis hace falta la figura sola y en archivo.

Reutiliza `graficos_valor_esperado.promediar_ens` / `figuras_ens`, así que la
figura sale IDÉNTICA a la de la app (mismos colores, fuentes y márgenes) y no
hay dos versiones del mismo gráfico que se desincronicen.

Uso típico (los 10 escenarios de la tanda severa):

    python graficar_ens_caso_base.py --tanda "sec-0-2-9-10 severo"

Sin `--tanda` promedia TODOS los baselines de la variante, que no es lo mismo
que la muestra del barrido.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

import pipeline as P
from graficos_valor_esperado import figuras_ens, promediar_ens

AQUI = Path(__file__).resolve().parent
from rutas import DATOS_DIR  # datos del caso activo (ver rutas.py)


def escenarios_de_tanda(tanda):
    """Los escenarios efectivamente corridos en esa tanda del barrido.

    Se leen del `resumen_individual.csv` en vez de listar los `.npz` de una
    medida cualquiera: una medida puede tener corridas de más (reintentos) o
    de menos (cortadas por tiempo), y el resumen es lo que la app considera
    la muestra.
    """
    ruta = P.ART_DIR / "barrido" / tanda / "resumen_individual.csv"
    if not ruta.exists():
        sys.exit(f"No existe {ruta}. Revisar el nombre de la tanda.")
    df = pd.read_csv(ruta)
    return sorted(int(e) for e in df["escenario"].unique())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variante", default="sec-0-2-9-10",
                    help="Supuesto de aislación de los baselines (carpeta en "
                         "artefactos/baselines). Por defecto 'sec-0-2-9-10'.")
    ap.add_argument("--tanda", default=None,
                    help="Tanda del barrido de la que tomar la muestra de "
                         "escenarios (p. ej. 'sec-0-2-9-10 severo').")
    ap.add_argument("--escenarios", default=None,
                    help="Lista explícita, separada por comas, en la "
                         "numeración interna (0-based, la del nombre de archivo).")
    ap.add_argument("--horas", type=int, default=8760)
    ap.add_argument("--salida", default=str(DATOS_DIR / "figuras_caso_base"))
    args = ap.parse_args()

    base_dir = P.base_dir(args.variante)
    if not base_dir.exists():
        sys.exit(f"No hay baselines en {base_dir}. Correr E5 de pipeline.py.")

    if args.escenarios:
        escs = sorted(int(e) for e in args.escenarios.split(","))
    elif args.tanda:
        escs = escenarios_de_tanda(args.tanda)
    else:
        escs = sorted(int(p.stem.split("_")[1]) for p in base_dir.glob("esc_*.npz"))

    rutas = [base_dir / f"esc_{e:03d}.npz" for e in escs]
    faltan = [r.name for r in rutas if not r.exists()]
    if faltan:
        sys.exit(f"Faltan baselines en {base_dir}: {', '.join(faltan)}")

    # El caso base no depende de la red de una medida: BUSES y bus_zone salen
    # del artefacto E0, sin pasar por data_loader (que importa streamlit).
    red = P._cargar("red.pkl")
    buses, bus_zone = red["BUSES"], red["bus_zone"]

    horario, zonal = promediar_ens(None, rutas, buses, bus_zone, args.horas)
    zona, hora = figuras_ens(horario, zonal, None)

    total = float(zonal["Caso base"].sum())
    print(f"Variante '{args.variante}' · {len(escs)} escenarios "
          f"(peso 1/{len(escs)}) · horizonte {args.horas} h")
    print("Escenarios (numeración de la app): "
          + ", ".join(str(e + 1) for e in escs))
    print(f"ENS esperada total: {total:,.0f} kWh/año ({total/1000:,.1f} MWh/año)")
    for z, v in zip(zonal["Zona"], zonal["Caso base"]):
        print(f"  {z}: {v:,.0f} kWh  ({v/total:.1%})" if total else f"  {z}: 0")

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    for fig, tabla, sufijo in ((zona, zonal, "zona"), (hora, horario, "horario")):
        fig.write_html(salida / f"ens_caso_base_{sufijo}.html",
                       include_plotlyjs="cdn")
        tabla.to_csv(salida / f"ens_caso_base_{sufijo}.csv", index=False,
                     encoding="utf-8-sig")
        try:
            fig.write_image(salida / f"ens_caso_base_{sufijo}.png", scale=2)
        except Exception as exc:                            # noqa: BLE001
            # kaleido es opcional: sin él quedan el HTML y el CSV, que es lo
            # que de verdad hace falta para rehacer la figura.
            print(f"  (sin PNG de '{sufijo}': {type(exc).__name__}: {exc})")
    print(f"Escrito en {salida}")


if __name__ == "__main__":
    main()
