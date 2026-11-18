"""Exporta figuras 16:9 transparentes para diapositivas, sin recalcular el MILP."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import barrido_vista as BV
from graficos_combinaciones import preparar_comparacion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tanda', default='barrido/sec-0-2-9-10 severo')
    from rutas import DATOS_DIR
    parser.add_argument('--salida', default=str(DATOS_DIR / 'figuras_combinaciones_presentacion'))
    args = parser.parse_args()
    b = BV.cargar_barrido(args.tanda, 0)
    datos = pd.concat([
        b['individual'].assign(_tipo='Individual', _caso=b['individual']['medida']),
        b['combinadas'].assign(_tipo='Combinada', _caso=b['combinadas']['combinacion'])], ignore_index=True)
    if datos['horas'].nunique() != 1:
        raise ValueError('Hay horizontes distintos en la tanda.')
    base = BV.cargar_base(b['variante'], sorted(datos.escenario.unique()), int(datos.horas.iloc[0]))
    datos = BV.enriquecer(datos, base, b['mip_gap'])
    excluidas = int(datos.sospechosa.sum())
    datos = datos[~datos.sospechosa].copy()
    cobertura = datos[datos._tipo == 'Combinada'].groupby('_caso').escenario.nunique()
    elegidas = set(cobertura[cobertura == cobertura.max()].index)
    grupos = list(datos[(datos._tipo == 'Combinada') & datos._caso.isin(elegidas)].groupby('_caso'))
    comunes = sorted(set.intersection(*(set(g.escenario) for _, g in grupos)))
    if not comunes:
        raise ValueError('No hay escenarios comunes entre todas las alternativas válidas.')
    individuales = {c for c, g in datos[datos._tipo == 'Individual'].groupby('_caso')
                    if set(comunes).issubset(set(g.escenario))}
    seleccion = datos[((datos._tipo == 'Combinada') & datos._caso.isin(elegidas)) |
                     ((datos._tipo == 'Individual') & datos._caso.isin(individuales))]
    resumen, aporte = preparar_comparacion(seleccion[seleccion.escenario.isin(comunes)])
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    combos = resumen[resumen.Tipo == 'Combinada'].copy()
    nombres = {
        'dg_sol': 'Solar', 'dg_eol': 'Eólica', 'dg_fos': 'Generación fósil',
        'ess_4h': 'ESS 4 h', 'ess_5h': 'ESS 5 h', 'ess_6h': 'ESS 6 h',
        'reconfig_topo': 'Reconfiguración', 'meg': 'Generación móvil',
        'mess': 'ESS móvil', 'edrp': 'EDRP', 'refuerzo': 'Refuerzo',
        'soterramiento': 'Soterramiento'}
    etiquetas = {}
    for _, c in combos.iterrows():
        miembros = c.Clave.removeprefix('combo_').split('+')
        partes = [nombres.get(m, m) for m in miembros]
        if all(m in miembros for m in ('ess_4h', 'ess_5h', 'ess_6h')):
            partes = [p for p in partes if not p.startswith('ESS ')] + ['ESS 4/5/6 h']
        etiqueta = ' + '.join(partes) if len(miembros) <= 5 else f'Cartera de {len(miembros)} medidas'
        etiquetas[c.Caso] = f'{c.Caso} · {etiqueta}'
    mejor = resumen[resumen.Tipo == 'Individual'].sort_values('ENS [kWh]').iloc[0]
    ens_base = (resumen['ENS [kWh]'] + resumen['ENS evitada [kWh]']).iloc[0] / 1000
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 18,
                         'text.color': '#172334', 'axes.labelcolor': '#172334',
                         'xtick.color': '#172334', 'ytick.color': '#172334',
                         'svg.fonttype': 'none'})

    def guardar(fig, ax, nombre):
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#94a3b8')
        ax.tick_params(axis='y', length=0)
        fig.text(.04, .025, f'{len(comunes)} escenarios comunes · igual peso · '
                 f'{int(datos.horas.iloc[0])} h · tanda severa', fontsize=12)
        for ext in ('png', 'svg'):
            fig.savefig(salida / f'{nombre}.{ext}', dpi=200, transparent=True)
        plt.close(fig)

    orden = combos.sort_values('ENS [kWh]', ascending=False)
    for pagina, inicio in enumerate(range(0, len(orden), 5), 1):
        sub = orden.iloc[inicio:inicio + 5]
        labels = ['Caso base', 'Mejor individual comparable:\n' + nombres.get(mejor.Clave, mejor.Clave)] + [etiquetas[c] for c in sub.Caso]
        valores = [ens_base, mejor['ENS [kWh]'] / 1000] + list(sub['ENS [kWh]'] / 1000)
        fig, ax = plt.subplots(figsize=(16, 9))
        fig.subplots_adjust(left=.43, right=.91, top=.84, bottom=.16)
        ax.barh(labels, valores, color=['#94a3b8', '#3b82f6'] + ['#059669'] * len(sub), height=.57)
        ax.invert_yaxis()
        ax.set_xlim(0, max(valores) * 1.17)
        for i, v in enumerate(valores):
            ax.text(v + max(valores) * .015, i, f'{v:.1f}', va='center', fontsize=18)
        ax.set_xlabel('ENS esperada [MWh] — menor es mejor', labelpad=16)
        fig.suptitle('¿Cuánta energía queda sin servir?', fontsize=29, x=.04, ha='left', y=.96)
        guardar(fig, ax, f'01_ens_combinaciones_{pagina}')

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=.11, right=.94, top=.85, bottom=.17)
    for _, c in combos.iterrows():
        x, y = c['Reducción ENS [%]'], c['Costo total [USD/año]'] / 1000
        if not np.isfinite([x, y]).all():
            continue
        ax.scatter(x, y, color='#059669', s=140, marker='D')
        ax.annotate(c.Caso, (x, y), xytext=(8, 8), textcoords='offset points', fontsize=17)
    ax.set_xlabel('Reducción esperada de ENS [%] — mayor es mejor', labelpad=15)
    ax.set_ylabel('Costo total [miles de USD/año]', labelpad=15)
    ax.margins(.15)
    ax.axhline(0, color='#94a3b8', lw=1, ls='--')
    fig.suptitle('Costo y beneficio de las combinaciones', fontsize=29, x=.04, ha='left', y=.96)
    guardar(fig, ax, '02_costo_beneficio')
    resumen.to_csv(salida / 'resumen.csv', index=False, encoding='utf-8-sig')
    aporte.to_csv(salida / 'composicion_y_aporte.csv', index=False, encoding='utf-8-sig')
    (salida / 'muestra.json').write_text(json.dumps({
        'tanda': args.tanda, 'escenarios_internos': comunes,
        'corridas_excluidas_por_calidad': excluidas,
        'criterio': 'Combinaciones con mayor número de escenarios válidos; individuales con cobertura completa de la muestra común.',
        'combinaciones_omitidas': sorted(set(b['combinadas'].combinacion) - elegidas),
        'etiquetas': etiquetas}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Figuras PNG y SVG: {salida}; escenarios comunes: {comunes}')


if __name__ == '__main__':
    # La presentación severa usa la muestra de la cartera completa, no la
    # intersección con combinaciones pertenecientes a otra muestra.
    from graficar_presentacion_severo import main as generar_severo
    generar_severo()
