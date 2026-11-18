"""Exporta el estilo de la app sobre la muestra severa de la cartera completa."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import barrido_vista as BV
from graficos_valor_esperado import figuras_ens, promediar_ens, resumir_indicadores


def main():
    tanda = 'barrido/sec-0-2-9-10 severo'
    b = BV.cargar_barrido(tanda, 0)
    combinadas = b['combinadas']
    claves = list(combinadas.combinacion.unique())
    completa = max(claves, key=lambda c: len(c.removeprefix('combo_').split('+')))
    muestra = sorted(combinadas.loc[combinadas.combinacion == completa, 'escenario'].unique())
    datos = pd.concat([
        b['individual'].assign(_tipo='Individual', _caso=b['individual']['medida']),
        combinadas.assign(_tipo='Combinada', _caso=combinadas.combinacion)], ignore_index=True)
    if datos.horas.isna().any() or datos.horas.nunique() != 1:
        raise ValueError('Horizontes incompatibles.')
    horas = int(datos.horas.iloc[0])
    base = BV.cargar_base(b['variante'], muestra, horas)
    datos = BV.enriquecer(datos[datos.escenario.isin(muestra)], base, b['mip_gap'])
    admitidos = [g for _, g in datos.groupby(['_tipo', '_caso'])
                 if set(g.escenario) == set(muestra) and not g.sospechosa.any()]
    filas = pd.concat(admitidos, ignore_index=True)
    if filas.duplicated(['_tipo', '_caso', 'escenario']).any():
        raise ValueError('Hay escenarios duplicados.')
    if completa not in set(filas._caso):
        raise ValueError('La cartera completa no tiene una muestra válida completa.')
    resumen = resumir_indicadores(filas)
    from rutas import DATOS_DIR
    salida = DATOS_DIR / 'figuras_presentacion_severo'
    salida.mkdir(parents=True, exist_ok=True)
    exportados = []

    def exportar(fig, nombre, alto=900, izquierda=125):
        fig.update_layout(
            template='plotly_white', width=1600, height=alto,
            font=dict(family='Arial', size=28, color='#6b6b78'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=25),
                        orientation='h', x=0, y=-.22),
            margin=dict(l=izquierda, r=95, t=90, b=180))
        fig.update_xaxes(title_font=dict(size=30), tickfont=dict(size=28),
                         gridcolor='#D9DDE3', zerolinecolor='#D9DDE3')
        fig.update_yaxes(title_font=dict(size=30), tickfont=dict(size=28),
                         gridcolor='#D9DDE3', zerolinecolor='#D9DDE3')
        horizontal = any(getattr(t, 'orientation', None) == 'h' for t in fig.data)
        fig.update_xaxes(showgrid=horizontal)
        fig.update_yaxes(showgrid=not horizontal)
        fig.write_html(salida / f'{nombre}.html', include_plotlyjs=True)
        for ext in ('png', 'svg'):
            fig.write_image(salida / f'{nombre}.{ext}', scale=2)
        exportados.append(nombre)

    red = BV.P._cargar('red.pkl')
    bases = [BV.P.base_dir(b['variante']) / f'esc_{e:03d}.npz' for e in muestra]
    rutas = [Path(b['dir']) / 'combinadas' / completa / f'esc_{e:03d}.npz' for e in muestra]
    horario, zonal = promediar_ens(rutas, bases, red['BUSES'], red['bus_zone'], horas)
    registro = resumen[resumen._caso == completa].iloc[0]
    if not np.isclose(zonal['Con medida'].sum(), registro.ens, rtol=1e-5, atol=1):
        raise ValueError('El resumen y los NPZ de la combinación no coinciden.')
    zona, hora = figuras_ens(horario, zonal, BV._nombre_combo(completa))
    zona.data[0].marker.pattern.update(
        fillmode='replace', bgcolor='rgba(0,0,0,0)',
        fgcolor=list(zona.data[0].marker.color), solidity=.5)
    hora.data[1].line.color = '#56B4E9'
    exportar(zona, 'ens_esperada_zona_12_medidas')
    exportar(hora, 'ens_esperada_horaria_12_medidas', alto=800)
    zonal.to_csv(salida / 'ens_zonal.csv', index=False, encoding='utf-8-sig')
    horario.to_csv(salida / 'ens_horaria.csv', index=False, encoding='utf-8-sig')

    # Barras separadas: cada combinación es una alternativa, no acumulable.
    combos = resumen[resumen._tipo == 'Combinada'].sort_values('d_ens_pct').copy()
    cortos = {'dg_eol': 'Eólica', 'dg_sol': 'Solar', 'dg_fos': 'GD fósil',
              'reconfig_topo': 'Reconfiguración', 'ess_4h': 'SAE 4 h',
              'ess_5h': 'SAE 5 h', 'ess_6h': 'SAE 6 h'}

    def etiqueta(c):
        ms = c.removeprefix('combo_').split('+')
        if len(ms) > 5:
            return f'Combinada ({len(ms)} medidas)'
        etiquetas = [cortos.get(m, m) for m in ms]
        if all(m in ms for m in ('ess_4h', 'ess_5h', 'ess_6h')):
            etiquetas = [cortos.get(m, m) for m in ms if not m.startswith('ess_')] + ['SAE 4/5/6 h']
        return ' + '.join(etiquetas)

    combos['Etiqueta'] = combos._caso.map(etiqueta)
    for col, titulo, nombre in [
        ('d_ens_pct', 'Reducción esperada de ENS [%]', 'reduccion_ens_combinaciones'),
        ('usd_kwh', 'Costo por ENS evitada [USD/kWh]', 'costo_unitario_combinaciones')]:
        valores = combos[col] * (100 if col == 'd_ens_pct' else 1)
        fig = go.Figure(go.Bar(y=combos.Etiqueta, x=valores, orientation='h',
                              marker_color='#A7F3D0' if col == 'd_ens_pct' else '#BFDBFE',
                              text=[f'{v:.1f}'.replace('.', ',') for v in valores],
                              textposition='outside', cliponaxis=False))
        fig.update_layout(xaxis_title=titulo, showlegend=False)
        fig.update_xaxes(range=[min(0, float(valores.min()) * 1.1), float(valores.max()) * 1.15])
        exportar(fig, nombre, izquierda=510)

    resumen.to_csv(salida / 'indicadores_esperados.csv', index=False, encoding='utf-8-sig')
    (salida / 'muestra.json').write_text(json.dumps({
        'tanda': tanda, 'variante_baseline': b['variante'],
        'escenarios_internos': [int(e) for e in muestra],
        'escenarios_app': [int(e) + 1 for e in muestra], 'peso': 1 / len(muestra),
        'cartera_referencia': completa, 'combinaciones_omitidas': sorted(set(claves) - set(filas._caso)),
        'ENS_zonal_base': zonal['Caso base'].tolist(),
        'ENS_zonal_combinada': zonal['Con medida'].tolist(),
        'archivos': exportados}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(zonal.to_string(index=False))
    print(f'{len(muestra)} escenarios; {len(combos)} combinaciones. Salida: {salida}')


if __name__ == '__main__':
    main()
