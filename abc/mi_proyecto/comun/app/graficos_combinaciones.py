"""Comparaciones de carteras sobre una muestra común de escenarios."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def preparar_comparacion(filas):
    """No suma beneficios como si fueran acumulables ni completa datos faltantes."""
    requeridas = {'_tipo', '_caso', 'escenario', 'horas', 'ens', 'ens_base'}
    if not requeridas.issubset(filas.columns):
        raise ValueError('Faltan columnas de ENS, casos o escenarios.')
    if filas.empty:
        return pd.DataFrame(), pd.DataFrame()
    if filas.duplicated(['_tipo', '_caso', 'escenario']).any():
        raise ValueError('Hay casos con escenarios duplicados.')
    grupos = list(filas.groupby(['_tipo', '_caso'], sort=True))
    muestra = set(grupos[0][1]['escenario'])
    if any(set(g['escenario']) != muestra for _, g in grupos):
        raise ValueError('Selecciona una muestra común para todos los casos.')
    if filas['horas'].isna().any() or filas['horas'].nunique() != 1:
        raise ValueError('Los horizontes de las corridas deben coincidir.')
    if not np.isfinite(filas[['ens', 'ens_base']].to_numpy(dtype=float)).all():
        raise ValueError('Hay valores de ENS faltantes o no finitos.')
    base = grupos[0][1].set_index('escenario')['ens_base'].sort_index()
    if any(not np.allclose(g.set_index('escenario')['ens_base'].sort_index(),
                           base, rtol=1e-8, atol=1e-6) for _, g in grupos):
        raise ValueError('Los casos no usan el mismo baseline por escenario.')

    registros = []
    for (tipo, caso), g in grupos:
        ens = g['ens'].mean()
        registro = dict(Tipo=tipo, Clave=caso, Escenarios=len(muestra),
                        **{'ENS [kWh]': ens,
                           'ENS evitada [kWh]': base.mean() - ens,
                           'Reducción ENS [%]': 100 * (base.mean() - ens) / base.mean()
                           if base.mean() > 0 else np.nan})
        for origen, destino in [('cst_costo_total', 'Costo total [USD/año]'),
                                ('cst_ahorro_neto', 'Ahorro neto [USD/año]'),
                                ('saidi', 'SAIDI [h/cl.]'),
                                ('saifi', 'SAIFI [int./cl.]')]:
            registro[destino] = (g[origen].mean(skipna=False)
                                 if origen in g else np.nan)
        registros.append(registro)
    resumen = pd.DataFrame(registros)
    resumen['Caso'] = [f'C{i + 1}' if t == 'Combinada' else c
                       for i, (t, c) in enumerate(zip(resumen.Tipo, resumen.Clave))]
    individuales = resumen[resumen.Tipo == 'Individual'].set_index('Clave')
    comparaciones = []
    for _, c in resumen[resumen.Tipo == 'Combinada'].iterrows():
        miembros = str(c.Clave).removeprefix('combo_').split('+')
        faltantes = sorted(set(miembros) - set(individuales.index))
        r = {'Caso': c.Caso, 'Medidas': ' + '.join(miembros),
             'Estado': 'Faltan individuales: ' + ', '.join(faltantes) if faltantes else 'Completa'}
        if not faltantes:
            ind = individuales.loc[miembros]
            # Una alternativa fija, elegida por su ENS esperada, no una medida
            # distinta elegida con conocimiento previo de cada escenario.
            mejor = ind.loc[ind['ENS [kWh]'].idxmin()]
            extra = mejor['ENS [kWh]'] - c['ENS [kWh]']
            costo_extra = c['Costo total [USD/año]'] - mejor['Costo total [USD/año]']
            r.update({'Mejor individual': mejor.name,
                      'ENS adicional evitada [kWh]': extra,
                      'Costo adicional [USD/año]': costo_extra,
                      'Costo incremental [USD/kWh]': costo_extra / extra if extra > 1 else np.nan,
                      'Suma individual evitada [kWh]': ind['ENS evitada [kWh]'].sum(),
                      'Beneficio conjunto [kWh]': c['ENS evitada [kWh]'],
                      'Diferencia frente a suma [kWh]': c['ENS evitada [kWh]'] - ind['ENS evitada [kWh]'].sum()})
        comparaciones.append(r)
    return resumen, pd.DataFrame(comparaciones)


def figuras_comparacion(resumen, nombre=lambda x: x):
    """ENS residual y costo frente a beneficio: cada punto es una alternativa."""
    ens, costo = go.Figure(), go.Figure()
    orden = resumen.sort_values('ENS [kWh]', ascending=False)
    base = float((resumen['ENS [kWh]'] + resumen['ENS evitada [kWh]']).iloc[0])
    ens.add_bar(name='Caso base', x=[base / 1000], y=['Caso base'],
                orientation='h', marker_color='#94a3b8')
    for tipo, color, simbolo in [('Individual', '#60a5fa', 'circle'),
                                ('Combinada', '#059669', 'diamond')]:
        g = orden[orden.Tipo == tipo]
        etiquetas = [r.Caso if tipo == 'Combinada' else nombre(r.Clave)
                     for _, r in g.iterrows()]
        ens.add_bar(name=tipo, x=g['ENS [kWh]'] / 1000, y=etiquetas,
                    orientation='h', marker_color=color,
                    text=g['ENS [kWh]'].map(lambda x: f'{x / 1000:.2f}'),
                    textposition='outside', cliponaxis=False)
        validos = np.isfinite(g['Costo total [USD/año]'])
        costo.add_scatter(name=tipo, x=g.loc[validos, 'Reducción ENS [%]'],
                          y=g.loc[validos, 'Costo total [USD/año]'],
                          text=np.array(etiquetas)[validos], mode='markers+text',
                          textposition='top center',
                          marker=dict(size=13, color=color, symbol=simbolo),
                          hovertemplate='%{text}<br>Reducción: %{x:.2f}%<br>Costo: %{y:,.0f} USD/año<extra></extra>')
    ens.update_layout(title='ENS esperada: medidas solas y combinaciones',
                      xaxis_title='ENS residual [MWh]',
                      height=max(550, 38 * (len(resumen) + 1)))
    costo.update_layout(title='Costo frente a reducción esperada de ENS',
                        xaxis_title='Reducción de ENS respecto al caso base [%]',
                        yaxis_title='Costo total [USD/año]', height=650)
    costo.add_hline(y=0, line_color='#94a3b8', line_dash='dot')
    for fig in (ens, costo):
        fig.update_layout(template='plotly_white', width=1400, font=dict(size=30),
                          margin=dict(l=180, r=100, t=85, b=110),
                          legend=dict(orientation='h', y=-.18))
    ens.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    return ens, costo


def render_combinaciones(filas, descargas):
    import streamlit as st
    from barrido_vista import _nombre

    if not (filas['_tipo'] == 'Combinada').any():
        return
    st.markdown('#### ¿Qué aporta combinar las medidas?')
    try:
        resumen, comparaciones = preparar_comparacion(filas)
    except ValueError as exc:
        st.warning(str(exc))
        return
    st.caption('Cada alternativa usa los mismos escenarios con igual peso. '
               'La reducción porcentual se calcula a partir de las ENS promedio. '
               'C1, C2, etc. identifican las combinaciones en la tabla de composición.')
    composicion = comparaciones[['Caso', 'Medidas', 'Estado']].copy()
    composicion['Medidas'] = composicion['Medidas'].map(
        lambda s: ' + '.join(_nombre(m) for m in s.split(' + ')))
    st.dataframe(composicion, hide_index=True, use_container_width=True)
    tabs = st.tabs(['ENS residual', 'Costo y beneficio', 'Aporte adicional'])
    for tab, fig, slug in zip(tabs[:2], figuras_comparacion(resumen, _nombre),
                              ['ens_combinaciones', 'costo_beneficio_combinaciones']):
        with tab:
            st.plotly_chart(fig, use_container_width=True, key=f've_{slug}')
            descargas(fig, slug, key=f've_exp_{slug}')
    with tabs[1]:
        st.caption('Más a la derecha significa mayor reducción; más abajo, menor costo. '
                   'Los casos sin costo disponible no aparecen en este gráfico.')
    with tabs[2]:
        st.dataframe(comparaciones, hide_index=True, use_container_width=True)
        st.caption('La mejor individual es una misma medida elegida por menor ENS promedio '
                   'entre los componentes de cada combinación. El costo incremental divide '
                   'el costo adicional por la ENS adicional evitada; solo se muestra si esta '
                   'supera 1 kWh. Una diferencia negativa frente a la suma indica subaditividad '
                   'y una positiva, superaditividad; también pueden influir el redimensionamiento '
                   'y la calidad de las soluciones. La suma individual es una referencia, '
                   'no un beneficio realizable ni una atribución causal a cada medida.')
    st.dataframe(resumen, hide_index=True, use_container_width=True)
    for tabla, slug in [(resumen, 'resumen_combinaciones'),
                         (comparaciones, 'aporte_combinaciones')]:
        st.download_button('Descargar ' + slug.replace('_', ' ') + ' (CSV)',
                           tabla.to_csv(index=False).encode('utf-8-sig'),
                           file_name=slug + '.csv', mime='text/csv', key=f've_csv_{slug}')
