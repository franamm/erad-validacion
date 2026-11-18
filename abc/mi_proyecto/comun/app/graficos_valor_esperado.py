"""ENS esperada y abatimiento a partir de corridas ya calculadas."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def promediar_ens(rutas, rutas_base, buses, bus_zone, n_horas):
    """Promedio equiprobable pareado, con orden de barras explícito.

    No sustituye archivos faltantes por ceros ni mezcla horizontes.
    Los baselines E5 conservan el orden BUSES de la red.

    Con `rutas=None` promedia SOLO el caso base: es la ENS esperada de no
    hacer nada, que es un resultado por sí misma (el diagnóstico contra el
    que se comparan todas las medidas) y no solo la barra de referencia de
    otra figura. En ese caso las tablas salen sin la columna 'Con medida'.
    """
    solo_base = rutas is None
    if not solo_base and (not rutas or len(rutas) != len(rutas_base)):
        raise ValueError('Se necesita un baseline por cada escenario de la medida.')
    buses = list(buses)
    zonas = sorted(set(bus_zone[b] for b in buses))
    acumulados = []
    for archivos in ((rutas_base,) if solo_base else (rutas_base, rutas)):
        total = np.zeros((n_horas, len(buses)), dtype=float)
        for ruta in archivos:
            with np.load(ruta) as z:
                matriz = z['P_LC']
                ids = list(z['BUSES']) if 'BUSES' in z else buses
                if (matriz.ndim != 2 or matriz.shape[0] < n_horas
                        or matriz.shape[1] != len(ids)
                        or len(set(ids)) != len(ids) or set(ids) != set(buses)):
                    raise ValueError(f'Horizonte o barras incompatibles: {ruta}')
                matriz = matriz[:n_horas, [ids.index(b) for b in buses]]
                if not np.isfinite(matriz).all():
                    raise ValueError(f'ENS no finita: {ruta}')
                total += matriz
        acumulados.append(total / len(archivos))
    base = acumulados[0]
    horario = pd.DataFrame({'Hora': np.arange(1, n_horas + 1),
                            'Caso base': base.sum(axis=1)})
    zonal = pd.DataFrame({
        'Zona': [f'Zona {z}' for z in zonas],
        'Caso base': [base[:, [bus_zone[b] == z for b in buses]].sum() for z in zonas],
    })
    if not solo_base:
        medida = acumulados[1]
        horario['Con medida'] = medida.sum(axis=1)
        zonal['Con medida'] = [medida[:, [bus_zone[b] == z for b in buses]].sum()
                               for z in zonas]
    return horario, zonal


def figuras_ens(horario, zonal, nombre):
    """Las dos figuras de ENS esperada. Con `nombre=None` (o sin columna
    'Con medida' en las tablas) sale solo el caso base."""
    zona, hora = go.Figure(), go.Figure()
    colores_zona = {'Zona 1': '#BFDBFE', 'Zona 2': '#FDE68A', 'Zona 3': '#A7F3D0'}
    colores = [colores_zona[z] for z in zonal['Zona']]
    series = [('Caso base', 'Caso base')]
    if nombre is not None and 'Con medida' in zonal.columns:
        series.append(('Con medida', nombre))
    for columna, etiqueta in series:
        marcador = dict(color=colores, opacity=1, pattern=dict(shape='/'))
        if columna != 'Caso base':
            marcador = dict(color=colores, opacity=.85,
                            line=dict(color=colores, width=1.5))
        # if columna == 'Caso base':
        #     marcador = dict(color=colores, opacity=.85,
                                    # line=dict(color=colores, width=1.5))
        zona.add_bar(name=etiqueta, x=zonal['Zona'], y=zonal[columna],
                     marker=marcador,
                     text=[f'{v:,.0f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
                           for v in zonal[columna]],
                     textposition='outside', textfont=dict(size=26))
        hora.add_scatter(name=etiqueta, x=horario['Hora'], y=horario[columna],
                         mode='lines',
                         **({'line': dict(color='#d62728', width=2)}
                            if columna == 'Caso base' else {}))
    # Mismo formato que ENS por zona / ENS por período de app_v7.
    for fig, eje_x, alto, fuente, margen in [
        (zona, 'Zona', 900, 40, dict(l=90, r=40, t=110, b=8)),
        (hora, 'Hora', 700, 30, dict(l=90, r=40, t=110, b=80)),
    ]:
        fig.update_layout(width=1600, height=alto, font=dict(size=fuente),
                          xaxis_title=eje_x, yaxis_title='ENS [kWh]',
                          xaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
                          yaxis=dict(title_font=dict(size=30), tickfont=dict(size=30)),
                          legend=dict(orientation='h', y=-.25, font=dict(size=25)),
                          margin=margen, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    zona.update_layout(barmode='group')
    return zona, hora


def etiqueta_caso(tipo, caso):
    """Nombre legible de un caso, sea medida individual o combinación."""
    import barrido_vista as BV
    return BV._nombre(caso) if tipo == 'Individual' else BV._nombre_combo(caso)


def subdir_caso(tipo):
    """Carpeta del barrido donde vive la corrida de ese tipo de caso."""
    return 'individual' if tipo == 'Individual' else 'combinadas'


UMBRALES = {'ens': 1.0, 'saidi': 1e-3, 'saifi': 1e-4}

# Centinela del "caso" que no es una corrida del barrido sino el propio caso
# base. No puede chocar con un (tipo, medida) real porque no hay carpeta
# 'Base' en el barrido.
CASO_BASE = ('Base', '_caso_base')


def curva_abatimiento_por_tipo(promedio, indicador):
    """La misma curva, identificando individuales y combinadas por color."""
    from html import escape
    from textwrap import wrap
    from helpers_inversion import construir_curva_abatimiento_2
    unidad, costo = {'ens': ('kWh', 'usd_kwh'),
                     'saidi': ('h/año', 'usd_saidi'),
                     'saifi': ('int./año', 'usd_saifi')}[indicador]
    datos = promedio.copy()
    datos['Estrategia'] = [etiqueta_caso(t, c) for t, c
                           in zip(datos['_tipo'], datos['_caso'])]
    # Claves breves únicas: las etiquetas largas se conservan en la tabla y hover.
    datos['Código'] = [f'{"C" if t == "Combinada" else "I"}{i + 1}'
                       for i, t in enumerate(datos['_tipo'])]
    datos['Reducción'] = datos[f'{indicador}_base'] - datos[indicador]
    fig, procesados = construir_curva_abatimiento_2(
        df=datos, columna_medida='Código', columna_reduccion='Reducción',
        columna_costo_unitario=costo, columna_costo_total='cst_costo_total',
        titulo=f'Abatimiento de {indicador.upper()}: individuales y combinadas',
        titulo_x=f'Reducción de {indicador.upper()} por alternativa [{unidad}] · tramos no acumulables',
        titulo_y=('Costo por ENS evitada [USD/kWh]' if indicador == 'ens' else
                  f'Costo por reducción de {indicador.upper()} [USD/({unidad})]'),
        umbral_ruido=UMBRALES[indicador], escala_y='symlog',
        mostrar_etiquetas=False, mostrar_excluidas_en_leyenda=False)
    if fig is None:
        return None, datos
    colores = {'Individual': '#56B4E9', 'Combinada': '#009E73'}
    por_codigo = datos.set_index('Código')
    tipos_presentes = set()
    nombres_por_tipo = {'Individual': [], 'Combinada': []}
    numeros = {}
    for traza in fig.data:
        if traza.type != 'bar' or traza.name not in por_codigo.index:
            continue
        fila = por_codigo.loc[traza.name]
        tipo = fila['_tipo']
        tipos_presentes.add(tipo)
        traza.marker.color = colores[tipo]
        traza.legendgroup = tipo
        traza.showlegend = False
        numero = len(numeros) + 1
        numeros[traza.name] = numero
        nombres_por_tipo[tipo].append((numero, fila['Estrategia']))
        traza.text = [str(numero)]
        traza.textposition = 'outside'
        # El helper ya conserva costo real, reducción y extremos del tramo.
        custom = [list(r) for r in traza.customdata]
        for r in custom:
            r[0] = f'{numero}. {fila["Estrategia"]} ({tipo})'
        traza.customdata = custom
    for tipo, color in colores.items():
        if tipo in tipos_presentes:
            fig.add_bar(x=[None], y=[None], name=tipo, legendgroup=tipo,
                        marker_color=color, showlegend=True, hoverinfo='skip')
    # Leyenda de nombres incluida en el PNG/SVG, con columnas por tipo.
    # Las coordenadas en píxeles evitan comprimir las combinaciones largas.
    max_lineas = 0
    for tipo, x in [('Individual', 0), ('Combinada', .49)]:
        linea = 0
        for numero, nombre in nombres_por_tipo[tipo]:
            partes = wrap(f'{numero}. {nombre}', width=48,
                          break_long_words=False, break_on_hyphens=False)
            fig.add_annotation(
                x=x, y=0, xref='paper', yref='paper',
                xanchor='left', yanchor='top', yshift=-125 - linea * 26,
                text='<br>'.join(escape(p) for p in partes), showarrow=False,
                align='left', font=dict(size=16, color=colores[tipo]))
            linea += len(partes) + .05
        max_lineas = max(max_lineas, linea)
    margen_inferior = int(150 + max_lineas * 26)
    fig.update_layout(width=1600, height=700 + margen_inferior,
                      margin=dict(l=165, r=80, t=140, b=margen_inferior),
                      legend=dict(groupclick='togglegroup', y=-.18))
    datos['Barra'] = datos['Código'].map(numeros).astype('Int64')
    datos['En curva'] = datos['Código'].isin(procesados['Código'])
    return fig, datos


def resumir_indicadores(filas):
    """Niveles esperados y costos unitarios de la curva, sin filtrar por signo del costo."""
    columnas = [c for c in (
        'ens', 'saidi', 'saifi', 'ens_base', 'saidi_base', 'saifi_base',
        'cst_capex_anual', 'cst_opex_anual', 'cst_costo_total',
        # El costo de la ENS que queda con la medida puesta: es lo que hace
        # comparable la fila del caso base en la tabla económica (su costo de
        # ENS contra el de cada medida; la resta es justamente 'ahorro_ens').
        'cst_costo_ens_estr',
        'cst_ahorro_ens', 'cst_ahorro_neto') if c in filas]
    grupo = filas.groupby(['_tipo', '_caso'])
    # No promediar parcialmente costos o indicadores faltantes.
    resultado = grupo[columnas].agg(lambda x: x.mean(skipna=False))
    resultado['n_esc'] = grupo.size()
    resultado['ens_std'] = grupo['ens'].std()
    resultado['d_ens_pct_std'] = (grupo['d_ens_pct'].std()
                                  if 'd_ens_pct' in filas else np.nan)
    for indicador, col in [('ens', 'usd_kwh'), ('saidi', 'usd_saidi'), ('saifi', 'usd_saifi')]:
        delta = resultado[f'{indicador}_base'] - resultado[indicador]
        resultado[f'd_{indicador}'] = delta
        resultado[f'd_{indicador}_pct'] = delta / resultado[f'{indicador}_base'].where(
            resultado[f'{indicador}_base'] > 0)
        if 'cst_costo_total' in resultado:
            resultado[col] = resultado['cst_costo_total'] / delta.where(delta > UMBRALES[indicador])
    return resultado.reset_index()


def seleccionar_corridas(todo):
    """Una selección de casos y escenarios para tablas y gráficos."""
    import streamlit as st
    # Un caso es (tipo, nombre): las combinaciones viven en `combinadas/` y las
    # medidas sueltas en `individual/`, así que el tipo no es decorativo, es
    # parte de la ruta. Antes esto se limitaba a las individuales y las
    # combinaciones no tenían ningún gráfico.
    casos = (todo[['_tipo', '_caso']].drop_duplicates()
             .itertuples(index=False, name=None))
    # Combinadas primero: son las que interesan cuando ya se miraron las sueltas.
    opciones = sorted(casos, key=lambda k: (k[0] != 'Combinada', str(k[1])))
    if not opciones:
        st.info('No hay casos que graficar en este barrido.')
        return todo.iloc[:0].copy()

    def _fmt(k):
        return ('🔗 ' if k[0] == 'Combinada' else '▪️ ') + etiqueta_caso(*k)

    seleccion = st.multiselect('Casos para los gráficos de valor esperado', opciones,
                              default=opciones, format_func=_fmt,
                              key='ve_graficos_medidas')
    if not seleccion:
        st.info('Selecciona al menos un caso.')
        return todo.iloc[:0].copy()
    _clave = todo[['_tipo', '_caso']].apply(tuple, axis=1)
    filas = todo[_clave.isin(seleccion)].copy()
    if filas.duplicated(['_tipo', '_caso', 'escenario']).any():
        st.warning('Hay corridas duplicadas por caso y escenario; revisa el resumen.')
        return todo.iloc[:0].copy()
    comunes = set.intersection(*(
        set(filas.loc[(filas['_tipo'] == t) & (filas['_caso'] == c), 'escenario'])
        for t, c in seleccion))
    if not comunes:
        st.warning('Los casos seleccionados no tienen escenarios comunes.')
        return todo.iloc[:0].copy()
    # Un caso con pocas corridas válidas (típico de las combinaciones grandes,
    # que se cortan por TimeLimit) arrastra a TODOS al mismo subconjunto: la
    # intersección es lo correcto para comparar, pero conviene verlo.
    _por_caso = {(t, c): len(set(filas.loc[(filas['_tipo'] == t)
                                           & (filas['_caso'] == c), 'escenario']))
                 for t, c in seleccion}
    _limita = [k for k, v in _por_caso.items() if v == len(comunes)]
    if len(comunes) < max(_por_caso.values()):
        st.warning(
            f'La comparación queda restringida a {len(comunes)} escenario(s) '
            f'porque es lo que tiene en común el caso más incompleto '
            f'({", ".join(etiqueta_caso(*k) for k in _limita)}). '
            f'Deselecciónalo para promediar el resto sobre más escenarios.')
    filas = filas[filas['escenario'].isin(comunes)]
    if filas['horas'].nunique() != 1:
        st.warning('Las corridas tienen horizontes distintos y no se pueden promediar juntas.')
        return todo.iloc[:0].copy()
    return filas


def render(todo, barrido, buses, bus_zone, fig_pantalla, descargas):
    import streamlit as st
    import barrido_vista as BV
    from helpers_inversion import construir_curva_abatimiento_2

    st.markdown('#### ENS y curvas de abatimiento — valores esperados')
    st.caption(f"Tanda usada en estos gráficos: {Path(barrido['dir']).parent.name} · "
               f"{Path(barrido['dir']).name}")

    filas = todo
    if filas.empty:
        return
    from graficos_combinaciones import render_combinaciones
    render_combinaciones(filas, descargas)
    comunes = set(filas['escenario'])
    seleccion = list(filas[['_tipo', '_caso']].drop_duplicates().itertuples(index=False, name=None))
    # El caso base solo, sin ninguna medida: es la figura del diagnóstico
    # (cuánta ENS deja el sistema tal como está), y antes solo existía como
    # la barra de referencia de la figura de alguna medida.
    seleccion.insert(0, CASO_BASE)

    def _fmt(k):
        return 'Caso base (sin medidas)' if k == CASO_BASE else etiqueta_caso(*k)
    n_horas = int(filas['horas'].iloc[0])
    escenarios = sorted(int(e) for e in comunes)
    st.caption(
        f'Caso base y todas las medidas: los mismos {len(escenarios)} escenarios, '
        f'con peso 1/{len(escenarios)} y horizonte {n_horas} h. '
        'Escenarios (numeración de la app): ' + ', '.join(str(e + 1) for e in escenarios)
        + '. Se usa la intersección de las corridas válidas; las faltantes o excluidas '
        'no se reemplazan por ceros. Valor esperado sobre esta muestra calculada.')
    caso = st.selectbox('Caso para ENS por zona y horario', seleccion,
                        format_func=_fmt, key='ve_graficos_medida')
    tipo_c, nom_c = caso
    solo_base = caso == CASO_BASE
    ruta = Path(barrido['dir'])
    bases = [BV.P.base_dir(barrido['variante']) / f'esc_{e:03d}.npz' for e in escenarios]
    archivos = None if solo_base else [
        ruta / subdir_caso(tipo_c) / str(nom_c) / f'esc_{e:03d}.npz'
        for e in escenarios]
    try:
        horario, zonal = promediar_ens(archivos, bases, buses, bus_zone, n_horas)
    except (OSError, KeyError, ValueError) as exc:
        st.warning(f'No se pudo construir la comparación de ENS: {exc}')
        return
    nombre = None if solo_base else etiqueta_caso(tipo_c, nom_c)
    zona, hora = figuras_ens(horario, zonal, nombre)
    if solo_base:
        total = float(zonal['Caso base'].sum())
        st.caption(
            f'Caso base sin ninguna medida, promediado sobre los '
            f'{len(escenarios)} escenarios. ENS total esperada: '
            f'**{total:,.0f} kWh/año** ({total / 1000:,.1f} MWh/año). '
            + ' · '.join(f"{z}: {v:,.0f} kWh ({v / total:.1%})"
                         for z, v in zip(zonal['Zona'], zonal['Caso base'])
                         if total > 0))
    t_zona, t_hora, t_mac = st.tabs(['ENS por zona', 'ENS total horario', 'Curvas de abatimiento'])
    for tab, fig, tabla, sufijo in [(t_zona, zona, zonal, 'zona'),
                                   (t_hora, hora, horario, 'horario')]:
        with tab:
            # Conservar en pantalla el formato original de Resultados:
            # la misma figura, fuentes, márgenes y altura que se descarga.
            st.plotly_chart(fig,
                            use_container_width=True, key=f've_fig_{sufijo}')
            # El nombre de una combinación puede pasar de 100 caracteres
            # ('combo_dg_sol+dg_eol+...'), así que se recorta para el archivo.
            slug = ('caso_base' if solo_base
                    else str(nom_c).replace('combo_', '')[:60])
            descargas(fig, f'ens_esperada_{sufijo}_{slug}', key=f've_exp_{sufijo}')
            st.download_button('Descargar datos CSV', tabla.to_csv(index=False).encode('utf-8-sig'),
                               file_name=f'ens_esperada_{sufijo}_{slug}.csv',
                               mime='text/csv', key=f've_csv_{sufijo}')
    with t_mac:
        render_mac(filas, descargas)


def render_mac(filas, descargas):
    """Curvas de abatimiento de ENS, SAIDI y SAIFI sobre `filas` (valor esperado).

    Separada de `render` para que la app v8 muestre exactamente las mismas curvas.
    """
    import streamlit as st
    from helpers_inversion import construir_curva_abatimiento_2

    # Promediar niveles y restarlos; nunca promediar cocientes de costos.
    columnas = ['ens', 'saidi', 'saifi', 'ens_base', 'saidi_base', 'saifi_base',
                'cst_costo_total']
    if not set(columnas).issubset(filas.columns):
        st.info('Faltan indicadores o costos para construir las curvas.')
        return
    if filas[columnas].isna().any().any():
        st.warning('Hay indicadores o costos faltantes; no se calcularán promedios parciales.')
        return
    promedio = resumir_indicadores(filas)
    promedio['Estrategia'] = [etiqueta_caso(t, c) for t, c
                              in zip(promedio['_tipo'], promedio['_caso'])]
    st.caption('Abatimiento = E[caso base] − E[con medida]. '
               'Costo unitario = E[costo total] / abatimiento esperado. '
               'Las anchuras comparan alternativas individuales; su suma no representa '
               'el efecto de instalar todas las medidas juntas.')
    if (promedio['_tipo'] == 'Combinada').any():
        st.warning(
            '⚠️ Con combinaciones en la curva, el eje X **no se puede leer '
            'como una cartera acumulable**: una combinación ya contiene a '
            'sus medidas sueltas, así que sus barras cuentan dos veces la '
            'misma ENS evitada. Sirve para comparar alternativas entre sí '
            '(cada barra es una decisión excluyente), no para sumarlas.')
    for indicador, unidad, umbral in [('ens', 'kWh', 1.0), ('saidi', 'h/año', 1e-3),
                                     ('saifi', 'int./año', 1e-4)]:
        delta = promedio[f'{indicador}_base'] - promedio[indicador]
        promedio['Reducción'] = delta
        promedio['Costo unitario'] = promedio[{'ens': 'usd_kwh', 'saidi': 'usd_saidi', 'saifi': 'usd_saifi'}[indicador]]
        fig, _ = construir_curva_abatimiento_2(
            df=promedio, columna_medida='Estrategia', columna_reduccion='Reducción',
            columna_costo_unitario='Costo unitario', columna_costo_total='cst_costo_total',
            titulo=f'Curva de costos de abatimiento de {indicador.upper()}',
            titulo_x=f'Reducción acumulada de {indicador.upper()} [{unidad}]',
            titulo_y=('Costo por ENS evitada [USD/kWh]' if indicador == 'ens' else
                      f'Costo por reducción de {indicador.upper()} [USD/({unidad})]'),
            umbral_ruido=umbral, incluir_variaciones_negativas=False, escala_y='symlog',
            ancho=1500, altura=850, fuente_general=18, fuente_titulo=26,
            fuente_ejes=20, fuente_ticks=17, fuente_leyenda=16, fuente_etiquetas=15,
            mostrar_etiquetas=True, escala_exportacion=2)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True,
                            key=f've_mac_{indicador}')
            descargas(fig, f'abatimiento_esperado_{indicador}', key=f've_exp_mac_{indicador}')
        else:
            st.info(f'No hay reducción suficiente de {indicador.upper()} para dibujar barras.')
        st.download_button(f'Datos de abatimiento de {indicador.upper()} (CSV)',
                           promedio.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f'abatimiento_esperado_{indicador}.csv', mime='text/csv',
                           key=f've_csv_mac_{indicador}')
    if (promedio['_tipo'] == 'Combinada').any():
        st.markdown('##### Curva adicional: individuales y combinadas por color')
        indicador = st.selectbox('Indicador de la curva por tipo',
                                 ['ens', 'saidi', 'saifi'], format_func=str.upper,
                                 key='ve_mac_tipo_indicador')
        fig_tipo, tabla_tipo = curva_abatimiento_por_tipo(promedio, indicador)
        st.caption('Azul: medidas individuales. Verde: combinaciones. '
                   'Cada barra es una alternativa; las anchuras no representan '
                   'una cartera acumulable. Cada número corresponde al nombre '
                   'incluido debajo de la curva.')
        if fig_tipo is not None:
            st.plotly_chart(fig_tipo, use_container_width=True, key='ve_mac_tipo')
            descargas(fig_tipo, f'abatimiento_{indicador}_por_tipo', key='ve_exp_mac_tipo')
        else:
            st.info('No hay reducción suficiente para construir esta curva.')
        columnas_tipo = [c for c in ['Barra', 'Estrategia', '_tipo', 'Reducción',
                                     'cst_costo_total', 'En curva'] if c in tabla_tipo]
        st.dataframe(tabla_tipo[columnas_tipo], hide_index=True, use_container_width=True)
        st.download_button('Datos de la curva por tipo (CSV)',
                           tabla_tipo.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f'abatimiento_{indicador}_por_tipo.csv',
                           mime='text/csv', key='ve_csv_mac_tipo')
