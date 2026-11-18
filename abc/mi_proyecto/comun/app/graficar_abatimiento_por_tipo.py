"""Curvas por tipo con los indicadores de los 10 escenarios severos ya exportados."""
from pathlib import Path
import pandas as pd
from graficos_valor_esperado import curva_abatimiento_por_tipo


def main():
    from rutas import DATOS_DIR
    salida = DATOS_DIR / 'figuras_presentacion_severo'
    promedio = pd.read_csv(salida / 'indicadores_esperados.csv')
    for indicador in ('ens', 'saidi', 'saifi'):
        fig, tabla = curva_abatimiento_por_tipo(promedio, indicador)
        if fig is None:
            continue
        fig.update_layout(
            width=1600, font=dict(family='Arial', color='#6b6b78'),
            legend=dict(font=dict(size=25), bgcolor='rgba(0,0,0,0)'),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        nombre = f'abatimiento_{indicador}_por_tipo'
        fig.write_html(salida / f'{nombre}.html', include_plotlyjs=True)
        for ext in ('png', 'svg'):
            fig.write_image(salida / f'{nombre}.{ext}', scale=2)
        tabla.to_csv(salida / f'{nombre}.csv', index=False, encoding='utf-8-sig')
        print(f'{nombre}: {int(tabla["En curva"].sum())} barras exportadas', flush=True)


if __name__ == '__main__':
    main()
