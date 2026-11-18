"""Curvas de costos de abatimiento de la pestaña Resultados (sección E) de la v7.

Un solo lugar para sus parámetros, que usan app_v7_carga_corregida.py y la app v8.
`df_mac` trae una fila por estrategia con `Estrategia`, `Δ ENS`, `Δ SAIDI`,
`Δ SAIFI` y los costos unitarios `Reducción ENS [USD/kWh]`,
`Reducción SAIDI [USD/(h/año)]` y `Reducción SAIFI [USD/(int./año)]`.
"""
import pandas as pd

from helpers_inversion import construir_curva_abatimiento_2

# Tamaño de tesis: es lo que se descarga; a la pantalla va `_fig_pantalla(fig, height=460)`.
TAMANO = dict(ancho=1500, altura=850, fuente_general=18, fuente_titulo=26, fuente_ejes=20,
              fuente_ticks=17, fuente_leyenda=16, fuente_etiquetas=15, escala_exportacion=2)

CURVAS = {
    "ens": dict(columna_reduccion="Δ ENS", columna_costo_unitario="Reducción ENS [USD/kWh]",
                titulo="Curva de costos de abatimiento de ENS",
                titulo_x="Reducción acumulada de ENS [kWh]",
                titulo_y="Costo por ENS evitada [USD/kWh]"),
    "saidi": dict(columna_reduccion="Δ SAIDI", columna_costo_unitario="Reducción SAIDI [USD/(h/año)]",
                  titulo="Curva de costos de abatimiento de SAIDI",
                  titulo_x="Reducción acumulada de SAIDI [h/año]",
                  titulo_y="Costo por reducción de SAIDI [USD/(h/año)]",
                  # Descarta "mejoras" que son ruido numérico del solver (0,001 h en
                  # vez de 0 exacto), para que no aparezcan como una barra falsa.
                  umbral_ruido=1e-2),
    "saifi": dict(columna_reduccion="Δ SAIFI", columna_costo_unitario="Reducción SAIFI [USD/(int./año)]",
                  titulo="Curva de costos de abatimiento de SAIFI",
                  titulo_x="Reducción acumulada de SAIFI [int./año]",
                  titulo_y="Costo por reducción de SAIFI [USD/(int./año)]"),
}

FORMATO_UNIDAD = {"Reducción ENS [USD/MWh]": "{:,.1f}",
                  "Reducción SAIDI [USD/(h/año)]": "{:,.0f}",
                  "Reducción SAIFI [USD/(int./año)]": "{:,.0f}"}


def curvas_mac_estrategias(df_mac):
    """{indicador: (figura o None, tabla procesada)} para ENS, SAIDI y SAIFI."""
    return {k: construir_curva_abatimiento_2(
                df=df_mac, columna_medida="Estrategia", mostrar_etiquetas=True,
                escala_y="symlog", incluir_variaciones_negativas=False, **p, **TAMANO)
            for k, p in CURVAS.items()}


def tabla_costos_unidad(df_mac, curvas):
    """Costo por unidad de mejora de cada estrategia, leído de las curvas.

    El costo por ENS evitada se TABULA en USD/MWh aunque el cálculo y la curva
    sigan en USD/kWh: en kWh los valores caen bajo 1 USD y dos medidas
    distintas salen con el mismo número a la precisión mostrada.
    """
    df = pd.DataFrame(index=df_mac["Estrategia"],
                      columns=[p["columna_costo_unitario"] for p in CURVAS.values()], dtype=float)
    for k, p in CURVAS.items():
        tabla = curvas[k][1]
        if tabla is not None and not tabla.empty and "Costo unitario real" in tabla:
            df[p["columna_costo_unitario"]] = (tabla.set_index("Estrategia")["Costo unitario real"]
                                               .reindex(df.index))
    df = df.rename(columns={"Reducción ENS [USD/kWh]": "Reducción ENS [USD/MWh]"})
    df["Reducción ENS [USD/MWh]"] *= 1000.0
    return df
