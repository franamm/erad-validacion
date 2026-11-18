# ─────────────────────────────────────────────────────────────
# FIGURAS: PANTALLA vs. EXPORTACIÓN
#
# Varias figuras de esta app están configuradas con el tamaño que pide la
# tesis: 1600×900 px y tipografía de 26-40 pt. Eso está bien para el PNG que
# se pega en el documento, pero es el mismo objeto que se manda a
# `st.plotly_chart(..., use_container_width=True)`, y ahí pasan dos cosas:
# el `width` se ignora (manda el ancho del contenedor) mientras que las
# fuentes NO se reescalan, así que en el navegador quedan títulos y ticks
# desproporcionados sobre un gráfico angosto.
#
# La solución es separar los dos usos: la figura se sigue construyendo en
# tamaño de exportación (es la que se descarga), y a la pantalla va una copia
# reescalada con `_fig_pantalla`.
#
# Viven acá (y no en app_v7_carga_corregida.py) para que la app v8 use
# exactamente los mismos helpers sin importar la v7 entera.
# ─────────────────────────────────────────────────────────────
import plotly.graph_objects as go
import streamlit as st


def _fig_pantalla(fig, height=420, font=13, text_size=None, margin=None):
    """Copia de `fig` dimensionada para el navegador.

    Devuelve una copia: la figura original conserva su tamaño de exportación
    para el PNG/HTML descargable. `text_size` reescala además las etiquetas
    dibujadas sobre las barras (las de `textfont`), que no siguen al `font`
    global del layout. `margin` se deja intacto salvo que se pase uno: los
    márgenes anchos suelen ser parte de la geometría de la figura (el grafo
    de la red, por ejemplo), y pisarlos la desarma.
    """
    f = go.Figure(fig)
    f.update_layout(
        width=None,
        height=height,
        font=dict(size=font),
        legend=dict(font=dict(size=max(font - 1, 9))),
    )
    if margin is not None:
        f.update_layout(margin=margin)
    f.update_xaxes(title_font=dict(size=font + 1), tickfont=dict(size=font - 1))
    f.update_yaxes(title_font=dict(size=font + 1), tickfont=dict(size=font - 1))
    if f.layout.title is not None and f.layout.title.text:
        f.update_layout(title=dict(font=dict(size=font + 4)))
    if text_size is not None:
        f.update_traces(textfont=dict(size=text_size),
                        selector=dict(type="bar"))
    return f


def _descargas_figura(fig, nombre_archivo, key, etiqueta="esta figura"):
    """Descarga PNG/HTML de `fig` EN SU TAMAÑO DE EXPORTACIÓN.

    Detrás de una casilla porque renderizar el PNG toma unos segundos y
    Streamlit re-ejecuta el script entero en cada interacción de cualquier
    widget: generarlo siempre haría lenta la pestaña completa. Es el mismo
    patrón del grafo resumen.
    """
    if not st.checkbox(
        f"🖼️ Preparar la descarga de {etiqueta} en alta resolución "
        "(tarda unos segundos)",
        key=f"chk_descarga_{key}",
    ):
        return

    col_png, col_html = st.columns(2)
    try:
        png = fig.to_image(format="png", scale=3)
    except Exception:
        col_png.caption(
            "📷 PNG no disponible: falta el paquete `kaleido` en este entorno "
            "(`pip install --upgrade kaleido`). Usa el HTML interactivo."
        )
    else:
        col_png.download_button(
            "📷 Descargar PNG (alta resolución)",
            data=png,
            file_name=f"{nombre_archivo}.png",
            mime="image/png",
            key=f"download_{key}_png",
            help="Tamaño y tipografía de la tesis, no los de la pantalla.",
        )
    col_html.download_button(
        "🌐 Descargar HTML interactivo",
        data=fig.to_html(include_plotlyjs="cdn"),
        file_name=f"{nombre_archivo}.html",
        mime="text/html",
        key=f"download_{key}_html",
        help="Conserva hover, zoom y leyenda clickeable.",
    )
