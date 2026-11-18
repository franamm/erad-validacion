"""Prueba de navegación con artefactos locales. No pulsa botones de ejecución."""
import logging
import sys

from streamlit.testing.v1 import AppTest

from estudio_v8.core import APP_DIR

ESTADO = {}


def correr(app, etiqueta, min_figuras=0):
    for clave, valor in ESTADO.items():
        app.session_state[clave] = valor
    app.run(timeout=300)
    errores = [x.message for x in app.exception]
    if errores:
        raise AssertionError(f"{etiqueta}: {errores}")
    n = len(app.get("plotly_chart"))
    if n < min_figuras:
        raise AssertionError(f"{etiqueta}: {n} figuras, se esperaban {min_figuras}")
    print(f"OK {etiqueta} · figuras={n}", flush=True)


def main():
    logging.disable(logging.WARNING)
    app = AppTest.from_file(str(APP_DIR / "app_v8_estudio.py"), default_timeout=300)
    correr(app, "Datos de entrada")
    ESTADO["pagina_v8"] = "⚙️ Optimización y análisis"
    correr(app, "Optimización (estudio de la tesis)")
    ESTADO.update(opt_mac=True, opt_solap=True, opt_decision=True)
    correr(app, "Optimización con los tres análisis")
    ESTADO["pagina_v8"] = "📊 Resultados"
    from estudio_v8.resultados import PESTANAS
    minimo = {PESTANAS[0]: 1, PESTANAS[2]: 2, PESTANAS[3]: 1, PESTANAS[4]: 1}
    from estudio_v8.core import ART_DIR
    import casos
    if not casos.es_validacion(ART_DIR):
        # Un caso nuevo parte con pocas medidas corridas: sin combinaciones el
        # solapamiento solo avisa. Lo que sí tiene que haber siempre es la red.
        minimo = {PESTANAS[0]: 1}
    for pestana in PESTANAS:
        ESTADO["resultados_seccion"] = pestana
        correr(app, pestana, minimo.get(pestana, 0))
    ESTADO["abrir_guia"] = True
    correr(app, "Guía")


if __name__ == "__main__":
    sys.exit(main())
