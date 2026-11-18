"""
check_entrada_red.py — Prueba de navegación de la pestaña "Datos de entrada"
============================================================================

Mismo enfoque que `estudio_v8/check_ui.py`: se recorre la vista con `AppTest`
de Streamlit y se falla si alguna rama lanza excepción. **No pulsa ningún
botón de ejecución**, así que no lanza etapas ni escribe artefactos.

Existe porque esta vista tiene muchas ramas que dependen de lo que hay en
disco (caso armado o vacío, con fallas o sin, validación u otro), y son justo
las que no se ven al abrirla una vez en el navegador.

    python check_entrada_red.py
"""

import logging
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parent
CLAVE = "entrada_red"


def correr(etiqueta, estado, esperado=()):
    app = AppTest.from_file(str(APP_DIR / "app_v8_estudio.py"), default_timeout=300)
    app.session_state["pagina_v8"] = "📥 Datos de entrada"
    app.session_state["datos_vista"] = "Origen de los datos"
    for k, v in estado.items():
        app.session_state[k] = v
    app.run()
    errores = [x.message for x in app.exception]
    if errores:
        raise AssertionError(f"{etiqueta}: {errores}")

    texto = " ".join(str(e.value) for e in list(app.markdown) + list(app.warning)
                     + list(app.info) + list(app.error) + list(app.success)
                     + list(app.caption))
    faltan = [t for t in esperado if t not in texto]
    if faltan:
        raise AssertionError(f"{etiqueta}: no apareció {faltan}")
    print(f"OK {etiqueta}", flush=True)
    return app


def main():
    logging.disable(logging.WARNING)
    import casos

    disponibles = casos.listar_casos()
    validacion = casos.art_dir_defecto()
    otros = [c for c in disponibles if c != validacion]

    # 1. Caso de validación: es el activo, así que la banda verde y nada de
    #    advertencias de reinicio.
    correr("caso de validación", {f"{CLAVE}_caso": str(validacion)},
           esperado=["es el caso que cargó la app"])

    # 2. Otro caso: tiene que avisar que las demás pestañas no lo muestran.
    if otros:
        correr(f"otro caso ({otros[0].name})", {f"{CLAVE}_caso": str(otros[0])},
               esperado=["Estás **preparando**"])
    else:
        print("… sin otros casos en disco: se omite esa rama", flush=True)

    # 3. Crear caso nuevo: el formulario aparece y no explota sin nombre.
    correr("crear caso nuevo", {f"{CLAVE}_caso": "➕ crear un caso nuevo…"},
           esperado=["Elige un caso existente o crea uno nuevo"])

    # 4. La rama larga: asistente completo con una red cargada. Es la que
    #    recorre validación, zonas, puntos, seccionadores, perfiles y las dos
    #    salidas. Se inyecta la red y los puntos en session_state porque el
    #    asistente corta antes si faltan.
    import pandapower.networks as pn

    import entrada_red as ER
    net, _, ties = ER.adaptar_para_milp(pn.mv_oberrhein(scenario="load"))
    destino = otros[0] if otros else validacion
    estado = {
        f"{CLAVE}_caso": str(destino),
        f"{CLAVE}_net": net,
        f"{CLAVE}_origen": "prueba: mv_oberrhein",
        f"{CLAVE}_ties_sug": ties,
        f"{CLAVE}_bus_zone": ER.bus_zone_inicial(net, 2),
        f"{CLAVE}_puntos": {1: (-33.45, -70.66), 2: (-36.82, -73.05)},
    }
    app = correr("asistente con red cargada", estado,
                 esperado=["Seccionadores existentes", "Perfiles horarios"])
    # La red de prueba no trae seccionadores, así que tiene que avisarlo en vez
    # de inventarlos: es el caso que motivó dejar la lista vacía como válida.
    textos = " ".join(str(e.value) for e in app.info)
    if "Sin seccionadores declarados" not in textos:
        raise AssertionError("no avisó que la lista de seccionadores está vacía")
    print("OK avisa que no hay seccionadores declarados", flush=True)

    print("\nTodas las ramas de la vista se renderizan sin excepciones.")


if __name__ == "__main__":
    sys.exit(main())
