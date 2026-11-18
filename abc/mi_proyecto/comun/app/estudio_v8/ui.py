"""Navegación de la herramienta v8. Cada pestaña vive en su propio módulo."""
import subprocess
import sys

import streamlit as st

import casos
from estudio_v8.core import APP_DIR, ART_DIR, COMBINACION_TESIS, MEDIDAS_TESIS


def _etiqueta_caso(ruta):
    """Nombre del caso más qué le falta, para elegir con criterio."""
    listo, faltan = casos.listo_para_ver(ruta)
    if casos.es_validacion(ruta):
        sufijo = "validación de la tesis"
    elif listo:
        sufijo = "completo"
    else:
        sufijo = "falta " + ", ".join(f.split(" ")[0] for f in faltan)
    return f"{ruta.name} · {sufijo}"


def selector_caso(consulta=False):
    """Selecciona la carpeta de consulta sin modificar el modelo ni reiniciar."""
    sb = st.sidebar
    sb.divider()
    activo = casos.art_dir_activo()
    disponibles = [d.resolve() for d in casos.listar_casos()]
    if activo not in disponibles:
        disponibles.insert(0, activo)
    elegido = sb.selectbox("Caso de estudio", disponibles,
                           index=disponibles.index(activo),
                           format_func=_etiqueta_caso, key="v8_caso")
    sb.caption(f"Artefactos seleccionados: `{elegido}`. "
               "Optimización y análisis lee este caso sin reiniciar.")


def datos():
    """Dos vistas: la red cargada y el origen de los datos.

    Ya no hay ficha de estado acá arriba ni vista "Preparar etapas": las dos
    quedaron duplicadas cuando la pestaña de origen de los datos pasó a
    organizarse por caso de estudio, y ahí el estado de cada etapa se ve junto
    al botón que la calcula, referido al caso correcto. Una ficha aparte
    describía siempre el caso activo del proceso, que puede no ser el que se
    está preparando.
    """
    vista = st.radio("Vista", ["Origen de los datos", "Modelo cargado"],
                     horizontal=True, key="datos_vista")
    if vista == "Modelo cargado":
        try:
            from estudio_v8.entradas_clasicas import render
            render()
        except Exception as exc:                                   # noqa: BLE001
            st.error(f"No se pudo cargar el modelo ({type(exc).__name__}: {exc}). "
                     "Revisa el estado del caso en «Origen de los datos» y corre "
                     "las etapas que falten.")
    else:
        import entrada_red_vista
        entrada_red_vista.render()


def guia():
    from estudio_v8 import estado as E, optimizacion as O
    st.markdown("""### Recorrido de trabajo

| Quiero… | Dónde | Qué se calcula |
|---|---|---|
| Revisar o preparar insumos | 📥 Datos de entrada | Etapas E0–E6 del pipeline |
| Red, plan instalado, indicadores y ENS de una combinación | ⚙️ Optimización (siempre activo) | Combinación × escenarios |
| Curvas MAC | ⚙️ Optimización → 📉 Curvas MAC | + cada medida por separado |
| Solapamiento con medidas activas | ⚙️ Optimización → 🔗 Solapamiento | + individuales y combinación |
| Minimax, Minimin, Maximin, Maximax, minimax regret | ⚙️ Optimización → ⚖️ Reglas de decisión | + planes fijos (inversión fija, operación libre) |
| Elegir escenarios por percentil | ⚙️ Optimización → 5 · Escenarios | Casos base de toda la población si faltan |
| Mirar una corrida hora a hora | 📊 Resultados → 🔍 Ver una corrida | Nada |

Los cálculos corren como procesos aparte y guardan cada corrida apenas termina: se puede cerrar el
navegador, detener y reanudar sin repetir lo guardado. Cambiar filtros o criterios en Resultados nunca
lanza una optimización. Escenarios con índices desde 0, como en los archivos.
""")
    root = ART_DIR / O.ESTUDIO_TESIS
    sup = O.supuestos_estudio(root)[0] if root.exists() else None
    if sup:
        escs = E.contenido_estudio(root)["escenarios"]
        st.markdown(f"#### Comandos exactos para el estudio de la tesis (`{O.ESTUDIO_TESIS}`)")
        comandos = [
            ("Estado de los insumos", [sys.executable, str(APP_DIR / "pipeline.py"), "--estado"]),
            ("Casos base de la muestra", O.cmd_e5(sup["seccionadores"], escs)),
            ("Muestra por percentil (requiere casos base de toda la población)",
             [sys.executable, str(APP_DIR / "seleccionar_escenarios.py"), "--seccionadores",
              ",".join(map(str, sup["seccionadores"]))]),
            ("Medidas individuales (curvas MAC y solapamiento)", O.cmd_barrido(root, sup, MEDIDAS_TESIS, escs, "1,2")),
            ("Combinación de las 12 medidas", O.cmd_barrido(root, sup, MEDIDAS_TESIS, escs, "3", combo=True)),
            ("Recostear DG con el E4 vigente (sin MILP)", O.cmd_recosteo(root, sup, escs)),
            ("Planes fijos (reglas de decisión)", O.cmd_plan_fijo(root, COMBINACION_TESIS, escs, escs, False, sup)),
            ("Rearmar la matriz sin resolver", O.cmd_plan_fijo(root, COMBINACION_TESIS, escs, escs, False, sup,
                                                               solo_matriz=True)),
        ]
        for titulo, cmd in comandos:
            st.markdown(f"**{titulo}**")
            st.code(subprocess.list2cmdline(cmd), language="powershell")
        st.caption("Agrega `--listar` a run_barrido_medidas.py o run_plan_fijo.py para ver el plan sin resolver. "
                   "Las figuras de la tesis se exportan con graficos_validacion.ipynb, que solo lee artefactos.")
    with st.expander("Exportar herramienta y estudio"):
        nombres = O.estudios_disponibles()
        nombre = st.selectbox("Estudio", nombres, key="estudio_exportar") if nombres else None
        completos = st.checkbox("Incluir detalle completo de corridas (.pkl.gz)", value=False, key="exp_completo")
        st.caption("Incluye código, dependencias declaradas, insumos preparados y resultados del estudio. "
                   "No incluye Python, la licencia de Gurobi ni los datos climáticos brutos.")
        if nombre and st.button("Preparar paquete ZIP", key="exp_zip"):
            from estudio_v8.package import exportar
            try:
                with st.spinner("Preparando paquete…"):
                    st.session_state["paquete_v8"] = str(exportar(nombre, completos))
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        if st.session_state.get("paquete_v8"):
            st.success(f"Paquete preparado en {st.session_state['paquete_v8']}")
    for archivo in ("GUIA_BARRIDO.md", "INSTRUCTIVO_DATOS.md", "README_APP_V8.md"):
        path = APP_DIR / archivo
        if path.exists():
            with st.expander(archivo):
                st.markdown(path.read_text(encoding="utf-8"))


def main():
    # Barra lateral abierta como en la v7: fija el mismo ancho de contenido, así las
    # figuras (que se ajustan al contenedor) se ven del mismo tamaño en ambas apps.
    st.set_page_config(page_title="Estudio de resiliencia · v8", page_icon="⚡", layout="wide",
                       initial_sidebar_state="expanded")
    st.title("⚡ Resiliencia de la red de distribución")
    st.sidebar.markdown("### ⚡ Estudio de resiliencia")
    st.sidebar.caption("1 · 📥 Revisa los insumos\n\n2 · ⚙️ Elige análisis y calcula lo que falta\n\n"
                       "3 · 📊 Consulta los resultados")
    from estudio_v8.resultados import navegacion
    paginas = ["📥 Datos de entrada", "⚙️ Optimización y análisis", "📊 Resultados"]
    pagina, espacio = navegacion(paginas, "pagina_v8")
    selector_caso(consulta=pagina == paginas[2])
    with espacio:
        if pagina == paginas[0]:
            datos()
        elif pagina == paginas[1]:
            from estudio_v8 import optimizacion
            optimizacion.render()
        else:
            from estudio_v8 import resultados
            resultados.render()
    st.divider()
    with st.expander("📖 Guía de uso y exportación"):
        if st.checkbox("Abrir guía de uso", key="abrir_guia"):
            guia()
