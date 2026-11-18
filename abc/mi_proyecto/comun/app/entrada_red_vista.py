# ============================================================
# entrada_red_vista.py — Pestaña "🗺️ Datos de entrada"
# ============================================================
# PREPARA INSUMOS, NO RESUELVE NADA. Igual que barrido_vista.py, esta
# pestaña respeta la regla del repo: cálculo largo -> consola, la app
# visualiza y configura.
#
# Todo gira alrededor de un CASO DE ESTUDIO = una carpeta de artefactos
# (ver `casos.py`). Primero se elige el caso, y solo después se muestra
# cualquier cosa: hasta el 2026-09-22 la pestaña abría con el estado de
# `artefactos/` aunque la usuaria hubiera entrado a cargar otra red, o sea
# mostraba datos correctos del caso equivocado, que es peor que no mostrar
# nada.
#
# Con el caso elegido, la pestaña hace dos cosas según lo que encuentre:
#
#   caso armado (tiene red.pkl) -> muestra qué tiene y qué falta
#   caso vacío                  -> asistente paso a paso hasta armarlo
#
# y en los dos casos ofrece las etapas que faltan, con un botón que las
# corre en segundo plano y el comando de consola equivalente al lado.
#
# Las dos salidas que abren las cadenas de cálculo:
#
#   SystemzoneV1.xlsx     -> pipeline climático (ARClim + Open-Meteo)
#   <caso>/red_nueva.pkl  -> pipeline.py --artefactos <caso> --fuente red_nueva
#
# ⚠️ Esta vista NUNCA llama `pipeline.usar_artefactos`: movería la carpeta
# bajo los pies de `data_loader`, `barrido_vista` y `estudio_v8`, que ya
# leyeron la suya al importarse. Para MIRAR otro caso usa
# `casos.estado_caso(ruta)`; para CALCULAR, lanza un subproceso con
# `--artefactos`. Por eso también el caso que se prepara acá no se ve en las
# otras pestañas hasta reiniciar la app (ver `_banner_caso`).
#
# El detalle completo está en INSTRUCTIVO_DATOS.md.
# ============================================================

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import casos
import entrada_red as ER
import pipeline as P
from estudio_v8 import optimizacion as O
from aislacion import variante as variante_aislacion
from estudio_v8.core import indices

# Carpetas Inputs/ del pipeline climático donde vive el SystemzoneV1.xlsx
# vigente. Son varias porque el estudio quedó con copias en paralelo; la
# pestaña ofrece escribir en todas para que no se desincronicen.
DIRS_CLIMA = [
    P.CASO_DIR / "Trend database with hourly resolution" / "Inputs",
    P.CASO_DIR / "Extreme Weather Events database" / "Inputs",
    P.CASO_DIR / "Proyecto oficial" / "Inputs",
]
RUTA_WIND_NC = P.RUTA_WIND_NC

CLAVE = "entrada_red"          # prefijo de session_state


# ════════════════════════════════════════════════════════════
# Estado de la cadena completa
# ════════════════════════════════════════════════════════════

def _insumos_clima():
    """(texto, ok) de los dos insumos climáticos, que son COMPARTIDOS.

    No pertenecen a ningún caso: se generaron para unos puntos geográficos
    concretos, que pueden no ser los del caso que se prepara. Por eso no van en
    una tabla de estado del caso, sino donde importan: en el panel de la cadena
    climática de fallas, que es lo único que los consume.
    """
    lineas, ok = [], True
    sz = next((d / "SystemzoneV1.xlsx" for d in DIRS_CLIMA
               if (d / "SystemzoneV1.xlsx").exists()), None)
    if sz is None:
        lineas.append("⬜ No hay `SystemzoneV1.xlsx` en ninguna carpeta `Inputs/`.")
        ok = False
    else:
        try:
            n = len(pd.read_excel(sz).dropna(subset=["Lat", "Lon"]))
            lineas.append(f"`SystemzoneV1.xlsx` con **{n} zona(s)** "
                          f"({sz.parent.parent.name}).")
        except Exception as e:                                      # noqa: BLE001
            lineas.append(f"⬜ `SystemzoneV1.xlsx` ilegible: {e}")
            ok = False

    if not RUTA_WIND_NC.exists():
        lineas.append(f"⬜ Falta `{RUTA_WIND_NC.name}`: hay que correr el pipeline "
                      f"climático (pasos 1 a 4).")
        ok = False
    else:
        detalle = f"{RUTA_WIND_NC.stat().st_size/1e6:.1f} MB"
        try:
            import xarray as xr
            with xr.open_dataset(RUTA_WIND_NC) as ds:
                detalle += (f", {ds['wind'].shape[0]} h × {ds['wind'].shape[1]} zonas "
                            f"({ds.attrs.get('start_year','?')}-{ds.attrs.get('end_year','?')})")
        except Exception:                                            # noqa: BLE001
            pass
        lineas.append(f"`{RUTA_WIND_NC.name}` ({detalle}).")
    return lineas, ok


# ════════════════════════════════════════════════════════════
# Caso de estudio: qué carpeta de artefactos se está preparando
# ════════════════════════════════════════════════════════════

OPCION_NUEVO = "➕ crear un caso nuevo…"


def _etiqueta_caso(ruta):
    est = casos.estado_caso(ruta)
    partes = []
    if est["es_validacion"]:
        partes.append("caso de validación de la tesis")
    if est["armado"]:
        listas = [e for e, i in est["etapas"].items() if i["existe"]]
        partes.append(f"{len(listas)}/5 etapas")
    else:
        partes.append("vacío")
    if est["es_activo"]:
        partes.append("cargado por la app")
    return f"{ruta.name}  ·  {' · '.join(partes)}"


def _selector_caso():
    """Carpeta del caso a preparar. None mientras no haya una válida.

    Es lo PRIMERO de la pestaña, y nada se muestra antes: el estado de los
    insumos solo significa algo referido a un caso concreto.
    """
    # Aplicar la selección pendiente antes de crear el widget del selector.
    pendiente = st.session_state.pop(f"{CLAVE}_caso_pendiente", None)
    if pendiente is not None:
        st.session_state[f"{CLAVE}_caso"] = pendiente
    disponibles = casos.listar_casos()
    opciones = [str(d) for d in disponibles] + [OPCION_NUEVO]
    activo = str(casos.art_dir_activo())
    inicial = opciones.index(activo) if activo in opciones else 0

    elegido = st.selectbox(
        "Caso de estudio (carpeta de artefactos)", opciones, index=inicial,
        format_func=lambda o: o if o == OPCION_NUEVO else _etiqueta_caso(Path(o)),
        key=f"{CLAVE}_caso",
        help="Cada caso es una red con sus propios perfiles, fallas y "
             "resultados. Trabajar en carpetas separadas es lo que evita "
             "pisar el caso de la tesis al probar otra red.")

    if elegido != OPCION_NUEVO:
        return Path(elegido)

    c1, c2 = st.columns([3, 1])
    nombre = c1.text_input(
        "Nombre del caso nuevo", key=f"{CLAVE}_caso_nuevo",
        placeholder="oberrhein",
        help="Se crea una carpeta por caso dentro de artefactos/.")
    ruta, error = casos.nombre_valido(nombre)
    if c2.button("Crear caso", key=f"{CLAVE}_btn_caso_nuevo",
                 disabled=not nombre, type="primary"):
        if error:
            st.error(error)
        else:
            ruta.mkdir(parents=True)
            (ruta / ".caso").touch()
            st.session_state[f"{CLAVE}_caso_pendiente"] = str(ruta)
            st.rerun()
    if nombre and error:
        st.caption(f"⚠️ {error}")
    elif nombre and ruta:
        st.caption(f"Se creará `{ruta.name}/`")
    return None


def _invalidar_si_cambio_caso(caso):
    """Al cambiar de caso, borra la red y las zonas cargadas del anterior.

    Sin esto, la red de 15 barras seguiría en `session_state` mientras la
    pestaña dice estar preparando otro caso, y se exportaría un paquete con
    la red equivocada. Mismo criterio que ya se aplica al recargar una red
    (`_cargar_red_ui`).

    Corre ANTES de instanciar cualquier widget del asistente y termina en
    `st.rerun()`: Streamlit no deja borrar la clave de un widget en el mismo
    run en que ese widget se crea.
    """
    conservar = {f"{CLAVE}_caso", f"{CLAVE}_caso_nuevo", f"{CLAVE}_btn_caso_nuevo"}
    previo = st.session_state.get(f"{CLAVE}_caso_previo")
    if previo == str(caso):
        return
    if previo is not None:
        for k in [k for k in st.session_state
                  if k.startswith(CLAVE) and k not in conservar
                  and k != f"{CLAVE}_caso_previo"]:
            st.session_state.pop(k, None)
        st.session_state[f"{CLAVE}_caso_previo"] = str(caso)
        st.rerun()
    st.session_state[f"{CLAVE}_caso_previo"] = str(caso)


def _banner_caso(caso, est):
    """Indica cómo consultar el caso preparado sin reiniciar la aplicación."""
    if est["es_activo"]:
        st.success(f"`{caso.name}` es el caso que cargó la app.")
    st.info(f"Estás preparando `{caso.name}`. Para usar sus artefactos en "
            "Optimización y análisis, selecciónalo en «Caso de estudio» de la "
            "barra lateral. No hace falta reiniciar.")


def _aviso_fallas(est, caso):
    """De dónde salieron las fallas de este caso, y qué implica.

    Cubre los tres orígenes posibles, y los tres importan para leer un
    resultado: el `.nc` real (lo válido), el fallback plano de literatura
    (vegetación y refuerzo sin curva de fragilidad) y las fallas sintéticas
    de `fallas_sinteticas.py` (nada validado).
    """
    meta = est["fallas_meta"]
    if not meta:
        return
    fuente = meta.get("fuente_viento")
    if meta.get("no_validado") or fuente == "sintetico":
        st.error(
            "**Las fallas de este caso son SINTÉTICAS y no valen para la tesis.** "
            f"Las generó `{meta.get('generador', 'fallas_sinteticas.py')}`: "
            "tormentas inventadas, sin viento ni curva de fragilidad. Sirven para "
            "comprobar que el modelo corre.\n\n"
            "Consecuencia concreta: las matrices mitigadas son **iguales** a la "
            "base, así que vegetación, refuerzo y soterramiento van a dar ahorro "
            "cero por construcción, y eso no es un resultado.")
        return
    if fuente == "nc_real":
        st.success("E2 calibrado con el viento real del .nc (`fuente_viento = nc_real`).")
        return
    st.warning(
        "**E2 se calculó sin el .nc de viento** (`fuente_viento = "
        f"{fuente}`). Vegetación y refuerzo estructural "
        "quedaron con un porcentaje plano de literatura (ρ_veg = 0.30, "
        "ρ_ref = 0.42) en vez de la curva de fragilidad calibrada hora a hora.\n\n"
        + (f"El `.nc` **ya está disponible**, así que conviene rehacer la etapa:\n\n"
           f"```\npython pipeline.py --artefactos {caso.name} --etapas E2 --forzar\n```\n"
           "Ojo: eso cambia `FaultlinMC_veg/_ref/_ug` y por lo tanto los "
           "resultados de esas medidas, no los del caso base."
           if RUTA_WIND_NC.exists() else
           "Falta generar el `.nc` con el pipeline climático."))


# ════════════════════════════════════════════════════════════
# Paso 1 — cargar la red
# ════════════════════════════════════════════════════════════

def _cargar_red_ui():
    st.markdown("Elige de dónde sale la topología. La opción **CIGRE MV** "
                "reproduce la red del estudio actual.")

    fmt = st.radio("Formato", list(ER.FORMATOS),
                   format_func=lambda k: ER.FORMATOS[k],
                   horizontal=True, key=f"{CLAVE}_fmt")

    net, origen, avisos = None, "", []

    if fmt == "cigre":
        aerea = st.checkbox(
            "Convertir a línea aérea de MT (parámetros del estudio)", value=True,
            help="La CIGRE MV de pandapower viene con cable subterráneo. El "
                 "estudio modela línea aérea, que es la expuesta al viento: "
                 "sin este cambio la curva de fragilidad no aplica.")
        if st.button("Cargar CIGRE MV", key=f"{CLAVE}_btn_cigre"):
            net = ER.red_cigre(aerea=aerea)
            origen = f"CIGRE MV ({'aérea' if aerea else 'original'})"

    elif fmt == "pandapower":
        c1, c2 = st.columns(2)
        subido = c1.file_uploader("Archivo pandapower", type=["json", "p", "pkl", "xlsx"],
                                  key=f"{CLAVE}_up_pp")
        ruta_txt = c2.text_input("…o ruta local", key=f"{CLAVE}_ruta_pp",
                                 placeholder=r"C:\...\mi_red.json")
        if st.button("Cargar red", key=f"{CLAVE}_btn_pp"):
            try:
                if subido is not None:
                    tmp = P.ART_DIR / f"_subida_red{Path(subido.name).suffix}"
                    tmp.parent.mkdir(parents=True, exist_ok=True)
                    tmp.write_bytes(subido.getbuffer())
                    net, origen = ER.desde_pandapower(tmp), f"pandapower: {subido.name}"
                elif ruta_txt.strip():
                    net = ER.desde_pandapower(ruta_txt.strip())
                    origen = f"pandapower: {Path(ruta_txt).name}"
                else:
                    st.warning("Sube un archivo o escribe una ruta.")
            except Exception as e:
                st.error(f"No se pudo leer la red: {type(e).__name__}: {e}")

    elif fmt == "opendss":
        st.caption(
            "Un caso OpenDSS son varios archivos que el Master.DSS incluye con "
            "`Redirect`, así que no basta subir uno solo: se pide la **ruta local** "
            "del Master.")
        ruta_txt = st.text_input("Ruta al Master.DSS", key=f"{CLAVE}_ruta_dss",
                                 placeholder=r"C:\...\IEEE_13_Bus-G\Master.DSS")
        st.caption("Se importan buses, líneas, cargas y coordenadas. Reguladores, "
                   "capacitores y curvas horarias se ignoran: el MILP no los usa.")
        if st.button("Importar desde OpenDSS", key=f"{CLAVE}_btn_dss"):
            try:
                net, avisos = ER.desde_opendss(ruta_txt.strip())
                origen = f"OpenDSS: {Path(ruta_txt).name}"
            except Exception as e:
                st.error(f"No se pudo importar el caso DSS: {type(e).__name__}: {e}")

    elif fmt == "csv":
        with st.expander("Formato esperado de cada tabla"):
            for nombre, texto in ER.PLANTILLA_CSV.items():
                st.markdown(f"**{nombre}.csv**")
                st.code(texto, language="csv")
        c1, c2, c3 = st.columns(3)
        f_bus = c1.file_uploader("buses.csv", type=["csv"], key=f"{CLAVE}_csv_bus")
        f_lin = c2.file_uploader("lineas.csv", type=["csv"], key=f"{CLAVE}_csv_lin")
        f_car = c3.file_uploader("cargas.csv", type=["csv"], key=f"{CLAVE}_csv_car")
        sn = st.number_input("Capacidad del transformador de cabecera [MVA]",
                             min_value=0.1, value=25.0, step=0.5,
                             key=f"{CLAVE}_csv_sn",
                             help="El MILP la usa como cota de inyección del alimentador.")
        if st.button("Construir red desde CSV", key=f"{CLAVE}_btn_csv"):
            if not (f_bus and f_lin and f_car):
                st.warning("Faltan tablas: se necesitan las tres.")
            else:
                try:
                    net = ER.desde_csv(pd.read_csv(f_bus), pd.read_csv(f_lin),
                                       pd.read_csv(f_car), sn_mva_trafo=sn)
                    origen = "CSV"
                except Exception as e:
                    st.error(f"No se pudo construir la red: {type(e).__name__}: {e}")

    if net is not None:
        st.session_state[f"{CLAVE}_net"] = net
        st.session_state[f"{CLAVE}_origen"] = origen
        st.session_state[f"{CLAVE}_avisos"] = avisos
        # Zonas y puntos se recalculan al cambiar de red: mantener los de la
        # red anterior es la forma más silenciosa de mapear mal el viento.
        st.session_state.pop(f"{CLAVE}_bus_zone", None)
        st.session_state.pop(f"{CLAVE}_puntos", None)
        st.success(f"Red cargada: {origen}")


# ════════════════════════════════════════════════════════════
# Paso 2 — validación
# ════════════════════════════════════════════════════════════

def _validacion_ui(net):
    r = ER.resumen(net)
    c = st.columns(5)
    c[0].metric("Buses", r["buses"])
    c[1].metric("Líneas", r["lineas"])
    # Con `scaling` aplicado, o sea la demanda que va a ver el modelo.
    c[2].metric("Demanda base", f"{r['demanda_kw']:,.0f} kW",
                help="Suma de P_D_base: potencia de las cargas en servicio "
                     "multiplicada por su `scaling`, que es el escenario de "
                     "carga que trae la red.")
    c[3].metric("Largo total", f"{r['largo_km']:.2f} km")
    if r["capacidad_kw"]:
        holgura = 100 * (1 - r["demanda_kw"] / r["capacidad_kw"])
        c[4].metric("Capacidad de subestaciones", f"{r['capacidad_kw']:,.0f} kW",
                    delta=f"{holgura:.0f}% de holgura",
                    delta_color="normal" if holgura > 10 else "inverse",
                    help=f"{r['trafo_mva']:.1f} MVA por cabecera. Si la demanda "
                         f"la supera, el modelo recorta carga incluso sin fallas.")
    else:
        c[4].metric("Trafo cabecera", "—")

    hallazgos = ER.validar(net)
    errores = [m for n, m in hallazgos if n == "error"]
    for nivel, msg in hallazgos:
        {"error": st.error, "aviso": st.warning, "ok": st.success}[nivel](msg)
    for msg in st.session_state.get(f"{CLAVE}_avisos", []):
        st.warning(msg)

    # Los errores de numeración son mecánicos y se arreglan renumerando, así
    # que no tiene sentido dejar a la usuaria trabada: casi ninguna red
    # bajada de un catálogo cumple los supuestos del modelo. Se ofrece el
    # arreglo acá mismo, y se muestra TODO lo que cambió — renumerar una red
    # en silencio sería peor que el error.
    if errores:
        st.markdown("##### Adaptar la red a los supuestos del modelo")
        st.caption(
            "Renumera los buses 0..n-1 dejando las cabeceras de subestación "
            "en 1 y 12, renumera las líneas 0..m-1, y reinicia el índice de "
            "las tablas de elementos. No cambia ni la topología, ni los "
            "parámetros eléctricos, ni las coordenadas: solo los números con "
            "que se llaman las cosas.")
        subs = ER.buses_subestacion(net)
        st.caption(f"Cabeceras detectadas en esta red: **{subs or 'ninguna'}** "
                   f"(lado de baja de los transformadores, o los buses de "
                   f"`ext_grid` si no hay transformador).")
        if st.button("🔧 Adaptar la red", key=f"{CLAVE}_btn_adaptar",
                     type="primary"):
            try:
                adaptada, cambios, ties = ER.adaptar_para_milp(net)
            except Exception as e:
                st.error(f"No se pudo adaptar: {type(e).__name__}: {e}")
            else:
                st.session_state[f"{CLAVE}_net"] = adaptada
                st.session_state[f"{CLAVE}_cambios"] = cambios
                st.session_state[f"{CLAVE}_ties_sug"] = ties
                # La renumeración invalida cualquier zona o punto elegido
                # sobre los índices viejos.
                st.session_state.pop(f"{CLAVE}_bus_zone", None)
                st.session_state.pop(f"{CLAVE}_puntos", None)
                st.rerun()

    cambios = st.session_state.get(f"{CLAVE}_cambios") or []
    if cambios:
        with st.expander(f"Red adaptada — {len(cambios)} cambio(s) de numeración",
                         expanded=bool(errores)):
            for c in cambios:
                st.markdown(f"- {c}")
            st.caption("Los números de bus y de línea de esta tabla ya NO son "
                       "los del archivo original. Si vas a cruzar resultados "
                       "con tu fuente, usa la columna de nombre.")

    if r["trafo_mva"] is None:
        sn = st.number_input("Declarar capacidad del transformador de cabecera [MVA]",
                             min_value=0.1, value=25.0, step=0.5, key=f"{CLAVE}_sn_fix")
        if st.button("Añadir transformador", key=f"{CLAVE}_btn_sn"):
            ER.asegurar_trafo(net, sn)
            st.session_state[f"{CLAVE}_net"] = net
            st.rerun()

    with st.expander("Ver tablas de la red"):
        st.markdown("**Buses**")
        st.dataframe(net.bus, use_container_width=True)
        st.markdown("**Líneas**")
        st.dataframe(net.line, use_container_width=True)
        st.markdown("**Cargas**")
        st.dataframe(net.load, use_container_width=True)

    return not errores


# ════════════════════════════════════════════════════════════
# Paso 3 — zonas por bus
# ════════════════════════════════════════════════════════════

def _zonas_ui(net):
    st.markdown(
        "La **zona climática** es el puente entre las dos cadenas: define de "
        "qué punto geográfico toma el viento cada línea. Una línea hereda la "
        "zona de su **bus de origen** (`from_bus`), igual que en el notebook y "
        "en `pipeline.py`.")

    n_zonas = st.number_input("N° de zonas climáticas", 1, 20,
                              value=int(st.session_state.get(f"{CLAVE}_nz", 3)),
                              key=f"{CLAVE}_nz")

    if f"{CLAVE}_bus_zone" not in st.session_state:
        st.session_state[f"{CLAVE}_bus_zone"] = ER.bus_zone_inicial(net, n_zonas)
    bus_zone = st.session_state[f"{CLAVE}_bus_zone"]

    df = pd.DataFrame({
        "Bus": list(net.bus.index),
        "Nombre": [net.bus.at[b, "name"] for b in net.bus.index],
        "Demanda [kW]": [ER.p_d_base(net).get(int(b), 0.0) for b in net.bus.index],
        "Zona": [int(bus_zone.get(int(b), 1)) for b in net.bus.index],
    })
    editado = st.data_editor(
        df, hide_index=True, use_container_width=True, key=f"{CLAVE}_ed_zonas",
        column_config={
            "Bus": st.column_config.NumberColumn(disabled=True),
            "Nombre": st.column_config.TextColumn(disabled=True),
            "Demanda [kW]": st.column_config.NumberColumn(disabled=True, format="%.1f"),
            "Zona": st.column_config.NumberColumn(min_value=1, max_value=int(n_zonas), step=1),
        })
    bus_zone = {int(r.Bus): int(r.Zona) for r in editado.itertuples()}
    st.session_state[f"{CLAVE}_bus_zone"] = bus_zone

    zonas_usadas = sorted(set(bus_zone.values()))
    resumen_zona = pd.DataFrame([{
        "Zona": z,
        "Buses": sum(1 for v in bus_zone.values() if v == z),
        "Líneas": sum(1 for l in net.line.index
                      if bus_zone.get(int(net.line.from_bus[l])) == z),
        "Demanda [kW]": sum(v for b, v in ER.p_d_base(net).items() if bus_zone.get(b) == z),
    } for z in zonas_usadas])
    st.dataframe(resumen_zona, hide_index=True, use_container_width=True)

    vacias = [z for z in range(1, int(n_zonas) + 1) if z not in zonas_usadas]
    if vacias:
        st.info(f"Zonas sin ningún bus: {vacias}. No se exportarán a SystemzoneV1.xlsx "
                f"(un punto climático sin líneas asociadas solo agrega descargas.)")
    return bus_zone, zonas_usadas


# ════════════════════════════════════════════════════════════
# Paso 4 — puntos geográficos
# ════════════════════════════════════════════════════════════

def _puntos_ui(net, bus_zone, zonas_usadas):
    coords = ER.coords_de_net(net)
    diag = ER.diagnostico_coords(coords)

    if diag == "chile":
        st.success(f"La red trae coordenadas geográficas en Chile "
                   f"({len(coords)} buses). Se proponen como centroide por zona.")
    elif diag == "fuera_de_chile":
        st.warning(
            "La red trae coordenadas, pero **no son lat/lon de Chile** — casi "
            "siempre son las cartesianas de dibujo del caso (la CIGRE de "
            "pandapower, por ejemplo, viene con (7, 16)). Hay que fijar el punto "
            "de cada zona a mano: ARClim y Open-Meteo se consultan con lat/lon "
            "reales y con cualquier otra cosa devuelven el punto de grilla "
            "equivocado, sin error.")
    else:
        st.info("La red no trae coordenadas. Fija el punto de cada zona abajo.")

    if f"{CLAVE}_puntos" not in st.session_state:
        base = ER.puntos_por_zona(net, bus_zone)
        if not base:
            base = _puntos_desde_systemzone(zonas_usadas)
        st.session_state[f"{CLAVE}_puntos"] = base
    puntos = st.session_state[f"{CLAVE}_puntos"]

    c1, c2 = st.columns([1, 3])
    if c1.button("Cargar del SystemzoneV1 actual", key=f"{CLAVE}_btn_sz"):
        st.session_state[f"{CLAVE}_puntos"] = _puntos_desde_systemzone(zonas_usadas)
        st.rerun()
    c2.caption("Lee el SystemzoneV1.xlsx que ya usa el pipeline climático, "
               "para partir de los puntos con los que se generó el `.nc` vigente.")

    comunas = st.session_state.get(f"{CLAVE}_comunas", {})
    df = pd.DataFrame([{
        "Zona": z,
        "Lat": float(puntos.get(z, (np.nan, np.nan))[0]),
        "Lon": float(puntos.get(z, (np.nan, np.nan))[1]),
        "Comuna": comunas.get(z, ""),
    } for z in zonas_usadas])

    editado = st.data_editor(
        df, hide_index=True, use_container_width=True, key=f"{CLAVE}_ed_puntos",
        column_config={
            "Zona": st.column_config.NumberColumn(disabled=True),
            "Lat": st.column_config.NumberColumn(format="%.4f", min_value=-56.0, max_value=-17.0),
            "Lon": st.column_config.NumberColumn(format="%.4f", min_value=-76.0, max_value=-66.0),
            "Comuna": st.column_config.TextColumn(help="Solo documental; no se usa en el cálculo."),
        })

    puntos = {int(r.Zona): (float(r.Lat), float(r.Lon))
              for r in editado.itertuples()
              if pd.notna(r.Lat) and pd.notna(r.Lon)}
    st.session_state[f"{CLAVE}_puntos"] = puntos
    st.session_state[f"{CLAVE}_comunas"] = {int(r.Zona): str(r.Comuna or "")
                                            for r in editado.itertuples()}

    faltan = [z for z in zonas_usadas if z not in puntos]
    if faltan:
        st.error(f"Zonas sin punto geográfico: {faltan}. El pipeline climático "
                 f"necesita una fila por zona.")
    else:
        st.map(pd.DataFrame([{"lat": la, "lon": lo} for la, lo in puntos.values()]),
               zoom=5)
        fuera = [z for z, (la, lo) in puntos.items() if not ER._en_chile(la, lo)]
        if fuera:
            st.warning(f"Zonas fuera de Chile continental: {fuera}. ARClim no "
                       f"tiene datos ahí.")
    return puntos, not faltan


def _puntos_desde_systemzone(zonas_usadas):
    """Puntos del SystemzoneV1.xlsx vigente, recortados a las zonas que la
    red realmente usa."""
    for d in DIRS_CLIMA:
        ruta = d / "SystemzoneV1.xlsx"
        if not ruta.exists():
            continue
        try:
            df = pd.read_excel(ruta).dropna(subset=["Lat", "Lon"])
            return {int(r["Zona"]): (float(r["Lat"]), float(r["Lon"]))
                    for _, r in df.iterrows() if int(r["Zona"]) in zonas_usadas}
        except Exception:
            continue
    return {}


# ════════════════════════════════════════════════════════════
# Paso 5 — parámetros del estudio y exportación
# ════════════════════════════════════════════════════════════

def _systemzone_bytes(df_sz):
    """El Excel de zonas como bytes, sin pasar por disco."""
    import io
    buffer = io.BytesIO()
    df_sz.to_excel(buffer, index=False)
    return buffer.getvalue()


def _seccionadores_ui(net, ties):
    """Qué líneas llevan seccionador. Puede quedar vacío, y eso es válido.

    El dato cambia radicalmente los indicadores: sin seccionadores, una falla
    deja fuera de servicio todo el alimentador hasta que se repara; con ellos,
    solo la sección donde ocurrió. Por eso es un DATO de la red y no una
    decisión del modelo.

    Se muestra la tabla de switches de la red cargada porque es de donde sale
    la información, pero un switch NO es necesariamente un seccionador de
    tramo: en MV Oberrhein, por ejemplo, son switches de extremo de línea y
    ninguno es seccionador. Por eso no se deducen automáticamente: se
    proponen y se marcan.
    """
    st.markdown("#### Seccionadores existentes")
    sw = getattr(net, "switch", None)
    tiene_sw = sw is not None and len(sw)

    if tiene_sw:
        cerrados = sorted({int(x) for x in sw[(sw.et == "l") & (sw.closed.astype(bool))].element
                           if int(x) in net.line.index})
        st.caption(
            f"La red trae {len(sw)} switches: {len(cerrados)} cerrados sobre línea "
            f"y {len(ties)} abiertos (los enlaces de respaldo de arriba). Un switch "
            f"cerrado **puede** ser un seccionador de tramo o puede ser solo un "
            f"switch de extremo de línea, que no aísla nada. Marca los que de "
            f"verdad seccionan.")
        with st.expander(f"Ver los {len(sw)} switches de la red"):
            st.dataframe(sw, use_container_width=True)
    else:
        cerrados = []
        st.caption("La red no trae tabla de switches, así que los seccionadores "
                   "hay que declararlos a mano (o dejarlo vacío).")

    candidatas = [l for l in net.line.index if l not in set(ties)]
    seccionadores = st.multiselect(
        "Líneas con seccionador", candidatas,
        default=[l for l in st.session_state.get(f"{CLAVE}_seccionadores", [])
                 if l in candidatas],
        key=f"{CLAVE}_seccionadores",
        help="Vacío = aislación perfecta implícita (el modelo saca de servicio "
             "solo la línea averiada). Es el supuesto optimista con el que se "
             "corrió el estudio; la variante realista necesita esta lista.")
    if not seccionadores:
        st.info("Sin seccionadores declarados: la variante de aislación realista "
                "dejará fuera el alimentador completo ante cualquier falla, y las "
                "medidas de inversión en switches no tendrán con qué compararse. "
                "Es un supuesto legítimo, pero conviene que sea a propósito.")
    elif tiene_sw and cerrados:
        ajenos = [l for l in seccionadores if l not in cerrados]
        if ajenos:
            st.caption(f"Ojo: {ajenos} no tienen switch cerrado en la red cargada. "
                       f"Se acepta (puedes estar modelando un seccionador que la "
                       f"fuente no traía), pero que sea deliberado.")


def _perfiles_ui(caso):
    """Archivos de perfiles del caso: demanda, solar y eólico.

    Los tres archivos de `app/` son los del estudio de validación. Para otra
    red son un préstamo, y heredarlos en silencio es el error que se quiere
    evitar: se terminaría concluyendo sobre una red con la curva de carga de
    otra. Acá se eligen explícitamente, y si se heredan queda dicho.
    """
    st.markdown("#### Perfiles horarios")
    defaults = {"demanda": P.RUTA_PERFIL_DEMANDA, "solar": P.RUTA_PERFIL_SOLAR,
                "eolico": P.RUTA_PERFIL_EOLICO}
    propios = dict(st.session_state.get(f"{CLAVE}_perfiles") or {})

    cols = st.columns(3)
    etiquetas = {"demanda": "Demanda (.csv, columna Power)",
                 "solar": "Solar por zona (.xlsx, columnas Z*)",
                 "eolico": "Eólico por zona (.xlsx, columnas Z*)"}
    for col, (clave, base) in zip(cols, defaults.items()):
        valor = col.text_input(
            etiquetas[clave], value=str(propios.get(clave) or base),
            key=f"{CLAVE}_perf_{clave}",
            help="Ruta al archivo. Puede ser absoluta o relativa a app/.")
        propios[clave] = valor.strip()

    st.session_state[f"{CLAVE}_perfiles"] = {
        k: v for k, v in propios.items() if v and Path(v) != defaults[k]}

    heredados = [k for k, v in propios.items() if not v or Path(v) == defaults[k]]
    faltantes = [f"{k}: {v}" for k, v in propios.items()
                 if v and not (Path(v) if Path(v).is_absolute() else P.DATOS_DIR / v).exists()]
    if faltantes:
        st.error(f"No existen: {faltantes}. E1 fallaría al buscarlos.")
    if heredados and not casos.es_validacion(caso):
        st.warning(
            f"Este caso va a usar los perfiles del estudio de validación para "
            f"**{', '.join(heredados)}**. Son los de la red de 15 barras: la "
            f"forma horaria se hereda tal cual, así que los resultados de este "
            f"caso llevan la curva de carga de otro sistema. Sirve para probar "
            f"el modelo, no para concluir.")


def _exportar_ui(caso, net, bus_zone, puntos):
    st.markdown("#### Parámetros del estudio")
    c1, c2, c3 = st.columns(3)
    numesc = c1.number_input("Escenarios Monte Carlo", 1, 1000, 100, key=f"{CLAVE}_numesc",
                             help="Columnas de FaultlinMC. Cada uno es una realización "
                                  "distinta del año de fallas.")
    n_horas = c2.number_input("Horas del horizonte", 24, 17520, 8760, step=24,
                              key=f"{CLAVE}_nhoras")
    n_eventos = c3.number_input("N_eventos (multiplicador anual del costo de ENS)",
                                1, 100, 1, key=f"{CLAVE}_nev",
                                help="Multiplica el costo de ENS en la función objetivo "
                                     "para anualizarlo.")

    horas_rep = st.number_input(
        "Horas de reparación de una línea fallada", 1, 168,
        value=int(P.horas_reparacion_registradas() or P.HORAS_REPARACION_DEFAULT),
        key=f"{CLAVE}_hrep",
        help="Cuánto tarda en reponerse una línea. Es un parámetro del estudio, "
             "no una constante: entra directo en SAIDI y ENS, y define cuánto "
             "vale toda medida que acorte la indisponibilidad. El mismo valor "
             "va al Monte Carlo (--horas-espera) y a E2 (--horas-reparacion).")

    c4, c5 = st.columns(2)
    # El default [6, 11, 14] son las barras críticas de la red de 15 barras del
    # estudio: en otra red esos números no significan nada, así que solo se
    # proponen para el caso de validación.
    criticos_def = ([b for b in (6, 11, 14) if b in net.bus.index]
                    if casos.es_validacion(caso) else [])
    criticos = c4.multiselect("Buses críticos", list(net.bus.index),
                              default=criticos_def,
                              key=f"{CLAVE}_criticos",
                              help="Su VoLL se multiplica ×100 cuando se activa la "
                                   "medida de priorización. En una red nueva hay "
                                   "que elegirlos: no hay default que valga.")
    # Por defecto, las líneas con switch ABIERTO, que es lo que significa un
    # enlace de respaldo. Antes el default eran "las tres últimas", una
    # convención que solo era cierta en la red del estudio.
    sugeridas = (st.session_state.get(f"{CLAVE}_ties_sug")
                 or ER.tie_indices_sugeridas(net)
                 or list(net.line.index[-3:]))
    ties = c5.multiselect("Enlaces de respaldo (tie-lines)", list(net.line.index),
                          default=[l for l in sugeridas if l in net.line.index],
                          key=f"{CLAVE}_ties",
                          help="Líneas normalmente abiertas: candidatas de la "
                               "reconfiguración topológica. El resto son candidatas "
                               "a seccionador. El valor por defecto son las líneas "
                               "con switch abierto en la red cargada.")
    if ER.tie_indices_sugeridas(net):
        c5.caption(f"Detectadas por switch abierto: {ER.tie_indices_sugeridas(net)}")
    else:
        c5.caption("La red no trae switches abiertos, así que los enlaces de "
                   "respaldo hay que elegirlos a mano.")

    _seccionadores_ui(net, ties)
    _perfiles_ui(caso)

    st.divider()
    st.markdown("#### Salida 1 — `SystemzoneV1.xlsx` (cadena climática)")
    df_sz = ER.df_systemzone(puntos, st.session_state.get(f"{CLAVE}_comunas", {}))
    st.dataframe(df_sz, hide_index=True, use_container_width=True)

    # El Excel se arma EN MEMORIA para el botón de descarga. Antes se escribía
    # en disco en cada re-ejecución del script —o sea en cada interacción con
    # cualquier widget de la pestaña— y el archivo aparecía modificado en git
    # sin que nadie hubiera pedido nada.
    st.download_button(
        "⬇️ Descargar SystemzoneV1.xlsx", _systemzone_bytes(df_sz),
        file_name="SystemzoneV1.xlsx", key=f"{CLAVE}_dl_sz",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    destinos = st.multiselect(
        "…o escribirlo directamente en las carpetas del pipeline climático",
        [str(d) for d in DIRS_CLIMA if d.exists()], key=f"{CLAVE}_dest_sz")
    if destinos and st.button("Escribir SystemzoneV1.xlsx en esas carpetas",
                              key=f"{CLAVE}_btn_sz_w"):
        for d in destinos:
            ER.exportar_systemzone(puntos, Path(d) / "SystemzoneV1.xlsx",
                                   st.session_state.get(f"{CLAVE}_comunas", {}))
        st.success(f"Escrito en {len(destinos)} carpeta(s). Ojo: los `.nc` ya "
                   f"generados siguen correspondiendo a los puntos ANTERIORES "
                   f"hasta que vuelvas a correr el pipeline climático.")

    st.divider()
    st.markdown(f"#### Salida 2 — `{caso.name}/red_nueva.pkl` (cadena del modelo)")
    st.caption(
        "Contiene solo lo que E0/E1 rescatan hoy de `app.pkl`: red, zonas, "
        "demanda base, seccionadores, perfiles y los parámetros de arriba. No se "
        "escribe un `app.pkl` nuevo — son 740 MB donde la red convive con los "
        "resultados congelados del estudio, y de ahí el pipeline solo lee seis "
        "claves.")

    if st.button("💾 Guardar red_nueva.pkl", key=f"{CLAVE}_btn_pkg", type="primary"):
        net_geo = ER.aplicar_geo(net, bus_zone, puntos)
        ruta = ER.guardar_paquete(
            net_geo, bus_zone, puntos,
            numesc=numesc, n_eventos=n_eventos, buses_criticos=criticos,
            tie_indices=ties, n_horas=n_horas,
            comunas=st.session_state.get(f"{CLAVE}_comunas", {}),
            horas_reparacion=horas_rep,
            seccionadores=st.session_state.get(f"{CLAVE}_seccionadores", []),
            perfiles=st.session_state.get(f"{CLAVE}_perfiles"),
            origen=st.session_state.get(f"{CLAVE}_origen", ""),
            ruta=caso / "red_nueva.pkl")
        st.success(f"Guardado: `{ruta}`")
        st.rerun()


# ════════════════════════════════════════════════════════════
# Caso ya armado: qué red tiene
# ════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _resumen_red_cacheado(ruta_str, mtime):
    """`casos.resumen_red` cacheado por mtime de red.pkl.

    Carga un pickle con una red de pandapower dentro, así que sin caché se
    pagaría en cada interacción de widget (Streamlit re-ejecuta el script
    completo). `mtime` está en la firma justamente para que el caché se
    invalide cuando la etapa E0 reescribe el archivo.
    """
    r = casos.resumen_red(ruta_str)
    if r:
        r.pop("net", None)        # la red no es serializable para el caché
    return r


def _resumen_caso(caso, est):
    r = _resumen_red_cacheado(str(caso), (caso / "red.pkl").stat().st_mtime)
    if not r:
        return
    c = st.columns(5)
    c[0].metric("Barras", r["n_buses"])
    c[1].metric("Líneas", r["n_lineas"])
    c[2].metric("Demanda base", f"{r['demanda_kw']:,.0f} kW")
    c[3].metric("Zonas", len(r["zonas"]))
    c[4].metric("Escenarios", r["numesc"] or "—")

    detalle = {
        "Enlaces de respaldo (tie-lines)": r["tie_indices"] or "ninguno",
        "Seccionadores declarados": r["seccionadores"] or
            "ninguno — la aislación de fallas deja fuera el alimentador completo",
        "Barras críticas": r["buses_criticos"] or "ninguna",
        "Largo total": f"{r['largo_km']:.1f} km" if r["largo_km"] else "—",
    }
    paq = est["paquete"] or {}
    if paq.get("origen"):
        detalle["Origen de la red"] = paq["origen"]
    if paq.get("perfiles"):
        detalle["Perfiles propios del caso"] = ", ".join(
            Path(v).name for v in paq["perfiles"].values())
    with st.expander("Características de la red de este caso", expanded=False):
        for k, v in detalle.items():
            st.markdown(f"- **{k}**: {v}")


# ════════════════════════════════════════════════════════════
# Etapas: botón en segundo plano + comando copiable
# ════════════════════════════════════════════════════════════

def _cmd_pipeline(caso, etapas, *, fuente=None, escenarios=None,
                  seccionadores=None, forzar=False, lim_dg=None,
                  lim_dg_modo=None, etiqueta=None, horas=None):
    cmd = [sys.executable, str(P.APP_DIR / "pipeline.py"),
           "--artefactos", str(caso), "--etapas", ",".join(etapas)]
    if fuente:
        cmd += ["--fuente", fuente]
    if escenarios:
        cmd += ["--escenarios", escenarios]
    if seccionadores:
        cmd += ["--seccionadores", _lista(seccionadores)]
    for flag, valor in (("--lim-dg", lim_dg), ("--lim-dg-modo", lim_dg_modo),
                        ("--etiqueta", etiqueta), ("--horas", horas)):
        if valor is not None:
            cmd += [flag, str(valor)]
    if forzar:
        cmd += ["--forzar"]
    return cmd


def _dir_baselines(caso, var):
    """Carpeta de casos base de una variante de aislación (ver pipeline.base_dir)."""
    raiz = caso / "baselines"
    return raiz if var == "base" else raiz / var


# Referencia de último recurso: lo que tarda un caso base de E5 en la red de 15
# barras del estudio (`optimizacion.SEG_POR_E5`). NO sirve para otra red ni otra
# máquina —el mismo caso base de 720 h tardó 18,7 s en un equipo y 60,2 s en
# otro, y la red de 179 barras es ~12 veces más grande— así que se usa solo
# mientras el caso no tenga un tiempo propio medido, y diciéndolo.
BARRAS_REFERENCIA = 15
HORAS_REFERENCIA = 8760


def _estimacion(est, etapa, n_horas, n_buses, factor=1):
    """(segundos estimados, de dónde sale). Prefiere lo MEDIDO en este caso.

    `meta.json` guarda cuánto tardó cada etapa en esta carpeta y esta máquina
    (lo escribe E6). Si esa etapa ya corrió, se reescala por horizonte y se
    usa: es el único número que no es una suposición. Si no, se parte de la
    referencia de 15 barras escalada por tamaño, que es una cota gruesa y se
    presenta como tal.
    """
    medidos = ((est.get("meta") or {}).get("duraciones") or {}).get(etapa)
    if medidos and medidos.get("segundos"):
        seg, horas_medidas = float(medidos["segundos"]), medidos.get("horas")
        if etapa == "E5" and medidos.get("escenarios"):
            seg /= float(medidos["escenarios"])       # a segundos por escenario
        if horas_medidas and n_horas:
            seg *= n_horas / float(horas_medidas)
        return seg * factor, f"medido en este caso ({medidos.get('cuando', '')[:10]})"

    seg = O.SEG_POR_E5 * factor
    if n_buses:
        seg *= n_buses / BARRAS_REFERENCIA
    if n_horas:
        seg *= n_horas / HORAS_REFERENCIA
    return seg, "estimación gruesa desde la red de 15 barras: sin medición propia"


def _panel_etapas(caso, est):
    """Las etapas que faltan, con botón y comando.

    Reutiliza `optimizacion.panel_ejecucion`, que ya resuelve lo difícil:
    lanza el subproceso desacoplado, muestra el comando exacto que va a
    correr, sigue el progreso leyendo el log y permite detenerlo. El comando
    copiable sale de la MISMA lista que se ejecuta, así que no pueden
    divergir.
    """
    st.markdown("### Etapas del pipeline")
    # De dónde sale la red en E0/E1. `app_pkl` es el estudio congelado de la
    # tesis (711 MB) y su red es la CIGRE de 15 barras, así que SOLO tiene
    # sentido para el caso de validación. Para cualquier otro caso la red sale
    # de su propio paquete: usar app_pkl lo llenaría con la red del estudio sin
    # dar error, y se descubriría cuando los resultados no calcen con la red que
    # se subió.
    hay_paquete = (caso / "red_nueva.pkl").exists()
    # `red_nueva` no significa "hay paquete": es el camino que NO lee el
    # app.pkl. E1 por ese camino solo necesita el `red.pkl` del caso más los
    # archivos de perfiles, así que funciona sin paquete (verificado). Lo único
    # que el paquete habilita es E0, que es quien lo lee.
    fuente = "app_pkl" if (est["es_validacion"] and not hay_paquete) else "red_nueva"
    puede_e0 = hay_paquete or est["es_validacion"]

    if not est["armado"] and not puede_e0:
        st.info("Este caso todavía no tiene red. Usa el asistente de abajo "
                "para cargarla y guardar el paquete; después vuelve acá a "
                "correr las etapas.")
        return

    if not puede_e0:
        st.warning(
            f"Este caso tiene su `red.pkl` pero no el paquete `red_nueva.pkl`, "
            f"así que **E0 no se puede rehacer**: su red tendría que salir de "
            f"`app.pkl`, que es el estudio congelado de 15 barras y no esta red. "
            f"El resto de las etapas sí corre, porque leen el `red.pkl` del caso.")
        st.caption(
            "`red.pkl` es la SALIDA de E0 y el paquete es su ENTRADA, pero el "
            "primero contiene siete de las ocho claves que E0 lee, así que el "
            "paquete se puede reconstruir desde él sin volver a pasar la red por "
            "el asistente.")
        if st.button("↩️ Reconstruir `red_nueva.pkl` desde el `red.pkl` del caso",
                     key=f"{CLAVE}_btn_recon"):
            try:
                ruta, faltantes = ER.paquete_desde_red_pkl(caso)
            except Exception as e:                                  # noqa: BLE001
                st.error(f"No se pudo reconstruir: {type(e).__name__}: {e}")
            else:
                st.success(f"Escrito `{ruta.name}`.")
                if faltantes:
                    st.warning("Quedó sin rellenar, porque no hay de dónde "
                               "sacarlo: " + "; ".join(faltantes)
                               + ". Para eso sí hace falta el asistente.")
                st.rerun()

    # Seccionadores del caso: definen la variante de aislación de E5 y, con
    # ella, en qué subcarpeta de baselines/ caen los casos base. Sin pasarlos,
    # E5 calcularía el supuesto histórico (aislación perfecta) y Optimización
    # buscaría los baselines de la variante realista, que no existirían.
    r0 = _resumen_red_cacheado(str(caso), (caso / "red.pkl").stat().st_mtime) \
        if est["armado"] else None
    sec_caso = list((r0 or {}).get("seccionadores")
                    or (est["paquete"] or {}).get("seccionadores_existentes") or [])

    st.markdown("#### Supuestos de E4 y E5")
    c4, c5 = st.columns(2)
    lim_dg = c4.number_input("E4 · Tope DG [fracción de la demanda]", 0.0, 5.0,
                             0.3, step=0.01, key=f"{CLAVE}_e4_lim_dg")
    modo_dg = c4.selectbox("E4 · Tope aplicado por", ["alimentador", "bus"],
                           key=f"{CLAVE}_e4_modo_dg")
    from limites_dg import etiqueta_variante
    etiqueta_e4 = etiqueta_variante({"lim_dg": lim_dg, "lim_dg_modo": modo_dg})
    ruta_e4 = P.ruta_sin_fallas(etiqueta_e4, caso)
    c4.caption(f"Se guardará en `{ruta_e4.name}`. El E4 histórico se conserva.")
    sec_txt = c5.text_input("E5 · Líneas con seccionador", _lista(sec_caso),
                            key=f"{CLAVE}_e5_seccionadores",
                            help="Índices de línea, por ejemplo 0,2,9,10. Vacío: aislación perfecta.")
    try:
        sec_caso = indices(sec_txt) if sec_txt.strip() else []
        if r0 is not None and r0.get("net") is not None:
            ajenas = set(sec_caso) - set(r0["net"].line.index)
            if ajenas:
                raise ValueError(f"Estas líneas no existen en la red del caso: {sorted(ajenas)}")
    except ValueError as exc:
        st.error(str(exc))
        return
    escenarios, var = _escenarios_e5_ui(caso, est, sec_caso)
    c5.caption(f"Casos base en `{_dir_baselines(caso, var).relative_to(caso)}`.")

    n_horas = int((est["paquete"] or {}).get("n_horas")
                  or (est["faultlin"]["forma"] or [0])[0] or 8760)
    n_buses = int((r0 or {}).get("n_buses") or 0)
    seg_e4, base_e4 = _estimacion(est, "E4", n_horas, n_buses, factor=4)
    seg_e5, base_e5 = _estimacion(est, "E5", n_horas, n_buses)

    forzar = st.checkbox("--forzar", key=f"{CLAVE}_forzar",
                         help="Rehacer las etapas que ya tienen artefacto. "
                              "Ojo: rehacer E2 invalida los baselines de E5 y "
                              "el barrido calculado con la matriz anterior.")

    base = {"fuente": fuente, "forzar": forzar}
    # E0 es la única que necesita el paquete: sin él no se ofrece, en vez de
    # ofrecerse con la red del estudio congelado.
    pasos = [O.paso("E0 · red y topología", _cmd_pipeline(caso, ["E0"], **base),
                    [caso / "red.pkl"], 0, "pipeline", milp=False)] if puede_e0 else []
    pasos += [
        O.paso("E1 · perfiles", _cmd_pipeline(caso, ["E1"], **base),
               [caso / "perfiles.npz"], 0, "pipeline", milp=False),
        O.paso("E2 · fallas mitigadas", _cmd_pipeline(caso, ["E2"], **base),
               [caso / "fallas.npz"], 0, "pipeline", milp=False),
        O.paso("E3 · costos", _cmd_pipeline(caso, ["E3"], **base),
               [caso / "costos.pkl"], 0, "pipeline", milp=False),
        # E4 y E5 sí son MILP: `milp=True` hace que el panel muestre el tiempo
        # estimado y que no se reofrezcan cuando su artefacto ya existe.
        # E4 resuelve CUATRO problemas del horizonte completo (base, DG solar,
        # DG eólica y las dos juntas), de ahí el factor 4.
        O.paso("E4 · caso sin fallas (4 MILP)",
               _cmd_pipeline(caso, ["E4"], lim_dg=lim_dg, lim_dg_modo=modo_dg,
                             etiqueta=etiqueta_e4, horas=n_horas, **base),
               [ruta_e4], seg_e4, "pipeline"),
        O.paso(f"E5 · baselines de {len(escenarios)} escenario(s), variante "
               f"`{var}` (MILP)",
               _cmd_pipeline(caso, ["E5"], escenarios=_lista(escenarios),
                             seccionadores=sec_caso, horas=n_horas, **base),
               [_dir_baselines(caso, var) / f"esc_{e:03d}.npz" for e in escenarios],
               seg_e5, "pipeline"),
    ]
    elegidas = st.multiselect("Etapas a preparar", [p["nombre"].split(" · ")[0] for p in pasos],
                               default=[p["nombre"].split(" · ")[0] for p in pasos],
                               key=f"{CLAVE}_etapas_elegidas")
    pasos = [p for p in pasos if p["nombre"].split(" · ")[0] in elegidas]
    if forzar:
        for paso in pasos:
            paso["pendientes"] = paso["total"]
    O.panel_ejecucion(caso / "trabajos_v8" / "pipeline", pasos,
                      {"etapas": [p["nombre"] for p in pasos], "caso": str(caso),
                       "e4": etiqueta_e4, "lim_dg": lim_dg, "lim_dg_modo": modo_dg,
                       "seccionadores": sec_caso, "horas": n_horas},
                      key=f"{CLAVE}_etapas")
    if fuente == "app_pkl":
        origen = "`app.pkl`, el estudio congelado de la tesis (solo lectura)"
    elif hay_paquete:
        origen = f"`{caso.name}/red_nueva.pkl`, el paquete que escribe esta pestaña"
    else:
        origen = (f"`{caso.name}/red.pkl` más los archivos de perfiles "
                  f"(E1 no necesita el paquete; E0 sí, y por eso no se ofrece)")
    st.caption(f"Se ejecuta `pipeline.py --artefactos {caso.name}`, o sea que "
               f"escribe **solo** en esa carpeta. La red sale de {origen}.")
    st.caption(
        f"Tiempos: **E4 ≈ {seg_e4/60:.0f} min** ({base_e4}) · "
        f"**E5 ≈ {seg_e5/60:.0f} min por escenario** ({base_e5}), o sea "
        f"≈ {seg_e5*len(escenarios)/3600:.1f} h por los {len(escenarios)} "
        f"escenarios elegidos, sobre un horizonte de {n_horas} h.")
    if base_e4.startswith("estimación gruesa"):
        st.caption(
            "⚠️ Esas cifras son una cota gruesa hasta que el caso tenga tiempos "
            "propios: salen de escalar lo que tarda la red de 15 barras. El "
            "tamaño de la red y la máquina cambian mucho el resultado — el mismo "
            "caso base de 720 h tardó 18,7 s en un equipo y 60,2 s en otro. Al "
            "correr una etapa queda medida en `meta.json` y esta estimación pasa "
            "a usar ese número.")


def _lista(escenarios):
    return ",".join(str(int(e)) for e in escenarios)


def _escenarios_e5_ui(caso, est, sec_caso):
    """Qué escenarios calcular en E5. Devuelve la lista de índices (desde 0).

    Lo que manda acá no es un rango escrito a mano —el viejo "0-9" no era una
    muestra de nada— sino **si la población está completa o no**, que es el
    mismo criterio con el que Optimización decide qué puede ofrecer
    (`completa = len(disp) >= n_pob`, `optimizacion.py:580`):

      * mientras falten casos base, lo único que interesa es completarlos, y
        E5 se saltea solo los que ya existen (`pipeline.py`: "ya existe, se
        omite"), así que pedir la población entera calcula exactamente los que
        faltan;
      * una vez completa, queda habilitada la selección por percentil de
        severidad en Optimización, que es la que elige la muestra del barrido.

    Las otras dos opciones existen para corridas parciales: la muestra
    estratificada que dejó `seleccionar_escenarios.py` en este caso, y una
    lista personalizada con el mismo parser (`core.indices`) del resto de la
    app.
    """
    r = _resumen_red_cacheado(str(caso), (caso / "red.pkl").stat().st_mtime) \
        if est["armado"] else None
    n_pob = int((r or {}).get("numesc") or (est["paquete"] or {}).get("numesc") or 0)
    # Los baselines viven en `baselines/` o en `baselines/<variante>/` según el
    # supuesto de aislación: dos supuestos distintos no son comparables. Si el
    # caso declara seccionadores, los suyos son los de su variante y no los de
    # la carpeta raíz.
    var = variante_aislacion(bool(sec_caso), sec_caso)
    hechos = set(est["escenarios_baseline"] if var == "base"
                 else _baselines_variante(caso, var))
    faltan_pob = [e for e in range(n_pob) if e not in hechos]
    muestra, meta_muestra = _muestra_estratificada(caso)

    if n_pob and faltan_pob:
        st.warning(
            f"La población del caso son **{n_pob} escenario(s)** y hay "
            f"**{len(faltan_pob)} sin caso base** (variante `{var}`). Hasta "
            f"completarlos, «Optimización y análisis» no puede elegir la muestra "
            f"por percentil de severidad, porque no tiene con qué ordenar.")
    elif n_pob:
        st.success(f"Los {n_pob} casos base de la variante `{var}` están "
                   f"calculados: la selección por percentil de severidad ya se "
                   f"puede usar en «Optimización y análisis».")

    modos = ([f"Población completa ({n_pob})"] if n_pob else []) \
        + ([f"Muestra estratificada ({len(muestra)})"] if muestra else []) \
        + ["Personalizada"]
    modo = st.radio("Escenarios para E5", modos, horizontal=True,
                    key=f"{CLAVE}_modo_esc")

    if modo.startswith("Población completa"):
        escenarios = list(range(n_pob))
    elif modo.startswith("Muestra estratificada"):
        escenarios = list(muestra)
        st.caption(
            f"Muestra de `seleccionar_escenarios.py` "
            f"(criterio {meta_muestra.get('criterio')}, "
            f"{meta_muestra.get('n_estratos')} estratos de "
            f"{meta_muestra.get('n_poblacion')}): {_lista(escenarios)}")
    else:
        try:
            escenarios = sorted(set(indices(st.text_input(
                "Índices desde 0 (ej: 1,9,15 o 0-9)", _lista(sorted(hechos)) or "0",
                key=f"{CLAVE}_esc_txt"))))
        except ValueError as exc:
            st.error(str(exc))
            escenarios = []

    if escenarios:
        faltan = [e for e in escenarios if e not in hechos]
        st.caption(
            f"{len(escenarios) - len(faltan)} ya calculados, "
            f"**{len(faltan)} pendiente(s)**"
            + (f" ≈ {len(faltan) * O.SEG_POR_E5 / 60:.0f} min. E5 no rehace los "
               f"que ya están, salvo con `--forzar`." if faltan else "."))
    return escenarios, var


def _baselines_variante(caso, var):
    """Escenarios con caso base en la variante de aislación `var` del caso."""
    d = caso / "baselines" / var
    if not d.exists():
        return []
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("esc_*.npz"))


def _muestra_estratificada(caso):
    """La muestra que dejó `seleccionar_escenarios.py` en ESTE caso.

    Se lee por ruta y no con `pipeline.escenarios_muestra()`, que resuelve
    contra la carpeta activa del proceso: acá el caso puede ser otro.
    """
    ruta = caso / "escenarios_estratificados.json"
    if not ruta.exists():
        return [], {}
    try:
        d = json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:                                               # noqa: BLE001
        return [], {}
    escs = sorted(int(e) for e in (d.get("escenarios") or []))
    meta = {k: d.get(k) for k in ("criterio", "n_estratos", "n_poblacion")}
    return escs, meta


# ════════════════════════════════════════════════════════════
# Fallas: la cadena real o el generador sintético
# ════════════════════════════════════════════════════════════

def _panel_fallas(caso, est):
    st.markdown("### Matriz de fallas")
    _aviso_fallas(est, caso)

    fl, r = est["faultlin"], None
    if est["armado"]:
        r = _resumen_red_cacheado(str(caso), (caso / "red.pkl").stat().st_mtime)

    if fl["existe"] and fl["forma"]:
        h, esc, lin = fl["forma"]
        st.write(f"`{fl['ruta'].name}`: **{h} h × {esc} escenarios × {lin} líneas**, "
                 f"{fl['fallas']} fallas ({fl['mb']:.1f} MB)")
        if r and lin != r["n_lineas"]:
            st.error(
                f"La matriz tiene {lin} líneas y la red de este caso tiene "
                f"{r['n_lineas']}. **No son la misma red**: hay que rehacer la "
                f"matriz, y con ella E2, E4 y E5.")
        meta = est["faultlin_meta"] or {}
        if meta:
            st.caption(f"curva={meta.get('curva')} · "
                       f"clima={meta.get('escenario_clima')} · "
                       f"semilla={meta.get('semilla')} · "
                       f"enlaces inmunes={meta.get('enlaces_inmunes')}")
    else:
        st.info("Este caso no tiene matriz de fallas. Hay dos caminos, y **no son "
                "equivalentes**.")

    if not est["armado"]:
        st.caption("Primero hay que tener la red del caso (etapa E0).")
        return

    real, sint = st.tabs(["🌪️ Cadena climática (validada)",
                          "🎲 Fallas sintéticas (solo para probar)"])
    with real:
        _fallas_reales_ui(caso, est, r)
    with sint:
        _fallas_sinteticas_ui(caso, est, r)


def _fallas_reales_ui(caso, est, r):
    st.markdown(
        "Viento real del `.nc` de eventos extremos → curva de fragilidad → "
        "Monte Carlo. Es el camino del estudio y el único que sirve para la "
        "tesis. Cada línea toma el viento de la **zona** de su bus de origen, "
        "así que el `.nc` tiene que tener al menos tantos puntos como zonas "
        "tiene la red.")
    lineas, ok = _insumos_clima()
    st.caption("Insumos climáticos (compartidos por todos los casos): "
               + " ".join(lineas))
    if not ok:
        st.error("Falta un insumo climático, así que este camino no se puede "
                 "correr todavía. Hay que generar el `.nc` con el pipeline "
                 "climático, usando el `SystemzoneV1.xlsx` de las zonas de este "
                 "caso.")
        return
    st.caption("⚠️ El `.nc` corresponde a los puntos geográficos con que se "
               "generó. Si las zonas de este caso son otras, hay que rehacer la "
               "cadena climática antes de sortear las fallas, o cada línea "
               "tomará el viento de un lugar que no le corresponde.")

    c1, c2, c3 = st.columns(3)
    horas = c1.number_input("Horas", 24, 17520,
                            int((est["paquete"] or {}).get("n_horas") or 8760),
                            step=24, key=f"{CLAVE}_fl_horas")
    esc = c2.number_input("Escenarios", 1, 1000,
                          int(r["numesc"] if r and r["numesc"] else 100),
                          key=f"{CLAVE}_fl_esc")
    semilla = c3.number_input("Semilla", 0, 10**6, 42, key=f"{CLAVE}_fl_semilla",
                              help="Sin semilla la corrida no es reproducible. "
                                   "El .npy histórico se generó sin ella.")
    curva = st.selectbox("Curva de fragilidad", ["dunn", "lognormal"],
                         key=f"{CLAVE}_fl_curva",
                         help="'dunn' es la vigente del estudio: empírica, escala "
                              "con el largo del tramo e incluye caída de árboles.")

    cmd_fl = [sys.executable, str(P.APP_DIR / "construir_faultlin.py"),
              "--artefactos", str(caso), "--horas", str(int(horas)),
              "--escenarios", str(int(esc)), "--curva", curva,
              "--semilla", str(int(semilla)),
              "--escenario-clima", "with_EWE", "--enlaces-inmunes", "--forzar"]
    pasos = [
        O.paso("Monte Carlo de fallas", cmd_fl,
               [casos.ruta_faultlin(caso)], 60, "faultlin", milp=False),
        O.paso("E2 · fallas mitigadas", _cmd_pipeline(caso, ["E2"], forzar=True),
               [caso / "fallas.npz"], 30, "pipeline", milp=False),
    ]
    O.panel_ejecucion(caso / "trabajos_v8" / "faultlin", pasos,
                      {"caso": str(caso)}, key=f"{CLAVE}_fl")


def _fallas_sinteticas_ui(caso, est, r):
    st.warning(
        "**Fallas inventadas: no sirven para la tesis.** Tormentas sorteadas sin "
        "viento ni curva de fragilidad. Existen para responder rápido *¿el modelo "
        "resuelve esta red?* sin esperar la cadena climática completa.\n\n"
        "Además, las matrices mitigadas quedan iguales a la base, así que "
        "vegetación, refuerzo y soterramiento darán ahorro cero por construcción.")

    if est["es_validacion"]:
        st.error("Este es el caso de validación de la tesis: sus fallas vienen del "
                 "viento real y la pestaña no permite pisarlas con fallas "
                 "inventadas. Crea otro caso arriba.")
        return

    c1, c2, c3 = st.columns(3)
    horas = c1.number_input("Horas", 24, 17520,
                            int((est["paquete"] or {}).get("n_horas") or 8760),
                            step=24, key=f"{CLAVE}_syn_horas")
    esc = c2.number_input("Escenarios", 1, 1000,
                          int(r["numesc"] if r and r["numesc"] else 1),
                          key=f"{CLAVE}_syn_esc")
    tormentas = c3.number_input("Tormentas al año", 1, 200, 12,
                                key=f"{CLAVE}_syn_torm")
    c4, c5, c6 = st.columns(3)
    p_linea = c4.slider("P(falla) por línea de la zona golpeada", 0.01, 0.5, 0.08,
                        0.01, key=f"{CLAVE}_syn_plin")
    p_multi = c5.slider("P(evento multizona)", 0.0, 1.0, 0.30, 0.05,
                        key=f"{CLAVE}_syn_pmulti")
    semilla = c6.number_input("Semilla", 0, 10**6, 7, key=f"{CLAVE}_syn_semilla")

    cmd = [sys.executable, str(P.APP_DIR / "fallas_sinteticas.py"),
           "--artefactos", str(caso), "--horas", str(int(horas)),
           "--escenarios", str(int(esc)), "--tormentas", str(int(tormentas)),
           "--p-linea", str(p_linea), "--p-multizona", str(p_multi),
           "--semilla", str(int(semilla)), "--forzar"]
    O.panel_ejecucion(
        caso / "trabajos_v8" / "fallas_sinteticas",
        [O.paso("Fallas sintéticas", cmd, [caso / "fallas.npz"], 0,
                "fallas_sint", milp=False)],
        {"caso": str(caso)}, key=f"{CLAVE}_syn")
    st.caption("Escribe `fallas.npz` y deja `fuente_viento = sintetico` en "
               "`fallas_meta.json`, que es lo que hace que el aviso de arriba "
               "aparezca en toda la app mientras esas fallas estén en uso.")


# ════════════════════════════════════════════════════════════
# render
# ════════════════════════════════════════════════════════════

def render():
    st.title("🗺️ Datos de entrada")
    st.caption(
        "Todo empieza eligiendo el **caso de estudio**: una carpeta de artefactos "
        "con su red, sus perfiles y sus fallas. Esta pestaña prepara los insumos "
        "y lanza las etapas; el cálculo corre en procesos aparte.")

    # El caso PRIMERO. Nada de estado antes: un ✅ solo significa algo referido
    # a un caso concreto.
    caso = _selector_caso()
    if caso is None:
        st.info("Elige un caso existente o crea uno nuevo para continuar.")
        return
    _invalidar_si_cambio_caso(caso)

    est = casos.estado_caso(caso)
    _banner_caso(caso, est)

    # Sin tabla de ✅/⬜ acá: qué etapa falta se ve en el panel de etapas, al
    # lado del botón que la calcula y con su tiempo estimado. Una lista de
    # chequeos separada del sitio donde se actúa solo repite información.
    if est["armado"]:
        _resumen_caso(caso, est)

    st.divider()
    _panel_etapas(caso, est)

    st.divider()
    _panel_fallas(caso, est)

    st.divider()
    armado = est["armado"] or (caso / "red_nueva.pkl").exists()
    with st.expander("🧭 Asistente: cargar o reemplazar la red de este caso",
                     expanded=not armado):
        _asistente(caso, est)

    with st.expander("Cómo encajan las dos cadenas", expanded=False):
        st.markdown("""
**Cadena 1 — clima (`Proyecto oficial/Scripts/`)**

`SystemzoneV1.xlsx` (lat/lon por zona) → Open-Meteo (histórico horario) +
ARClim (futuro diario) → `BaseFhorPELP4v20z2anos.nc` (base horaria por zona)
→ `BaseHoraEWEPELP20z.nc` (viento, humedad, FFDI, incendio hora a hora).

**Puente — matriz de fallas (`construir_faultlin.py`)**

Viento del `.nc` a 12 m de altura → curva de fragilidad lognormal
(θ = 30 m/s, β = 0.25) → Monte Carlo de 100 escenarios → `FaultlinMC.npy`.
Acá se cruzan las dos cadenas: cada **línea** toma el viento de la **zona**
de su bus de origen.

**Cadena 2 — modelo (`pipeline.py`)**

`red_nueva.pkl` (o `app.pkl`) + `FaultlinMC.npy` → E0 red · E1 perfiles ·
E2 fallas mitigadas · E3 costos · E4 caso sin fallas · E5 baselines →
la carpeta del caso → la app y el barrido.
        """)


def _asistente(caso, est):
    """Los cinco pasos para dejar armado el paquete de red del caso.

    Mantiene los cortes tempranos de siempre: sin red no hay paso 2, con
    errores de validación no hay paso 3, y sin puntos geográficos no hay
    exportación. Cada corte explica qué falta.
    """
    st.markdown("#### 1 · Cargar la red")
    _cargar_red_ui()

    net = st.session_state.get(f"{CLAVE}_net")
    if net is None:
        st.info("Carga una red para continuar con los pasos 2 a 5.")
        return

    st.divider()
    st.markdown(f"#### 2 · Validar — *{st.session_state.get(f'{CLAVE}_origen','')}*")
    if not _validacion_ui(net):
        st.error("Hay errores que impiden usar esta red en el modelo. "
                 "Corrígelos antes de seguir.")
        return

    st.divider()
    st.markdown("#### 3 · Zonas climáticas por bus")
    bus_zone, zonas_usadas = _zonas_ui(net)

    st.divider()
    st.markdown("#### 4 · Punto geográfico de cada zona")
    puntos, completo = _puntos_ui(net, bus_zone, zonas_usadas)

    st.divider()
    st.markdown("#### 5 · Exportar")
    if not completo:
        st.info("Completa los puntos geográficos para poder exportar.")
        return
    _exportar_ui(caso, net, bus_zone, puntos)
