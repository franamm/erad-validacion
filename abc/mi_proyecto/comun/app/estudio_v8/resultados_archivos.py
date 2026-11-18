"""Consulta de resultados por ruta, independiente del modelo activo y del solver."""
from pathlib import Path
import gzip
import pickle

import numpy as np
import pandas as pd
import streamlit as st


def archivos_disponibles(carpeta):
    carpeta = Path(carpeta)
    patrones = ("*.csv", "caso_base_*.pkl", "caso_base_*.npz", "prueba_milp.pkl",
                "barrido*/*/*.csv", "barrido*/*/individual/*/esc_*.npz",
                "barrido*/*/combinadas/*/esc_*.npz", "baselines/**/esc_*.npz",
                "barrido*/*/individual/*/esc_*.pkl.gz",
                "barrido*/*/combinadas/*/esc_*.pkl.gz")
    return sorted({p for patron in patrones for p in carpeta.glob(patron) if p.is_file()})


def _series(obj, nombre="Resultado"):
    """Recorre los formatos históricos: escenario -> horas, historial y base/estrategia."""
    if isinstance(obj, dict):
        if "results" in obj:
            yield from _series(obj["results"], nombre)
        elif "historial" in obj:
            yield from _series(obj["historial"], nombre)
        else:
            for clave, valor in obj.items():
                if isinstance(valor, (dict, list, tuple)):
                    yield from _series(valor, f"{nombre} / {clave}")
    elif isinstance(obj, (list, tuple)) and obj:
        if any(isinstance(h, dict) and "P_LC" in h for h in obj):
            yield nombre, obj
        else:
            for i, valor in enumerate(obj):
                yield from _series(valor, f"{nombre} / {i}")


def _total(valor):
    if isinstance(valor, dict):
        valor = list(valor.values())
    return float(np.asarray(valor, dtype=float).sum())


def resumir_pickle(obj):
    salida = {}
    for nombre, horas in _series(obj):
        filas = []
        for i, hora in enumerate(horas, 1):
            fila = {"Hora": i}
            if isinstance(hora, dict):
                for campo in ("P_LC", "P_SUB", "P_D", "Pb"):
                    if campo in hora:
                        fila[campo] = _total(hora[campo])
            filas.append(fila)
        salida[nombre] = pd.DataFrame(filas).set_index("Hora")
    return salida


@st.cache_data(show_spinner="Leyendo resultados guardados…", max_entries=4)
def leer_resultado(ruta, mtime_ns, tamano):
    """Cachea solo las tablas pequeñas; ruta y sello evitan cruces entre casos."""
    ruta = Path(ruta)
    if ruta.suffix == ".csv":
        from barrido_csv import leer_resumen
        return {"Tabla": leer_resumen(ruta)}
    if ruta.suffix == ".npz":
        with np.load(ruta, allow_pickle=False) as z:
            columnas = {}
            for campo in ("P_LC", "P_SUB", "P_D", "Pb"):
                if campo in z:
                    a = np.asarray(z[campo], dtype=float)
                    if a.ndim in (1, 2):
                        columnas[campo] = a if a.ndim == 1 else a.sum(axis=1)
            if not columnas:
                return {}
            df = pd.DataFrame(columnas)
            df.index = pd.RangeIndex(1, len(df) + 1, name="Hora")
            return {"Serie horaria": df}
    abrir = gzip.open if ruta.suffix == ".gz" else open
    with abrir(ruta, "rb") as fh:
        return resumir_pickle(pickle.load(fh))


def render(carpeta):
    carpeta = Path(carpeta).resolve()
    st.caption(f"Carpeta consultada: `{carpeta}`")
    archivos = archivos_disponibles(carpeta)
    if not archivos:
        st.info("Esta carpeta todavía no contiene resultados guardados legibles.")
        return
    # La ruta del caso forma parte de la clave: dos carpetas nunca comparten selección.
    clave = f"archivos_resultados:{carpeta}"
    preferido = next((p for p in archivos if p.name == "escalera_caso_base.csv"), archivos[0])
    ruta = st.selectbox("Resultado guardado", archivos, index=archivos.index(preferido),
                        format_func=lambda p: p.relative_to(carpeta).as_posix(), key=clave)
    if st.button("Actualizar archivos", key=f"{clave}:actualizar"):
        leer_resultado.clear()
    sello = ruta.stat()
    try:
        tablas = leer_resultado(str(ruta), sello.st_mtime_ns, sello.st_size)
    except Exception as exc:
        st.error(f"No se pudo leer {ruta.name}: {exc}")
        return
    if not tablas:
        st.info("El archivo no contiene series horarias reconocibles.")
        return
    nombre = st.selectbox("Serie", list(tablas), key=f"{clave}:{ruta.name}:serie") if len(tablas) > 1 else next(iter(tablas))
    tabla = tablas[nombre]
    if tabla.index.name == "Hora":
        if "P_LC" in tabla:
            st.metric("ENS del horizonte [kWh]", f"{tabla['P_LC'].sum(min_count=1):,.2f}")
        st.caption(f"{len(tabla)} horas guardadas. Los valores ausentes se conservan como ausentes.")
        campo = st.selectbox("Variable horaria [kW]", list(tabla.columns), key=f"{clave}:variable")
        st.line_chart(tabla[[campo]])
    st.dataframe(tabla, use_container_width=True)
    st.download_button("Descargar tabla CSV", tabla.to_csv(index=tabla.index.name == "Hora").encode("utf-8-sig"),
                       file_name=f"{ruta.stem}.csv", mime="text/csv", key=f"{clave}:csv")
