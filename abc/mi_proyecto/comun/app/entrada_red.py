# ============================================================
# entrada_red.py — Lectura de la red y definición de zonas/puntos
# ============================================================
# Lógica PURA (sin streamlit) detrás de la pestaña "🗺️ Datos de entrada".
# Se separa de la vista por la misma razón que `indicadores.py` no importa
# `data_loader`: así se puede usar desde consola, desde un notebook o desde
# un test, sin levantar Streamlit.
#
# Qué resuelve
# ------------
# Hoy la red del estudio nace de dos celdas del notebook (celdas 2 y 5 de
# "Prueba 12 oficial.ipynb") que hacen tres cosas mezcladas:
#   1. crear la CIGRE MV de pandapower,
#   2. pisarle los parámetros de línea (subterránea -> aérea),
#   3. colgarle coordenadas y zonas leídas de SystemzoneV1.xlsx.
# Y de ahí sale, por un lado, `app.pkl` (la red del modelo) y por otro el
# propio SystemzoneV1.xlsx que alimenta al pipeline climático. Los dos
# tienen que hablar de las MISMAS zonas: si la red tiene 3 zonas y el Excel
# 4 puntos, `construir_faultlin.py` mapea mal el viento y nadie se entera.
#
# Este módulo hace explícito ese contrato: carga la red (varios formatos),
# la valida contra lo que el MILP realmente exige, deja fijar zona por bus
# y un punto geográfico por zona, y exporta LAS DOS salidas coherentes
# entre sí:
#   - SystemzoneV1.xlsx  -> entrada del pipeline climático
#   - artefactos/red_nueva.pkl -> entrada de pipeline.py --fuente red_nueva
# ============================================================

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import casos

APP_DIR = Path(__file__).resolve().parent
# Carpeta del caso activo (ver casos.py). Son defaults: la vista pasa la ruta
# explícita del caso que la usuaria eligió, porque puede estar preparando un
# caso distinto del que cargó la app.
ART_DIR = casos.art_dir_activo()
RUTA_PAQUETE = ART_DIR / "red_nueva.pkl"

# Parámetros de línea aérea de MT con los que el notebook (celda 5) pisa los
# valores subterráneos de la CIGRE. Se dejan acá con nombre y no sueltos en
# un bloque de código: son un supuesto del estudio, no un detalle.
LINEA_AEREA_MT = {
    "std_type": "OHL_CIGRE_MV",
    "r_ohm_per_km": 0.510,
    "x_ohm_per_km": 0.366,
    "c_nf_per_km": 10.09679,
    "max_i_ka": 0.145,
    "type": "ol",
}

# Columnas que el modelo lee de verdad (grep en optimization_6_oficial.py,
# optimization_base.py, grafo.py e indicadores.py). Si falta una, el MILP
# no falla al armarse: falla a mitad de la primera hora.
REQUERIDO = {
    "bus": ["vn_kv"],
    "line": ["from_bus", "to_bus", "length_km", "x_ohm_per_km", "max_i_ka"],
    "load": ["bus", "p_mw"],
    "trafo": ["sn_mva"],
}

FORMATOS = {
    "cigre": "CIGRE MV integrada (la del estudio)",
    "pandapower": "pandapower (.json / .p / .xlsx)",
    "opendss": "OpenDSS (.dss)",
    "csv": "CSV (buses + líneas + cargas)",
}


# ════════════════════════════════════════════════════════════
# 1. Carga de la red
# ════════════════════════════════════════════════════════════

def red_cigre(aerea=True):
    """Red CIGRE MV de pandapower, con los parámetros del estudio.

    `aerea=True` reproduce la celda 5 del notebook: la CIGRE trae cables
    subterráneos y el estudio modela líneas aéreas, que son las expuestas
    al viento. Sin este cambio la curva de fragilidad no tendría sentido
    físico sobre esta red.
    """
    from pandapower.networks import create_cigre_network_mv

    net = create_cigre_network_mv()
    if aerea:
        for col, val in LINEA_AEREA_MT.items():
            net.line[col] = val
    return net


def desde_pandapower(ruta):
    """Lee .json, .p/.pickle o .xlsx de pandapower."""
    import pandapower as pp

    ruta = Path(ruta)
    suf = ruta.suffix.lower()
    if suf == ".json":
        return pp.from_json(str(ruta))
    if suf in (".p", ".pickle", ".pkl"):
        return _red_desde_pickle(ruta)
    if suf in (".xlsx", ".xls"):
        return pp.from_excel(str(ruta))
    raise ValueError(f"Extensión no reconocida para pandapower: '{suf}'")


def _red_desde_pickle(ruta):
    """Red desde un .pkl, venga en el formato de pandapower o en el nuestro.

    Un .pkl puede ser tres cosas distintas y el usuario no tiene por qué
    distinguirlas:

      1. el formato de `pp.to_pickle`;
      2. una `pandapowerNet` guardada directamente con `pickle.dump`;
      3. un paquete del pipeline (`artefactos/red.pkl`), que es un dict
         con la red en la clave "net" más BUSES, P_D_base, etc.

    `pp.from_pickle` solo entiende el primero, y con los otros dos falla
    con "'pandapowerNet' instance has no attribute 'version'", que no le
    dice a nadie qué hacer. Por eso se mira el contenido antes.
    """
    import pickle

    import pandapower as pp

    try:
        with open(ruta, "rb") as f:
            objeto = pickle.load(f)
    except Exception:
        # No es un pickle legible por sí solo: que lo intente pandapower,
        # que sabe de sus propios formatos comprimidos.
        return pp.from_pickle(str(ruta))

    if isinstance(objeto, pp.pandapowerNet):
        return objeto
    if isinstance(objeto, dict):
        if isinstance(objeto.get("net"), pp.pandapowerNet):
            return objeto["net"]
        # Formato propio de pandapower (dict de tablas serializadas).
        try:
            return pp.from_pickle(str(ruta))
        except Exception:
            claves = list(objeto)[:8]
            raise ValueError(
                f"'{ruta.name}' es un diccionario sin red adentro. Claves: "
                f"{claves}. Si es un paquete del pipeline, la red va en la "
                f"clave 'net'.") from None
    raise ValueError(
        f"'{ruta.name}' contiene un {type(objeto).__name__}, no una red de "
        f"pandapower.")


def desde_opendss(ruta_master):
    """Convierte un caso OpenDSS a una red pandapower mínima.

    NO es un conversor general: extrae buses, líneas, cargas y coordenadas,
    que es exactamente lo que consume el MILP (flujo DC linealizado sobre
    x_ohm_per_km y max_i_ka). Todo lo que OpenDSS modela y este estudio no
    usa —regulación de tensión, capacitores, curvas de carga horarias,
    devanados— se ignora a propósito: arrastrarlo daría una falsa sensación
    de fidelidad sobre un modelo que igual lo descarta.

    Devuelve (net, avisos).
    """
    import opendssdirect as dss
    import pandapower as pp

    ruta_master = Path(ruta_master)
    avisos = []

    dss.Text.Command("Clear")
    dss.Text.Command(f'Redirect "{ruta_master.as_posix()}"')

    net = pp.create_empty_network(name=ruta_master.stem)

    # — Buses —
    nombres = list(dss.Circuit.AllBusNames())
    idx_por_nombre = {}
    for nombre in nombres:
        dss.Circuit.SetActiveBus(nombre)
        try:
            vn = float(dss.Bus.kVBase()) * (3 ** 0.5)   # kVBase es fase-neutro
        except Exception:
            vn = 20.0
        if not np.isfinite(vn) or vn <= 0:
            vn = 20.0
        b = pp.create_bus(net, vn_kv=vn, name=nombre)
        idx_por_nombre[nombre] = b
        # Coordenadas: OpenDSS las guarda en el sistema del caso (suelen ser
        # metros o pies, NO lat/lon). Se conservan como vienen y la vista
        # avisa que hay que georreferenciarlas.
        try:
            if dss.Bus.Coorddefined():
                net.bus.at[b, "x_dss"] = float(dss.Bus.X())
                net.bus.at[b, "y_dss"] = float(dss.Bus.Y())
        except Exception:
            pass

    # — Líneas —
    dss.Lines.First()
    while True:
        nombre = dss.Lines.Name()
        if not nombre:
            break
        bus1 = dss.Lines.Bus1().split(".")[0]
        bus2 = dss.Lines.Bus2().split(".")[0]
        if bus1 in idx_por_nombre and bus2 in idx_por_nombre:
            # OpenDSS entrega el largo Y las impedancias en la unidad de la
            # línea (ft, m, mi...). El mismo factor que convierte el largo a
            # km divide a R1/X1/C1 para pasarlos a "por km": olvidarlo deja
            # líneas de 2000 km con 2e-5 Ω/km, que es exactamente lo que se
            # ve si uno confía en los valores crudos.
            km_por_unidad = _factor_a_km(dss.Lines.Units())
            largo_km = float(dss.Lines.Length()) * km_por_unidad
            if largo_km <= 0:
                largo_km = 0.01
                avisos.append(f"Línea '{nombre}' con largo 0: se asumió 0.01 km.")
            pp.create_line_from_parameters(
                net,
                from_bus=idx_por_nombre[bus1],
                to_bus=idx_por_nombre[bus2],
                length_km=largo_km,
                r_ohm_per_km=float(dss.Lines.R1()) / km_por_unidad or LINEA_AEREA_MT["r_ohm_per_km"],
                x_ohm_per_km=float(dss.Lines.X1()) / km_por_unidad or LINEA_AEREA_MT["x_ohm_per_km"],
                c_nf_per_km=float(dss.Lines.C1()) / km_por_unidad or LINEA_AEREA_MT["c_nf_per_km"],
                max_i_ka=(float(dss.Lines.NormAmps()) or 145.0) / 1000.0,
                name=nombre,
            )
        if dss.Lines.Next() == 0:
            break

    # — Cargas —
    dss.Loads.First()
    while True:
        nombre = dss.Loads.Name()
        if not nombre:
            break
        dss.Circuit.SetActiveElement(f"Load.{nombre}")
        try:
            bus_carga = dss.CktElement.BusNames()[0].split(".")[0]
        except Exception:
            bus_carga = None
        if bus_carga in idx_por_nombre:
            pp.create_load(net, bus=idx_por_nombre[bus_carga],
                           p_mw=float(dss.Loads.kW()) / 1000.0,
                           q_mvar=float(dss.Loads.kvar()) / 1000.0,
                           name=nombre)
        if dss.Loads.Next() == 0:
            break

    if net.load.empty:
        avisos.append("El caso DSS no trae cargas: P_D_base quedaría en cero.")
    if net.trafo.empty:
        avisos.append(
            "El caso DSS no aporta transformador de cabecera. El MILP usa "
            "net.trafo.sn_mva[0] como capacidad del alimentador: hay que "
            "declararla antes de exportar.")

    return net, avisos


def _factor_a_km(unidad_dss):
    """Unidades de largo de OpenDSS -> km por unidad.

    `Lines.Units()` devuelve un enum `LineUnits`, cuyo str() es
    'LineUnits.ft'. Se usa `.name` cuando existe: comparar el str completo
    dejaba TODO en el default de 1.0 km, es decir 2000 ft leídos como
    2000 km.
    """
    nombre = getattr(unidad_dss, "name", None) or str(unidad_dss).split(".")[-1]
    return {"km": 1.0, "m": 1e-3, "mi": 1.609344, "kft": 0.3048,
            "ft": 3.048e-4, "in": 2.54e-5, "cm": 1e-5}.get(nombre.lower(), 1.0)


PLANTILLA_CSV = {
    "buses": "bus,name,vn_kv,zona,lat,lon\n0,Bus 0,20.0,1,-33.45,-70.66\n",
    "lineas": "line,name,from_bus,to_bus,length_km,r_ohm_per_km,x_ohm_per_km,max_i_ka\n"
              "0,Line 0-1,0,1,2.82,0.510,0.366,0.145\n",
    "cargas": "bus,p_mw,q_mvar,name\n1,15.3,3.0,Load R1\n",
}


def desde_csv(df_buses, df_lineas, df_cargas, sn_mva_trafo=25.0):
    """Arma una red pandapower desde tres tablas planas.

    Es el camino de menor fricción para una red que no está en ningún
    formato de simulador: basta un Excel con barras, tramos y consumos.
    Las columnas esperadas están en PLANTILLA_CSV.
    """
    import pandapower as pp

    net = pp.create_empty_network(name="red_csv")

    orden = {}
    for _, fila in df_buses.iterrows():
        b = pp.create_bus(
            net,
            vn_kv=float(fila.get("vn_kv", 20.0)),
            name=str(fila.get("name", f"Bus {fila.get('bus')}")),
        )
        orden[int(fila["bus"])] = b
        for col_src, col_dst in (("zona", "zona"), ("lat", "lat"), ("lon", "lon")):
            if col_src in df_buses.columns and pd.notna(fila.get(col_src)):
                net.bus.at[b, col_dst] = fila[col_src]

    for _, fila in df_lineas.iterrows():
        pp.create_line_from_parameters(
            net,
            from_bus=orden[int(fila["from_bus"])],
            to_bus=orden[int(fila["to_bus"])],
            length_km=float(fila["length_km"]),
            r_ohm_per_km=float(fila.get("r_ohm_per_km", LINEA_AEREA_MT["r_ohm_per_km"])),
            x_ohm_per_km=float(fila.get("x_ohm_per_km", LINEA_AEREA_MT["x_ohm_per_km"])),
            c_nf_per_km=float(fila.get("c_nf_per_km", LINEA_AEREA_MT["c_nf_per_km"])),
            max_i_ka=float(fila.get("max_i_ka", LINEA_AEREA_MT["max_i_ka"])),
            name=str(fila.get("name", "")),
        )

    for _, fila in df_cargas.iterrows():
        pp.create_load(net, bus=orden[int(fila["bus"])],
                       p_mw=float(fila["p_mw"]),
                       q_mvar=float(fila.get("q_mvar", 0.0)),
                       name=str(fila.get("name", "")))

    asegurar_trafo(net, sn_mva_trafo)
    return net


# ════════════════════════════════════════════════════════════
# 1b. Adaptación a lo que el MILP tiene escrito a fuego
# ════════════════════════════════════════════════════════════
# El modelo NO es agnóstico a la numeración de la red. Tres supuestos
# viven en el código de optimization_6_oficial / optimization_base y no se
# pueden configurar:
#
#   * las subestaciones inyectan en los buses 1 y 12
#     (`sub_p = m.P_SUB[t,0] if b == 1 else ... if b == 12`);
#   * el bus 0 es la referencia angular (`model.theta[t, 0].fix(0)`);
#   * los índices se usan como posición y como etiqueta indistintamente
#     (`net.line.x_ohm_per_km.iloc[i]` junto a `net.line.from_bus[l]`, y
#     `net.trafo.sn_mva[0]`).
#
# Una red bajada de cualquier parte no cumple nada de eso. Antes, la
# pestaña solo podía reportar el problema y dejar a la usuaria arreglarlo
# por fuera; `adaptar_para_milp` lo arregla y devuelve el detalle de lo
# que cambió, porque renumerar una red a sus espaldas sería peor que el
# error original.


def _pp_func(nombre):
    """Función de pandapower esté en el paquete raíz (3.x) o en toolbox (2.x)."""
    import importlib

    import pandapower as pp

    if hasattr(pp, nombre):
        return getattr(pp, nombre)
    for modulo in ("pandapower.toolbox", "pandapower.toolbox.element_selection"):
        try:
            mod = importlib.import_module(modulo)
        except ImportError:
            continue
        if hasattr(mod, nombre):
            return getattr(mod, nombre)
    raise AttributeError(f"'{nombre}' no existe en pandapower {pp.__version__}")


def buses_subestacion(net):
    """Buses donde el MILP espera la inyección de las subestaciones.

    Son los lados de baja de los transformadores de cabecera, o sea las
    cabeceras de alimentador. Si la red no tiene transformador se usan los
    buses de `ext_grid`. Devuelve hasta dos, que son las dos inyecciones
    que el modelo admite (`S = [0, 1]`).
    """
    if len(getattr(net, "trafo", [])):
        return [int(b) for b in net.trafo.lv_bus.tolist()[:2]]
    if len(getattr(net, "ext_grid", [])):
        return [int(b) for b in net.ext_grid.bus.tolist()[:2]]
    return []


def tie_indices_sugeridas(net):
    """Líneas normalmente abiertas = las que llevan un switch abierto.

    Es la definición física de un enlace de respaldo, y la que corresponde
    ofrecer por defecto. El viejo `LINES[-3:]` era una convención de la
    red del estudio, donde las tres últimas líneas resultaban ser las de
    respaldo; en una red cualquiera, "las tres últimas" no significan nada.
    """
    sw = getattr(net, "switch", None)
    if sw is None or not len(sw):
        return []
    abiertas = sw[(sw.et == "l") & (~sw.closed.astype(bool))].element
    return sorted({int(x) for x in abiertas if int(x) in net.line.index})


def adaptar_para_milp(net, buses_sub=None):
    """Renumera la red para que cumpla los supuestos del MILP.

    Devuelve `(net_adaptado, cambios, tie_sugeridas)`. `cambios` es la
    lista de lo que se tocó, en castellano, para mostrarla en la vista: sin
    eso la usuaria no tendría cómo saber que el bus 57 de su archivo pasó a
    ser el bus 1.

    `tie_sugeridas` se calcula ANTES de renumerar y se devuelve ya
    traducida a los índices nuevos, porque después de renumerar las líneas
    los índices del archivo original ya no sirven para nada.
    """
    import copy

    # `copy.deepcopy` y no `net.deepcopy()`: ese método no existe en
    # pandapower 3.x (AttributeError), así que el botón "Adaptar la red"
    # reventaba con cualquier red que llegara hasta acá.
    net = copy.deepcopy(net)
    cambios = []
    ties_viejas = tie_indices_sugeridas(net)

    # ── Buses: contiguos, con las cabeceras en 1 y 12 ────────────────
    if buses_sub is None:
        buses_sub = buses_subestacion(net)
    destino = {}
    hv = [int(b) for b in getattr(net, "trafo", pd.DataFrame()).get(
        "hv_bus", pd.Series(dtype=int)).tolist()[:2]]
    # El bus 0 es solo la referencia angular y en la red del estudio es el
    # lado de alta del transformador, que no tiene líneas ni demanda.
    if hv:
        destino[hv[0]] = 0
        # Solo si es OTRO bus: en la CIGRE los dos trafos cuelgan del mismo
        # bus de alta, y asignarlo dos veces pisaba el 0 con un 2 —intercambiando
        # los buses 0 y 2 de una red que ya cumplía todo—.
        if len(hv) > 1 and hv[1] not in destino:
            destino[hv[1]] = 2
    if len(buses_sub) > 0:
        destino[buses_sub[0]] = 1
    if len(buses_sub) > 1:
        destino[buses_sub[1]] = 12

    ocupados = set(destino.values())
    libres = [i for i in range(len(net.bus)) if i not in ocupados]
    resto = [int(b) for b in net.bus.index if int(b) not in destino]
    mapa_bus = dict(destino)
    mapa_bus.update(dict(zip(resto, libres)))
    if any(k != v for k, v in mapa_bus.items()):
        _pp_func("reindex_buses")(net, mapa_bus)
        net.bus = net.bus.sort_index()
        movidos = {k: v for k, v in mapa_bus.items() if k != v}
        cambios.append(
            f"Buses renumerados 0..{len(net.bus)-1} ({len(movidos)} cambiaron "
            f"de número). Cabeceras de subestación: "
            + ", ".join(f"bus {b} → {destino[b]}" for b in buses_sub if b in destino))

    # ── Líneas: índices contiguos, mismo orden ───────────────────────
    if list(net.line.index) != list(range(len(net.line))):
        mapa_lin = {int(v): i for i, v in enumerate(net.line.index)}
        try:
            _pp_func("reindex_elements")(net, "line", lookup=mapa_lin)
        except Exception:
            # Respaldo: renumerar a mano y arrastrar las referencias de los
            # switches, que son las únicas que apuntan a líneas por índice.
            net.line = net.line.rename(index=mapa_lin).sort_index()
            if len(getattr(net, "switch", [])):
                es_linea = net.switch.et == "l"
                net.switch.loc[es_linea, "element"] = (
                    net.switch.loc[es_linea, "element"].map(mapa_lin))
        cambios.append(f"Líneas renumeradas 0..{len(net.line)-1}.")
        ties = sorted(mapa_lin[l] for l in ties_viejas if l in mapa_lin)
    else:
        ties = list(ties_viejas)

    # ── Tablas de elementos: índice desde 0 ──────────────────────────
    # `net.trafo.sn_mva[0]` es acceso POR ETIQUETA: con el índice original
    # de una red bajada (en MV Oberrhein los trafos son 114 y 142) el
    # modelo revienta con KeyError: 0 al construirse.
    reseteadas = []
    for tabla in ("trafo", "load", "sgen", "ext_grid", "switch"):
        df = getattr(net, tabla, None)
        if df is None or not len(df):
            continue
        if list(df.index) != list(range(len(df))):
            setattr(net, tabla, df.reset_index(drop=True))
            reseteadas.append(tabla)
    if reseteadas:
        cambios.append(f"Índice reiniciado en: {', '.join(reseteadas)} "
                       f"(el modelo lee net.trafo.sn_mva[0] por etiqueta).")

    if ties:
        cambios.append(f"Enlaces de respaldo detectados por switch abierto: {ties}.")
    return net, cambios, ties


def asegurar_trafo(net, sn_mva):
    """Garantiza net.trafo.sn_mva[0]: el MILP la usa como capacidad máxima
    del alimentador (optimization_6_oficial: `net.trafo.sn_mva[0]*1000`).
    Sin fila de trafo, el modelo revienta con IndexError al construirse."""
    import pandapower as pp

    if not net.trafo.empty:
        return net
    if len(net.bus) < 2:
        return net
    hv, lv = int(net.bus.index[0]), int(net.bus.index[1])
    pp.create_transformer_from_parameters(
        net, hv_bus=hv, lv_bus=lv, sn_mva=float(sn_mva),
        vn_hv_kv=float(net.bus.vn_kv.iloc[0]), vn_lv_kv=float(net.bus.vn_kv.iloc[1]),
        vkr_percent=0.5, vk_percent=10.0, pfe_kw=0.0, i0_percent=0.0,
        name="Trafo cabecera (declarado en la app)", in_service=False)
    return net


# ════════════════════════════════════════════════════════════
# 2. Validación contra lo que el modelo exige
# ════════════════════════════════════════════════════════════

def validar(net):
    """Devuelve una lista de (nivel, mensaje) con nivel en
    {"error", "aviso", "ok"}. 'error' = el MILP no va a correr."""
    out = []

    for tabla, columnas in REQUERIDO.items():
        df = getattr(net, tabla, None)
        if df is None or len(df) == 0:
            nivel = "error" if tabla in ("bus", "line") else "aviso"
            out.append((nivel, f"net.{tabla} está vacío."))
            continue
        faltan = [c for c in columnas if c not in df.columns]
        if faltan:
            out.append(("error", f"net.{tabla}: faltan columnas {faltan}."))

    if len(net.bus) and len(net.line):
        # Índices: el código recorre LINES = list(net.line.index) y usa
        # .iloc[i] junto a [l] indistintamente (optimization_6_oficial:
        # length_km.iloc[i] y from_bus[l]). Con índices no contiguos las dos
        # formas dejan de coincidir y el modelo mezcla líneas en silencio.
        if list(net.line.index) != list(range(len(net.line))):
            out.append(("error",
                        "net.line tiene índices no contiguos. El modelo mezcla "
                        ".iloc[i] con [l]: reindexar 0..n-1 antes de exportar."))
        if list(net.bus.index) != list(range(len(net.bus))):
            out.append(("error", "net.bus tiene índices no contiguos: reindexar 0..n-1."))

        huerfanos = set(net.line.from_bus) | set(net.line.to_bus)
        huerfanos -= set(net.bus.index)
        if huerfanos:
            out.append(("error", f"Líneas que apuntan a buses inexistentes: {sorted(huerfanos)}"))

        # Buses de subestación: el modelo inyecta en 1 y 12 y NADA avisa si
        # la red los tiene en otra parte. Sin este chequeo, una red subida
        # resolvería con la subestación colgada de un bus cualquiera —
        # típicamente uno interior, sin relación con la cabecera real — y el
        # resultado sería plausible y falso.
        subs = buses_subestacion(net)
        esperados = [1, 12][:len(subs)]
        if subs and list(subs) != esperados:
            out.append(("error",
                        f"Las cabeceras de subestación de esta red son los buses "
                        f"{subs}, pero el modelo inyecta en {esperados} "
                        f"(optimization_6_oficial: `if b == 1 ... elif b == 12`). "
                        f"Hay que renumerar."))
        if not subs:
            out.append(("aviso",
                        "La red no declara transformador ni ext_grid, así que no "
                        "se puede saber dónde está la cabecera. El modelo va a "
                        "inyectar en los buses 1 y 12, sean lo que sean."))

        # Índice por etiqueta en las tablas de elementos: net.trafo.sn_mva[0].
        for tabla in ("trafo", "load"):
            df = getattr(net, tabla, None)
            if df is not None and len(df) and list(df.index) != list(range(len(df))):
                out.append(("error",
                            f"net.{tabla} tiene índices no contiguos desde 0. El "
                            f"modelo lee `net.trafo.sn_mva[0]` por etiqueta y "
                            f"fallaría con KeyError."))

        if (net.line.length_km <= 0).any():
            out.append(("aviso", "Hay líneas con length_km <= 0: las medidas por km "
                                 "(vegetación, refuerzo, soterramiento) costarían 0."))
        if (net.line.max_i_ka <= 0).any():
            out.append(("aviso", "Hay líneas con max_i_ka <= 0: capacidad de transporte nula."))

    if len(getattr(net, "load", [])) == 0:
        out.append(("error", "Sin cargas no hay demanda: P_D_base quedaría en cero "
                             "y todos los indicadores (ENS, SAIDI, SAIFI) valdrían 0."))
    elif len(getattr(net, "trafo", [])):
        # Demanda contra capacidad. Si la demanda pico supera lo que pueden
        # inyectar las subestaciones, el MILP recorta carga incluso SIN
        # fallas, y la ENS del caso base deja de medir el efecto del clima:
        # mide un dimensionamiento inconsistente. Es el chequeo que delata
        # una red importada con el escenario de carga equivocado.
        dem = sum(p_d_base(net).values())
        cap = float(net.trafo.sn_mva.iloc[0]) * 1000 * max(1, len(buses_subestacion(net)))
        if dem > cap:
            out.append(("error",
                        f"La demanda base ({dem:,.0f} kW) supera la capacidad de "
                        f"las subestaciones ({cap:,.0f} kW). El modelo recortaría "
                        f"carga sin que haya ninguna falla. Revisa el `scaling` de "
                        f"las cargas o la capacidad declarada del transformador."))
        elif dem > 0.9 * cap:
            out.append(("aviso",
                        f"La demanda base ({dem:,.0f} kW) usa más del 90% de la "
                        f"capacidad de las subestaciones ({cap:,.0f} kW): queda "
                        f"muy poco margen para redistribuir carga tras una falla."))

    if not out:
        out.append(("ok", "La red cumple lo que el modelo exige."))
    return out


def resumen(net):
    """Cifras de una línea para la cabecera de la vista.

    `demanda_kw` sale de `p_d_base`, o sea con `scaling` aplicado, para que
    la cifra que muestra la pestaña sea EXACTAMENTE la que va a ver el
    modelo. Antes sumaba `p_mw` crudo y en una red con scaling distinto de
    1 mostraba una demanda que nadie usaba.
    """
    return {
        "buses": len(net.bus),
        "lineas": len(net.line),
        "cargas": len(net.load),
        "demanda_kw": sum(p_d_base(net).values()) if len(net.load) else 0.0,
        "largo_km": float(net.line.length_km.sum()) if len(net.line) else 0.0,
        "trafo_mva": float(net.trafo.sn_mva.iloc[0]) if len(net.trafo) else None,
        "capacidad_kw": (float(net.trafo.sn_mva.iloc[0]) * 1000
                         * max(1, len(buses_subestacion(net)))
                         if len(net.trafo) else None),
    }


# ════════════════════════════════════════════════════════════
# 3. Zonas y puntos geográficos
# ════════════════════════════════════════════════════════════

def p_d_base(net):
    """Demanda base por bus en kW.

    Es la fórmula del notebook (celda 27) más dos correcciones que en la
    red del estudio no cambian nada y en una red subida sí:

      * se aplica `scaling`. pandapower lo usa para representar un
        escenario de carga y el flujo de potencia lo respeta, así que
        ignorarlo infla la demanda. MV Oberrhein viene con scaling 0.6:
        sin esto son 61.9 MW contra 50 MVA de capacidad instalada, o sea
        el modelo recortaría carga incluso sin ninguna falla y la ENS del
        caso base saldría inventada.
      * se descartan las cargas fuera de servicio, que por definición no
        consumen.

    En la red del estudio `scaling` es 1 y todas las cargas están en
    servicio, así que el total sigue siendo 44.742,1 kW (verificado
    contra `artefactos/red.pkl`).
    """
    cargas = net.load
    if "in_service" in cargas.columns:
        cargas = cargas[cargas.in_service.astype(bool)]
    escala = (cargas["scaling"] if "scaling" in cargas.columns
              else pd.Series(1.0, index=cargas.index))
    kw = cargas.p_mw.astype(float) * escala.astype(float) * 1000.0
    return {int(b): float(kw[cargas.bus == b].sum()) for b in net.bus.index}


def bus_zone_inicial(net, n_zonas=3):
    """Zona por bus de partida para la tabla editable.

    Prioridad: columna 'zona' de la red -> reparto por tramos contiguos de
    índice. El reparto por índice NO pretende ser correcto: es un punto de
    partida para que la usuaria corrija en la tabla, porque en una red
    radial los índices suelen seguir el orden del alimentador.
    """
    if "zona" in net.bus.columns and net.bus["zona"].notna().any():
        salida = {}
        for b in net.bus.index:
            v = net.bus.at[b, "zona"]
            salida[int(b)] = _a_zona(v)
        if len(set(salida.values())) > 1:
            return salida

    n = len(net.bus)
    corte = np.array_split(np.array(list(net.bus.index)), max(1, n_zonas))
    return {int(b): z + 1 for z, grupo in enumerate(corte) for b in grupo}


def _a_zona(valor):
    """'Z2' / 2 / '2' -> 2."""
    if isinstance(valor, str):
        digitos = "".join(ch for ch in valor if ch.isdigit())
        return int(digitos) if digitos else 1
    try:
        return int(valor)
    except (TypeError, ValueError):
        return 1


# Caja envolvente de Chile continental. Sirve para UNA cosa: decidir si un
# par de números que trae la red son de verdad coordenadas geográficas o
# son las cartesianas de dibujo que traen los casos de prueba (la CIGRE de
# pandapower viene con (7, 16), que es un lat/lon perfectamente válido en
# África). Sin este chequeo la app mostraría un mapa plausible y falso.
BBOX_CHILE = {"lat": (-56.0, -17.0), "lon": (-76.0, -66.0)}


def _en_chile(lat, lon):
    return (BBOX_CHILE["lat"][0] <= lat <= BBOX_CHILE["lat"][1]
            and BBOX_CHILE["lon"][0] <= lon <= BBOX_CHILE["lon"][1])


def coords_de_net(net):
    """{bus: (lat, lon)} desde net.bus, si la red trae georreferencia.

    Acepta las tres formas que aparecen en el repo: columnas lat/lon,
    columnas x_dss/y_dss (importadas de OpenDSS) y la columna 'geo' con
    GeoJSON, que es como el notebook deja la CIGRE.

    Ojo con el orden: el estándar GeoJSON es [lon, lat], pero la celda 2
    del notebook guarda [lat, lon]. Se desambigua probando ambos contra la
    caja de Chile; si ninguno cae dentro, se devuelve el par tal cual vino
    y `diagnostico_coords` lo marca como no georreferenciado.
    """
    salida = {}
    if {"lat", "lon"} <= set(net.bus.columns):
        for b in net.bus.index:
            la, lo = net.bus.at[b, "lat"], net.bus.at[b, "lon"]
            if pd.notna(la) and pd.notna(lo):
                salida[int(b)] = (float(la), float(lo))
        if salida:
            return salida

    if "geo" in net.bus.columns:
        for b in net.bus.index:
            g = net.bus.at[b, "geo"]
            # pandapower 3.x guarda GeoJSON como texto, pero una red
            # construida a mano puede traer ya el dict.
            if isinstance(g, str):
                if not g.strip().startswith("{"):
                    continue
                try:
                    par = json.loads(g)["coordinates"]
                except Exception:
                    continue
            elif isinstance(g, dict):
                par = g.get("coordinates")
                if par is None:
                    continue
            else:
                continue
            if len(par) != 2:
                continue
            a, c = float(par[0]), float(par[1])
            if _en_chile(a, c):
                salida[int(b)] = (a, c)          # ya viene [lat, lon]
            elif _en_chile(c, a):
                salida[int(b)] = (c, a)          # viene [lon, lat]
            else:
                # Ninguna orientación cae en Chile: red extranjera. Se
                # respeta el estándar GeoJSON, que es [lon, lat] (así viene
                # MV Oberrhein), para que el mapa de la red original se
                # dibuje en el lugar correcto del mundo. `puntos_por_zona`
                # igual se niega a proponer estos puntos como punto
                # climático, porque ARClim solo tiene datos de Chile.
                salida[int(b)] = (c, a)
        if salida:
            return salida

    if {"x_dss", "y_dss"} <= set(net.bus.columns):
        for b in net.bus.index:
            x, y = net.bus.at[b, "x_dss"], net.bus.at[b, "y_dss"]
            if pd.notna(x) and pd.notna(y):
                salida[int(b)] = (float(y), float(x))
        if salida:
            return salida

    # pandapower 2.x guardaba la georreferencia en una tabla aparte. Una red
    # subida desde un archivo de esa época no tiene columna 'geo'.
    geodata = getattr(net, "bus_geodata", None)
    if geodata is not None and len(geodata) and {"x", "y"} <= set(geodata.columns):
        for b in geodata.index:
            x, y = geodata.at[b, "x"], geodata.at[b, "y"]
            if pd.notna(x) and pd.notna(y):
                # (y, x) = (lat, lon): en pandapower x es la longitud.
                salida[int(b)] = (float(y), float(x))
    return salida


def diagnostico_coords(coords):
    """'chile' | 'fuera_de_chile' | 'sin_coordenadas'.

    Solo 'chile' habilita usar esas coordenadas como punto climático: el
    pipeline de ARClim y Open-Meteo consulta APIs chilenas y devolvería
    NaN (o el punto de grilla más cercano, que puede estar en otro país)
    para cualquier otra cosa.
    """
    if not coords:
        return "sin_coordenadas"
    dentro = sum(1 for lat, lon in coords.values() if _en_chile(lat, lon))
    return "chile" if dentro >= max(1, len(coords) // 2) else "fuera_de_chile"


def puntos_por_zona(net, bus_zone, solo_chile=True):
    """Un punto (lat, lon) por zona: centroide de los buses de esa zona que
    tengan coordenadas. Zona sin coordenadas -> no aparece en el dict, y la
    vista la pide a mano.

    Con `solo_chile` (el default) se devuelve {} si las coordenadas de la
    red no son geográficas chilenas: es preferible pedirlas a mano que
    proponer un centroide inventado que después se consulta contra ARClim.
    """
    coords = coords_de_net(net)
    if solo_chile and diagnostico_coords(coords) != "chile":
        return {}
    acumulado = {}
    for b, z in bus_zone.items():
        if b in coords:
            acumulado.setdefault(z, []).append(coords[b])
    return {z: (float(np.mean([p[0] for p in ps])), float(np.mean([p[1] for p in ps])))
            for z, ps in acumulado.items()}


def aplicar_geo(net, bus_zone, puntos):
    """Escribe en net.bus['geo'] el punto de la zona de cada bus.

    Reproduce lo que hace la celda 2 del notebook: todos los buses de una
    zona comparten el punto de la zona (`x_bus = x0`, sin desplazamiento).
    Es deliberado — el estudio no modela la posición individual de cada
    barra, solo a qué zona climática pertenece — pero conviene saberlo
    antes de leer un mapa donde 5 buses caen exactamente encima.
    """
    net = net.deepcopy() if hasattr(net, "deepcopy") else net
    for b in net.bus.index:
        z = bus_zone.get(int(b))
        if z in puntos:
            lat, lon = puntos[z]
            net.bus.at[b, "geo"] = json.dumps({"coordinates": [lat, lon], "type": "Point"})
    net.bus["zona"] = [f"Z{bus_zone.get(int(b), 1)}" for b in net.bus.index]

    if "geo" in net.line.columns or True:
        for l in net.line.index:
            zf = bus_zone.get(int(net.line.from_bus[l]), 1)
            zt = bus_zone.get(int(net.line.to_bus[l]), 1)
            if zf in puntos and zt in puntos:
                net.line.at[l, "geo"] = json.dumps({
                    "type": "LineString",
                    "coordinates": [list(puntos[zf]), list(puntos[zt])]})
    net.line["zone"] = [bus_zone.get(int(net.line.from_bus[l]), 1) for l in net.line.index]
    return net


# ════════════════════════════════════════════════════════════
# 4. Salidas
# ════════════════════════════════════════════════════════════

def df_systemzone(puntos, comunas=None):
    """Tabla con la forma exacta que leen los scripts del pipeline
    climático: columnas 'Zona', 'Lat', 'Lon' (read_coordinates de
    Extreme_events_arclim_final.py exige Lat y Lon, y usa Zona si está).

    El ORDEN DE LAS FILAS importa: es el que fija el índice `point_id` con
    el que queda indexado el eje `lat` del NetCDF de salida, y por lo tanto
    la columna de viento que `construir_faultlin.py` le asigna a cada zona.
    Por eso se ordena por número de zona y no por el orden de edición.
    """
    comunas = comunas or {}
    filas = [{"Zona": int(z), "Lat": round(float(lat), 6), "Lon": round(float(lon), 6),
              "Comuna": comunas.get(z, "")}
             for z, (lat, lon) in sorted(puntos.items())]
    return pd.DataFrame(filas, columns=["Zona", "Lat", "Lon", "Comuna"])


def exportar_systemzone(puntos, ruta, comunas=None):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df_systemzone(puntos, comunas).to_excel(ruta, index=False)
    return ruta


def guardar_paquete(net, bus_zone, puntos, *, numesc, n_eventos, buses_criticos,
                    tie_indices, n_horas, comunas=None, horas_reparacion=None,
                    seccionadores=None, perfiles=None,
                    ramales_fusible=None, largo_expuesto_km=None,
                    clientes_bus=None, clientes_denominador=None,
                    ruta=RUTA_PAQUETE, origen=""):
    """Escribe artefactos/red_nueva.pkl: todo lo que E0/E1 necesitan y que
    hoy solo sale de app.pkl.

    No escribe un app.pkl nuevo a propósito. app.pkl son 740 MB donde la
    red convive con los resultados congelados del estudio; lo que E0
    realmente rescata de ahí son seis claves (pipeline.py:236-260). Este
    paquete son esas seis claves y nada más, así que se puede versionar,
    revisar y rehacer sin mover 740 MB.
    """
    paquete = {
        "net": net,
        "bus_zone": {int(k): int(v) for k, v in bus_zone.items()},
        "puntos_zona": {int(z): (float(a), float(b)) for z, (a, b) in puntos.items()},
        "comunas": comunas or {},
        "P_D_base": p_d_base(net),
        "numesc": int(numesc),
        "N_eventos": int(n_eventos),
        "buses_criticos": [int(b) for b in buses_criticos],
        "tie_indices": [int(l) for l in tie_indices],
        # Líneas que llevan seccionador. Es un DATO de la red, no una decisión
        # del modelo, y sin él la variante de aislación realista no se puede
        # correr en un caso nuevo: `config_medida` lee esta lista de red.pkl y,
        # vacía, cualquier falla deja fuera al alimentador completo. Puede
        # quedar vacía con toda legitimidad: hay redes cuyos switches son de
        # extremo de línea y no seccionadores de tramo.
        "seccionadores_existentes": [int(l) for l in (seccionadores or [])],
        # Archivos de perfiles de ESTE caso. Sin esto, E1 usa los del estudio
        # de validación y una red nueva hereda en silencio su demanda.
        "perfiles": {k: str(v) for k, v in (perfiles or {}).items() if v},
        "n_horas": int(n_horas),
        # Parámetro del estudio, no constante del código: define cuánto dura
        # la interrupción y entra directo en SAIDI y ENS. Se guarda acá para
        # que quede junto a la red con la que se eligió.
        "horas_reparacion": (None if horas_reparacion is None else int(horas_reparacion)),
        "_meta": {
            "origen": origen,
            "generado_por": "entrada_red.guardar_paquete",
            "n_buses": len(net.bus),
            "n_lineas": len(net.line),
        },
    }
    # Opcionales de una red reducida (ver ramales.py): ramales con fusible como
    # elementos de falla propios y el largo de cada línea expuesto al viento.
    # Solo se escriben si vienen: un paquete sin ellos queda igual que antes.
    if ramales_fusible:
        import ramales as RAM
        RAM.validar(ramales_fusible, list(net.bus.index))
        paquete["ramales_fusible"] = {
            "id": [str(x) for x in ramales_fusible["id"]],
            "km": [float(x) for x in ramales_fusible["km"]],
            "zona": [int(x) for x in ramales_fusible["zona"]],
            "carga": [(int(i), int(b), float(fr)) for i, b, fr in ramales_fusible["carga"]],
        }
        if ramales_fusible.get("padre") is not None:
            paquete["ramales_fusible"]["padre"] = [int(x) for x in ramales_fusible["padre"]]
    if largo_expuesto_km:
        paquete["largo_expuesto_km"] = {int(l): float(v) for l, v in largo_expuesto_km.items()}
    # Clientes reales por barra (SAIDI/SAIFI en indicadores.py). Sin la clave,
    # los indicadores siguen con la regla de clientes por kW.
    if clientes_bus:
        paquete["clientes_bus"] = {int(b): float(c) for b, c in clientes_bus.items()}
    if clientes_denominador is not None:
        paquete["clientes_denominador"] = float(clientes_denominador)
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(paquete, f, protocol=pickle.HIGHEST_PROTOCOL)
    return ruta


def paquete_desde_red_pkl(art_dir, ruta=None):
    """Reconstruye `red_nueva.pkl` a partir del `red.pkl` de un caso.

    `red.pkl` (la salida de E0) es casi un superconjunto del paquete de
    entrada: de las ocho claves que E0 lee del origen, `red.pkl` tiene siete
    —todas menos `seccionadores_existentes`, que es opcional—. Así que un caso
    cuyo `red.pkl` vino de otra parte (un notebook, por ejemplo) puede
    recuperar su paquete sin volver a pasar la red por el asistente.

    Lo que el paquete tiene y `red.pkl` no, se rellena de donde se pueda:

      * `n_horas` y `numesc` de la forma de `fallas.npz`;
      * `horas_reparacion` de `fallas_meta.json`;
      * `puntos_zona` de las coordenadas de la red, y SOLO si son lat/lon
        chilenas: inventar un punto climático sería peor que no tenerlo,
        porque ARClim se consultaría en el lugar equivocado sin dar error;
      * `perfiles` vacío, o sea que E1 usará los archivos compartidos.

    Devuelve (ruta, faltantes) — `faltantes` nombra lo que no se pudo
    rellenar, para que la vista lo diga en vez de dejarlo implícito.
    """
    art_dir = Path(art_dir)
    with open(art_dir / "red.pkl", "rb") as f:
        red = pickle.load(f)

    n_horas, numesc = None, red.get("numesc")
    fallas = art_dir / "fallas.npz"
    if fallas.exists():
        with np.load(fallas) as z:
            forma = z["faultlinMC"].shape
        n_horas, numesc = int(forma[0]), int(forma[1])

    horas_rep = None
    meta_fallas = art_dir / "fallas_meta.json"
    if meta_fallas.exists():
        try:
            horas_rep = int(json.loads(meta_fallas.read_text(encoding="utf-8"))
                            .get("horas_reparacion"))
        except Exception:                                       # noqa: BLE001
            horas_rep = None

    net, bus_zone = red["net"], red["bus_zone"]
    puntos = puntos_por_zona(net, bus_zone)      # {} si no son chilenas

    faltantes = []
    if not puntos:
        faltantes.append("puntos geográficos por zona (los pide la cadena climática)")
    if n_horas is None:
        faltantes.append("horizonte en horas (no hay fallas.npz de donde leerlo)")
    if horas_rep is None:
        faltantes.append("horas de reparación (no hay fallas_meta.json)")

    ruta = Path(ruta) if ruta else art_dir / "red_nueva.pkl"
    guardar_paquete(
        net, bus_zone, puntos,
        numesc=int(numesc or 1), n_eventos=int(red.get("N_eventos") or 1),
        buses_criticos=red.get("buses_criticos") or [],
        tie_indices=red.get("tie_indices") or [],
        n_horas=int(n_horas or 8760),
        horas_reparacion=horas_rep,
        seccionadores=red.get("seccionadores_existentes") or [],
        clientes_bus=red.get("clientes_bus"),
        clientes_denominador=red.get("clientes_denominador"),
        perfiles={},
        origen=f"reconstruido desde {art_dir.name}/red.pkl",
        ruta=ruta)
    return ruta, faltantes


def cargar_paquete(ruta=RUTA_PAQUETE):
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe '{ruta.name}'. Se genera desde la pestaña "
            f"'🗺️ Datos de entrada' de la app.")
    with open(ruta, "rb") as f:
        return pickle.load(f)
