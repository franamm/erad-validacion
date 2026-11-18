"""
limites_dg.py — Tope de capacidad instalable de DG, por grupo de barras.

El tope histórico es POR BARRA:

    CAP_DG[g] <= lim_dg * P_D_base[g]

Es una cota que cierra el problema (sin ella el LP es no acotado cuando el
LCOE de la DG es menor que el precio de compra a la subestación), pero
además IMPONE EL EMPLAZAMIENTO: cada barra recibe capacidad en proporción
exacta a su propia demanda, así que el modelo nunca decide *dónde* conviene
instalar — solo puede llenar cada barra hasta su tope. Con costos lineales
en kW y sin costo fijo por punto de conexión, el óptimo es llenar todas.

Este módulo generaliza el tope a un AGREGADO POR GRUPO:

    sum_{g in G} CAP_DG[g] <= lim_dg * sum_{b in G} P_D_base[b]

Con el mismo `lim_dg` la cota TOTAL es idéntica, pero la asignación dentro
del grupo queda libre: el modelo puede concentrar la capacidad del grupo en
las barras donde más rinde. Eso es lo que convierte el emplazamiento en un
resultado en vez de un supuesto.

Modos de agrupación (`config["lim_dg_modo"]`):

  "bus"          una barra por grupo → equivale EXACTAMENTE al tope
                 histórico. Es el default, así que nada cambia salvo que
                 se pida explícitamente otro modo.
  "alimentador"  grupos topológicos: cada devanado de baja de un trafo de
                 la subestación es la cabecera de un alimentador, y el
                 grupo es todo lo que cuelga de ella recorriendo las
                 líneas normalmente cerradas (las tie-lines, normalmente
                 abiertas, no propagan pertenencia). Es el significado
                 eléctrico de "alimentador" y el que justifica la cota:
                 el trafo de cabecera es el que limita la inyección.
  "zona"         grupos según `bus_zone`. OJO: en esta red `bus_zone` es
                 una zonificación geográfica/climática (la que usan los
                 escenarios de viento), NO una partición radial — la zona
                 3 son las colas de dos alimentadores distintos. Sirve si
                 el tope que se quiere justificar es territorial, no
                 eléctrico.
  "seccion"      grupos según los seccionadores existentes: cada sección
                 de aislación es un grupo. Es el agregado alineado con la
                 física de la resiliencia (la sección es lo que queda
                 islanded), a costa de que los grupos cambian si cambia
                 la configuración de seccionadores.

Tope por barra adicional (`config["lim_dg_bus"]`, en múltiplos de la
demanda local):

    CAP_DG[g] <= lim_dg_bus * P_D_base[g]

Sin esto, el agregado por alimentador permite soluciones físicamente
absurdas: en esta red las barras 1 y 12 concentran el 89 % de la demanda
(19,8 y 20,0 MW de 44,7 MW), así que un tope de alimentador al 30 % deja
~7 MW de holgura que el modelo puede colgar entera de una barra cuyo
lateral sirve 431 kW. El tope por barra es el límite térmico/de tensión
del lateral, y es lo que en los estudios de hosting capacity acompaña
siempre al límite de cabecera. `None` lo desactiva.

Vive aparte de los optimizadores por la misma razón que `aislacion.py`:
`optimization_base` (que calcula E4/E5) y `optimization_6_oficial` (que
calcula las estrategias) TIENEN que aplicar el mismo tope, o la estrategia
se compara contra un caso base construido con otra física.
"""

MODOS = ("bus", "alimentador", "zona", "seccion")


def _grupos_por_bus(BUSES):
    return {f"bus_{b}": [b] for b in BUSES}


def _grupos_por_zona(BUSES, bus_zone):
    grupos = {}
    for b in BUSES:
        z = bus_zone.get(b)
        if z is None:
            continue
        grupos.setdefault(f"zona_{z}", []).append(b)
    return grupos


def _componentes(semillas, BUSES, lineas_activas, from_bus, to_bus):
    """Componentes conexas alcanzables desde cada semilla.

    Recorre solo `lineas_activas`. Una barra alcanzable desde dos semillas
    se asigna a la primera que la alcanza (no debería ocurrir en una red
    radial con las tie-lines excluidas; si ocurre, es señal de que la
    configuración normal no es radial).
    """
    incidentes = {}
    for l in lineas_activas:
        incidentes.setdefault(from_bus[l], []).append(l)
        incidentes.setdefault(to_bus[l], []).append(l)

    asignado = {}
    grupos = {}
    for nombre, raiz in semillas:
        if raiz in asignado:
            continue
        pila, miembros = [raiz], []
        asignado[raiz] = nombre
        while pila:
            b = pila.pop()
            miembros.append(b)
            for l in incidentes.get(b, ()):
                otro = to_bus[l] if from_bus[l] == b else from_bus[l]
                if otro not in asignado:
                    asignado[otro] = nombre
                    pila.append(otro)
        grupos[nombre] = sorted(miembros)
    return grupos


def _grupos_por_alimentador(net, BUSES, LINES, from_bus, to_bus, tie_indices):
    """Un grupo por devanado de baja de trafo de subestación.

    Las barras AT de la subestación (las `hv_bus`) quedan deliberadamente
    fuera de todo grupo: no son un punto de conexión de DG en media tensión.
    """
    hv = {int(b) for b in net.trafo.hv_bus}
    semillas = [(f"alim_{int(b)}", int(b)) for b in net.trafo.lv_bus]
    activas = [l for l in LINES
               if l not in tie_indices
               and from_bus[l] not in hv and to_bus[l] not in hv]
    return _componentes(semillas, BUSES, activas, from_bus, to_bus)


def _grupos_por_seccion(BUSES, LINES, from_bus, to_bus, tie_indices,
                        sec_existentes):
    """Un grupo por sección de aislación.

    Las líneas con seccionador se excluyen del recorrido, así que lo que
    queda conexo entre seccionadores es exactamente la sección que sale
    completa ante una falla interna (misma definición que `aislacion.py`).
    """
    excluidas = set(tie_indices) | set(sec_existentes)
    activas = [l for l in LINES if l not in excluidas]
    semillas = [(f"sec_{b}", b) for b in sorted(BUSES)]
    return _componentes(semillas, BUSES, activas, from_bus, to_bus)


def construir_grupos(modo, net, BUSES, LINES, from_bus, to_bus,
                     tie_indices=(), bus_zone=None, sec_existentes=()):
    """{nombre_grupo: [barras]} según `modo`. Ver MODOS."""
    if modo not in MODOS:
        raise ValueError(f"lim_dg_modo='{modo}' no válido. Opciones: {MODOS}")
    if modo == "bus":
        return _grupos_por_bus(BUSES)
    if modo == "zona":
        return _grupos_por_zona(BUSES, bus_zone or {})
    if modo == "alimentador":
        return _grupos_por_alimentador(net, BUSES, LINES, from_bus, to_bus,
                                       set(tie_indices))
    return _grupos_por_seccion(BUSES, LINES, from_bus, to_bus,
                               set(tie_indices), sec_existentes)


def leer_config(config, data, net, BUSES, LINES, from_bus, to_bus,
                tie_indices=(), bus_zone=None, sec_existentes=(),
                lim_default=1.0, verbose=False):
    """(grupos, Lim, lim_bus) listos para `aplicar_tope`.

    `lim_dg_modo` default "bus" → comportamiento histórico intacto.
    `lim_dg_bus` default None; se recomienda fijarlo cuando el modo no es
    "bus" (ver docstring del módulo).

    `lim_default` existe solo para conservar los defaults distintos que
    traían los dos optimizadores (1.0 en `optimization_6_oficial`, 0.3 en
    `optimization_base`) cuando `config` no trae `lim_dg`. En el camino
    normal `pipeline.config_medida` siempre lo fija explícitamente.
    """
    modo = str(config.get("lim_dg_modo", "bus"))
    Lim = float(config.get("lim_dg", lim_default))
    lim_bus = config.get("lim_dg_bus")
    lim_bus = None if lim_bus is None else float(lim_bus)

    grupos = construir_grupos(modo, net, BUSES, LINES, from_bus, to_bus,
                              tie_indices, bus_zone, sec_existentes)
    if verbose and modo != "bus":
        print(f"[LIM DG] modo={modo}  lim_dg={Lim}  lim_dg_bus={lim_bus}")
        for nombre, miembros in sorted(grupos.items()):
            print(f"[LIM DG]   {nombre}: {miembros}")
        sueltas = sorted(set(BUSES) - {b for m in grupos.values() for b in m})
        if sueltas:
            print(f"[LIM DG]   sin grupo (CAP forzada a 0): {sueltas}")
    return grupos, Lim, lim_bus


def aplicar_tope(model, prefijo, var, GENS_act, grupos, Lim, P_D_base,
                 lim_bus=None):
    """Agrega a `model` el tope agregado por grupo (y el por barra si aplica).

    `var` es la Var de capacidad (p.ej. `model.CAP_DG_sol`) y `GENS_act` las
    barras donde esa tecnología está habilitada. Las barras de `GENS_act`
    que no pertenecen a ningún grupo quedan con capacidad forzada a 0: no
    tienen cota que las respalde y dejarlas libres volvería a abrir el
    problema no acotado.

    Las restricciones se registran con nombres derivados de `prefijo` para
    que convivan las tres tecnologías en el mismo modelo.
    """
    import pyomo.environ as pyo

    activas = set(GENS_act)
    miembros = {n: [g for g in ms if g in activas] for n, ms in grupos.items()}
    miembros = {n: ms for n, ms in miembros.items() if ms}

    # Cota del grupo: se calcula sobre TODA la demanda del grupo, no solo
    # sobre las barras habilitadas, porque lo que la justifica es la carga
    # servida por la cabecera (o la sección), no dónde se puede conectar.
    cotas = {n: Lim * sum(float(P_D_base.get(b, 0.0)) for b in grupos[n])
             for n in miembros}

    setattr(model, f"{prefijo}_grupos", pyo.Set(initialize=sorted(miembros)))
    setattr(model, f"{prefijo}_cap_grupo", pyo.Constraint(
        getattr(model, f"{prefijo}_grupos"),
        rule=lambda m, n: sum(var[g] for g in miembros[n]) <= cotas[n]))

    if lim_bus is not None:
        setattr(model, f"{prefijo}_cap_bus", pyo.Constraint(
            GENS_act,
            rule=lambda m, g: var[g] <= lim_bus * float(P_D_base.get(g, 0.0))))

    huerfanas = sorted(activas - {g for ms in miembros.values() for g in ms})
    if huerfanas:
        setattr(model, f"{prefijo}_cap_cero", pyo.Constraint(
            huerfanas, rule=lambda m, g: var[g] <= 0.0))

    return miembros, cotas


def aplicar_tope_conjunto(model, prefijo, techs, grupos, Lim, P_D_base,
                          lim_bus=None):
    """Tope COMPARTIDO por varias tecnologías de DG (p.ej. solar + eólica).

    `aplicar_tope` acota cada tecnología por separado: con solar y eólica
    encendidas, cada una puede llegar a `Lim` y la DG total a 2·Lim. Acá el
    tope es sobre la SUMA de las capacidades de todas las tecnologías de
    `techs` (lista de `(var, GENS_act)`):

        sum_{t} sum_{g in G} CAP_t[g] <= Lim * sum_{b in G} P_D_base[b]

    y, si `lim_bus` no es None, por barra sobre la suma de las tecnologías.
    Con una sola tecnología es idéntico a `aplicar_tope`.
    """
    import pyomo.environ as pyo

    activas = [set(g) for _, g in techs]
    todas = set().union(*activas) if activas else set()
    miembros = {}
    for n, ms in grupos.items():
        por_tec = [[g for g in ms if g in act] for act in activas]
        if any(por_tec):
            miembros[n] = por_tec
    cotas = {n: Lim * sum(float(P_D_base.get(b, 0.0)) for b in grupos[n])
             for n in miembros}

    setattr(model, f"{prefijo}_grupos", pyo.Set(initialize=sorted(miembros)))
    setattr(model, f"{prefijo}_cap_grupo", pyo.Constraint(
        getattr(model, f"{prefijo}_grupos"),
        rule=lambda m, n: sum(var[g] for (var, _), gs in zip(techs, miembros[n])
                              for g in gs) <= cotas[n]))

    if lim_bus is not None:
        setattr(model, f"{prefijo}_cap_bus", pyo.Constraint(
            sorted(todas),
            rule=lambda m, g: sum(var[g] for (var, gens), act in zip(techs, activas)
                                  if g in act) <= lim_bus * float(P_D_base.get(g, 0.0))))

    en_grupo = {g for por_tec in miembros.values() for gs in por_tec for g in gs}
    huerfanas = sorted(todas - en_grupo)
    if huerfanas:
        setattr(model, f"{prefijo}_cap_cero", pyo.Constraint(
            huerfanas, rule=lambda m, g: sum(var[g] for (var, _), act in zip(techs, activas)
                                             if g in act) <= 0.0))
    return miembros, cotas


def etiqueta_variante(config):
    """Sufijo para los artefactos de E4/E5.

    Dos topes distintos NO son comparables (el CAPEX/OPEX de DG sin fallas
    que E4 entrega al objetivo depende de cuánta capacidad se dejó
    instalar y dónde), así que cada combinación necesita su propio
    artefacto. En modo "bus" devuelve la etiqueta histórica para no mover
    de sitio nada de lo ya calculado.
    """
    modo = str(config.get("lim_dg_modo", "bus"))
    lim = float(config.get("lim_dg", 1))
    conj = "-CONJ" if config.get("lim_dg_conjunto") else ""   # tope compartido sol+eol
    if modo == "bus":
        return f"LIM-{lim:.2f}{conj}"
    lim_bus = config.get("lim_dg_bus")
    suf = "" if lim_bus is None else f"-B{float(lim_bus):.2f}"
    return f"LIM-{lim:.2f}-{modo.upper()[:4]}{suf}{conj}"
