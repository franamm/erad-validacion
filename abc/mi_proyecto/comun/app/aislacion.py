"""
aislacion.py — Zona de aislación de la falla por seccionadores.

Sin esta capa, un optimizador que ve fallar la línea `l` saca de servicio
ÚNICAMENTE esa línea, lo que equivale a suponer un seccionador en cada
extremo de cada línea: la falla se aísla siempre, gratis y al instante.
Es un supuesto optimista que subestima ENS/SAIDI/SAIFI y deja sin
beneficio a la inversión en switches.

El comportamiento real es que los seccionadores particionan el alimentador
en secciones, y una falla dentro de una sección deja FUERA A TODA LA
SECCIÓN: la protección aguas arriba despeja y la reposición solo alcanza
hasta el seccionador más cercano (zona de aislación / protection zone).

Este módulo lo resuelve para un conjunto FIJO de seccionadores (dato de
entrada, no decisión del modelo), así que recorre el grafo en vez de
agregar restricciones al MILP. Vive aparte de los optimizadores para que
`optimization_base` y `optimization_6_oficial` compartan exactamente la
misma lógica sin depender uno del otro.

Referencias: Ettehad et al. (2024), Electric Power Systems Research
232:110361, ecs. (32)-(34); Shahbazian et al. (2020); Chen et al. (2019).
"""


def adyacencia_lineas(LINES, from_bus, to_bus):
    """{bus: [líneas incidentes]} — soporte del recorrido de `zona_aislada`."""
    lineas_en_bus = {}
    for l in LINES:
        lineas_en_bus.setdefault(from_bus[l], []).append(l)
        lineas_en_bus.setdefault(to_bus[l], []).append(l)
    return lineas_en_bus


def zona_aislada(dam_vec, LINES, from_bus, to_bus, lineas_en_bus,
                 sec_existentes, tie_indices=frozenset()):
    """Líneas fuera de servicio al aislar las fallas de `dam_vec`.

    La salida se propaga entre líneas que comparten un bus hasta topar con
    una línea que lleva seccionador: esa se abre —queda fuera de servicio—
    y detiene la propagación.

    Ejemplo: seccionadores en {2-3, 6-7, 9-10} y falla en 4-5 → la salida
    se propaga 4-5 → 3-4 → 2-3 (corta) y 4-5 → 5-6 → 6-7 (corta). Fuera de
    servicio = {2-3, 3-4, 4-5, 5-6, 6-7}; el tramo desde el bus 7 hacia
    abajo queda sano y solo se repone si una tie-line lo re-alimenta.

    Casos borde que fijan la interpretación:
      * sin seccionadores → cae el alimentador completo (solo despeja el
        interruptor de cabecera);
      * seccionador en cada línea → sale solo la línea averiada, que es
        el supuesto implícito cuando la aislación está apagada.

    Las tie-lines quedan al margen de la propagación: normalmente abiertas,
    no conducen corriente de falla. NO son seccionadores — son líneas de
    enlace que sirven para REPONER el tramo sano aguas abajo, mientras que
    el seccionador es el equipo que CORTA para acotar la sección.

    Devuelve un vector 0/1 del mismo tipo que `dam_vec`.
    """
    sec_existentes = set(sec_existentes)
    tie_indices = set(tie_indices)
    fuera = {l for l in LINES if dam_vec[l] == 1 and l not in tie_indices}
    frontera = [l for l in fuera if l not in sec_existentes]
    while frontera:
        l = frontera.pop()
        for b in (from_bus[l], to_bus[l]):
            for m in lineas_en_bus.get(b, ()):
                if m in fuera or m in tie_indices:
                    continue
                fuera.add(m)
                if m not in sec_existentes:
                    frontera.append(m)
    out = dam_vec.copy()
    for l in fuera:
        out[l] = 1
    return out


def leer_config(config, data, LINES):
    """(usar_aislacion, set de seccionadores) desde `config`/`data`.

    Centraliza la lectura para que los dos optimizadores acepten las
    mismas llaves y no se desincronicen.
    """
    usar = bool(config.get("aislacion_fallas_fija", False))
    crudo = config.get("seccionadores_existentes",
                       data.get("seccionadores_existentes", []) if data else [])
    sec = {l for l in (crudo or []) if l in set(LINES)}
    return usar, sec


def variante(usar_aislacion, sec_existentes):
    """Etiqueta corta y estable del supuesto de aislación.

    Se usa para separar los artefactos de caso base: dos supuestos
    distintos NO son comparables entre sí, y sobrescribir uno con el otro
    destruye la comparación 'aislación perfecta vs. realista', que es un
    resultado en sí. Devuelve 'base' para el supuesto histórico (sin
    aislación), de modo que los baselines ya calculados siguen siendo
    válidos y no hay que rehacerlos.
    """
    sec = sorted(sec_existentes or [])
    if not usar_aislacion or not sec:
        return "base"
    etiqueta = "sec-" + "-".join(str(l) for l in sec)
    if len(etiqueta) <= 48:
        return etiqueta
    import hashlib
    h = hashlib.md5("-".join(str(l) for l in sec).encode()).hexdigest()[:10]
    return f"sec-{len(sec)}x-{h}"


def expandir_dano(por_hora, horas, LINES, from_bus, to_bus, lineas_en_bus,
                  sec_existentes, tie_indices):
    """Aplica `zona_aislada` a un dict {hora: vector de daño}."""
    if por_hora is None:
        return None
    return {t: zona_aislada(por_hora[t], LINES, from_bus, to_bus,
                            lineas_en_bus, sec_existentes, tie_indices)
            for t in horas}
