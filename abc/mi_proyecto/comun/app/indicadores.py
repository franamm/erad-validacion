"""
indicadores.py — Métricas de confiabilidad: SAIDI, SAIFI, ENS, ASAI.
"""
import numpy as np

# Todas las funciones de este módulo reciben BUSES / P_D_base / horas como
# argumentos, así que NO necesitan el estado global de data_loader. El import
# se sacó a propósito: data_loader arrastra streamlit y los artefactos del
# pipeline, lo que impedía usar los indicadores desde un script de consola
# (p. ej. run_barrido_medidas.py) sin levantar la app.


def _results_to_list(results_dict):
    """{esc: [dict por hora]} -> [[dict por hora], ...] ordenado por escenario.

    Copia local de helpers_operativos._results_to_list, por la misma razón:
    ese módulo también depende de data_loader.
    """
    return [results_dict[k] for k in sorted(results_dict.keys())]

# BLOQUE 2 — Indicadores de confiabilidad
# ─────────────────────────────────────────────────────────────

# def fn_SAIDI(results_filtrados, P_D_base, BUSES, horas, clientes_por_kW=0.5):
#     """SAIDI [h/cliente] = horas de interrupción ponderadas por clientes."""
#     if not results_filtrados or len(results_filtrados) < len(horas):
#         return 0.0
#     N_bus   = {b: clientes_por_kW * P_D_base[b] for b in BUSES}
#     N_total = sum(N_bus.values())
#     if N_total == 0:
#         return 0.0
#     Ui = {b: 0 for b in BUSES}
#     for h in horas:
#         P_LC = results_filtrados[h - 1].get("P_LC", {})
#         for b in BUSES:
#             if P_LC.get(b, 0) > 0:
#                 # P_D_ = results_filtrados[h - 1].get("Pb", {}).get(b, 0) - P_LC.get(b, 0)
#                 Ui[b] += 1
#     return sum(N_bus[b]* (Ui[b]) for b in BUSES) / N_total

# def fn_SAIDI(results_filtrados, P_D_base, BUSES, horas, clientes_por_kW=0.5):
#     """SAIDI [h/cliente] = horas de interrupción ponderadas por clientes."""
#     if not results_filtrados or len(results_filtrados) < len(horas):
#         return 0.0
#     N_bus   = {b: clientes_por_kW * P_D_base[b] for b in BUSES}
#     N_total = sum(N_bus.values())
#     if N_total == 0:
#         return 0.0
#     Ui = {(b,h): 0.0 for b in BUSES for h in horas}
#     P_D_ = {}
#     for h in horas:
#         P_LC = results_filtrados[h - 1].get("P_LC", {})
#         for b in BUSES:
#             P_D_[b,h] = results_filtrados[h - 1].get("Pb", {}).get(b, 0) + P_LC.get(b, 0)
#             if P_LC.get(b, 0) > 0:
#                 Ui[b,h] += 1
#     return sum(N_bus[b] * (Ui[b,h]/P_D_[b,h]) if Ui[b,h] != 0.0 else 0.0 for b in BUSES for h in horas) / N_total 

# def fn_SAIDI(results_filtrados, P_D_base, BUSES, horas, clientes_por_kW=0.5):
#     """SAIDI [h/cliente] = horas de interrupción ponderadas por clientes."""
#     if not results_filtrados or len(results_filtrados) < len(horas):
#         return 0.0
#     N_bus   = {b: clientes_por_kW * P_D_base[b] for b in BUSES}
#     N_total = sum(N_bus.values())
#     if N_total == 0:
#         return 0.0
#     Ui = {(b,h): 0.0 for b in BUSES for h in horas}
#     P_D_ = {}
#     P_LC = {}
#     for h in horas:
#         for b in BUSES:
#             P_LC[b,h] = results_filtrados[h - 1].get("P_LC", {}).get(b,0)
#             P_D_[b,h] = results_filtrados[h - 1].get("Pb", {}).get(b, 0) + P_LC[b,h]
#     return sum(N_bus[b] * (P_LC[b,h]/P_D_[b,h]) if P_LC[b,h] > 0 else 0.0 for b in BUSES for h in horas) / N_total 
def fn_SAIDI(
    results_filtrados,
    P_D_base,
    BUSES,
    horas,
    clientes_por_kW=0.5,
    delta_t=1.0,
    umbral_potencia=1e-9,
    clientes=None
):
    """
    SAIDI equivalente [h/cliente].

    `clientes` ({barra: n.º de clientes}, red.pkl["clientes_bus"]) pondera
    cada barra por sus clientes reales. Sin él se usa la regla
    clientes_por_kW · P_D_base, y el resultado no cambia respecto de antes.

    Se aproxima la fracción de clientes sin suministro mediante:

        q_bh = P_LC_bh / P_D_bh

    donde:

        P_D_bh = Pb_bh + P_LC_bh

    Supuestos:
    - Pb es la potencia suministrada en la barra [kW o MW].
    - P_LC es la potencia no suministrada o carga recortada.
    - P_D es la demanda total de la barra.
    - q_bh representa aproximadamente la fracción de clientes
      equivalentes sin suministro.
    - delta_t es la duración de cada intervalo en horas.
    """

    if not results_filtrados:
        return 0.0

    if len(results_filtrados) != len(horas):
        raise ValueError(
            "results_filtrados y horas deben tener la misma longitud"
        )

    if clientes is not None:
        N_bus = {b: max(float(clientes.get(b, 0.0)), 0.0) for b in BUSES}
    else:
        N_bus = {
            b: clientes_por_kW * max(
                float(P_D_base.get(b, 0.0)),
                0.0
            )
            for b in BUSES
        }

    N_total = sum(N_bus.values())

    if N_total <= 0.0:
        return 0.0

    saidi = 0.0

    for idx, h in enumerate(horas):

        resultado_h = results_filtrados[idx]

        for b in BUSES:

            P_LC = max(
                float(
                    resultado_h.get("P_LC", {}).get(b, 0.0)
                ),
                0.0
            )

            P_serv = max(
                float(
                    resultado_h.get("Pb", {}).get(b, 0.0)
                ),
                0.0
            )

            P_D = P_serv + P_LC

            if P_LC <= umbral_potencia or P_D <= 0.0:
                q = 0.0
            else:
                q = min(P_LC / P_D, 1.0)

            saidi += (
                N_bus[b]
                * q
                * delta_t
            )

    return saidi / N_total
# def _contar_eventos_interrupcion(results_esc, buses_sel=None, umbral=1e-9):
#     """Cuenta eventos por bloques consecutivos de horas con P_LC > umbral por bus.
#     Coherente con optimization_3.py, donde los eventos se manejan como bloques horarios consecutivos."""
#     buses_sel = buses_sel or BUSES
#     if not results_esc:
#         return {b: 0 for b in buses_sel}

#     eventos = {b: 0 for b in buses_sel}
#     en_evento = {b: False for b in buses_sel}

#     for h in horas:
#         if h - 1 >= len(results_esc):
#             continue
#         plc_now = results_esc[h - 1].get("P_LC", {})
#         P_D = {}
#         for b in buses_sel:
#             interrumpido = plc_now.get(b, 0) > umbral
#             if interrumpido and not en_evento[b]:
#                 eventos[b] += 1
#                 en_evento[b] = True
#             elif not interrumpido:
#                 en_evento[b] = False
#     return eventos

# # def fn_SAIFI(results_filtrados, P_D_base, BUSES, horas, clientes_por_MW=0.5):
# #     """SAIFI [int/cliente] = número de eventos de interrupción por cliente,
# #     contando bloques consecutivos de horas con P_LC > 0 como un único evento."""
# #     if not results_filtrados or len(results_filtrados) < len(horas):
# #         return 0.0
# #     N_bus   = {b: clientes_por_MW * P_D_base[b] for b in BUSES}
# #     N_total = sum(N_bus.values())
# #     if N_total == 0:
# #         return 0.0
# #     Fi = _contar_eventos_interrupcion(results_filtrados, BUSES)
# #     return sum(Fi[b] * N_bus[b] for b in BUSES) / N_total

# def fn_SAIFI(results_filtrados,P_D_base,BUSES,horas,):
#     """
#     SAIFI ponderado por demanda.

#     Supuesto: cualquier P_LC > 0 implica que toda la barra
#     experimenta una interrupción.
#     """
#     if not results_filtrados:
#         return 0.0

#     pesos = {b: max(float(P_D_base[b]), 0.0) for b in BUSES}
#     peso_total = sum(pesos.values())

#     if peso_total <= 0:
#         return 0.0

#     Fi = _contar_eventos_interrupcion(results_filtrados,buses_sel=BUSES,)
#     return sum(Fi[b]*pesos[b] for b in BUSES) / peso_total


def _fraccion_clientes_interrumpidos(
    resultado_h,
    b,
    umbral_potencia=1e-9
):
    """
    Calcula la fracción equivalente de clientes sin suministro
    en una barra durante una hora.

    Supuestos:
    Pb   : potencia suministrada en la barra [MW].
    P_LC : potencia no suministrada o carga recortada [MW].
    P_D  : demanda total de la barra [MW] = Pb + P_LC.

    La fracción P_LC / P_D se interpreta como la fracción
    equivalente de clientes interrumpidos.
    """

    P_serv = max(
        float(resultado_h.get("Pb", {}).get(b, 0.0)),
        0.0
    )

    P_LC = max(
        float(resultado_h.get("P_LC", {}).get(b, 0.0)),
        0.0
    )

    if P_LC <= umbral_potencia:
        return 0.0

    P_D = P_serv + P_LC

    if P_D <= 0.0:
        return 0.0

    q = P_LC / P_D

    return min(max(q, 0.0), 1.0)
def _eventos_de_falla(results):
    """(activa, nueva) por hora, o None si los resultados no traen fallas.

    activa[h]: hay líneas dañadas o fuera de servicio (falla en curso).
    nueva[h] : aumenta el n.º de líneas dañadas respecto de la hora anterior,
               o sea, empieza una falla distinta. Solo usa conteos para dar lo
               mismo en el baseline y en las corridas guardadas (.npz).
    """
    if not any("n°lineas_danadas" in r for r in results):
        return None
    activa, nueva, prev = [], [], 0
    for r in results:
        dan = int(r.get("n°lineas_danadas", 0) or 0)
        fue = int(r.get("n°lineas_fuera_servicio", 0) or 0)
        activa.append(dan > 0 or fue > 0)
        nueva.append(dan > prev)
        prev = dan
    return activa, nueva


def _saifi_por_falla(results, BUSES, pesos, eventos, umbral_potencia):
    """Σ_barras peso · Σ_fallas (fracción de clientes interrumpida por la falla).

    Por barra se lleva `contado`: la mayor fracción q ya contada en la falla en
    curso. Solo la parte de q que supera lo contado suma, así que una falla es
    un único término aunque q baje (reposición por bloques) y vuelva a subir
    (la generación se agota). Si empieza otra falla, lo ya interrumpido en ese
    instante (q_anterior) no se vuelve a contar. La falla termina cuando no hay
    líneas dañadas ni fuera de servicio y q llegó a 0. Sin información de
    fallas, cada tramo continuo con q > 0 es un evento.
    """
    activa, nueva = eventos if eventos else (None, None)
    total = 0.0
    for b in BUSES:
        contado, q_ant = 0.0, 0.0
        for idx, r in enumerate(results):
            q = _fraccion_clientes_interrumpidos(
                resultado_h=r, b=b, umbral_potencia=umbral_potencia
            )
            if nueva is not None and nueva[idx]:
                contado = min(contado, q_ant)
            if q > contado:
                total += pesos[b] * (q - contado)
                contado = q
            if q <= 0.0 and (activa is None or not activa[idx]):
                contado = 0.0
            q_ant = q
    return total


def fn_SAIFI(
    results_filtrados,
    P_D_base,
    BUSES,
    horas,
    umbral_potencia=1e-9,
    clientes=None,
    clientes_denominador=None,
    por_falla=True
):
    """
    SAIFI equivalente [interrupciones/cliente].

    `por_falla=True` (default) sigue la NT de Calidad de Servicio en
    Distribución: SAIFI = Σ_i Clientes_fs_i / Clientes_inst, con un término
    por falla o desconexión i, y cada cliente cuenta UNA vez por falla. La
    reposición gradual (bloques de reposición) y los cortes posteriores de la
    misma falla no suman. Ver `_saifi_por_falla`. `por_falla=False` conserva la
    definición anterior (suma de todas las subidas de q).

    `clientes_denominador`, si se entrega, es el total de clientes de la
    comuna/area y puede incluir clientes fuera de la red modelada. La carga
    interrumpida se reparte proporcionalmente entre los clientes/SED de cada
    barra mediante la fraccion P_LC / (Pb + P_LC).

    `clientes` ({barra: n.º de clientes}) pondera cada barra por sus
    clientes reales; sin él, el peso es la demanda P_D_base, como antes.

    q_bh = P_LC_bh / P_D_bh representa la fracción
    equivalente de clientes sin suministro.

    Solo los incrementos positivos de q se interpretan
    como nuevos clientes interrumpidos.

    Una disminución de q representa restauración.
    Un aumento posterior representa una nueva interrupción
    de clientes equivalentes.
    """

    if not results_filtrados:
        return 0.0

    if len(results_filtrados) != len(horas):
        raise ValueError(
            "results_filtrados y horas deben tener la misma longitud"
        )

    fuente = clientes if clientes is not None else P_D_base
    pesos = {
        b: max(float(fuente.get(b, 0.0)), 0.0)
        for b in BUSES
    }

    peso_total = sum(pesos.values())
    denominador = (
        max(float(clientes_denominador), 0.0)
        if clientes_denominador is not None
        else peso_total
    )

    if peso_total <= 0.0 or denominador <= 0.0:
        return 0.0

    if por_falla:
        eventos = _eventos_de_falla(results_filtrados)
        return _saifi_por_falla(
            results_filtrados, BUSES, pesos, eventos, umbral_potencia
        ) / denominador

    q_anterior = {
        b: 0.0 for b in BUSES
    }

    saifi = 0.0

    for idx, h in enumerate(horas):

        resultado_h = results_filtrados[idx]

        for b in BUSES:

            q_actual = _fraccion_clientes_interrumpidos(
                resultado_h=resultado_h,
                b=b,
                umbral_potencia=umbral_potencia
            )

            nuevos_clientes = max(
                q_actual - q_anterior[b],
                0.0
            )

            saifi += (
                pesos[b]
                * nuevos_clientes
            )

            q_anterior[b] = q_actual

    return saifi / denominador

def fn_ENS_total(results_filtrados, BUSES, horas):
    """ENS acumulada total [MWh]."""
    if not results_filtrados or len(results_filtrados) < len(horas):
        return 0.0
    return sum(results_filtrados[h - 1].get("P_LC", {}).get(b, 0)
            for h in horas for b in BUSES
            if h - 1 < len(results_filtrados))


def fn_ASAI(saidi, periodo_horas=8760):
    """ASAI = 1 - SAIDI/periodo  (índice de disponibilidad promedio)."""
    return 1.0 - (saidi / periodo_horas if periodo_horas > 0 else 0.0)


def fn_ens_costo_base(results_sin_dict, horas, BUSES, VoLL, N_eventos):
    """Costo de ENS caso base = sum_b(VoLL_b * P_LC_b) * N_eventos.
    Usa VoLL diferenciado por bus para reflejar correctamente cargas críticas vs. no críticas."""
    esc_list   = _results_to_list(results_sin_dict)
    costos_esc = []
    for esc in esc_list:
        if not esc:
            continue
        c = sum(
            VoLL.get(b, 1.0) * esc[h - 1].get("P_LC", {}).get(b, 0)
            for h in horas for b in BUSES if h - 1 < len(esc)
            ) * N_eventos
        costos_esc.append(c)
    return np.mean(costos_esc) if costos_esc else 0.0

# ─────────────────────────────────────────────────────────────
