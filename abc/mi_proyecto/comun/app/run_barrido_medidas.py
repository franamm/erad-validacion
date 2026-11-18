#!/usr/bin/env python
# ============================================================
# run_barrido_medidas.py — barrido medida x escenario, promedio de
# indicadores, escenario representativo y corrida combinada
# ============================================================
# Responde a la pregunta metodológica: "no puedo correr 100 escenarios con
# todas las medidas; ¿cuál escenario uso como representativo?".
#
# FASE 1 — cada medida POR SEPARADO sobre N escenarios (default 10).
#          Una corrida determinista de optimization_6_oficial por
#          (medida, escenario), con esc_debug = escenario.
# FASE 2 — promedio de los indicadores sobre esos escenarios y elección,
#          PARA CADA MEDIDA, del escenario más cercano a su promedio.
#          Ese es el escenario "representativo" de la medida.
# FASE 3 — corrida con TODAS las medidas de la fase 1 encendidas a la vez,
#          sobre cada escenario representativo distinto (y sobre las
#          combinaciones extra que se pidan con --combo).
#
# Todo es reanudable: cada corrida se guarda en disco apenas termina, y una
# segunda ejecución omite lo que ya existe (--forzar lo rehace).
#
# USO
#   # ver qué se va a correr, sin resolver nada
#   python run_barrido_medidas.py --escenarios 0-9 --listar
#
#   # fase 1 y 2 sobre 10 escenarios, horizonte de 24 h (prueba rápida)
#   python run_barrido_medidas.py --escenarios 0-9 --horas 24 --fases 1,2
#
#   # barrido completo con un subconjunto de medidas
#   python run_barrido_medidas.py --escenarios 0-9 \
#       --medidas dg_sol,ess_4h,veg_mgmt,inv_switch_rcs --fases 1,2,3
#
#   # solo rehacer la fase 2 con otro criterio (no re-resuelve nada)
#   python run_barrido_medidas.py --escenarios 0-9 --fases 2 --criterio multi
#
# CRITERIO DE "MÁS CERCA DEL PROMEDIO" (--criterio)
#   objetivo (default) : |fn_objetivo_esc - promedio|. Es la métrica que el
#                        modelo minimiza: CAPEX+OPEX de la medida + ENS
#                        valorizada (VoLL·P_LC·N_eventos) - ahorro de
#                        subestación (ver optimization_6_oficial:1317).
#                        Las constantes -CAPEX/OPEX_sin_fallas de DG sol/eól
#                        son iguales en todos los escenarios de una misma
#                        medida, así que no alteran el argmin.
#   ens                : |ENS_esc - promedio|, solo energía no suministrada.
#   multi              : distancia euclidiana sobre (SAIDI, SAIFI, ENS)
#                        estandarizados (z-score entre escenarios).
#
# OJO con el MIPGap: el objetivo trae hasta `--mip_gap` de holgura, así que
# dos escenarios separados por menos que eso se ordenan por ruido del solver.
# El resumen imprime la dispersión para que se vea si el caso aplica.
# ============================================================

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline as P
from aislacion import variante as _variante
import limites_dg as LDG
from indicadores import fn_ASAI, fn_ENS_total, fn_SAIDI, fn_SAIFI

APP_DIR = Path(__file__).resolve().parent
ART_DIR = P.ART_DIR


# ════════════════════════════════════════════════════════════
# Catálogo de medidas
# ════════════════════════════════════════════════════════════
# La clave es la misma que lee optimization_6_oficial desde `config`.
# `microredes` NO está acá a propósito: no es una medida por sí sola, es un
# modificador de los switches (la app lo exige junto a RCS/MS). Se activa
# con --microredes y solo aplica a las corridas que llevan switch.

MEDIDAS = {
    "dg_sol":         "DG solar",
    "dg_eol":         "DG eolica",
    "dg_fos":         "DG fosil",
    "ess_4h":         "ESS 4h",
    "ess_5h":         "ESS 5h",
    "ess_6h":         "ESS 6h",
    "reconfig_topo":  "Reconfiguracion topologica",
    "inv_tie":        "Enlaces nuevos",
    "inv_switch_rcs": "Switches RCS",
    "inv_switch_ms":  "Switches MS",
    "veg_mgmt":       "Gestion de vegetacion",
    "refuerzo":       "Refuerzo estructural",
    "soterramiento":  "Soterramiento",
    "meg":            "MEG (generacion movil)",
    "meg_arriendo":   "MEG arrendado (por evento)",
    "mess":           "MESS (almacenamiento movil)",
    "edrp":           "EDRP",
    "prior_criticos": "Priorizacion de cargas criticas",
}

# Variables de decisión de inversión que vale la pena guardar de cada corrida
# (son constantes en el tiempo, así que basta con leerlas de la hora 0).
CLAVES_PLAN = [
    "CAP_DG_sol_kW", "CAP_DG_eol_kW", "CAP_DG_fos_kW",
    "CAP_ESS_kWh_4h", "CAP_ESS_kWh_5h", "CAP_ESS_kWh_6h",
    "y_tie", "y_rcs", "y_ms", "x_veg", "x_ref", "x_ug",
    "CAP_MEG_MW", "y_mess", "CAP_MESS_MWh",
]

INDICADORES = ["objetivo", "ens", "saidi", "saifi", "asai", "costo_ens"]

# Descomposicion economica del objetivo (helpers_inversion, la misma que usa
# la app). Permite comparar costo-efectividad entre medidas leyendo solo el
# CSV, sin abrir los .pkl.gz.
# OJO: la clave "beneficio" del helper NO es un beneficio, es una copia literal
# de fn_objetivo (helpers_inversion:379 y :695). Se deja fuera por duplicada y
# porque el nombre invita a leerla mal; en su lugar va `ahorro_neto`, derivado
# acá de forma explícita.
# `opex_anual` es el TOTAL y se conserva para no romper lecturas existentes;
# `opex_medidas` y `termino_subestacion` son sus dos mitades, que suman igual.
# Se reportan las tres porque mirar solo el total impide distinguir un OPEX alto
# de un crédito de subestación bajo, que es lo que pasa en la GD renovable.
COSTOS = ["capex_anual", "opex_anual", "opex_medidas", "termino_subestacion",
          "termino_subestacion_falla", "opex_resiliencia",
          "costo_ens_base", "costo_ens_estr", "ahorro_ens", "costo_total",
          "sobrecosto_dg", "objetivo_plan"]
_COLS_COSTOS = [f"cst_{k}" for k in COSTOS] + ["cst_ahorro_neto"]

# Salud del solver. Viajan al resumen porque un incumbente cortado por
# TimeLimit se ve idéntico a un óptimo en todas las demás columnas.
_COLS_SOLVER = ["solver_term", "solver_gap", "limite_tiempo"]

# Resultados físicos de la corrida: capacidad instalada, equipos, energía por
# tecnología y operación. Son los mismos que la app muestra para una corrida
# suelta; acá se guardan para las 170 del barrido.
AGREGADOS = [
    # capacidad instalada
    "cap_dg_sol_kW", "cap_dg_eol_kW", "cap_dg_fos_kW",
    "cap_ess_4h_kWh", "cap_ess_5h_kWh", "cap_ess_6h_kWh",
    "cap_meg_MW", "cap_mess_MWh",
    # equipos instalados (conteo de binarias) y km intervenidos
    "n_rcs", "n_ms", "n_tie_nuevas", "n_mess",
    "n_veg", "n_ref", "n_ug", "km_veg", "km_ref", "km_ug",
    # energía del año
    "E_dg_sol", "E_dg_eol", "E_dg_fos", "E_meg",
    "E_ess_descarga", "E_ess_carga", "E_mess_descarga", "E_mess_carga",
    "E_edrp", "E_sub", "E_servida",
    # operación
    "maniobras_tie", "maniobras_sw", "horas_isla",
    # MEG arrendado (al final: las tandas viejas quedan con estas en NaN)
    "E_meg_arr", "n_arriendos_meg",
]

# Dimensionamiento del plan: cuánto instala la medida. Es la única familia de
# resultados que NO queda descrita por su valor esperado. Los indicadores son
# magnitudes continuas y su promedio equiprobable es el valor que se espera
# enfrentar; la capacidad instalada es una DECISIÓN, y ahí un promedio de
# 6.700 kW puede venir de "los diez escenarios instalan 6.700" o de "cinco
# instalan 13.400 y cinco nada", que son planes de inversión distintos. Por eso
# estas columnas llevan además el rango [min, max] entre escenarios.
_COLS_DIMENSION = [c for c in AGREGADOS
                   if c.startswith(("cap_", "n_", "km_"))]


# ════════════════════════════════════════════════════════════
# Utilidades
# ════════════════════════════════════════════════════════════

def _log(msg):
    print(f"[barrido] {msg}", flush=True)


def parse_lista_int(txt):
    """'0-9' | '0,3,7' | '3' -> [0,1,...]"""
    out = []
    for parte in str(txt).split(","):
        parte = parte.strip()
        if not parte:
            continue
        if "-" in parte:
            a, b = parte.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(parte))
    return sorted(set(out))


def parse_medidas(txt):
    if not txt or txt.strip().lower() in ("todas", "all"):
        return list(MEDIDAS)
    pedidas = [m.strip() for m in txt.split(",") if m.strip()]
    malas = [m for m in pedidas if m not in MEDIDAS]
    if malas:
        raise SystemExit(f"Medidas desconocidas: {malas}\nValidas: {list(MEDIDAS)}")
    return pedidas


# ════════════════════════════════════════════════════════════
# Datos y configuración
# ════════════════════════════════════════════════════════════

# Supuestos que hacen que dos corridas NO sean comparables entre sí. La
# variante de aislación ya separa carpetas por sí sola; estas no, y por
# eso hay que vigilarlas.
#
# `lim_dg_modo`/`lim_dg_bus` entran por la misma razón que `lim_dg`: un tope
# agregado por alimentador y uno por barra dan la misma capacidad TOTAL pero
# distinto emplazamiento, así que ni el objetivo ni los indicadores son
# comparables entre sí.
# Dimensionamiento del MEG (defaults = flota de El Melón, 2026-10-25): una flota existente o un traslado distinto cambian
# el resultado de la medida y de toda combinación que la incluya.
SUPUESTOS = ("lim_dg", "lim_dg_modo", "lim_dg_bus", "lim_dg_conjunto", "e4_sin_fallas", "horas",
             "n_meg", "P_meg_max", "t_tra_meg", "meg_existentes")


def _lim_bus_reg(v):
    """Valor de `lim_dg_bus` tal como se registra en disco.

    `verificar_supuestos` interpreta None como "corrida anterior a que se
    anotara este supuesto" y la deja pasar. Pero None es TAMBIÉN el valor
    legítimo de "sin tope por barra", así que se escribe como "sin" para
    que no se confunda con lo desconocido y una corrida sin tope por barra
    no se mezcle con una que sí lo tiene.
    """
    return "sin" if v is None else float(v)


def _supuestos_en_disco(salida):
    """{supuestos: [rutas]} de las corridas ya guardadas."""
    vistos = {}
    for j in sorted(Path(salida).glob("*/*/esc_*.json")):
        try:
            reg = json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            continue
        if reg.get("status") != "optimal":
            continue
        vistos.setdefault(tuple(reg.get(k) for k in SUPUESTOS), []).append(j)
    return vistos


def verificar_supuestos(salida, actuales, forzar=False):
    """Impide mezclar en una carpeta corridas hechas con supuestos distintos.

    La carpeta de salida solo codifica la variante de aislación, así que dos
    barridos con distinto `lim_dg` (o distinto E4, o distinto horizonte) caen
    en el mismo sitio. Y como el barrido es reanudable, el segundo no solo
    sobrescribe: OMITE las corridas que ya existen, dejando una tabla que
    mezcla dos supuestos sin que nada lo advierta.

    Es el mismo criterio que ya aplican `base_dir` para los baselines y
    `ruta_sin_fallas` para E4: cada supuesto, su propia carpeta.
    """
    salida = Path(salida)
    actual = tuple(actuales[k] for k in SUPUESTOS)
    distintas = {k: v for k, v in _supuestos_en_disco(salida).items() if k != actual}

    # Un valor en None significa "corrida anterior a que se registrara ese
    # supuesto", no "corrida con otro supuesto". Desconocido no es lo mismo que
    # incompatible: se avisa, pero no se bloquea, porque si no cualquier
    # barrido viejo dejaría la carpeta inutilizable.
    def _solo_desconocidos(k):
        return all(v is None or v == a for v, a in zip(k, actual))

    dudosas = {k: v for k, v in distintas.items() if _solo_desconocidos(k)}
    ajenas = {k: v for k, v in distintas.items() if not _solo_desconocidos(k)}

    if dudosas:
        n = sum(len(v) for v in dudosas.values())
        _log(f"[!] {n} corridas guardadas no registran alguno de estos supuestos "
             f"({', '.join(SUPUESTOS)}): son anteriores a que se anotaran. Se "
             f"asumen compatibles con esta corrida; --forzar las rehace para "
             f"dejar la carpeta homogenea.")
    if not ajenas:
        return

    lineas = ["", "=" * 96,
              "  SUPUESTOS INCOMPATIBLES EN LA CARPETA DE SALIDA",
              "=" * 96,
              f"  {salida}", "",
              f"  Esta corrida usa : " +
              ", ".join(f"{k}={v}" for k, v in zip(SUPUESTOS, actual)),
              "  Ya hay guardadas :"]
    for k, rutas in ajenas.items():
        desc = ", ".join(f"{n}={v}" for n, v in zip(SUPUESTOS, k))
        ejemplo = rutas[0].relative_to(salida)
        lineas.append(f"    {len(rutas):>4} corridas con {desc}   (ej. {ejemplo})")
    if forzar:
        lineas += ["", "  --forzar las va a recalcular todas con los supuestos "
                       "nuevos.", "=" * 96]
        print("\n".join(lineas), flush=True)
        return

    sug = f"{salida.name}__lim{actuales['lim_dg']:g}"
    lineas += [
        "",
        "  Mezclarlas daria una tabla sin sentido: el barrido es reanudable, asi",
        "  que las que ya existen se omitirian y quedarian promediadas junto a",
        "  las nuevas. Dos salidas:",
        "",
        f"    1) Carpeta aparte (recomendado, conserva lo ya calculado):",
        f"       --salida artefactos/barrido/{sug}",
        "",
        f"    2) Rehacer todo en esta carpeta con los supuestos nuevos:",
        f"       --forzar",
        "=" * 96, ""]
    raise SystemExit("\n".join(lineas))


def verificar_lim_dg(sf, lim_dg, etiqueta_e4, horas=None,
                     lim_dg_modo="bus", lim_dg_bus=None, lim_dg_conjunto=False):
    """Falla si el E4 elegido no es compatible con esta corrida.

    Es el error que ya se coló una vez: la estrategia instalaba hasta
    1.0·P_D_base mientras el CAPEX/OPEX de DG sin fallas del objetivo venía
    de un E4 resuelto con 0.3. No da error en ninguna parte y el resultado
    queda mal en silencio, así que conviene detenerse acá.

    Se revisa lo mismo para el horizonte: un E4 de 24 h no instala nada
    (el CAPEX anualizado no se recupera en un día), así que usarlo para una
    corrida de 8760 h mete un cero donde debería ir el crédito de DG.
    """
    meta = sf.get("_meta") or {}

    # Tope compartido sol+eol o uno por tecnología: otra capacidad instalada y
    # otro CAPEX/OPEX sin fallas, así que el E4 tiene que coincidir. Un E4 sin
    # la clave es anterior a la opción, o sea por tecnología.
    if bool(meta.get("lim_dg_conjunto", False)) != bool(lim_dg_conjunto):
        raise SystemExit(
            f"\nINCONSISTENCIA: el E4 '{P.ruta_sin_fallas(etiqueta_e4).name}' se calculo con "
            f"lim_dg_conjunto={bool(meta.get('lim_dg_conjunto', False))} y esta corrida usa "
            f"lim_dg_conjunto={bool(lim_dg_conjunto)}.\n"
            f"  Generar el que corresponde: python pipeline.py --etapas E4 --lim-dg {lim_dg} "
            f"--lim-dg-modo {lim_dg_modo}"
            + (" --lim-dg-conjunto" if lim_dg_conjunto else "")
            + f" --etiqueta {LDG.etiqueta_variante(dict(lim_dg=lim_dg, lim_dg_modo=lim_dg_modo, lim_dg_bus=lim_dg_bus, lim_dg_conjunto=lim_dg_conjunto))}")

    h_e4 = meta.get("horas")
    if horas is not None and h_e4 is not None and int(h_e4) != int(horas):
        raise SystemExit(
            f"\nINCONSISTENCIA: el E4 '{P.ruta_sin_fallas(etiqueta_e4).name}' se "
            f"calculo con {h_e4} h y esta corrida usa {horas} h.\n"
            f"Un E4 de horizonte corto no alcanza a amortizar el CAPEX y deja el "
            f"credito de DG en cero.\n"
            f"  Generar el que corresponde: python pipeline.py --etapas E4 "
            f"--lim-dg {lim_dg} --etiqueta <etiqueta>")

    lim_e4 = meta.get("lim_dg")
    if lim_e4 is None:
        _log(f"[!] El artefacto de E4 '{P.ruta_sin_fallas(etiqueta_e4).name}' es "
             f"anterior al registro de lim_dg, asi que no se puede verificar que "
             f"coincida con el --lim-dg {lim_dg} de esta corrida. Si no estas "
             f"seguro, recalcularlo: python pipeline.py --etapas E4 --lim-dg "
             f"{lim_dg} --etiqueta LIM-{lim_dg:.2f}")
        return
    if abs(float(lim_e4) - float(lim_dg)) > 1e-9:
        raise SystemExit(
            f"\nINCONSISTENCIA: el E4 '{P.ruta_sin_fallas(etiqueta_e4).name}' se "
            f"calculo con lim_dg={lim_e4} y esta corrida usa --lim-dg {lim_dg}.\n"
            f"El CAPEX/OPEX de DG sin fallas del objetivo quedaria referido a otra "
            f"capacidad maxima.\n"
            f"  - para usar el E4 tal cual:  --lim-dg {lim_e4}\n"
            f"  - o generar el que falta:    python pipeline.py --etapas E4 "
            f"--lim-dg {lim_dg} --etiqueta LIM-{lim_dg:.2f}")

    # Lo mismo para la agrupación del tope: mismo lim_dg pero agregado por
    # alimentador en vez de por barra da OTRA capacidad instalada y otro
    # CAPEX/OPEX sin fallas, así que el E4 tampoco sirve.
    modo_e4 = meta.get("lim_dg_modo")
    bus_e4 = meta.get("lim_dg_bus", "__falta__")
    if modo_e4 is None:
        if str(lim_dg_modo) != "bus" or lim_dg_bus is not None:
            raise SystemExit(
                f"\nINCONSISTENCIA: el E4 '{P.ruta_sin_fallas(etiqueta_e4).name}' es "
                f"anterior al registro de lim_dg_modo, o sea que se calculo con el "
                f"tope POR BARRA, y esta corrida usa --lim-dg-modo {lim_dg_modo} "
                f"--lim-dg-bus {lim_dg_bus}.\n"
                f"  Generar el que corresponde: python pipeline.py --etapas E4 "
                f"--lim-dg {lim_dg} --lim-dg-modo {lim_dg_modo}"
                + (f" --lim-dg-bus {lim_dg_bus}" if lim_dg_bus is not None else "")
                + f" --etiqueta {LDG.etiqueta_variante(dict(lim_dg=lim_dg, lim_dg_modo=lim_dg_modo, lim_dg_bus=lim_dg_bus))}")
        return

    bus_e4 = None if bus_e4 == "__falta__" else bus_e4
    if str(modo_e4) != str(lim_dg_modo) or _lim_bus_reg(bus_e4) != _lim_bus_reg(lim_dg_bus):
        raise SystemExit(
            f"\nINCONSISTENCIA: el E4 '{P.ruta_sin_fallas(etiqueta_e4).name}' se "
            f"calculo con lim_dg_modo={modo_e4}, lim_dg_bus={bus_e4} y esta corrida "
            f"usa lim_dg_modo={lim_dg_modo}, lim_dg_bus={lim_dg_bus}.\n"
            f"Con otra agrupacion la DG se emplaza distinto y el CAPEX/OPEX sin "
            f"fallas del objetivo queda referido a otra solucion.\n"
            f"  - para usar el E4 tal cual:  --lim-dg-modo {modo_e4}"
            + (f" --lim-dg-bus {bus_e4}" if bus_e4 is not None else "") + "\n"
            f"  - o generar el que falta:    python pipeline.py --etapas E4 "
            f"--lim-dg {lim_dg} --lim-dg-modo {lim_dg_modo}"
            + (f" --lim-dg-bus {lim_dg_bus}" if lim_dg_bus is not None else "")
            + f" --etiqueta {LDG.etiqueta_variante(dict(lim_dg=lim_dg, lim_dg_modo=lim_dg_modo, lim_dg_bus=lim_dg_bus))}")


def construir_data(escenarios, variante="base", etiqueta_e4="", lim_dg=None,
                   horas=None, lim_dg_modo="bus", lim_dg_bus=None,
                   lim_dg_conjunto=False):
    """`data` completo para optimization_6, desde los artefactos del pipeline.

    Mismo contrato que arma data_loader para la app, pero sin streamlit:
    pipeline.construir_data deja fuera el baseline sin fallas (E4) y el
    ΣP_SUB por escenario (E5), que se agregan acá.

    `etiqueta_e4` elige QUÉ artefacto de E4 usar. Importa porque el CAPEX/
    OPEX de DG sin fallas depende del límite de capacidad con que se resolvió
    (ver pipeline.ruta_sin_fallas): mezclar un E4 viejo con el modelo actual
    mete una constante equivocada en el objetivo de dg_sol/dg_eol.
    """
    data, red = P.construir_data(sin_fallas=False)

    sf = P.cargar_sin_fallas(etiqueta_e4)
    if lim_dg is not None:
        verificar_lim_dg(sf, lim_dg, etiqueta_e4, horas,
                         lim_dg_modo=lim_dg_modo, lim_dg_bus=lim_dg_bus,
                         lim_dg_conjunto=lim_dg_conjunto)
    data.update({
        "P_SUB_tot_sin_fallas":    sf["P_SUB_tot_sin_fallas"],
        "CAPEX_DG_sol_sin_fallas": sf["CAPEX_DG_sol_sin_fallas"],
        "OPEX_DG_sol_sin_fallas":  sf["OPEX_DG_sol_sin_fallas"],
        "CAPEX_DG_eol_sin_fallas": sf["CAPEX_DG_eol_sin_fallas"],
        "OPEX_DG_eol_sin_fallas":  sf["OPEX_DG_eol_sin_fallas"],
        # Referencia conjunta (E4/4), para las corridas que encienden las DOS
        # tecnologias DG. Con ambas activas el optimo sin fallas no es la suma
        # de los optimos individuales, porque compiten por el mismo tope de
        # `lim_dg` por alimentador. Ver ECUACIONES_OBJETIVO_DG.md 8.2.
        # `.get` y no `[]`: un E4 anterior a esta correccion no las trae, y en
        # ese caso el objetivo cae al comportamiento historico.
        "CAPEX_DG_ambas_sin_fallas": sf.get("CAPEX_DG_ambas_sin_fallas"),
        "OPEX_DG_ambas_sin_fallas":  sf.get("OPEX_DG_ambas_sin_fallas"),
        # ΣP_SUB sin fallas CON cada tecnología. No entra al MILP: permite a
        # helpers_inversion separar `OPEX_DG_<tec>_sin_fallas` en sus dos
        # mitades (OPEX puro y crédito de subestación), que el artefacto de E4
        # guarda sumadas en un solo escalar.
        "P_SUB_sf_sol": float(sf["sin_fallas_DG_sol"]["P_SUB"].sum()),
        "P_SUB_sf_eol": float(sf["sin_fallas_DG_eol"]["P_SUB"].sum()),
        # Las mismas series HORA A HORA, para poder restringir el término de
        # subestación a las horas con falla (la fracción atribuible a
        # resiliencia). ~140 kB cada una, no entran al MILP.
        "P_SUB_sf_h":     sf["sin_fallas"]["P_SUB"].sum(axis=1),
        "P_SUB_sf_sol_h": sf["sin_fallas_DG_sol"]["P_SUB"].sum(axis=1),
        "P_SUB_sf_eol_h": sf["sin_fallas_DG_eol"]["P_SUB"].sum(axis=1),
        "faultlinMC_veg_disponible": True,
    })

    # El baseline con fallas tiene que ser el de la MISMA variante de
    # aislación con la que se van a resolver las estrategias: comparar una
    # estrategia con aislación realista contra un caso base con aislación
    # perfecta da ahorros negativos aunque la medida sirva (ver pipeline.base_dir).
    p_sub, faltan = {}, []
    for e in escenarios:
        ruta = P.base_dir(variante) / f"esc_{e:03d}.npz"
        if not ruta.exists():
            faltan.append(e)
            continue
        p_sub[e] = float(np.load(ruta)["P_SUB"].sum())
    if faltan:
        extra = ""
        if variante != "base":
            extra = " --seccionadores " + variante[4:].replace("-", ",")
        raise FileNotFoundError(
            f"Faltan baselines (variante '{variante}') para los escenarios "
            f"{faltan}. Correr:\n  python pipeline.py --etapas E5 --escenarios "
            f"{','.join(str(x) for x in faltan)}{extra}")

    data["P_SUB_tot_por_esc"] = p_sub
    data["P_SUB_tot"] = p_sub[escenarios[0]]
    return data, red


def parse_enlaces_nuevos(txt):
    """'5-10:1.0,13-3:2.0' -> [(5, 10, 1.0), (13, 3, 2.0)] (barra, barra, km)."""
    out = []
    for tok in (txt or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        par, _, km = tok.partition(":")
        a, b = (int(x) for x in par.split("-"))
        out.append((a, b, float(km) if km else 1.0))
    return out


def agregar_enlaces_nuevos(data, red, enlaces):
    """Suma líneas CANDIDATAS (tie-lines por construir) a la red, solo en memoria.

    Los artefactos del pipeline no se tocan: las líneas nuevas se agregan a
    `net`, a `LINES`/`longitud_km_linea` y a las matrices de falla (columnas en
    cero: el enlace nuevo es inmune, como en reconfig_topo). Quedan en
    `tie_indices` (flujo por P_tie, nunca "siempre cerradas") y en
    `red["tie_candidatas_nuevas"]`, que construir_config pasa como
    `tie_candidatas` para que `inv_tie` decida construirlas. Los enlaces
    existentes siguen siendo los de `reconfig_topo`.
    """
    import copy
    net = copy.deepcopy(data["net"])
    base = net.line.index.max() + 1
    plantilla = net.line.loc[red["tie_indices"][0]]
    nuevas = []
    for k, (a, b, km) in enumerate(enlaces):
        for bus in (a, b):
            if bus not in net.bus.index:
                raise ValueError(f"enlace nuevo {a}-{b}: la barra {bus} no existe")
        l = int(base + k)
        fila = plantilla.copy()
        fila["name"] = f"Line {a}-{b} (nueva)"
        fila["from_bus"], fila["to_bus"], fila["length_km"] = a, b, km
        if "zone" in fila.index:
            fila["zone"] = red["bus_zone"].get(a, fila["zone"])
        net.line.loc[l] = fila
        nuevas.append(l)

    for clave in ("faultlinMC", "repairMC", "faultlinMC_veg", "faultlinMC_ref", "faultlinMC_ug"):
        m = data.get(clave)
        if m is not None:
            pad = np.zeros(m.shape[:-1] + (len(nuevas),), dtype=m.dtype)
            data[clave] = np.concatenate([m, pad], axis=-1)

    km_lin = dict(data["longitud_km_linea"])
    km_lin.update({l: e[2] for l, e in zip(nuevas, enlaces)})
    red = dict(red)
    red.update({
        "net": net,
        "LINES": list(red["LINES"]) + nuevas,
        "tie_indices": list(red["tie_indices"]) + nuevas,
        "longitud_km_linea": km_lin,
        "tie_candidatas_nuevas": nuevas,
    })
    data.update({"net": net, "longitud_km_linea": km_lin, "tie_candidatas": nuevas})
    for l, (a, b, km) in zip(nuevas, enlaces):
        _log(f"enlace nuevo candidato: línea {l} = barras {a}-{b}, {km:g} km")
    return data, red


def construir_config(red, args, esc, medidas_on, usar_aisl, sec):
    """Config de optimization_6 con SOLO `medidas_on` encendidas."""
    cfg = P.config_medida(red, args.horas, esc, lim_dg=args.lim_dg,
                          lim_dg_modo=args.lim_dg_modo,
                          lim_dg_bus=args.lim_dg_bus,
                          lim_dg_conjunto=args.lim_dg_conjunto,
                          **{m: True for m in medidas_on})

    # config_medida deja las unidades móviles dimensionadas en cero (todas
    # las medidas apagadas). Si MEG/MESS están encendidos hay que darles
    # tamaño; los defaults replican los de la UI de la app.
    con_switch = bool({"inv_switch_rcs", "inv_switch_ms"} & set(medidas_on))
    e_cap_mess = args.p_mess_max * args.h_mess
    cfg.update({
        "n_meg": args.n_meg,
        "P_meg_max": args.p_meg_max,
        "T_tra_meg": {b: args.t_tra_meg for b in red["BUSES"]},
        "meg_existentes": args.meg_existentes,
        "n_meg_arr": args.n_meg_arr,
        "P_meg_arr": args.p_meg_arr,
        "h_meg_arr": args.h_meg_arr,
        "T_tra_meg_arr": {b: (args.t_tra_meg if args.t_tra_meg_arr is None else args.t_tra_meg_arr)
                          for b in red["BUSES"]},
        "n_mess": args.n_mess,
        "h_mess": args.h_mess,
        "P_mess_max": args.p_mess_max,
        "E_cap_mess": e_cap_mess,
        "E_cap_mess_total": args.n_mess * e_cap_mess,
        "T_tra_mess": args.t_tra_mess,
        "T_maniobra_MS": args.t_maniobra_ms,
        "T_maniobra_tie": args.t_maniobra_tie,
        "radial_tie": bool(args.radial_tie),
        "gurobi_mem_gb": args.gurobi_mem_gb,   # None = 60 % de la RAM (default del modelo)
        "microredes": bool(args.microredes) and con_switch,
        "aislacion_fallas_fija": usar_aisl,
        "seccionadores_existentes": sec,
        "mip_gap": args.mip_gap,
        "gurobi_time_limit": args.time_limit,
        # Con guion bajo: no es un parámetro del modelo (el optimizador solo
        # lee claves conocidas), viaja acá solo para quedar registrado en el
        # JSON de cada corrida y poder rastrear con qué E4 se calculó.
        "_e4": args.sin_fallas or "historico",
    })
    if red.get("tie_candidatas_nuevas"):
        # inv_tie decide solo sobre los enlaces nuevos; los existentes son de reconfig_topo.
        cfg["tie_candidatas"] = list(red["tie_candidatas_nuevas"])
        cfg["tie_expuesta_a_falla"] = False
    # Solo se pisan si el usuario los pidió: en None, el optimizador los
    # calcula según los núcleos y la RAM de la máquina.
    if args.threads is not None:
        cfg["gurobi_threads"] = args.threads
    if args.mem_gb is not None:
        cfg["gurobi_mem_gb"] = args.mem_gb
    if args.method is not None:
        cfg["gurobi_method"] = args.method
    if args.presolve is not None:
        cfg["gurobi_presolve"] = args.presolve
    if args.bar_homogeneous is not None:
        cfg["gurobi_bar_homogeneous"] = args.bar_homogeneous
    if args.mipfocus is not None:
        cfg["gurobi_mipfocus"] = args.mipfocus
    return cfg


# ════════════════════════════════════════════════════════════
# Indicadores y persistencia de una corrida
# ════════════════════════════════════════════════════════════

def indicadores_de(res, red, data):
    """SAIDI/SAIFI/ENS/ASAI + objetivo de UNA corrida (lista de dicts-hora)."""
    BUSES, P_D_base = red["BUSES"], red["P_D_base"]
    horas_idx = range(1, len(res) + 1)      # fn_* exige len(res) == len(horas)

    cli = red.get("clientes_bus")           # None: regla de clientes por kW
    cli_total = P.clientes_denominador_saifi(red, ART_DIR)
    saidi = fn_SAIDI(res, P_D_base, BUSES, horas_idx, clientes=cli)
    saifi = fn_SAIFI(res, P_D_base, BUSES, horas_idx, clientes=cli,
                     clientes_denominador=cli_total)
    ens = fn_ENS_total(res, BUSES, horas_idx)
    # ASAI sobre el horizonte realmente simulado, no sobre 8760 fijo: con
    # --horas 24 el 8760 del default daría un ASAI de 0.9999 sin sentido.
    asai = fn_ASAI(saidi, len(res))

    VoLL, N_ev = data["VoLL"], data["N_eventos"]
    costo_ens = sum(VoLL.get(b, 1.0) * h.get("P_LC", {}).get(b, 0.0)
                    for h in res for b in BUSES) * N_ev

    return {
        "objetivo": float(res[0].get("fn_objetivo", float("nan"))),
        "ens": float(ens),
        "saidi": float(saidi),
        "saifi": float(saifi),
        "asai": float(asai),
        "costo_ens": float(costo_ens),
        "horas": len(res),
    }


def plan_de(res):
    """Cartera de inversión elegida por el solver (solo valores no nulos)."""
    h0 = res[0]
    plan = {}
    for clave in CLAVES_PLAN:
        d = h0.get(clave, {}) or {}
        nz = {str(k): float(v) for k, v in d.items() if abs(float(v)) > 1e-6}
        if nz:
            plan[clave] = nz
    return plan


def _suma(d):
    """Suma los valores de un dict del resultado; 0.0 si la clave no existe."""
    return float(sum(float(v) for v in (d or {}).values()))


def _por_bus(res, clave, BUSES):
    """Energía acumulada por barra de una magnitud horaria.

    Acepta las dos formas de indexación que usa el optimizador: {bus: valor}
    (DG, EDRP) y {(unidad, bus): valor} (MEG y MESS, que están indexados por
    unidad móvil además de por barra).
    """
    idx = {b: i for i, b in enumerate(BUSES)}
    out = np.zeros(len(BUSES), dtype=np.float64)
    for h in res:
        for k, v in (h.get(clave, {}) or {}).items():
            b = k[1] if isinstance(k, tuple) and len(k) == 2 else k
            if b in idx:
                out[idx[b]] += float(v)
    return out


def _json_seguro(obj):
    """Convierte tipos de numpy a tipos de Python para poder serializar."""
    if isinstance(obj, dict):
        return {str(k): _json_seguro(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_seguro(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


_BASE_CACHE = {}


def _baseline(esc, variante):
    """Baseline rehidratado, cacheado: varias medidas comparten escenario y
    rehidratar 8760 horas en cada corrida sería tiempo regalado."""
    clave = (int(esc), variante)
    if clave not in _BASE_CACHE:
        _BASE_CACHE[clave] = P.cargar_baseline(esc, variante)
    return _BASE_CACHE[clave]


_SF_CACHE = {}


def _data_sin_fallas(data):
    """`data` y `red` sin fallas (faultlinMC = 0), cargados una sola vez."""
    if "data" not in _SF_CACHE:
        _SF_CACHE["data"], _SF_CACHE["red"] = P.construir_data(sin_fallas=True)
    d = dict(_SF_CACHE["data"])
    d["P_SUB_tot_sin_fallas"] = data["P_SUB_tot_sin_fallas"]
    return d, _SF_CACHE["red"]


def referencia_dg_plan(res, cfg, data):
    """Referencia sin fallas evaluada con la DG que eligió ESTA corrida.

    El E4 es el óptimo sin fallas con DG libre: otra cantidad y otra ubicación
    que las de la corrida con fallas. Restarlo línea por línea (CAPEX contra
    CAPEX) da CAPEX negativo cada vez que el plan instala menos DG que el E4,
    aunque el total no lo sea. Acá se resuelve el caso sin fallas con la DG del
    plan fija por nodo y por tecnología, y se entrega un `data` con las mismas
    claves que el E4 (`CAPEX/OPEX_DG_<tec>_sin_fallas`, `P_SUB_sf_<tec>[_h]`)
    pero referidas a ese plan. Con eso helpers_inversion no cambia: el CAPEX y
    el OPEX de la DG se cancelan exactamente y el término de subestación queda
    como lo que cuestan las fallas.

    No toca las decisiones del MILP: el objetivo restaba una constante, y una
    constante no cambia el óptimo. Solo cambia la contabilidad del reporte.

    Devuelve (data_referido_al_plan, sobrecosto_dg). `sobrecosto_dg` es el
    costo sin fallas del plan menos el del E4: cuánto cuesta en un año normal
    haber decidido la DG con fallas. Debería ser >= 0 porque el E4 es el óptimo
    con el mismo tope; si sale claramente negativo, algo no calza.
    """
    from optimization_base import resolver_optimizacion_base

    tecs = [t for t in ("sol", "eol") if cfg.get(f"dg_{t}")]
    hora0 = res[0]
    cap = {t: {g: float(v) for g, v in (hora0.get(f"CAP_DG_{t}_kW") or {}).items()}
           for t in tecs}

    d_sf, red = _data_sin_fallas(data)
    cfg_sf = P.config_medida(
        red, len(res), 0,
        lim_dg=cfg.get("lim_dg", P.LIM_DG_DEFAULT),
        lim_dg_modo=cfg.get("lim_dg_modo", P.LIM_DG_MODO_DEFAULT),
        lim_dg_bus=cfg.get("lim_dg_bus"),
        lim_dg_conjunto=cfg.get("lim_dg_conjunto", False),
        **{f"dg_{t}": True for t in tecs})
    cfg_sf["cap_dg_fija"] = cap
    r = resolver_optimizacion_base(data=d_sf, config=cfg_sf)
    res_sf = [h for h in (r or {}).get(0, []) if h]
    if not res_sf:
        raise RuntimeError("el caso sin fallas con la DG del plan no tiene solución")

    C_sub0 = data["C_sub"].get(0, 1.0)
    p_sf0 = float(data["P_SUB_tot_sin_fallas"])
    p_h = np.array([sum(h.get("P_SUB", {}).values()) for h in res_sf], dtype=np.float64)
    p_tot = float(p_h.sum())
    credito = -(p_sf0 - p_tot) * C_sub0

    GENS = list(data["net"].bus.index)
    frp = data["frp"]
    ref = dict(data)
    costo_plan = credito
    for i, t in enumerate(tecs):
        C_inv, C_op = data[f"DG_{t}"][0], data[f"DG_{t}"][1]
        capex = sum(cap[t].get(g, 0.0) * C_inv[g] * frp for g in GENS)
        opex = sum(cap[t].get(g, 0.0) * C_op[g] for g in GENS)
        costo_plan += capex + opex
        # El crédito de subestación es UNO por corrida, no uno por tecnología:
        # se asigna entero a la primera y la segunda queda con crédito cero
        # (P_SUB_sf igual al del caso sin DG), para no contarlo dos veces.
        principal = i == 0
        ref[f"CAPEX_DG_{t}_sin_fallas"] = capex
        ref[f"OPEX_DG_{t}_sin_fallas"] = opex + (credito if principal else 0.0)
        ref[f"P_SUB_sf_{t}"] = p_tot if principal else p_sf0
        ref[f"P_SUB_sf_{t}_h"] = p_h if principal else np.asarray(data["P_SUB_sf_h"])

    if len(tecs) == 2 and data.get("CAPEX_DG_ambas_sin_fallas") is not None:
        costo_e4 = data["CAPEX_DG_ambas_sin_fallas"] + data["OPEX_DG_ambas_sin_fallas"]
    else:
        costo_e4 = sum(data[f"CAPEX_DG_{t}_sin_fallas"] + data[f"OPEX_DG_{t}_sin_fallas"]
                       for t in tecs)
    return ref, costo_plan - costo_e4


def costos_de(res, esc, cfg, data, variante):
    """Descomposición económica del objetivo, con la MISMA función que la app.

    Devuelve capex_anual, opex_anual, costo_ens_base, costo_ens_estr,
    ahorro_ens, beneficio y costo_total. Sin esto no se puede comparar
    costo-efectividad entre medidas sin abrir cada .pkl.gz, y re-derivar los
    costos acá garantizaría que el barrido y la app terminen dando cifras
    distintas para lo mismo.

    Import perezoso y a prueba de fallos por la misma razón que `resumen_red`:
    arrastra data_loader, y es un extra de análisis, no el resultado.
    """
    try:
        from helpers_inversion import _extraer_costos_objetivo
        # El término de subestación del helper compara contra `P_SUB_tot`, un
        # único escalar: el helper recorre los escenarios con `enumerate`, así
        # que pierde el índice y no tiene con qué elegir el baseline del que
        # está costeando. Se le pasa acá el del escenario que se resolvió, que
        # es EXACTAMENTE el que usó el objetivo del MILP
        # (optimization_6_oficial:378). Sin esto, las diez corridas de una
        # medida se costean todas contra el baseline del primer escenario del
        # barrido: son ±0,03% de diferencia, pero multiplicados por los ~156 GWh
        # del baseline dan hasta 4.500 USD/año, que en una medida barata como
        # reconfiguración (8.033 USD/año) es más de la mitad del costo.
        # Con DG renovable, la referencia sin fallas es la del MISMO plan y no
        # la del E4 (ver `referencia_dg_plan`). Si no se puede resolver, se
        # cae a la del E4 y queda avisado: sobrecosto_dg vacío lo delata.
        base_data, sobrecosto = data, None
        if cfg.get("dg_sol") or cfg.get("dg_eol"):
            try:
                base_data, sobrecosto = referencia_dg_plan(res, cfg, data)
            except Exception as e:              # noqa: BLE001
                _log(f"  [!] sin referencia sin fallas del plan DG "
                     f"({type(e).__name__}: {e}); se usa la del E4")
        d_esc = dict(base_data)  # copia: `data` lo comparten todas las corridas
        p_esc = (data.get("P_SUB_tot_por_esc") or {}).get(esc)
        if p_esc is not None:
            d_esc["P_SUB_tot"] = p_esc
        out = _extraer_costos_objetivo({esc: res}, {esc: _baseline(esc, variante)},
                                       cfg, d_esc)
        # Devuelve un dict cuando hay un solo escenario y una lista cuando hay
        # varios; acá siempre es uno, pero se aceptan las dos formas.
        if isinstance(out, (list, tuple)):
            out = out[0] if out else None
        if not out:
            return None
        d = _json_seguro(out)
        # Lo que uno esperaría que fuera "beneficio": el ahorro de ENS menos lo
        # que costó conseguirlo. Se calcula acá porque el helper no lo entrega.
        d["ahorro_neto"] = float(d.get("ahorro_ens", 0.0)) - float(d.get("costo_total", 0.0))
        if sobrecosto is not None:
            d["sobrecosto_dg"] = float(sobrecosto)
        return d
    except Exception as e:                      # noqa: BLE001
        _log(f"  [!] sin descomposicion de costos ({type(e).__name__}: {e})")
        return None


def resumen_red(res):
    """El `resumen` de grafo.resumir_simulacion, guardado con la corrida.

    Permite que la pestaña del barrido dibuje EXACTAMENTE la misma figura de
    resumen de red que la pestaña Resultados, en vez de reimplementarla y que
    las dos se desincronicen. Se calcula acá, donde todavía existe la lista de
    dicts por hora; el resumen pesa unos pocos KB, frente a los ~100 MB que
    ocuparía guardar las 8760 horas completas.

    Import perezoso porque `grafo` arrastra data_loader y streamlit, que no
    hacen falta para el resto del barrido. Si falla, la corrida se guarda
    igual: es un extra para visualizar, no un resultado.
    """
    try:
        from grafo import resumir_simulacion
        return _json_seguro(resumir_simulacion(res))
    except Exception as e:                      # noqa: BLE001
        _log(f"  [!] sin resumen de red ({type(e).__name__}: {e})")
        return None


def _matriz_por_elemento(res, clave, dtype=np.int8):
    """(horas × elementos) de una variable indexada por línea o por barra.

    La lista de elementos se lee de las claves del propio resultado, no de un
    orden fijo, para que no se desalinee si cambia el conjunto de candidatas.
    Devuelve (None, []) si la variable no existe en esta corrida.
    """
    elems = sorted((res[0].get(clave, {}) or {}).keys())
    if not elems:
        return None, []
    M = np.zeros((len(res), len(elems)), dtype=dtype)
    for t, h in enumerate(res):
        d = h.get(clave, {}) or {}
        for j, e in enumerate(elems):
            M[t, j] = int(round(float(d.get(e, 0))))
    return M, elems


def _ubicacion_movil(res, clave):
    """(horas × unidades) con la BARRA donde está cada unidad móvil, -1 si no
    está desplegada esa hora.

    `u_mess` y `x_re` vienen indexados por (unidad, barra) con una binaria por
    combinación; guardar esa matriz completa por hora sería 8760×k×b casi todo
    ceros, así que se colapsa a "en qué barra está", que es la pregunta que uno
    le hace después.
    """
    llaves = [k for k in (res[0].get(clave, {}) or {}) if isinstance(k, tuple)]
    if not llaves:
        return None, []
    unidades = sorted({k[0] for k in llaves})
    pos = {u: i for i, u in enumerate(unidades)}
    M = np.full((len(res), len(unidades)), -1, dtype=np.int16)
    for t, h in enumerate(res):
        for (u, b), v in (h.get(clave, {}) or {}).items():
            if float(v) > 0.5:
                M[t, pos[u]] = int(b)
    return M, unidades


def ubicaciones_de(res, red):
    """Resumen legible de DÓNDE actuó cada medida.

    Complementa a `plan`, que ya dice qué líneas reciben inversión fija
    (switches, refuerzo, poda, soterramiento, enlaces nuevos). Esto cubre lo
    que varía en el tiempo: qué enlace se cierra, qué switch se maniobra y por
    dónde pasan las unidades móviles.
    """
    out = {}

    def _conteo(clave, criterio):
        M, elems = _matriz_por_elemento(res, clave)
        if M is None:
            return {}
        return {str(e): int(criterio(M[:, j])) for j, e in enumerate(elems)
                if criterio(M[:, j]) > 0}

    # Enlaces y switches: horas en servicio y número de maniobras por línea.
    out["tie_horas_cerrada"] = _conteo("z_tie", lambda c: c.sum())
    out["tie_maniobras"] = _conteo("w_tie", lambda c: c.sum())
    out["sw_maniobras"] = _conteo("w_sw", lambda c: c.sum())
    # z_sw = 1 cuando la candidata está EN servicio, así que lo informativo es
    # cuántas horas estuvo abierta.
    M_sw, elems_sw = _matriz_por_elemento(res, "z_sw")
    if M_sw is not None:
        out["sw_horas_abierta"] = {str(e): int((M_sw[:, j] == 0).sum())
                                   for j, e in enumerate(elems_sw)
                                   if (M_sw[:, j] == 0).any()}

    # Unidades móviles: a qué barra fue cada una y cuántas horas estuvo ahí.
    for clave, nombre in (("x_re", "meg"), ("u_mess", "mess")):
        M, unidades = _ubicacion_movil(res, clave)
        if M is None:
            continue
        por_unidad, por_bus = {}, {}
        for j, u in enumerate(unidades):
            buses, cuentas = np.unique(M[:, j][M[:, j] >= 0], return_counts=True)
            if len(buses) == 0:
                continue
            por_unidad[str(u)] = {str(int(b)): int(c) for b, c in zip(buses, cuentas)}
            for b, c in zip(buses, cuentas):
                por_bus[str(int(b))] = por_bus.get(str(int(b)), 0) + int(c)
        if por_unidad:
            out[f"{nombre}_horas_por_unidad_y_bus"] = por_unidad
            out[f"{nombre}_horas_por_bus"] = por_bus

    # Barras que operaron como fuente de isla.
    out["isla_horas_por_bus"] = _conteo("z_fuente", lambda c: c.sum())

    return {k: v for k, v in out.items() if v}


def _detalle_por_elemento(res):
    """Matrices (horas × elemento) de las variables de operación, con la lista
    de ids del elemento al lado.

    Son binarias, así que van en int8 y comprimen muchísimo (casi todo ceros).
    Sin esto solo se sabe CUÁNTOS enlaces se cerraron en una hora, no cuáles.
    """
    out = {}
    for clave, nombre in (("z_tie", "tie_cerrada"), ("w_tie", "tie_maniobra"),
                          ("z_sw", "sw_en_servicio"), ("w_sw", "sw_maniobra"),
                          ("z_fuente", "isla_fuente")):
        M, elems = _matriz_por_elemento(res, clave)
        if M is not None:
            out[f"det_{nombre}"] = M
            out[f"det_{nombre}_ids"] = np.array(elems, dtype=np.int32)
    for clave, nombre in (("x_re", "meg_bus"), ("u_mess", "mess_bus")):
        M, unidades = _ubicacion_movil(res, clave)
        if M is not None:
            out[f"det_{nombre}"] = M
            out[f"det_{nombre}_ids"] = np.array(unidades, dtype=np.int32)
    return out


def series_de(res, red):
    """Series horarias de la corrida: el formato slim de E5 más las magnitudes
    por tecnología que la app grafica.

    Se guardan TOTALES del sistema por hora (1D) y no el detalle por barra de
    cada tecnología: eso último serían ~1 MB por arreglo y por corrida, y con
    170 corridas no se justifica. El detalle espacial se conserva como energía
    acumulada por barra (`E_*_bus`), que es lo que se usa para mapas y para
    saber dónde terminó instalándose cada recurso.
    """
    BUSES = red["BUSES"]

    def serie(clave):
        return np.array([_suma(h.get(clave, {})) for h in res], dtype=np.float64)

    return {
        # ── mismo contenido que el baseline de E5 ────────────────────────
        "P_LC": np.array([[h.get("P_LC", {}).get(b, 0.0) for b in BUSES] for h in res],
                         dtype=np.float64),
        "P_SUB": np.array([[h.get("P_SUB", {}).get(s, 0.0) for s in P.S] for h in res],
                          dtype=np.float64),
        "dano": np.array([h.get("n°lineas_danadas", 0) for h in res], dtype=np.int16),
        "fuera": np.array([h.get("n°lineas_fuera_servicio", h.get("n°lineas_danadas", 0))
                           for h in res], dtype=np.int16),
        # ── generación ──────────────────────────────────────────────────
        "P_DG_sol": serie("P_DG_sol"),
        "P_DG_eol": serie("P_DG_eol"),
        "P_DG_fos": serie("P_DG_fos"),
        "P_MEG": serie("P_MEG"),
        "P_MEG_arr": serie("P_MEG_arr"),
        # ── almacenamiento: P_ESS = descarga - carga, o sea POSITIVO es
        #    inyección a la red y negativo es consumo (optimization_6:1524)
        "P_ESS_4h": serie("P_ESS_4h"),
        "P_ESS_5h": serie("P_ESS_5h"),
        "P_ESS_6h": serie("P_ESS_6h"),
        "SOC_4h": serie("SOC_MWh_4h"),
        "SOC_5h": serie("SOC_MWh_5h"),
        "SOC_6h": serie("SOC_MWh_6h"),
        "P_MESS": serie("P_mess"),
        "SOC_MESS": serie("SOC_mess"),
        # ── demanda y operación ─────────────────────────────────────────
        "EDRP": serie("EDRP_total"),
        "carga_servida": np.array([float(h.get("carga_servida_MW", 0.0)) for h in res],
                                  dtype=np.float64),
        "tie_cerradas": serie("z_tie"),
        "maniobras_tie": serie("w_tie"),
        "sw_en_servicio": serie("z_sw"),
        "maniobras_sw": serie("w_sw"),
        "buses_isla": serie("z_fuente"),
        # ── energía acumulada por barra (detalle espacial) ──────────────
        "BUSES": np.array(BUSES, dtype=np.int32),
        # ── quién y cuándo (detalle por elemento) ───────────────────────
        # Matrices (horas × elemento) con su lista de ids al lado, para poder
        # responder "qué enlace se cerró en la hora 4312" y no solo "cuántos".
        **_detalle_por_elemento(res),
        "E_DG_sol_bus": _por_bus(res, "P_DG_sol", BUSES),
        "E_DG_eol_bus": _por_bus(res, "P_DG_eol", BUSES),
        "E_DG_fos_bus": _por_bus(res, "P_DG_fos", BUSES),
        "E_MEG_bus": _por_bus(res, "P_MEG", BUSES),
        "E_MEG_arr_bus": _por_bus(res, "P_MEG_arr", BUSES),
        "E_MESS_bus": _por_bus(res, "P_mess", BUSES),
        "E_EDRP_bus": _por_bus(res, "EDRP_total", BUSES),
        "E_LC_bus": _por_bus(res, "P_LC", BUSES),
    }


def agregados_de(res, red, S):
    """Capacidad instalada, equipos, energía por tecnología y operación.

    Las capacidades y las binarias de inversión son constantes en el tiempo,
    así que se leen de la hora 0; las energías se integran de las series.
    """
    h0 = res[0]
    km = red.get("longitud_km_linea", {})

    def n_bin(clave):
        return int(sum(1 for v in (h0.get(clave, {}) or {}).values() if float(v) > 0.5))

    def km_bin(clave):
        return float(sum(km.get(l, 0.0)
                         for l, v in (h0.get(clave, {}) or {}).items() if float(v) > 0.5))

    pos = lambda a: float(np.maximum(a, 0.0).sum())     # noqa: E731
    neg = lambda a: float(-np.minimum(a, 0.0).sum())    # noqa: E731
    ess = S["P_ESS_4h"] + S["P_ESS_5h"] + S["P_ESS_6h"]

    return {
        "cap_dg_sol_kW": _suma(h0.get("CAP_DG_sol_kW")),
        "cap_dg_eol_kW": _suma(h0.get("CAP_DG_eol_kW")),
        "cap_dg_fos_kW": _suma(h0.get("CAP_DG_fos_kW")),
        "cap_ess_4h_kWh": _suma(h0.get("CAP_ESS_kWh_4h")),
        "cap_ess_5h_kWh": _suma(h0.get("CAP_ESS_kWh_5h")),
        "cap_ess_6h_kWh": _suma(h0.get("CAP_ESS_kWh_6h")),
        "cap_meg_MW": _suma(h0.get("CAP_MEG_MW")),
        "cap_mess_MWh": _suma(h0.get("CAP_MESS_MWh")),
        "n_rcs": n_bin("y_rcs"),
        "n_ms": n_bin("y_ms"),
        "n_tie_nuevas": n_bin("y_tie"),
        "n_mess": n_bin("y_mess"),
        "n_veg": n_bin("x_veg"),
        "n_ref": n_bin("x_ref"),
        "n_ug": n_bin("x_ug"),
        "km_veg": km_bin("x_veg"),
        "km_ref": km_bin("x_ref"),
        "km_ug": km_bin("x_ug"),
        "E_dg_sol": float(S["P_DG_sol"].sum()),
        "E_dg_eol": float(S["P_DG_eol"].sum()),
        "E_dg_fos": float(S["P_DG_fos"].sum()),
        "E_meg": float(S["P_MEG"].sum()),
        "E_meg_arr": float(S["P_MEG_arr"].sum()),
        "n_arriendos_meg": len(h0.get("arriendos_MEG", []) or []),
        "E_ess_descarga": pos(ess),
        "E_ess_carga": neg(ess),
        "E_mess_descarga": pos(S["P_MESS"]),
        "E_mess_carga": neg(S["P_MESS"]),
        "E_edrp": float(S["EDRP"].sum()),
        "E_sub": float(S["P_SUB"].sum()),
        "E_servida": float(S["carga_servida"].sum()),
        "maniobras_tie": float(S["maniobras_tie"].sum()),
        "maniobras_sw": float(S["maniobras_sw"].sum()),
        "horas_isla": int((S["buses_isla"] > 0).sum()),
    }


def guardar_completo(res, esc, cfg, etiqueta, ind, destino, firma):
    """Escribe la corrida COMPLETA en el .pkl.gz que lee la app.

    El slim .npz sirve para las vistas que cruzan las 120 corridas; este
    archivo sirve para abrir UNA corrida en la pestaña Resultados con toda la
    maquinaria que ya existe. Pesa ~4.5 MB contra 33 KB del slim, así que son
    complementarios, no alternativos.

    Se usa la convención de la app: `results` re-indexado a 0 (el `_norm` que
    hace app_v7 en modo determinista), y el escenario real en la etiqueta.
    """
    historial = {
        f"{etiqueta} · esc {esc}": {
            "results": {0: res},
            "saidi": {1: round(float(ind["saidi"]), 2)},
            "config": cfg,
            "tiempo": ind.get("segundos", 0),
        }
    }
    P.escribir_historial(destino, historial, firma)
    return destino


def resolver_y_guardar(resolver, data, red, cfg, esc, destino_json, destino_npz,
                       etiqueta, forzar=False, completo=False, firma=None):
    """Una corrida. Devuelve el dict de indicadores (o None si no hubo solución).

    Reanudable: si el .json ya existe se lee de disco y no se re-resuelve.
    Las corridas SIN solución también se registran, para no reintentarlas en
    cada pasada; --forzar las vuelve a intentar.
    """
    if destino_json.exists() and not forzar:
        reg = json.loads(destino_json.read_text(encoding="utf-8"))
        estado = "OK" if reg.get("status") == "optimal" else "SIN SOLUCION"
        _log(f"  {etiqueta} esc {esc}: ya existe ({estado}), se omite")
        return reg if reg.get("status") == "optimal" else None

    t0 = time.time()
    try:
        res_dict = resolver(data=data, config=cfg)
    except Exception as e:            # noqa: BLE001 — un fallo no debe botar el barrido
        import traceback
        traceback.print_exc()
        res_dict = {}
        _log(f"  {etiqueta} esc {esc}: EXCEPCION {type(e).__name__}: {e}")

    dt = time.time() - t0
    res = [h for h in (res_dict or {}).get(esc, []) if h]

    destino_json.parent.mkdir(parents=True, exist_ok=True)
    if not res:
        destino_json.write_text(json.dumps(
            {"medida": etiqueta, "escenario": esc, "status": "sin_solucion",
             "segundos": round(dt, 1)}, indent=2, ensure_ascii=False), encoding="utf-8")
        _log(f"  {etiqueta} esc {esc}: SIN SOLUCION ({dt:.0f} s)")
        return None

    ind = indicadores_de(res, red, data)
    series = series_de(res, red)
    agg = agregados_de(res, red, series)
    # Cómo terminó el solver, tal como lo reporta (optimization_6_oficial lo
    # propaga en la hora 0). Antes acá se escribía "optimal" fijo por el solo
    # hecho de haber resultado, así que una corrida cortada por TimeLimit —con
    # un incumbente que puede ser pésimo— quedaba registrada como óptima y
    # entraba a los promedios sin que nada avisara.
    #
    # `status` se mantiene como "hay solución utilizable" y NO se degrada por
    # un corte de tiempo: es la clave con la que este script decide si una
    # corrida ya está hecha y si entra al resumen. Degradarla haría que la
    # corrida desapareciera del CSV en silencio, que es peor que verla
    # marcada. Quien analiza mira `solver_term` y `solver_gap`.
    term = str(res[0].get("solver_term", "desconocido"))
    gap = res[0].get("solver_gap")
    reg = {
        "medida": etiqueta,
        "escenario": esc,
        "status": "optimal",
        "solver_term": term,
        "solver_gap": float(gap) if gap is not None else None,
        "limite_tiempo": cfg.get("gurobi_time_limit"),
        "segundos": round(dt, 1),
        "medidas_on": sorted(m for m in MEDIDAS if cfg.get(m)),
        "microredes": bool(cfg.get("microredes")),
        "variante_aislacion": _variante(cfg.get("aislacion_fallas_fija", False),
                                        cfg.get("seccionadores_existentes", [])),
        "e4_sin_fallas": cfg.get("_e4", "historico"),
        "lim_dg": cfg.get("lim_dg"),
        "lim_dg_modo": cfg.get("lim_dg_modo"),
        "lim_dg_bus": _lim_bus_reg(cfg.get("lim_dg_bus")),
        "lim_dg_conjunto": bool(cfg.get("lim_dg_conjunto", False)),
        "n_meg": cfg.get("n_meg"),
        "P_meg_max": cfg.get("P_meg_max"),
        "t_tra_meg": next(iter((cfg.get("T_tra_meg") or {}).values()), None),
        "meg_existentes": cfg.get("meg_existentes", 0),
        "voll": next(iter(set(data["VoLL"].values())), None) if len(set(data["VoLL"].values())) == 1 else None,
        "mip_gap": cfg.get("mip_gap"),
        "mipfocus": cfg.get("gurobi_mipfocus", 1),
        "t_maniobra_tie": cfg.get("T_maniobra_tie", 0),
        "radial_tie": bool(cfg.get("radial_tie", False)),
        **ind,
        **agg,
        "plan": plan_de(res),
        # Dónde actuó cada medida: complementa a `plan`, que solo cubre la
        # inversión fija (qué líneas se refuerzan, dónde van los switches).
        "ubicacion": ubicaciones_de(res, red),
        "resumen_red": resumen_red(res),
        **{f"cst_{k}": v for k, v in (costos_de(
            res, esc, cfg,
            data, _variante(cfg.get("aislacion_fallas_fija", False),
                            cfg.get("seccionadores_existentes", []))) or {}
           ).items() if k in COSTOS or k == "ahorro_neto"},
    }
    reg["cst_objetivo_plan"] = _objetivo_plan(reg)
    destino_json.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(destino_npz, **series)

    extra = ""
    if completo:
        try:
            ruta_gz = destino_json.with_suffix(".pkl.gz")
            guardar_completo(res, esc, cfg, etiqueta, {**ind, "segundos": round(dt, 1)},
                             ruta_gz, firma)
            reg["archivo_completo"] = ruta_gz.name
            destino_json.write_text(json.dumps(reg, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
            extra = f"  +{ruta_gz.stat().st_size/1e6:.1f} MB completo"
        except Exception as e:                  # noqa: BLE001 — no debe botar el barrido
            _log(f"  [!] no se pudo guardar el .pkl.gz ({type(e).__name__}: {e})")

    _log(f"  {etiqueta} esc {esc}: obj={ind['objetivo']:,.0f}  ENS={ind['ens']:,.1f}  "
         f"SAIDI={ind['saidi']:.3f}  ({dt:.0f} s){extra}")
    return reg


# ════════════════════════════════════════════════════════════
# FASE 1 — cada medida por separado, escenario por escenario
# ════════════════════════════════════════════════════════════

def fase1(args, data, red, medidas, escenarios, salida, usar_aisl, sec, firma=None):
    from optimization_6_oficial import resolver_optimizacion

    filas = []
    total = len(medidas) * len(escenarios)
    hecho = 0
    t_ini = time.time()

    for m in medidas:
        _log(f"--- medida '{m}' ({MEDIDAS[m]}) ---")
        for esc in escenarios:
            hecho += 1
            dir_m = salida / "individual" / m
            cfg = construir_config(red, args, esc, [m], usar_aisl, sec)
            reg = resolver_y_guardar(
                resolver_optimizacion, data, red, cfg, esc,
                dir_m / f"esc_{esc:03d}.json", dir_m / f"esc_{esc:03d}.npz",
                etiqueta=m, forzar=args.forzar,
                completo=args.completo, firma=firma)
            if reg:
                filas.append({"medida": m, "escenario": esc,
                              **{k: reg.get(k, float("nan")) for k in INDICADORES + AGREGADOS + _COLS_COSTOS + _COLS_SOLVER},
                              "horas": reg.get("horas", args.horas),
                              "segundos": reg["segundos"]})
            _log(f"  progreso {hecho}/{total}  ({time.time()-t_ini:.0f} s acumulados)")

    if not filas:
        _log("FASE 1 sin resultados utilizables.")
        return pd.DataFrame()

    # El resumen se rearma desde TODOS los .json de la carpeta, no solo desde
    # las corridas de esta pasada. Antes se escribía `filas` tal cual, así que
    # correr el barrido acotado a una medida ("--medidas reconfig_topo
    # --escenarios 0") dejaba el CSV con una sola fila y borraba el resumen de
    # las otras 119 corridas, que seguían en disco pero desaparecían de la app.
    # Los .json son la fuente de verdad; el CSV es una vista de ellos.
    m_disco, e_disco = corridas_en_disco(salida)
    df = cargar_fase1(salida, m_disco, e_disco)
    if df.empty:
        df = pd.DataFrame(filas)
    df = df.sort_values(["medida", "escenario"])
    ruta = salida / "resumen_individual.csv"
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    extra = (f" ({len(df) - len(filas)} de pasadas anteriores)"
             if len(df) > len(filas) else "")
    _log(f"-> {ruta.name} ({len(df)} corridas{extra})")
    return df


def resolver_salida(salida, var):
    """Carpeta de resultados del barrido, a partir de `--salida`.

    Un `--salida` con un NOMBRE suelto ("sec-0-2-9-10 severo") se resuelve
    bajo `artefactos/barrido/`, que es donde la app busca los barridos. Antes
    se usaba tal cual, o sea relativo al directorio desde el que se lanzara el
    script: un barrido completo terminó escrito en `app/sec-0-2-9-10 severo no/`
    y la pestaña 🧪 Barrido no lo veía, aunque las corridas estaban en disco.
    Es el mismo criterio que ya aplica el lanzador de la app
    (`barrido_vista.seccion_lanzar`: `DIR_BARRIDO / nombre`).

    Una ruta explícita se respeta tal cual —absoluta, o relativa con separador
    ("./x", "../otro/x", "artefactos/barrido/x")—, así que los comandos que ya
    pasan la ruta completa siguen funcionando igual.
    """
    if not salida:
        return ART_DIR / "barrido" / var
    p = Path(salida)
    # Se mira la CADENA cruda, no el Path: `Path("./x")` se normaliza a
    # `Path("x")` y pierde el "./", así que un `--salida ./x` —que dice
    # explícitamente "acá"— acabaría redirigido igual que un nombre suelto.
    # Con separador (de cualquier tipo) es una ruta y se respeta; sin
    # separador es un nombre y va bajo artefactos/barrido/.
    tiene_sep = any(s in str(salida) for s in ("/", "\\"))
    if not p.is_absolute() and not tiene_sep:
        return ART_DIR / "barrido" / p.name
    return p


def corridas_en_disco(salida):
    """Todas las (medida, escenario) que ya tienen .json en la carpeta.

    Se lee del directorio y no de los argumentos de esta pasada: correr el
    barrido acotado a una medida NO significa que el resumen deba quedar con
    una sola fila.
    """
    d = Path(salida) / "individual"
    if not d.exists():
        return [], []
    medidas = sorted(p.name for p in d.iterdir() if p.is_dir())
    escs = sorted({int(f.stem.split("_")[1])
                   for m in medidas for f in (d / m).glob("esc_*.json")})
    return medidas, escs


def cargar_fase1(salida, medidas, escenarios):
    """Rearma la tabla de la fase 1 desde los .json ya escritos en disco."""
    filas = []
    for m in medidas:
        for esc in escenarios:
            ruta = salida / "individual" / m / f"esc_{esc:03d}.json"
            if not ruta.exists():
                continue
            reg = json.loads(ruta.read_text(encoding="utf-8"))
            if reg.get("status") != "optimal":
                continue
            filas.append({"medida": m, "escenario": esc,
                          **{k: reg.get(k, float("nan")) for k in INDICADORES + AGREGADOS + _COLS_COSTOS + _COLS_SOLVER},
                          "horas": reg.get("horas", 0),
                          "segundos": reg.get("segundos", 0)})
    return pd.DataFrame(filas)


# ════════════════════════════════════════════════════════════
# FASE 2 — promedio y escenario representativo POR MEDIDA
# ════════════════════════════════════════════════════════════

# Columna que resume el criterio en una sola métrica (para dispersión/CV).
COL_CRITERIO = {"objetivo": "objetivo", "ens": "ens", "multi": "ens"}


def _distancia(sub, criterio):
    """Serie de distancias al promedio, indexada igual que `sub`."""
    if criterio in ("objetivo", "ens"):
        col = criterio
        return (sub[col] - sub[col].mean()).abs()
    if criterio == "multi":
        # z-score por indicador para que SAIDI (h/cliente) y ENS (kWh) pesen
        # lo mismo; un indicador con std=0 no discrimina entre escenarios y
        # se deja fuera de la distancia.
        acum = pd.Series(0.0, index=sub.index)
        for col in ("saidi", "saifi", "ens"):
            sd = sub[col].std(ddof=0)
            if sd > 1e-12:
                acum += ((sub[col] - sub[col].mean()) / sd) ** 2
        return np.sqrt(acum)
    raise ValueError(f"criterio desconocido: {criterio}")


def fase2(df, criterio, salida, mip_gap=0.02):
    if df.empty:
        _log("FASE 2: no hay resultados de la fase 1.")
        return {}, pd.DataFrame()

    proms, reps = [], {}
    for m, sub in df.groupby("medida"):
        sub = sub.sort_values("escenario")
        dist = _distancia(sub, criterio)
        i = dist.idxmin()
        esc_rep = int(sub.loc[i, "escenario"])

        fila = {"medida": m, "n_escenarios": len(sub)}
        for col in INDICADORES:
            fila[f"{col}_prom"] = float(sub[col].mean())
            fila[f"{col}_std"] = float(sub[col].std(ddof=0))
            fila[f"{col}_rep"] = float(sub.loc[i, col])
        # De los agregados físicos basta el promedio y el valor del
        # representativo: la desviación de, por ejemplo, la energía por
        # tecnología rara vez se reporta y triplicaría el ancho de la tabla.
        for col in AGREGADOS:
            if col in sub:
                fila[f"{col}_prom"] = float(sub[col].mean())
                fila[f"{col}_rep"] = float(sub.loc[i, col])
        # Excepción: el dimensionamiento va como rango (ver _COLS_DIMENSION).
        # `n_instala` es en cuántos escenarios la medida instala algo de esa
        # magnitud, y es lo que distingue un promedio bajo porque todos
        # instalan poco de uno bajo porque casi ninguno instala.
        for col in _COLS_DIMENSION:
            if col in sub:
                fila[f"{col}_min"] = float(sub[col].min())
                fila[f"{col}_max"] = float(sub[col].max())
                fila[f"{col}_n_instala"] = int((sub[col].abs() > 1e-9).sum())
        # Descomposición económica, con el mismo tratamiento que los
        # indicadores. Los costos UNITARIOS (USD/kWh evitado y similares) no se
        # promedian acá a propósito: se calculan como cociente de promedios
        # (barrido_vista._resumen_eco), porque el promedio de razones pondera
        # igual un escenario que evitó 100 kWh y otro que evitó 10.000.
        for col in _COLS_COSTOS:
            if col in sub:
                fila[f"{col}_prom"] = float(sub[col].mean())
                fila[f"{col}_std"] = float(sub[col].std(ddof=0))
                fila[f"{col}_rep"] = float(sub.loc[i, col])
        # Dispersión relativa del criterio entre escenarios. Si cae por
        # debajo del MIPGap, la elección del representativo está dominada por
        # el ruido de optimalidad del solver, no por la física del escenario.
        col_cv = COL_CRITERIO[criterio]
        base = abs(sub[col_cv].mean())
        fila["cv_criterio"] = float(sub[col_cv].std(ddof=0) / base) if base > 0 else 0.0
        fila["esc_representativo"] = esc_rep
        fila["distancia_rep"] = float(dist.loc[i])
        proms.append(fila)

        reps[m] = {
            "escenario": esc_rep,
            "criterio": criterio,
            "distancia": float(dist.loc[i]),
            "ranking": [{"escenario": int(e), "distancia": float(d)}
                        for e, d in sorted(zip(sub["escenario"], dist),
                                           key=lambda p: p[1])],
            "promedios": {c: float(sub[c].mean()) for c in INDICADORES},
            "valores_rep": {c: float(sub.loc[i, c]) for c in INDICADORES},
        }

    df_prom = pd.DataFrame(proms).sort_values("medida")
    df_prom.to_csv(salida / "promedios.csv", index=False, encoding="utf-8-sig")
    (salida / "representativos.json").write_text(
        json.dumps({"criterio": criterio, "por_medida": reps,
                    "escenarios_representativos": sorted({v["escenario"]
                                                          for v in reps.values()})},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 96)
    print(f"  FASE 2 - escenario representativo por medida (criterio: {criterio})")
    print("=" * 96)
    print(f"  {'medida':<16} {'n':>3} {'objetivo prom':>16} {'ENS prom':>13} "
          f"{'SAIDI prom':>11} {'esc rep':>8} {'desv. rep':>10}")
    for _, f in df_prom.iterrows():
        rel = (abs(f["objetivo_rep"] - f["objetivo_prom"]) / abs(f["objetivo_prom"])
               if f["objetivo_prom"] else 0.0)
        print(f"  {f['medida']:<16} {int(f['n_escenarios']):>3} "
              f"{f['objetivo_prom']:>16,.0f} {f['ens_prom']:>13,.1f} "
              f"{f['saidi_prom']:>11.3f} {int(f['esc_representativo']):>8} "
              f"{rel:>9.2%}")
    print()
    print(f"  Escenarios representativos distintos: "
          f"{sorted({v['escenario'] for v in reps.values()})}")
    ruidosos = list(df_prom.loc[df_prom["cv_criterio"] < mip_gap, "medida"])
    if ruidosos:
        print(f"  [!] Dispersion entre escenarios menor al MIPGap ({mip_gap:.2%}) en: "
              f"{ruidosos}. Para esas medidas los escenarios son casi "
              f"indistinguibles y el representativo es poco robusto; bajar "
              f"--mip_gap o usar --criterio ens/multi.")
    print("=" * 96)
    return reps, df_prom


# ════════════════════════════════════════════════════════════
# FASE 3 — todas las medidas juntas sobre los representativos
# ════════════════════════════════════════════════════════════

def fase3(args, data, red, medidas, reps, salida, usar_aisl, sec, combos_extra,
          firma=None):
    from optimization_6_oficial import resolver_optimizacion

    # Por defecto la fase 3 corre sobre los escenarios representativos, que es
    # su razón de ser. Con --combo-escenarios se corre sobre los que se pidan,
    # y entonces NO hace falta haber corrido la fase 2: si el usuario dice
    # sobre qué escenario quiere la combinación, exigirle un representativo
    # sería pedirle permiso para algo que ya decidió.
    if args.combo_escenarios:
        escenarios_rep = parse_lista_int(args.combo_escenarios)
        ajenos = [e for e in escenarios_rep
                  if e not in {v["escenario"] for v in reps.values()}]
        if reps and ajenos:
            _log(f"[!] {ajenos} no son escenarios representativos; se corren "
                 f"igual, pero no son comparables con el resto de la fase 3.")
    elif not reps:
        _log("FASE 3: no hay escenarios representativos. Correr la fase 2 "
             "antes, o indicar los escenarios con --combo-escenarios.")
        return pd.DataFrame()
    else:
        escenarios_rep = sorted({v["escenario"] for v in reps.values()})

    # Combo por defecto: todas las medidas que se corrieron individualmente,
    # encendidas a la vez. El MILP elige la cartera dentro de ese menú.
    # --no-todas lo desactiva para correr solo los --combo explícitos.
    combos = ([("todas", list(medidas))] if args.todas else []) + [
        ("combo_" + "+".join(c), c) for c in combos_extra]
    if not combos:
        _log("FASE 3: no hay ninguna combinacion que correr (--no-todas sin "
             "ningun --combo).")
        return pd.DataFrame()

    filas = []
    for nombre, ms in combos:
        _log(f"--- combinacion '{nombre}': {ms} ---")
        for esc in escenarios_rep:
            dir_c = salida / "combinadas" / nombre
            cfg = construir_config(red, args, esc, ms, usar_aisl, sec)
            reg = resolver_y_guardar(
                resolver_optimizacion, data, red, cfg, esc,
                dir_c / f"esc_{esc:03d}.json", dir_c / f"esc_{esc:03d}.npz",
                etiqueta=nombre, forzar=args.forzar,
                completo=args.completo, firma=firma)
            if reg:
                filas.append({
                    "combinacion": nombre,
                    "escenario": esc,
                    "medidas": "+".join(ms),
                    "representa_a": "+".join(sorted(m for m, v in reps.items()
                                                    if v["escenario"] == esc)),
                    **{k: reg.get(k, float("nan")) for k in INDICADORES + AGREGADOS + _COLS_COSTOS + _COLS_SOLVER},
                    "horas": reg.get("horas", args.horas),
                    "segundos": reg["segundos"],
                })

    if not filas:
        return pd.DataFrame()
    # Mismo criterio que la fase 1: el CSV es una vista de los .json, así que
    # correr UNA combinación no puede borrar del resumen las que ya existían.
    df = cargar_fase3(salida, reps)
    if df.empty:
        df = pd.DataFrame(filas)
    df.to_csv(salida / "resumen_combinadas.csv", index=False, encoding="utf-8-sig")
    extra = (f" ({len(df) - len(filas)} de pasadas anteriores)"
             if len(df) > len(filas) else "")
    _log(f"-> resumen_combinadas.csv ({len(df)} corridas{extra})")
    return df


def cargar_fase3(salida, reps=None):
    """Rearma la tabla de la fase 3 desde los .json de `combinadas/`."""
    d = Path(salida) / "combinadas"
    if not d.exists():
        return pd.DataFrame()
    reps = reps or {}
    filas = []
    for ruta in sorted(d.glob("*/esc_*.json")):
        try:
            reg = json.loads(ruta.read_text(encoding="utf-8"))
        except Exception:                       # noqa: BLE001
            continue
        if reg.get("status") != "optimal":
            continue
        esc = int(reg["escenario"])
        ms = reg.get("medidas_on") or []
        filas.append({
            "combinacion": ruta.parent.name,
            "escenario": esc,
            "medidas": "+".join(ms),
            # Se re-deriva de los representativos vigentes: es información de
            # la fase 2, no de la corrida, y no viaja en el .json.
            "representa_a": "+".join(sorted(m for m, v in reps.items()
                                            if v.get("escenario") == esc)),
            **{k: reg.get(k, float("nan"))
               for k in INDICADORES + AGREGADOS + _COLS_COSTOS + _COLS_SOLVER},
            "horas": reg.get("horas", 0),
            "segundos": reg.get("segundos", 0),
        })
    if not filas:
        return pd.DataFrame()
    return pd.DataFrame(filas).sort_values(["combinacion", "escenario"])


# ════════════════════════════════════════════════════════════

def _objetivo_plan(reg):
    """Objetivo referido al caso sin fallas del MISMO plan, no al E4.

    El MILP resta la constante del E4, así que su objetivo lleva adentro
    `sobrecosto_dg` (costo sin fallas del plan menos el del E4). Restarlo deja
    el objetivo como si la referencia hubiera sido el sin fallas con la DG
    instalada donde la puso la estrategia. Es un valor de reporte: `objetivo`
    se conserva tal cual porque con él se eligen los representativos (fase 2).
    Sin DG renovable no hay sobrecosto y los dos coinciden.
    """
    obj = reg.get("objetivo")
    if obj is None:
        return None
    return float(obj) - float(reg.get("cst_sobrecosto_dg") or 0.0)


def recostear_dg(salida, data):
    """Recalcula los `cst_*` de las corridas YA GUARDADAS que llevan DG renovable.

    Las decisiones del MILP no cambian con la referencia del reporte, así que
    no hace falta re-resolver: se lee la corrida completa del .pkl.gz, se
    evalúa sin fallas con su DG (`referencia_dg_plan`) y se reescribe el .json.
    Una corrida sin .pkl.gz no se puede recostear y queda avisada.
    """
    hechas = 0
    for ruta in sorted(salida.glob("*/*/esc_*.json")):   # individual/<m>/, combinadas/<c>/
        reg = json.loads(ruta.read_text(encoding="utf-8"))
        if reg.get("status") != "optimal":
            continue
        if not {"dg_sol", "dg_eol"} & set(reg.get("medidas_on", [])):
            continue
        nombre = f"{ruta.parent.name}/{ruta.name}"
        gz = ruta.with_suffix(".pkl.gz")
        if not gz.exists():
            _log(f"  [!] {nombre}: falta {gz.name}, no se puede recostear")
            continue
        entrada = next(iter(P.leer_historial(gz).values()))
        res = [h for h in entrada["results"][0] if h]
        cfg = entrada["config"]
        cst = costos_de(res, int(reg["escenario"]), cfg, data,
                        _variante(cfg.get("aislacion_fallas_fija", False),
                                  cfg.get("seccionadores_existentes", [])))
        if not cst:
            continue
        antes = reg.get("cst_capex_anual")
        reg.update({f"cst_{k}": v for k, v in cst.items()
                    if k in COSTOS or k == "ahorro_neto"})
        reg["cst_objetivo_plan"] = _objetivo_plan(reg)
        ruta.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
        hechas += 1
        _log(f"  {nombre}: capex {antes if antes is None else f'{antes:,.0f}'} -> "
             f"{reg['cst_capex_anual']:,.0f}  sobrecosto_dg="
             f"{reg.get('cst_sobrecosto_dg', float('nan')):,.0f}")
    return hechas


def reparar_corridas(salida, data, escenarios=None, medidas=None):
    """Completa lo que una corrida no alcanzó a guardar: resumen de red y costos.

    Pasa si el proceso del barrido se quedó con módulos viejos en memoria (se
    editó código mientras corría) o si `grafo`/`data_loader` fallaron al
    importarse: la corrida se guarda igual, pero sin `resumen_red` ni `cst_*`.
    La solución del MILP está en el .pkl.gz, así que no se vuelve a resolver:
    se relee, se rehace el resumen y se recalculan los costos (para la DG eso
    incluye el caso sin fallas con su plan, que sí es un MILP más corto).
    Solo toca las corridas a las que les falte algo; `escenarios` y `medidas`
    las acotan.
    """
    hechas = 0
    for ruta in sorted(salida.glob("*/*/esc_*.json")):
        reg = json.loads(ruta.read_text(encoding="utf-8"))
        if reg.get("status") != "optimal":
            continue
        if escenarios is not None and int(reg["escenario"]) not in escenarios:
            continue
        if medidas is not None and not (set(reg.get("medidas_on", [])) & set(medidas)):
            continue
        falta_red = not reg.get("resumen_red")
        falta_cst = "cst_capex_anual" not in reg
        if not (falta_red or falta_cst):
            continue
        nombre = f"{ruta.parent.name}/{ruta.name}"
        gz = ruta.with_suffix(".pkl.gz")
        if not gz.exists():
            _log(f"  [!] {nombre}: falta {gz.name}, no se puede reparar")
            continue
        entrada = next(iter(P.leer_historial(gz).values()))
        res = [h for h in entrada["results"][0] if h]
        cfg = entrada["config"]
        hecho = []
        if falta_red:
            r = resumen_red(res)
            if r:
                reg["resumen_red"] = r
                hecho.append("resumen de red")
        if falta_cst:
            cst = costos_de(res, int(reg["escenario"]), cfg, data,
                            _variante(cfg.get("aislacion_fallas_fija", False),
                                      cfg.get("seccionadores_existentes", [])))
            if cst:
                reg.update({f"cst_{k}": v for k, v in cst.items()
                            if k in COSTOS or k == "ahorro_neto"})
                hecho.append("costos")
        if not hecho:
            continue
        reg["cst_objetivo_plan"] = _objetivo_plan(reg)
        ruta.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
        hechas += 1
        _log(f"  {nombre}: reparado ({', '.join(hecho)})")
    return hechas


def main():
    ap = argparse.ArgumentParser(
        description="Barrido medida x escenario, promedio de indicadores y "
                    "eleccion del escenario representativo.")
    ap.add_argument("--escenarios", default="0-9",
                    help="indices MC del barrido: '0-9', '0,3,7' (default: 0-9)")
    ap.add_argument("--medidas", default="todas",
                    help="claves separadas por coma, o 'todas' (default). "
                         f"Validas: {','.join(MEDIDAS)}")
    ap.add_argument("--horas", type=int, default=8760,
                    help="horizonte de cada corrida (default: 8760)")
    ap.add_argument("--fases", default="1,2,3",
                    help="fases a ejecutar (default: 1,2,3)")
    ap.add_argument("--criterio", default="objetivo",
                    choices=["objetivo", "ens", "multi"],
                    help="metrica de cercania al promedio (default: objetivo)")
    ap.add_argument("--todas", action=argparse.BooleanOptionalAction, default=True,
                    help="fase 3: correr ademas la combinacion con TODAS las "
                         "medidas de --medidas juntas (default). --no-todas la "
                         "omite y deja solo los --combo explicitos.")
    ap.add_argument("--combo-escenarios", dest="combo_escenarios", default="",
                    help="fase 3: escenarios sobre los que correr las "
                         "combinaciones. Default: los representativos de la "
                         "fase 2.")
    ap.add_argument("--combo", action="append", default=[],
                    help="combinacion EXTRA para la fase 3, ej: "
                         "--combo dg_sol,ess_4h (repetible)")
    ap.add_argument("--salida", default=None,
                    help="carpeta de resultados (default: artefactos/barrido/"
                         "<variante>). Un NOMBRE suelto ('sec-0-2-9-10 severo') "
                         "se crea dentro de artefactos/barrido/, que es donde la "
                         "app busca los barridos. Para escribir en otro lado hay "
                         "que dar una ruta: './x', '../x' o absoluta")
    ap.add_argument("--seccionadores", default="",
                    help="lineas con seccionador preexistente, ej: 0,4,9. "
                         "Activa la aislacion de fallas y exige baselines de "
                         "esa misma variante (pipeline --etapas E5 --seccionadores ...)")
    ap.add_argument("--completo", action=argparse.BooleanOptionalAction, default=True,
                    help="ademas del .npz slim, guardar la corrida COMPLETA en el "
                         ".pkl.gz que lee la app (~4.5 MB y ~9 s por corrida). "
                         "Es lo que permite abrirla en la pestana Resultados. "
                         "--no-completo lo desactiva.")
    ap.add_argument("--lim-dg", dest="lim_dg", type=float, default=P.LIM_DG_DEFAULT,
                    help=f"tope de capacidad de DG, como fraccion de la demanda "
                         f"base del grupo (default: {P.LIM_DG_DEFAULT}). Tiene que "
                         f"ser el MISMO con el que se calculo el E4 de --sin-fallas.")
    ap.add_argument("--lim-dg-modo", dest="lim_dg_modo", default=P.LIM_DG_MODO_DEFAULT,
                    choices=list(LDG.MODOS),
                    help="sobre que grupo se agrega el tope de DG: 'bus' (default, "
                         "el historico: cada barra topada por su propia demanda) o "
                         "'alimentador'/'zona'/'seccion' (tope agregado, el modelo "
                         "elige donde instalar dentro del grupo). Tiene que ser el "
                         "MISMO del E4 de --sin-fallas.")
    ap.add_argument("--lim-dg-bus", dest="lim_dg_bus", type=float, default=None,
                    help="tope adicional POR BARRA, en multiplos de su demanda "
                         "base. Recomendado con --lim-dg-modo distinto de 'bus'.")
    ap.add_argument("--lim-dg-conjunto", dest="lim_dg_conjunto", action="store_true",
                    help="solar + eolica comparten UN tope (la suma <= --lim-dg x "
                         "demanda del grupo) en vez de uno por tecnologia. Tiene que "
                         "ser el MISMO del E4 de --sin-fallas (pipeline.py "
                         "--lim-dg-conjunto).")
    ap.add_argument("--sin-fallas", dest="sin_fallas", default="",
                    help="etiqueta del artefacto de E4 a usar (default: el "
                         "historico sin_fallas.pkl). Ver las disponibles con "
                         "'python pipeline.py --estado'.")
    ap.add_argument("--microredes", action="store_true",
                    help="permite operacion en isla; solo aplica a las corridas "
                         "que incluyen switch RCS/MS")
    ap.add_argument("--forzar", action="store_true", help="rehacer corridas ya guardadas")
    ap.add_argument("--recostear-dg", dest="recostear_dg", action="store_true",
                    help="no resuelve nada: recalcula los costos de las corridas guardadas "
                         "con dg_sol/dg_eol contra el caso sin fallas de su propia DG, y "
                         "rearma resumen_individual, promedios y resumen_combinadas")
    ap.add_argument("--reparar", action="store_true",
                    help="no resuelve el MILP de las medidas: completa el resumen de red y los "
                         "costos que le falten a las corridas ya guardadas (de --escenarios y "
                         "--medidas) leyendo su .pkl.gz, y rearma resumen_individual")
    ap.add_argument("--listar", action="store_true",
                    help="mostrar el plan de corridas y salir, sin resolver")

    # Dimensionamiento de unidades moviles y switches (defaults = los de la UI)
    ap.add_argument("--n_meg", type=int, default=3)
    ap.add_argument("--p_meg_max", type=float, default=270.0)
    ap.add_argument("--t_tra_meg", type=int, default=4)
    ap.add_argument("--meg_existentes", type=int, default=1,
                    help="de las n_meg unidades, cuántas ya son de la empresa (sin CAPEX; "
                         "se usan antes de comprar). 0 = MEG comprado, como siempre")
    ap.add_argument("--n_meg_arr", type=int, default=1, help="unidades arrendables por evento")
    ap.add_argument("--p_meg_arr", type=float, default=200.0, help="kW por unidad arrendada")
    ap.add_argument("--h_meg_arr", type=int, default=6, help="horas de servicio por evento")
    ap.add_argument("--t_tra_meg_arr", type=int, default=None,
                    help="traslado del arrendado [h] (default: --t_tra_meg)")
    ap.add_argument("--n_mess", type=int, default=3)
    ap.add_argument("--h_mess", type=float, default=3.0)
    ap.add_argument("--p_mess_max", type=float, default=200.0)
    ap.add_argument("--t_tra_mess", type=int, default=2)
    ap.add_argument("--t_maniobra_ms", type=int, default=1)
    ap.add_argument("--t_maniobra_tie", type=int, default=0,
                    help="Horas que tarda cerrar una tie desde el inicio del evento "
                         "(reconfig_topo / inv_tie). 0 = instantánea (default).")
    ap.add_argument("--gurobi_mem_gb", type=float, default=None,
                    help="SoftMemLimit de Gurobi en GB. Default: 60 %% de la RAM (el resto "
                         "queda para el modelo Pyomo, que vive en el proceso Python).")
    ap.add_argument("--enlaces-nuevos", default="",
                    help="Enlaces CANDIDATOS a construir para inv_tie, como 'barra-barra:km' "
                         "separados por coma (ej. 5-10:1.0,13-3:2.0). Se agregan a la red solo en "
                         "memoria, inmunes a falla; los enlaces existentes siguen siendo de "
                         "reconfig_topo. Sin esto, inv_tie usa los enlaces existentes.")
    ap.add_argument("--radial_tie", action="store_true",
                    help="Cierre de enlaces sin mallar la red (radialidad.py). "
                         "Apagado por defecto.")

    # Solver
    ap.add_argument("--mip_gap", type=float, default=0.02)
    ap.add_argument("--time_limit", type=float, default=1800)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--mem_gb", type=float, default=None)
    ap.add_argument("--method", type=int, default=None, choices=[-1, 0, 1, 2, 3],
                    help="Method de Gurobi (por defecto 2 = barrier; 1 = simplex dual, usa menos memoria)")
    ap.add_argument("--bar_homogeneous", type=int, default=None, choices=[-1, 0, 1],
                    help="BarHomogeneous de Gurobi (por defecto: lo decide Gurobi; "
                         "1 = algoritmo homogeneo, mas estable si la raiz de barrier no converge)")
    ap.add_argument("--presolve", type=int, default=None, choices=[-1, 0, 1, 2],
                    help="Presolve de Gurobi (por defecto 2 = agresivo)")
    ap.add_argument("--mipfocus", type=int, default=None, choices=[0, 1, 2, 3],
                    help="MIPFocus de Gurobi (por defecto 1 = privilegia encontrar soluciones "
                         "factibles; 3 = privilegia mejorar la cota, util cuando el incumbente "
                         "ya es bueno y el gap no baja)")
    ap.add_argument("--voll", type=float, default=None,
                    help="Sensibilidad: valor de la energia no servida [USD/kWh], igual para todas "
                         "las barras. Por defecto el de costos.pkl (5). Cambia el objetivo, asi que "
                         "usar una --salida propia por valor.")

    ap.add_argument("--artefactos", default=None, help="Carpeta del caso de estudio")
    args = ap.parse_args()
    if args.artefactos:
        P.usar_artefactos(args.artefactos)
        global ART_DIR
        ART_DIR = P.ART_DIR
    # Antes de cualquier import perezoso de grafo / helpers_inversion: los dos
    # arrastran data_loader, que carga el E4 al importarse. Así cargan el
    # mismo E4 con que se resuelve el barrido (ver data_loader._cargar_artefactos).
    if args.sin_fallas:
        import os
        os.environ["ERAD_SIN_FALLAS"] = args.sin_fallas

    escenarios = parse_lista_int(args.escenarios)
    medidas = parse_medidas(args.medidas)
    fases = set(parse_lista_int(args.fases))
    combos_extra = [parse_medidas(c) for c in args.combo]

    sec = parse_lista_int(args.seccionadores) if args.seccionadores else []
    usar_aisl = bool(sec)
    var = _variante(usar_aisl, sec)

    salida = resolver_salida(args.salida, var)
    print("=" * 96)
    print("  BARRIDO DE MEDIDAS POR ESCENARIO")
    print("=" * 96)
    print(f"  escenarios      : {escenarios}  ({len(escenarios)})")
    print(f"  medidas         : {medidas}  ({len(medidas)})")
    print(f"  horizonte       : {args.horas} h")
    print(f"  aislacion       : {'si, seccionadores en ' + str(sec) if usar_aisl else 'no (perfecta)'}"
          f"  -> variante '{var}'")
    print(f"  E4 (sin fallas) : {P.ruta_sin_fallas(args.sin_fallas).name}")
    print(f"  tope DG         : {args.lim_dg} x P_D_base agregado por "
          f"{'barra' if args.lim_dg_modo == 'bus' else args.lim_dg_modo}"
          + (f"  (+ tope por barra {args.lim_dg_bus} x P_D_base)"
             if args.lim_dg_bus is not None else "")
          + ("  [CONJUNTO solar+eolica]" if args.lim_dg_conjunto else "  [por tecnologia]"))
    print(f"  criterio rep.   : {args.criterio}")
    print(f"  fases           : {sorted(fases)}")
    print(f"  MIPGap          : {args.mip_gap:.2%}   TimeLimit: {args.time_limit:.0f} s")
    # Absoluta a propósito: es la única forma de ver de un vistazo si el
    # barrido va a quedar donde la app lo busca.
    print(f"  salida          : {salida.resolve()}")
    print(f"  corridas fase 1 : {len(medidas) * len(escenarios)}")
    print("=" * 96)
    if args.horas < 8760:
        print("  [!] Con --horas < 8760 el credito de subestacion del objetivo compara")
        print("      contra un baseline (E5) de 8760 h, asi que fn_objetivo queda")
        print("      distorsionado en valor absoluto. Sirve para probar el flujo, NO")
        print("      para elegir el representativo con --criterio objetivo.")
        print("=" * 96)
    if args.listar:
        return

    # Recién acá: con --listar no se crea nada en disco, para que un vistazo
    # al plan no deje carpetas vacías dando vueltas.
    salida.mkdir(parents=True, exist_ok=True)

    # Antes de resolver nada: que esta corrida no contamine lo ya guardado.
    verificar_supuestos(salida, {"lim_dg": args.lim_dg,
                                 "lim_dg_modo": args.lim_dg_modo,
                                 "lim_dg_bus": _lim_bus_reg(args.lim_dg_bus),
                                 "lim_dg_conjunto": bool(args.lim_dg_conjunto),
                                 "e4_sin_fallas": args.sin_fallas or "historico",
                                 "enlaces_nuevos": args.enlaces_nuevos or "",
                                 "horas": int(args.horas),
                                 "n_meg": args.n_meg,
                                 "P_meg_max": args.p_meg_max,
                                 "t_tra_meg": args.t_tra_meg,
                                 "meg_existentes": args.meg_existentes}, args.forzar)

    data, red = construir_data(escenarios, var, args.sin_fallas, args.lim_dg,
                               args.horas, lim_dg_modo=args.lim_dg_modo,
                               lim_dg_bus=args.lim_dg_bus,
                               lim_dg_conjunto=args.lim_dg_conjunto)
    if args.enlaces_nuevos:
        data, red = agregar_enlaces_nuevos(data, red, parse_enlaces_nuevos(args.enlaces_nuevos))
    if args.voll is not None:
        # Solo la sensibilidad al VoLL: el modelo y el costo del caso base leen data["VoLL"].
        data["VoLL"] = {b: float(args.voll) for b in data["VoLL"]}
        _log(f"VoLL = {args.voll:g} USD/kWh en todas las barras (sensibilidad)")
    # Se calcula una vez: es la misma para todas las corridas del barrido.
    firma = P.firma_modelo() if args.completo else None

    if args.recostear_dg or args.reparar:
        if args.reparar:
            n = reparar_corridas(salida, data, escenarios=set(escenarios),
                                 medidas=None if args.medidas == "todas" else args.medidas.split(","))
            _log(f"REPARACIÓN: {n} corridas completadas")
        else:
            n = recostear_dg(salida, data)
            _log(f"RECOSTEO DG: {n} corridas actualizadas")
        m_disco, e_disco = corridas_en_disco(salida)
        df = cargar_fase1(salida, m_disco, e_disco)
        reps = {}
        if not df.empty:
            df.sort_values(["medida", "escenario"]).to_csv(
                salida / "resumen_individual.csv", index=False, encoding="utf-8-sig")
            # Los representativos no cambian: el criterio no usa costos.
            reps, _ = fase2(df, args.criterio, salida, mip_gap=args.mip_gap)
        df3 = cargar_fase3(salida, reps)
        if not df3.empty:
            df3.to_csv(salida / "resumen_combinadas.csv", index=False, encoding="utf-8-sig")
        return

    df = pd.DataFrame()
    if 1 in fases:
        df = fase1(args, data, red, medidas, escenarios, salida, usar_aisl, sec, firma)
    if df.empty:
        df = cargar_fase1(salida, medidas, escenarios)

    reps = {}
    if 2 in fases:
        reps, _ = fase2(df, args.criterio, salida, mip_gap=args.mip_gap)
    elif (salida / "representativos.json").exists():
        reps = json.loads((salida / "representativos.json")
                          .read_text(encoding="utf-8"))["por_medida"]

    if 3 in fases:
        df3 = fase3(args, data, red, medidas, reps, salida, usar_aisl, sec, combos_extra,
                    firma)
        if not df3.empty:
            print()
            print("=" * 96)
            print("  FASE 3 - medidas combinadas sobre los escenarios representativos")
            print("=" * 96)
            print(f"  {'combinacion':<24} {'esc':>5} {'objetivo':>16} {'ENS':>13} "
                  f"{'SAIDI':>10}")
            for _, f in df3.iterrows():
                print(f"  {f['combinacion']:<24} {int(f['escenario']):>5} "
                      f"{f['objetivo']:>16,.0f} {f['ens']:>13,.1f} {f['saidi']:>10.3f}")
            print("=" * 96)

    _log(f"listo. Resultados en {salida}")


if __name__ == "__main__":
    main()
