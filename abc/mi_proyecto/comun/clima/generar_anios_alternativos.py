#!/usr/bin/env python3
"""
Genera los anos operativos alternativos, para la sensibilidad del barrido.

NO hace falta correrlo ahora. Queda listo para cuando quieras verificar que el
ranking de medidas no depende del ano de amenaza que elegiste.

CONTEXTO
--------
El ano base del estudio es 2024 (mediana del periodo 2017-2025). La
variabilidad interanual en ese periodo es de 3.6x entre el ano mas benigno y
el mas severo, asi que conviene mostrar que las conclusiones aguantan.

Severidad medida en Arauco (horas con rafaga > 20.8 m/s, suma de las 3 zonas,
ECMWF IFS 2017-2025):

    ano    Z1    Z2    Z3   TOTAL   comentario
    2017   232   110    40    382
    2018   211   108    25    344
    2019   248   105    31    384
    2020   291   153    43    487   <- SEVERO (1.7x la mediana)
    2021   145    77    42    264   rafaga maxima del periodo (34.5 m/s)
    2022   141    63     4    208   <- EVITAR: Z3 casi sin eventos, contraste 35x
    2023   110    85    16    211
    2024   140   119    30    289   <- BASE (mediana exacta)
    2025    74    51    11    136   <- BENIGNO

PRIORIDAD
---------
El ano pesa mucho mas que el miembro GCM:
    variabilidad interanual  ->  3.6x
    dispersion entre GCM     ->  +/-5%
Si vas a correr solo un barrido extra, que sea 2020 (severo), no otro miembro.

USO
---
    python generar_anios_alternativos.py --listar
    python generar_anios_alternativos.py --solo severo
    python generar_anios_alternativos.py --solo severo benigno
    python generar_anios_alternativos.py --todos
    python generar_anios_alternativos.py --todos --miembros

Cada corrida consulta ARClim (6 requests) y escribe un NetCDF en Outputs/.
Los que ya existen se saltan salvo que pases --forzar.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
CONSTRUCTOR = os.path.join(AQUI, "construir_ano_operativo.py")
from caso_clima import raiz_proyecto  # noqa: E402
SALIDAS = os.path.join(raiz_proyecto(), "Outputs")

# etiqueta -> (ano_previo, ano_operativo, h_totales, nota)
ANIOS = {
    "severo":  (2019, 2020, 487, "1.7x la mediana. El caso de sensibilidad prioritario."),
    "benigno": (2024, 2025, 136, "0.47x la mediana. Cota inferior."),
    "alto":    (2018, 2019, 384, "Segundo mas severo, sin ser extremo."),
    "2017":    (2017, 2017, 382, "Primer ano con IFS HRES; previo = el mismo ano."),
    "2018":    (2017, 2018, 344, ""),
    "2021":    (2020, 2021, 264, "Contiene la rafaga maxima del periodo (34.5 m/s)."),
    "2023":    (2022, 2023, 211, ""),
    # 2022 queda fuera a proposito: Z3 tiene solo 4 h sobre umbral y el
    # contraste entre zonas se dispara a 35x, que no es comparable con el resto.
}

# variantes por miembro GCM, sobre el ano base 2024
MIEMBROS = {
    "awi":     ("AWI-CM-1-1-MR", "El mismo GCM que usa el paper de referencia."),
    "canesm5": ("CanESM5", "Extremo alto del ensemble."),
    "miroc6":  ("MIROC6", "Extremo bajo del ensemble."),
}


def correr(args_extra: list[str], salida: str, forzar: bool) -> bool:
    destino = os.path.join(SALIDAS, salida)
    if os.path.exists(destino) and not forzar:
        print(f"  [SKIP] ya existe: {salida}")
        return True
    cmd = [sys.executable, CONSTRUCTOR, "--salida-nc", salida] + args_extra
    print(f"  $ {' '.join(cmd[1:])}")
    r = subprocess.run(cmd, cwd=AQUI)
    if r.returncode != 0:
        print(f"  [ERROR] fallo con codigo {r.returncode}")
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Genera anos operativos alternativos para sensibilidad.")
    p.add_argument("--solo", nargs="+", metavar="ETIQUETA",
                   help="Etiquetas a generar. Ver --listar.")
    p.add_argument("--todos", action="store_true", help="Todos los anos de la tabla.")
    p.add_argument("--miembros", action="store_true", help="Ademas, las variantes por GCM sobre 2024.")
    p.add_argument("--listar", action="store_true", help="Muestra la tabla y sale.")
    p.add_argument("--forzar", action="store_true", help="Rehacer aunque el archivo exista.")
    a = p.parse_args()

    if a.listar or not (a.solo or a.todos or a.miembros):
        print("ANOS DISPONIBLES  (base del estudio: 2024, 289 h)\n")
        print(f"  {'etiqueta':<10} {'previo':>6} {'operativo':>10} {'h/ano':>7}  nota")
        for k, (prev, op, h, nota) in ANIOS.items():
            print(f"  {k:<10} {prev:>6} {op:>10} {h:>7}  {nota}")
        print("\nVARIANTES POR MIEMBRO GCM (sobre el ano base 2024)\n")
        for k, (m, nota) in MIEMBROS.items():
            print(f"  {k:<10} {m:<18} {nota}")
        print("\n  2022 excluido a proposito: Z3 con 4 h sobre umbral, contraste 35x.")
        print("\nEjemplo:  python generar_anios_alternativos.py --solo severo")
        return

    tareas: list[tuple[str, list[str], str]] = []
    etiquetas = list(ANIOS) if a.todos else (a.solo or [])
    for k in etiquetas:
        if k not in ANIOS:
            print(f"[!] etiqueta desconocida: {k}. Use --listar.")
            continue
        prev, op, _, _ = ANIOS[k]
        tareas.append((k, ["--anio-previo", str(prev), "--anio-operativo", str(op)],
                       f"BaseHoraEWE_arauco_{k}.nc"))

    if a.miembros:
        for k, (m, _) in MIEMBROS.items():
            tareas.append((f"miembro {k}", ["--miembro", m], f"BaseHoraEWE_arauco_gcm_{k}.nc"))

    if not tareas:
        print("Nada que hacer.")
        return

    print(f"Generando {len(tareas)} archivo(s) en {SALIDAS}\n")
    ok = 0
    for etiqueta, extra, salida in tareas:
        print(f"[{etiqueta}] -> {salida}")
        if correr(extra, salida, a.forzar):
            ok += 1
        print()

    print(f"Listos {ok}/{len(tareas)}.")
    print("\nDespues, por cada archivo:")
    print("  python ../../app/app/construir_faultlin.py --viento <archivo> "
          "--escenario with_EWE --offset 8760")
    print("y luego el barrido. Compare el RANKING de medidas, no los valores absolutos.")


if __name__ == "__main__":
    main()
