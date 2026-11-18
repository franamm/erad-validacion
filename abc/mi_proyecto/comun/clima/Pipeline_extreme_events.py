#!/usr/bin/env python3
"""
pipeline_extreme_events.py
==========================
Script maestro que automatiza el flujo completo de procesamiento de eventos extremos
para el análisis de curva de fragilidad de vientos extremos (Prueba-12-oficial.ipynb).

ORDEN DE EJECUCIÓN
------------------
Paso 1: download_openmeteo_historical_ready_v2.py
        → Descarga datos horarios históricos (Open-Meteo API)
        → Genera: Inputs/Hourlyhistoricdata.nc

Paso 2: download_arclim_future_to_netcdf_ready_v4.py
        → Descarga variables futuras diarias (ARClim API)
        → Genera: Inputs/Dailyfuturedata.nc
                  Inputs/Dailyfuturedatarelativehumidity.nc

Paso 3: Hourly_climate_zone_humidity_daily_patch.py
        → Combina histórico + futuro en base horaria por zona climática
        → Lee:    Inputs/Hourlyhistoricdata.nc
                  Inputs/Dailyfuturedata.nc
                  Inputs/Dailyfuturedatarelativehumidity.nc
                  Inputs/SystemzoneV1.xlsx
        → Genera: Outputs/BaseFhorPELP4v20z2anos.nc

Paso 4: Extreme_events_arclim_final.py
        → Calcula eventos extremos (viento, humedad, FFDI, incendios)
        → Lee:    Outputs/BaseFhorPELP4v20z2anos.nc
                  Inputs/Dailyfuturedata.nc
                  Inputs/Dailyfuturedatarelativehumidity.nc
                  Inputs/SystemzoneV1.xlsx
        → Genera: Outputs/BaseHoraEWEPELP20z.nc       ← entrada para el notebook
                  Outputs/resumenEWEarclim.csv
                  Outputs/EWEwindhoraanual.npy
                  Outputs/ordenzonasBaseHoraEWEPELP20z.csv

ARCHIVO DE SALIDA PRINCIPAL para Prueba-12-oficial.ipynb:
        Outputs/BaseHoraEWEPELP20z.nc

USO
---
  # Ejecutar el pipeline completo (todos los pasos):
  python pipeline_extreme_events.py

  # Ejecutar solo desde un paso específico (ej: si ya tienes el histórico):
  python pipeline_extreme_events.py --start-step 2

  # Ejecutar solo pasos individuales:
  python pipeline_extreme_events.py --only-step 4

  # Pasar argumentos extra a un script específico:
  python pipeline_extreme_events.py --step2-args "--scenario ssp245 --start-year 2029 --end-year 2030"

  # Modo dry-run (muestra los comandos sin ejecutarlos):
  python pipeline_extreme_events.py --dry-run

  # Omitir un paso si su(s) archivo(s) de salida ya existen:
  python pipeline_extreme_events.py --skip-if-exists

ESTRUCTURA DE DIRECTORIOS ESPERADA
-----------------------------------
  <proyecto>/
  ├── Scripts/                    ← carpeta donde viven los scripts Python
  │   ├── pipeline_extreme_events.py   (este archivo)
  │   ├── download_openmeteo_historical_ready_v2.py
  │   ├── download_arclim_future_to_netcdf_ready_v4.py
  │   ├── Hourly_climate_zone_humidity_daily_patch.py
  │   └── Extreme_events_arclim_final.py
  ├── Inputs/
  │   └── SystemzoneV1.xlsx       ← requerido antes de correr el pipeline
  └── Outputs/

  Si los scripts viven en la raíz del proyecto (sin subcarpeta Scripts/),
  el pipeline los detecta igualmente.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuración del pipeline
# ---------------------------------------------------------------------------

# Nombre de los scripts en el orden de ejecución
SCRIPTS = {
    1: "download_openmeteo_historical_ready_v2.py",
    2: "download_arclim_future_to_netcdf_ready_v4.py",
    3: "Hourly_climate_zone_humidity_daily_patch.py",
    4: "Extreme_events_arclim_final.py",
}

STEP_DESCRIPTIONS = {
    1: "Descarga histórico horario desde Open-Meteo",
    2: "Descarga futuro diario desde ARClim",
    3: "Genera base horaria por zona climática + humedad",
    4: "Calcula eventos extremos (viento, FFDI, incendios)",
}

# Archivos de salida esperados por cada paso (relativos a <proyecto>/)
STEP_OUTPUTS = {
    1: ["Inputs/Hourlyhistoricdata.nc"],
    2: [
        "Inputs/Dailyfuturedata.nc",
        "Inputs/Dailyfuturedatarelativehumidity.nc",
    ],
    3: ["Outputs/BaseFhorPELP4v20z2anos.nc"],
    4: [
        "Outputs/BaseHoraEWEPELP20z.nc",
        "Outputs/resumenEWEarclim.csv",
        "Outputs/EWEwindhoraanual.npy",
    ],
}

# Archivo principal de salida para el notebook
NOTEBOOK_INPUT_FILE = "Outputs/BaseHoraEWEPELP20z.nc"


# ---------------------------------------------------------------------------
# Resolución de rutas
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """
    Detecta la raíz del proyecto buscando la carpeta 'Inputs' con
    'SystemzoneV1.xlsx'. Busca en: directorio del script → directorio padre.
    """
    from caso_clima import raiz_proyecto
    this_dir = Path(__file__).resolve().parent
    candidates = [Path(raiz_proyecto()), this_dir, this_dir.parent]
    for root in candidates:
        if (root / "Inputs" / "SystemzoneV1.xlsx").exists():
            return root
        if (root / "Inputs").is_dir():
            return root
    # Fallback: usar el directorio del script
    return this_dir


def find_script(script_name: str, project_root: Path) -> Optional[Path]:
    """Busca un script Python en Scripts/ o en la raíz del proyecto."""
    candidates = [
        project_root / "Scripts" / script_name,
        project_root / script_name,
        Path(__file__).resolve().parent / script_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# Ejecución de pasos
# ---------------------------------------------------------------------------

def outputs_exist(step: int, project_root: Path) -> bool:
    """Verifica si todos los archivos de salida de un paso ya existen."""
    for rel_path in STEP_OUTPUTS.get(step, []):
        if not (project_root / rel_path).exists():
            return False
    return True


def run_step(
    step: int,
    project_root: Path,
    extra_args: str = "",
    dry_run: bool = False,
    skip_if_exists: bool = False,
) -> bool:
    """
    Ejecuta un paso del pipeline.

    Returns
    -------
    bool
        True si el paso terminó con éxito (o fue saltado), False si hubo error.
    """
    script_name = SCRIPTS[step]
    description = STEP_DESCRIPTIONS[step]

    print()
    print("=" * 70)
    print(f"  PASO {step}/4: {description}")
    print(f"  Script : {script_name}")
    print("=" * 70)

    # Verificar si se puede saltar
    if skip_if_exists and outputs_exist(step, project_root):
        existing = [str(project_root / p) for p in STEP_OUTPUTS[step]]
        print(f"  [SKIP] Archivos de salida ya existen:")
        for f in existing:
            print(f"         {f}")
        return True

    # Localizar el script
    script_path = find_script(script_name, project_root)
    if script_path is None:
        print(f"  [ERROR] No se encontró el script: {script_name}")
        print(f"          Buscado en: Scripts/ y raíz del proyecto.")
        return False

    print(f"  Ruta   : {script_path}")

    # Construir el comando
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        import shlex
        cmd.extend(shlex.split(extra_args))

    print(f"  Comando: {' '.join(cmd)}")
    print()

    if dry_run:
        print("  [DRY-RUN] Comando NO ejecutado.")
        return True

    # Ejecutar
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            check=False,
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            print()
            print(f"  [OK] Paso {step} completado en {elapsed:.1f} s")
            # Verificar archivos generados
            missing = [
                str(project_root / p)
                for p in STEP_OUTPUTS.get(step, [])
                if not (project_root / p).exists()
            ]
            if missing:
                print("  [AVISO] Los siguientes archivos de salida esperados NO fueron creados:")
                for f in missing:
                    print(f"          {f}")
            else:
                print("  [OK] Archivos de salida verificados:")
                for rel in STEP_OUTPUTS.get(step, []):
                    print(f"       {project_root / rel}")
            return True
        else:
            print()
            print(f"  [ERROR] Paso {step} terminó con código {result.returncode}")
            return False

    except Exception as exc:
        print(f"  [EXCEPCIÓN] {exc}")
        return False


# ---------------------------------------------------------------------------
# Reporte final
# ---------------------------------------------------------------------------

def print_summary(project_root: Path, steps_run: list[int], steps_ok: list[int]) -> None:
    """Imprime un resumen del resultado del pipeline."""
    print()
    print("=" * 70)
    print("  RESUMEN DEL PIPELINE")
    print("=" * 70)

    all_ok = True
    for step in range(1, 5):
        if step not in steps_run:
            status = "  [--] omitido"
        elif step in steps_ok:
            status = "  [OK]"
        else:
            status = "  [FAIL]"
            all_ok = False
        print(f"  Paso {step}: {status}  {STEP_DESCRIPTIONS[step]}")

    print()
    notebook_input = project_root / NOTEBOOK_INPUT_FILE
    if notebook_input.exists():
        print(f"  ✔  Archivo listo para el notebook:")
        print(f"     {notebook_input}")
        print()
        print("  Para ejecutar el notebook:")
        print("     jupyter notebook Prueba-12-oficial.ipynb")
        print()
        print("  Asegúrate de que la ruta en el notebook apunte a:")
        print(f"     {notebook_input}")
    else:
        print(f"  ✘  El archivo principal del notebook NO fue generado:")
        print(f"     {notebook_input}")
        all_ok = False

    print()
    if all_ok:
        print("  Pipeline completado exitosamente.")
    else:
        print("  Pipeline terminó con errores. Revise los mensajes anteriores.")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline automatizado de eventos extremos → BaseHoraEWEPELP20z.nc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Iniciar desde el paso N (default: 1, ejecuta todo el pipeline).",
    )
    parser.add_argument(
        "--only-step",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="Ejecutar únicamente el paso N.",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Detenerse después del paso N (default: 4).",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Saltar un paso si sus archivos de salida ya existen.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostrar los comandos sin ejecutarlos.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Ruta raíz del proyecto (auto-detectada si se omite).",
    )
    # Argumentos extra por paso
    for s in [1, 2, 3, 4]:
        parser.add_argument(
            f"--step{s}-args",
            default="",
            metavar="ARGS",
            help=f'Argumentos extra para el script del paso {s} (entre comillas). '
                 f'Ej: --step2-args "--scenario ssp245 --end-year 2035"',
        )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolver raíz del proyecto
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = find_project_root()

    print()
    print("╔══════════════════════════════════╗")
    print("║   PIPELINE DE EVENTOS EXTREMOS   ║")
    print("╚══════════════════════════════════╝")
    print()
    print(f"  Raíz del proyecto : {project_root}")
    print(f"  Python            : {sys.executable}")
    if args.dry_run:
        print("  Modo              : DRY-RUN (sin ejecución real)")
    if args.skip_if_exists:
        print("  Skip-if-exists    : activado")
    print()

    # Crear carpeta Outputs si no existe
    outputs_dir = project_root / "Outputs"
    if not args.dry_run:
        outputs_dir.mkdir(parents=True, exist_ok=True)

    # Determinar qué pasos ejecutar
    if args.only_step is not None:
        steps_to_run = [args.only_step]
    else:
        steps_to_run = list(range(args.start_step, args.end_step + 1))

    extra_args_map = {
        1: args.step1_args,
        2: args.step2_args,
        3: args.step3_args,
        4: args.step4_args,
    }

    steps_ok: list[int] = []
    pipeline_ok = True

    for step in steps_to_run:
        ok = run_step(
            step=step,
            project_root=project_root,
            extra_args=extra_args_map[step],
            dry_run=args.dry_run,
            skip_if_exists=args.skip_if_exists,
        )
        if ok:
            steps_ok.append(step)
        else:
            pipeline_ok = False
            print()
            print(f"  [ABORT] El paso {step} falló. Pipeline detenido.")
            break

    print_summary(project_root, steps_to_run, steps_ok)
    sys.exit(0 if pipeline_ok else 1)


if __name__ == "__main__":
    main()
