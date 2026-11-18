# ============================================================
# rutas.py — Dónde está el código y dónde están los datos del caso
# ============================================================
# El código vive una sola vez en `mi_proyecto/comun/app/` y cada caso de
# estudio es una carpeta con solo datos:
#
#   mi_proyecto/
#   ├── comun/app/            ← este módulo y todo el código
#   ├── comun/clima/          ← scripts de la cadena climática
#   ├── caso_validacion/      ← un caso
#   │   ├── .caso_erad        ← marca que identifica la carpeta como caso
#   │   ├── datos/            ← artefactos/, app.pkl, FaultlinMC.npy, notebooks
#   │   ├── Proyecto oficial/ ← Inputs/ y Outputs/ del clima
#   │   └── Scenario Database/Inputs/
#   └── caso_chilquinta/      ← otro caso, misma estructura
#
# El caso activo se decide así, del más explícito al más cómodo:
#
#   1. `$ERAD_CASO` — ruta a la carpeta del caso.
#   2. La primera carpeta con `.caso_erad` desde el directorio actual hacia
#      arriba. Un notebook abierto en `caso_x/datos/` o un comando lanzado
#      desde `caso_x/` toman ese caso sin configurar nada.
#   3. `caso_validacion/`, el estudio de la tesis, como funcionó siempre.
#
# Al importarse publica el caso en `$ERAD_CASO`, para que los subprocesos
# (jobs de la app, barridos, construir_faultlin) hereden el mismo caso aunque
# se lancen con otro directorio de trabajo.
#
# `datos/` está a la misma profundidad que tenía `app/app`, así que las rutas
# relativas de los notebooks (`cwd.parent / "Proyecto oficial"`,
# `cwd.parents[3] / "Tesis_..."`) siguen valiendo sin tocarlos.
# ============================================================

import os
from pathlib import Path

CODIGO_DIR = Path(__file__).resolve().parent
MI_PROYECTO = CODIGO_DIR.parents[1]
VAR_CASO = "ERAD_CASO"
MARCA = ".caso_erad"
CASO_DEFECTO = MI_PROYECTO / "caso_validacion"


def _con_marca(desde):
    for d in (desde, *desde.parents):
        if (d / MARCA).is_file():
            return d
    return None


def resolver_caso():
    env = os.environ.get(VAR_CASO)
    if env:
        return Path(env).resolve()
    return _con_marca(Path.cwd().resolve()) or CASO_DEFECTO


CASO_DIR = resolver_caso()
DATOS_DIR = CASO_DIR / "datos"
os.environ[VAR_CASO] = str(CASO_DIR)
