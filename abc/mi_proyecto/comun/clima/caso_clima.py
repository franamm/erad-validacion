"""Carpeta 'Proyecto oficial' (Inputs/ y Outputs/) del caso activo.

Los scripts de clima viven una sola vez en `mi_proyecto/comun/clima/` y
leen y escriben en el caso que corresponda. Misma regla que
`comun/app/rutas.py`:

  1. `$ERAD_CASO` — ruta a la carpeta del caso.
  2. La primera carpeta con `.caso_erad` desde el directorio actual hacia
     arriba (correr los scripts desde dentro del caso basta).
  3. `caso_validacion/`, el estudio de la tesis.

Publica el caso en `$ERAD_CASO`, para que `Pipeline_extreme_events.py`
pase el mismo caso a cada paso que lanza.
"""
import os
from pathlib import Path

MI_PROYECTO = Path(__file__).resolve().parents[2]
VAR_CASO = "ERAD_CASO"
MARCA = ".caso_erad"


def caso_dir():
    env = os.environ.get(VAR_CASO)
    if env:
        return Path(env).resolve()
    aqui = Path.cwd().resolve()
    for d in (aqui, *aqui.parents):
        if (d / MARCA).is_file():
            return d
    return MI_PROYECTO / "caso_validacion"


def raiz_proyecto():
    """Ruta (str) a `<caso>/Proyecto oficial`, que contiene Inputs/ y Outputs/."""
    caso = caso_dir()
    os.environ[VAR_CASO] = str(caso)
    return str(caso / "Proyecto oficial")
