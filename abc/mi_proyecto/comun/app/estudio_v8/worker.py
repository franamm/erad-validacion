"""Supervisor independiente de la sesión del navegador."""
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from estudio_v8.core import APP_DIR, leer_json
from rutas import DATOS_DIR  # datos del caso activo (ver rutas.py)
from estudio_v8.jobs import guardar


def main():
    meta = Path(sys.argv[1])
    info = leer_json(meta)
    info.update(estado="ejecutando", pid=os.getpid(), create_time=psutil.Process().create_time())
    guardar(meta, info)
    try:
        # Un trabajo puede encadenar pasos (casos base → barrido → plan fijo):
        # se detiene en el primero que falla.
        comandos = info.get("comandos") or [info["comando"]]
        codigo = 0
        for n, comando in enumerate(comandos, 1):
            info.update(paso=n, pasos=len(comandos), comando=comando)
            guardar(meta, info)
            print(f"[v8] paso {n}/{len(comandos)}: {subprocess.list2cmdline(comando)}", flush=True)
            # stdout/stderr EXPLÍCITOS al log del trabajo. Sin esto, en Windows y
            # con CREATE_NO_WINDOW el hijo queda sin consola y su salida se
            # descarta: el log solo juntaba las líneas "[v8] paso n/N" y se
            # perdía todo lo demás — el progreso `[pipeline]`, el gap de Gurobi
            # que `progreso.ultimo_solver` busca, y el traceback cuando una
            # etapa falla, que es justo lo que hace falta para entender por qué
            # falló. El seguimiento en vivo mostraba un log casi vacío mientras
            # el cálculo avanzaba a ciegas.
            codigo = subprocess.call(comando, cwd=DATOS_DIR,
                                     stdout=sys.stdout, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                     env={**os.environ, "PYTHONUTF8": "1"})
            if codigo:
                break
        pendientes = [p for p in info.get("esperados", []) if not Path(p).exists()
                      or (p.endswith(".json") and leer_json(p).get("status") != "optimal")]
        info.update(codigo_salida=codigo,
                    estado="fallido" if codigo else ("incompleto" if pendientes else "terminado"),
                    pendientes=len(pendientes))
    except Exception as exc:
        info.update(estado="fallido", error=str(exc))
        raise
    finally:
        info["fin"] = time.time()
        guardar(meta, info)
        (meta.parent / ".trabajo_v8.lock").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
