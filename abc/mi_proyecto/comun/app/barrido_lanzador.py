# ============================================================
# barrido_lanzador.py — lanzar el barrido como proceso aparte
# ============================================================
# La app NO ejecuta el barrido: lo LANZA. La diferencia importa.
#
# Si el MILP corriera dentro del proceso de Streamlit, moriría con él: al
# cerrar la pestaña, al dormirse el navegador, o al tocar cualquier widget
# (Streamlit re-ejecuta el script entero en cada interacción). Desactivar la
# UI mientras calcula no arregla nada de eso, solo impide usarla.
#
# Con un subproceso desacoplado:
#   - la página sigue navegable mientras corre;
#   - el cálculo sobrevive a cerrar el navegador, porque es un proceso del
#     sistema operativo y no una sesión de Streamlit;
#   - el progreso se lee de los artefactos, que el script ya guarda a medida
#     que termina cada corrida;
#   - cancelar es matar el PID.
#
# Este módulo no importa streamlit a propósito: es manejo de procesos, y así
# se puede probar desde consola.
# ============================================================

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

APP_DIR = Path(__file__).resolve().parent
SCRIPT = APP_DIR / "run_barrido_medidas.py"
from rutas import DATOS_DIR  # datos del caso activo (ver rutas.py)

# Un barrido por carpeta de salida: dos procesos escribiendo los mismos
# artefactos se pisarían los .json a medio escribir.
NOMBRE_LOCK = "barrido_en_curso.json"


def ruta_lock(salida):
    return Path(salida) / NOMBRE_LOCK


def ruta_log(salida):
    return Path(salida) / "barrido.log"


def construir_comando(*, escenarios, medidas, seccionadores="", horas=8760,
                      fases="1,2", criterio="objetivo", sin_fallas="",
                      lim_dg=1.0, mip_gap=0.02, time_limit=1800,
                      completo=True, microredes=False, forzar=False,
                      salida=None, extra=None, combos=None, todas=None,
                      combo_escenarios=""):
    """La lista de argumentos exacta que se va a ejecutar.

    Se devuelve para MOSTRARLA antes de lanzar: así el usuario ve el comando
    reproducible, lo puede copiar a una consola y no queda nada oculto detrás
    de un botón.

    `combos` es una lista de combinaciones para la fase 3, cada una como lista
    de claves de medida. Va aparte de `extra` porque --combo es repetible y un
    dict solo admitiría una; `todas=False` añade --no-todas para correr SOLO
    las combinaciones pedidas y no además la de todas las medidas juntas.
    """
    cmd = [sys.executable, str(SCRIPT),
           "--escenarios", str(escenarios),
           "--medidas", ",".join(medidas) if not isinstance(medidas, str) else medidas,
           "--horas", str(int(horas)),
           "--fases", str(fases),
           "--criterio", str(criterio),
           "--lim-dg", str(lim_dg),
           "--mip_gap", str(mip_gap),
           "--time_limit", str(int(time_limit))]
    if seccionadores:
        cmd += ["--seccionadores", str(seccionadores)]
    if sin_fallas:
        cmd += ["--sin-fallas", str(sin_fallas)]
    cmd += ["--completo"] if completo else ["--no-completo"]
    if microredes:
        cmd += ["--microredes"]
    if forzar:
        cmd += ["--forzar"]
    if salida:
        cmd += ["--salida", str(salida)]
    for c in (combos or []):
        cmd += ["--combo", ",".join(c) if not isinstance(c, str) else c]
    if todas is not None:
        cmd += ["--todas"] if todas else ["--no-todas"]
    if combo_escenarios:
        cmd += ["--combo-escenarios", str(combo_escenarios)]
    for k, v in (extra or {}).items():
        cmd += [f"--{k}", str(v)]
    return cmd


def como_texto(cmd):
    """El comando en una línea copiable a la consola."""
    partes = []
    for c in cmd:
        partes.append(f'"{c}"' if " " in c else c)
    return " ".join(partes)


def esta_corriendo(salida):
    """(True, info) si hay un barrido vivo para esa carpeta.

    Si el lock existe pero el proceso ya murió (corte de luz, kill -9), se
    limpia solo: un lock huérfano bloquearía la carpeta para siempre.
    """
    lock = ruta_lock(salida)
    if not lock.exists():
        return False, None
    try:
        info = json.loads(lock.read_text(encoding="utf-8"))
        pid = int(info["pid"])
    except Exception:
        lock.unlink(missing_ok=True)
        return False, None

    if not psutil.pid_exists(pid):
        lock.unlink(missing_ok=True)
        return False, info
    try:
        proc = psutil.Process(pid)
        # El PID pudo reciclarse en otro programa: se compara la hora de
        # arranque que se guardó al lanzar.
        if abs(proc.create_time() - float(info.get("create_time", 0))) > 1.0:
            lock.unlink(missing_ok=True)
            return False, info
    except psutil.Error:
        lock.unlink(missing_ok=True)
        return False, info
    return True, info


def lanzar(cmd, salida):
    """Arranca el barrido desacoplado de la app y deja el lock escrito."""
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)

    vivo, _ = esta_corriendo(salida)
    if vivo:
        raise RuntimeError("Ya hay un barrido en curso para esta carpeta.")

    log = ruta_log(salida)
    f = open(log, "a", encoding="utf-8", buffering=1)
    f.write(f"\n{'='*70}\n{time.strftime('%Y-%m-%d %H:%M:%S')}  {como_texto(cmd)}\n{'='*70}\n")

    # En Windows hay que desacoplar explícitamente para que el proceso
    # sobreviva a que se cierre la app; en POSIX basta con start_new_session.
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                   | subprocess.DETACHED_PROCESS)
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                            cwd=str(DATOS_DIR), **kwargs)
    try:
        create_time = psutil.Process(proc.pid).create_time()
    except psutil.Error:
        create_time = 0.0

    ruta_lock(salida).write_text(json.dumps({
        "pid": proc.pid,
        "create_time": create_time,
        "comando": cmd,
        "inicio": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inicio_ts": time.time(),
        "log": log.name,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return proc.pid


def detener(salida):
    """Termina el barrido en curso. Lo ya calculado queda en disco: el script
    es reanudable, así que relanzarlo sigue donde iba."""
    vivo, info = esta_corriendo(salida)
    if not vivo:
        return False
    try:
        proc = psutil.Process(int(info["pid"]))
        for hijo in proc.children(recursive=True):    # Gurobi puede ser hijo
            hijo.terminate()
        proc.terminate()
        proc.wait(timeout=10)
    except psutil.TimeoutExpired:
        proc.kill()
    except psutil.Error:
        pass
    ruta_lock(salida).unlink(missing_ok=True)
    return True


def progreso(salida, medidas, escenarios):
    """Cuántas corridas de fase 1 ya están en disco.

    Se cuenta el .json de cada corrida, que el script escribe apenas termina:
    no hace falta que el proceso reporte nada.
    """
    salida = Path(salida)
    hechas = sum(1 for m in medidas
                 for e in escenarios
                 if (salida / "individual" / m / f"esc_{e:03d}.json").exists())
    return hechas, len(medidas) * len(escenarios)


# Ruido que mete streamlit al importarse fuera de la app (data_loader entra por
# helpers_inversion y por grafo). No aporta nada al seguimiento del barrido.
_RUIDO = ("ScriptRunContext", "MemoryCacheStorageManager", "No runtime found",
          "missing ScriptRunContext")


def cola_log(salida, n=40, filtrar=True):
    """Últimas `n` líneas del log, para ver en qué va sin abrir el archivo."""
    log = ruta_log(salida)
    if not log.exists():
        return ""
    try:
        lineas = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    if filtrar:
        lineas = [l for l in lineas if not any(r in l for r in _RUIDO)]
    return "\n".join(lineas[-n:])
