"""Trabajos persistentes con un bloqueo exclusivo por carpeta de salida."""
import json
import os
import subprocess
import sys
import time
import hashlib
import threading
from pathlib import Path

from estudio_v8.core import APP_DIR, leer_json
from rutas import DATOS_DIR  # datos del caso activo (ver rutas.py)
import casos


def guardar(path, data):
    path = Path(path)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def vivo(info):
    import psutil
    try:
        proc = psutil.Process(info["pid"])
        return abs(proc.create_time() - info["create_time"]) < 1 and proc.is_running()
    except (psutil.Error, KeyError):
        return False


def estado(salida):
    path = Path(salida) / "trabajo_v8.json"
    info = leer_json(path)
    if info.get("estado") == "ejecutando" and not vivo(info):
        info["estado"] = "interrumpido"
    return info


def lanzar(cmd, salida, config, esperados=()):
    salida = Path(salida)
    art_dir = casos.resolver_ruta(config.get("artefactos") or casos.art_dir_activo())
    config = {**config, "artefactos": str(art_dir)}
    salida.mkdir(parents=True, exist_ok=True)
    lock = salida / ".trabajo_v8.lock"
    # Bloquear también el lanzador histórico si usa esta salida.
    if vivo(leer_json(salida / "barrido_en_curso.json")):
        raise RuntimeError("La versión anterior está escribiendo en esta carpeta.")
    if lock.exists():
        info = estado(salida)
        if vivo(info) or time.time() - lock.stat().st_mtime < 15:
            raise RuntimeError("Ya hay un trabajo en esta carpeta.")
        lock.unlink()
    with lock.open("x", encoding="utf-8") as f:
        f.write(str(time.time()))
    try:
        if "medidas" in config:
            # La misma carpeta no puede reutilizar resultados de otros insumos.
            snapshot = {k: config.get(k) for k in ("seccionadores", "horas", "e4", "lim_dg", "lim_dg_modo", "extras", "microredes")}
            archivos = [art_dir / n for n in ("red.pkl", "perfiles.npz", "fallas.npz", "costos.pkl")]
            archivos += [art_dir / (f"sin_fallas_{config['e4']}.pkl" if config['e4'] else "sin_fallas.pkl")]
            snapshot["insumos_sha256"] = {}
            for archivo in archivos:
                with archivo.open("rb") as f:
                    snapshot["insumos_sha256"][archivo.name] = hashlib.file_digest(f, "sha256").hexdigest()
            manifiesto = salida / "estudio_v8.json"
            anterior = leer_json(manifiesto)
            if anterior and anterior != snapshot:
                raise ValueError("Los insumos o supuestos cambiaron. Usa otro nombre de estudio para conservar los resultados previos.")
            guardar(manifiesto, snapshot)
    except Exception:
        lock.unlink(missing_ok=True)
        raise
    meta = salida / "trabajo_v8.json"
    # `cmd` es un comando o una lista de comandos que el supervisor encadena.
    comandos = [list(c) for c in cmd] if cmd and isinstance(cmd[0], (list, tuple)) else [list(cmd)]
    info = {"estado": "iniciando", "inicio": time.time(), "comando": comandos[0],
            "comandos": comandos, "config": config, "esperados": [str(p) for p in esperados]}
    guardar(meta, info)
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
    try:
        with (salida / "trabajo_v8.log").open("a", encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable, "-m", "estudio_v8.worker", str(meta)],
                             cwd=DATOS_DIR, stdout=log, stderr=subprocess.STDOUT,
                             env={**os.environ, casos.VAR_ENTORNO: str(art_dir), "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)},
                             creationflags=flags, start_new_session=os.name != "nt")
            threading.Thread(target=proc.wait, daemon=True).start()
    except Exception:
        lock.unlink(missing_ok=True)
        raise


def detener(salida):
    import psutil
    info = estado(salida)
    if not vivo(info):
        return
    proc = psutil.Process(info["pid"])
    hijos = proc.children(recursive=True)
    for p in hijos:
        try:
            p.terminate()
        except psutil.Error:
            pass
    proc.terminate()
    _, pendientes = psutil.wait_procs(hijos + [proc], timeout=3)
    for p in pendientes:
        try:
            p.kill()
        except psutil.Error:
            pass
    info.update(estado="detenido", fin=time.time())
    guardar(Path(salida) / "trabajo_v8.json", info)
    (Path(salida) / ".trabajo_v8.lock").unlink(missing_ok=True)
