"""Avance de un cálculo en segundo plano, leído de disco y del registro.

Sin Streamlit, para probarlo con registros de ejemplo. Nada de esto le pide
algo al proceso que calcula: solo mira lo que ya escribió.

- Avance y resultados preliminares: los `.json` esperados que ya existen
  (los scripts guardan cada corrida apenas termina).
- Corrida en curso: la primera esperada que todavía no existe.
- Cronómetro: tiempo desde el último `.json` guardado (o desde el inicio).
- Solver: última línea de avance de Gurobi en el registro (tee=True).
"""
import re
import time
from pathlib import Path

from estudio_v8.core import leer_json

# Línea del árbol de Gurobi: "... Incumbent BestBd Gap% It/Node Time"
# p. ej. "H    0     0                    -9.870e+05 -1.021e+06  3.45%     -    5s"
_NUM = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
RE_NODO = re.compile(rf"(?P<inc>{_NUM})\s+(?P<cota>{_NUM})\s+(?P<gap>\d+(?:\.\d+)?)%\s+\S+\s+(?P<t>\d+)s\s*$")
RE_FINAL = re.compile(rf"Best objective (?P<inc>{_NUM}), best bound (?P<cota>{_NUM}), gap (?P<gap>\d+(?:\.\d+)?)%")
# Líneas de los scripts ([barrido], [pipeline], [v8]…); Gurobi nunca empieza con "[".
RE_SCRIPT = re.compile(r"^\[[a-z0-9_\-]+\]")


def cola(ruta, n_bytes=60000):
    """Últimos `n_bytes` del registro, sin cargar el archivo entero."""
    ruta = Path(ruta)
    if not ruta.exists():
        return ""
    with ruta.open("rb") as f:
        f.seek(max(0, ruta.stat().st_size - n_bytes))
        return f.read().decode("utf-8", errors="replace")


def ultimo_solver(texto):
    """{incumbente, cota, gap_pct, segundos, final} de la última línea de Gurobi, o None."""
    for linea in reversed(texto.splitlines()):
        m = RE_FINAL.search(linea)
        if m:
            return {"incumbente": float(m["inc"]), "cota": float(m["cota"]),
                    "gap_pct": float(m["gap"]), "segundos": None, "final": True}
        m = RE_NODO.search(linea)
        if m:
            return {"incumbente": float(m["inc"]), "cota": float(m["cota"]),
                    "gap_pct": float(m["gap"]), "segundos": int(m["t"]), "final": False}
        if RE_SCRIPT.match(linea.strip()):
            # Una línea del script posterior a la última de Gurobi significa
            # que ese solve ya terminó: no mostrar un gap de la corrida anterior.
            return None
    return None


def ultimas_lineas_script(texto, n=6):
    return [l for l in texto.splitlines() if RE_SCRIPT.match(l.strip())][-n:]


def avance(esperados, inicio, ahora=None):
    """Estado de las corridas esperadas y estimación de tiempo."""
    ahora = ahora or time.time()
    rutas = [Path(p) for p in esperados]
    hechas = [p for p in rutas if p.exists()]
    registros = [leer_json(p) for p in hechas]
    optimas = sum(r.get("status") == "optimal" for r in registros)
    recientes = [p.stat().st_mtime for p in hechas if p.stat().st_mtime >= (inicio or 0)]
    desde = max(recientes) if recientes else (inicio or ahora)
    actual = next((p for p in rutas if not p.exists()), None)
    duraciones = [float(r["segundos"]) for r in registros if r.get("segundos")]
    promedio = sum(duraciones) / len(duraciones) if duraciones else None
    pendientes = len(rutas) - len(hechas)
    return {"total": len(rutas), "hechas": len(hechas), "optimas": optimas,
            "pendientes": pendientes, "actual": str(actual) if actual else None,
            "en_curso_s": max(0.0, ahora - desde), "promedio_s": promedio,
            "restante_s": promedio * pendientes if promedio is not None else None}


def formato_duracion(s):
    if s is None:
        return "—"
    s = int(round(s))
    h, m = divmod(s, 3600)
    m, s = divmod(m, 60)
    return f"{h} h {m:02d} min" if h else (f"{m} min {s:02d} s" if m else f"{s} s")
