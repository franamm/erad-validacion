# ============================================================
# recursos.py — Núcleos y RAM que de verdad le tocan a ESTE proceso
# ============================================================
# Los optimizadores dimensionan Gurobi (Threads, SoftMemLimit,
# NodefileStart) según la máquina. En el notebook de Windows "la máquina"
# es todo el equipo, y `os.cpu_count()` / `psutil.virtual_memory().total`
# bastaban. En un servidor Linux compartido no:
#
#   * un trabajo de SLURM, un contenedor o un `taskset` ven TODOS los
#     núcleos del nodo con `os.cpu_count()`, aunque solo tengan asignados
#     unos pocos: Gurobi lanza 16 hilos sobre 4 núcleos y cada hilo guarda
#     su propia copia de la relajación del nodo;
#   * `psutil` reporta la RAM física del nodo, no el límite del cgroup.
#     SoftMemLimit queda en el 60 % de 512 GB cuando el trabajo tiene 64,
#     y al cruzar el límite el kernel mata el proceso con SIGKILL. Eso no
#     es el "Out of memory" ordenado de Gurobi que devuelve el incumbente:
#     es una corrida de horas perdida sin una línea de log.
#
# Este módulo devuelve el MÍNIMO entre todo lo que acota al proceso. No
# importa nada pesado (ni pyomo ni streamlit) para que lo use cualquiera.
#
# Variables de entorno para forzar los valores (ganan sobre todo lo demás):
#   ERAD_CPUS     núcleos a considerar disponibles
#   ERAD_RAM_GB   GB de RAM a considerar disponibles
# ============================================================

import os
from pathlib import Path

_GB = 1024 ** 3
# cgroup v1 escribe un número gigantesco (≈ 2^63) para decir "sin límite".
_SIN_LIMITE = 1 << 60


def _float_env(nombre):
    try:
        v = float(os.environ.get(nombre, ""))
        return v if v > 0 else None
    except ValueError:
        return None


def _leer(ruta):
    try:
        return Path(ruta).read_text().strip()
    except OSError:
        return None


def _rutas_cgroup():
    """Carpetas de cgroup de este proceso, de la propia hacia la raíz.

    Se recorren todas porque el límite puede estar puesto en un ancestro
    (SLURM lo pone en el del trabajo, no en el del paso).
    """
    contenido = _leer("/proc/self/cgroup")
    if not contenido:
        return []
    rutas = []
    for linea in contenido.splitlines():
        partes = linea.split(":", 2)
        if len(partes) != 3:
            continue
        jerarquia, controladores, rel = partes
        if jerarquia == "0":                                   # cgroup v2
            base = Path("/sys/fs/cgroup")
        elif "memory" in controladores.split(",") or "cpu" in controladores.split(","):
            base = Path("/sys/fs/cgroup") / controladores.split(",")[0]
            if not base.exists():
                base = Path("/sys/fs/cgroup") / controladores
        else:
            continue
        p = base / rel.lstrip("/")
        while True:
            rutas.append(p)
            if p == base or p.parent == p:
                break
            p = p.parent
    return rutas


def _limite_ram_cgroup():
    """Límite de memoria del cgroup en bytes, o None si no hay."""
    limites = []
    for d in _rutas_cgroup():
        for nombre in ("memory.max", "memory.limit_in_bytes"):
            txt = _leer(d / nombre)
            if txt and txt != "max":
                try:
                    v = int(txt)
                except ValueError:
                    continue
                if 0 < v < _SIN_LIMITE:
                    limites.append(v)
    return min(limites) if limites else None


def _limite_cpu_cgroup():
    """Núcleos según la cuota de CPU del cgroup (cpu.max), o None."""
    limites = []
    for d in _rutas_cgroup():
        txt = _leer(d / "cpu.max")                              # v2: "cuota periodo"
        if txt:
            cuota, _, periodo = txt.partition(" ")
            if cuota != "max":
                try:
                    limites.append(int(cuota) / int(periodo or 100000))
                except ValueError:
                    pass
        cuota = _leer(d / "cpu.cfs_quota_us")                   # v1
        periodo = _leer(d / "cpu.cfs_period_us")
        if cuota and periodo:
            try:
                if int(cuota) > 0:
                    limites.append(int(cuota) / int(periodo))
            except ValueError:
                pass
    return max(1, int(min(limites))) if limites else None


def _ram_fisica_bytes():
    try:
        import psutil
        return psutil.virtual_memory().total
    except Exception:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        pass
    try:  # Windows sin psutil
        import ctypes

        class _MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MEMSTAT()
        st.dwLength = ctypes.sizeof(_MEMSTAT)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullTotalPhys
    except Exception:
        pass
    return None


def _ram_slurm_bytes():
    """Memoria pedida a SLURM (--mem o --mem-per-cpu), en bytes."""
    por_nodo = _float_env("SLURM_MEM_PER_NODE")                 # MB
    if por_nodo:
        return por_nodo * 1024 ** 2
    por_cpu = _float_env("SLURM_MEM_PER_CPU")
    cpus = _float_env("SLURM_CPUS_ON_NODE") or _float_env("SLURM_CPUS_PER_TASK")
    if por_cpu and cpus:
        return por_cpu * cpus * 1024 ** 2
    return None


def ram_disponible_gb(default=8.0):
    """GB de RAM que este proceso puede usar sin que lo maten.

    Mínimo entre la RAM física, el límite del cgroup y lo pedido a SLURM.
    `ERAD_RAM_GB` lo fija a mano.
    """
    forzado = _float_env("ERAD_RAM_GB")
    if forzado:
        return forzado
    candidatos = [v for v in (_ram_fisica_bytes(), _limite_ram_cgroup(), _ram_slurm_bytes())
                  if v]
    return min(candidatos) / _GB if candidatos else float(default)


def cpus_disponibles(default=8):
    """Núcleos asignados a este proceso (affinity, cgroup, SLURM).

    `ERAD_CPUS` lo fija a mano.
    """
    forzado = _float_env("ERAD_CPUS")
    if forzado:
        return max(1, int(forzado))
    candidatos = []
    try:
        candidatos.append(len(os.sched_getaffinity(0)))          # solo Linux
    except (AttributeError, OSError):
        if os.cpu_count():
            candidatos.append(os.cpu_count())
    for v in (_limite_cpu_cgroup(), _float_env("SLURM_CPUS_PER_TASK")):
        if v:
            candidatos.append(int(v))
    return max(1, min(candidatos)) if candidatos else int(default)


def resumen():
    """Una línea para el log: qué se detectó y de dónde salió."""
    partes = [f"{cpus_disponibles()} nucleos asignados (equipo: {os.cpu_count()})",
              f"{ram_disponible_gb():.1f} GB RAM utilizables"]
    fisica = _ram_fisica_bytes()
    if fisica:
        partes[-1] += f" (fisica: {fisica / _GB:.1f} GB)"
    if os.environ.get("ERAD_CPUS") or os.environ.get("ERAD_RAM_GB"):
        partes.append("forzado por ERAD_CPUS/ERAD_RAM_GB")
    elif os.environ.get("SLURM_JOB_ID"):
        partes.append(f"trabajo SLURM {os.environ['SLURM_JOB_ID']}")
    return ", ".join(partes)


if __name__ == "__main__":
    print(resumen())
