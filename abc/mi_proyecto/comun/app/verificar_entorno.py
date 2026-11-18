# ============================================================
# verificar_entorno.py — ¿Está este equipo listo para correr la cadena?
# ============================================================
# Pensado para correrlo en un servidor recién instalado, ANTES de lanzar
# horas de MILP: cada falla de acá es una corrida que habría muerto a mitad
# de camino. Revisa, en orden:
#
#   1. Python y paquetes (versión contra servidor/requirements_linux.txt);
#   2. la licencia de Gurobi, resolviendo un MILP por sobre el límite de la
#      licencia restringida (2000 variables) — "importa" no es "resuelve";
#   3. las dos interfaces de Pyomo que usa el código: gurobi_direct
#      (optimization_6_oficial) y "gurobi" por archivo LP (optimization_base,
#      E4/E5), y QUÉ ejecutable usa la segunda;
#   4. los recursos que ve el proceso (recursos.py) y el espacio en disco de
#      los temporales;
#   5. que los módulos de la cadena importen;
#   6. con --red, que se llegue a las APIs (Open-Meteo, ARClim, Gurobi WLS).
#
# USO
#   python verificar_entorno.py          # todo menos la red
#   python verificar_entorno.py --red    # también conectividad
# ============================================================

import argparse
import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from rutas import DATOS_DIR                      # datos del caso activo (ERAD_CASO)
REQ = APP_DIR / "servidor" / "requirements_linux.txt"

for _flujo in (sys.stdout, sys.stderr):      # ver pipeline._forzar_utf8_en_consola
    try:
        _flujo.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

FALLAS, AVISOS = [], []


def ok(msg):
    print(f"  [ok]    {msg}")


def aviso(msg):
    AVISOS.append(msg)
    print(f"  [aviso] {msg}")


def falla(msg):
    FALLAS.append(msg)
    print(f"  [FALLA] {msg}")


def seccion(titulo):
    print(f"\n── {titulo} " + "─" * max(0, 60 - len(titulo)))


def requeridos():
    out = {}
    if REQ.exists():
        for linea in REQ.read_text(encoding="utf-8").splitlines():
            linea = linea.split("#")[0].strip()
            if "==" in linea:
                nombre, ver = linea.split("==")
                out[nombre.strip()] = ver.strip()
    return out


def paquetes():
    seccion("1. Python y paquetes")
    v = sys.version_info
    (ok if (v.major, v.minor) == (3, 11) else aviso)(
        f"Python {v.major}.{v.minor}.{v.micro} en {sys.executable}"
        + ("" if (v.major, v.minor) == (3, 11) else " (el estudio corrió en 3.11)"))
    from importlib.metadata import PackageNotFoundError, version
    for nombre, esperada in requeridos().items():
        try:
            real = version(nombre)
        except PackageNotFoundError:
            falla(f"{nombre} no está instalado (pip install -r {REQ.relative_to(APP_DIR)})")
            continue
        (ok if real == esperada else aviso)(
            f"{nombre} {real}" + ("" if real == esperada else f" (fijada: {esperada})"))


def licencia():
    seccion("2. Licencia de Gurobi")
    lic = os.environ.get("GRB_LICENSE_FILE") or str(Path.home() / "gurobi.lic")
    if Path(lic).exists():
        ok(f"archivo de licencia: {lic}")
    elif not os.environ.get("GRB_LICENSE_FILE"):
        aviso(f"no hay {lic} ni GRB_LICENSE_FILE: gurobipy usará la licencia de "
              f"prueba (2000 variables), que no alcanza ni para E4")
    try:
        import gurobipy as gp
    except ImportError:
        falla("gurobipy no importa")
        return
    try:
        env = gp.Env(params={"OutputFlag": 0})
        m = gp.Model(env=env)
        x = m.addVars(3000, vtype=gp.GRB.BINARY)
        m.addConstr(x.sum() <= 10)
        m.setObjective(x.sum(), gp.GRB.MAXIMIZE)
        m.optimize()
        if m.Status == gp.GRB.OPTIMAL and abs(m.ObjVal - 10) < 1e-6:
            ok(f"Gurobi {'.'.join(map(str, gp.gurobi.version()))} resolvió un MILP "
               f"de 3000 binarias: licencia completa")
        else:
            falla(f"el MILP de prueba terminó con status {m.Status}")
        m.dispose()
        env.dispose()
    except gp.GurobiError as e:
        falla(f"Gurobi no resuelve: {e}. Con licencia WLS, el servidor necesita "
              f"salida a internet hacia token.gurobi.com (puerto 443)")


def pyomo_interfaces():
    seccion("3. Pyomo → Gurobi")
    try:
        import pyomo.environ as pyo
    except ImportError:
        falla("pyomo no importa")
        return
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], domain=pyo.Binary)
    m.o = pyo.Objective(expr=m.x[1] + m.x[2], sense=pyo.maximize)
    for nombre, quien in (("gurobi_direct", "optimization_6_oficial / barrido"),
                          ("gurobi", "optimization_base / E4-E5")):
        try:
            s = pyo.SolverFactory(nombre)
            if not s.available(exception_flag=False):
                falla(f"'{nombre}' no disponible ({quien})")
                continue
            r = s.solve(m)
            term = str(r.solver.termination_condition)
            (ok if term == "optimal" else falla)(f"'{nombre}' resuelve: {term} ({quien})")
        except Exception as e:                                  # noqa: BLE001
            falla(f"'{nombre}': {type(e).__name__}: {e}")
    # La interfaz por archivo LP busca primero gurobi.sh en el PATH. Si el
    # servidor tiene un Gurobi de sistema, E4/E5 corren con ESE (otra versión,
    # otra licencia) y el barrido con el de pip, sin que nada lo diga.
    exe = shutil.which("gurobi.sh") or shutil.which("gurobi.bat")
    if exe:
        aviso(f"hay un Gurobi de sistema en el PATH ({exe}): E4/E5 lo usarán en vez "
              f"del gurobipy de pip. Sacarlo del PATH en el trabajo o verificar que "
              f"sea la misma versión")
    else:
        ok("sin gurobi.sh en el PATH: E4/E5 usan el gurobipy de este entorno")


def recursos_y_disco():
    seccion("4. Recursos y disco")
    import recursos
    ok(recursos.resumen())
    ram = recursos.ram_disponible_gb()
    if ram < 32:
        aviso(f"{ram:.0f} GB de RAM: un MILP de 8760 h sobre una red de ~180 barras "
              f"con todas las medidas pidió más de 64 GB en Oberrhein")
    for nombre, ruta in (("TMPDIR", tempfile.gettempdir()),
                         ("artefactos", str(DATOS_DIR / "artefactos")),
                         ("nodefiles", os.environ.get("ERAD_GUROBI_NODEFILE_DIR", ""))):
        if not ruta:
            continue
        p = Path(ruta)
        while not p.exists() and p != p.parent:
            p = p.parent
        libre = shutil.disk_usage(p).free / 1024 ** 3
        (ok if libre >= 50 else aviso)(f"{nombre}: {ruta} — {libre:.0f} GB libres"
                                       + ("" if libre >= 50 else
                                          " (un LP de 8760 h y los nodefiles pueden "
                                          "ocupar decenas de GB)"))


def modulos():
    seccion("5. Módulos de la cadena")
    sys.path.insert(0, str(APP_DIR))
    for mod in ("casos", "entrada_red", "pipeline", "aislacion", "limites_dg",
                "indicadores", "optimization_base", "optimization_6_oficial",
                "construir_faultlin", "construir_perfiles", "fallas_sinteticas",
                "seleccionar_escenarios", "crear_caso"):
        try:
            importlib.import_module(mod)
            ok(mod)
        except SystemExit:
            ok(mod)
        except Exception as e:                                  # noqa: BLE001
            falla(f"{mod}: {type(e).__name__}: {e}")


def red():
    seccion("6. Conectividad")
    import requests
    for nombre, url in (("Open-Meteo (clima, perfiles)", "https://archive-api.open-meteo.com/v1/archive"),
                        ("ARClim (delta climático)", "https://arclim.mma.gob.cl/"),
                        ("Gurobi WLS (licencia)", "https://token.gurobi.com/")):
        try:
            r = requests.get(url, timeout=15)
            ok(f"{nombre}: HTTP {r.status_code}")
        except Exception as e:                                  # noqa: BLE001
            (falla if "Gurobi" in nombre else aviso)(
                f"{nombre}: sin acceso ({type(e).__name__}). "
                + ("La licencia WLS no se renueva sin esto." if "Gurobi" in nombre else
                   "Correr las etapas clima/perfiles en otro nodo y copiar el .nc."))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--red", action="store_true", help="probar también la conectividad")
    args = ap.parse_args()
    paquetes()
    licencia()
    pyomo_interfaces()
    recursos_y_disco()
    modulos()
    if args.red:
        red()
    print()
    if FALLAS:
        print(f"{len(FALLAS)} falla(s), {len(AVISOS)} aviso(s). Corregir las fallas antes de lanzar MILP.")
        sys.exit(1)
    print(f"Listo para correr. {len(AVISOS)} aviso(s).")


if __name__ == "__main__":
    main()
