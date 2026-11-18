"""Cliente mínimo de api.minenergia.cl (Exploradores Solar y Eólico del Ministerio de Energía).

Lee EXPLORADOR_USUARIO / EXPLORADOR_CLAVE del `.env` de la raíz del repo (ignorado por git) sin
imprimirlos. Las consultas van a POST /api/proxy con el JSON que arma la interfaz web
(`export.format = "json"` devuelve los datos en la respuesta).
"""
import pathlib
import re
import sys

import requests

BASE = "https://api.minenergia.cl"


def credenciales(ruta=pathlib.Path(__file__).resolve().parents[4] / ".env"):
    env = {}
    for linea in ruta.read_text(encoding="utf-8-sig").splitlines():
        if "=" in linea and not linea.strip().startswith("#"):
            k, v = linea.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    faltan = [k for k in ("EXPLORADOR_USUARIO", "EXPLORADOR_CLAVE") if not env.get(k)]
    if faltan:
        sys.exit(f"faltan en .env: {faltan} (claves presentes: {sorted(env)})")
    return env["EXPLORADOR_USUARIO"], env["EXPLORADOR_CLAVE"]


def sesion_web():
    """Sesión del sitio (formulario de ingreso + CSRF), la que usa la interfaz web."""
    u, c = credenciales()
    s = requests.Session()
    r = s.get(f"{BASE}/login/", timeout=60)
    tok = s.cookies.get("csrftoken") or re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text).group(1)
    r = s.post(f"{BASE}/login/", data={"username": u, "password": c, "csrfmiddlewaretoken": tok},
               headers={"Referer": f"{BASE}/login/"}, timeout=60)
    if "sessionid" not in s.cookies:
        sys.exit(f"ingreso web falló: HTTP {r.status_code}")
    s.get(f"{BASE}/api/", timeout=60)
    s.headers.update({"X-CSRFToken": s.cookies.get("csrftoken", tok), "Referer": f"{BASE}/api/"})
    return s
