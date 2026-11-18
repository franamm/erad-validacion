"""Contratos del estudio. Sin Streamlit ni ejecución de optimizadores al importar."""
from pathlib import Path
import json
import re
import sys

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
import casos  # noqa: E402  (necesita APP_DIR en sys.path)

# Caso activo del proceso (ver casos.py). La v8 trabaja siempre contra el caso
# con el que arrancó la app: cambiarlo en caliente movería la carpeta bajo los
# pies de data_loader, que ya leyó sus artefactos al importarse.
ART_DIR = casos.art_dir_activo()
MEDIDAS_TESIS = (
    "reconfig_topo", "dg_sol", "dg_eol", "dg_fos", "ess_4h", "ess_5h",
    "ess_6h", "meg", "mess", "edrp", "refuerzo", "soterramiento",
)
COMBINACION_TESIS = "combo_" + "+".join(MEDIDAS_TESIS)


def leer_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def indices(texto):
    valores = set()
    for parte in texto.replace(" ", "").split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", parte):
            raise ValueError("Usa índices no negativos: 0,3,7 o 0-9.")
        extremos = [int(x) for x in parte.split("-")]
        if len(extremos) == 2:
            if extremos[1] < extremos[0] or extremos[1] - extremos[0] > 10000:
                raise ValueError("Rango de escenarios inválido.")
            valores.update(range(extremos[0], extremos[1] + 1))
        else:
            valores.add(extremos[0])
    return sorted(valores)


def salida_estudio(nombre):
    if not re.fullmatch(r"[\w .+-]+", nombre) or nombre in (".", ".."):
        raise ValueError("El nombre debe ser una carpeta, sin separadores de ruta.")
    return ART_DIR / "barrido" / nombre


def comando_barrido(config):
    medidas, escenarios = config["medidas"], config["escenarios"]
    if not medidas or not escenarios:
        raise ValueError("Selecciona medidas y escenarios.")
    modo = config["modo"]
    fases = {"Individuales": "1,2", "Combinación": "3",
             "Corrida puntual": "3", "Estudio completo": "1,2,3"}[modo]
    if modo == "Corrida puntual" and len(escenarios) != 1:
        raise ValueError("La corrida puntual necesita exactamente un escenario.")
    cmd = [sys.executable, str(APP_DIR / "run_barrido_medidas.py"),
           "--medidas", ",".join(medidas), "--escenarios", ",".join(map(str, escenarios)),
           "--fases", fases, "--salida", str(salida_estudio(config["nombre"])),
           "--no-todas", "--completo", "--criterio", config["criterio"]]
    for clave, flag in (("horas", "horas"), ("lim_dg", "lim-dg"),
                        ("lim_dg_modo", "lim-dg-modo"), ("e4", "sin-fallas"),
                        ("mip_gap", "mip_gap"), ("time_limit", "time_limit")):
        cmd += ["--" + flag, str(config[clave])]
    if config["seccionadores"]:
        cmd += ["--seccionadores", ",".join(map(str, config["seccionadores"]))]
    if modo != "Individuales":
        cmd += ["--combo", ",".join(medidas),
                "--combo-escenarios", ",".join(map(str, escenarios))]
    for clave, valor in config.get("extras", {}).items():
        cmd += ["--" + clave, str(valor)]
    if config.get("microredes"):
        cmd += ["--microredes"]
    return cmd


def esperadas(config):
    root = salida_estudio(config["nombre"])
    rutas = []
    if config["modo"] in ("Individuales", "Estudio completo"):
        rutas += [root / "individual" / m / f"esc_{e:03d}.json"
                  for m in config["medidas"] for e in config["escenarios"]]
    if config["modo"] != "Individuales":
        combo = "combo_" + "+".join(config["medidas"])
        rutas += [root / "combinadas" / combo / f"esc_{e:03d}.json"
                  for e in config["escenarios"]]
    return rutas


def miembros(combo, registros=()):
    for reg in registros:
        if reg.get("medidas_on"):
            return list(reg["medidas_on"])
    return combo.removeprefix("combo_").split("+")


def reglas(matriz, beneficio=False, pesos=None):
    """Criterios sobre columnas completas, referencia = mejor candidato por columna."""
    import numpy as np
    import pandas as pd
    M = matriz.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    M = M.dropna(axis=1, how="any")
    if M.empty:
        raise ValueError("No hay escenarios evaluados para todos los planes.")
    w = pd.Series(1.0, index=M.columns) if pesos is None else pd.Series(pesos).reindex(M.columns)
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Los pesos deben ser no negativos y cubrir todos los escenarios.")
    w = w / w.sum()
    R = M.max(axis=0) - M if beneficio else M - M.min(axis=0)
    out = pd.DataFrame({"Valor esperado": M.mul(w, axis=1).sum(axis=1),
                        "Mínimo": M.min(axis=1), "Máximo": M.max(axis=1),
                        "Arrepentimiento máximo": R.max(axis=1)})
    columnas = {"Valor esperado": "Valor esperado",
                "Maximin" if beneficio else "Minimax": "Mínimo" if beneficio else "Máximo",
                "Maximax" if beneficio else "Minimin": "Máximo" if beneficio else "Mínimo",
                "Minimax regret": "Arrepentimiento máximo"}
    elegidos = {}
    for regla, col in columnas.items():
        s = out[col]
        mejor = s.max() if beneficio and regla != "Minimax regret" else s.min()
        elegidos[regla] = list(s.index[np.isclose(s, mejor, rtol=1e-9, atol=1e-9)])
    return out, R, elegidos, list(M.columns)


def rangos(registros):
    """Incluye ceros de instalaciones ausentes al comparar planes."""
    import pandas as pd
    claves = {(k, str(i)) for r in registros for k, d in r.get("plan", {}).items() for i in d}
    filas = []
    for k, i in sorted(claves):
        vals = pd.Series([float(r.get("plan", {}).get(k, {}).get(i, 0)) for r in registros])
        filas.append({"Variable": k, "Ubicación": i, "Mínimo": vals.min(),
                      "Mediana": vals.median(), "Máximo": vals.max(),
                      "Frecuencia de instalación": float((vals > 1e-6).mean()),
                      "Planes": len(registros)})
    return pd.DataFrame(filas)


def matriz_planes(regs, planes, escenarios, salida):
    """Conservar calidad y desvío de inversión junto con cada celda."""
    import pandas as pd
    filas = []
    for i in planes:
        for j in escenarios:
            reg = regs.get(j, {}) if i == j else leer_json(Path(salida) / f"plan_esc_{i:03d}" / f"esc_{j:03d}.json")
            if reg.get("status") != "optimal":
                continue
            filas.append({**{k: v for k, v in reg.items() if not isinstance(v, (dict, list))},
                          "plan": i, "escenario": j, "origen": "barrido" if i == j else "plan fijo"})
    return pd.DataFrame(filas)
