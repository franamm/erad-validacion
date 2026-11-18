"""Exporta las referencias históricas de app.pkl sin ejecutar el solver.

Uso: python importar_referencia_app_pkl.py
El destino tiene etiqueta APP-PKL y nunca se sobrescribe.
"""
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

import pipeline as P


def main():
    origen = Path(__file__).resolve().parent / "app.pkl"
    destino = P.ruta_sin_fallas("APP-PKL")
    if destino.exists():
        raise SystemExit(f"Ya existe {destino}; no se sobrescribe.")
    with origen.open("rb") as f:
        firma = hashlib.file_digest(f, "sha256").hexdigest()
        f.seek(0)
        data = pickle.load(f)

    claves = ["P_SUB_tot_sin_fallas"] + [
        f"{costo}_DG_{tec}_sin_fallas"
        for tec in ("sol", "eol") for costo in ("CAPEX", "OPEX")
    ]
    salida = {k: data[k] for k in claves}
    for k in claves:
        if not np.isfinite(salida[k]):
            raise ValueError(f"Referencia no finita: {k}")
    buses = list(data["net"].bus.index)
    escenarios = {}
    for sufijo, tec in (("", None), ("_DG_sol", "sol"), ("_DG_eol", "eol")):
        resultados = data[f"results_sin_fallas{sufijo}"]
        if not isinstance(resultados, dict) or len(resultados) != 1:
            raise ValueError(f"Se esperaba un único escenario histórico: {sufijo}")
        esc, horas = next(iter(resultados.items()))
        if len(horas) != 8760 or not all(isinstance(h, dict) and h for h in horas):
            raise ValueError(f"Serie histórica incompleta: {sufijo}")
        for h in horas:
            for campo, indices in (("P_SUB", P.S), ("P_LC", buses)):
                if any(i not in h.get(campo, {}) for i in indices):
                    raise ValueError(f"Faltan valores de {campo}: {sufijo}")
        if tec:
            # El código histórico usaba 0.3 por barra. Las capacidades
            # guardadas saturan ese límite en ambas tecnologías.
            capacidades = horas[0][f"CAP_DG_{tec}_kW"]
            for b in buses:
                if not np.isclose(capacidades[b], 0.3 * data["P_D_base"][b]):
                    raise ValueError("Las capacidades no respaldan el tope histórico 0.3 por barra.")
        slim = P._slim_sin_fallas(horas, buses, tec)
        for campo in ("P_SUB", "P_LC"):
            if not np.isfinite(slim[campo]).all():
                raise ValueError(f"Serie no finita: {sufijo}/{campo}")
        salida[f"sin_fallas{sufijo}"] = slim
        escenarios[sufijo or "base"] = int(esc)
    if not np.isclose(salida["sin_fallas"]["P_SUB"].sum(),
                      salida["P_SUB_tot_sin_fallas"], rtol=1e-12):
        raise ValueError("El total de subestación no coincide con la serie histórica.")
    salida["_meta"] = {
        "etiqueta": "APP-PKL", "generado": datetime.now().isoformat(timespec="seconds"),
        "horas": 8760, "lim_dg": 0.3, "lim_dg_modo": "bus", "lim_dg_bus": None,
        "origen": "app.pkl", "origen_sha256": firma,
        "escenarios_origen": escenarios,
        "lim_dg_evidencia": "Default histórico 0.3 por barra, corroborado por capacidades guardadas.",
        "referencia_conjunta_disponible": False,
        "nota": "Valores históricos copiados sin recalcular. No contiene una solución solar y eólica conjunta.",
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("xb") as f:
        pickle.dump(salida, f, protocol=pickle.HIGHEST_PROTOCOL)
    recargado = P.cargar_sin_fallas("APP-PKL")
    for k in claves:
        if recargado[k] != data[k]:
            raise AssertionError(f"La referencia cambió durante la exportación: {k}")
    print(destino)
    print(json.dumps({k: float(recargado[k]) for k in claves}, indent=2))


if __name__ == "__main__":
    main()
