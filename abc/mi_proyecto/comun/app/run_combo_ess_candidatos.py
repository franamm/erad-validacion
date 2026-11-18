"""Barrido con el ESS restringido a una lista de barras candidatas.

En el modelo vigente el ESS es candidato en TODAS las barras (`ESS = BUSES`):
179 cadenas de SOC de 8760 h, que es lo que agota la memoria en los combos
grandes. Este script NO edita optimization_6_oficial.py: carga su código,
le aplica en memoria el filtro de candidatas y corre run_barrido_medidas
con esa versión. Si el modelo oficial cambia y algún reemplazo ya no calza,
falla en vez de correr un modelo distinto al esperado.

Las candidatas salen de una corrida ya guardada (por defecto, las barras
donde la corrida individual de ess_5h instaló capacidad). Todos los demás
argumentos pasan tal cual a run_barrido_medidas. Usar SIEMPRE una --salida
propia, para no mezclar estas corridas con las del modelo vigente.

    python run_combo_ess_candidatos.py \\
        --ess-desde artefactos/oberrhein/barrido/individual-esc0/individual/ess_5h/esc_000.json \\
        --salida artefactos/oberrhein/barrido/individual-esc0-ess-cand ...
"""
import argparse
import json
import re
import sys
import types
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODELO = APP_DIR / "optimization_6_oficial.py"

# (texto original, texto nuevo, apariciones esperadas en el modelo oficial)
_REEMPLAZOS = [
    ("    ESS   = BUSES\n",
     "    ESS   = BUSES\n"
     "    _cand = config.get('ess_candidatos')\n"
     "    ESS_CAND = [e for e in BUSES if e in set(_cand)] if _cand else BUSES\n",
     1),
    ("= ESS  if usar_ess_", "= ESS_CAND  if usar_ess_", 3),
    # La extracción recorre todas las barras; las no candidatas no tienen
    # variable y se reportan en cero.
    ("            fn_objetivo   = pyo.value(model.obj)\n",
     "            def _val_ess(var, idx):\n"
     "                return _val(var[idx]) if idx in var else 0.0\n"
     "\n"
     "            fn_objetivo   = pyo.value(model.obj)\n",
     1),
]
_EXTRACCION = re.compile(
    r"_val\(model\.((?:P_discharge|P_charge|SOC|CAP_ESS)_[456]h)\[([^\]]+)\]\)")
_EXTRACCION_N = 12


def _modelo_parchado():
    src = MODELO.read_text(encoding="utf-8")
    for viejo, nuevo, n in _REEMPLAZOS:
        hallados = src.count(viejo)
        if hallados != n:
            raise SystemExit(f"optimization_6_oficial cambió: se esperaban {n} "
                             f"apariciones de {viejo.strip()!r} y hay {hallados}.")
        src = src.replace(viejo, nuevo)
    src, hallados = _EXTRACCION.subn(r"_val_ess(model.\1, (\2))", src)
    if hallados != _EXTRACCION_N:
        raise SystemExit(f"optimization_6_oficial cambió: se esperaban "
                         f"{_EXTRACCION_N} lecturas de ESS y hay {hallados}.")
    mod = types.ModuleType("optimization_6_oficial")
    mod.__file__ = str(MODELO)
    exec(compile(src, str(MODELO), "exec"), mod.__dict__)
    return mod


def _candidatas(ruta, top, umbral_kwh=1e-3):
    reg = json.loads(Path(ruta).read_text(encoding="utf-8"))
    cap = {}
    for clave in ("CAP_ESS_kWh_5h", "CAP_ESS_kWh_4h", "CAP_ESS_kWh_6h"):
        for b, v in ((reg.get("plan") or {}).get(clave) or {}).items():
            if v and v > umbral_kwh:
                cap[int(b)] = cap.get(int(b), 0.0) + v
    if not cap:
        raise SystemExit(f"{ruta} no tiene capacidad de ESS instalada.")
    orden = sorted(cap, key=cap.get, reverse=True)
    return sorted(orden[:top] if top else orden), cap


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--ess-desde", required=True,
                    help=".json de una corrida guardada; sus barras con ESS > 0 son las candidatas")
    ap.add_argument("--ess-top", type=int, default=0,
                    help="quedarse solo con las N barras de mayor capacidad (0 = todas)")
    args, resto = ap.parse_known_args()

    candidatas, cap = _candidatas(args.ess_desde, args.ess_top)
    print(f"[ess-cand] {len(candidatas)} barras candidatas para ESS "
          f"(desde {args.ess_desde}): {candidatas}", flush=True)

    sys.modules["optimization_6_oficial"] = _modelo_parchado()
    import run_barrido_medidas as R

    construir = R.construir_config

    def construir_con_candidatas(*a, **k):
        cfg = construir(*a, **k)
        cfg["ess_candidatos"] = candidatas
        return cfg

    R.construir_config = construir_con_candidatas

    sys.argv = [str(APP_DIR / "run_barrido_medidas.py")] + resto
    R.main()

    if "--listar" not in resto and "--salida" in resto:
        # La misma carpeta en que escribió el barrido (nombre suelto → artefactos
        # del caso/barrido/), no una relativa al código.
        salida = Path(R.resolver_salida(resto[resto.index("--salida") + 1], None))
        salida.mkdir(parents=True, exist_ok=True)
        (salida / "ess_candidatos.json").write_text(json.dumps(
            {"desde": str(args.ess_desde), "top": args.ess_top,
             "barras": candidatas,
             "cap_origen_kWh": {str(b): cap[b] for b in candidatas}},
            indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
