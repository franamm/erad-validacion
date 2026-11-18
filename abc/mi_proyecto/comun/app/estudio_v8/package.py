"""Paquete transportable de ejecución desde insumos ya preparados."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from estudio_v8.core import APP_DIR, ART_DIR, leer_json
from rutas import DATOS_DIR  # datos del caso activo (ver rutas.py)
from estudio_v8.jobs import estado, vivo


def exportar(nombre, completos=True):
    root = (ART_DIR / nombre).resolve()
    if not root.is_relative_to(ART_DIR.resolve()) or not root.is_dir():
        raise ValueError("El estudio debe estar dentro de artefactos.")
    for path in [root, *root.glob("plan_fijo/*")]:
        if vivo(estado(path)):
            raise ValueError("Detén el trabajo del estudio antes de exportar una instantánea consistente.")
    archivos = set(APP_DIR.glob("*.py")) | set(APP_DIR.glob("*.md"))
    archivos |= set(APP_DIR.glob("requirements_v8.txt"))
    archivos |= set((APP_DIR / "estudio_v8").glob("*.py"))
    archivos |= {p for p in root.rglob("*") if p.is_file()
                 and not p.name.endswith((".lock", ".tmp", ".log"))
                 and (completos or not p.name.endswith(".pkl.gz"))}
    archivos |= set(ART_DIR.glob("sin_fallas*.pkl"))
    archivos |= {ART_DIR / n for n in ("red.pkl", "perfiles.npz", "fallas.npz", "costos.pkl", "escenarios_estratificados.json")}
    for ruta in root.glob("**/esc_*.json"):
        r = leer_json(ruta)
        if "escenario" not in r:
            continue
        var = r.get("variante_aislacion", "base")
        base = ART_DIR / "baselines"
        if var != "base":
            base /= var
        archivos |= set(base.glob(f"esc_{r['escenario']:03d}.*"))
    archivos = {p for p in archivos if p.is_file()}
    salida = ART_DIR / "exportaciones_v8" / f"{root.name}.zip"
    salida.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(salida, "w", compression=ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(archivos):
            # el código cuelga de APP_DIR y los datos de DATOS_DIR: en el zip quedan juntos, como antes
            z.write(p, str(p.relative_to(APP_DIR if p.is_relative_to(APP_DIR) else DATOS_DIR)))
        z.writestr("PAQUETE.json", json.dumps({"estudio": nombre, "detalle_completo": completos,
                    "alcance": "Ejecución con artefactos preparados; fuentes climáticas externas no incluidas",
                    "archivos": len(archivos)}, ensure_ascii=False, indent=2))
    return salida
