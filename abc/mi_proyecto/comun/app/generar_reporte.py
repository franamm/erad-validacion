"""Reporte del estudio en Word y PDF, desde consola. Solo lee artefactos: no optimiza.

Ejemplo, desde la carpeta del caso:
    cd abc/mi_proyecto/caso_elmelon
    python ../comun/app/generar_reporte.py --estudio "barrido/todas10_sin_ess_v3" --combinacion todas

Escribe en <estudio>/reporte/: Reporte_<caso>_<estudio>.docx y .pdf, las figuras y
manifiesto.json (opciones, archivos leídos y su fecha). Configuración opcional del
estudio en <estudio>/reporte.json (ver estudio_v8/reporte_datos.py).
"""
import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--estudio", required=True, help='Carpeta del estudio dentro de artefactos, p. ej. "barrido/x"')
    ap.add_argument("--combinacion", default="todas", help="Combinación cuyos planes se evalúan (por defecto: todas)")
    ap.add_argument("--plan", default=None, help="Forzar el plan reportado (P4 o número de escenario)")
    ap.add_argument("--formato", default="docx,pdf", help="docx, pdf o ambos separados por coma")
    ap.add_argument("--salida", default=None, help="Carpeta de salida (por defecto <estudio>/reporte)")
    a = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    from estudio_v8.core import ART_DIR
    from estudio_v8 import reporte, reporte_datos, reporte_docx, reporte_pdf

    t0 = time.time()
    root = ART_DIR / a.estudio
    if not root.is_dir():
        sys.exit(f"No existe el estudio {root}")
    salida = Path(a.salida) if a.salida else root / "reporte"
    salida.mkdir(parents=True, exist_ok=True)
    print(f"[reporte] caso {ART_DIR.name} · estudio {a.estudio} · combinación {a.combinacion}", flush=True)
    D = reporte_datos.reunir(a.estudio, a.combinacion, a.plan)
    print(f"[reporte] datos reunidos en {time.time() - t0:.0f} s", flush=True)
    bloques = reporte.construir(D, salida / "figuras")
    nombre = f"Reporte_{ART_DIR.name}_{Path(a.estudio).name}".replace(" ", "_")
    encabezado = f"Reporte de resiliencia · {D['meta']['nombre_red']}"
    hechos = []
    formatos = {f.strip().lower() for f in a.formato.split(",")}
    if "docx" in formatos:
        hechos.append(reporte_docx.escribir(bloques, salida / f"{nombre}.docx", encabezado))
    if "pdf" in formatos:
        hechos.append(reporte_pdf.escribir(bloques, salida / f"{nombre}.pdf", encabezado))

    # Manifiesto: qué se leyó y de cuándo es, para saber si el reporte quedó desactualizado.
    fuentes = sorted({*root.glob(f"combinadas/{a.combinacion}/esc_*.json"),
                      *root.glob(f"plan_fijo/{a.combinacion}/plan_esc_*/esc_*.json"),
                      *(ART_DIR / n for n in ("red.pkl", "costos.pkl", "fallas_meta.json", "faultlin_meta.json"))})
    manifiesto = {"generado": datetime.now().isoformat(timespec="seconds"), "caso": str(ART_DIR),
                  "estudio": a.estudio, "combinacion": a.combinacion, "plan": D["recomendado"]["etiqueta"]
                  if D["recomendado"] else None, "plan_forzado": a.plan, "git": D["meta"]["git"],
                  "archivos": [str(p) for p in hechos],
                  "ultima_fuente": max((p.stat().st_mtime for p in fuentes if p.exists()), default=0),
                  "n_fuentes": len(fuentes)}
    (salida / "manifiesto.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding="utf-8")
    for p in hechos:
        print(f"[reporte] {p}", flush=True)
    print(f"[reporte] listo en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
