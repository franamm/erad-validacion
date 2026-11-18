"""La selección del caso aísla lecturas, corridas pendientes y nuevos trabajos."""
import hashlib
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import casos
import pipeline as P
from estudio_v8 import jobs, optimizacion as O


class CasoOptimizacionTests(unittest.TestCase):
    def test_estudios_red_y_pendientes_pertenecen_al_caso(self):
        with tempfile.TemporaryDirectory() as tmp:
            antes = P.ART_DIR
            sup = dict(seccionadores=[], horas=24, mip_gap=.02, time_limit=60, e4="")
            for nombre, n in (("a", 2), ("b", 7)):
                caso = Path(tmp) / nombre
                estudio = caso / "barrido" / "mismo_nombre"
                estudio.mkdir(parents=True)
                (estudio / "resumen_individual.csv").write_text("medida,escenario\n", encoding="utf-8")
                with (caso / "red.pkl").open("wb") as fh:
                    pickle.dump({"numesc": n}, fh)
                os.utime(caso / "red.pkl", (100, 100))
                if nombre == "a":
                    (caso / "baselines").mkdir()
                    (caso / "baselines/esc_000.npz").touch()
                self.assertEqual(O.estudios_disponibles(caso), ["barrido/mismo_nombre"])
                self.assertEqual(O.numesc(caso), n)
                pasos, _ = O.armar_pasos(estudio, sup, ["dg_sol"], [0], set(), {})
                self.assertEqual(any(p["tipo"] == "e5" for p in pasos), nombre == "b")
                for paso in pasos:
                    cmd = paso["comando"]
                    self.assertEqual(Path(cmd[cmd.index("--artefactos") + 1]), caso)
                    self.assertTrue(all(p.is_relative_to(caso) for p in paso["rutas"]))
            self.assertEqual(P.ART_DIR, antes)

    def test_lanzador_firma_y_entorno_del_caso_elegido(self):
        with tempfile.TemporaryDirectory() as tmp:
            caso = Path(tmp)
            archivos = ["red.pkl", "perfiles.npz", "fallas.npz", "costos.pkl", "sin_fallas.pkl"]
            for nombre in archivos:
                (caso / nombre).write_bytes(b"caso seleccionado")
            entorno = os.environ.get(casos.VAR_ENTORNO)
            salida = caso / "barrido/prueba"
            with patch.object(jobs.subprocess, "Popen") as popen, patch.object(jobs.threading, "Thread"):
                jobs.lanzar(["python", "ejemplo.py"], salida,
                            {"medidas": ["dg_sol"], "e4": "", "artefactos": str(caso)})
            self.assertEqual(popen.call_args.kwargs["env"][casos.VAR_ENTORNO], str(caso))
            self.assertEqual(os.environ.get(casos.VAR_ENTORNO), entorno)
            firma = jobs.leer_json(salida / "estudio_v8.json")["insumos_sha256"]
            self.assertEqual(firma, {n: hashlib.sha256(b"caso seleccionado").hexdigest() for n in archivos})


if __name__ == "__main__":
    unittest.main()
