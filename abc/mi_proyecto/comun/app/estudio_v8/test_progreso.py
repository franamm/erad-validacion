import json
import tempfile
import time
import unittest
from pathlib import Path

from estudio_v8 import progreso, estado

LOG = """[barrido] --- medida 'dg_sol' (DG solar) ---
    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0 -1.021e+06    0 1234          - -1.021e+06      -     -    5s
H    0     0                    -9.870e+05 -1.021e+06  3.45%     -    7s
*  123    45             12    -1.0100e+06 -1.0200e+06  0.99%   5.0   30s
"""


class ProgresoTests(unittest.TestCase):
    def test_gurobi_en_curso(self):
        s = progreso.ultimo_solver(LOG)
        self.assertEqual(s["segundos"], 30)
        self.assertAlmostEqual(s["gap_pct"], 0.99)
        self.assertAlmostEqual(s["incumbente"], -1.01e6)

    def test_gurobi_terminado_y_linea_posterior(self):
        fin = LOG + "Best objective -1.010000000000e+06, best bound -1.012000000000e+06, gap 0.1980%\n"
        self.assertTrue(progreso.ultimo_solver(fin)["final"])
        otra = fin + "[barrido]   dg_sol esc 1: obj=-1,010,000  ENS=10.0  SAIDI=0.100  (30 s)\n"
        self.assertIsNone(progreso.ultimo_solver(otra))

    def test_avance_y_estado(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            esperadas = estado.rutas_esperadas(root, ["dg_sol"], [1, 2, 3])
            rutas = [r for *_, r in esperadas]
            rutas[0].parent.mkdir(parents=True)
            rutas[0].write_text(json.dumps({"status": "optimal", "segundos": 60, "mip_gap": .02,
                                            "solver_term": "optimal", "ens": 5}), encoding="utf-8")
            rutas[1].write_text(json.dumps({"status": "sin_solucion", "segundos": 20}), encoding="utf-8")
            a = progreso.avance(rutas, inicio=time.time() - 100)
            self.assertEqual((a["hechas"], a["optimas"], a["pendientes"]), (2, 1, 1))
            self.assertTrue(a["actual"].endswith("esc_003.json"))
            self.assertEqual(a["restante_s"], 40)
            inv = estado.inventario(esperadas, ens_base={1: 4})
            self.assertEqual(list(inv["estado"]), ["sospechosa", "sin_solucion", "falta"])
            self.assertEqual(estado.resumen(inv)["pendientes"], 1)

    def test_seccionadores(self):
        self.assertEqual(estado.seccionadores_de("sec-0-2-9-10"), [0, 2, 9, 10])
        self.assertEqual(estado.seccionadores_de("base"), [])


if __name__ == "__main__":
    unittest.main()
