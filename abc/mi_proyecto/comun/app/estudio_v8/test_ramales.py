"""Ramales con fusible (ramales.py): el corte forzado y el modelo sin ramales.

Se resuelve el caso de validación (24 h, escenario 0, sin fallas de línea) con
optimization_base, para que todo P_LC venga de los ramales:
  * sin ramales, y con ramales que no fallan, el modelo es idéntico;
  * un ramal fundido obliga a cortar su fracción de la demanda durante la
    reparación, ni más ni menos, y SAIDI/SAIFI/ENS lo reflejan;
  * varios ramales de la misma barra no pueden pedir más que su demanda.
"""
import unittest
from unittest.mock import patch

import numpy as np
import pyomo.environ as pyo

import ramales as RAM

_SF = pyo.SolverFactory


def _solver_de_prueba():
    """HiGHS (libre) si está; si no, Gurobi. None si no hay ninguno usable."""
    try:
        if _SF("appsi_highs").available(exception_flag=False):
            return "appsi_highs"
    except Exception:
        pass
    try:
        import gurobipy
        gurobipy.Model().dispose()
        return "gurobi"
    except Exception:
        return None


SOLVER = _solver_de_prueba()


class _Solver:
    """Reemplaza al solver de los optimizadores; ignora las opciones de Gurobi."""
    def __init__(self, *a, **k):
        self.options = {}

    def solve(self, model, tee=False):
        return _SF(SOLVER).solve(model, tee=False)


class PisoCorteTests(unittest.TestCase):
    def _data(self, f, carga, horas_rep):
        return {"faultlinMC_ramales": f, "repairMC_ramales": RAM.matriz_reparacion(f, horas_rep),
                "ramales_carga": carga}

    def test_sin_ramales_no_hay_piso(self):
        self.assertEqual(RAM.piso_corte({}, 0, range(24)), {})
        f = np.zeros((24, 1, 2), dtype=np.int8)
        self.assertEqual(RAM.piso_corte(self._data(f, [(0, 3, .5)], 4), 0, range(24)), {})

    def test_fundido_hasta_la_reparacion(self):
        f = np.zeros((24, 1, 1), dtype=np.int8)
        f[3, 0, 0] = 1
        piso = RAM.piso_corte(self._data(f, [(0, 5, .3)], 4), 0, range(24))
        self.assertEqual(sorted(t for t, _ in piso), [3, 4, 5, 6])
        self.assertTrue(all(b == 5 and v == .3 for (_, b), v in piso.items()))

    def test_varios_ramales_se_topan_en_la_demanda(self):
        f = np.zeros((10, 1, 2), dtype=np.int8)
        f[2, 0, :] = 1
        piso = RAM.piso_corte(self._data(f, [(0, 1, .7), (1, 1, .6)], 3), 0, range(10))
        self.assertEqual(piso[2, 1], 1.0)

    def test_anidado_no_cuenta_doble(self):
        f = np.zeros((10, 1, 2), dtype=np.int8)
        f[2, 0, :] = 1                                   # se funden el de afuera (0) y el de adentro (1)
        data = {**self._data(f, [(0, 4, .5), (1, 4, .2)], 3), "ramales_padre": [-1, 0]}
        self.assertAlmostEqual(RAM.piso_corte(data, 0, range(10))[2, 4], .5)
        f[2, 0, 0] = 0                                   # solo el de adentro
        data = {**self._data(f, [(0, 4, .5), (1, 4, .2)], 3), "ramales_padre": [-1, 0]}
        self.assertAlmostEqual(RAM.piso_corte(data, 0, range(10))[2, 4], .2)

    def test_validar_formato(self):
        ok = {"id": ["a"], "km": [1.0], "zona": [1], "carga": [(0, 2, .4)]}
        self.assertEqual(RAM.validar(ok, [1, 2]), 1)
        with self.assertRaises(ValueError):
            RAM.validar({**ok, "carga": [(0, 9, .4)]}, [1, 2])
        with self.assertRaises(ValueError):
            RAM.validar({**ok, "carga": [(0, 2, 1.4)]}, [1, 2])


@unittest.skipIf(SOLVER is None, "sin solver disponible (HiGHS o Gurobi con licencia)")
class OptimizadorConRamalesTests(unittest.TestCase):
    HORAS, BUS, FRAC, FALLA, REP = 24, 5, 0.3, 3, 6

    @classmethod
    def setUpClass(cls):
        import pipeline as P
        cls.P = P
        cls.data0, cls.red = P.construir_data(sin_fallas=True)
        cls.cfg = P.config_medida(cls.red, cls.HORAS, 0)

    def _resolver(self, extra):
        from optimization_base import resolver_optimizacion_base
        data = {**self.data0, **extra}
        with patch("optimization_base.pyo.SolverFactory", _Solver):
            return resolver_optimizacion_base(data=data, config=dict(self.cfg))[0]

    def _ramales(self, fallas_en, carga):
        n = 1 + max(i for i, _, _ in carga)
        f = np.zeros((self.data0["faultlinMC"].shape[0], self.data0["numesc"], n), dtype=np.int8)
        for t, i in fallas_en:
            f[t, 0, i] = 1
        return {"faultlinMC_ramales": f, "repairMC_ramales": RAM.matriz_reparacion(f, self.REP),
                "ramales_carga": carga}

    def test_sin_ramales_identico_a_ramales_que_no_fallan(self):
        a = self._resolver({})
        b = self._resolver(self._ramales([], [(0, self.BUS, self.FRAC)]))
        self.assertEqual([h["P_LC"] for h in a], [h["P_LC"] for h in b])
        self.assertEqual([h["P_SUB"] for h in a], [h["P_SUB"] for h in b])

    def test_ramal_fundido_corta_su_fraccion_durante_la_reparacion(self):
        import indicadores as I
        res = self._resolver(self._ramales([(self.FALLA, 0)], [(0, self.BUS, self.FRAC)]))
        fuera = range(self.FALLA, self.FALLA + self.REP)
        for t, h in enumerate(res):
            p_lc = h["P_LC"][self.BUS]
            p_d = h["Pb"][self.BUS] + p_lc
            if t in fuera:
                self.assertAlmostEqual(p_lc, self.FRAC * p_d, places=3, msg=f"hora {t}")
            self.assertAlmostEqual(sum(v for b, v in h["P_LC"].items() if not (t in fuera and b == self.BUS)),
                                   0.0, places=4, msg=f"corte fuera del ramal en la hora {t}")
        BUSES, P_D_base = list(self.red["BUSES"]), self.red["P_D_base"]
        horas = list(range(self.HORAS))
        peso = P_D_base[self.BUS] / sum(P_D_base[b] for b in BUSES)
        self.assertAlmostEqual(I.fn_SAIDI(res, P_D_base, BUSES, horas), self.FRAC * self.REP * peso, places=4)
        self.assertAlmostEqual(I.fn_SAIFI(res, P_D_base, BUSES, horas), self.FRAC * peso, places=4)
        ens = sum(h["P_LC"][self.BUS] for h in res)
        esperado = sum(self.FRAC * (res[t]["Pb"][self.BUS] + res[t]["P_LC"][self.BUS]) for t in fuera)
        self.assertAlmostEqual(ens, esperado, places=3)

    def test_ramales_que_superan_la_demanda_no_dejan_infactible(self):
        res = self._resolver(self._ramales([(self.FALLA, 0), (self.FALLA, 1)],
                                           [(0, self.BUS, .7), (1, self.BUS, .6)]))
        self.assertEqual(len(res), self.HORAS)
        h = res[self.FALLA]
        self.assertAlmostEqual(h["Pb"][self.BUS], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
