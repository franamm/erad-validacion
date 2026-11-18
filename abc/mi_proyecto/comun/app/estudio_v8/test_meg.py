"""Generador móvil: flota existente + compra, vida útil propia y arriendo por evento.

Caso de validación, escenario 26 (falla en la hora 36), 120 h, con la ENS a
500 USD/kWh para que las unidades convengan. Se resuelve con HiGHS si está.
  * una unidad existente ahorra exactamente su CAPEX y se ubica antes que la comprada;
  * `frp_meg` cambia el CAPEX del MEG comprado y nada más;
  * el arrendado sirve h_meg_arr horas desde su llegada, a lo más P_meg_arr kW,
    y se paga por evento y unidad; si el arriendo es carísimo, no se arrienda.
"""
import contextlib
import io
import unittest
from unittest.mock import patch

from test_ramales import SOLVER, _Solver

ESC, HORAS, VOLL = 26, 120, 500
P_UNIDAD = 300.0


@unittest.skipIf(SOLVER is None, "sin solver disponible (HiGHS o Gurobi con licencia)")
class MegTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pipeline as P
        import optimization_6_oficial as O
        cls.P, cls.O = P, O
        cls.data, cls.red = P.construir_data()
        cls.data["VoLL"] = {b: VOLL for b in cls.red["BUSES"]}
        cls.B = cls.red["BUSES"]
        cls.meg = dict(meg=True, n_meg=2, P_meg_max=P_UNIDAD, T_tra_meg={b: 2 for b in cls.B})
        cls.arr = dict(meg_arriendo=True, n_meg_arr=2, P_meg_arr=200.0, h_meg_arr=6,
                       T_tra_meg_arr={b: 2 for b in cls.B})
        cls.comprado = cls._resolver(**cls.meg)

    @classmethod
    def _resolver(cls, extra_data=None, **cfg_extra):
        cfg = cls.P.config_medida(cls.red, HORAS, ESC)
        cfg.update(cfg_extra)
        data = {**cls.data, **(extra_data or {})}
        with patch.object(cls.O.pyo, "SolverFactory", _Solver), contextlib.redirect_stdout(io.StringIO()):
            return cls.O.resolver_optimizacion(data=data, config=cfg)[ESC]

    def _frp(self, n):
        r = self.data["r"]
        return r * (1 + r) ** n / ((1 + r) ** n - 1)

    def test_unidad_existente_ahorra_su_capex(self):
        res = self._resolver(meg_existentes=1, **self.meg)
        cap0 = res[0]["CAP_MEG_MW"][0]
        ahorro = self.comprado[0]["fn_objetivo"] - res[0]["fn_objetivo"]
        self.assertAlmostEqual(ahorro, self.data["MEG"][0] * self._frp(self.data["años"]) * cap0, places=2)
        ubicada = lambda k: any(res[0]["x_re"][(k, b)] for b in self.B)
        self.assertTrue(ubicada(0) or not ubicada(1), "se ubicó la comprada sin la existente")

    def test_vida_util_propia_solo_cambia_el_capex(self):
        n = 10
        res = self._resolver(extra_data={"frp_meg": self._frp(n)}, **self.meg)
        cap = sum(res[0]["CAP_MEG_MW"].values())
        dif = res[0]["fn_objetivo"] - self.comprado[0]["fn_objetivo"]
        esperado = self.data["MEG"][0] * cap * (self._frp(n) - self._frp(self.data["años"]))
        self.assertAlmostEqual(dif, esperado, delta=1e-6 * abs(esperado) + 1e-3)

    def test_arriendo_respeta_ventana_y_potencia(self):
        res = self._resolver(extra_data={"MEG_arriendo": [867.0]}, **self.arr)
        arriendos = res[0]["arriendos_MEG"]
        self.assertTrue(arriendos, "con ENS a 500 USD/kWh debería arrendar")
        horas = sorted({h["hora"] - 1 for h in res if sum(h["P_MEG_arr"].values()) > 1e-6})
        self.assertLessEqual(len(horas), 6)
        self.assertGreaterEqual(min(horas), 36 + 2)          # falla en la 36, llega en 2 h
        self.assertLessEqual(max(v for h in res for v in h["P_MEG_arr"].values()), 200.0 + 1e-6)
        self.assertEqual(sum(sum(h["P_MEG"].values()) for h in res), 0.0)

    def test_arriendo_carisimo_no_se_usa(self):
        base = self._resolver()
        res = self._resolver(extra_data={"MEG_arriendo": [1e9]}, **self.arr)
        self.assertEqual(res[0]["arriendos_MEG"], [])
        self.assertAlmostEqual(res[0]["fn_objetivo"], base[0]["fn_objetivo"], places=4)


if __name__ == "__main__":
    unittest.main()
