"""PerfilPorAnio (pipeline): el escenario e lee el perfil del año e // sorteos, como el arreglo completo."""
import unittest

import numpy as np

from pipeline import PerfilPorAnio


class PerfilPorAnioTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.por_anio = rng.random((3, 24, 5))              # años × horas × barras
        self.p = PerfilPorAnio(self.por_anio, 4, 12)        # 12 escenarios, 4 por año
        self.completo = np.stack([self.por_anio[e // 4] for e in range(12)], axis=1)   # horas × esc × barras

    def test_forma(self):
        self.assertEqual(self.p.shape, (24, 12, 5))
        self.assertEqual(len(self.p), 24)

    def test_valor_por_valor(self):
        for t, e, b in [(0, 0, 0), (5, 3, 2), (5, 4, 2), (23, 11, 4), (10, 8, 1)]:
            self.assertEqual(self.p[t, e, b], self.completo[t, e, b])

    def test_rebanadas_y_materializar(self):
        np.testing.assert_array_equal(self.p[:, 7, :], self.completo[:, 7, :])
        np.testing.assert_array_equal(self.p[3, :, 1], self.completo[3, :, 1])
        np.testing.assert_array_equal(np.asarray(self.p), self.completo)

    def test_faltan_anios(self):
        with self.assertRaises(ValueError):
            PerfilPorAnio(self.por_anio, 4, 13)


if __name__ == "__main__":
    unittest.main()
