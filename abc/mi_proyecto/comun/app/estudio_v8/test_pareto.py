import unittest

import pandas as pd

from estudio_v8.pareto import frontera_pareto, promediar_planes


class TestPareto(unittest.TestCase):
    def test_frontera(self):
        df = pd.DataFrame({"ens": [10, 8, 12, 8, 9], "costo": [5, 7, 4, 9, 7]}, index=list("abcde"))
        fr, dom = frontera_pareto(df, ["ens", "costo"])
        # d (8, 9) lo domina b (8, 7); e (9, 7) lo domina b.
        self.assertEqual(set(fr.index), {"a", "b", "c"})
        self.assertEqual(set(dom.index), {"d", "e"})

    def test_empate_no_domina(self):
        df = pd.DataFrame({"ens": [1.0, 1.0], "costo": [2.0, 2.0]}, index=["x", "y"])
        fr, _ = frontera_pareto(df, ["ens", "costo"])
        self.assertEqual(set(fr.index), {"x", "y"})

    def test_promedio_solo_planes_completos(self):
        larga = pd.DataFrame({"plan": [1, 1, 2], "escenario": [10, 20, 10], "ens": [4.0, 6.0, 1.0]})
        p = promediar_planes(larga, [10, 20], ("ens",))
        self.assertEqual(list(p.index), [1])
        self.assertAlmostEqual(p.loc[1, "ens"], 5.0)


if __name__ == "__main__":
    unittest.main()
