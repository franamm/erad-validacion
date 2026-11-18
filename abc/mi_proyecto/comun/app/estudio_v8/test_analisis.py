"""Pruebas de analisis.py: casos conocidos y contraste con las tablas de la tesis."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from estudio_v8 import analisis as A
from estudio_v8.core import APP_DIR, ART_DIR, COMBINACION_TESIS
from rutas import DATOS_DIR  # datos del caso activo (ver rutas.py)

TANDA = ART_DIR / "barrido" / "sec-0-2-9-10 severo"
TESIS = APP_DIR.parents[3] / "Tesis_Francisca_Muñoz_M" / "Figuras" / "Resultados"
MUESTRA = DATOS_DIR / "figuras_presentacion_severo" / "muestra.json"


class ReglasTests(unittest.TestCase):
    # Planes diseñados para los escenarios 1, 2 y 3; columnas = evaluación.
    M = pd.DataFrame([[10, 30, 20],
                      [14, 12, 25],
                      [13, 16, 18]], index=[1, 2, 3], columns=[1, 2, 3], dtype=float)

    def test_costos_referencia_diagonal(self):
        r = A.reglas(self.M, referencia="diagonal")
        R = r["regret"]
        self.assertEqual(R.loc[1, 2], 18)          # 30 - M[2,2]
        self.assertEqual(R.loc[3, 3], 0)
        self.assertEqual(r["elegidos"]["Valor esperado"], [3])   # 47/3
        self.assertEqual(r["elegidos"]["Minimax"], [3])          # máx 18
        self.assertEqual(r["elegidos"]["Minimin"], [1])          # mín 10
        self.assertEqual(r["elegidos"]["Minimax regret"], [3])   # máx(3, 4, 0)
        # EVPI = min E[plan] - E[diagonal] = 47/3 - 40/3
        self.assertAlmostEqual(r["evpi"], 7 / 3)

    def test_referencia_mejor_candidato_difiere(self):
        M = self.M.copy()
        M.loc[3, 2] = 11                            # un plan ajeno supera la diagonal
        d = A.reglas(M, referencia="diagonal")["regret"]
        c = A.reglas(M, referencia="mejor_candidato")["regret"]
        self.assertEqual(d.loc[3, 2], -1)          # regret negativo = ruido o gap
        self.assertEqual(c.loc[3, 2], 0)
        self.assertEqual(c.loc[2, 2], 1)

    def test_beneficio_orientacion_y_empates(self):
        U = pd.DataFrame([[5, 1], [3, 3], [3, 3]], index=[0, 1, 2], columns=[0, 1], dtype=float)
        r = A.reglas(U, beneficio=True, referencia="mejor_candidato")
        self.assertEqual(r["elegidos"]["Maximax"], [0])
        self.assertEqual(r["elegidos"]["Maximin"], [1, 2])
        # Referencias 5 y 3: los tres planes pierden como máximo 2 → empate.
        self.assertEqual(r["elegidos"]["Minimax regret"], [0, 1, 2])
        self.assertEqual(r["regret"].loc[0, 1], 2)

    def test_columnas_incompletas_y_pesos(self):
        M = self.M.copy()
        M.loc[2, 3] = np.nan
        r = A.reglas(M, pesos={1: 3, 2: 1, 3: 99})
        self.assertEqual(r["columnas"], [1, 2])
        self.assertAlmostEqual(r["criterios"].loc[1, "Valor esperado"], (3 * 10 + 30) / 4)
        with self.assertRaises(ValueError):
            A.reglas(M, pesos={1: 1})


class PercentilTests(unittest.TestCase):
    def test_reproduce_escenarios_estratificados(self):
        ruta = ART_DIR / "escenarios_estratificados.json"
        if not ruta.exists():
            self.skipTest("sin escenarios_estratificados.json")
        meta = json.loads(ruta.read_text(encoding="utf-8"))
        muestra = A.muestra_por_percentil(meta["indicadores_por_escenario"],
                                          meta["n_estratos"], meta["criterio"])
        self.assertEqual(sorted(muestra["escenario"]), meta["escenarios"])
        # Decil más severo: con estos casos base da la muestra de la tanda severa.
        severa = A.muestra_mas_severa(meta["indicadores_por_escenario"], 10, "ens")
        self.assertEqual(sorted(severa["escenario"]), [1, 9, 15, 21, 41, 57, 78, 82, 93, 98])
        self.assertTrue(severa["ens"].is_monotonic_decreasing)
        self.assertEqual(severa["Percentil"].iloc[0], 100)
        pct = A.percentil_en_poblacion(meta["indicadores_por_escenario"], meta["escenarios"])
        self.assertTrue(pct["Percentil"].between(0, 100).all())


@unittest.skipUnless(TANDA.exists() and MUESTRA.exists() and TESIS.exists(),
                     "sin la tanda severo o las tablas de la tesis")
class ContrasteTesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from barrido_csv import leer_resumen
        cls.escenarios = json.loads(MUESTRA.read_text(encoding="utf-8"))["escenarios_internos"]
        cls.dce = leer_resumen(TANDA / "resumen_combinadas.csv")

    def test_medidas_activas(self):
        esperado = pd.read_csv(TESIS / "combinaciones_medidas_activas.csv",
                               index_col=0, encoding="utf-8-sig")
        nombre_a_clave = {v: k for k, v in A.NOMBRE_COMBO_TESIS.items()}
        combos = [nombre_a_clave[i.split(". ", 1)[1]] for i in esperado.index]
        obtenido = A.tabla_activas(self.dce, combos, self.escenarios)[A.ORDEN_TABLA]
        obtenido.columns = [A.NOMBRE_MEDIDA[m] for m in obtenido.columns]
        obtenido.index = esperado.index
        pd.testing.assert_frame_equal(obtenido.astype(float), esperado.astype(float),
                                      check_names=False)

    def test_rangos_libreria_completa(self):
        import pipeline as P
        esperado = pd.read_csv(TESIS / "rangos_libreria_completa.csv", encoding="utf-8-sig")
        carpeta = TANDA / "combinadas" / COMBINACION_TESIS
        regs = {e: json.loads((carpeta / f"esc_{e:03d}.json").read_text(encoding="utf-8"))
                for e in self.escenarios}
        edrp = A.edrp_por_bus({e: carpeta / f"esc_{e:03d}.npz" for e in self.escenarios})
        nombres = P._cargar("red.pkl")["net"].line["name"].to_dict()
        obtenido = A.rangos_plan(regs, self.escenarios, nombres, edrp)
        # La tesis solo lista DG renovable, reconfiguración y EDRP; la función
        # cubre además las demás claves de `plan`.
        obtenido = obtenido[obtenido["Medida"].isin(set(esperado["Medida"]))].reset_index(drop=True)
        pd.testing.assert_frame_equal(obtenido[["Medida", "Elemento", "Unidad", "Escenarios"]],
                                      esperado[["Medida", "Elemento", "Unidad", "Escenarios"]])
        np.testing.assert_allclose(obtenido[["Min", "Max", "Media"]].to_numpy(),
                                   esperado[["Min", "Max", "Media"]].to_numpy(), rtol=1e-6, atol=1e-6)

    def test_solapamiento_conjunto_igual_a_esperados(self):
        import barrido_vista as BV
        esperados = pd.read_csv(DATOS_DIR / "figuras_presentacion_severo" / "indicadores_esperados.csv",
                                encoding="utf-8-sig")
        B = BV.cargar_barrido("barrido/sec-0-2-9-10 severo", 0)
        base = BV.cargar_base(B["variante"], tuple(self.escenarios), 8760)
        ind = B["individual"][B["individual"]["escenario"].isin(self.escenarios)]
        ind = BV.enriquecer(ind, base, B["mip_gap"])
        dce = BV.enriquecer(self.dce[self.dce["escenario"].isin(self.escenarios)], base, B["mip_gap"])
        s = A.solapamiento_activas(ind, dce, COMBINACION_TESIS, self.escenarios, "ens")
        fila = esperados[esperados["_caso"] == COMBINACION_TESIS].iloc[0]
        self.assertAlmostEqual(s["conjunto"], fila["d_ens"], delta=1e-3)
        self.assertFalse(s["faltan_individuales"])
        self.assertGreaterEqual(s["suma"], s["conjunto"] * 0.5)


if __name__ == "__main__":
    unittest.main()
