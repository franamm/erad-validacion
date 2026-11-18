import unittest
import pandas as pd
from estudio_v8.core import MEDIDAS_TESIS, COMBINACION_TESIS, indices, reglas, rangos, comando_barrido


class EstudioTests(unittest.TestCase):
    def test_catalogo_y_combinacion(self):
        self.assertEqual(len(MEDIDAS_TESIS), 12)
        self.assertEqual(COMBINACION_TESIS, "combo_reconfig_topo+dg_sol+dg_eol+dg_fos+ess_4h+ess_5h+ess_6h+meg+mess+edrp+refuerzo+soterramiento")

    def test_matriz_costos_con_faltante_y_empate(self):
        M = pd.DataFrame([[1, 10, None], [5, 6, 7], [5, 6, 7]], index=[1, 2, 3])
        crit, R, elegidos, usados = reglas(M)
        self.assertEqual(usados, [0, 1])
        self.assertEqual(elegidos['Minimax'], [2, 3])
        self.assertEqual(elegidos['Minimin'], [1])
        self.assertEqual(elegidos['Minimax regret'], [1, 2, 3])
        self.assertEqual(R.loc[1, 1], 4)

    def test_beneficio_y_pesos(self):
        M = pd.DataFrame([[1, 10], [5, 6]])
        _, _, e, _ = reglas(M, beneficio=True, pesos={0: .9, 1: .1})
        self.assertEqual(e['Maximax'], [0])
        self.assertEqual(e['Maximin'], [1])
        self.assertEqual(e['Valor esperado'], [1])

    def test_rangos_incluyen_ausencias(self):
        r = rangos([{'plan': {'solar': {'4': 10}}}, {'plan': {}}]).iloc[0]
        self.assertEqual(r['Mínimo'], 0)
        self.assertEqual(r['Mediana'], 5)
        self.assertEqual(r['Frecuencia de instalación'], .5)

    def test_indices_invalidos(self):
        self.assertEqual(indices('1,3-5,3'), [1, 3, 4, 5])
        for x in ['-1', '9-1', 'a', '']:
            with self.assertRaises(ValueError):
                indices(x)

    def test_combinacion_usa_muestra_explicita(self):
        c = dict(nombre='prueba', modo='Combinación', medidas=list(MEDIDAS_TESIS), escenarios=[1, 41],
                 criterio='ens', horas=24, lim_dg=.3, lim_dg_modo='alimentador', e4='prueba',
                 mip_gap=.02, time_limit=60, seccionadores=[0,2])
        cmd = comando_barrido(c)
        self.assertIn('--no-todas', cmd)
        self.assertEqual(cmd[cmd.index('--combo-escenarios')+1], '1,41')
        self.assertEqual(cmd[cmd.index('--fases')+1], '3')


if __name__ == '__main__':
    unittest.main()
