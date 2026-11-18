import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from graficos_valor_esperado import resumir_indicadores, seleccionar_corridas


def fila(caso, esc, base=100, ens=50, costo=100):
    return dict(_tipo='Individual', _caso=caso, escenario=esc, horas=8760,
                ens_base=base, ens=ens, saidi_base=10, saidi=5,
                saifi_base=2, saifi=1, cst_costo_total=costo)


class IndicadoresEsperadosTest(unittest.TestCase):
    def test_cociente_de_esperanzas_y_porcentaje(self):
        filas = pd.DataFrame([fila('A', 0, 100, 90, 100),
                              fila('A', 1, 1000, 100, 900)])
        r = resumir_indicadores(filas).iloc[0]
        self.assertAlmostEqual(r.usd_kwh, 1000 / 910)
        self.assertAlmostEqual(r.d_ens_pct, 910 / 1100)

    def test_costos_negativos_cero_y_mejora_insuficiente(self):
        filas = pd.DataFrame([fila('ahorro', 0, costo=-100),
                              fila('gratis', 0, costo=0),
                              fila('sin mejora', 0, ens=100),
                              fila('ruido', 0, ens=99.9)])
        r = resumir_indicadores(filas).set_index('_caso')
        self.assertEqual(r.loc['ahorro', 'usd_kwh'], -2)
        self.assertEqual(r.loc['gratis', 'usd_kwh'], 0)
        self.assertTrue(np.isnan(r.loc['sin mejora', 'usd_kwh']))
        self.assertTrue(np.isnan(r.loc['ruido', 'usd_kwh']))

    def test_no_promedia_costos_parciales(self):
        r = resumir_indicadores(pd.DataFrame([
            fila('A', 0), fila('A', 1, costo=np.nan)])).iloc[0]
        self.assertTrue(np.isnan(r.usd_kwh))

    def test_tabla_y_curva_comparten_escenarios(self):
        st = types.ModuleType('streamlit')
        st.multiselect = lambda *a, **k: k['default']
        st.warning = st.info = lambda *a, **k: None
        datos = pd.DataFrame([fila('A', 0, costo=10000), fila('A', 1, costo=100),
                              fila('B', 1, costo=200)])
        with patch.dict(sys.modules, streamlit=st), patch(
                'graficos_valor_esperado.etiqueta_caso', side_effect=lambda t, c: c):
            seleccion = seleccionar_corridas(datos)
        self.assertEqual(set(seleccion.escenario), {1})
        r = resumir_indicadores(seleccion).set_index('_caso')
        self.assertEqual(r.loc['A', 'usd_kwh'], 2)
        self.assertEqual(r.loc['B', 'usd_kwh'], 4)


if __name__ == '__main__':
    unittest.main()
