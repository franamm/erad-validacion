"""Los valores iniciales pertenecen al E4 y se verifican antes de lanzar."""
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from estudio_v8 import optimizacion as O


class SupuestosE4Tests(unittest.TestCase):
    def test_defaults_y_bloqueo_antes_de_ejecutar(self):
        with tempfile.TemporaryDirectory() as tmp:
            caso = Path(tmp)
            for nombre, obj in {
                "red.pkl": {"numesc": 1},
                "sin_fallas.pkl": {"_meta": {"lim_dg": 1.0, "lim_dg_modo": "bus", "lim_dg_bus": None, "horas": 8760}},
                "sin_fallas_otro.pkl": {"_meta": {"lim_dg": .3, "lim_dg_modo": "alimentador", "lim_dg_bus": None, "horas": 24}},
            }.items():
                with (caso / nombre).open("wb") as f:
                    pickle.dump(obj, f)
            app = AppTest.from_string('from estudio_v8.optimizacion import render\nrender()', default_timeout=60)
            app.session_state['v8_caso'] = caso
            with patch('estudio_v8.jobs.lanzar', side_effect=AssertionError('No ejecutar')):
                app.run()
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertEqual(app.number_input(key='opt_lim').value, 1.0)
                self.assertEqual(app.selectbox(key='opt_lim_modo').value, 'bus')
                self.assertEqual(app.number_input(key='opt_horas').value, 8760)
                app.number_input(key='opt_lim').set_value(.3).run()
                self.assertFalse(app.exception)
                self.assertTrue(any('INCONSISTENCIA' in e.value for e in app.error))
                self.assertFalse(any('Ejecutar' in b.label for b in app.button))
                app.selectbox(key='opt_e4').select('otro').run()
                self.assertFalse(app.exception)
                self.assertEqual(app.number_input(key='opt_lim').value, .3)
                self.assertEqual(app.selectbox(key='opt_lim_modo').value, 'alimentador')
                self.assertEqual(app.number_input(key='opt_horas').value, 24)
                self.assertFalse(app.error)

    def test_e4_antiguo_y_supuestos_incompatibles(self):
        meta = {"lim_dg": 1.0, "horas": 8760}
        sup = dict(lim_dg=1.0, horas=8760, lim_dg_modo='bus', e4='')
        self.assertIsNone(O.incompatibilidad_e4(meta, sup))
        self.assertIsNotNone(O.incompatibilidad_e4(meta, {**sup, 'lim_dg_modo': 'alimentador'}))
        self.assertIsNotNone(O.incompatibilidad_e4(meta, {**sup, 'horas': 24}))


if __name__ == '__main__':
    unittest.main()
