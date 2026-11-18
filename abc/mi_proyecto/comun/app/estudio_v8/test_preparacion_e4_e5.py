"""E4/E5 se preparan con los supuestos elegidos sin tocar otros artefactos."""
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from estudio_v8 import optimizacion as O
import entrada_red_vista as V


class PreparacionE4E5Tests(unittest.TestCase):
    def test_comandos_rutas_y_seleccion_de_etapas(self):
        with tempfile.TemporaryDirectory() as tmp:
            caso = Path(tmp)
            (caso/'red.pkl').touch()
            (caso/'sin_fallas.pkl').write_bytes(b'historico')
            est = dict(es_validacion=False, armado=True, paquete={}, faultlin={'forma': [8760, 1, 12]},
                       meta={}, escenarios_baseline=[0])
            resumen = dict(seccionadores=[], numesc=1, n_buses=15,
                           net=SimpleNamespace(line=pd.DataFrame(index=range(12))))
            code = 'from pathlib import Path\nimport entrada_red_vista as V\nV._panel_etapas(Path('+repr(tmp)+'), '+repr(est)+')'
            with patch.object(V, '_resumen_red_cacheado', return_value=resumen), patch.object(O, 'panel_ejecucion') as panel:
                at = AppTest.from_string(code, default_timeout=60).run()
                self.assertFalse(at.exception, [e.message for e in at.exception])
                at.text_input(key='entrada_red_e5_seccionadores').set_value('0,2,9,10').run()
                at.multiselect(key='entrada_red_etapas_elegidas').set_value(['E4','E5']).run()
                self.assertFalse(at.exception)
                salida, pasos, config = panel.call_args.args
                self.assertEqual(len(pasos), 2)
                e4, e5 = pasos
                self.assertIn('--lim-dg', e4['comando'])
                self.assertEqual(e4['comando'][e4['comando'].index('--lim-dg')+1], '0.3')
                self.assertEqual(e4['comando'][e4['comando'].index('--lim-dg-modo')+1], 'alimentador')
                self.assertEqual(e4['rutas'], [caso/'sin_fallas_LIM-0.30-ALIM.pkl'])
                self.assertEqual(e4['pendientes'], 1)
                self.assertEqual(e5['comando'][e5['comando'].index('--seccionadores')+1], '0,2,9,10')
                self.assertEqual(e5['rutas'], [caso/'baselines/sec-0-2-9-10/esc_000.npz'])
                self.assertEqual(config['seccionadores'], [0,2,9,10])
                self.assertEqual((caso/'sin_fallas.pkl').read_bytes(), b'historico')
                at.text_input(key='entrada_red_e5_seccionadores').set_value('999').run()
                self.assertFalse(at.exception)
                self.assertTrue(any('no existen' in e.value for e in at.error))

    def test_optimizacion_hereda_preparacion_guardada(self):
        with tempfile.TemporaryDirectory() as tmp:
            caso=Path(tmp)
            (caso/'trabajos_v8/pipeline').mkdir(parents=True)
            (caso/'trabajos_v8/pipeline/trabajo_v8.json').write_text(json.dumps({'config': {
                'e4':'LIM-0.30-ALIM', 'seccionadores':[0,2,9,10]}}),encoding='utf-8')
            for archivo,obj in [('red.pkl',{'numesc':1}), ('sin_fallas_LIM-0.30-ALIM.pkl',{
                '_meta': {'lim_dg':.3,'lim_dg_modo':'alimentador','lim_dg_bus':None,'horas':8760}})]:
                with (caso/archivo).open('wb') as f:
                    pickle.dump(obj,f)
            at=AppTest.from_string('from estudio_v8.optimizacion import render\nrender()',default_timeout=60)
            at.session_state['v8_caso']=caso
            with patch.object(O,'panel_ejecucion'):
                at.run()
            self.assertFalse(at.exception, [e.message for e in at.exception])
            self.assertEqual(at.selectbox(key='opt_e4').value,'LIM-0.30-ALIM')
            self.assertEqual(at.text_input(key='opt_secc').value,'0,2,9,10')
            self.assertEqual(at.number_input(key='opt_lim').value,.3)


if __name__=='__main__':
    unittest.main()
