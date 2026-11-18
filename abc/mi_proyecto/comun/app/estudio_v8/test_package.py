import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
from estudio_v8 import package
from estudio_v8.core import APP_DIR


class PackageTests(unittest.TestCase):
    def test_paquete_contiene_insumos_y_respeta_detalle(self):
        with tempfile.TemporaryDirectory(dir=APP_DIR, prefix='_test_v8_') as tmp:
            app = Path(tmp)
            art = app / 'artefactos'
            corrida = art / 'barrido' / 'demo' / 'individual' / 'dg_sol'
            corrida.mkdir(parents=True)
            (app / 'app_v8_estudio.py').write_text('# entrada', encoding='utf-8')
            (art / 'red.pkl').write_bytes(b'red')
            (corrida / 'esc_001.json').write_text('{"escenario":1,"variante_aislacion":"base"}', encoding='utf-8')
            (corrida / 'esc_001.pkl.gz').write_bytes(b'detalle')
            (art / 'baselines').mkdir()
            (art / 'baselines' / 'esc_001.npz').write_bytes(b'base')
            with patch.object(package, 'APP_DIR', app), patch.object(package, 'ART_DIR', art):
                ruta = package.exportar('barrido/demo', completos=False)
                with ZipFile(ruta) as z:
                    self.assertIsNone(z.testzip())
                    nombres = z.namelist()
                    self.assertIn('app_v8_estudio.py', nombres)
                    self.assertIn('artefactos/baselines/esc_001.npz', nombres)
                    self.assertFalse(any(n.endswith('.pkl.gz') for n in nombres))
                    self.assertIn('PAQUETE.json', nombres)


if __name__ == '__main__':
    unittest.main()
