import sys
import tempfile
import time
import unittest
from pathlib import Path
from estudio_v8 import jobs
from estudio_v8.core import APP_DIR


class JobsTests(unittest.TestCase):
    TERMINALES = ('terminado', 'incompleto', 'fallido', 'detenido')

    def esperar(self, salida, estados, segundos=12):
        limite = time.monotonic() + segundos
        while time.monotonic() < limite:
            info = jobs.estado(salida)
            if info.get('estado') in estados:
                # El worker escribe el estado final ANTES de terminar, así que
                # en un estado terminal su proceso puede seguir vivo un instante
                # con `trabajo_v8.log` abierto. En Windows eso hacía fallar la
                # limpieza de `TemporaryDirectory` con WinError 32 una de cada
                # tres corridas, un falso negativo del test y no del código.
                if info.get('estado') in self.TERMINALES:
                    while time.monotonic() < limite and jobs.vivo(info):
                        time.sleep(.05)
                return info
            time.sleep(.1)
        self.fail(f"No alcanzó {estados}: {jobs.estado(salida)}")

    def test_terminacion_y_codigo_error(self):
        with tempfile.TemporaryDirectory(dir=APP_DIR, prefix='_test_v8_') as tmp:
            for codigo, estado in [(0, 'terminado'), (3, 'fallido')]:
                salida = Path(tmp) / str(codigo)
                jobs.lanzar([sys.executable, '-c', f'print("prueba"); raise SystemExit({codigo})'], salida, {})
                info = self.esperar(salida, [estado])
                self.assertEqual(info['codigo_salida'], codigo)

    def test_exclusion_cancelacion_y_reanudacion(self):
        with tempfile.TemporaryDirectory(dir=APP_DIR, prefix='_test_v8_') as tmp:
            salida = Path(tmp)
            cmd = [sys.executable, '-c', 'import time; time.sleep(30)']
            try:
                jobs.lanzar(cmd, salida, {})
                self.esperar(salida, ['ejecutando'])
                with self.assertRaises(RuntimeError):
                    jobs.lanzar(cmd, salida, {})
                jobs.detener(salida)
                self.assertEqual(jobs.estado(salida)['estado'], 'detenido')
                jobs.lanzar([sys.executable, '-c', 'print("reanudado")'], salida, {})
                self.esperar(salida, ['terminado'])
            finally:
                jobs.detener(salida)


if __name__ == '__main__':
    unittest.main()
