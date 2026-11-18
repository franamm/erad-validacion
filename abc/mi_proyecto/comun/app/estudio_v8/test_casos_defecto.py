"""Carpeta de artefactos por defecto de cada caso y E4 con el que abre la app.

Un caso nuevo (caso_x/datos/artefactos/x/) tiene que abrir su propia carpeta y no la
contenedora, y su E4 etiquetado cuenta como E4: antes la app decía "no hay red ni E4".
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import casos


def _caso(tmp, nombre_caso, subcarpetas, archivos=()):
    datos = Path(tmp) / nombre_caso / "datos"
    for s in subcarpetas:
        (datos / "artefactos" / s).mkdir(parents=True, exist_ok=True)
    for ruta in archivos:
        (datos / "artefactos" / ruta).write_bytes(b"x")
    return datos


class ArtDirDefectoTests(unittest.TestCase):
    def _con(self, tmp, nombre_caso, datos):
        return patch.multiple(casos, APP_DIR=datos, CASO_DIR=Path(tmp) / nombre_caso)

    def test_caso_nuevo_abre_su_carpeta(self):
        with tempfile.TemporaryDirectory() as tmp:
            datos = _caso(tmp, "caso_elmelon", ["elmelon"])
            with self._con(tmp, "caso_elmelon", datos):
                self.assertEqual(casos.art_dir_defecto(), datos / "artefactos" / "elmelon")
                self.assertFalse(casos.es_validacion(datos / "artefactos" / "elmelon"))

    def test_validacion_no_cambia(self):
        with tempfile.TemporaryDirectory() as tmp:
            datos = _caso(tmp, "caso_validacion", ["validacion", "oberrhein"])
            with self._con(tmp, "caso_validacion", datos):
                self.assertEqual(casos.art_dir_defecto(), datos / "artefactos" / "validacion")
                self.assertTrue(casos.es_validacion(datos / "artefactos" / "validacion"))
                self.assertFalse(casos.es_validacion(datos / "artefactos" / "oberrhein"))

    def test_sin_carpeta_propia_cae_en_la_historica(self):
        with tempfile.TemporaryDirectory() as tmp:
            datos = _caso(tmp, "caso_otro", [])
            with self._con(tmp, "caso_otro", datos):
                self.assertEqual(casos.art_dir_defecto(), datos / "artefactos")


class E4Tests(unittest.TestCase):
    def setUp(self):
        self._env = patch.dict(os.environ)
        self._env.start()
        os.environ.pop("ERAD_SIN_FALLAS", None)

    def tearDown(self):
        self._env.stop()

    def test_solo_etiquetado_cuenta_como_e4(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for n in ("red.pkl", "perfiles.npz", "fallas.npz", "costos.pkl", "sin_fallas_LIM-0.30-ALIM.pkl"):
                (d / n).write_bytes(b"x")
            self.assertEqual(casos.etiqueta_e4(d), "LIM-0.30-ALIM")
            self.assertEqual(casos.listo_para_ver(d), (True, []))
            self.assertTrue(casos.estado_caso(d)["etapas"]["E4"]["existe"])

    def test_historico_manda_si_existe(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "sin_fallas.pkl").write_bytes(b"x")
            (d / "sin_fallas_X.pkl").write_bytes(b"x")
            self.assertEqual(casos.etiqueta_e4(d), "")

    def test_variable_de_entorno_manda(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "sin_fallas.pkl").write_bytes(b"x")
            os.environ["ERAD_SIN_FALLAS"] = "LIM-0.30-ALIM"
            self.assertEqual(casos.etiqueta_e4(d), "LIM-0.30-ALIM")

    def test_sin_ninguno_falta_el_historico(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, faltan = casos.listo_para_ver(Path(tmp))
            self.assertFalse(ok)
            self.assertIn("E4 (sin_fallas.pkl)", faltan)


if __name__ == "__main__":
    unittest.main()
