"""Creación y selección de un caso desde el widget real de Streamlit."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import casos
from entrada_red_vista import OPCION_NUEVO


class CrearCasoTests(unittest.TestCase):
    def test_crear_selecciona_el_caso_sin_modificar_un_widget_instanciado(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(casos, "APP_DIR", Path(tmp)):
            base = Path(tmp) / "artefactos/validacion"
            base.mkdir(parents=True)
            (base / ".caso").touch()
            app = AppTest.from_string(
                "from entrada_red_vista import _selector_caso\n_selector_caso()",
                default_timeout=60,
            ).run()
            self.assertFalse(app.exception)
            app.selectbox(key="entrada_red_caso").select(OPCION_NUEVO).run()
            app.text_input(key="entrada_red_caso_nuevo").set_value("caso_nuevo").run()
            app.button(key="entrada_red_btn_caso_nuevo").click().run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            nuevo = Path(tmp) / "artefactos/caso_nuevo"
            self.assertTrue((nuevo / ".caso").is_file())
            self.assertEqual(app.selectbox(key="entrada_red_caso").value, str(nuevo))
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="entrada_red_caso").value, str(nuevo))
            # Intentar repetir el nombre muestra el error sin pisar la carpeta.
            app.selectbox(key="entrada_red_caso").select(OPCION_NUEVO).run()
            app.text_input(key="entrada_red_caso_nuevo").set_value("caso_nuevo").run()
            app.button(key="entrada_red_btn_caso_nuevo").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("ya existe" in e.value for e in app.error))


if __name__ == "__main__":
    unittest.main()
