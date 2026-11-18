"""Pruebas de lectura: aislamiento por caso y formatos guardados, sin solver."""
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from estudio_v8.resultados_archivos import archivos_disponibles, leer_resultado
import casos


class ResultadosArchivosTest(unittest.TestCase):
    def test_casos_con_mismo_nombre_y_sello_no_comparten_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            for caso, valor in (("a", 2.0), ("b", 8.0)):
                carpeta = raiz / caso
                carpeta.mkdir()
                p = carpeta / "caso_base_1h.pkl"
                with p.open("wb") as fh:
                    pickle.dump({0: [{"P_LC": {1: valor}, "P_SUB": {0: 3.0}}]}, fh)
                tablas = leer_resultado(str(p), 1, 100)
                self.assertEqual(next(iter(tablas.values()))["P_LC"].sum(), valor)
                self.assertEqual(archivos_disponibles(carpeta), [p])
            p = raiz / "a/caso_base_1h.pkl"
            with p.open("wb") as fh:
                pickle.dump({0: [{"P_LC": {1: 11.0}}]}, fh)
            self.assertEqual(next(iter(leer_resultado(str(p), 2, 100).values()))["P_LC"].sum(), 11)

    def test_rutas_antiguas_y_casos_nuevos(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(casos, "APP_DIR", Path(tmp)):
            for nombre in ("validacion", "oberrhein"):
                p = Path(tmp) / "artefactos" / nombre
                p.mkdir(parents=True)
                (p / "red.pkl").touch()
            self.assertEqual(casos.resolver_ruta("artefactos_oberrhein"), p)
            self.assertEqual([p.name for p in casos.listar_casos()], ["validacion", "oberrhein"])
            ruta, error = casos.nombre_valido("nuevo")
            self.assertIsNone(error)
            self.assertEqual(ruta, Path(tmp) / "artefactos/nuevo")
            for nombre in ("..", "artefactos_..", "../otro", "artefactos_"):
                self.assertIsNotNone(casos.nombre_valido(nombre)[1])


if __name__ == "__main__":
    unittest.main()
