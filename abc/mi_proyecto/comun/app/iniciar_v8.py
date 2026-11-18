"""Arranque local: python iniciar_v8.py [--server.port=8508]."""
from pathlib import Path
import importlib.util
import sys


def main():
    if importlib.util.find_spec("streamlit") is None:
        # Entorno de verificación instalado dentro de este workspace, si existe.
        runtime = next((p / ".codex_runtime_v8" for p in Path(__file__).resolve().parents
                        if (p / ".codex_runtime_v8" / "streamlit").is_dir()), None)
        if runtime is None:
            raise SystemExit("Instala las dependencias: python -m pip install -r requirements_v8.txt")
        # Usar las bibliotecas científicas del entorno; completar solo las faltantes.
        sys.path.insert(0, str(runtime))
        import typing_extensions  # noqa: F401 -- requerido por AnyIO del runtime
        sys.path.remove(str(runtime))
        sys.path.append(str(runtime))
    # El caso se decide con el directorio desde donde se lanza (ver rutas.py) y
    # queda publicado en ERAD_CASO; después se trabaja desde la carpeta del
    # código, que es donde Streamlit busca .streamlit/config.toml.
    import os
    import rutas
    os.chdir(rutas.CODIGO_DIR)
    from streamlit.web import cli
    app = Path(__file__).with_name("app_v8_estudio.py")
    sys.argv = ["streamlit", "run", str(app), "--global.developmentMode=false", *sys.argv[1:]]
    cli.main()


if __name__ == "__main__":
    main()
