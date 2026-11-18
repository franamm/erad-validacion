#!/usr/bin/env bash
# ============================================================
# instalar.sh — Entorno de Python para correr la cadena en Linux
# ============================================================
# Crea erad-env/ en la raíz del repo (el mismo nombre que en Windows, y ya
# ignorado por git), instala las versiones fijas de requirements_linux.txt
# y corre verificar_entorno.py.
#
# USO (desde cualquier carpeta)
#   bash abc/mi_proyecto/comun/app/servidor/instalar.sh
#   PY=/ruta/a/python3.11 bash .../instalar.sh      # otro intérprete
#
# Necesita Python 3.11. Si el servidor no lo trae:
#   module load python/3.11            (clústeres con Lmod)
#   uv python install 3.11 && PY="$(uv python find 3.11)" bash .../instalar.sh
#   conda create -n py311 python=3.11 && PY="$(conda run -n py311 which python)" ...
# ============================================================
set -Eeuo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$APP_DIR/../../../.." && pwd)"
VENV="$REPO/erad-env"
PY="${PY:-python3.11}"

command -v "$PY" >/dev/null || { echo "No encuentro '$PY'. Ver las opciones al inicio de este script." >&2; exit 1; }
VER="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "$VER" = "3.11" ] || echo "[!] $PY es Python $VER; el estudio corrió en 3.11 y las versiones fijas están probadas ahí."

if [ ! -x "$VENV/bin/python" ]; then
  echo "[instalar] creando $VENV con $PY"
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip wheel
"$VENV/bin/python" -m pip install -r "$APP_DIR/servidor/requirements_linux.txt"

# Licencia: gurobipy busca ~/gurobi.lic o $GRB_LICENSE_FILE. Se copia a mano
# (scp) porque trae credenciales y NO va en git (*.lic está en .gitignore).
if [ -z "${GRB_LICENSE_FILE:-}" ] && [ ! -f "$HOME/gurobi.lic" ]; then
  cat <<EOF

[!] Falta la licencia de Gurobi. Desde el computador con la licencia:
      scp ~/gurobi.lic  <usuario>@<servidor>:~/gurobi.lic
    (en Windows: C:\\Users\\<usuario>\\gurobi.lic). Es una licencia WLS: el
    servidor necesita salida HTTPS a token.gurobi.com. Revisar en
    https://license.gurobi.com cuántas sesiones simultáneas admite antes de
    correr en dos máquinas a la vez.
EOF
fi

chmod +x "$APP_DIR/servidor/"*.sh 2>/dev/null || true
echo
"$VENV/bin/python" "$APP_DIR/verificar_entorno.py" --red
