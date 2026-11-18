#!/usr/bin/env bash
# ============================================================
# trabajo_slurm.sbatch — Las etapas de MILP como trabajo de SLURM
# ============================================================
# Solo si el servidor usa SLURM (si `sbatch` existe). Si es una máquina
# compartida sin gestor de colas, usar tmux o nohup (README_SERVIDOR.md §5).
#
# Las etapas que bajan datos (clima, perfiles) NO van acá: en muchos
# clústeres los nodos de cómputo no tienen internet. Correrlas antes en el
# nodo de login, que son minutos:
#   servidor/correr_caso.sh servidor/casos/mi_red.env clima,e0,perfiles,e1,fallas,e2,e3
#
# Lanzar (desde abc/mi_proyecto/comun/app; el caso va en ERAD_CASO del .env):
#   sbatch --export=ALL,CASO_ENV=servidor/casos/mi_red.env servidor/trabajo_slurm.sbatch
#   squeue -u $USER          # estado
#   tail -f slurm-<id>.out   # progreso
#
# Ajustar los recursos a la red. Referencia medida en MV Oberrhein (179
# barras, 8760 h): E4 tardó 4,5 h; la combinación de las 12 medidas se quedó
# sin memoria con 64 GB tras 4,8 h. recursos.py lee lo que se pida acá
# (SLURM_CPUS_PER_TASK, SLURM_MEM_PER_NODE y el cgroup) y dimensiona Gurobi
# con eso, no con el tamaño del nodo.
# ============================================================
#SBATCH --job-name=erad-milp
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%j.out
# #SBATCH --partition=<particion>
# #SBATCH --mail-type=END,FAIL --mail-user=<correo>

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${CASO_ENV:?pasar --export=ALL,CASO_ENV=servidor/casos/<caso>.env}"

# Temporales en el disco local del nodo si existe (más rápido y no llena el
# home): el LP de 8760 h y los nodefiles de Gurobi pueden ser decenas de GB.
if [ -n "${SLURM_TMPDIR:-}" ]; then export TMPDIR="$SLURM_TMPDIR";
elif [ -d "/scratch/$USER" ]; then export TMPDIR="/scratch/$USER/erad_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"; fi

echo "nodo $(hostname) · trabajo ${SLURM_JOB_ID} · ${SLURM_CPUS_PER_TASK:-?} cpus · TMPDIR=${TMPDIR:-default}"
bash servidor/correr_caso.sh "$CASO_ENV" "${ETAPAS:-e4,e5,seleccion,barrido}"
