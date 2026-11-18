#!/bin/bash
# Validación · evento 2020 (semilla 43): rehace en PARALELO lo que dependía del E4 equivocado (viento 2024).
#
# Contexto: ver caso_validacion/TRASPASO_E4_2020.md. El E4 nuevo (sin_fallas_LIM-0.30-ALIM.pkl, ya en git) está
# calculado con el viento de 2020. En Windows quedaron hechas 7 de las 10 combinadas; 0, 7, 23 y 32 terminaron con
# solver_term=error (la 0 con gap 42 %) y faltan 52, 91 y 95. Los planes fijos de la matriz 20 x 10 no se rehicieron.
#
# Etapas (cada una un proceso por escenario, todos a la vez):
#   combinadas  librería completa (13 medidas) en los 10 escenarios del evento 2020. Solo rehace 0, 7, 23, 32 con
#               --forzar; 52, 91 y 95 son nuevas; 26, 34 y 38 (optimal) se saltan por reanudable.
#   plan_fijo   20 planes (10 propios P11-P20 + 10 del evento 2024) x 10 escenarios. Un proceso por escenario evaluado.
#               Necesita las 10 combinadas terminadas, así que va DESPUÉS de la etapa anterior.
#
# USO (desde comun/app, dentro de tmux o con nohup)
#   servidor/validacion_2020_paralelo.sh combinadas --dry-run
#   servidor/validacion_2020_paralelo.sh combinadas
#   servidor/validacion_2020_paralelo.sh plan_fijo
#
# Variables opcionales: THREADS (default 6), MEM_GB (default 45, SoftMemLimit de Gurobi por combinada),
#   FORZAR_ESC (default "0 7 23 32"), ESC (default los 10 del evento).
# RAM: en Windows las combinadas fallaron al llegar a ~19 GB con el tope en 14 GB. Acá cada una tiene 45 GB de Gurobi y
# se corren 4 a la vez (MAX_PAR_COMB); las otras 6 esperan turno. Para más memoria por corrida, bajar el paralelismo:
# MEM_GB=60 MAX_PAR_COMB=3.

cd "$(dirname "$0")/.." || exit 1
APP_DIR="$(pwd)"
ETAPA="${1:-}"
DRY=""; [ "${2:-}" = "--dry-run" ] && DRY=1

PYTHON="${PYTHON:-$APP_DIR/../../../../erad-env/bin/python}"
ART="$APP_DIR/../../caso_validacion/datos/artefactos"
ART2020="$ART/validacion_2020_s43"
BARRIDO="sec-0-2-9-10 2020 semilla43 peores"
PLANES_2024="$ART/validacion/barrido/sec-0-2-9-10 severo/combinadas/combo_reconfig_topo+dg_sol+dg_eol+dg_fos+ess_4h+ess_5h+ess_6h+meg+mess+edrp+refuerzo+soterramiento"
ESC="${ESC:-0 7 23 26 32 34 38 52 91 95}"
FORZAR_ESC="${FORZAR_ESC:-0 7 23 32}"
THREADS="${THREADS:-6}"
MEM_GB="${MEM_GB:-45}"              # SoftMemLimit de Gurobi por combinada (en Windows con ~14 GB fallaron)
MAX_PAR_COMB="${MAX_PAR_COMB:-4}"   # combinadas simultáneas: 4 x (45 GB Gurobi + ~15 GB Python) ~ 240 GB como peor caso
MAX_PAR_PLAN="${MAX_PAR_PLAN:-10}"  # los plan_fijo son modelos chicos (~190 s)
MAX_PAR=0
LOGS="$ART2020/_logs_paralelo"
MEDIDAS="dg_sol,dg_eol,dg_fos,ess_4h,ess_5h,ess_6h,reconfig_topo,inv_tie,refuerzo,soterramiento,meg,mess,edrp"  # las 13 de la combinación original
ESC_CSV="$(echo $ESC | tr ' ' ',')"

[ -n "$ETAPA" ] || { sed -n 2,25p "$0"; exit 1; }

# ── Comprobaciones previas ───────────────────────────────────────────────────
falta=0
for f in sin_fallas_LIM-0.30-ALIM.pkl perfiles.npz fallas.npz; do
    [ -f "$ART2020/$f" ] || { echo "FALTA $ART2020/$f  (git pull)"; falta=1; }
done
# red.pkl y costos.pkl no vienen en git para esta carpeta: son los mismos del evento 2024 (misma red CIGRE y costos)
for f in red.pkl costos.pkl; do
    if [ ! -f "$ART2020/$f" ]; then
        if [ -f "$ART/validacion/$f" ]; then echo "copiando $f desde validacion/"; [ -n "$DRY" ] || cp -n "$ART/validacion/$f" "$ART2020/$f"
        else echo "FALTA $f en validacion_2020_s43 y en validacion"; falta=1; fi
    fi
done
[ -d "$ART2020/baselines" ] || { echo "FALTA $ART2020/baselines"; falta=1; }
[ "$falta" = 0 ] || exit 1
# Guardia: el E4 de 2020 tiene que ser distinto al de 2024
if cmp -s "$ART/validacion/sin_fallas_LIM-0.30-ALIM.pkl" "$ART2020/sin_fallas_LIM-0.30-ALIM.pkl"; then
    echo "ERROR: el E4 de 2020 es idéntico al de 2024"; exit 1; fi

mkdir -p "$LOGS"
correr() {   # correr <log> <comando...>  → en segundo plano, o solo lo imprime con --dry-run
    local log="$1"; shift
    if [ -n "$DRY" ]; then printf '%q ' "$@"; echo " > $log"; else
        while [ "$MAX_PAR" -gt 0 ] && [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do sleep 20; done
        nohup "$@" > "$log" 2>&1 &
    fi
}

# ── Etapa 1: combinadas ──────────────────────────────────────────────────────
combinadas() {
    MAX_PAR="$MAX_PAR_COMB"
    for e in $ESC; do
        extra=""; for f in $FORZAR_ESC; do [ "$f" = "$e" ] && extra="--forzar"; done
        correr "$LOGS/combinada_$e.log" "$PYTHON" run_barrido_medidas.py \
            --artefactos "$ART2020" --salida "$BARRIDO" \
            --escenarios "$e" --combo-escenarios "$e" --fases 3 --medidas "$MEDIDAS" \
            --seccionadores 0,2,9,10 --sin-fallas LIM-0.30-ALIM --lim-dg 0.3 --lim-dg-modo alimentador \
            --n_meg 3 --p_meg_max 300 --t_tra_meg 2 --meg_existentes 0 --t_maniobra_tie 0 \
            --mip_gap 0.02 --time_limit 7200 --mipfocus 1 \
            --threads "$THREADS" --gurobi_mem_gb "$MEM_GB" $extra
    done
    [ -n "$DRY" ] && return
    wait
    echo "combinadas listas. Rearmando resumen_combinadas.csv (los procesos en paralelo se lo pisan entre sí)..."
    "$PYTHON" run_barrido_medidas.py --artefactos "$ART2020" --salida "$BARRIDO" \
        --escenarios "$ESC_CSV" --combo-escenarios "$ESC_CSV" --fases 3 --medidas "$MEDIDAS" \
        --seccionadores 0,2,9,10 --sin-fallas LIM-0.30-ALIM --lim-dg 0.3 --lim-dg-modo alimentador \
        --n_meg 3 --p_meg_max 300 --t_tra_meg 2 --meg_existentes 0 --t_maniobra_tie 0 \
        --mip_gap 0.02 --time_limit 7200 --mipfocus 1 --threads "$THREADS" > "$LOGS/resumen_combinadas.log" 2>&1
    echo "Revisar status/gap de cada esc_XXX.json antes de seguir (solver_term, solver_gap, cst_sobrecosto_dg ≈ 0 o chico)."
}

# ── Etapa 2: plan fijo 20 x 10 ───────────────────────────────────────────────
plan_fijo() {
    MAX_PAR="$MAX_PAR_PLAN"
    n=$(ls "$ART2020/barrido/$BARRIDO/combinadas/todas"/esc_*.json 2>/dev/null | wc -l)
    [ "$n" -ge 10 ] || { echo "Solo hay $n combinadas; hacen falta 10. Corre primero 'combinadas'."; exit 1; }
    for e in $ESC; do
        correr "$LOGS/plan_fijo_$e.log" "$PYTHON" run_plan_fijo.py \
            --artefactos "$ART2020" --barrido "$BARRIDO" --combinacion todas \
            --planes-de "$PLANES_2024" --offset-planes-de 100 --salida-nombre todas_2eventos \
            --escenarios "$e" --mip_gap 0.02 --time_limit 7200 --threads "$THREADS"
    done
    [ -n "$DRY" ] && return
    wait
    echo "Armando la matriz con los 10 escenarios..."
    "$PYTHON" run_plan_fijo.py --artefactos "$ART2020" --barrido "$BARRIDO" --combinacion todas \
        --planes-de "$PLANES_2024" --offset-planes-de 100 --salida-nombre todas_2eventos \
        --escenarios "$ESC_CSV" --solo-matriz > "$LOGS/plan_fijo_matriz.log" 2>&1
    echo "Listo: $ART2020/barrido/$BARRIDO/plan_fijo/todas_2eventos/"
}

case "$ETAPA" in
    combinadas) combinadas ;;
    plan_fijo)  plan_fijo ;;
    *) echo "Etapa desconocida: $ETAPA (combinadas | plan_fijo)"; exit 1 ;;
esac
